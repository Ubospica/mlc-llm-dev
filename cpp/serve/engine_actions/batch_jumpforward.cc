/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_verify.cc
 */

#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/threading_backend.h>

#include <cmath>
#include <exception>

#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs verification for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 */
class BatchJumpForwardActionObj : public EngineActionObj {
 public:
  explicit BatchJumpForwardActionObj(Array<Model> models, Tokenizer tokenizer,
                                     Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        tokenizer_(tokenizer),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Do not run decode when there are multiple models or no running requests.
    if (models_.size() > 1 || estate->running_queue.empty()) {
      return {};
    }

    // Preempt request state entries when decode cannot apply.
    std::vector<RequestStateEntry> running_rsentries;
    {
      NVTXScopedRange nvtx_scope("BatchJumpForward getting requests");
      running_rsentries = GetRunningRequestStateEntries(estate);
      while (!CheckMemForJumpForward(running_rsentries.size())) {
        if (estate->prefix_cache->TryFreeMemory()) continue;
        RequestStateEntry preempted =
            PreemptLastRunningRequestStateEntry(estate, models_, NullOpt, trace_recorder_);
        if (preempted.same_as(running_rsentries.back())) {
          running_rsentries.pop_back();
        }
      }
    }

    if (running_rsentries.empty()) {
      return {};
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    for (auto rsentry : running_rsentries) {
      if (!CanJumpForward(rsentry)) {
        continue;
      }

      auto mstate = rsentry->mstates[0];
      auto jump_forward_str = mstate->grammar_state_matcher.value()->FindJumpForwardString();

      if (jump_forward_str.empty()) {
        continue;
      }

      auto [rollback_cnt, new_tokens] =
          RetokenizeWithNewString(mstate, jump_forward_str, MAX_ROLLBACK_TOKENS_);

      // handle output
      if (rollback_cnt >
          static_cast<int>(mstate->committed_tokens.size()) - rsentry->next_callback_token_pos) {
        const auto& token_table = tokenizer_->PostProcessedTokenTable();
        for (auto i = rsentry->next_callback_token_pos; i < mstate->committed_tokens.size(); ++i) {
          auto token_id = mstate->committed_tokens[i].GetTokenId();
          rsentry->extra_prefix_string += token_table[token_id];
        }
        rsentry->extra_prefix_string += jump_forward_str;
        rsentry->next_callback_token_pos = static_cast<int>(mstate->committed_tokens.size()) -
                                           rollback_cnt + static_cast<int>(new_tokens.size());
      }

      if (rollback_cnt > 0) {
        mstate->RollbackTokens(rollback_cnt);
      }

      if (rollback_cnt > mstate->num_pending_kv_cache_tokens) {
        models_[0]->PopNFromKVCache(mstate->internal_id,
                                    rollback_cnt - mstate->num_pending_kv_cache_tokens);
        mstate->num_pending_kv_cache_tokens = 0;
      } else {
        mstate->num_pending_kv_cache_tokens -= rollback_cnt;
      }

      // Add new tokens
      for (auto token_id : new_tokens) {
        mstate->CommitToken({{token_id, 1.0}, {}});
      }

      mstate->require_retokenization_in_next_decode = true;

      rsentry->rstate->metrics.num_output_tokens +=
          static_cast<int>(new_tokens.size()) - rollback_cnt;
    }

    auto tend = std::chrono::high_resolution_clock::now();
    // std::cout << "BatchJumpForward time: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count()
    //           << " us" << std::endl;
    estate->metrics.engine_jump_forward_time_sum +=
        static_cast<double>((tend - tstart).count()) / 1e9;

    return {};
  }

 private:
  /*! \brief Check if the input request state entries can be decoded under conditions. */
  bool CheckMemForJumpForward(int num_rsentries) {
    static constexpr int MAX_AVG_JUMPFORWARD_PAGES_PER_REQUEST = 10;
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries * MAX_AVG_JUMPFORWARD_PAGES_PER_REQUEST <= num_available_pages;
  }

  bool CanJumpForward(const RequestStateEntry& rsentry) {
    if (rsentry->request->generation_cfg->logprobs) {
      return false;
    }
    if (!rsentry->mstates[0]->grammar_state_matcher) {
      return false;
    }
    return true;
  }

  std::pair<int, std::vector<int32_t>> RetokenizeWithNewString(RequestModelState mstate,
                                                               const std::string& new_string,
                                                               int max_rollback_tokens) {
    // Step 1. Get past tokens
    // past_tokens = mstate[-max_rollback_tokens:]
    // past_string = detokenize(past_tokens)
    const auto& token_table = tokenizer_->PostProcessedTokenTable();
    std::vector<int32_t> past_tokens;
    std::string past_string;
    auto past_begin_it = mstate->committed_tokens.size() >= max_rollback_tokens
                             ? mstate->committed_tokens.end() - max_rollback_tokens
                             : mstate->committed_tokens.begin();
    for (auto it = past_begin_it; it != mstate->committed_tokens.end(); ++it) {
      past_tokens.push_back(it->GetTokenId());
      past_string += token_table[it->GetTokenId()];
    }

    // Step 2. Retokenize
    // Compare tokenize(past_string + new_string) and past_tokens
    auto new_tokens = tokenizer_->EncodeNoPrependSpace(past_string + new_string);

    // Pop last token
    if (tokenizer_->GetPrefixTokenMask()[new_tokens.back()]) {
      auto last_token = token_table[new_tokens.back()];
      if (last_token.length() >= new_string.length()) {
        return {0, {}};
      }

      new_tokens.pop_back();
    }

    int first_differ_idx = past_tokens.size();
    for (int i = 0; i < static_cast<int>(past_tokens.size()); ++i) {
      if (i == static_cast<int>(new_tokens.size()) || past_tokens[i] != new_tokens[i]) {
        first_differ_idx = i;
        break;
      }
    }

    return {past_tokens.size() - first_differ_idx,
            std::vector<int32_t>(new_tokens.begin() + first_differ_idx, new_tokens.end())};
  }

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  Tokenizer tokenizer_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  const int MAX_ROLLBACK_TOKENS_ = 10;
};

EngineAction EngineAction::BatchJumpForward(Array<Model> models, Tokenizer tokenizer,
                                            Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<BatchJumpForwardActionObj>(
      std::move(models), std::move(tokenizer), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
