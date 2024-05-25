/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_verify.cc
 */

#include <tvm/runtime/nvtx.h>
#include <tvm/runtime/threading_backend.h>

#include <cmath>
#include <exception>

#include "../../random.h"
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
      while (!CanJumpForward(running_rsentries.size())) {
        RequestStateEntry preempted =
            PreemptLastRunningRequestStateEntry(estate, models_, trace_recorder_);
        if (preempted.same_as(running_rsentries.back())) {
          running_rsentries.pop_back();
        }
      }
    }

    if (running_rsentries.empty()) {
      return {};
    }

    std::cout << "in jumpforward\n";

    auto tstart = std::chrono::high_resolution_clock::now();

    for (auto rsentry : running_rsentries) {
      auto mstate = rsentry->mstates[0];
      if (!mstate->grammar_state_matcher) {
        continue;
      }
      auto jump_forward_str = mstate->grammar_state_matcher.value()->FindJumpForwardString();

      if (jump_forward_str.empty()) {
        continue;
      }

      // std::cout << "request id: " << rsentry->request->id << " jump forward: <" <<
      // jump_forward_str
      //           << ">\n";

      std::vector<int32_t> past_tokens;

      auto it_past_tokens = mstate->committed_tokens.size() >= CHECK_PREV_TOKENS_CNT_
                                ? mstate->committed_tokens.end() - CHECK_PREV_TOKENS_CNT_
                                : mstate->committed_tokens.begin();
      for (; it_past_tokens != mstate->committed_tokens.end(); ++it_past_tokens) {
        past_tokens.push_back(it_past_tokens->sampled_token_id.first);
      }

      auto past_string = tokenizer_->DecodeNoBOS(past_tokens);
      auto new_tokens = tokenizer_->EncodeNoBOS(past_string + jump_forward_str);

      // print past tokens
      // std::cout << "past tokens: ";
      // for (int i = 0; i < past_tokens.size(); ++i) {
      //   std::cout << past_tokens[i] << " <" << tokenizer_->IdToToken(past_tokens[i]) << "> ";
      // }
      // std::cout << std::endl;

      // print new tokens
      // std::cout << "new tokens: ";
      // for (int i = 0; i < new_tokens.size(); ++i) {
      //   std::cout << new_tokens[i] << " <" << tokenizer_->IdToToken(new_tokens[i]) << "> ";
      // }
      // std::cout << std::endl;

      int same_size = past_tokens.size();
      for (int i = 0; i < static_cast<int>(past_tokens.size()); ++i) {
        if (i == static_cast<int>(new_tokens.size()) || past_tokens[i] != new_tokens[i]) {
          same_size = i;
          break;
        }
      }

      // Rollback
      int rollback_len = past_tokens.size() - same_size;

      if (rollback_len >
          static_cast<int>(mstate->committed_tokens.size()) - rsentry->next_callback_token_pos) {
        // handle output
        std::vector<int32_t> not_callback_tokens;
        for (auto i = rsentry->next_callback_token_pos; i < mstate->committed_tokens.size(); ++i) {
          not_callback_tokens.push_back(mstate->committed_tokens[i].sampled_token_id.first);
        }
        rsentry->next_callback_prefix_string +=
            tokenizer_->DecodeNoBOS(not_callback_tokens) + jump_forward_str;
        int next_callback_token_pos = static_cast<int>(mstate->committed_tokens.size()) -
                                      static_cast<int>(past_tokens.size()) +
                                      static_cast<int>(new_tokens.size());
        rsentry->additional_num_delta_tokens +=
            next_callback_token_pos - rsentry->next_callback_token_pos;
        rsentry->next_callback_token_pos = next_callback_token_pos;
      }

      if (rollback_len > 0) {
        mstate->RollbackTokens(rollback_len);
      }

      if (rollback_len > mstate->num_tokens_for_next_decode) {
        models_[0]->PopNFromKVCache(mstate->internal_id,
                                    rollback_len - mstate->num_tokens_for_next_decode);
      }

      mstate->num_tokens_for_next_decode =
          std::max(0, mstate->num_tokens_for_next_decode - rollback_len);

      // std::cout << "past size: " << past_tokens.size() << " new size: " << new_tokens.size()
      //           << " same size: " << same_size << std::endl;

      // std::cout << "request id: " << rsentry->request->id << " rollback tokens:";
      // for (int i = 0; i < rollback_len; ++i) {
      //   std::cout << past_tokens[past_tokens.size() - rollback_len + i] << " <"
      //             << tokenizer_->IdToToken(past_tokens[past_tokens.size() - rollback_len + i])
      //             << "> ";
      // }
      // std::cout << std::endl;

      // Add new tokens
      for (int i = same_size; i < new_tokens.size(); ++i) {
        // std::cout << "new tokens: " << new_tokens[i] << " <" <<
        // tokenizer_->IdToToken(new_tokens[i])
        //           << ">" << std::endl;
        mstate->CommitToken({{new_tokens[i], 1.0}, {}});
      }

      mstate->require_retokenization_in_next_decode = true;
    }

    auto tend = std::chrono::high_resolution_clock::now();
    estate->stats.engine_total_decode_time += static_cast<double>((tend - tstart).count()) / 1e9;

    return {};
  }

 private:
  /*! \brief Check if the input request state entries can be decoded under conditions. */
  bool CanJumpForward(int num_rsentries) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries <= num_available_pages;
  }

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  Tokenizer tokenizer_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  const int CHECK_PREV_TOKENS_CNT_ = 10;
};

EngineAction EngineAction::BatchJumpForward(Array<Model> models, Tokenizer tokenizer,
                                            Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<BatchJumpForwardActionObj>(
      std::move(models), std::move(tokenizer),
      std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
