/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine_actions/batch_decode.cc
 */

#include <tvm/runtime/nvtx.h>

#include <numeric>

#include "../../support/random.h"
#include "../config.h"
#include "../model.h"
#include "../sampler/sampler.h"
#include "action.h"
#include "action_commons.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief The action that runs one-step decode for requests in the
 * `running_queue` of engine state. Preempt low-priority requests
 * accordingly when it is impossible to decode all the running requests.
 * \note The BatchDecode action **does not** take effect for speculative
 * decoding scenarios where there are multiple models. For speculative
 * decoding in the future, we will use other specific actions.
 */
class BatchDecodeActionObj : public EngineActionObj {
 public:
  explicit BatchDecodeActionObj(Array<Model> models, Tokenizer tokenizer,
                                LogitProcessor logit_processor, Sampler sampler,
                                EngineConfig engine_config,
                                Optional<EventTraceRecorder> trace_recorder)
      : models_(std::move(models)),
        tokenizer_(std::move(tokenizer)),
        logit_processor_(std::move(logit_processor)),
        sampler_(std::move(sampler)),
        engine_config_(std::move(engine_config)),
        trace_recorder_(std::move(trace_recorder)) {}

  Array<Request> Step(EngineState estate) final {
    // - Do not run decode when there are multiple models or no running requests.
    if (models_.size() > 1 || estate->running_queue.empty()) {
      return {};
    }

    // Preempt request state entries when decode cannot apply.
    std::vector<RequestStateEntry> running_rsentries;
    {
      NVTXScopedRange nvtx_scope("BatchDecode getting requests");
      running_rsentries = GetRunningRequestStateEntries(estate);
      while (!CanDecode(running_rsentries.size())) {
        if (estate->prefix_cache->TryFreeMemory()) continue;
        RequestStateEntry preempted =
            PreemptLastRunningRequestStateEntry(estate, models_, NullOpt, trace_recorder_);
        if (preempted.same_as(running_rsentries.back())) {
          running_rsentries.pop_back();
        }
      }
    }

    auto tstart = std::chrono::high_resolution_clock::now();

    // NOTE: Right now we only support decode all the running request states at a time.
    int num_rsentries = running_rsentries.size();
    ICHECK_GT(num_rsentries, 0)
        << "There should be at least one request state entry that can run decode. "
           "Possible failure reason: none of the prefill phase of the running requests is finished";
    ICHECK_LE(num_rsentries, engine_config_->max_num_sequence)
        << "The number of running requests exceeds the max number of sequence in EngineConfig. "
           "Possible failure reason: the prefill action allows new sequence in regardless of the "
           "max num sequence.";
    // Collect
    // - the last committed token,
    // - the request id,
    // - the generation config,
    // - the random number generator,
    // of each request state entry.
    std::vector<int> input_tokens;
    std::vector<int> lengths;
    Array<String> request_ids;
    std::vector<int64_t> request_internal_ids;
    Array<RequestModelState> mstates;
    Array<GenerationConfig> generation_cfg;
    std::vector<RandomGenerator*> rngs;
    bool is_all_request_single_token = true;

    input_tokens.reserve(num_rsentries);
    request_ids.reserve(num_rsentries);
    request_internal_ids.reserve(num_rsentries);
    mstates.reserve(num_rsentries);
    generation_cfg.reserve(num_rsentries);
    rngs.reserve(num_rsentries);

    for (const RequestStateEntry& rsentry : running_rsentries) {
      auto mstate = rsentry->mstates[0];
      // std::cout << "num tokens for next decode: " << mstate->num_pending_kv_cache_tokens
      //           << std::endl;
      ICHECK(mstate->num_pending_kv_cache_tokens > 0 &&
             mstate->num_pending_kv_cache_tokens <=
                 static_cast<int>(mstate->committed_tokens.size()));

      for (auto begin = mstate->committed_tokens.end() - mstate->num_pending_kv_cache_tokens;
           begin != mstate->committed_tokens.end(); ++begin) {
        input_tokens.push_back(begin->GetTokenId());
      }
      lengths.push_back(mstate->num_pending_kv_cache_tokens);
      is_all_request_single_token =
          is_all_request_single_token && mstate->num_pending_kv_cache_tokens == 1;

      mstate->num_pending_kv_cache_tokens = 0;

      request_ids.push_back(rsentry->request->id);
      request_internal_ids.push_back(mstate->internal_id);
      mstates.push_back(mstate);
      generation_cfg.push_back(rsentry->request->generation_cfg);
      rngs.push_back(&rsentry->rng);
    }

    // - Compute embeddings.
    RECORD_EVENT(trace_recorder_, request_ids, "start embedding");
    ObjectRef embeddings =
        models_[0]->TokenEmbed({IntTuple(input_tokens.begin(), input_tokens.end())});
    RECORD_EVENT(trace_recorder_, request_ids, "finish embedding");

    // - Invoke model decode.
    RECORD_EVENT(trace_recorder_, request_ids, "start decode");
    NDArray logits;
    if (is_all_request_single_token) {
      logits = models_[0]->BatchDecode(embeddings, request_internal_ids);
      ICHECK_EQ(logits->ndim, 3);
      ICHECK_EQ(logits->shape[0], num_rsentries);
      ICHECK_EQ(logits->shape[1], 1);
    } else {
      logits = models_[0]->BatchPrefill(embeddings, request_internal_ids, lengths);
      ICHECK_EQ(logits->ndim, 3);
      ICHECK_EQ(logits->shape[0], 1);
      ICHECK_EQ(logits->shape[1], num_rsentries);
    }
    RECORD_EVENT(trace_recorder_, request_ids, "finish decode");

    // - Update logits.
    logits = logits.CreateView({num_rsentries, logits->shape[2]}, logits->dtype);
    logit_processor_->InplaceUpdateLogits(logits, generation_cfg, mstates, request_ids);

    // - Compute probability distributions.
    NDArray probs_on_device =
        logit_processor_->ComputeProbsFromLogits(logits, generation_cfg, request_ids);

    // - Sample tokens.
    // Fill range [0, num_rsentries) into `sample_indices`.
    std::vector<int> sample_indices(num_rsentries);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    NDArray renormalized_probs = sampler_->BatchRenormalizeProbsByTopP(
        probs_on_device, sample_indices, request_ids, generation_cfg);
    std::vector<SampleResult> sample_results = sampler_->BatchSampleTokensWithProbAfterTopP(
        renormalized_probs, sample_indices, request_ids, generation_cfg, rngs);
    ICHECK_EQ(sample_results.size(), num_rsentries);

    // - Update the committed tokens of states.
    for (int i = 0; i < num_rsentries; ++i) {
      auto mstate = mstates[i];
      std::cout << "decode result: " << sample_results[i].GetTokenId() << " <"
                << tokenizer_->PostProcessedTokenTable()[sample_results[i].GetTokenId()]
                << "> other choices: ";
      for (auto i : sample_results[i].top_prob_tokens) {
        std::cout << "(" << i.first << ", " << i.second << ") <"
                  << tokenizer_->PostProcessedTokenTable()[i.first] << "> ";
      }
      std::cout << std::endl;

      if (!mstate->require_retokenization_in_next_decode) {
        auto start = std::chrono::high_resolution_clock::now();
        mstates[i]->CommitToken(sample_results[i]);
        // std::cout << "commit time: "
        //           << std::chrono::duration_cast<std::chrono::microseconds>(
        //                  std::chrono::high_resolution_clock::now() - start)
        //                  .count()
        //           << " us" << std::endl;
      } else {
        auto start = std::chrono::high_resolution_clock::now();
        CommitTokenMayRetokenize(running_rsentries[i], mstate, sample_results[i]);
        mstate->require_retokenization_in_next_decode = false;
        // std::cout << "commit retokenize time: "
        //           << std::chrono::duration_cast<std::chrono::microseconds>(
        //                  std::chrono::high_resolution_clock::now() - start)
        //                  .count()
        //           << " us" << std::endl;
      }
      // Metrics update
      // live update the output metrics
      running_rsentries[i]->rstate->metrics.num_output_tokens += 1;
    }

    auto tend = std::chrono::high_resolution_clock::now();

    // std::cout << "Decode time: "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count()
    //           << " us" << std::endl;
    double elapsed_time = static_cast<double>((tend - tstart).count()) / 1e9;
    estate->metrics.engine_decode_time_sum += elapsed_time;
    estate->metrics.UpdateDecodeTimeByBatchSize(num_rsentries, elapsed_time);

    return estate->running_queue;
  }

 private:
  /*! \brief Check if the input request state entries can be decoded under conditions. */
  bool CanDecode(int num_rsentries) {
    int num_available_pages = models_[0]->GetNumAvailablePages();
    return num_rsentries <= num_available_pages;
  }

  void CommitTokenMayRetokenize(RequestStateEntry rsentry, RequestModelState mstate,
                                const SampleResult& sample_result) {
    auto generation_cfg = rsentry->request->generation_cfg;
    if (!generation_cfg->debug_config.ignore_eos &&
        std::any_of(generation_cfg->stop_token_ids.begin(), generation_cfg->stop_token_ids.end(),
                    [&](int32_t token) { return token == sample_result.GetTokenId(); })) {
      mstate->CommitToken(sample_result);
      return;
    }

    const auto& committed_tokens = mstate->committed_tokens;
    std::vector<int> past_tokens;

    for (auto start_it = static_cast<int>(committed_tokens.size()) <= MAX_ROLLBACK_TOKENS_
                             ? committed_tokens.begin()
                             : committed_tokens.end() - MAX_ROLLBACK_TOKENS_;
         start_it != committed_tokens.end(); ++start_it) {
      past_tokens.push_back(start_it->GetTokenId());
    }

    // std::cout << "past tokens: ";
    // for (auto i : past_tokens) {
    //   std::cout << i << " <" << tokenizer_->IdToToken(i) << "> ";
    // }
    // std::cout << std::endl;

    std::string new_string = tokenizer_->DecodeNoStripSpace(past_tokens) +
                             tokenizer_->PostProcessedTokenTable()[sample_result.GetTokenId()];
    // std::cout << "new string: <" << new_string << ">" << std::endl;
    std::vector<int32_t> new_tokens = tokenizer_->EncodeNoPrependSpace(new_string);

    // std::cout << "new tokens: ";
    // for (auto i : new_tokens) {
    //   std::cout << i << " <" << tokenizer_->IdToToken(i) << "> ";
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
        not_callback_tokens.push_back(mstate->committed_tokens[i].GetTokenId());
      }
      not_callback_tokens.push_back(sample_result.GetTokenId());
      rsentry->extra_prefix_string += tokenizer_->DecodeNoStripSpace(not_callback_tokens);
      int next_callback_token_pos = static_cast<int>(mstate->committed_tokens.size()) -
                                    static_cast<int>(past_tokens.size()) +
                                    static_cast<int>(new_tokens.size());
      rsentry->next_callback_token_pos = next_callback_token_pos;
    }

    if (rollback_len > 0) {
      mstate->RollbackTokens(rollback_len);
      models_[0]->PopNFromKVCache(mstate->internal_id, rollback_len);
    }

    for (int i = same_size; i < static_cast<int>(new_tokens.size()); ++i) {
      mstate->CommitToken({{new_tokens[i], 1.0}, {}});
    }
  }

  /*!
   * \brief The model to run decode in. When there are multiple
   * models, the `Step` function of the created action will not take effect.
   */
  Array<Model> models_;
  Tokenizer tokenizer_;
  /*! \brief The logit processor. */
  LogitProcessor logit_processor_;
  /*! \brief The sampler to sample new tokens. */
  Sampler sampler_;
  /*! \brief The engine config. */
  EngineConfig engine_config_;
  /*! \brief Event trace recorder. */
  Optional<EventTraceRecorder> trace_recorder_;
  const int MAX_ROLLBACK_TOKENS_ = 10;
};

EngineAction EngineAction::BatchDecode(Array<Model> models, Tokenizer tokenizer,
                                       LogitProcessor logit_processor, Sampler sampler,
                                       EngineConfig engine_config,
                                       Optional<EventTraceRecorder> trace_recorder) {
  return EngineAction(make_object<BatchDecodeActionObj>(
      std::move(models), std::move(tokenizer), std::move(logit_processor), std::move(sampler),
      std::move(engine_config), std::move(trace_recorder)));
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
