/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/sampler.h
 * \brief The header for runtime module of sampler functions.
 */

#ifndef MLC_LLM_SERVE_SAMPLER_H_
#define MLC_LLM_SERVE_SAMPLER_H_

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "../base.h"
#include "../random.h"
#include "event_trace_recorder.h"
#include "model.h"
#include "request_state.h"

namespace mlc {
namespace llm {
namespace serve {

using tvm::Device;
using namespace tvm::runtime;

/*!
 * \brief The base class of runtime sampler.
 * Its main function is `BatchSampleTokens`, which takes a batch of
 * logits and corresponding configuration, and sample one token
 * for each instance of the batch.
 */
class SamplerObj : public Object {
 public:
  /*!
   * \brief Sample tokens from the input batch of logits.
   * \param logits_on_device The logits to sample tokens from.
   * \param model The LLM model which contains the softmax
   * function on device that might be used to compute probability distribution.
   * \param request_mstates The request states of each sequence in
   * the batch with regard to the given model.
   * \param generation_cfg The generation config of each request
   * in the input batch.
   * \param rngs The random number generator of each sequence.
   * \param output_prob_dist The output probability distribution
   * \param output_token_probs The output token probabilities
   * \return The sampled tokens, one for each request in the batch.
   */
  virtual std::vector<int32_t> BatchSampleTokens(
      NDArray logits_on_device, Model model, Array<RequestModelState> request_mstates,
      Array<GenerationConfig> generation_cfg, const std::vector<RandomGenerator*>& rngs,
      TokenizerConfig tokenizer_config, std::vector<NDArray>* output_prob_dist = nullptr,
      std::vector<float>* output_token_probs = nullptr) = 0;

  /*!
   * \brief Verify draft tokens generated by small models in the large model
   * in speculative decoding. The input corresponds to a batch of sequences.
   * \param logits_on_device The logits of the large model.
   * \param cum_verify_lengths The cumulative draft lengths to verify of all sequences.
   * \param model The LLM model which contains the softmax
   * function on device that might be used to compute probability distribution.
   * \param request_mstates The request states of each sequence in
   * the batch with regard to the large model.
   * \param generation_cfg The generation config of each request
   * in the input batch.
   * \param rngs The random number generator of each sequence.
   * \param draft_output_tokens The draft tokens generated by the small model for
   * each sequence.
   * \param draft_output_token_prob The draft tokens' probabilities computed from
   * the small model for each sequence.
   * \param draft_output_prob_dist The probability distribution computed from the
   * small model for each sequence.
   * \return The list of accepted tokens for each request.
   */
  virtual std::vector<std::vector<int32_t>> BatchVerifyDraftTokens(
      NDArray logits_on_device, const std::vector<int>& cum_verify_lengths, Model model,
      const Array<RequestModelState>& request_mstates,
      const Array<GenerationConfig>& generation_cfg, const std::vector<RandomGenerator*>& rngs,
      const std::vector<std::vector<int>>& draft_output_tokens,
      const std::vector<std::vector<float>>& draft_output_token_prob,
      const std::vector<std::vector<NDArray>>& draft_output_prob_dist) = 0;

  static constexpr const char* _type_key = "mlc.serve.Sampler";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(SamplerObj, Object);
};

class Sampler : public ObjectRef {
 public:
  /*!
   * \brief Create the runtime sampler module.
   * \param sampler_kind The sampler name denoting which sampler to create.
   * \param trace_recorder The event trace recorder for requests.
   * \return The created runtime module.
   */
  TVM_DLL static Sampler Create(std::string sampler_kind,
                                Optional<EventTraceRecorder> trace_recorder);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Sampler, ObjectRef, SamplerObj);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_SAMPLER_H_
