/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_TOKENIZER_CONFIG_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_TOKENIZER_CONFIG_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "../../support/encoding.h"
#include "../../tokenizers.h"
#include "grammar.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*! \brief A token and its id. */
struct TokenAndId {
  std::vector<TCodepoint> token;
  int32_t id;
  /*! \brief Compare tokens by their unicode codepoint sequence. */
  bool operator<(const TokenAndId& other) const;
};

/*!
 * \brief Preprocessed information, for a given specific rule and position, divides the token set
 * into three categories:
 * - accepted_indices: If a token is accepted by this rule
 * - rejected_indices: If a token is rejected by this rule
 * - uncertain_indices: If a prefix of a token is accepted by this rule and comes to the end
 * of the rule. In real matching process, it still can be accepted or rejected by some parent rule.
 * So it is uncertain.
 * \note Since the union of these three sets is the whole token set, we only need to store the
 * smaller two sets. The unsaved set is specified by not_saved_index.
 * \note These indices are the indices of sorted_token_and_ids in the GrammarTokenizerConfig object,
 * instead of the token ids. That helps the matching process.
 */
struct CatagorizedTokens {
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  std::vector<int32_t> uncertain_indices;
  enum class NotSavedIndex { kAccepted = 0, kRejected = 1, kUncertain = 2 };
  NotSavedIndex not_saved_index;

  CatagorizedTokens() = default;

  CatagorizedTokens(std::vector<int32_t>&& accepted_indices,
                    std::vector<int32_t>&& rejected_indices,
                    std::vector<int32_t>&& uncertain_indices) {
    auto size_acc = accepted_indices.size();
    auto size_rej = rejected_indices.size();
    auto size_unc = uncertain_indices.size();
    not_saved_index =
        (size_acc >= size_rej && size_acc >= size_unc)
            ? NotSavedIndex::kAccepted
            : (size_rej >= size_unc ? NotSavedIndex::kRejected : NotSavedIndex::kUncertain);

    if (not_saved_index != NotSavedIndex::kAccepted) {
      this->accepted_indices = std::move(accepted_indices);
    }
    if (not_saved_index != NotSavedIndex::kRejected) {
      this->rejected_indices = std::move(rejected_indices);
    }
    if (not_saved_index != NotSavedIndex::kUncertain) {
      this->uncertain_indices = std::move(uncertain_indices);
    }
  }
};

/*!
 * \brief All information that we need to match tokens in the tokenizer to the specified grammar.
 * It is the result of preprocessing.
 * \sa mlc::llm::serve::GrammarMatcher
 * \sa mlc::llm::serve::Sampler
 */
class GrammarTokenizerConfigNode : public Object {
 public:
  /*! \brief The vocabulary size of the tokenizer. */
  size_t vocab_size;
  /*! \brief The sorted token and its id. Tokens are sorted to reuse the common prefix during
   * matching. */
  std::vector<TokenAndId> sorted_token_and_ids;
  /*! \brief The stop tokens. They can be accepted iff GramamrMatcher can reach the end of the
   * grammar. */
  std::vector<int32_t> stop_token_ids;
  /*! \brief The special tokens. Currently we will ignore these tokens during grammar-guided
   * matching. */
  std::vector<int32_t> special_token_ids;
  /*! \brief Mapping from token id to token. */
  std::unordered_map<int32_t, TokenAndId> token_lookup_map;

  /*! \brief A sequence id and its position. */
  struct SequenceIdAndPosition {
    int32_t sequence_id;
    int32_t element_id;
    bool operator==(const SequenceIdAndPosition& other) const {
      return sequence_id == other.sequence_id && element_id == other.element_id;
    }
  };

  /*! \brief Hash function for SequenceIdAndPosition. */
  struct SequenceIdAndPositionHash {
    std::size_t operator()(const SequenceIdAndPosition& k) const {
      return std::hash<int32_t>()(k.sequence_id) ^ (std::hash<int32_t>()(k.element_id) << 1);
    }
  };

  /*! \brief Mapping from sequence id and its position to the catagorized tokens. */
  std::unordered_map<SequenceIdAndPosition, CatagorizedTokens, SequenceIdAndPositionHash>
      catagorized_tokens_for_grammar;

  /*! \brief Tokenizer will replace space with a special underscore. We will replace it back to
   * space. */
  static constexpr const char* kSpecialUnderscore = "▁";
};

class GrammarTokenizerConfig : public ObjectRef {
 public:
  /*! \brief Construct a GrammarTokenizerConfig from a tokenizer and a grammar. The grammar should
   * be the same as the grammar used to construct the GrammarMatcher. */
  GrammarTokenizerConfig(const Tokenizer& tokenizer, const BNFGrammar& grammar);

  TVM_DEFINE_OBJECT_REF_METHODS(GrammarTokenizerConfig, ObjectRef, GrammarTokenizerConfigNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_TOKENIZER_CONFIG_H_
