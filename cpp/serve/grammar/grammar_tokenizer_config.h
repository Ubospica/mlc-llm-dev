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

#include "../../tokenizers.h"
#include "../encoding.h"
#include "grammar.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

struct TokenAndId {
  std::vector<TCodepoint> token;
  int32_t id;
  bool operator<(const TokenAndId& other) const;
};

struct SequenceIdAndPosition {
  int32_t sequence_id;
  int32_t element_id;
  bool operator==(const SequenceIdAndPosition& other) const {
    return sequence_id == other.sequence_id && element_id == other.element_id;
  }
};

struct SequenceIdAndPositionHash {
  std::size_t operator()(const SequenceIdAndPosition& k) const {
    return std::hash<int32_t>()(k.sequence_id) ^ (std::hash<int32_t>()(k.element_id) << 1);
  }
};

struct CatagorizedTokensForGrammar {
  std::unordered_set<int32_t> accepted_indices;
  std::unordered_set<int32_t> rejected_indices;
  std::unordered_set<int32_t> uncertain_indices;
  enum class NotSavedIndex { kAccepted = 0, kRejected = 1, kUncertain = 2 };
  NotSavedIndex not_saved_index;

  CatagorizedTokensForGrammar() = default;

  CatagorizedTokensForGrammar(const std::unordered_set<int32_t>& accepted_indices,
                              const std::unordered_set<int32_t>& rejected_indices,
                              const std::unordered_set<int32_t>& uncertain_indices) {
    auto size_acc = accepted_indices.size();
    auto size_rej = rejected_indices.size();
    auto size_unc = uncertain_indices.size();
    not_saved_index =
        size_acc < size_rej && size_acc < size_unc
            ? NotSavedIndex::kAccepted
            : (size_rej < size_unc ? NotSavedIndex::kRejected : NotSavedIndex::kUncertain);

    if (not_saved_index != NotSavedIndex::kAccepted) {
      this->accepted_indices = accepted_indices;
    }
    if (not_saved_index != NotSavedIndex::kRejected) {
      this->rejected_indices = rejected_indices;
    }
    if (not_saved_index != NotSavedIndex::kUncertain) {
      this->uncertain_indices = uncertain_indices;
    }
  }
};

class GrammarTokenizerConfigNode : public Object {
 public:
  size_t vocab_size;
  std::vector<TokenAndId> sorted_token_and_ids;
  std::vector<int32_t> stop_token_ids;
  std::vector<int32_t> special_token_ids;
  std::unordered_map<int32_t, TokenAndId> token_lookup_map;
  std::unordered_map<SequenceIdAndPosition, CatagorizedTokensForGrammar, SequenceIdAndPositionHash>
      catagorized_tokens_for_grammar;

  static constexpr const char* kSpecialUnderscore = "▁";
};

class GrammarTokenizerConfig : public ObjectRef {
 public:
  GrammarTokenizerConfig(const Tokenizer& tokenizer, const BNFGrammar& grammar);

  TVM_DEFINE_OBJECT_REF_METHODS(GrammarTokenizerConfig, ObjectRef, GrammarTokenizerConfigNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_TOKENIZER_CONFIG_H_
