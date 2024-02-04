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

struct KnownStateTokens {
  std::unordered_set<int32_t> accepted_ids;
  std::unordered_set<int32_t> rejected_ids;
  std::unordered_set<int32_t> unknown_state_ids;
  const std::unordered_map<int32_t, TokenAndId>* token_lookup_map = nullptr;

  KnownStateTokens() = default;

  KnownStateTokens(const std::unordered_map<int32_t, TokenAndId>* token_lookup_map,
                   const std::unordered_set<int32_t>& accepted_ids,
                   const std::unordered_set<int32_t>& rejected_ids,
                   const std::unordered_set<int32_t>& unknown_state_ids)
      : token_lookup_map(token_lookup_map) {
    auto size_acc = accepted_ids.size();
    auto size_rej = rejected_ids.size();
    auto size_unk = unknown_state_ids.size();
    int not_save_index =
        size_acc < size_rej && size_acc < size_unk ? 0 : (size_rej < size_unk ? 1 : 2);

    if (not_save_index != 0) {
      this->accepted_ids = accepted_ids;
    }
    if (not_save_index != 1) {
      this->rejected_ids = rejected_ids;
    }
    if (not_save_index != 2) {
      this->unknown_state_ids = unknown_state_ids;
    }
  }

  bool IsAccepted(int32_t token_id) const {
    return accepted_ids.count(token_id) > 0 ||
           (token_lookup_map->count(token_id) && !rejected_ids.count(token_id) &&
            !unknown_state_ids.count(token_id));
  }

  bool IsRejected(int32_t token_id) const {
    return rejected_ids.count(token_id) > 0 ||
           (token_lookup_map->count(token_id) && !accepted_ids.count(token_id) &&
            !unknown_state_ids.count(token_id));
  }
};

class GrammarTokenizerConfigNode : public Object {
 public:
  size_t vocab_size;
  std::vector<TokenAndId> sorted_token_and_ids;
  std::vector<int32_t> stop_token_ids;
  std::vector<int32_t> special_token_ids;
  std::unordered_map<int32_t, TokenAndId> token_lookup_map;
  std::unordered_map<SequenceIdAndPosition, KnownStateTokens, SequenceIdAndPositionHash>
      known_state_tokens;

  bool IsTokenRejectedForRulePosition(int32_t token_id, int32_t rule_id, int32_t element_id) const {
    auto it = known_state_tokens.find({rule_id, element_id});
    if (it == known_state_tokens.end()) {
      return false;
    }
    return it->second.IsRejected(token_id);
  }

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
