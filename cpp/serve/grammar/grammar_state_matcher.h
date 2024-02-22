/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_matcher.h
 * \brief The header for the support of matching characters to a BNF grammar. This is the core
 * logic of the grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../../support/dynamic_bitset.h"
#include "../../support/encoding.h"
#include "grammar.h"
#include "grammar_tokenizer_config.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief Match character or string or tokens to the given BNF grammar. This class is the core logic
 * of the grammar-guided generation.
 * \details This class implements the non-deterministic pushdown automaton (NPDA) matching algorithm
 * to match a string to a BNF grammar. It keep track of the current state of the matching process by
 * maintaining several stacks internally as possible paths in the NPDA. Therefore, it supports
 * continuous matching of characters and backtracking.
 *
 * It also supports detecting the rejected tokens at the current position, which helps the
 * grammar-guided generation.
 */
class GrammarStateMatcherNode : public Object {
 public:
  /*!
   * \brief Accept one unicode character to the current state.
   * \param codepoint The unicode codepoint of the character to be accepted.
   * \param drop_old If true, the old state will be dropped after accepting the new character when
   * the number of states exceeds the limit of saved history.
   * \param verbose If true, the function will print the previous state, the new state and the
   * matching status of the new character. Mainly for debugging purpose.
   */
  virtual bool AcceptToken(int32_t token_id) = 0;

  /*!
   * \brief Given a tokenizer config, find the tokens that are rejected at the current position.
   * \param tokenizer_config Information about the current tokenizer, which is particularly useful
   * for efficient matching.
   * \param rejected_ids A mutable bitset. This funciton will set its size to the number of tokens
   * of the tokenizer, the positions of rejected tokens to 1, and other positions to 0.
   * \sa mlc::llm::serve::Sampler.
   */
  virtual void FindNextTokenBitmask(DLTensor* next_token_bitmask) = 0;

  /*!
   * \brief Rollback the matcher to a previous state.
   * \param rollback_steps The number of steps to rollback. Should not be greater than the number of
   * steps in the current history.
   */
  virtual void Rollback(int num_tokens) = 0;

  /*!
   * \brief Print the current state of the matcher to a string as a set of stacks, mainly for
   * debugging purpose.
   * \param steps_behind_latest The number of steps behind the latest state to print. If 0, print
   * the latest state.
   */
  virtual std::string PrintMatcherState(int steps_behind_latest = 0) const = 0;

  static constexpr const char* _type_key = "mlc.serve.GrammarStateMatcher";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarStateMatcherNode, Object);
};

class TokenInfoForMatcher {
 public:
  TokenInfoForMatcher(const BNFGrammar& grammar, const std::vector<std::string>& token_table);

 private:
  class TokenInfoForMatcherData;
  std::shared_ptr<TokenInfoForMatcherData> data_;
  friend class GrammarStateMatcherNode;
};

class GrammarStateMatcher : public ObjectRef {
 public:
  /*!
   * \brief Construct a grammar matcher from a BNF grammar.
   * \param grammar The BNF grammar to match.
   * \param max_rollback_steps The maximum number of steps to rollback when backtracking.
   * \param init_rule_position The initial position of the grammar matcher. If not specified, the
   * matcher will start from all start positions in the main rule.
   */
  GrammarStateMatcher(const BNFGrammar& grammar, const TokenInfoForMatcher& token_info,
                      int max_rollback_tokens = 0);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarStateMatcher, ObjectRef, GrammarStateMatcherNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
