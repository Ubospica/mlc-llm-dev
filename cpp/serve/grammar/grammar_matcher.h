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
 * \brief Refers to a specific position in the grammar. Used to specify the initial position of
 * the grammar matcher.
 * \sa mlc::llm::serve::GrammarMatcher::GrammarMatcher
 */
struct RulePosition {
  /*! \brief The rule's id. */
  int32_t rule_id = -1;
  /*! \brief Which choice in this rule is selected. */
  int32_t sequence_id = -1;
  /*! \brief Which element of the choice sequence is being visited. */
  int32_t element_id = -1;
  /*!
   * \brief If the element refers to another rule, and another rule is a star quantifier of
   * a character class, this field will be set to the id of the character class.
   * This is part of the special support of star quantifiers of character classes.
   * \sa mlc::llm::serve::BNFGrammarNode.
   */
  int32_t char_class_id = -1;
};

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
class GrammarMatcherNode : public Object {
 public:
  /*!
   * \brief Accept one unicode character to the current state.
   * \param codepoint The unicode codepoint of the character to be accepted.
   * \param drop_old If true, the old state will be dropped after accepting the new character when
   * the number of states exceeds the limit of saved history.
   * \param verbose If true, the function will print the previous state, the new state and the
   * matching status of the new character. Mainly for debugging purpose.
   */
  virtual bool AcceptChar(TCodepoint codepoint, bool drop_old = true, bool verbose = false) = 0;

  /*!
   * \brief Returns true if the matcher already reached the end of the grammar.
   * \note Since the matcher maintains a non-deterministic state internally, even though the matcher
   * reaches the end, it may still have other paths that can continue to accept new characters.
   */
  virtual bool CanReachEnd() const = 0;

  /*!
   * \brief Given a tokenizer config, find the tokens that are rejected at the current position.
   * \param tokenizer_config Information about the current tokenizer, which is particularly useful
   * for efficient matching.
   * \param rejected_ids A mutable bitset. This funciton will set its size to the number of tokens
   * of the tokenizer, the positions of rejected tokens to 1, and other positions to 0.
   * \sa mlc::llm::serve::Sampler.
   */
  virtual void FindRejectedTokens(const GrammarTokenizerConfig& tokenizer_config,
                                  DynamicBitSet* rejected_ids) = 0;

  /*!
   * \brief Preprocess the token set according to the specified position in the grammar. This is
   * part of the preprocessing for GrammarTokenizerConfig.
   * \param sorted_token_and_ids The token set to be preprocessed. It should be sorted by the token
   * codepoints.
   * \param is_main_rule Whether the specified position is the main rule of the grammar.
   * \return A CatagorizedTokens object. It splits the given token set into three parts: accepted
   * by the current position, rejected by the current position, and uncertain.
   * \sa mlc::llm::serve::GrammarTokenizerConfig.
   */
  virtual CatagorizedTokens GetCatagorizedTokens(
      const std::vector<TokenAndId>& sorted_token_and_ids, bool is_main_rule) = 0;

  /*!
   * \brief Rollback the matcher to a previous state.
   * \param rollback_steps The number of steps to rollback. Should not be greater than the number of
   * steps in the current history.
   */
  virtual void Rollback(int rollback_steps) = 0;

  /*!
   * \brief Print the current state of the matcher to a string as a set of stacks, mainly for
   * debugging purpose.
   * \param steps_behind_latest The number of steps behind the latest state to print. If 0, print
   * the latest state.
   */
  virtual std::string PrintStackState(int steps_behind_latest = 0) const = 0;

  static constexpr const char* _type_key = "mlc.serve.GrammarMatcher";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarMatcherNode, Object);
};

class GrammarMatcher : public ObjectRef {
 public:
  /*!
   * \brief Construct a grammar matcher from a BNF grammar.
   * \param grammar The BNF grammar to be matched.
   * \param max_rollback_steps The maximum number of steps to rollback when backtracking.
   * \param init_rule_position The initial position of the grammar matcher. If not specified, the
   * matcher will start from all start positions in the main rule.
   */
  GrammarMatcher(const BNFGrammar& grammar, int max_rollback_steps = 0,
                 RulePosition init_rule_position = {});

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarMatcher, ObjectRef, GrammarMatcherNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
