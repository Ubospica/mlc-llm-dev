/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../../support/dynamic_bitset.h"
#include "../encoding.h"
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

/*! \brief Refers to a specific position in the grammar. Used to specify the initial position of
 * the grammar matcher. */
class GrammarMatcherNode : public Object {
 public:
  virtual bool AcceptChar(TCodepoint codepoint, bool drop_old = true, bool verbose = false) = 0;
  virtual bool MatchCompleteString(String str) = 0;
  virtual bool CanReachEnd() const = 0;

  virtual void FindRejectedTokens(const GrammarTokenizerConfig& tokenizer_config,
                                  DynamicBitSet* rejected_ids) = 0;

  virtual CatagorizedTokensForGrammar GetCatagorizedTokens(
      const std::vector<TokenAndId>& sorted_token_and_ids, bool is_root_rule) = 0;
  virtual void Rollback(int rollback_steps) = 0;

  virtual std::string PrintStackState(int steps_behind_latest = 0) const = 0;

  static constexpr const char* _type_key = "mlc.serve.GrammarMatcher";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarMatcherNode, Object);
};

class GrammarMatcher : public ObjectRef {
 public:
  GrammarMatcher(const BNFGrammar& grammar, int max_rollback_steps = 0,
                 RulePosition init_rule_position = {});

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarMatcher, ObjectRef, GrammarMatcherNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
