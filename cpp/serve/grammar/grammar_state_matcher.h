/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher.h
 * \brief The header for the support of matching characters to a BNF grammar. This is the core
 * logic of the grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_grammar_state_matcher_H_
#define MLC_LLM_SERVE_GRAMMAR_grammar_state_matcher_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <vector>

#include "../../support/dynamic_bitset.h"
#include "../../support/encoding.h"
#include "grammar.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief Refers to a specific position in the grammar. Used to specify the initial position of
 * the grammar state matcher.
 * \sa mlc::llm::serve::GrammarStateMatcher::GrammarStateMatcher
 */
class GrammarStateInitContext;

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
  virtual bool AcceptCodepoint(TCodepoint codepoint, bool drop_old = true,
                               bool verbose = false) = 0;

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
  virtual void FindRejectedTokens(DynamicBitSet* rejected_ids) = 0;

  /*!
   * \brief RollbackCodepoint the matcher to a previous state.
   * \param rollback_steps The number of steps to rollback. Should not be greater than the number of
   * steps in the current history.
   */
  virtual void RollbackSteps(int rollback_steps) = 0;

  static constexpr const char* _type_key = "mlc.serve.GrammarStateMatcher";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarStateMatcherNode, Object);
};

class GrammarStateMatcher : public ObjectRef {
 public:
  /*! \brief Construct a GrammarStateInitContext from a tokenizer and a grammar. The grammar should
   * be the same as the grammar used to construct the GrammarStateMatcher. */
  static std::shared_ptr<GrammarStateInitContext> CreateInitContext(
      const BNFGrammar& grammar, const std::vector<std::string>& token_table);
  GrammarStateMatcher(std::shared_ptr<GrammarStateInitContext> init_ctx,
                      int max_rollback_steps = 0);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarStateMatcher, ObjectRef, GrammarStateMatcherNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_grammar_state_matcher_H_
