/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar_matcher.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(GrammarMatcherNode);

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcher")
    .set_body_typed([](BNFGrammar grammar, int max_rollback_steps) {
      return GrammarMatcher(grammar, max_rollback_steps);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcherAcceptChar")
    .set_body_typed([](GrammarMatcher matcher, int32_t codepoint, bool drop_old) {
      return matcher->AcceptChar(codepoint, drop_old);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcherCanAcceptEnd")
    .set_body_typed([](GrammarMatcher matcher) { return matcher->CanAcceptEnd(); });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcherMatchCompleteString")
    .set_body_typed([](GrammarMatcher matcher, String str) {
      std::cout << "29 Grammar: " << matcher->grammar_ << "\n";
      return matcher->MatchCompleteString(str);
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
