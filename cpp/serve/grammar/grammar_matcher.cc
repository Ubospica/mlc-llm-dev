/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar_matcher.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(GrammarMatcherNode);

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarMatchString")
    .set_body_typed([](BNFGrammar grammar, String input) {
      return GrammarMatcher(grammar)->AcceptString(input);
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
