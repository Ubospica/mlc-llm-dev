/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_simplifier.cc
 */

#include "grammar_simplifier.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarToSimplified")
    .set_body_typed([](const BNFGrammar& grammar) {
      return BNFGrammarSimplifier(grammar).Apply();
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
