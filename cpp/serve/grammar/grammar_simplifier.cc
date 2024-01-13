/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar_simplifier.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarToNormalized")
    .set_body_typed([](const BNFGrammar& grammar) {
      return BNFGrammarNormalizer(grammar).Apply();
    });

}  // namespace serve
}  // namespace llm
}  // namespace mlc
