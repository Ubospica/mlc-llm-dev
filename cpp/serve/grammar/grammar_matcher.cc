/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar_matcher.h"

#include <chrono>

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
      return matcher->MatchCompleteString(str);
    });

IntTuple GetRejectedTokenIdsForTokenizer(GrammarMatcher matcher, Tokenizer tokenizer) {
  auto start = std::chrono::high_resolution_clock::now();
  static Tokenizer cached_tokenizer = tokenizer;
  static TokenizerConfig tokenizer_config = TokenizerConfig(tokenizer);
  if (cached_tokenizer != tokenizer) {
    tokenizer_config = TokenizerConfig(tokenizer);
    cached_tokenizer = tokenizer;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  std::cout << "Step 1: " << duration.count() << " ms" << std::endl;
  // std::cout << "Stack: " << matcher->PrintStackState() << std::endl;
  start = std::chrono::high_resolution_clock::now();
  matcher->handle_past_time = std::chrono::milliseconds(0);
  matcher->rollback_total_time = std::chrono::milliseconds(0);
  matcher->accept_total_time = std::chrono::milliseconds(0);
  matcher->codepoint_set_total_time = std::chrono::milliseconds(0);
  auto res = matcher->FindRejectedTokenIds(tokenizer_config->sorted_token_and_ids);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Step 2: " << duration.count() << " ms" << std::endl;
  std::cout << "Handle past time: " << matcher->handle_past_time.count() << " ms" << std::endl;
  std::cout << "Rollback time: " << matcher->rollback_total_time.count() << " ms" << std::endl;
  std::cout << "Accept time: " << matcher->accept_total_time.count() << " ms" << std::endl;
  std::cout << "Codepoint set time: " << matcher->codepoint_set_total_time.count() << " ms"
            << std::endl;
  start = std::chrono::high_resolution_clock::now();
  auto ret = IntTuple(std::vector<long>(res.begin(), res.end()));
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Step 3: " << duration.count() << " ms" << std::endl;
  return ret;
}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcherGetRejectedTokenIdsForTokenizer")
    .set_body_typed(GetRejectedTokenIdsForTokenizer);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
