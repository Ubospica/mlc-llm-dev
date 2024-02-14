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

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcherCanReachEnd")
    .set_body_typed([](GrammarMatcher matcher) { return matcher->CanReachEnd(); });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcherMatchCompleteString")
    .set_body_typed([](GrammarMatcher matcher, String str) {
      return matcher->MatchCompleteString(str);
    });

IntTuple GetRejectedTokenIdsForTokenizer(GrammarMatcher matcher, BNFGrammar grammar,
                                         Tokenizer tokenizer) {
  auto start = std::chrono::high_resolution_clock::now();
  static BNFGrammar cached_grammar = grammar;
  static Tokenizer cached_tokenizer = tokenizer;
  static GrammarTokenizerConfig tokenizer_config = GrammarTokenizerConfig(tokenizer, grammar);
  if (cached_tokenizer != tokenizer || cached_grammar != grammar) {
    tokenizer_config = GrammarTokenizerConfig(tokenizer, cached_grammar);
    cached_tokenizer = tokenizer;
    cached_grammar = grammar;
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  // std::cout << "Step 1: " << duration.count() << " ms" << std::endl;
  // std::cout << "Stack: " << matcher->PrintStackState() << std::endl;
  matcher->process_time = std::chrono::milliseconds(0);
  matcher->process_1_time = std::chrono::milliseconds(0);
  matcher->process_2_time = std::chrono::milliseconds(0);
  matcher->process_3_time = std::chrono::milliseconds(0);
  matcher->process_4_time = std::chrono::milliseconds(0);
  matcher->overhead_time = std::chrono::milliseconds(0);
  DynamicBitSet bitset;
  start = std::chrono::high_resolution_clock::now();
  matcher->FindRejectedTokens(tokenizer_config, &bitset);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Total time: " << duration.count() << " ms" << std::endl;
  std::cout << "process_time: " << matcher->process_time.count() << " ms" << std::endl;
  std::cout << "process_1_time: " << matcher->process_1_time.count() << " ms" << std::endl;
  std::cout << "process_2_time: " << matcher->process_2_time.count() << " ms" << std::endl;
  std::cout << "Overhead time: " << matcher->overhead_time.count() << " ms" << std::endl;
  std::cout << "process_3_time: " << matcher->process_3_time.count() << " ms" << std::endl;
  std::cout << "process_4_time: " << matcher->process_4_time.count() << " ms" << std::endl;

  // start = std::chrono::high_resolution_clock::now();

  std::vector<long> res_vector;
  for (int i = 0; i < bitset.Size(); i++) {
    if (bitset[i]) {
      res_vector.push_back(i);
    }
  }

  auto ret = IntTuple(res_vector);
  // end = std::chrono::high_resolution_clock::now();
  // duration = end - start;
  // std::cout << "Step 3: " << duration.count() << " ms" << std::endl;
  return ret;
}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarMatcherGetRejectedTokenIdsForTokenizer")
    .set_body_typed(GetRejectedTokenIdsForTokenizer);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
