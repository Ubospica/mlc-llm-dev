/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar_tokenizer_config.h"

#include "grammar_matcher.h"
#include "grammar_serializer.h"

namespace mlc {
namespace llm {
namespace serve {

bool TokenAndId::operator<(const TokenAndId& other) const {
  for (size_t i = 0; i < token.size(); ++i) {
    if (i >= other.token.size()) {
      return false;
    }
    if (token[i] < other.token[i]) {
      return true;
    } else if (token[i] > other.token[i]) {
      return false;
    }
  }
  return token.size() < other.token.size();
}

std::string ReplaceUnderscoreWithSpace(const std::string& str,
                                       const std::string& kSpecialUnderscore) {
  std::string res;
  size_t pos = 0;
  while (pos < str.size()) {
    size_t found = str.find(kSpecialUnderscore, pos);
    if (found == std::string::npos) {
      res += str.substr(pos);
      break;
    }
    res += str.substr(pos, found - pos) + " ";
    pos = found + kSpecialUnderscore.size();
  }
  return res;
}

GrammarTokenizerConfig::GrammarTokenizerConfig(const Tokenizer& tokenizer,
                                               const BNFGrammar& grammar) {
  ObjectPtr<GrammarTokenizerConfigNode> n = make_object<GrammarTokenizerConfigNode>();
  n->vocab_size = tokenizer->GetVocabSize();
  for (int i = 0; i < tokenizer->GetVocabSize(); ++i) {
    auto token = tokenizer->IdToToken(i);
    if (token == "<unk>" || token == "<pad>" || token == "<s>") {
      n->special_token_ids.push_back(i);
    } else if (token == "</s>") {
      n->stop_token_ids.push_back(i);
    } else if (token[0] == '<' && token[token.size() - 1] == '>') {
      n->special_token_ids.push_back(i);
    } else {
      auto token_underscore_replaced = ReplaceUnderscoreWithSpace(token, n->kSpecialUnderscore);
      auto codepoints = Utf8StringToCodepoints(token_underscore_replaced.c_str());
      ICHECK(!codepoints.empty() &&
             codepoints[0] != static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8))
          << "Invalid token: " << token;
      n->sorted_token_and_ids.push_back({codepoints, i});
      n->token_lookup_map[i] = {codepoints, i};
    }
  }
  std::sort(n->sorted_token_and_ids.begin(), n->sorted_token_and_ids.end());

  std::chrono::duration<double, std::milli> duration;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
    auto rule = grammar->GetRule(i);
    auto rule_expr = grammar->GetRuleExpr(rule.body_expr_id);
    for (auto sequence_id : rule_expr) {
      auto sequence_expr = grammar->GetRuleExpr(sequence_id);
      if (sequence_expr.type == BNFGrammarNode::RuleExprType::kEmptyStr) {
        continue;
      }
      ICHECK(sequence_expr.type == BNFGrammarNode::RuleExprType::kSequence);
      for (int element_id = 0; element_id < sequence_expr.size(); ++element_id) {
        auto element_expr = grammar->GetRuleExpr(sequence_expr[element_id]);
        if (element_expr.type == BNFGrammarNode::RuleExprType::kRuleRef) {
          continue;
        }
        auto cur_rule_position = RulePosition{i, sequence_id, element_id};

        // std::cout << "Rule: " << rule.name
        //           << " Sequence: " << BNFGrammarPrinter(grammar).PrintRuleExpr(sequence_expr)
        //           << " Position: " << element_id << std::endl;

        // auto start = std::chrono::high_resolution_clock::now();
        auto grammar_matcher = GrammarMatcher(grammar, 0, cur_rule_position);
        // auto end = std::chrono::high_resolution_clock::now();
        auto cur_catagorized_tokens_for_grammar =
            grammar_matcher->GetCatagorizedTokens(n->sorted_token_and_ids, i == 0);
        // auto end1 = std::chrono::high_resolution_clock::now();
        // std::cout << "preprocess step1: "
        //           << std::chrono::duration<double, std::milli>(end - start).count() << " ms"
        //           << std::endl;
        // std::cout << "preprocess step2: "
        //           << std::chrono::duration<double, std::milli>(end1 - end).count() << " ms"
        //           << std::endl;
        n->catagorized_tokens_for_grammar[{sequence_id, element_id}] =
            cur_catagorized_tokens_for_grammar;
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  std::cout << "Preprocess time: " << duration.count() << " ms" << std::endl;
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
