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
  using RuleExprType = BNFGrammarNode::RuleExprType;

  ObjectPtr<GrammarTokenizerConfigNode> n = make_object<GrammarTokenizerConfigNode>();
  n->vocab_size = tokenizer->GetVocabSize();
  for (int i = 0; i < tokenizer->GetVocabSize(); ++i) {
    auto token = tokenizer->IdToToken(i);
    if (token == "<unk>" || token == "<pad>" || token == "<s>") {
      n->special_token_ids.push_back(i);
    } else if (token == "</s>") {
      n->stop_token_ids.push_back(i);
    } else if (token[0] == '<' && token[token.size() - 1] == '>') {
      // Currently we consider all <...> tokens as special tokens.
      n->special_token_ids.push_back(i);
    } else {
      // First replace the special underscore with space.
      auto token_underscore_replaced = ReplaceUnderscoreWithSpace(token, n->kSpecialUnderscore);
      auto codepoints = Utf8StringToCodepoints(token_underscore_replaced.c_str());
      DCHECK(!codepoints.empty() &&
             codepoints[0] != static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8))
          << "Invalid token: " << token;
      n->sorted_token_and_ids.push_back({codepoints, i});
      n->token_lookup_map[i] = {codepoints, i};
    }
  }
  std::sort(n->sorted_token_and_ids.begin(), n->sorted_token_and_ids.end());

  // Find the corresponding catagorized tokens for:
  // 1. All character elements in the grammar
  // 2. All RuleRef elements that refers to a rule of a StarQuantifier of a character class
  for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
    auto rule = grammar->GetRule(i);
    auto rule_expr = grammar->GetRuleExpr(rule.body_expr_id);
    // Skip StarQuantifier since we just handle it at the reference element during matching.
    if (rule_expr.type == RuleExprType::kStarQuantifier) {
      continue;
    }
    DCHECK(rule_expr.type == RuleExprType::kChoices);
    for (auto sequence_id : rule_expr) {
      auto sequence_expr = grammar->GetRuleExpr(sequence_id);
      if (sequence_expr.type == RuleExprType::kEmptyStr) {
        continue;
      }
      DCHECK(sequence_expr.type == RuleExprType::kSequence);
      for (int element_id = 0; element_id < sequence_expr.size(); ++element_id) {
        auto element_expr = grammar->GetRuleExpr(sequence_expr[element_id]);
        auto cur_rule_position = RulePosition{i, sequence_id, element_id};
        if (element_expr.type == RuleExprType::kRuleRef) {
          auto ref_rule = grammar->GetRule(element_expr[0]);
          auto ref_rule_expr = grammar->GetRuleExpr(ref_rule.body_expr_id);
          if (ref_rule_expr.type == RuleExprType::kChoices) {
            continue;
          } else {
            // Reference to a StarQuantifier of a character class.
            cur_rule_position.char_class_id = ref_rule_expr[0];
          }
        }

        auto grammar_matcher = GrammarMatcher(grammar, 0, cur_rule_position);
        auto cur_catagorized_tokens_for_grammar =
            grammar_matcher->GetCatagorizedTokens(n->sorted_token_and_ids, i == 0);
        n->catagorized_tokens_for_grammar[{sequence_id, element_id}] =
            cur_catagorized_tokens_for_grammar;
      }
    }
  }
  data_ = std::move(n);
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
