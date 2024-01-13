/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_serializer.h
 * \brief The header for printing the AST of a BNF grammar.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_

#include <string>

#include "grammar.h"
#include "grammar_builder.h"

namespace mlc {
namespace llm {
namespace serve {

/*!
 * \brief Serialize the abstract syntax tree of a BNF grammar to a string.
 1. recursive -> non recursive
  1. rule is or rule
  2. or -> or: unroll
  3. or -> seq ok
  3. or -> char range -> seq
  4. or -> empty str -> ok
  5. or -> ruleref -> seq
  6. seq -> or: new rule
  7. seq -> seq: unroll
  8. seq -> char range ok
  9. seq -> empty str: check others: others, no others: empty str
  10. seq -> ruleref: ok
 2. left recursive -> right recursive
  1. epsilon elimination
  2. unit production elimination
  3. left recursion elimination
  4. unreachable elimination
 3. inline
  1. one choice
  2. not char starting
 */

class BNFGrammarMutator {
 public:
  BNFGrammarMutator(const BNFGrammar& grammar) : grammar_(grammar) {}
  virtual BNFGrammar Apply() = 0;

 protected:
  using Rule = BNFGrammarNode::Rule;
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using DataKind = BNFGrammarNode::DataKind;
  const BNFGrammar& grammar_;
  BNFGrammarBuilder builder_;
};

// input: rule -> choices -> sequence
class BNFGrammarFlattener : public BNFGrammarMutator {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply() final {
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      builder_.AddEmptyRule(grammar_->GetRule(i).name);
    }
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_expr.kind == BNFGrammarNode::DataKind::kChoices)
          << "The rule body is expected to be a choice, but rule " << rule.name << " is not.";
      cur_rule_name_ = rule.name;
      auto choice_ids = HandleChoices(rule_expr);
      auto new_rule_expr_id = builder_.AddChoices(choice_ids);
      builder_.UpdateRuleBody(i, new_rule_expr_id);
    }
    return builder_.Get();
  }

 private:
  std::vector<int32_t> HandleChoices(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_choice_ids;
    bool found_empty = false;
    for (int i = 0; i < rule_expr.data_len; ++i) {
      auto choice_expr = grammar_->GetRuleExpr(rule_expr[i]);
      ICHECK(choice_expr.kind == BNFGrammarNode::DataKind::kSequence)
          << "The body of a choice is expected to be a sequence, but RuleExpr with id " << i
          << " is not.";

      auto sub_sequence_ids = HandleSequence(rule_expr);
      if (sub_sequence_ids.size() == 0) {
        found_empty = true;
      } else if (sub_sequence_ids.size() == 1) {
        auto sub_sequence_expr = grammar_->GetRuleExpr(sub_sequence_ids[0]);
        HandleSingleSequenceInChoices(sub_sequence_expr, &new_choice_ids, &found_empty);
      } else {
        new_choice_ids.push_back(builder_.AddSequence(sub_sequence_ids));
      }
    }
    if (found_empty) {
      new_choice_ids.insert(new_choice_ids.begin(), builder_.AddEmptyStr());
    }
    ICHECK_GE(new_choice_ids.size(), 1);
    return new_choice_ids;
  }

  void HandleSingleSequenceInChoices(const RuleExpr& rule_expr,
                                     std::vector<int32_t>* new_choice_ids, bool* found_empty) {
    switch (choice_expr.kind) {
      case DataKind::kSequence:
        HandleSequenceInChoices(choice_expr, &new_choice_ids, &found_empty);
        break;
      case DataKind::kChoices:
        HandleChoicesInChoices(choice_expr, &new_choice_ids, &found_empty);
        break;
      case DataKind::kEmptyStr:
        found_empty = true;
        break;
      case DataKind::kCharacterRange:
      case DataKind::kNegCharacterRange:
      case DataKind::kRuleRef:
        HandleAtomInChoices(choice_expr, &new_choice_ids);
        break;
      default:
        LOG(FATAL) << "Unexpected choice kind: " << static_cast<int>(choice_expr.kind);
    }
  }
  
  std::vector<int32_t> HandleSequence(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_sequence_ids;
    for (int i = 0; i < rule_expr.data_len; ++i) {
      auto seq_expr = grammar_->GetRuleExpr(rule_expr[i]);
      switch (seq_expr.kind) {
        case DataKind::kSequence:
          HandleSequenceInSequence(seq_expr, &new_sequence_ids);
          break;
        case DataKind::kChoices:
          HandleChoiceInSequence(seq_expr, &new_sequence_ids);
          break;
        case DataKind::kEmptyStr:
          break;
        case DataKind::kCharacterRange:
        case DataKind::kNegCharacterRange:
        case DataKind::kRuleRef:
          HandleAtomInSequence(seq_expr, &new_sequence_ids);
          break;
        default:
          LOG(FATAL) << "Unexpected sequence kind: " << static_cast<int>(seq_expr.kind);
      }
    }
    return new_sequence_ids;
  }

  void HandleSequenceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_sequence_ids = HandleSequence(rule_expr);
    new_sequence_ids->insert(new_sequence_ids->end(), sub_sequence_ids.begin(),
                             sub_sequence_ids.end());
  }

  void HandleChoiceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_choice_ids = HandleChoices(rule_expr);
    if (sub_choice_ids.size() == 1) {
      auto choice_element_expr = builder_.GetRuleExpr(sub_choice_ids[0]);
      if (choice_element_expr.kind != DataKind::kEmptyStr) {
        new_sequence_ids->insert(new_sequence_ids->end(), choice_element_expr.begin(),
                                 choice_element_expr.end());
      }
    } else {
      auto new_choice_id = builder_.AddChoices(sub_choice_ids);
      auto new_choice_rule_id = builder_.AddRuleWithHint(cur_rule_name_ + "_choice", new_choice_id);
      new_sequence_ids->push_back(builder_.AddRuleRef(new_choice_rule_id));
    }
  }

  void HandleAtomInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    new_sequence_ids->push_back(builder_.AddRuleExpr(rule_expr));
  }

  std::string cur_rule_name_;
};

// class UnreachableEliminator : public BNFGrammarMutator {
//  public:
//   UnreachableEliminator(const BNFGrammar& grammar) : BNFGrammarMutator(grammar, false) {}
//   BNFGrammar Apply() final {}
// };

class BNFGrammarNormalizer : public BNFGrammarMutator {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply() final {
    auto flattened = BNFGrammarFlattener(grammar_).Apply();
    return flattened;
  }
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_
