/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_serializer.h
 * \brief The header for printing the AST of a BNF grammar.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_

#include <queue>
#include <string>

#include "grammar.h"
#include "grammar_builder.h"
#include "grammar_serializer.h"

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

template <typename T = int32_t, typename ReturnType = BNFGrammar>
class BNFGrammarMutator {
 public:
  explicit BNFGrammarMutator(const BNFGrammar& grammar) : grammar_(grammar) {}
  virtual ReturnType Apply() {
    if constexpr (std::is_same<T, int32_t>::value && std::is_same<ReturnType, BNFGrammar>::value) {
      for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
        auto rule = grammar_->GetRule(i);
        auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
        auto new_rule_expr_id = VisitExpr(rule_expr);
        builder_.AddRule(rule.name, new_rule_expr_id);
      }
      return builder_.Get();
    } else if constexpr (!std::is_same<ReturnType, void>::value) {
      return ReturnType();
    }
  }

 protected:
  using Rule = BNFGrammarNode::Rule;
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using DataKind = BNFGrammarNode::DataKind;

  virtual T VisitExpr(const RuleExpr& rule_expr) {
    switch (rule_expr.kind) {
      case DataKind::kSequence:
        return VisitSequence(rule_expr);
      case DataKind::kChoices:
        return VisitChoices(rule_expr);
      case DataKind::kEmptyStr:
        return VisitEmptyStr(rule_expr);
      case DataKind::kCharacterRange:
      case DataKind::kNegCharacterRange:
        return VisitCharacterRange(rule_expr);
      case DataKind::kRuleRef:
        return VisitRuleRef(rule_expr);
      default:
        LOG(FATAL) << "Unexpected sequence kind: " << static_cast<int>(rule_expr.kind);
    }
  }

  virtual T VisitSequence(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(grammar_->GetRuleExpr(i));
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<T> sequence_ids;
      for (int32_t i : rule_expr) {
        sequence_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
      }
      return builder_.AddSequence(sequence_ids);
    } else {
      return T();
    }
  }

  virtual T VisitChoices(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      for (auto i : rule_expr) {
        VisitExpr(grammar_->GetRuleExpr(i));
      }
    } else if constexpr (std::is_same<T, int32_t>::value) {
      std::vector<int32_t> choice_ids;
      for (int32_t i : rule_expr) {
        choice_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
      }
      return builder_.AddChoices(choice_ids);
    } else {
      return T();
    }
  }

  virtual T VisitElement(const RuleExpr& rule_expr) {
    if constexpr (std::is_same<T, void>::value) {
      return;
    } else if constexpr (std::is_same<T, int32_t>::value) {
      return builder_.AddRuleExpr(rule_expr);
    } else {
      return T();
    }
  }

  virtual T VisitEmptyStr(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  virtual T VisitCharacterRange(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  virtual T VisitRuleRef(const RuleExpr& rule_expr) { return VisitElement(rule_expr); }

  BNFGrammar grammar_;
  BNFGrammarBuilder builder_;
};

class SingleElementSequenceOrChoiceEliminator : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::Apply;
  using BNFGrammarMutator::BNFGrammarMutator;

 private:
  int32_t VisitSequence(const RuleExpr& rule_expr) {
    std::vector<int32_t> sequence_ids;
    for (int32_t i : rule_expr) {
      sequence_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
    }
    if (sequence_ids.size() == 1) {
      return sequence_ids[0];
    } else {
      return builder_.AddSequence(sequence_ids);
    }
  }

  int32_t VisitChoices(const RuleExpr& rule_expr) {
    std::vector<int32_t> choice_ids;
    for (int32_t i : rule_expr) {
      choice_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
    }
    if (choice_ids.size() == 1) {
      return choice_ids[0];
    } else {
      return builder_.AddChoices(choice_ids);
    }
  }
};

class BNFGrammarFlattener : public BNFGrammarMutator<std::vector<int32_t>, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply() final {
    grammar_ = SingleElementSequenceOrChoiceEliminator(grammar_).Apply();
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      builder_.AddEmptyRule(grammar_->GetRule(i).name);
    }
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      cur_rule_name_ = rule.name;
      auto new_rule_expr_id = VisitRuleBody(rule_expr);
      builder_.UpdateRuleBody(i, new_rule_expr_id);
    }
    return builder_.Get();
  }

 private:
  int32_t VisitRuleBody(const RuleExpr& rule_expr) {
    switch (rule_expr.kind) {
      case DataKind::kSequence:
        return builder_.AddChoices({builder_.AddSequence(VisitSequence(rule_expr))});
      case DataKind::kChoices:
        return builder_.AddChoices(VisitChoices(rule_expr));
      case DataKind::kEmptyStr:
        return builder_.AddChoices({builder_.AddEmptyStr()});
      case DataKind::kCharacterRange:
      case DataKind::kNegCharacterRange:
      case DataKind::kRuleRef:
        return builder_.AddChoices({builder_.AddSequence({builder_.AddRuleExpr(rule_expr)})});
      default:
        LOG(FATAL) << "Unexpected sequence kind: " << static_cast<int>(rule_expr.kind);
    }
  }

  std::vector<int32_t> VisitChoices(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_choice_ids;
    bool found_empty = false;
    for (auto i : rule_expr) {
      auto choice_expr = grammar_->GetRuleExpr(i);
      switch (choice_expr.kind) {
        case DataKind::kSequence:
          VisitSequenceInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case DataKind::kChoices:
          VisitChoicesInChoices(choice_expr, &new_choice_ids, &found_empty);
          break;
        case DataKind::kEmptyStr:
          found_empty = true;
          break;
        case DataKind::kCharacterRange:
        case DataKind::kNegCharacterRange:
        case DataKind::kRuleRef:
          VisitAtomInChoices(choice_expr, &new_choice_ids);
          break;
        default:
          LOG(FATAL) << "Unexpected choice kind: " << static_cast<int>(choice_expr.kind);
      }
    }
    if (found_empty) {
      new_choice_ids.insert(new_choice_ids.begin(), builder_.AddEmptyStr());
    }
    ICHECK_GE(new_choice_ids.size(), 1);
    return new_choice_ids;
  }

  void VisitSequenceInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids,
                              bool* found_empty) {
    auto sub_sequence_ids = VisitSequence(rule_expr);
    if (sub_sequence_ids.size() == 0) {
      *found_empty = true;
    } else {
      new_choice_ids->push_back(builder_.AddSequence(sub_sequence_ids));
    }
  }

  void VisitChoicesInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids,
                             bool* found_empty) {
    auto sub_choice_ids = VisitChoices(rule_expr);
    bool contains_empty = builder_.GetRuleExpr(sub_choice_ids[0]).kind == DataKind::kEmptyStr;
    if (contains_empty) {
      *found_empty = true;
      new_choice_ids->insert(new_choice_ids->end(), sub_choice_ids.begin() + 1,
                             sub_choice_ids.end());
    } else {
      new_choice_ids->insert(new_choice_ids->end(), sub_choice_ids.begin(), sub_choice_ids.end());
    }
  }

  void VisitAtomInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids) {
    auto sub_expr_id = builder_.AddRuleExpr(rule_expr);
    new_choice_ids->push_back(builder_.AddSequence({sub_expr_id}));
  }

  std::vector<int32_t> VisitSequence(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_sequence_ids;
    for (auto i : rule_expr) {
      auto seq_expr = grammar_->GetRuleExpr(i);
      switch (seq_expr.kind) {
        case DataKind::kSequence:
          VisitSequenceInSequence(seq_expr, &new_sequence_ids);
          break;
        case DataKind::kChoices:
          VisitChoiceInSequence(seq_expr, &new_sequence_ids);
          break;
        case DataKind::kEmptyStr:
          break;
        case DataKind::kCharacterRange:
        case DataKind::kNegCharacterRange:
        case DataKind::kRuleRef:
          VisitAtomInSequence(seq_expr, &new_sequence_ids);
          break;
        default:
          LOG(FATAL) << "Unexpected sequence kind: " << static_cast<int>(seq_expr.kind);
      }
    }
    return new_sequence_ids;
  }

  void VisitSequenceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_sequence_ids = VisitSequence(rule_expr);
    new_sequence_ids->insert(new_sequence_ids->end(), sub_sequence_ids.begin(),
                             sub_sequence_ids.end());
  }

  void VisitChoiceInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    auto sub_choice_ids = VisitChoices(rule_expr);
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

  void VisitAtomInSequence(const RuleExpr& rule_expr, std::vector<int32_t>* new_sequence_ids) {
    new_sequence_ids->push_back(builder_.AddRuleExpr(rule_expr));
  }

  std::string cur_rule_name_;
};

class RuleVisitGraph {
 public:
  RuleVisitGraph() = default;
  explicit RuleVisitGraph(int num_rules) : in_edges_(num_rules), out_edges_(num_rules) {}
  void AddEdge(int32_t from, int32_t to) {
    out_edges_[from].push_back(to);
    in_edges_[to].push_back(from);
  }
  const std::vector<int32_t>& GetOutEdges(int32_t from) const { return out_edges_[from]; }
  const std::vector<int32_t>& GetInEdges(int32_t to) const { return in_edges_[to]; }
  std::string ToString() const {
    std::string result;
    for (int i = 0; i < static_cast<int>(out_edges_.size()); ++i) {
      result += std::to_string(i) + ": [";
      for (auto j = out_edges_[i].begin(); j != out_edges_[i].end(); ++j) {
        result += std::to_string(*j);
        if (j + 1 != out_edges_[i].end()) {
          result += ", ";
        }
      }
      result += "]\n";
    }
    return result;
  }

 private:
  std::vector<std::vector<int32_t>> in_edges_;
  std::vector<std::vector<int32_t>> out_edges_;
};

class RuleReachGraphFinder : public BNFGrammarMutator<void, RuleVisitGraph> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  RuleVisitGraph Apply() {
    rule_visit_graph_ = RuleVisitGraph(grammar_->NumRules());
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      cur_rule_id_ = i;
      VisitExpr(rule_expr);
    }
    return rule_visit_graph_;
  }

 private:
  void VisitRuleRef(const RuleExpr& rule_expr) {
    rule_visit_graph_.AddEdge(cur_rule_id_, rule_expr[0]);
  }

  RuleVisitGraph rule_visit_graph_;
  int32_t cur_rule_id_;
};

class UnreachableEliminator : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  BNFGrammar Apply() final {
    rule_visit_graph_ = RuleReachGraphFinder(grammar_).Apply();
    FindMainRule();
    FindRuleReachable();
    RenumberRules();
    BuildRuleBody();
    return builder_.Get();
  }

 private:
  void FindMainRule() {
    root_rule_id_ = -1;
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      if (grammar_->GetRule(i).name == "main") {
        ICHECK_EQ(root_rule_id_, -1);
        root_rule_id_ = i;
        break;
      }
    }
    ICHECK_GE(root_rule_id_, 0);
    CHECK(rule_visit_graph_.GetInEdges(root_rule_id_).empty()) << "Main rule should not be used.";
  }

  void FindRuleReachable() {
    rule_reachable_ = std::vector<bool>(grammar_->NumRules(), false);
    std::unordered_set<int32_t> visited;
    std::queue<int32_t> queue;
    queue.push(root_rule_id_);
    while (!queue.empty()) {
      auto rule_id = queue.front();
      queue.pop();
      if (visited.count(rule_id)) {
        continue;
      }
      visited.insert(rule_id);
      rule_reachable_[rule_id] = true;
      for (auto next_rule_id : rule_visit_graph_.GetOutEdges(rule_id)) {
        queue.push(next_rule_id);
      }
    }
  }

  void RenumberRules() {
    rule_id_map_.clear();
    rule_id_map_[root_rule_id_] = builder_.AddEmptyRule("main");
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      if (i == root_rule_id_) {
        continue;
      }
      if (rule_reachable_[i]) {
        rule_id_map_[i] = builder_.AddEmptyRule(grammar_->GetRule(i).name);
      }
    }
  }

  void BuildRuleBody() {
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      if (!rule_reachable_[i]) {
        continue;
      }
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      auto new_rule_expr_id = VisitExpr(rule_expr);
      builder_.UpdateRuleBody(rule_id_map_[i], new_rule_expr_id);
    }
  }

  int32_t VisitRuleRef(const RuleExpr& rule_expr) {
    auto rule_id = rule_expr[0];
    ICHECK(rule_reachable_[rule_id]);
    ICHECK(rule_id_map_.count(rule_id));
    return builder_.AddRuleRef(rule_id_map_[rule_id]);
  }

  RuleVisitGraph rule_visit_graph_;
  int32_t root_rule_id_;
  std::vector<bool> rule_reachable_;
  std::unordered_map<int32_t, int32_t> rule_id_map_;
};

class EpsilonEliminator : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  BNFGrammar Apply() final {
    rule_visit_graph_ = RuleReachGraphFinder(grammar_).Apply();
    CollectExplicitEpsilonRules();
    FindEpsilonRules();
    FindEpsilonOnlyRules();
    EliminateEpsilonRules();
    return builder_.Get();
  }

 private:
  void CollectExplicitEpsilonRules() {
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      if (grammar_->GetRuleExpr(rule_expr[0]).kind == DataKind::kEmptyStr) {
        epsilon_rule_id_set_.insert(i);
        if (rule_expr.data_len == 1) {
          epsilon_only_rule_id_set_.insert(i);
        }
      }
    }
  }

  void FindEpsilonRules() {
    std::queue<int32_t> queue;
    for (auto i : epsilon_rule_id_set_) {
      queue.push(i);
    }
    while (!queue.empty()) {
      auto rule_id = queue.front();
      queue.pop();
      for (auto next_rule_id : rule_visit_graph_.GetInEdges(rule_id)) {
        if (epsilon_rule_id_set_.count(next_rule_id)) {
          continue;
        }
        auto rule = grammar_->GetRule(next_rule_id);
        auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);

        bool is_epsilon = std::any_of(rule_expr.begin(), rule_expr.end(), [&](int32_t i) {
          auto sub_expr = grammar_->GetRuleExpr(i);
          return SubRuleExprIsEpsilon(sub_expr, epsilon_rule_id_set_);
        });

        if (is_epsilon) {
          epsilon_rule_id_set_.insert(next_rule_id);
          queue.push(next_rule_id);
        }
      }
    }
  }

  void FindEpsilonOnlyRules() {
    std::queue<int32_t> queue;
    for (auto i : epsilon_only_rule_id_set_) {
      queue.push(i);
    }
    while (!queue.empty()) {
      auto rule_id = queue.front();
      queue.pop();
      for (auto next_rule_id : rule_visit_graph_.GetInEdges(rule_id)) {
        if (epsilon_only_rule_id_set_.count(next_rule_id)) {
          continue;
        }
        auto rule = grammar_->GetRule(next_rule_id);
        auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);

        bool is_epsilon_only = std::all_of(rule_expr.begin(), rule_expr.end(), [&](int32_t i) {
          auto sub_expr = grammar_->GetRuleExpr(i);
          return SubRuleExprIsEpsilon(sub_expr, epsilon_only_rule_id_set_);
        });

        if (is_epsilon_only) {
          epsilon_only_rule_id_set_.insert(next_rule_id);
          queue.push(next_rule_id);
        }
      }
    }
  }

  bool SubRuleExprIsEpsilon(const RuleExpr& sub_expr,
                            const std::unordered_set<int32_t>& epsilon_rule_id_set) {
    if (sub_expr.kind == DataKind::kEmptyStr) {
      return true;
    }
    ICHECK(sub_expr.kind == DataKind::kSequence);

    bool all_epsilon = std::all_of(sub_expr.begin(), sub_expr.end(), [&](int32_t i) {
      auto sub_sub_expr = grammar_->GetRuleExpr(i);
      return sub_sub_expr.kind == DataKind::kRuleRef && epsilon_rule_id_set.count(sub_sub_expr[0]);
    });

    return all_epsilon;
  }

  void EliminateEpsilonRules() {
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_expr.kind == DataKind::kChoices);
      cur_rule_id_ = i;
      auto new_rule_expr_id = VisitChoices(rule_expr);
      builder_.AddRule(rule.name, new_rule_expr_id);
    }
  }

  int32_t VisitChoices(const RuleExpr& rule_expr) final {
    if (epsilon_only_rule_id_set_.count(cur_rule_id_)) {
      return builder_.AddChoices({builder_.AddEmptyStr()});
    }
    std::vector<int32_t> new_choice_ids;
    for (auto i : rule_expr) {
      auto choice_expr = grammar_->GetRuleExpr(i);
      if (choice_expr.kind == DataKind::kEmptyStr) {
        continue;
      }
      ICHECK(choice_expr.kind == DataKind::kSequence);
      VisitSequenceInChoices(choice_expr, &new_choice_ids);
    }
    if (new_choice_ids.empty() || (cur_rule_id_ == 0 && epsilon_rule_id_set_.count(0))) {
      new_choice_ids.insert(new_choice_ids.begin(), builder_.AddEmptyStr());
    }
    return builder_.AddChoices(new_choice_ids);
  }

  void VisitSequenceInChoices(const RuleExpr& rule_expr, std::vector<int32_t>* new_choice_ids,
                              int rule_iter = 0) {
    if (rule_iter == static_cast<int>(rule_expr.data_len)) {
      if (cur_sequence_.size() != 0) {
        new_choice_ids->push_back(builder_.AddSequence(cur_sequence_));
      }
      return;
    }
    auto sub_expr = grammar_->GetRuleExpr(rule_expr[rule_iter]);
    ICHECK(sub_expr.kind != DataKind::kEmptyStr);
    if (sub_expr.kind == DataKind::kRuleRef && epsilon_only_rule_id_set_.count(sub_expr[0])) {
      VisitSequenceInChoices(rule_expr, new_choice_ids, rule_iter + 1);
    } else if (sub_expr.kind == DataKind::kRuleRef && epsilon_rule_id_set_.count(sub_expr[0])) {
      VisitSequenceInChoices(rule_expr, new_choice_ids, rule_iter + 1);
      cur_sequence_.push_back(VisitExpr(sub_expr));
      VisitSequenceInChoices(rule_expr, new_choice_ids, rule_iter + 1);
      cur_sequence_.pop_back();
    } else {
      cur_sequence_.push_back(VisitExpr(sub_expr));
      VisitSequenceInChoices(rule_expr, new_choice_ids, rule_iter + 1);
      cur_sequence_.pop_back();
    }
  }

  std::unordered_set<int32_t> epsilon_rule_id_set_;
  std::unordered_set<int32_t> epsilon_only_rule_id_set_;
  RuleVisitGraph rule_visit_graph_;
  int32_t cur_rule_id_;
  std::vector<int32_t> cur_sequence_;
};

class UnitProductionEliminator : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  BNFGrammar Apply() final {
    while (true) {
      UpdateGrammar();
      if (!updated_) {
        break;
      }
    }
    return grammar_;
  }

 private:
  void UpdateGrammar() {
    Reset();
    GetUnitVisitGraph();
    GetUnitClosure();
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_expr.kind == DataKind::kChoices);
      cur_rule_id_ = i;
      auto new_rule_expr_id = VisitChoices(rule_expr);
      builder_.AddRule(rule.name, new_rule_expr_id);
    }
    grammar_ = EpsilonEliminator(builder_.Get()).Apply();
  }

  void Reset() {
    unit_closure_.clear();
    empty_rules_.clear();
    builder_ = BNFGrammarBuilder();
    updated_ = false;
  }

  void GetUnitVisitGraph() {
    unit_visit_graph_ = RuleVisitGraph(grammar_->NumRules());
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_expr.kind == DataKind::kChoices);
      for (auto j : rule_expr) {
        auto sequence_expr = grammar_->GetRuleExpr(j);
        if (sequence_expr.kind != DataKind::kSequence || sequence_expr.data_len != 1) {
          continue;
        }
        auto atom_expr = grammar_->GetRuleExpr(sequence_expr[0]);
        if (atom_expr.kind == DataKind::kRuleRef) {
          updated_ = true;
          unit_visit_graph_.AddEdge(i, atom_expr[0]);
          std::cout << i << " " << atom_expr[0] << "\n";
        }
      }
    }
  }

  void GetUnitClosure() {
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      std::unordered_set<int32_t> visited;
      std::queue<int32_t> queue;
      queue.push(i);
      while (!queue.empty()) {
        auto rule_id = queue.front();
        queue.pop();
        if (visited.count(rule_id)) {
          continue;
        }
        visited.insert(rule_id);
        for (auto next_rule_id : unit_visit_graph_.GetOutEdges(rule_id)) {
          queue.push(next_rule_id);
        }
      }
      unit_closure_.push_back(visited);
    }
  }

  int32_t VisitChoices(const RuleExpr& rule_expr) final {
    std::vector<int32_t> new_choice_ids = GetNonUnitChoices(rule_expr);
    for (auto to : unit_closure_[cur_rule_id_]) {
      if (to == cur_rule_id_) {
        continue;
      }
      auto rule = grammar_->GetRule(to);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_expr.kind == DataKind::kChoices);
      auto choices_to_append = GetNonUnitChoices(rule_expr);
      new_choice_ids.insert(new_choice_ids.end(), choices_to_append.begin(),
                            choices_to_append.end());
    }
    if (new_choice_ids.empty()) {
      return builder_.AddChoices({builder_.AddEmptyStr()});
    }
    return builder_.AddChoices(new_choice_ids);
  }

  std::vector<int32_t> GetNonUnitChoices(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_choice_ids;
    for (auto i : rule_expr) {
      auto choice_expr = grammar_->GetRuleExpr(i);
      if (choice_expr.kind == DataKind::kSequence && choice_expr.data_len == 1) {
        auto sub_expr = grammar_->GetRuleExpr(choice_expr[0]);
        if (sub_expr.kind == DataKind::kRuleRef) {
          continue;
        }
      }
      auto new_choice_id = VisitExpr(choice_expr);
      new_choice_ids.push_back(new_choice_id);
    }
    return new_choice_ids;
  }

  RuleVisitGraph unit_visit_graph_;
  std::vector<std::unordered_set<int32_t>> unit_closure_;
  std::unordered_set<int32_t> empty_rules_;
  int32_t cur_rule_id_;
  bool updated_;
};

class LeftRecursionEliminator : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  BNFGrammar Apply() final {
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      builder_.AddEmptyRule(rule.name);
    }
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_expr.kind == DataKind::kChoices);
      cur_rule_id_ = i;
      cur_rule_name_ = rule.name;
      auto new_rule_expr_id = VisitChoices(rule_expr);
      builder_.UpdateRuleBody(i, new_rule_expr_id);
    }
    std::cout << builder_.Get() << "\n";
    return UnreachableEliminator(builder_.Get()).Apply();
  }

 private:
  int32_t VisitChoices(const RuleExpr& rule_expr) final {
    std::vector<int32_t> new_choice_ids;
    new_choice_ids = InlinePriorRules(rule_expr);
    auto new_expr_id = builder_.AddChoices(new_choice_ids);
    std::cout << BNFGrammarPrinter(builder_.Get()).PrintRuleExpr(new_expr_id) << "\n";
    new_choice_ids = EliminateInstantLeftRecursion(new_choice_ids);
    new_expr_id = builder_.AddChoices(new_choice_ids);
    std::cout << BNFGrammarPrinter(builder_.Get()).PrintRuleExpr(new_expr_id) << "\n";
    return new_expr_id;
  }

  std::vector<int32_t> InlinePriorRules(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_choice_ids;
    for (auto i : rule_expr) {
      auto choice_expr = grammar_->GetRuleExpr(i);
      if (choice_expr.kind != DataKind::kSequence) {
        new_choice_ids.push_back(VisitExpr(choice_expr));
        continue;
      }
      auto atom_expr = grammar_->GetRuleExpr(choice_expr[0]);
      if (atom_expr.kind != DataKind::kRuleRef || atom_expr[0] >= cur_rule_id_) {
        new_choice_ids.push_back(VisitExpr(choice_expr));
        continue;
      }
      std::vector<int32_t> rest_sequence;
      for (auto j = 1; j < choice_expr.data_len; ++j) {
        rest_sequence.push_back(VisitExpr(grammar_->GetRuleExpr(choice_expr[j])));
      }

      auto choices_to_append = GetChoicesWithInlinedRule(atom_expr[0], rest_sequence);
      new_choice_ids.insert(new_choice_ids.end(), choices_to_append.begin(),
                            choices_to_append.end());
    }
    return new_choice_ids;
  }

  std::vector<int32_t> GetChoicesWithInlinedRule(int32_t refered_rule_id,
                                                 std::vector<int32_t> rest_sequence) {
    std::vector<int32_t> new_choice_ids;
    auto referred_rule = builder_.GetRule(refered_rule_id);
    auto referred_rule_expr = builder_.GetRuleExpr(referred_rule.rule_expr_id);

    ICHECK(referred_rule_expr.kind == DataKind::kChoices);
    for (auto j : referred_rule_expr) {
      auto referred_choice_expr = grammar_->GetRuleExpr(j);
      if (referred_choice_expr.kind == DataKind::kSequence) {
        std::vector<int32_t> new_sequence_ids;
        for (auto k : referred_choice_expr) {
          new_sequence_ids.push_back(VisitExpr(grammar_->GetRuleExpr(k)));
        }
        new_sequence_ids.insert(new_sequence_ids.end(), rest_sequence.begin(), rest_sequence.end());
        new_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
      } else {
        new_choice_ids.push_back(builder_.AddSequence(rest_sequence));
      }
    }
    return new_choice_ids;
  }

  std::vector<int32_t> EliminateInstantLeftRecursion(const std::vector<int32_t>& choice_ids) {
    bool require_elimination = std::any_of(choice_ids.begin(), choice_ids.end(), [&](int32_t i) {
      auto choice_expr = builder_.GetRuleExpr(i);
      if (choice_expr.kind != DataKind::kSequence) {
        return false;
      }
      auto atom_expr = builder_.GetRuleExpr(choice_expr[0]);
      return atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == cur_rule_id_;
    });

    if (!require_elimination) {
      return choice_ids;
    }

    auto new_rule_id = builder_.AddEmptyRule(cur_rule_name_ + "_left_recursion");
    std::vector<int32_t> cur_rule_choice_ids;
    std::vector<int32_t> new_rule_choice_ids;
    new_rule_choice_ids.push_back(builder_.AddEmptyStr());

    for (auto i : choice_ids) {
      auto choice_expr = builder_.GetRuleExpr(i);
      if (choice_expr.kind != DataKind::kSequence) {
        cur_rule_choice_ids.push_back(i);
        continue;
      }
      auto atom_expr = builder_.GetRuleExpr(choice_expr[0]);
      if (atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == cur_rule_id_) {
        auto new_sequence_ids = std::vector<int32_t>(choice_expr.begin() + 1, choice_expr.end());
        new_sequence_ids.push_back(builder_.AddRuleRef(new_rule_id));
        new_rule_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
      } else {
        auto new_sequence_ids = std::vector<int32_t>(choice_expr.begin(), choice_expr.end());
        new_sequence_ids.push_back(builder_.AddRuleRef(new_rule_id));
        cur_rule_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
      }
    }
    builder_.UpdateRuleBody(cur_rule_id_, builder_.AddChoices(cur_rule_choice_ids));
    return new_rule_choice_ids;
  }

  int32_t cur_rule_id_;
  std::string cur_rule_name_;
};

class BNFGrammarNormalizer : public BNFGrammarMutator<void, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply() final {
    grammar_ = BNFGrammarFlattener(grammar_).Apply();
    // std::cout << "1st finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    grammar_ = UnreachableEliminator(grammar_).Apply();
    // std::cout << "2nd finished\n";
    grammar_ = EpsilonEliminator(grammar_).Apply();
    grammar_ = UnreachableEliminator(grammar_).Apply();
    grammar_ = UnitProductionEliminator(grammar_).Apply();
    grammar_ = UnreachableEliminator(grammar_).Apply();
    grammar_ = LeftRecursionEliminator(grammar_).Apply();
    // grammar_ = UnreachableEliminator(grammar_).Apply();
    return grammar_;
  }
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_
