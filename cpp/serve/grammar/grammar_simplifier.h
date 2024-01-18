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

class RuleVisitGraph {
 public:
  RuleVisitGraph() = default;
  explicit RuleVisitGraph(int num_rules) : in_edges_(num_rules), out_edges_(num_rules) {}

  void AddEdge(int32_t from, int32_t to) {
    out_edges_[from].push_back(to);
    in_edges_[to].push_back(from);
  }

  void DelEdge(int32_t from, int32_t to) {
    auto& out_edges = out_edges_[from];
    auto& in_edges = in_edges_[to];
    auto out_it = std::find(out_edges.begin(), out_edges.end(), to);
    ICHECK(out_it != out_edges.end());
    out_edges.erase(out_it);
    auto in_it = std::find(in_edges.begin(), in_edges.end(), from);
    ICHECK(in_it != in_edges.end());
    in_edges.erase(in_it);
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

int32_t GetMainRuleId(const BNFGrammar& grammar) {
  for (int i = 0; i < static_cast<int>(grammar->NumRules()); ++i) {
    if (grammar->GetRule(i).name == "main") {
      return i;
    }
  }
  LOG(FATAL) << "The grammar should have a rule named \"main\"";
}

class UnreachableEliminator : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  BNFGrammar Apply() final {
    rule_visit_graph_ = RuleReachGraphFinder(grammar_).Apply();
    main_rule_id_ = GetMainRuleId(grammar_);
    FindRuleReachable();
    RenumberRules();
    BuildRuleBody();
    return builder_.Get();
  }

 private:
  void FindRuleReachable() {
    rule_reachable_ = std::vector<bool>(grammar_->NumRules(), false);
    std::unordered_set<int32_t> visited;
    std::queue<int32_t> queue;
    queue.push(main_rule_id_);
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
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
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

  int32_t VisitRuleRef(const RuleExpr& rule_expr) final {
    auto rule_id = rule_expr[0];
    ICHECK(rule_reachable_[rule_id]);
    ICHECK(rule_id_map_.count(rule_id));
    return builder_.AddRuleRef(rule_id_map_[rule_id]);
  }

  RuleVisitGraph rule_visit_graph_;
  int32_t main_rule_id_;
  std::vector<bool> rule_reachable_;
  std::unordered_map<int32_t, int32_t> rule_id_map_;
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
    return UnreachableEliminator(builder_.Get()).Apply();
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

class MainRuleNormalizer : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  MainRuleNormalizer(const BNFGrammar& grammar, bool grammar_can_be_empty)
      : BNFGrammarMutator(grammar), grammar_can_be_empty_(grammar_can_be_empty) {}

  BNFGrammar Apply() final {
    rule_visit_graph_ = RuleReachGraphFinder(grammar_).Apply();
    main_rule_id_ = GetMainRuleId(grammar_);
    RenumberRules();
    return builder_.Get();
  }

 private:
  void RenumberRules() {
    rule_id_map_.clear();
    bool create_new_main = rule_visit_graph_.GetInEdges(main_rule_id_).size() > 0;
    if (create_new_main) {
      builder_.AddEmptyRule("main");
      rule_id_map_[main_rule_id_] = builder_.AddEmptyRule(builder_.GetNewRuleName("main"));
    } else {
      rule_id_map_[main_rule_id_] = builder_.AddEmptyRule("main");
    }

    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      if (rule.name == "main") {
        continue;
      }
      rule_id_map_[i] = builder_.AddEmptyRule(rule.name);
    }

    if (create_new_main) {
      auto rule = grammar_->GetRule(main_rule_id_);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      cur_rule_id_ = 0;
      auto new_rule_expr_id = VisitExpr(rule_expr);
      builder_.UpdateRuleBody(0, new_rule_expr_id);
    }

    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = grammar_->GetRuleExpr(rule.rule_expr_id);
      cur_rule_id_ = rule_id_map_[i];
      auto new_rule_expr_id = VisitExpr(rule_expr);
      builder_.UpdateRuleBody(rule_id_map_[i], new_rule_expr_id);
    }
  }

  int32_t VisitChoices(const RuleExpr& rule_expr) final {
    std::vector<int32_t> choice_ids;
    for (int32_t i : rule_expr) {
      choice_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
    }
    if (grammar_can_be_empty_ && cur_rule_id_ == 0) {
      choice_ids.insert(choice_ids.begin(), builder_.AddEmptyStr());
    }
    return builder_.AddChoices(choice_ids);
  }

  int32_t VisitRuleRef(const RuleExpr& rule_expr) final {
    auto rule_id = rule_expr[0];
    ICHECK(rule_id_map_.count(rule_id));
    return builder_.AddRuleRef(rule_id_map_[rule_id]);
  }

  RuleVisitGraph rule_visit_graph_;
  int32_t main_rule_id_;
  std::unordered_map<int32_t, int32_t> rule_id_map_;
  int32_t cur_rule_id_;
  bool grammar_can_be_empty_;
};

class EpsilonEliminator : public BNFGrammarMutator<int32_t, std::tuple<BNFGrammar, bool, bool>> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  std::tuple<BNFGrammar, bool, bool> Apply() final {
    rule_visit_graph_ = RuleReachGraphFinder(grammar_).Apply();
    CollectExplicitEpsilonRules();
    FindEpsilonRules();
    FindEpsilonOnlyRules();
    EliminateEpsilonRules();
    grammar_ = UnreachableEliminator(builder_.Get()).Apply();
    return std::make_tuple(grammar_, grammar_can_be_empty_, grammar_must_be_empty_);
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
      if (rule.name == "main") {
        grammar_can_be_empty_ = epsilon_rule_id_set_.count(i);
        grammar_must_be_empty_ = epsilon_only_rule_id_set_.count(i);
      }
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
    if (new_choice_ids.empty()) {
      new_choice_ids.push_back(builder_.AddEmptyStr());
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
  bool grammar_can_be_empty_ = false;
  bool grammar_must_be_empty_ = false;
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
    bool grammar_can_be_empty, grammar_must_be_empty;
    std::tie(grammar_, grammar_can_be_empty, grammar_must_be_empty) =
        EpsilonEliminator(builder_.Get()).Apply();
    ICHECK(!grammar_must_be_empty && !grammar_can_be_empty);
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
        ICHECK(sequence_expr.kind == DataKind::kSequence);
        if (sequence_expr.data_len != 1) {
          continue;
        }
        auto atom_expr = grammar_->GetRuleExpr(sequence_expr[0]);
        if (atom_expr.kind == DataKind::kRuleRef) {
          updated_ = true;
          unit_visit_graph_.AddEdge(i, atom_expr[0]);
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
    bool grammar_can_be_empty, grammar_must_be_empty;
    std::tie(grammar_, grammar_can_be_empty, grammar_must_be_empty) =
        EpsilonEliminator(builder_.Get()).Apply();
    ICHECK(!grammar_must_be_empty && !grammar_can_be_empty);
    return grammar_;
  }

 private:
  int32_t VisitChoices(const RuleExpr& rule_expr) final {
    std::vector<int32_t> new_choice_ids = InlinePriorRules(rule_expr);
    new_choice_ids = EliminateInstantLeftRecursion(new_choice_ids);
    return builder_.AddChoices(new_choice_ids);
  }

  std::vector<int32_t> InlinePriorRules(const RuleExpr& rule_expr) {
    std::vector<int32_t> new_choice_ids;
    for (auto i : rule_expr) {
      new_choice_ids.push_back(VisitExpr(grammar_->GetRuleExpr(i)));
    }
    while (RequireInlinePriorRules(new_choice_ids)) {
      new_choice_ids = InlinePriorRulesOnce(new_choice_ids);
    }
    return new_choice_ids;
  }

  bool RequireInlinePriorRules(const std::vector<int32_t>& choice_ids) {
    auto check_sequence_start_with_prior_rule = [&](int32_t i) {
      auto sequence_expr = builder_.GetRuleExpr(i);
      if (sequence_expr.kind != DataKind::kSequence) {
        return false;
      }
      auto atom_expr = builder_.GetRuleExpr(sequence_expr[0]);
      return atom_expr.kind == DataKind::kRuleRef && atom_expr[0] < cur_rule_id_;
    };
    return std::any_of(choice_ids.begin(), choice_ids.end(), check_sequence_start_with_prior_rule);
  }

  std::vector<int32_t> InlinePriorRulesOnce(const std::vector<int32_t>& choice_ids) {
    std::vector<int32_t> new_choice_ids;
    for (auto i : choice_ids) {
      auto sequence_expr = builder_.GetRuleExpr(i);
      if (sequence_expr.kind != DataKind::kSequence) {
        new_choice_ids.push_back(i);
        continue;
      }
      auto atom_expr = builder_.GetRuleExpr(sequence_expr[0]);
      if (atom_expr.kind != DataKind::kRuleRef || atom_expr[0] >= cur_rule_id_) {
        new_choice_ids.push_back(i);
        continue;
      }
      std::vector<int32_t> rest_sequence;
      for (int j = 1; j < static_cast<int>(sequence_expr.data_len); ++j) {
        rest_sequence.push_back(VisitExpr(builder_.GetRuleExpr(sequence_expr[j])));
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
      auto referred_sequence_expr = builder_.GetRuleExpr(j);
      ICHECK(referred_sequence_expr.kind == DataKind::kSequence);
      std::vector<int32_t> new_sequence_ids;
      for (auto k : referred_sequence_expr) {
        new_sequence_ids.push_back(VisitExpr(builder_.GetRuleExpr(k)));
      }
      new_sequence_ids.insert(new_sequence_ids.end(), rest_sequence.begin(), rest_sequence.end());
      new_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
    }
    return new_choice_ids;
  }

  std::vector<int32_t> EliminateInstantLeftRecursion(const std::vector<int32_t>& choice_ids) {
    auto check_sequence_is_left_recursion = [&](int32_t i) {
      auto sequence_expr = builder_.GetRuleExpr(i);
      if (sequence_expr.kind != DataKind::kSequence) {
        return false;
      }
      auto atom_expr = builder_.GetRuleExpr(sequence_expr[0]);
      return atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == cur_rule_id_;
    };

    bool require_elimination =
        std::any_of(choice_ids.begin(), choice_ids.end(), check_sequence_is_left_recursion);
    if (!require_elimination) {
      return choice_ids;
    }

    auto new_rule_id = builder_.AddEmptyRule(cur_rule_name_ + "_recursion");
    std::vector<int32_t> cur_rule_choice_ids;
    std::vector<int32_t> new_rule_choice_ids;
    new_rule_choice_ids.push_back(builder_.AddEmptyStr());

    for (auto i : choice_ids) {
      auto sequence_expr = builder_.GetRuleExpr(i);
      if (sequence_expr.kind != DataKind::kSequence) {
        cur_rule_choice_ids.push_back(i);
        continue;
      }
      auto atom_expr = builder_.GetRuleExpr(sequence_expr[0]);
      if (atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == cur_rule_id_) {
        auto new_sequence_ids =
            std::vector<int32_t>(sequence_expr.begin() + 1, sequence_expr.end());
        new_sequence_ids.push_back(builder_.AddRuleRef(new_rule_id));
        new_rule_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
      } else {
        auto new_sequence_ids = std::vector<int32_t>(sequence_expr.begin(), sequence_expr.end());
        new_sequence_ids.push_back(builder_.AddRuleRef(new_rule_id));
        cur_rule_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
      }
    }
    builder_.UpdateRuleBody(new_rule_id, builder_.AddChoices(new_rule_choice_ids));

    return cur_rule_choice_ids;
  }

  int32_t cur_rule_id_;
  std::string cur_rule_name_;
};

class SequenceRuleInliner : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  BNFGrammar Apply() final {
    builder_ = BNFGrammarBuilder(grammar_);
    rule_to_inline_ = -1;
    for (int i = 0; i < static_cast<int>(grammar_->NumRules()); ++i) {
      auto rule = grammar_->GetRule(i);
      auto rule_expr = builder_.GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_expr.kind == DataKind::kChoices);
      builder_.AddRule(rule.name, VisitChoices(rule_expr)
    }
    while (GetRuleToInline()) {
      UpdateGrammar();
    }
    return UnreachableEliminator(builder_.Get()).Apply();
  }

 private:
  bool GetRuleToInline() {
    for (int i = 0; i < static_cast<int>(builder_.NumRules()); ++i) {
      auto rule = builder_.GetRule(i);
      if (rule.name == "main" || inlined_rules_.count(i)) {
        continue;
      }
      auto rule_body = builder_.GetRuleExpr(rule.rule_expr_id);
      ICHECK(rule_body.kind == DataKind::kChoices);
      if (rule_body.data_len > 1) {
        continue;
      }
      auto sequence_expr = builder_.GetRuleExpr(rule_body[0]);
      ICHECK(sequence_expr.kind == DataKind::kSequence);

      bool contain_self_ref =
          std::any_of(sequence_expr.begin(), sequence_expr.end(), [&](int32_t i) {
            auto atom_expr = builder_.GetRuleExpr(i);
            return atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == i;
          });
      if (contain_self_ref) {
        continue;
      }

      rule_to_inline_ = i;
      sequences_to_inline_ = std::vector<int32_t>(sequence_expr.begin(), sequence_expr.end());
      inlined_rules_.insert(rule_to_inline_);
      std::cout << "Rule to inline: " << rule.name << std::endl;
      return true;
    }
    std::cout << "No rule to inline" << std::endl;
    return false;
  }

  void UpdateGrammar() {
    for (int i = 0; i < static_cast<int>(builder_.NumRules()); ++i) {
      if (inlined_rules_.count(i)) {
        continue;
      }
      auto rule = builder_.GetRule(i);
      auto rule_expr = builder_.GetRuleExpr(rule.rule_expr_id);
      builder_.UpdateRuleBody(i, VisitExpr(rule_expr));
      std::cout << "updated rule body: " << BNFGrammarPrinter(builder_.Get()).PrintRule(i)
                << std::endl;
    }
  }

  int32_t VisitChoices(const RuleExpr& choices_expr) final {
    std::vector<int32_t> choice_ids;
    for (int32_t i : choices_expr) {
      choice_ids.push_back(VisitExpr(builder_.GetRuleExpr(i)));
    }
    return builder_.AddChoices(choice_ids);
  }

  int32_t VisitSequence(const RuleExpr& sequence_expr) final {
    std::vector<int32_t> new_sequence_ids;
    for (auto i : sequence_expr) {
      const auto atom_expr = builder_.GetRuleExpr(i);
      if (atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == rule_to_inline_) {
        new_sequence_ids.insert(new_sequence_ids.end(), sequences_to_inline_.begin(),
                                sequences_to_inline_.end());
      } else {
        new_sequence_ids.push_back(VisitExpr(atom_expr));
      }
    }
    return builder_.AddSequence(new_sequence_ids);
  }

  int32_t rule_to_inline_;
  std::vector<int32_t> sequences_to_inline_;
  std::unordered_set<int32_t> inlined_rules_;
};

class RuleInliner : public BNFGrammarMutator<int32_t, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;
  BNFGrammar Apply() final {
    grammar_ = SequenceRuleInliner(grammar_).Apply();
    return grammar_;
  }

  //  private:
  //   int32_t VisitChoices(const RuleExpr& rule_expr) final {
  //     std::vector<int32_t> new_choice_ids;
  //     new_choice_ids = InlinePriorRules(rule_expr);
  //     auto new_expr_id = builder_.AddChoices(new_choice_ids);
  //     new_choice_ids = EliminateInstantLeftRecursion(new_choice_ids);
  //     new_expr_id = builder_.AddChoices(new_choice_ids);
  //     return new_expr_id;
  //   }

  //   std::vector<int32_t> InlinePriorRules(const RuleExpr& rule_expr) {
  //     std::vector<int32_t> new_choice_ids;
  //     for (auto i : rule_expr) {
  //       auto sequence_expr = grammar_->GetRuleExpr(i);
  //       if (sequence_expr.kind != DataKind::kSequence) {
  //         new_choice_ids.push_back(VisitExpr(sequence_expr));
  //         continue;
  //       }
  //       auto atom_expr = grammar_->GetRuleExpr(sequence_expr[0]);
  //       if (atom_expr.kind != DataKind::kRuleRef || atom_expr[0] >= cur_rule_id_) {
  //         new_choice_ids.push_back(VisitExpr(sequence_expr));
  //         continue;
  //       }
  //       std::vector<int32_t> rest_sequence;
  //       for (auto j = 1; j < sequence_expr.data_len; ++j) {
  //         rest_sequence.push_back(VisitExpr(grammar_->GetRuleExpr(sequence_expr[j])));
  //       }

  //       auto choices_to_append = GetChoicesWithInlinedRule(atom_expr[0], rest_sequence);
  //       new_choice_ids.insert(new_choice_ids.end(), choices_to_append.begin(),
  //                             choices_to_append.end());
  //     }
  //     return new_choice_ids;
  //   }

  //   std::vector<int32_t> GetChoicesWithInlinedRule(int32_t refered_rule_id,
  //                                                  std::vector<int32_t> rest_sequence) {
  //     std::vector<int32_t> new_choice_ids;
  //     auto referred_rule = builder_.GetRule(refered_rule_id);
  //     auto referred_rule_expr = builder_.GetRuleExpr(referred_rule.rule_expr_id);

  //     ICHECK(referred_rule_expr.kind == DataKind::kChoices);
  //     for (auto j : referred_rule_expr) {
  //       auto referred_sequence_expr = builder_.GetRuleExpr(j);
  //       ICHECK(referred_sequence_expr.kind == DataKind::kSequence);
  //       std::vector<int32_t> new_sequence_ids;
  //       for (auto k : referred_sequence_expr) {
  //         new_sequence_ids.push_back(VisitExpr(builder_.GetRuleExpr(k)));
  //       }
  //       new_sequence_ids.insert(new_sequence_ids.end(), rest_sequence.begin(),
  //       rest_sequence.end()); new_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
  //     }
  //     return new_choice_ids;
  //   }

  //   std::vector<int32_t> EliminateInstantLeftRecursion(const std::vector<int32_t>& choice_ids) {
  //     auto check_sequence_left_recursion = [&](int32_t i) {
  //       auto sequence_expr = builder_.GetRuleExpr(i);
  //       if (sequence_expr.kind != DataKind::kSequence) {
  //         return false;
  //       }
  //       auto atom_expr = builder_.GetRuleExpr(sequence_expr[0]);
  //       return atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == cur_rule_id_;
  //     };

  //     bool require_elimination =
  //         std::any_of(choice_ids.begin(), choice_ids.end(), check_sequence_left_recursion);
  //     if (!require_elimination) {
  //       return choice_ids;
  //     }

  //     auto new_rule_id = builder_.AddEmptyRule(cur_rule_name_ + "_left_recursion");
  //     std::vector<int32_t> cur_rule_choice_ids;
  //     std::vector<int32_t> new_rule_choice_ids;
  //     new_rule_choice_ids.push_back(builder_.AddEmptyStr());

  //     for (auto i : choice_ids) {
  //       auto sequence_expr = builder_.GetRuleExpr(i);
  //       ICHECK(sequence_expr.kind == DataKind::kSequence);
  //       auto atom_expr = builder_.GetRuleExpr(sequence_expr[0]);
  //       if (atom_expr.kind == DataKind::kRuleRef && atom_expr[0] == cur_rule_id_) {
  //         auto new_sequence_ids =
  //             std::vector<int32_t>(sequence_expr.begin() + 1, sequence_expr.end());
  //         new_sequence_ids.push_back(builder_.AddRuleRef(new_rule_id));
  //         new_rule_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
  //       } else {
  //         auto new_sequence_ids = std::vector<int32_t>(sequence_expr.begin(),
  //         sequence_expr.end()); new_sequence_ids.push_back(builder_.AddRuleRef(new_rule_id));
  //         cur_rule_choice_ids.push_back(builder_.AddSequence(new_sequence_ids));
  //       }
  //     }
  //     builder_.UpdateRuleBody(new_rule_id, builder_.AddChoices(new_rule_choice_ids));

  //     return cur_rule_choice_ids;
  //   }

  //   int32_t cur_rule_id_;
  //   std::string cur_rule_name_;
};

class BNFGrammarNormalizer : public BNFGrammarMutator<void, BNFGrammar> {
 public:
  using BNFGrammarMutator::BNFGrammarMutator;

  BNFGrammar Apply() final {
    grammar_ = BNFGrammarFlattener(grammar_).Apply();
    std::cout << "1st finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    bool grammar_can_be_empty, grammar_must_be_empty;
    std::cout << "2 finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    std::tie(grammar_, grammar_can_be_empty, grammar_must_be_empty) =
        EpsilonEliminator(grammar_).Apply();
    std::cout << "3 finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    if (grammar_must_be_empty) {
      return grammar_;
    }
    grammar_ = UnitProductionEliminator(grammar_).Apply();
    std::cout << "4 finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    grammar_ = LeftRecursionEliminator(grammar_).Apply();
    std::cout << "5 finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    grammar_ = RuleInliner(grammar_).Apply();
    std::cout << "6 finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    grammar_ = MainRuleNormalizer(grammar_, grammar_can_be_empty).Apply();
    std::cout << "7 finished\nresult:" << BNFGrammarPrinter(grammar_).ToString() << "\n";
    return grammar_;
  }
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_SIMPLIFIER_H_
