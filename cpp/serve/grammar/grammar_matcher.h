/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "../encoding.h"
#include "grammar.h"
#include "grammar_serializer.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

class CodepointSet {
 private:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

 public:
  CodepointSet() = default;

  explicit CodepointSet(const RuleExpr& rule_expr) {
    ICHECK(rule_expr.type == RuleExprType::kCharacterRange ||
           rule_expr.type == RuleExprType::kNegCharacterRange);
    for (int i = 0; i < rule_expr.size(); i += 2) {
      ranges_.push_back({rule_expr.data[i], rule_expr.data[i + 1]});
    }
    Simplify();
    if (rule_expr.type == RuleExprType::kNegCharacterRange) {
      ToNegative();
    }
  }

  bool Contains(int32_t codepoint) const {
    for (const auto& range : ranges_) {
      if (range.start <= codepoint && codepoint <= range.end) {
        return true;
      }
    }
    return false;
  }

  void Union(const CodepointSet& other) {
    ranges_.insert(ranges_.end(), other.ranges_.begin(), other.ranges_.end());
    Simplify();
  }

 private:
  void Simplify() {
    std::sort(ranges_.begin(), ranges_.end());
    for (auto it = ranges_.begin(); it != ranges_.end() - 1; ++it) {
      auto next = it + 1;
      if (it->end >= next->start) {
        it->end = std::max(it->end, next->end);
        ranges_.erase(next);
      }
    }
  }

  void ToNegative() {
    static constexpr int32_t kMaxCodepoint = 0x10FFFF;
    std::vector<Range> new_ranges;
    int32_t start = 0;
    for (const auto& range : ranges_) {
      if (start < range.start) {
        new_ranges.push_back({start, range.start - 1});
      }
      start = range.end + 1;
    }
    if (start <= kMaxCodepoint) {
      new_ranges.push_back({start, kMaxCodepoint});
    }
    ranges_ = std::move(new_ranges);
  }

  struct Range {
    int32_t start;
    int32_t end;
    bool operator<(const Range& other) const {
      return start == other.start ? end < other.end : start < other.start;
    }
  };
  std::vector<Range> ranges_;
};

struct RulePosition {
  /*! \brief The rule's id. */
  int32_t rule_id;
  /*! \brief Which choice in this rule is selected. */
  int sequence_id;
  /*! \brief Which element of the choice sequence is being visited. */
  int element_id;

  int32_t parent_id = -1;
  int reference_count = 0;
};

class RulePositionBuffer {
 public:
  int32_t Allocate(int32_t rule_id, int32_t sequence_id, int32_t element_id, int32_t parent_id,
                   int32_t reference_count = 0) {
    int32_t id;
    if (free_nodes_.empty()) {
      buffer_.emplace_back();
      id = buffer_.size() - 1;
    } else {
      id = free_nodes_.back();
      ICHECK(buffer_[id].rule_id == -1);
      free_nodes_.pop_back();
    }
    buffer_[id] = RulePosition{rule_id, sequence_id, element_id, parent_id, reference_count};
    return id;
  }

  void Free(int32_t id) {
    ICHECK(buffer_[id].rule_id != -1);
    buffer_[id] = RulePosition{-1, -1, -1, -1, 0};
    free_nodes_.push_back(id);
  }

  RulePosition& operator[](int32_t id) { return buffer_[id]; }
  const RulePosition& operator[](int32_t id) const { return buffer_[id]; }

 private:
  std::vector<RulePosition> buffer_;
  std::vector<int32_t> free_nodes_;
};

class RulePositionTree {
 public:
  static constexpr int32_t kNoParent = -1;

  int32_t NewNode(int32_t rule_id, int32_t sequence_id, int32_t element_id,
                  int32_t parent_id = kNoParent) {
    auto id = node_buffer_.Allocate(rule_id, sequence_id, element_id, parent_id);
    if (parent_id != kNoParent) {
      node_buffer_[parent_id].reference_count++;
    }
    return id;
  }

  void AttachRefTo(int32_t id) {
    ICHECK(id != kNoParent);
    node_buffer_[id].reference_count++;
  }

  void RemoveRefTo(int32_t id) {
    ICHECK(id != kNoParent);
    auto cur_node = id;
    while (cur_node != kNoParent) {
      node_buffer_[cur_node].reference_count--;
      if (node_buffer_[cur_node].reference_count != 0) {
        break;
      }
      node_buffer_.Free(cur_node);
      cur_node = node_buffer_[cur_node].parent_id;
    }
  }

  const RulePosition& operator[](int32_t id) const {
    ICHECK(id != kNoParent);
    return node_buffer_[id];
  }

  std::string PrintStackByTopId(const BNFGrammar& grammar, int32_t top_id) {
    std::stringstream ss;
    std::vector<int32_t> stack;
    for (auto cur_id = top_id; cur_id != kNoParent; cur_id = node_buffer_[cur_id].parent_id) {
      stack.push_back(cur_id);
    }
    ss << "{\n";
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
      const auto& rule_position = node_buffer_[*it];
      ss << "id: " << *it << ", rule: " << grammar->GetRule(rule_position.rule_id).name
         << ", sequence: " << BNFGrammarPrinter(grammar).PrintRuleExpr(rule_position.sequence_id)
         << ", element id: " << rule_position.element_id << "\n";
    }
    ss << "}";
    return ss.str();
  }

 private:
  RulePositionBuffer node_buffer_;
};

/*!
 * \brief Store the state in the past `max_rollback_steps` steps. The state is a list of stacks,
 * representing all possible paths on the pushdown automata.
 * Every stack is a list of RulePosition. They organize in a tree structure.
 * \details This history is managed as a circular buffer.
 */
class StacksListWithHistory {
 public:
  StacksListWithHistory(RulePositionTree* tree, int max_rollback_steps)
      : tree_(tree), max_rollback_steps_(max_rollback_steps) {}

  void PushStackTops(const std::vector<int32_t>& stack_tops, bool drop_old = true) {
    stack_tops_history_.push_back(stack_tops);
    for (auto id : stack_tops) {
      tree_->AttachRefTo(id);
    }
    if (drop_old) {
      while (stack_tops_history_.size() > max_rollback_steps_ + 1) {
        PopFront();
      }
    }
  }

  void Rollback(int rollback_cnt) {
    CHECK(rollback_cnt < stack_tops_history_.size())
        << "The number of requested rollback steps is greater than or equal to the current history "
        << "size: " << rollback_cnt << " vs " << stack_tops_history_.size() << ".";
    while (rollback_cnt--) {
      PopBack();
    }
  }

  const std::vector<int32_t>& Back() const { return stack_tops_history_.back(); }

  std::string PrintBack(const BNFGrammar& grammar) const {
    std::stringstream ss;
    ss << "Stacks list size: " << stack_tops_history_.back().size() << std::endl;
    int cnt = 0;
    for (auto id : stack_tops_history_.back()) {
      ss << "Stack #" << cnt << ": " << tree_->PrintStackByTopId(grammar, id) << "\n";
      ++cnt;
    }
    return ss.str();
  }

  int Size() const { return stack_tops_history_.size(); }

 private:
  void PopFront() {
    auto old_stack_tops = stack_tops_history_.front();
    stack_tops_history_.pop_front();
    for (auto id : old_stack_tops) {
      tree_->RemoveRefTo(id);
    }
  }

  void PopBack() {
    auto new_stack_tops = stack_tops_history_.back();
    stack_tops_history_.pop_back();
    for (auto id : new_stack_tops) {
      tree_->RemoveRefTo(id);
    }
  }

  RulePositionTree* tree_;
  int max_rollback_steps_;
  std::deque<std::vector<int32_t>> stack_tops_history_;
};

class GrammarMatcherNode : public Object {
 private:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

 public:
  GrammarMatcherNode(const BNFGrammar& grammar, int max_rollback_steps = 1)
      : grammar_(grammar), stacks_list_with_history_(&tree_, max_rollback_steps) {
    PushInitStackTops();
  }

  bool AcceptChar(TCodepoint codepoint, bool drop_old = true) {
    auto last_stack_tops = stacks_list_with_history_.Back();
    std::vector<int32_t> new_stack_tops;
    for (auto id : last_stack_tops) {
      const auto& rule_position = tree_[id];
      auto sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
      if (rule_position.element_id == sequence.size()) {
        // The state of this stack means previous elements has matched the whole rule.
        // But we are still accepting characters, so this stack cannot match the current elements.
        continue;
      }
      auto element = grammar_->GetRuleExpr(sequence[rule_position.element_id]);
      if (element.type == RuleExprType::kCharacterRange ||
          element.type == RuleExprType::kNegCharacterRange) {
        if (!CodepointSet(element).Contains(codepoint)) {
          continue;
        }
        new_stack_tops.push_back(
            GetUpdatedRulePositionId(rule_position.rule_id, rule_position.sequence_id,
                                     rule_position.element_id + 1, rule_position.parent_id));

      } else {
        ICHECK(element.type == RuleExprType::kRuleRef);
        auto new_rule_id = element[0];
        auto new_rule = grammar_->GetRule(new_rule_id);
        auto new_rule_expr = grammar_->GetRuleExpr(new_rule.rule_expr_id);
        ICHECK(new_rule_expr.type == RuleExprType::kChoices);

        for (auto j : new_rule_expr) {
          auto sequence = grammar_->GetRuleExpr(j);
          ICHECK(sequence.type == RuleExprType::kSequence);
          auto sequence_first_element = grammar_->GetRuleExpr(sequence[0]);
          ICHECK(sequence_first_element.type == RuleExprType::kCharacterRange ||
                 sequence_first_element.type == RuleExprType::kNegCharacterRange);
          if (!CodepointSet(sequence_first_element).Contains(codepoint)) {
            continue;
          }
          new_stack_tops.push_back(GetUpdatedRulePositionId(new_rule_id, j, 1, id));
        }
      }
    }
    if (new_stack_tops.empty()) {
      return false;
    }
    stacks_list_with_history_.PushStackTops(new_stack_tops, drop_old);
    std::cout << "Codepoint: " << codepoint << " \"" << CodepointToPrintable(codepoint)
              << "\" Stack size : " << stacks_list_with_history_.Back().size() << std::endl;
    return true;
  }

  bool CanAcceptEnd() const {
    auto last_stack_tops = stacks_list_with_history_.Back();
    for (auto id : last_stack_tops) {
      const auto& rule_position = tree_[id];
      auto sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
      if (rule_position.element_id == sequence.size()) {
        return true;
      }
    }
    return false;
  }

  bool MatchCompleteString(const std::string& str) {
    for (auto c : str) {
      if (!AcceptChar(c)) {
        return false;
      }
    }
    return CanAcceptEnd();
  }

  bool CanAcceptString(const std::string& str) {
    int rollback_cnt = 0;
    for (auto c : str) {
      if (!AcceptChar(c, false)) {
        Rollback(rollback_cnt);
        return false;
      }
      ++rollback_cnt;
    }
    Rollback(rollback_cnt);
    return true;
  }

  void Rollback(int rollback_cnt) { stacks_list_with_history_.Rollback(rollback_cnt); }

  static constexpr const char* _type_key = "mlc.serve.GrammarMatcher";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarMatcherNode, Object);

 private:
  void PushInitStackTops() {
    auto main_rule = grammar_->GetRule(0);
    auto main_rule_expr = grammar_->GetRuleExpr(main_rule.rule_expr_id);
    std::vector<int32_t> new_stack_tops;
    for (auto i : main_rule_expr) {
      auto sequence = grammar_->GetRuleExpr(i);
      ICHECK(sequence.type == RuleExprType::kSequence || sequence.type == RuleExprType::kEmptyStr);
      new_stack_tops.push_back(tree_.NewNode(0, i, 0, RulePositionTree::kNoParent));
    }
    stacks_list_with_history_.PushStackTops(new_stack_tops);
  }

  int32_t GetUpdatedRulePositionId(int32_t rule_id, int32_t sequence_id, int32_t element_id,
                                   int32_t parent_id) {
    while (parent_id != RulePositionTree::kNoParent &&
           grammar_->GetRuleExpr(sequence_id).size() == element_id) {
      auto parent_rule_position = tree_[parent_id];
      rule_id = parent_rule_position.rule_id;
      sequence_id = parent_rule_position.sequence_id;
      element_id = parent_rule_position.element_id + 1;
      parent_id = parent_rule_position.parent_id;
    }
    return tree_.NewNode(rule_id, sequence_id, element_id, parent_id);
  }

  const BNFGrammar& grammar_;
  RulePositionTree tree_;
  StacksListWithHistory stacks_list_with_history_;
};

class GrammarMatcher : public ObjectRef {
 public:
  GrammarMatcher(const BNFGrammar& grammar, int max_rollback_steps = 0) {
    data_ = make_object<GrammarMatcherNode>(grammar, max_rollback_steps);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarMatcher, ObjectRef, GrammarMatcherNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
