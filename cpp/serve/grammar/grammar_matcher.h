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

#include "../config.h"
#include "../encoding.h"
#include "grammar.h"
#include "grammar_serializer.h"
#include "grammar_tokenizer_config.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

inline bool CharacterRangeContains(const BNFGrammarNode::RuleExpr& rule_expr,
                                   TCodepoint codepoint) {
  DCHECK(rule_expr.type == BNFGrammarNode::RuleExprType::kCharacterRange ||
         rule_expr.type == BNFGrammarNode::RuleExprType::kNegCharacterRange);
  for (int i = 0; i < rule_expr.size(); i += 2) {
    if (rule_expr.data[i] <= codepoint && codepoint <= rule_expr.data[i + 1]) {
      return rule_expr.type == BNFGrammarNode::RuleExprType::kCharacterRange;
    }
  }
  return rule_expr.type == BNFGrammarNode::RuleExprType::kNegCharacterRange;
}

struct RulePosition {
  /*! \brief The rule's id. */
  int32_t rule_id;
  /*! \brief Which choice in this rule is selected. */
  int sequence_id;
  /*! \brief Which element of the choice sequence is being visited. */
  int element_id;

  int32_t parent_id = -1;
  int reference_count = 0;

  bool operator==(const RulePosition& other) const {
    return rule_id == other.rule_id && sequence_id == other.sequence_id &&
           element_id == other.element_id && parent_id == other.parent_id;
  }
  bool operator!=(const RulePosition& other) const { return !(*this == other); }
};

inline constexpr RulePosition kInvalidRulePosition = {-1, -1, -1, -1, 0};

class RulePositionBuffer {
 public:
  int32_t Allocate(RulePosition rule_position) {
    int32_t id;
    if (free_nodes_.empty()) {
      buffer_.emplace_back();
      id = buffer_.size() - 1;
    } else {
      id = free_nodes_.back();
      DCHECK(buffer_[id] == kInvalidRulePosition);
      free_nodes_.pop_back();
    }
    rule_position.reference_count = 0;
    buffer_[id] = rule_position;
    return id;
  }

  void Free(int32_t id) {
    DCHECK(buffer_[id] != kInvalidRulePosition);
    buffer_[id] = kInvalidRulePosition;
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
  RulePositionTree(const BNFGrammar& grammar) : grammar_(grammar) {}

  static constexpr int32_t kNoParent = -1;

  int32_t NewNode(const RulePosition& rule_position) {
    auto id = node_buffer_.Allocate(rule_position);
    if (rule_position.parent_id != kNoParent) {
      node_buffer_[rule_position.parent_id].reference_count++;
    }
    return id;
  }

  bool IsEndPosition(const RulePosition& rule_position) const {
    return rule_position.parent_id == RulePositionTree::kNoParent &&
           grammar_->GetRuleExpr(rule_position.sequence_id).size() == rule_position.element_id;
  }

  RulePosition GetNextPosition(const RulePosition& rule_position) const {
    if (IsEndPosition(rule_position)) {
      return kInvalidRulePosition;
    }
    auto rule_id = rule_position.rule_id;
    auto sequence_id = rule_position.sequence_id;
    auto element_id = rule_position.element_id + 1;
    auto parent_id = rule_position.parent_id;
    while (parent_id != RulePositionTree::kNoParent &&
           grammar_->GetRuleExpr(sequence_id).size() == element_id) {
      auto parent_rule_position = node_buffer_[parent_id];
      rule_id = parent_rule_position.rule_id;
      sequence_id = parent_rule_position.sequence_id;
      element_id = parent_rule_position.element_id + 1;
      parent_id = parent_rule_position.parent_id;
    }
    return {rule_id, sequence_id, element_id, parent_id};
  }

  void AttachRefTo(int32_t id) {
    DCHECK(id != kNoParent);
    node_buffer_[id].reference_count++;
  }

  void RemoveRefTo(int32_t id) {
    DCHECK(id != kNoParent);
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
    DCHECK(id != kNoParent);
    return node_buffer_[id];
  }

  std::string PrintStackByTopId(int32_t top_id) const {
    std::stringstream ss;
    std::vector<int32_t> stack;
    for (auto cur_id = top_id; cur_id != kNoParent; cur_id = node_buffer_[cur_id].parent_id) {
      stack.push_back(cur_id);
    }
    ss << "{\n";
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
      const auto& rule_position = node_buffer_[*it];
      ss << "id: " << *it << ", rule: " << grammar_->GetRule(rule_position.rule_id).name
         << ", sequence: " << BNFGrammarPrinter(grammar_).PrintRuleExpr(rule_position.sequence_id)
         << ", element id: " << rule_position.element_id << "\n";
    }
    ss << "}";
    return ss.str();
  }

 private:
  BNFGrammar grammar_;
  RulePositionBuffer node_buffer_;
};

/*!
 * \brief Store the state in the past `max_rollback_steps` steps. The state is a list of stacks,
 * representing all possible paths on the pushdown automata.
 * Every stack is a list of RulePosition. They organize in a tree structure.
 * \details This history is managed as a circular buffer.
 */
class StackTopsWithHistory {
 public:
  StackTopsWithHistory(RulePositionTree* tree, int max_rollback_steps)
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

  void Rollback(int rollback_steps) {
    DCHECK(rollback_steps < stack_tops_history_.size())
        << "The number of requested rollback steps is greater than or equal to the current "
           "history "
        << "size: " << rollback_steps << " vs " << stack_tops_history_.size() << ".";
    while (rollback_steps--) {
      PopBack();
    }
  }

  const std::vector<int32_t>& LatestStackTops() const { return stack_tops_history_.back(); }

  std::string PrintLatest() const {
    const auto& latest_tops = LatestStackTops();
    std::stringstream ss;
    ss << "Stacks list size: " << latest_tops.size() << std::endl;
    int cnt = 0;
    for (auto id : latest_tops) {
      ss << "Stack #" << cnt << ": " << tree_->PrintStackByTopId(id) << "\n";
      ++cnt;
    }
    return ss.str();
  }

  int Size() const { return stack_tops_history_.size(); }

 private:
  void PopFront() {
    const auto& old_stack_tops = stack_tops_history_.front();
    for (auto id : old_stack_tops) {
      tree_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_front();
  }

  void PopBack() {
    const auto& new_stack_tops = stack_tops_history_.back();
    for (auto id : new_stack_tops) {
      tree_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_back();
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
  GrammarMatcherNode(const BNFGrammar& grammar, int max_rollback_steps = 0,
                     RulePosition init_rule_position = kInvalidRulePosition)
      : grammar_(grammar), tree_(grammar), stack_tops_with_history_(&tree_, max_rollback_steps) {
    InitStackState(init_rule_position);
  }

  bool AcceptChar(TCodepoint codepoint, bool drop_old = true, bool verbose = false) {
    if (verbose) {
      std::cout << "before stack: " << PrintStackState() << std::endl;
    }
    auto prev_stack_tops = stack_tops_with_history_.LatestStackTops();
    accept_char_stack_tops_buffer_.clear();
    for (auto old_top : prev_stack_tops) {
      const auto& rule_position = tree_[old_top];
      auto current_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
      if (rule_position.parent_id == RulePositionTree::kNoParent &&
          rule_position.element_id == current_sequence.size()) {
        // This RulePosition means previous elements has matched the complete rule.
        // But we are still need to accept a new character, so this stack will become invalid.
        continue;
      }
      auto current_char_range = grammar_->GetRuleExpr(current_sequence[rule_position.element_id]);
      ICHECK(current_char_range.type == RuleExprType::kCharacterRange ||
             current_char_range.type == RuleExprType::kNegCharacterRange);
      auto ok = CharacterRangeContains(current_char_range, codepoint);
      // start = std::chrono::high_resolution_clock::now();
      // end = std::chrono::high_resolution_clock::now();
      // overhead_time += end - start;
      if (!ok) {
        continue;
      }
      AddNewStackTops(old_top, &accept_char_stack_tops_buffer_);
    }
    if (accept_char_stack_tops_buffer_.empty()) {
      if (verbose) {
        std::cout << "Codepoint: " << codepoint << " \"" << CodepointToPrintable(codepoint)
                  << "\" Rejected" << std::endl;
      }
      return false;
    }
    stack_tops_with_history_.PushStackTops(accept_char_stack_tops_buffer_, drop_old);
    if (verbose) {
      std::cout << "Codepoint: " << codepoint << " \"" << CodepointToPrintable(codepoint)
                << "\" Stack size : " << stack_tops_with_history_.LatestStackTops().size()
                << std::endl;
      std::cout << "after stack: " << PrintStackState() << std::endl;
    }
    return true;
  }

  std::chrono::duration<double, std::milli> codepoint_set_total_time;
  std::chrono::duration<double, std::milli> overhead_time;

  bool MatchCompleteString(String str) {
    auto codepoints = Utf8StringToCodepoints(str.c_str());
    int accepted_cnt = 0;
    for (auto codepoint : codepoints) {
      if (!AcceptChar(codepoint, false, false)) {
        Rollback(accepted_cnt);
        return false;
      }
      ++accepted_cnt;
    }
    auto accepted = CanAcceptEnd();
    Rollback(accepted_cnt);
    return accepted;
  }

  bool CanAcceptEnd() const {
    auto last_stack_tops = stack_tops_with_history_.LatestStackTops();
    return std::any_of(last_stack_tops.begin(), last_stack_tops.end(),
                       [&](int32_t id) { return tree_.IsEndPosition(tree_[id]); });
  }

  std::vector<int32_t> FindRejectedTokenIds(const GrammarTokenizerConfig& tokenizer_config) {
    const auto& sorted_token_and_ids = tokenizer_config->sorted_token_and_ids;
    const auto& catagorized_tokens_for_grammar = tokenizer_config->catagorized_tokens_for_grammar;
    std::vector<int32_t> rejected_ids;
    int prev_matched_size = 0;
    std::cout << "Stack size: " << stack_tops_with_history_.LatestStackTops().size() << std::endl;
    for (int i = 0; i < static_cast<int>(sorted_token_and_ids.size()); ++i) {
      // Step 1. Find the length of the previous token that is useful for matching the current
      // token. (denoted by prev_useful_size)
      // prev_useful_size = min(prev_matched_size, len(longest_common_prefix(prev_token,
      // current_token))
      auto start = std::chrono::high_resolution_clock::now();
      auto prev_useful_size =
          std::min(prev_matched_size, static_cast<int>(sorted_token_and_ids[i].token.size()));
      if (i > 0) {
        for (int j = 0; j < prev_useful_size; ++j) {
          if (sorted_token_and_ids[i].token[j] != sorted_token_and_ids[i - 1].token[j]) {
            prev_useful_size = j;
            break;
          }
        }
      }
      auto end = std::chrono::high_resolution_clock::now();
      handle_past_time += end - start;

      // Step 2. Rollback the stack before matching the current token.
      start = std::chrono::high_resolution_clock::now();
      Rollback(prev_matched_size - prev_useful_size);
      end = std::chrono::high_resolution_clock::now();
      rollback_total_time += end - start;

      // Step 3. Match the current token, and update the prev_matched_size.
      start = std::chrono::high_resolution_clock::now();
      bool accepted = true;
      prev_matched_size = prev_useful_size;
      for (int j = prev_useful_size; j < sorted_token_and_ids[i].token.size(); ++j) {
        if (!AcceptChar(sorted_token_and_ids[i].token[j], false, false)) {
          accepted = false;
          break;
        }
        prev_matched_size = j + 1;
      }
      end = std::chrono::high_resolution_clock::now();
      accept_total_time += end - start;

      // Step 4. If the current token is accepted, push its id to the result.
      if (!accepted) {
        rejected_ids.push_back(sorted_token_and_ids[i].id);
      }
    }
    auto start = std::chrono::high_resolution_clock::now();
    Rollback(prev_matched_size);
    auto end = std::chrono::high_resolution_clock::now();
    rollback_total_time += end - start;
    return rejected_ids;
  }

  std::chrono::duration<double, std::milli> handle_past_time;
  std::chrono::duration<double, std::milli> rollback_total_time;
  std::chrono::duration<double, std::milli> accept_total_time;

  CatagorizedTokensForGrammar GetCatagorizedTokens(
      const std::vector<TokenAndId>& sorted_token_and_ids, bool is_root_rule) {
    std::unordered_set<int32_t> accepted_indices;
    std::unordered_set<int32_t> rejected_indices;
    std::unordered_set<int32_t> uncertain_indices;
    std::vector<bool> can_see_end_stack{CanAcceptEnd()};
    int prev_matched_size = 0;
    std::cout << "Stack size: " << stack_tops_with_history_.LatestStackTops().size() << std::endl;
    for (int i = 0; i < static_cast<int>(sorted_token_and_ids.size()); ++i) {
      const auto& token = sorted_token_and_ids[i].token;
      const auto* prev_token = i > 0 ? &sorted_token_and_ids[i - 1].token : nullptr;

      auto prev_useful_size = 0;
      if (prev_token) {
        prev_useful_size = std::min(prev_matched_size, static_cast<int>(token.size()));
        for (int j = 0; j < prev_useful_size; ++j) {
          if (token[j] != (*prev_token)[j]) {
            prev_useful_size = j;
            break;
          }
        }
        Rollback(prev_matched_size - prev_useful_size);
        can_see_end_stack.erase(can_see_end_stack.end() - (prev_matched_size - prev_useful_size),
                                can_see_end_stack.end());
      }

      bool accepted = true;
      bool can_see_end = can_see_end_stack.back();
      prev_matched_size = prev_useful_size;
      for (int j = prev_useful_size; j < token.size(); ++j) {
        if (!AcceptChar(token[j], false, false)) {
          accepted = false;
          break;
        }
        if (CanAcceptEnd()) {
          can_see_end = true;
        }
        can_see_end_stack.push_back(can_see_end);
        prev_matched_size = j + 1;
      }
      if (accepted) {
        accepted_indices.insert(i);
      } else if (can_see_end && !is_root_rule) {
        uncertain_indices.insert(i);
      } else {
        rejected_indices.insert(i);
      }
    }
    Rollback(prev_matched_size);
    std::cout << "Accepted: " << accepted_indices.size();
    if (accepted_indices.size() < 100) {
      std::cout << " (";
      for (auto i : accepted_indices) {
        std::cout << "<";
        for (auto cp : sorted_token_and_ids[i].token) {
          std::cout << CodepointToPrintable(cp);
        }
        std::cout << "> ";
      }
      std::cout << ")";
    }
    std::cout << std::endl;
    std::cout << "Rejected: " << rejected_indices.size();
    if (rejected_indices.size() < 100) {
      std::cout << " (";
      for (auto i : rejected_indices) {
        std::cout << "<";
        for (auto cp : sorted_token_and_ids[i].token) {
          std::cout << CodepointToPrintable(cp);
        }
        std::cout << "> ";
      }
      std::cout << ")";
    }
    std::cout << std::endl;
    std::cout << "Uncertain: " << uncertain_indices.size();
    if (uncertain_indices.size() < 100) {
      std::cout << " (";
      for (auto i : uncertain_indices) {
        std::cout << "<";
        for (auto cp : sorted_token_and_ids[i].token) {
          std::cout << CodepointToPrintable(cp);
        }
        std::cout << "> ";
      }
      std::cout << ")";
    }
    std::cout << std::endl;
    return CatagorizedTokensForGrammar(accepted_indices, rejected_indices, uncertain_indices);
  }

  void Rollback(int rollback_steps) { stack_tops_with_history_.Rollback(rollback_steps); }

  std::string PrintStackState() const { return stack_tops_with_history_.PrintLatest(); }

  static constexpr const char* _type_key = "mlc.serve.GrammarMatcher";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarMatcherNode, Object);

 public:
  void InitStackState(RulePosition init_rule_position) {
    if (init_rule_position == kInvalidRulePosition) {
      // Initialize the stack with the main rule.
      auto main_rule = grammar_->GetRule(0);
      auto main_rule_expr = grammar_->GetRuleExpr(main_rule.body_expr_id);
      std::vector<int32_t> new_stack_tops;
      for (auto i : main_rule_expr) {
        DCHECK(grammar_->GetRuleExpr(i).type == RuleExprType::kSequence ||
               grammar_->GetRuleExpr(i).type == RuleExprType::kEmptyStr);
        new_stack_tops.push_back(tree_.NewNode(RulePosition{0, i, 0, RulePositionTree::kNoParent}));
      }
      stack_tops_with_history_.PushStackTops(new_stack_tops);
    } else {
      init_rule_position.parent_id = RulePositionTree::kNoParent;
      stack_tops_with_history_.PushStackTops({tree_.NewNode(init_rule_position)});
    }
  }

  void AddNewStackTops(int32_t old_rule_position_id, std::vector<int32_t>* new_stack_tops) {
    auto cur_rule_position = tree_.GetNextPosition(tree_[old_rule_position_id]);

    while (!tree_.IsEndPosition(cur_rule_position)) {
      auto sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
      auto element = grammar_->GetRuleExpr(sequence[cur_rule_position.element_id]);
      if (element.type == RuleExprType::kCharacterRange ||
          element.type == RuleExprType::kNegCharacterRange) {
        new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
        break;
      } else {
        DCHECK(element.type == RuleExprType::kRuleRef);
        auto new_rule_id = element[0];
        auto new_rule = grammar_->GetRule(new_rule_id);
        auto new_rule_expr = grammar_->GetRuleExpr(new_rule.body_expr_id);
        DCHECK(new_rule_expr.type == RuleExprType::kChoices);

        bool contain_empty = false;

        for (auto j : new_rule_expr) {
          auto sequence = grammar_->GetRuleExpr(j);
          if (sequence.type == RuleExprType::kEmptyStr) {
            contain_empty = true;
            continue;
          }
          DCHECK(sequence.type == RuleExprType::kSequence);
          DCHECK(grammar_->GetRuleExpr(sequence[0]).type == RuleExprType::kCharacterRange ||
                 grammar_->GetRuleExpr(sequence[0]).type == RuleExprType::kNegCharacterRange);
          // Note: rule_position is not inserted to the tree yet, so it need to be inserted first
          auto parent_id = tree_.NewNode(cur_rule_position);
          new_stack_tops->push_back(tree_.NewNode(RulePosition{new_rule_id, j, 0, parent_id}));
        }

        if (!contain_empty) {
          break;
        }
      }
      cur_rule_position = tree_.GetNextPosition(cur_rule_position);
    }

    if (tree_.IsEndPosition(cur_rule_position)) {
      new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
    }
  }

  BNFGrammar grammar_;
  RulePositionTree tree_;
  StackTopsWithHistory stack_tops_with_history_;

  std::vector<int32_t> accept_char_stack_tops_buffer_;
};

class GrammarMatcher : public ObjectRef {
 public:
  GrammarMatcher(const BNFGrammar& grammar, int max_rollback_steps = 0,
                 RulePosition init_rule_position = kInvalidRulePosition)
      : ObjectRef(
            make_object<GrammarMatcherNode>(grammar, max_rollback_steps, init_rule_position)) {}

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(GrammarMatcher, ObjectRef, GrammarMatcherNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_MATCHER_H_
