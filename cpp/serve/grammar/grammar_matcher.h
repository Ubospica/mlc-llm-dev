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

#include "../../support/dynamic_bitset.h"
#include "../config.h"
#include "../encoding.h"
#include "grammar.h"
#include "grammar_serializer.h"
#include "grammar_tokenizer_config.h"

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

inline bool CharacterClassContains(const BNFGrammarNode::RuleExpr& rule_expr,
                                   TCodepoint codepoint) {
  DCHECK(rule_expr.type == BNFGrammarNode::RuleExprType::kCharacterClass ||
         rule_expr.type == BNFGrammarNode::RuleExprType::kNegCharacterClass);
  for (int i = 0; i < rule_expr.size(); i += 2) {
    if (rule_expr.data[i] <= codepoint && codepoint <= rule_expr.data[i + 1]) {
      return rule_expr.type == BNFGrammarNode::RuleExprType::kCharacterClass;
    }
  }
  return rule_expr.type == BNFGrammarNode::RuleExprType::kNegCharacterClass;
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

  static constexpr int32_t kNoParent = -1;

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

  size_t Capacity() const { return buffer_.size(); }

  size_t Size() const {
    DCHECK(buffer_.size() >= free_nodes_.size());
    return buffer_.size() - free_nodes_.size();
  }

  RulePosition& operator[](int32_t id) { return buffer_[id]; }
  const RulePosition& operator[](int32_t id) const { return buffer_[id]; }

  friend class RulePositionTree;

 private:
  std::vector<RulePosition> buffer_;
  std::vector<int32_t> free_nodes_;
};

class RulePositionTree {
 public:
  RulePositionTree(const BNFGrammar& grammar) : grammar_(grammar) {}

  int32_t NewNode(const RulePosition& rule_position) {
    auto id = node_buffer_.Allocate(rule_position);
    // std::cout << "parent: " << rule_position.parent_id << std::endl;
    if (rule_position.parent_id != RulePosition::kNoParent) {
      DCHECK(rule_position.parent_id < static_cast<int32_t>(node_buffer_.Capacity()) &&
             node_buffer_[rule_position.parent_id] != kInvalidRulePosition);
      node_buffer_[rule_position.parent_id].reference_count++;
    }
    return id;
  }

  bool IsEndPosition(const RulePosition& rule_position) const {
    return rule_position.parent_id == RulePosition::kNoParent &&
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
    while (parent_id != RulePosition::kNoParent &&
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
    DCHECK(id != RulePosition::kNoParent);
    node_buffer_[id].reference_count++;
  }

  void RemoveRefTo(int32_t id) {
    DCHECK(id != RulePosition::kNoParent);
    auto cur_node = id;
    while (cur_node != RulePosition::kNoParent) {
      node_buffer_[cur_node].reference_count--;
      if (node_buffer_[cur_node].reference_count != 0) {
        break;
      }
      auto next_node = node_buffer_[cur_node].parent_id;
      node_buffer_.Free(cur_node);
      cur_node = next_node;
    }
  }

  const RulePosition& operator[](int32_t id) const {
    DCHECK(id != RulePosition::kNoParent);
    return node_buffer_[id];
  }

  std::string PrintStackByTopId(int32_t top_id) const {
    std::stringstream ss;
    std::vector<int32_t> stack;
    for (auto cur_id = top_id; cur_id != RulePosition::kNoParent;
         cur_id = node_buffer_[cur_id].parent_id) {
      stack.push_back(cur_id);
    }
    ss << "{\n";
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
      ss << PrintNode(*it) << "\n";
    }
    ss << "}";
    return ss.str();
  }

  std::string PrintNode(int32_t id) const {
    std::stringstream ss;
    const auto& rule_position = node_buffer_[id];
    ss << "id: " << id << ", rule: " << grammar_->GetRule(rule_position.rule_id).name
       << ", sequence: " << BNFGrammarPrinter(grammar_).PrintRuleExpr(rule_position.sequence_id)
       << ", element id: " << rule_position.element_id << ", parent id: " << rule_position.parent_id
       << ", ref count: " << rule_position.reference_count;
    return ss.str();
  }

  size_t NodeCount() const { return node_buffer_.Size(); }
  size_t BufferCapacity() const { return node_buffer_.Capacity(); }

  void CheckWellFormed(const std::vector<int32_t>& outside_pointers) const {
    const auto& buffer = node_buffer_.buffer_;
    std::unordered_set<int32_t> free_nodes_set(node_buffer_.free_nodes_.begin(),
                                               node_buffer_.free_nodes_.end());
    int buffer_size = static_cast<int>(buffer.size());
    std::vector<int> new_reference_counter(buffer_size, 0);
    std::vector<bool> visited(buffer_size, false);
    std::queue<int> visit_queue;
    for (auto id : outside_pointers) {
      CHECK(id >= 0 && id < buffer_size);
      CHECK(buffer[id] != kInvalidRulePosition);
      new_reference_counter[id]++;
      if (visited[id] == false) {
        visited[id] = true;
        visit_queue.push(id);
      }
    }
    while (!visit_queue.empty()) {
      auto cur_id = visit_queue.front();
      visit_queue.pop();
      const auto& rule_position = buffer[cur_id];
      if (rule_position.parent_id != RulePosition::kNoParent) {
        CHECK(rule_position.parent_id >= 0 && rule_position.parent_id < buffer_size);
        CHECK(buffer[rule_position.parent_id] != kInvalidRulePosition);
        new_reference_counter[rule_position.parent_id]++;
        if (visited[rule_position.parent_id] == false) {
          visited[rule_position.parent_id] = true;
          visit_queue.push(rule_position.parent_id);
        }
      }
    }

    for (int i = 0; i < static_cast<int32_t>(buffer.size()); ++i) {
      if (free_nodes_set.count(i)) {
        CHECK(visited[i] == false);
        CHECK(buffer[i] == kInvalidRulePosition);
      } else {
        CHECK(visited[i] == true);
        CHECK(buffer[i] != kInvalidRulePosition);
        CHECK(new_reference_counter[i] == buffer[i].reference_count)
            << "Reference counters unmatch for node #" << i << ": Updated "
            << new_reference_counter[i] << ", Original " << buffer[i].reference_count;
      }
    }
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

  std::string PrintStackTops(int steps_behind_latest = 0) const {
    const auto& latest_tops =
        stack_tops_history_[stack_tops_history_.size() - 1 - steps_behind_latest];
    std::stringstream ss;
    ss << "Stacks tops size: " << latest_tops.size() << std::endl;
    int cnt = 0;
    for (auto id : latest_tops) {
      ss << "Stack #" << cnt << ": " << tree_->PrintStackByTopId(id) << "\n";
      ++cnt;
    }
    return ss.str();
  }

  int Size() const { return stack_tops_history_.size(); }

  void CheckWellFormed() const {
    std::vector<int32_t> outside_pointers;
    for (const auto& stack_tops : stack_tops_history_) {
      outside_pointers.insert(outside_pointers.end(), stack_tops.begin(), stack_tops.end());
    }
    tree_->CheckWellFormed(outside_pointers);
  }

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
      std::cout << "before stack: " << PrintStackTops() << std::endl;
    }
    auto prev_stack_tops = stack_tops_with_history_.LatestStackTops();
    accept_char_stack_tops_buffer_.clear();
    for (auto old_top : prev_stack_tops) {
      const auto& rule_position = tree_[old_top];
      auto current_sequence = grammar_->GetRuleExpr(rule_position.sequence_id);
      if (rule_position.parent_id == RulePosition::kNoParent &&
          rule_position.element_id == current_sequence.size()) {
        // This RulePosition means previous elements has matched the complete rule.
        // But we are still need to accept a new character, so this stack will become invalid.
        continue;
      }
      auto current_char_class = grammar_->GetRuleExpr(current_sequence[rule_position.element_id]);
      ICHECK(current_char_class.type == RuleExprType::kCharacterClass ||
             current_char_class.type == RuleExprType::kNegCharacterClass);
      auto ok = CharacterClassContains(current_char_class, codepoint);
      // start = std::chrono::high_resolution_clock::now();
      // end = std::chrono::high_resolution_clock::now();
      // overhead_time += end - start;
      if (!ok) {
        continue;
      }
      UpdateNewStackTops(old_top, &accept_char_stack_tops_buffer_);
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
      std::cout << "after stack: " << PrintStackTops() << std::endl;
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
    auto accepted = CanReachEnd();
    Rollback(accepted_cnt);
    return accepted;
  }

  bool CanReachEnd() const {
    auto last_stack_tops = stack_tops_with_history_.LatestStackTops();
    return std::any_of(last_stack_tops.begin(), last_stack_tops.end(),
                       [&](int32_t id) { return tree_.IsEndPosition(tree_[id]); });
  }

  // void FindRejectedTokenIds0(const GrammarTokenizerConfig& tokenizer_config,
  //                            DynamicBitSet* rejected_ids) {
  //   const auto& sorted_token_and_ids = tokenizer_config->sorted_token_and_ids;
  //   const auto& catagorized_tokens_for_grammar =
  //   tokenizer_config->catagorized_tokens_for_grammar;
  //   rejected_ids->Reset(tokenizer_config->vocab_size);
  //   int prev_matched_size = 0;
  //   std::cout << "Stack size: " << stack_tops_with_history_.LatestStackTops().size() <<
  //   std::endl; std::vector<int32_t> accepted_indices; for (int i = 0; i <
  //   static_cast<int>(sorted_token_and_ids.size()); ++i) {
  //     const auto& token = sorted_token_and_ids[i].token;
  //     const auto* prev_token = i > 0 ? &sorted_token_and_ids[i - 1].token : nullptr;
  //     // Step 1. Find the length of the previous token that is useful for matching the current
  //     // token. (denoted by prev_useful_size)
  //     // prev_useful_size = min(prev_matched_size, len(longest_common_prefix(prev_token,
  //     // current_token))
  //     auto start = std::chrono::high_resolution_clock::now();
  //     auto prev_useful_size = 0;
  //     if (prev_token) {
  //       prev_useful_size = std::min(prev_matched_size, static_cast<int>(token.size()));
  //       for (int j = 0; j < prev_useful_size; ++j) {
  //         if (token[j] != (*prev_token)[j]) {
  //           prev_useful_size = j;
  //           break;
  //         }
  //       }
  //       Rollback(prev_matched_size - prev_useful_size);
  //     }
  //     auto end = std::chrono::high_resolution_clock::now();
  //     handle_past_time += end - start;

  //     // Step 3. Match the current token, and update the prev_matched_size.
  //     start = std::chrono::high_resolution_clock::now();
  //     bool accepted = true;
  //     prev_matched_size = prev_useful_size;
  //     for (int j = prev_useful_size; j < token.size(); ++j) {
  //       if (!AcceptChar(token[j], false, false)) {
  //         accepted = false;
  //         break;
  //       }
  //       prev_matched_size = j + 1;
  //     }
  //     end = std::chrono::high_resolution_clock::now();
  //     accept_total_time += end - start;

  //     // Step 4. If the current token is accepted, push its id to the result.
  //     if (!accepted) {
  //       rejected_ids->SetConst(sorted_token_and_ids[i].id);
  //     } else {
  //       accepted_indices.push_back(i);
  //     }
  //   }
  //   if (accepted_indices.size() < 200) {
  //     std::cout << "Accepted size: " << accepted_indices.size() << " contents: (";
  //     for (auto i : accepted_indices) {
  //       std::cout << "<";
  //       for (auto cp : sorted_token_and_ids[i].token) {
  //         std::cout << CodepointToPrintable(cp);
  //       }
  //       std::cout << "> ";
  //     }
  //     std::cout << ")" << std::endl;
  //   }
  //   auto start = std::chrono::high_resolution_clock::now();
  //   Rollback(prev_matched_size);
  //   auto end = std::chrono::high_resolution_clock::now();
  //   rollback_total_time += end - start;
  // }

  static void UnionizeWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
    // target and source are sorted, and avoid memory allocation in this function
    static std::vector<int32_t> result;
    result.clear();
    result.reserve(target->size() + source.size());
    auto it1 = target->begin();
    auto it2 = source.begin();
    while (it1 != target->end() && it2 != source.end()) {
      if (*it1 < *it2) {
        result.push_back(*it1);
        ++it1;
      } else if (*it1 > *it2) {
        result.push_back(*it2);
        ++it2;
      } else {
        result.push_back(*it1);
        ++it1;
        ++it2;
      }
    }
    while (it1 != target->end()) {
      result.push_back(*it1);
      ++it1;
    }
    while (it2 != source.end()) {
      result.push_back(*it2);
      ++it2;
    }
    target->swap(result);
  }

  static void IntersectWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
    // target and source are sorted, and avoid memory allocation in this function
    static std::vector<int32_t> result;

    if (!target->empty() && target->at(0) == -1) {
      *target = source;
      return;
    }

    result.clear();
    result.reserve(std::min(target->size(), source.size()));
    auto it1 = target->begin();
    auto it2 = source.begin();
    while (it1 != target->end() && it2 != source.end()) {
      if (*it1 < *it2) {
        ++it1;
      } else if (*it1 > *it2) {
        ++it2;
      } else {
        result.push_back(*it1);
        ++it1;
        ++it2;
      }
    }
    target->swap(result);
  }

  static void DifferenceWith(std::vector<int32_t>* target, const std::vector<int32_t>& source) {
    // target and source are sorted, and avoid memory allocation in this function
    static std::vector<int32_t> result;
    result.clear();
    result.reserve(target->size());
    auto it1 = target->begin();
    auto it2 = source.begin();
    while (it1 != target->end() && it2 != source.end()) {
      if (*it1 < *it2) {
        result.push_back(*it1);
        ++it1;
      } else if (*it1 > *it2) {
        ++it2;
      } else {
        ++it1;
        ++it2;
      }
    }
    while (it1 != target->end()) {
      result.push_back(*it1);
      ++it1;
    }
    target->swap(result);
  }

  // todo: if stack is root stack, do not need to consider uncertain
  void FindRejectedTokenIds(const GrammarTokenizerConfig& tokenizer_config,
                            DynamicBitSet* rejected_ids) {
    const auto& sorted_token_and_ids = tokenizer_config->sorted_token_and_ids;
    const auto& catagorized_tokens_for_grammar = tokenizer_config->catagorized_tokens_for_grammar;
    const auto& latest_stack_tops = stack_tops_with_history_.LatestStackTops();
    std::cout << "Top size: " << latest_stack_tops.size() << std::endl;

    std::vector<int32_t> accepted_indices;
    std::vector<int32_t> rejected_indices{-1};
    std::vector<int32_t> accepted_indices_delta;
    std::vector<int32_t> rejected_indices_delta;

    for (auto top : latest_stack_tops) {
      auto cur_rule_position = tree_[top];
      const auto& catagorized_tokens = catagorized_tokens_for_grammar.at(
          {cur_rule_position.sequence_id, cur_rule_position.element_id});

      stack_tops_with_history_.PushStackTops({tree_.NewNode(cur_rule_position)}, false);

      // std::cout << "top: " << tree_.PrintNode(top) << std::endl;
      // std::cout << "catagorized_tokens sizes: " << catagorized_tokens.accepted_indices.size() <<
      // " "
      //           << catagorized_tokens.rejected_indices.size() << " "
      //           << catagorized_tokens.uncertain_indices.size()
      //           << " Unsaved: " << static_cast<int>(catagorized_tokens.not_saved_index)
      //           << std::endl;

      bool is_find_accept_mode = catagorized_tokens.not_saved_index !=
                                 CatagorizedTokensForGrammar::NotSavedIndex::kAccepted;
      bool is_uncertain_saved = catagorized_tokens.not_saved_index !=
                                CatagorizedTokensForGrammar::NotSavedIndex::kUncertain;

      int idx = -1;
      int idx_unc = -1;
      int idx_acc = 0;
      int idx_rej = 0;
      const std::vector<TCodepoint>* cur_token;
      const std::vector<TCodepoint>* prev_token = nullptr;

      accepted_indices_delta.clear();
      rejected_indices_delta.clear();

      int prev_matched_size = 0;
      while (true) {
        if (is_uncertain_saved) {
          ++idx_unc;
          if (idx_unc >= static_cast<int>(catagorized_tokens.uncertain_indices.size())) {
            break;
          }
          idx = catagorized_tokens.uncertain_indices[idx_unc];
        } else {
          ++idx;
          while (idx < static_cast<int>(sorted_token_and_ids.size())) {
            while (idx_acc < static_cast<int>(catagorized_tokens.accepted_indices.size()) &&
                   catagorized_tokens.accepted_indices[idx_acc] < idx) {
              ++idx_acc;
            }
            if (idx_acc < static_cast<int>(catagorized_tokens.accepted_indices.size()) &&
                catagorized_tokens.accepted_indices[idx_acc] == idx) {
              ++idx_acc;
              ++idx;
              continue;
            }
            while (idx_rej < static_cast<int>(catagorized_tokens.rejected_indices.size()) &&
                   catagorized_tokens.rejected_indices[idx_rej] < idx) {
              ++idx_rej;
            }
            if (idx_rej < static_cast<int>(catagorized_tokens.rejected_indices.size()) &&
                catagorized_tokens.rejected_indices[idx_rej] == idx) {
              ++idx_rej;
              ++idx;
              continue;
            }
            break;
          }
          if (idx >= static_cast<int>(sorted_token_and_ids.size())) {
            break;
          }
        }

        cur_token = &sorted_token_and_ids[idx].token;

        int prev_useful_size = 0;
        if (prev_token) {
          prev_useful_size = std::min(prev_matched_size, static_cast<int>(cur_token->size()));
          for (int j = 0; j < prev_useful_size; ++j) {
            if ((*cur_token)[j] != (*prev_token)[j]) {
              prev_useful_size = j;
              break;
            }
          }
          Rollback(prev_matched_size - prev_useful_size);
        }

        bool accepted = true;
        prev_matched_size = prev_useful_size;
        for (int j = prev_useful_size; j < cur_token->size(); ++j) {
          if (!AcceptChar((*cur_token)[j], false, false)) {
            accepted = false;
            break;
          }
          prev_matched_size = j + 1;
        }

        // Step 4. If the current token is accepted, push its id to the result.
        if (accepted && is_find_accept_mode) {
          accepted_indices_delta.push_back(idx);
        } else if (!accepted && !is_find_accept_mode) {
          rejected_indices_delta.push_back(idx);
        }

        prev_token = cur_token;
      }

      Rollback(prev_matched_size + 1);

      if (is_find_accept_mode) {
        UnionizeWith(&accepted_indices_delta, catagorized_tokens.accepted_indices);
        UnionizeWith(&accepted_indices, accepted_indices_delta);
      } else {
        UnionizeWith(&rejected_indices_delta, catagorized_tokens.rejected_indices);
        IntersectWith(&rejected_indices, rejected_indices_delta);
      }
    }

    rejected_ids->Reset(tokenizer_config->vocab_size);
    // Find all indices that is not in accepted_indices, but in rejected_indices
    int idx_acc = 0;
    int idx_rej = 0;
    if (rejected_indices.size() == 1 && rejected_indices[0] == -1) {
      int cnt = 0;
      for (int i = 0; i < static_cast<int>(sorted_token_and_ids.size()); ++i) {
        while (idx_acc < static_cast<int>(accepted_indices.size()) &&
               accepted_indices[idx_acc] < i) {
          ++idx_acc;
        }
        if (idx_acc < static_cast<int>(accepted_indices.size()) && accepted_indices[idx_acc] == i) {
          ++idx_acc;
          continue;
        }
        ++cnt;
        rejected_ids->SetConst(sorted_token_and_ids[i].id);
      }
    } else {
      DifferenceWith(&rejected_indices, accepted_indices);
      for (int idx : rejected_indices) {
        rejected_ids->SetConst(sorted_token_and_ids[idx].id);
      }
    }
  }

  std::chrono::duration<double, std::milli> handle_past_time;
  std::chrono::duration<double, std::milli> rollback_total_time;
  std::chrono::duration<double, std::milli> accept_total_time;

  CatagorizedTokensForGrammar GetCatagorizedTokens(
      const std::vector<TokenAndId>& sorted_token_and_ids, bool is_root_rule) {
    std::vector<int32_t> accepted_indices;
    std::vector<int32_t> rejected_indices;
    std::vector<int32_t> uncertain_indices;
    std::vector<bool> can_see_end_stack{CanReachEnd()};
    int prev_matched_size = 0;
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
        if (CanReachEnd()) {
          can_see_end = true;
        }
        can_see_end_stack.push_back(can_see_end);
        prev_matched_size = j + 1;
      }
      if (accepted) {
        accepted_indices.push_back(i);
      } else if (can_see_end && !is_root_rule) {
        uncertain_indices.push_back(i);
      } else {
        rejected_indices.push_back(i);
      }
    }
    Rollback(prev_matched_size);
    // std::cout << "Accepted: " << accepted_indices.size();
    // if (accepted_indices.size() < 200) {
    //   std::cout << " (";
    //   for (auto i : accepted_indices) {
    //     std::cout << "<";
    //     for (auto cp : sorted_token_and_ids[i].token) {
    //       std::cout << CodepointToPrintable(cp);
    //     }
    //     std::cout << "> ";
    //   }
    //   std::cout << ")";
    // }
    // std::cout << std::endl;
    // std::cout << "Rejected: " << rejected_indices.size();
    // if (rejected_indices.size() < 200) {
    //   std::cout << " (";
    //   for (auto i : rejected_indices) {
    //     std::cout << "<";
    //     for (auto cp : sorted_token_and_ids[i].token) {
    //       std::cout << CodepointToPrintable(cp);
    //     }
    //     std::cout << "> ";
    //   }
    //   std::cout << ")";
    // }
    // std::cout << std::endl;
    // std::cout << "Uncertain: " << uncertain_indices.size();
    // if (uncertain_indices.size() < 200) {
    //   std::cout << " (";
    //   for (auto i : uncertain_indices) {
    //     std::cout << "<";
    //     for (auto cp : sorted_token_and_ids[i].token) {
    //       std::cout << CodepointToPrintable(cp);
    //     }
    //     std::cout << "> ";
    //   }
    //   std::cout << ")";
    // }
    // std::cout << std::endl;
    return CatagorizedTokensForGrammar(accepted_indices, rejected_indices, uncertain_indices);
  }

  void Rollback(int rollback_steps) { stack_tops_with_history_.Rollback(rollback_steps); }

  std::string PrintStackTops(int steps_behind_latest = 0) const {
    return stack_tops_with_history_.PrintStackTops(steps_behind_latest);
  }

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
        new_stack_tops.push_back(tree_.NewNode(RulePosition{0, i, 0, RulePosition::kNoParent}));
      }
      stack_tops_with_history_.PushStackTops(new_stack_tops);
    } else {
      init_rule_position.parent_id = RulePosition::kNoParent;
      stack_tops_with_history_.PushStackTops({tree_.NewNode(init_rule_position)});
    }
  }

  void UpdateNewStackTops(int32_t old_rule_position_id, std::vector<int32_t>* new_stack_tops) {
    auto cur_rule_position = tree_.GetNextPosition(tree_[old_rule_position_id]);

    while (!tree_.IsEndPosition(cur_rule_position)) {
      auto sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
      auto element = grammar_->GetRuleExpr(sequence[cur_rule_position.element_id]);
      if (element.type == RuleExprType::kCharacterClass ||
          element.type == RuleExprType::kNegCharacterClass) {
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
          DCHECK(grammar_->GetRuleExpr(sequence[0]).type == RuleExprType::kCharacterClass ||
                 grammar_->GetRuleExpr(sequence[0]).type == RuleExprType::kNegCharacterClass);
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
