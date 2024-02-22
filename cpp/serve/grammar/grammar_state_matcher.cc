/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_matcher.cc
 */

#include "grammar_state_matcher.h"

#include <chrono>

#include "../../support/set_operation.h"
#include "grammar.h"
#include "grammar_serializer.h"
#include "grammar_tokenizer_config.h"

namespace mlc {
namespace llm {
namespace serve {

/*
 * Note on the matching algorithm
 *
 * Given a context-free grammar, we match the characters in a string one by one.
 *
 * We adopt a non-deterministic pushdown automata (NPDA) in matching. To be specific, we maintain
 * several stacks, each of which represents a possible path in the NPDA, and update the stacks
 * during matching.
 *
 * ## Stack Structure
 * The element of every stack is a RulePosition object, referring to a position in the grammar. In a
 * stack, if a RulePosition refers to a RuleRef element (referring to another rule), the next
 * element in the stack will be a position in this rule. If a RulePosition refers to a
 * CharacterClass element, it will be the last (top) element in the stack, meaning *the next*
 * character to match.
 *
 * ## Matching Process
 * When accepting a new character and it is accepted by a stack, the last element of the stack will
 * be advanced to the next position in the grammar. If it gets to the end of the rule, several
 * elements at the end may be popped out, and the last element of the stack will be advanced.
 *
 * One stack may split since there may be multiple possible next positions. In this case, similar
 * stacks with different top elements will be added. When ome stack cannot accept the new character,
 * it will be removed from the stacks.
 *
 * ## Storage of Stacks
 * Note these stacks form a tree structure as when splitting, the new stacks share the same prefix.
 * We store all RulePositions as a tree, where every path from tree root to a node represents a
 * stack. To represent stack tops, we attach additional pointers pointing the stack top nodes.
 * Also, We maintain a history of the stack top pointers, so we can rollback to the previous state.
 *
 * All tree nodes are maintained by a buffer, and utilize reference counting to recycle. If a node
 * is neither pointed by a stack top pointer, not pointed by some child nodes, it will be freed.
 *
 * ## Example
 * ### Grammar
 * main ::= [a] R
 * R ::= [b] S [c] | [b] [c] T
 * S ::= "" | [c] [d]
 * T ::= [e]
 *
 * ### Previous step
 * Previous accepted string: ab
 * Previous stack tree:
 * A------
 * |  \   \
 * B   D<  E<
 * |
 * C<
 *
 * A: (rule main, choice 0, element 1)
 * B: (rule R, choice 0, element 1)
 * C: (rule S, choice 1, element 0)
 * D: (rule R, choice 0, element 2)
 * E: (rule R, choice 1, element 1)
 * < means the stack top pointers in the previous step.
 * The stacks in the previous step is: (A, B, C), (A, D), (A, E)
 *
 * ### Current step
 * Current accepted string: abc
 * Current stack tree:
 * A-----------------      G<<
 * |     \     \     \
 * B---   D<    E<    H
 * |   \              |
 * C<   F<<           I<<
 *
 * F: (rule S, choice 1, element 1)
 * G: (rule main, choice 0, element 2) (means the matching process has finished, and will be deleted
 * when next char comes)
 * H: (rule R, choice 1, element 2)
 * I: (rule T, choice 0, element 0)
 * << means the stack top pointers in the current step.
 * The stacks in the current step is: (A, B, F), (A, H, I), (G,)
 */

using namespace tvm::runtime;

TVM_REGISTER_OBJECT_TYPE(GrammarStateMatcherNode);

/*! \brief Check the codepoint is contained in the character class. */
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

/*! \brief A RulePosition object with additional structural information. */
struct RulePosition {
  /*! \brief The rule's id. */
  int32_t rule_id = -1;
  /*! \brief Which choice in this rule is selected. */
  int32_t sequence_id = -1;
  /*! \brief Which element of the choice sequence is being visited. */
  int32_t element_id = -1;
  /*!
   * \brief If the element refers to another rule, and another rule is a star quantifier of
   * a character class, this field will be set to the id of the character class.
   * This is part of the special support of star quantifiers of character classes.
   * \sa mlc::llm::serve::BNFGrammarNode.
   */
  int32_t char_class_id = -1;
  /*! \brief The id of the parent node in the RulePositionTree. */
  int32_t parent_id = -1;
  /*! \brief The reference count of this RulePosition. If reduces to zero, the node will be
   * removed from the RulePositionBuffer. */
  int reference_count = 0;

  /*! \brief A parent_id value of kNoParent means this RulePosition is the root of the tree. */
  static constexpr int32_t kNoParent = -1;

  /*! \brief Default constructor. */
  constexpr RulePosition() = default;
  /*! \brief Construct a RulePosition with the given values. */
  constexpr RulePosition(int32_t rule_id, int32_t sequence_id, int32_t element_id,
                         int32_t parent_id = kNoParent, int32_t char_class_id = -1)
      : rule_id(rule_id),
        sequence_id(sequence_id),
        element_id(element_id),
        char_class_id(char_class_id),
        parent_id(parent_id) {}

  bool operator==(const RulePosition& other) const {
    return rule_id == other.rule_id && sequence_id == other.sequence_id &&
           element_id == other.element_id && parent_id == other.parent_id &&
           char_class_id == other.char_class_id;
  }

  bool operator!=(const RulePosition& other) const { return !(*this == other); }
};

/*! \brief A special value for invalid RulePosition. */
inline constexpr RulePosition kInvalidRulePosition(-1, -1, -1, -1, -1);

/*! \brief A buffer to manage all RulePositions. */
class RulePositionBuffer {
 public:
  /*!
   * \brief Allocate a new RulePosition. with given initial value.
   * \returns The id of the allocated node.
   */
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

  /*! \brief Free the RulePosition with the given id. */
  void Free(int32_t id) {
    DCHECK(buffer_[id] != kInvalidRulePosition);
    buffer_[id] = kInvalidRulePosition;
    free_nodes_.push_back(id);
  }

  /*! \brief Get the capacity of the buffer. */
  size_t Capacity() const { return buffer_.size(); }

  /*! \brief Get the number of allocated nodes. */
  size_t Size() const {
    DCHECK(buffer_.size() >= free_nodes_.size());
    return buffer_.size() - free_nodes_.size();
  }

  /*! \brief Get the RulePosition with the given id. */
  RulePosition& operator[](int32_t id) { return buffer_[id]; }
  const RulePosition& operator[](int32_t id) const { return buffer_[id]; }

  friend class RulePositionTree;

 private:
  /*! \brief The buffer to store all RulePositions. */
  std::vector<RulePosition> buffer_;
  /*! \brief A stack to store all free node ids. */
  std::vector<int32_t> free_nodes_;
};

/*!
 * \brief A tree structure to store all stacks. Every stack contains several RulePositions, and
 * is represented as a path from the root to a leaf node.
 */
class RulePositionTree {
 public:
  /*! \brief Construct a RulePositionTree associated with the given grammar. */
  RulePositionTree(const BNFGrammar& grammar) : grammar_(grammar) {}

  /*!
   * \brief Create a new node with the given RulePosition. The reference count of the new node
   * is zero.
   *
   * \note Later, this node should either be pointed by some child rule, or become a stack top
   * node (so it will be pointed to by an attached pointer) to be maintained in the
   * reference-counting based memory management.
   */
  int32_t NewNode(const RulePosition& rule_position) {
    auto id = node_buffer_.Allocate(rule_position);
    if (rule_position.parent_id != RulePosition::kNoParent) {
      DCHECK(rule_position.parent_id < static_cast<int32_t>(node_buffer_.Capacity()) &&
             node_buffer_[rule_position.parent_id] != kInvalidRulePosition);
      node_buffer_[rule_position.parent_id].reference_count++;
    }
    return id;
  }

  /*! \brief See GetNextPosition. */
  bool IsEndPosition(const RulePosition& rule_position) const {
    return rule_position.parent_id == RulePosition::kNoParent &&
           grammar_->GetRuleExpr(rule_position.sequence_id).size() == rule_position.element_id;
  }

  /*!
   * \brief Update a node in the stack to the next position. Next position means either the next
   * element in the current rule, or if the current element is the last element in the rule, the
   * next element in the parent rule. If the current node is the last element in the main rule, it
   * is at the end position.
   */
  RulePosition GetNextPosition(RulePosition rule_position) const {
    if (IsEndPosition(rule_position)) {
      return kInvalidRulePosition;
    }
    rule_position = RulePosition(rule_position.rule_id, rule_position.sequence_id,
                                 rule_position.element_id + 1, rule_position.parent_id);
    while (rule_position.parent_id != RulePosition::kNoParent &&
           grammar_->GetRuleExpr(rule_position.sequence_id).size() == rule_position.element_id) {
      auto parent_rule_position = node_buffer_[rule_position.parent_id];
      rule_position =
          RulePosition(parent_rule_position.rule_id, parent_rule_position.sequence_id,
                       parent_rule_position.element_id + 1, parent_rule_position.parent_id);
    }
    return rule_position;
  }

  /*! \brief Attach an additional reference to the node with the given id. */
  void AttachRefTo(int32_t id) {
    DCHECK(id != RulePosition::kNoParent);
    node_buffer_[id].reference_count++;
  }

  /*! \brief Remove a reference to the node with the given id. If the reference count becomes zero,
   * free the node and recursively all its ancestors with zero reference count. */
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

  /*! \brief Get the RulePosition with the given id. */
  const RulePosition& operator[](int32_t id) const {
    DCHECK(id != RulePosition::kNoParent);
    return node_buffer_[id];
  }

  /*! \brief Print the stack with the given top id to a string. */
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

  /*! \brief Print the node with the given id to a string. */
  std::string PrintNode(int32_t id) const {
    std::stringstream ss;
    const auto& rule_position = node_buffer_[id];
    ss << "id: " << id;
    ss << ", rule " << rule_position.rule_id << ": "
       << grammar_->GetRule(rule_position.rule_id).name;
    ss << ", sequence " << rule_position.sequence_id << ": "
       << BNFGrammarPrinter(grammar_).PrintRuleExpr(rule_position.sequence_id);
    ss << ", element id: " << rule_position.element_id << ", parent id: " << rule_position.parent_id
       << ", ref count: " << rule_position.reference_count;
    return ss.str();
  }

  /*!
   * \brief Check the well-formedness of the tree and the associated buffer.
   * \details This function checks the following properties:
   * 1. Every node is pointed directly or indirectly by a outside pointer.
   * 2. Every node's reference count is consistent with the actual reference count.
   * 3. All ids and positions are valid.
   * 4. If a node in the buffer is free, it should be equal to kInvalidRulePosition.
   */
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
        CHECK(buffer[i] == kInvalidRulePosition);
        CHECK(visited[i] == false);
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
  /*! \brief The grammar associated with this RulePositionTree. */
  BNFGrammar grammar_;
  /*! \brief The buffer to store all RulePositions. */
  RulePositionBuffer node_buffer_;
};

/*!
 * \brief A class to maintain the stack tops and its history to support rollback.
 * \details This class helps to maintain nodes by automatically maintaining the attached references.
 * If a node is not existing in any stack in the history record, it will be freed.
 *
 * It can store up to the previous max_rollback_steps + 1 steps of history, and thus supports
 * rolling back up to max_rollback_steps steps.
 */
class StackTopsWithHistory {
 public:
  /*!
   * \param tree The RulePositionTree to be associated with. Possibly modify the tree by attaching
   * and removing references to the stack top nodes.
   * \param max_rollback_steps The maximum number of rollback steps to be supported.
   */
  StackTopsWithHistory(RulePositionTree* tree)
      : tree_(tree) {}

  /*!
   * \brief Push a new history record consisting a list of stack tops. These nodes will be recorded
   * as existing in a stack (by attaching a reference to them).
   * \param stack_tops The stack tops to be pushed.
   * \param drop_old Whether to drop the oldest history record if the history size exceeds the
   * limit. If the history is dropped, node that do not exist in any stack any more will be freed.
   */
  void PushHistory(const std::vector<int32_t>& stack_tops, bool drop_old = true) {
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

  /*!
   * \brief Roll back to several previous steps. Possibly frees node that do not exist in any stack
   * any more.
   * \param rollback_steps The number of steps to be rolled back.
   */
  void Rollback(int rollback_steps) {
    DCHECK(rollback_steps < stack_tops_history_.size())
        << "The number of requested rollback steps is greater than or equal to the current "
           "history "
        << "size: " << rollback_steps << " vs " << stack_tops_history_.size() << ".";
    while (rollback_steps--) {
      PopBack();
    }
  }

  /*!
   * \brief Get the latest stack tops.
   * \returns The latest stack tops.
   */
  const std::vector<int32_t>& LatestHistory() const { return stack_tops_history_.back(); }

  /*!
   * \brief Print one history record.
   * \param steps_behind_latest The number of steps behind the latest record. 0 means the latest
   * record.
   */
  std::string PrintHistory(int steps_behind_latest = 0) const {
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

  /*! \brief Get the number of history records. */
  int Size() const { return stack_tops_history_.size(); }

  /*! \brief Check the well-formedness of the tree and the associated buffer. */
  void CheckWellFormed() const {
    std::vector<int32_t> outside_pointers;
    for (const auto& stack_tops : stack_tops_history_) {
      outside_pointers.insert(outside_pointers.end(), stack_tops.begin(), stack_tops.end());
    }
    tree_->CheckWellFormed(outside_pointers);
  }

 private:
  /*! \brief Pop the oldest history record. Possibly frees node that do not exist in any stack any
   * more. */
  void PopFront() {
    const auto& old_stack_tops = stack_tops_history_.front();
    for (auto id : old_stack_tops) {
      tree_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_front();
  }

  /*! \brief Pop the latest history record. Possibly frees node that do not exist in any stack any
   * more. */
  void PopBack() {
    const auto& new_stack_tops = stack_tops_history_.back();
    for (auto id : new_stack_tops) {
      tree_->RemoveRefTo(id);
    }
    stack_tops_history_.pop_back();
  }

  /*! \brief Modifiable pointer to the RulePositionTree. */
  RulePositionTree* tree_;
  /*! \brief The maximum number of rollback steps to be supported. */
  int max_rollback_steps_;
  /*! \brief The history of stack tops. */
  std::deque<std::vector<int32_t>> stack_tops_history_;
};

/*! \brief A token and its id. */
struct TokenAndId {
  std::vector<TCodepoint> token;
  int32_t id;
  /*! \brief Compare tokens by their unicode codepoint sequence. */
  bool operator<(const TokenAndId& other) const;
};

/*!
 * \brief Preprocessed information, for a given specific rule and position, divides the token set
 * into three categories:
 * - accepted_indices: If a token is accepted by this rule
 * - rejected_indices: If a token is rejected by this rule
 * - uncertain_indices: If a prefix of a token is accepted by this rule and comes to the end
 * of the rule. In real matching process, it still can be accepted or rejected by some parent rule.
 * So it is uncertain.
 * \note Since the union of these three sets is the whole token set, we only need to store the
 * smaller two sets. The unsaved set is specified by not_saved_index.
 * \note These indices are the indices of sorted_token_and_ids in the TokenInfoForMatcher object,
 * instead of the token ids. That helps the matching process.
 */
struct CatagorizedTokens {
  std::vector<int32_t> accepted_indices;
  std::vector<int32_t> rejected_indices;
  std::vector<int32_t> uncertain_indices;
  enum class NotSavedIndex { kAccepted = 0, kRejected = 1, kUncertain = 2 };
  NotSavedIndex not_saved_index;

  CatagorizedTokens() = default;

  CatagorizedTokens(std::vector<int32_t>&& accepted_indices,
                    std::vector<int32_t>&& rejected_indices,
                    std::vector<int32_t>&& uncertain_indices) {
    auto size_acc = accepted_indices.size();
    auto size_rej = rejected_indices.size();
    auto size_unc = uncertain_indices.size();
    not_saved_index =
        (size_acc >= size_rej && size_acc >= size_unc)
            ? NotSavedIndex::kAccepted
            : (size_rej >= size_unc ? NotSavedIndex::kRejected : NotSavedIndex::kUncertain);

    if (not_saved_index != NotSavedIndex::kAccepted) {
      this->accepted_indices = std::move(accepted_indices);
    }
    if (not_saved_index != NotSavedIndex::kRejected) {
      this->rejected_indices = std::move(rejected_indices);
    }
    if (not_saved_index != NotSavedIndex::kUncertain) {
      this->uncertain_indices = std::move(uncertain_indices);
    }
  }
};

/*!
 * \brief All information that we need to match tokens in the tokenizer to the specified grammar.
 * It is the result of preprocessing.
 * \sa mlc::llm::serve::GrammarStateMatcher
 * \sa mlc::llm::serve::Sampler
 */
class TokenInfoForMatcher::TokenInfoForMatcherData {
 public:
  /*! \brief The vocabulary size of the tokenizer. */
  size_t vocab_size;
  /*! \brief The sorted token and its id. Tokens are sorted to reuse the common prefix during
   * matching. */
  std::vector<TokenAndId> sorted_token_and_ids;
  /*! \brief The stop tokens. They can be accepted iff GramamrMatcher can reach the end of the
   * grammar. */
  std::vector<int32_t> stop_token_ids;
  /*! \brief The special tokens. Currently we will ignore these tokens during grammar-guided
   * matching. */
  std::vector<int32_t> special_token_ids;
  /*! \brief Mapping from token id to token. */
  std::unordered_map<int32_t, TokenAndId> token_lookup_map;

  /*! \brief A sequence id and its position. */
  struct SequenceIdAndPosition {
    int32_t sequence_id;
    int32_t element_id;
    bool operator==(const SequenceIdAndPosition& other) const {
      return sequence_id == other.sequence_id && element_id == other.element_id;
    }
  };

  /*! \brief Hash function for SequenceIdAndPosition. */
  struct SequenceIdAndPositionHash {
    std::size_t operator()(const SequenceIdAndPosition& k) const {
      return std::hash<int32_t>()(k.sequence_id) ^ (std::hash<int32_t>()(k.element_id) << 1);
    }
  };

  /*! \brief Mapping from sequence id and its position to the catagorized tokens. */
  std::unordered_map<SequenceIdAndPosition, CatagorizedTokens, SequenceIdAndPositionHash>
      catagorized_tokens_for_grammar;

  /*! \brief Tokenizer will replace space with a special underscore. We will replace it back to
   * space. */
  static constexpr const char* kSpecialUnderscore = "▁";
};

/* \brief The concrete implementation of GrammarStateMatcherNode. */
class GrammarStateMatcherNodeImpl : public GrammarStateMatcherNode {
 private:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

 public:
  GrammarStateMatcherNodeImpl(const BNFGrammar& grammar, int max_rollback_steps = 0,
                              RulePosition init_rule_position = {})
      : grammar_(grammar), tree_(grammar), stack_tops_with_history_(&tree_, max_rollback_steps) {
    InitStackState(init_rule_position);
  }

  bool AcceptCodepoint(TCodepoint codepoint, bool drop_old = true, bool verbose = false) {
    if (verbose) {
      std::cout << "Stack before accepting: " << PrintStackState() << std::endl;
    }
    static std::vector<int32_t> new_stack_tops;
    new_stack_tops.clear();

    const auto& prev_stack_tops = stack_tops_with_history_.LatestHistory();
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
      // Special support for star quantifiers of character classes.
      if (current_char_class.type == RuleExprType::kRuleRef) {
        DCHECK(rule_position.char_class_id != -1);
        current_char_class = grammar_->GetRuleExpr(rule_position.char_class_id);
      }
      DCHECK(current_char_class.type == RuleExprType::kCharacterClass ||
             current_char_class.type == RuleExprType::kNegCharacterClass);
      auto ok = CharacterClassContains(current_char_class, codepoint);
      if (!ok) {
        continue;
      }
      UpdateNewStackTops(old_top, &new_stack_tops);
    }
    if (new_stack_tops.empty()) {
      if (verbose) {
        std::cout << "Codepoint: " << codepoint << " \"" << CodepointToPrintable(codepoint)
                  << "\" Rejected" << std::endl;
      }
      return false;
    }
    stack_tops_with_history_.PushHistory(new_stack_tops, drop_old);
    if (verbose) {
      std::cout << "Codepoint: " << codepoint << " \"" << CodepointToPrintable(codepoint)
                << "\" Accepted" << std::endl;
      std::cout << "Stack after accepting: " << PrintStackState() << std::endl;
    }
    return true;
  }

  bool CanReachEnd() const {
    auto last_stack_tops = stack_tops_with_history_.LatestHistory();
    return std::any_of(last_stack_tops.begin(), last_stack_tops.end(),
                       [&](int32_t id) { return tree_.IsEndPosition(tree_[id]); });
  }

  void FindRejectedTokens(const TokenInfoForMatcher& tokenizer_config,
                          DynamicBitSet* rejected_ids) {
    const auto& sorted_token_and_ids = tokenizer_config->sorted_token_and_ids;
    const auto& catagorized_tokens_for_grammar = tokenizer_config->catagorized_tokens_for_grammar;
    const auto& latest_stack_tops = stack_tops_with_history_.LatestHistory();

    // For every stack, we will either find all tokens it accepts, or all tokens it rejects.
    // The final rejected tokens should be not accepted by any stack, and also rejected by every
    // stack.

    // Per stack temporary data.
    // Note here indices store the indices in sorted_token_and_ids, instead of the token ids.
    std::vector<int32_t> accepted_indices;
    // {-1} means the universal set, i.e. all tokens
    std::vector<int32_t> rejected_indices{-1};
    std::vector<int32_t> accepted_indices_delta;
    std::vector<int32_t> rejected_indices_delta;
    DynamicBitSet unc_tokens;

    for (auto top : latest_stack_tops) {
      // Step 1. Find the current catagorized_tokens
      auto cur_rule_position = tree_[top];
      auto current_sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
      if (cur_rule_position.parent_id == RulePosition::kNoParent &&
          cur_rule_position.element_id == current_sequence.size()) {
        continue;
      }
      const auto& catagorized_tokens = catagorized_tokens_for_grammar.at(
          {cur_rule_position.sequence_id, cur_rule_position.element_id});

      // For each stack, we will check every uncertain token and put them into the accepted or
      // rejected list.
      // For effeciency, we will only find the accepted tokens or the rejected tokens.
      // If the accepted tokens are saved, it means it is likely to be smaller than the rejected
      // tokens, so we will just find the accepted tokens, and vice versa.
      bool is_find_accept_mode =
          catagorized_tokens.not_saved_index != CatagorizedTokens::NotSavedIndex::kAccepted;

      // If uncertain tokens are saved, we will iterate over the uncertain tokens.
      // Otherwise, we will iterate over all_tokens - accepted_tokens - rejected_tokens.
      bool is_uncertain_saved =
          catagorized_tokens.not_saved_index != CatagorizedTokens::NotSavedIndex::kUncertain;

      // Step 2. Update the accepted tokens in accepted_indices_delta, or the rejected tokens in
      // rejected_indices_delta.

      // Examine only the current one stack
      stack_tops_with_history_.PushHistory({tree_.NewNode(cur_rule_position)}, false);

      const std::vector<TCodepoint>* prev_token = nullptr;
      int prev_matched_size = 0;

      accepted_indices_delta.clear();
      rejected_indices_delta.clear();

      int idx = -1;
      auto it_unc = catagorized_tokens.uncertain_indices.begin() - 1;

      if (!is_uncertain_saved) {
        // unc_tokens = all_tokens - accepted_tokens - rejected_tokens
        unc_tokens.Reset<true>(sorted_token_and_ids.size());
        for (auto idx : catagorized_tokens.accepted_indices) {
          unc_tokens.SetConst<false>(idx);
        }
        for (auto idx : catagorized_tokens.rejected_indices) {
          unc_tokens.SetConst<false>(idx);
        }
      }

      while (true) {
        // Step 2.1. Find the current token.
        idx = GetNextUncertainToken(is_uncertain_saved, &it_unc,
                                    catagorized_tokens.uncertain_indices, idx, unc_tokens);
        if (idx == -1) {
          break;
        }
        const auto& cur_token = sorted_token_and_ids[idx].token;

        // Step 2.2. Find the longest common prefix with the accepted part of the previous token.
        // We can reuse the previous matched size to avoid unnecessary matching.
        int prev_useful_size = 0;
        if (prev_token) {
          prev_useful_size = std::min(prev_matched_size, static_cast<int>(cur_token.size()));
          for (int j = 0; j < prev_useful_size; ++j) {
            if (cur_token[j] != (*prev_token)[j]) {
              prev_useful_size = j;
              break;
            }
          }
          Rollback(prev_matched_size - prev_useful_size);
        }

        // Step 2.3. Find if the current token is accepted or rejected.
        bool accepted = true;
        prev_matched_size = prev_useful_size;

        for (int j = prev_useful_size; j < cur_token.size(); ++j) {
          if (!AcceptCodepoint(cur_token[j], false, false)) {
            accepted = false;
            break;
          }
          prev_matched_size = j + 1;
        }

        // Step 2.4. Push the result to the delta list.
        if (accepted && is_find_accept_mode) {
          accepted_indices_delta.push_back(idx);
        } else if (!accepted && !is_find_accept_mode) {
          rejected_indices_delta.push_back(idx);
        }

        prev_token = &cur_token;
      }

      Rollback(prev_matched_size + 1);

      // Step 3. Update the accepted_indices and rejected_indices
      if (is_find_accept_mode) {
        // accepted_indices += catagorized_tokens.accepted_indices + accepted_indices_delta
        UnionizeWith(&accepted_indices_delta, catagorized_tokens.accepted_indices);
        UnionizeWith(&accepted_indices, accepted_indices_delta);
      } else {
        // rejected_indices = Intersect(
        //     rejected_indices,
        //     catagorized_tokens.rejected_indices + rejected_indices_delta)
        UnionizeWith(&rejected_indices_delta, catagorized_tokens.rejected_indices);
        IntersectWith(&rejected_indices, rejected_indices_delta);
      }
    }

    // Finally update the rejected_ids bitset
    // rejected_ids = Intersect(all_tokens - accepted_indices, rejected_indices)
    rejected_ids->Reset(tokenizer_config->vocab_size);
    int idx_acc = 0;
    if (rejected_indices.size() == 1 && rejected_indices[0] == -1) {
      // When rejected_indices = all_tokens, we can just set rejected_ids = all_tokens - accepted
      SetRejectIdsWithComplement(rejected_ids, accepted_indices, sorted_token_and_ids);
    } else {
      // Otherwise, rejected_ids = rejected_indices - accepted_indices
      DifferenceWith(&rejected_indices, accepted_indices);
      for (int idx : rejected_indices) {
        rejected_ids->SetConst(sorted_token_and_ids[idx].id);
      }
    }
  }

  CatagorizedTokens GetCatagorizedTokens(const std::vector<TokenAndId>& sorted_token_and_ids,
                                         bool is_main_rule) {
    // Support the current stack contains only one stack with one RulePosition.
    // Iterate over all tokens. Split them into three categories:
    // - accepted_indices: If a token is accepted by current rule
    // - rejected_indices: If a token is rejected by current rule
    // - uncertain_indices: If a prefix of a token is accepted by current rule and comes to the end
    // of the rule. In real matching process, it still can be accepted or rejected by a parent rule.
    // So it is uncertain.

    // Note many tokens may contain the same prefix, so we will avoid unnecessary matching
    static std::vector<int32_t> accepted_indices;
    static std::vector<int32_t> rejected_indices;
    static std::vector<int32_t> uncertain_indices;

    // For every character in the current token, stores whether it is possible to reach the end of
    // the rule when matching until this character. Useful for rollback.
    static std::vector<bool> can_see_end_stack;

    accepted_indices.clear();
    rejected_indices.clear();
    uncertain_indices.clear();
    can_see_end_stack.clear();
    can_see_end_stack.push_back(CanReachEnd());

    int prev_matched_size = 0;
    for (int i = 0; i < static_cast<int>(sorted_token_and_ids.size()); ++i) {
      const auto& token = sorted_token_and_ids[i].token;
      const auto* prev_token = i > 0 ? &sorted_token_and_ids[i - 1].token : nullptr;

      // Find the longest common prefix with the accepted part of the previous token.
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

      // Find if the current token is accepted or rejected or uncertain.
      bool accepted = true;
      bool can_see_end = can_see_end_stack.back();
      prev_matched_size = prev_useful_size;
      for (int j = prev_useful_size; j < token.size(); ++j) {
        if (!AcceptCodepoint(token[j], false, false)) {
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
      } else if (can_see_end && !is_main_rule) {
        // If the current rule is the main rule, there will be no uncertain indices since we will
        // never consider its parent rule. Unaccepted tokens are just rejected.
        uncertain_indices.push_back(i);
      } else {
        rejected_indices.push_back(i);
      }
    }
    Rollback(prev_matched_size);
    return CatagorizedTokens(std::move(accepted_indices), std::move(rejected_indices),
                             std::move(uncertain_indices));
  }

  void Rollback(int rollback_steps) { stack_tops_with_history_.Rollback(rollback_steps); }

  std::string PrintStackState(int steps_behind_latest = 0) const {
    return stack_tops_with_history_.PrintHistory(steps_behind_latest);
  }

 private:
  // Init the stack state according to the given rule position.
  // If init_rule_position is {-1, -1, -1, -1}, init the stack with the main rule.
  void InitStackState(RulePosition init_rule_position) {
    if (init_rule_position == kInvalidRulePosition) {
      // Initialize the stack with the main rule.
      auto main_rule = grammar_->GetRule(0);
      auto main_rule_expr = grammar_->GetRuleExpr(main_rule.body_expr_id);
      std::vector<int32_t> new_stack_tops;
      for (auto i : main_rule_expr) {
        DCHECK(grammar_->GetRuleExpr(i).type == RuleExprType::kSequence ||
               grammar_->GetRuleExpr(i).type == RuleExprType::kEmptyStr);
        new_stack_tops.push_back(tree_.NewNode(RulePosition(0, i, 0, RulePosition::kNoParent)));
      }
      stack_tops_with_history_.PushHistory(new_stack_tops);
    } else {
      stack_tops_with_history_.PushHistory({tree_.NewNode(init_rule_position)});
    }
  }

  // Update the old stack top to the next position, and push the new stack tops to new_stack_tops.
  void UpdateNewStackTops(int32_t old_node_id, std::vector<int32_t>* new_stack_tops) {
    auto old_rule_position = tree_[old_node_id];
    // For char_class*, the old rule position itself is also the next position
    if (old_rule_position.char_class_id != -1) {
      new_stack_tops->push_back(tree_.NewNode(old_rule_position));
    }

    auto cur_rule_position = tree_.GetNextPosition(tree_[old_node_id]);

    // Continuously iterate to the next position (if reachs the end of the current rule, go to the
    // next position of the parent rule). Push it into new_stack_tops. If this position can not
    // be empty, exit the loop.
    // Positions that can be empty: reference to a rule that can be empty, or a star quantifier
    // rule.
    for (; !tree_.IsEndPosition(cur_rule_position);
         cur_rule_position = tree_.GetNextPosition(cur_rule_position)) {
      auto sequence = grammar_->GetRuleExpr(cur_rule_position.sequence_id);
      auto element = grammar_->GetRuleExpr(sequence[cur_rule_position.element_id]);
      if (element.type == RuleExprType::kCharacterClass ||
          element.type == RuleExprType::kNegCharacterClass) {
        // Character class: cannot be empty. Break the loop.
        new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
        break;
      } else {
        // RuleRef
        DCHECK(element.type == RuleExprType::kRuleRef);
        auto new_rule_id = element[0];
        auto new_rule = grammar_->GetRule(new_rule_id);
        auto new_rule_expr = grammar_->GetRuleExpr(new_rule.body_expr_id);
        if (new_rule_expr.type == RuleExprType::kStarQuantifier) {
          cur_rule_position.char_class_id = new_rule_expr[0];
          new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
        } else {
          DCHECK(new_rule_expr.type == RuleExprType::kChoices);

          bool contain_empty = false;

          // For rule containing choices, expand the rule and push all positions into new_stack_tops
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
            new_stack_tops->push_back(tree_.NewNode(RulePosition(new_rule_id, j, 0, parent_id)));
          }

          if (!contain_empty) {
            break;
          }
        }
      }
    }

    // Reaches the end of the main rule. Insert a special node to indicate the end.
    if (tree_.IsEndPosition(cur_rule_position)) {
      new_stack_tops->push_back(tree_.NewNode(cur_rule_position));
    }
  }

  // Set the tokens in sorted_token_and_ids that not in accepted_indices to rejected_ids.
  void SetRejectIdsWithComplement(DynamicBitSet* rejected_ids,
                                  const std::vector<int32_t>& accepted_indices,
                                  const std::vector<TokenAndId>& sorted_token_and_ids) {
    auto it_acc = accepted_indices.begin();
    for (int i = 0; i < static_cast<int>(sorted_token_and_ids.size()); ++i) {
      while (it_acc != accepted_indices.end() && *it_acc < i) {
        ++it_acc;
      }
      if (it_acc != accepted_indices.end() && *it_acc == i) {
        ++it_acc;
        continue;
      }
      rejected_ids->SetConst(sorted_token_and_ids[i].id);
    }
  }

  // If is_uncertain_saved is true, find the next token in uncertain_indices with iterator it_unc.
  // Otherwise, find the next token in unc_tokens with iterative index idx.
  // Return the index of the next token, or -1 if no more token.
  int GetNextUncertainToken(bool is_uncertain_saved, std::vector<int>::const_iterator* it_unc,
                            const std::vector<int>& uncertain_indices, int idx,
                            const DynamicBitSet& unc_tokens) {
    if (is_uncertain_saved) {
      ++*it_unc;
      if (*it_unc == uncertain_indices.end()) {
        return -1;
      }
      return **it_unc;
    } else {
      ++idx;
      while (idx < unc_tokens.Size() && !unc_tokens[idx]) {
        ++idx;
      }
      if (idx == unc_tokens.Size()) {
        return -1;
      }
      return idx;
    }
  }

  BNFGrammar grammar_;
  RulePositionTree tree_;
  StackTopsWithHistory stack_tops_with_history_;
};

/*!
 * \brief Find the rejected tokens among all tokens in the tokenizer for the specified
 * GrammarStateMatcher. For test purpose.
 * \param matcher The GrammarStateMatcher to be used.
 * \param grammar The grammar associated to the matcher.
 * \param tokenizer The specified tokenizer.
 * \returns A tuple of rejected token ids.
 */
IntTuple GetRejectedTokenIdsForTokenizer(GrammarStateMatcher matcher, BNFGrammar grammar,
                                         Tokenizer tokenizer) {
  // Cache the processed TokenInfoForMatcher
  static BNFGrammar cached_grammar = grammar;
  static Tokenizer cached_tokenizer = tokenizer;
  static TokenInfoForMatcher tokenizer_config = TokenInfoForMatcher(tokenizer, grammar);
  if (cached_tokenizer != tokenizer || cached_grammar != grammar) {
    tokenizer_config = TokenInfoForMatcher(tokenizer, cached_grammar);
    cached_tokenizer = tokenizer;
    cached_grammar = grammar;
  }

  DynamicBitSet bitset;
  matcher->FindRejectedTokens(tokenizer_config, &bitset);

  std::vector<long> res_vector;
  for (int i = 0; i < bitset.Size(); i++) {
    if (bitset[i]) {
      res_vector.push_back(i);
    }
  }

  auto ret = IntTuple(res_vector);
  return ret;
}

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

TokenInfoForMatcher::TokenInfoForMatcher(const BNFGrammar& grammar,
                                         const std::vector<std::string>& token_table) {
  using RuleExprType = BNFGrammarNode::RuleExprType;

  data_ = std::make_shared<TokenInfoForMatcherData>();

  data_->vocab_size = token_table.size();
  for (int i = 0; i < token_table.size(); ++i) {
    auto token = token_table[i];
    if (token == "<unk>" || token == "<pad>" || token == "<s>") {
      data_->special_token_ids.push_back(i);
    } else if (token == "</s>") {
      data_->stop_token_ids.push_back(i);
    } else if (token[0] == '<' && token[token.size() - 1] == '>') {
      // Currently we consider all <...> tokens as special tokens.
      data_->special_token_ids.push_back(i);
    } else {
      // First replace the special underscore with space.
      auto token_underscore_replaced = ReplaceUnderscoreWithSpace(token, data_->kSpecialUnderscore);
      auto codepoints = Utf8StringToCodepoints(token_underscore_replaced.c_str());
      DCHECK(!codepoints.empty() &&
             codepoints[0] != static_cast<TCodepoint>(CharHandlingError::kInvalidUtf8))
          << "Invalid token: " << token;
      data_->sorted_token_and_ids.push_back({codepoints, i});
      data_->token_lookup_map[i] = {codepoints, i};
    }
  }
  std::sort(data_->sorted_token_and_ids.begin(), data_->sorted_token_and_ids.end());

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

        auto grammar_matcher = GrammarStateMatcherNodeImpl(grammar, 0, cur_rule_position);
        auto cur_catagorized_tokens_for_grammar =
            grammar_matcher.GetCatagorizedTokens(data_->sorted_token_and_ids, i == 0);
        data_->catagorized_tokens_for_grammar[{sequence_id, element_id}] =
            cur_catagorized_tokens_for_grammar;
      }
    }
  }
}

GrammarStateMatcher::GrammarStateMatcher(const BNFGrammar& grammar, int max_rollback_steps,
                                         RulePosition init)
    : ObjectRef(make_object<GrammarStateMatcherNodeImpl>(grammar, max_rollback_steps, init)) {}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcher")
    .set_body_typed([](BNFGrammar grammar, int max_rollback_steps) {
      return GrammarStateMatcher(grammar, max_rollback_steps);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherAcceptCodepoint")
    .set_body_typed([](GrammarStateMatcher matcher, int32_t codepoint, bool drop_old) {
      return matcher->AcceptCodepoint(codepoint, drop_old);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherCanReachEnd")
    .set_body_typed([](GrammarStateMatcher matcher) { return matcher->CanReachEnd(); });

/*! \brief Check if a matcher can accept the complete string, and then reach the end of the grammar.
 * For test purpose. */
bool MatchCompleteString(GrammarStateMatcher matcher, String str) {
  auto codepoints = Utf8StringToCodepoints(str.c_str());
  int accepted_cnt = 0;
  for (auto codepoint : codepoints) {
    if (!matcher->AcceptCodepoint(codepoint, false, false)) {
      matcher->Rollback(accepted_cnt);
      return false;
    }
    ++accepted_cnt;
  }
  return matcher->CanReachEnd();
}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherMatchCompleteString")
    .set_body_typed([](GrammarStateMatcher matcher, String str) {
      return MatchCompleteString(matcher, str);
    });

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherGetRejectedTokenIdsForTokenizer")
    .set_body_typed(GetRejectedTokenIdsForTokenizer);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
