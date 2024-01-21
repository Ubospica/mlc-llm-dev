/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <vector>

#include "grammar.h"

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

template <typename T>
class TypedElementBuffer {
 public:
  int32_t Allocate() {
    if (free_nodes_.empty()) {
      buffer_.emplace_back();
      return buffer_.size() - 1;
    } else {
      int32_t id = free_nodes_.back();
      free_nodes_.pop_back();
      return id;
    }
  }

  void Free(int32_t id) { free_nodes_.push_back(id); }

  T& operator[](int32_t id) { return buffer_[id]; }

 private:
  std::vector<T> buffer_;
  std::vector<int32_t> free_nodes_;
};

/*!
 * \brief Store the state in the past `max_rollback_steps` steps. The state is a list of stacks,
 * representing all possible paths on the pushdown automata.
 * Every stack is a list of RuleWithPosition. They organize in a tree structure.
 * \details This history is managed as a circular buffer.
 */
class RuleStackTree {
 public:
  void Push(const std::vector<int32_t>& stack_tops);
  void Pop();
  std::vector<int32_t>& GetTop() const;
  int GetSize() const;

  struct RuleWithPosition {
    /*! \brief The rule's id.*/
    int32_t rule_id;
    /*! \brief Which choice in this rule is selected */
    int choice_id;
    /*! \brief Which element of the choice sequence is being visited */
    int element_id;
    /*! \brief Refers to the parent element */
    int32_t parent_id;
    /*!
     * \brief How many elements this element is referenced by as a parent node or the stack top.
     * If reference count is zero, we can recycle this element.
     */
    int reference_count;
  };

 private:
  TypedElementBuffer<RuleWithPosition> buffer_;
  std::vector<std::vector<int32_t>> stack_tops_;
};

class GrammarStateNode : public Object {
 public:
  virtual void GetRefusedTokens(Tokenizer tokenizer, std::vector<int>* refused_tokens_buffer) = 0;
  virtual void AcceptToken(int token_id) = 0;
  virtual void Rollback(int rollback_cnt) = 0;

  const BNFGrammar& grammar;
  const int max_rollback_steps;

 public:
  GrammarStateNode(const BNFGrammar& grammar, int max_rollback_steps)
      : grammar_(grammar), max_rollback_steps_(max_rollback_steps) {
    auto start_element_id = buffer_.Allocate();
    auto& start_element = buffer_[start_element_id];
  }

 private:
  RuleStackTree stack_tops_history_;

 public:
  static constexpr const char* _type_key = "mlc.serve.GrammarState";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(GrammarStateNode, Object);

 protected:
  GrammarStateNode(const BNFGrammar& grammar, int max_rollback_steps)
      : grammar(grammar), max_rollback_steps(max_rollback_steps) {}
};

class GrammarState : public ObjectRef {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(GrammarState, ObjectRef, GrammarStateNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_STATE_H_
