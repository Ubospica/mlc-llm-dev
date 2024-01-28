/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.h
 * \brief The header for the support of grammar-guided generation.
 */

#ifndef MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_
#define MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <vector>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

/*!
 * \brief This class stores the abstract syntax tree (AST) of the Backus-Naur Form (BNF) grammar.
 * The BNF definition here is standard BNF, and the characters are represented using regex-style
 * character ranges (e.g. [a-z], [^a-z]).
 *
 * \details The BNF grammar consists of a set of rules. Each rule is represented by a name and a
 * definition, and corresponds to a production rule. Each rule has a rule_id for reference.
 *
 * The definition of a rule is a RuleExpr. RuleExpr can be the definition of a rule or part of the
 * definition of a rule.
 *
 * For example, in the following rule: rule ::= ("a" "b") | "c"
 * ("a" "b"), "c", ("a" "b") | "c" are all RuleExprs.
 *
 * Every RuleExpr is represented by a type as well as a variable-length array containing its data.
 * There are several types for RuleExpr:
 * - Character range: a range of characters (each character is a unicode codepoint), e.g. [a-z],
 *   [ac-z].
 *   A char is represented by a character range with the same lower and upper bound. A string is
 *   represented by a sequence of character ranges.
 * - Negative character range: all characters that are not in the range, e.g. [^a-z], [^ac-z]
 * - EmptyStr: an empty string, i.e. ""
 * - Rule reference: a reference to another rule
 * - Sequence: a sequence of rule_exprs, e.g. ("a" "b"). These rule_exprs are concatenated together.
 * - Choices: a choice of rule_exprs, e.g. ("a" "b") | "c". Each rule_expr can be matched.
 *
 * For the internal representation of the data, see docs in BNFGrammarNode::RuleExprType. Each
 * RuleExpr corresponds to an rule_expr_id for reference.
 *
 * We store all RuleExprs in csr_matrix style. That is, they are stored consecutively in one vector
 * (data vector) and the starting position of each RuleExpr is recorded in the indptr vector.
 */
class BNFGrammarNode : public Object {
 public:
  /*! \brief A rule with name. */
  struct Rule {
    /*! \brief The name of the rule. */
    std::string name;
    /*! \brief The RuleExpr id of the definition of the rule. */
    int32_t rule_expr_id;
  };

  /*! \brief Get the number of rules. */
  size_t NumRules() const { return rules_.size(); }
  /*! \brief Get the rule with the given id. */
  const Rule& GetRule(int32_t rule_id) const {
    CHECK(rule_id >= 0 && rule_id < static_cast<int32_t>(rules_.size()))
        << "rule_id " << rule_id << " is out of bound";
    return rules_[rule_id];
  }

  /*! \brief The type of the rule expr. */
  enum class RuleExprType : int32_t {
    // data format: [lower0, upper0, lower1, upper1, ...]
    // to represent a single character, just add the same lower and upper bound.
    kCharacterRange,
    // data format: [lower0, upper0, lower1, upper1, ...]
    kNegCharacterRange,
    // data format: []
    kEmptyStr,
    // data format: [rule_id]
    kRuleRef,
    // data format: [rule_expr_id0, rule_expr_id1, ...]
    kSequence,
    // data format: [rule_expr_id0, rule_expr_id1, ...]
    kChoices,
  };

  /*! \brief The object representing a rule expr. */
  struct RuleExpr {
    /*! \brief The type of the rule expr. */
    RuleExprType type;
    /*! \brief The data of the RuleExpr. A variable-length array. */
    const int32_t* data;
    /*! \brief The length of the data array. */
    size_t data_len;

    const size_t size() const { return data_len; }
    /*! \brief Get the i-th element of the data array. */
    const int32_t& operator[](int i) const { return data[i]; }
    const int32_t* begin() const { return data; }
    const int32_t* end() const { return data + data_len; }
  };

  /*! \brief Get the number of rule_exprs. */
  size_t NumRuleExprs() const { return rule_expr_indptr_.size(); }
  /*! \brief Get the rule_expr with the given id. */
  RuleExpr GetRuleExpr(int32_t rule_expr_id) const {
    CHECK(rule_expr_id >= 0 && rule_expr_id < static_cast<int32_t>(rule_expr_indptr_.size()))
        << "rule_expr_id " << rule_expr_id << " is out of bound";
    int start_index = rule_expr_indptr_[rule_expr_id];
    RuleExprType type = static_cast<RuleExprType>(rule_expr_data_[start_index]);
    ++start_index;
    int end_index;
    if (rule_expr_id == static_cast<int32_t>(rule_expr_indptr_.size()) - 1) {
      end_index = rule_expr_data_.size();
    } else {
      end_index = rule_expr_indptr_[rule_expr_id + 1];
    }
    ICHECK_GE(end_index, start_index);
    return {type, rule_expr_data_.data() + start_index,
            static_cast<size_t>(end_index - start_index)};
  }

  /*! \brief Whether the grammar can generate empty string. */
  bool CanBeEmpty() const { return can_be_empty_; }

  static constexpr const char* _type_key = "mlc.serve.BNFGrammar";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(BNFGrammarNode, Object);

 private:
  /*! \brief The rules of the grammar. rule_id corresponds the index of this vector. */
  std::vector<Rule> rules_;
  /*! \brief The data of all rule_exprs. */
  std::vector<int32_t> rule_expr_data_;
  /*! \brief The start index of every rule_expr in rule_expr_data_. rule_expr_id corresponds the
   * index of this vector. */
  std::vector<int32_t> rule_expr_indptr_;
  bool can_be_empty_;

  friend class BNFGrammarBuilder;
  friend class BNFGrammarJSONSerializer;
  friend class BNFJSONParser;
};

class BNFGrammar : public ObjectRef {
 public:
  static BNFGrammar FromEBNFString(const String& ebnf_string, bool unwrap_nesting_rules = true,
                                   bool simplify = true);
  static BNFGrammar FromJSON(const String& json_string);
  static BNFGrammar GetJSONGrammar();

  friend std::ostream& operator<<(std::ostream& os, const BNFGrammar& grammar);

  TVM_DEFINE_OBJECT_REF_METHODS(BNFGrammar, ObjectRef, BNFGrammarNode);
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SERVE_GRAMMAR_GRAMMAR_H_
