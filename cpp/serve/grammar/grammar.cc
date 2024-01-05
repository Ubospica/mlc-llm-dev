/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar.cc
 */

#include "grammar.h"

#include <tvm/runtime/registry.h>

#include "../../metadata/json_parser.h"
#include "grammar_impl.h"
#include "grammar_parser.h"

namespace mlc {
namespace llm {
namespace serve {

TVM_REGISTER_OBJECT_TYPE(BNFGrammarNode);

BNFGrammar BNFGrammar::FromEBNFString(String ebnf_string) {
  auto node = make_object<BNFGrammarImpl>();
  auto parser = EBNFParser(node.get());
  parser.Parse(ebnf_string);
  // node->Simplify();
  // enable only when debugging?
  // ICHECK(node->WellFormed()) << "The parsed BNF AST is not well-formed.";
  return BNFGrammar(std::move(node));
}

BNFGrammar BNFGrammar::FromJSON(String json) {
  auto node = make_object<BNFGrammarImpl>();
  auto grammar_json = json::ParseToJsonObject(json);
  auto rules_json = json::Lookup<picojson::array>(grammar_json, "rules");
  for (const auto& rule_json : rules_json) {
    auto rule_json_obj = rule_json.get<picojson::object>();
    auto name = json::Lookup<std::string>(rule_json.get<picojson::object>(), "name");
    auto subrule =
        static_cast<int32_t>(json::Lookup<int64_t>(rule_json.get<picojson::object>(), "subrule"));
    node->rules_.push_back(BNFGrammarImpl::Rule({name, subrule}));
  }
  auto subrule_storage_json = json::Lookup<picojson::object>(grammar_json, "subrule_storage_json");
  auto subrule_storage_data_json = json::Lookup<picojson::array>(subrule_storage_json, "data");
  for (const auto& data_json : subrule_storage_data_json) {
    node->subrule_storage_.data_.push_back(static_cast<int32_t>(data_json.get<int64_t>()));
  }
  auto subrule_storage_start_index_json =
      json::Lookup<picojson::array>(subrule_storage_json, "start_index");
  for (const auto& start_index_json : subrule_storage_start_index_json) {
    node->subrule_storage_.start_index_.push_back(
        static_cast<int32_t>(start_index_json.get<int64_t>()));
  }
  return BNFGrammar(std::move(node));
}

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromEBNFString").set_body_typed([](String ebnf_string) {
  return BNFGrammar::FromEBNFString(ebnf_string);
});

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarFromJSON").set_body_typed([](String json) {
  return BNFGrammar::FromJSON(json);
});

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarAsString").set_body_typed([](const BNFGrammar& grammar) {
  return grammar->AsString();
});

TVM_REGISTER_GLOBAL("mlc.serve.BNFGrammarAsJSON")
    .set_body_typed([](const BNFGrammar& grammar, bool prettify = true) {
      return grammar->AsJSON(prettify);
    });

TVM_REGISTER_OBJECT_TYPE(GrammarStateNode);

class GrammarStateImpl : public GrammarStateNode {
 public:
  GrammarStateImpl(const BNFGrammar& grammar, int max_rollback_steps)
      : GrammarStateNode(grammar, max_rollback_steps) {
    auto start_element_id = buffer_.Allocate();
    auto& start_element = buffer_[start_element_id];
  }

 private:
  using TStackElementId = int32_t;

  struct RuleWithPosition {
    /*! \brief The rule's id.*/
    BNFGrammarImpl::TRuleId rule_id;
    /*! \brief Which choice in this rule is selected */
    int choice_id;
    /*! \brief Which element of the choice sequence is being visited */
    int element_id;
    /*! \brief Refers to the parent element */
    TStackElementId parent_id;
    /*!
     * \brief How many elements this element is referenced by as a parent node or the stack top.
     * If reference count is zero, we can recycle this element.
     */
    int reference_count;
  };

  class StackElementBuffer {
   public:
    TStackElementId Allocate() {
      if (free_nodes_.empty()) {
        buffer_.emplace_back();
        return buffer_.size() - 1;
      } else {
        TStackElementId id = free_nodes_.back();
        free_nodes_.pop_back();
        return id;
      }
    }

    void Free(TStackElementId id) { free_nodes_.push_back(id); }

    RuleWithPosition& operator[](TStackElementId id) { return buffer_[id]; }

   private:
    std::vector<RuleWithPosition> buffer_;
    std::vector<TStackElementId> free_nodes_;
  };

  /*!
   * \brief Store the state in the past `max_rollback_steps` steps. The state is a list of stacks,
   * representing all possible paths on the pushdown automata.
   * Every stack is a list of RuleWithPosition. They organize in a tree structure.
   * \details This history is managed as a circular buffer.
   */
  class StackTopsHistory {
   public:
    void Push(const std::vector<TStackElementId>& stack_tops);
    void Pop();
    std::vector<TStackElementId>& GetTop() const;
    int GetSize() const;

   private:
    std::vector<std::vector<TStackElementId>> stack_tops_;
  };

  StackElementBuffer buffer_;
  StackTopsHistory stack_tops_history_;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc
