/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher.cc
 */
#include "grammar_state_matcher.h"

#include <chrono>
#include <queue>

#include "../../support/set_operation.h"
#include "../../tokenizers.h"
#include "grammar.h"
#include "grammar_serializer.h"
#include "grammar_state_matcher_base.h"
#include "grammar_state_matcher_preproc.h"
#include "grammar_state_matcher_state.h"

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
 * The element of every stack is a RulePosition object, referring a position in the grammar. If a
 * RulePosition is a RuleRef element (referring to another rule), the next element of the stack will
 * be a position in this rule. If a RulePosition is a CharacterClass element, it will be the last
 * in the stack, meaning *the next* character to match.
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

/* \brief The concrete implementation of GrammarStateMatcherNode. */
class GrammarStateMatcherNodeImpl : public GrammarStateMatcherNode, public GrammarStateMatcherBase {
 private:
  using RuleExpr = BNFGrammarNode::RuleExpr;
  using RuleExprType = BNFGrammarNode::RuleExprType;

 public:
  GrammarStateMatcherNodeImpl(std::shared_ptr<GrammarStateInitContext> init_ctx,
                              int max_rollback_tokens = 0)
      : GrammarStateMatcherBase(init_ctx->grammar),
        init_ctx_(init_ctx),
        max_history_size_(max_rollback_tokens + 1) {}

  bool AcceptToken(int32_t token_id) final {
    CHECK(init_ctx_->codepoint_tokens_lookup.count(token_id) > 0);
    const auto& token = init_ctx_->codepoint_tokens_lookup[token_id].token;
    for (auto codepoint : token) {
      if (!AcceptCodepoint(codepoint, false)) {
        return false;
      }
    }
    token_size_history_.push_back(token.size());
    if (token_size_history_.size() > max_history_size_) {
      DiscardEarliestSteps(token_size_history_.front());
      token_size_history_.pop_front();
    }
  }

  void FindNextTokenBitmask(DLTensor* next_token_bitmask) {
    const auto& tokens_sorted_by_codepoint = init_ctx_->tokens_sorted_by_codepoint;
    const auto& catagorized_tokens_for_grammar = init_ctx_->catagorized_tokens_for_grammar;
    const auto& latest_stack_tops = stack_tops_history_.GetLatest();

    // For every stack, we will either find all tokens it accepts, or all tokens it rejects.
    // The final rejected tokens should be not accepted by any stack, and also rejected by every
    // stack.

    // Per stack temporary data.
    // Note here indices store the indices in tokens_sorted_by_codepoint, instead of the token ids.
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
      stack_tops_history_.PushHistory({tree_.NewNode(cur_rule_position)});

      const std::vector<TCodepoint>* prev_token = nullptr;
      int prev_matched_size = 0;

      accepted_indices_delta.clear();
      rejected_indices_delta.clear();

      int idx = -1;
      auto it_unc = catagorized_tokens.uncertain_indices.begin() - 1;

      if (!is_uncertain_saved) {
        // unc_tokens = all_tokens - accepted_tokens - rejected_tokens
        unc_tokens.Reset(tokens_sorted_by_codepoint.size(), true);
        for (auto idx : catagorized_tokens.accepted_indices) {
          unc_tokens.Set(idx, false);
        }
        for (auto idx : catagorized_tokens.rejected_indices) {
          unc_tokens.Set(idx, false);
        }
      }

      while (true) {
        // Step 2.1. Find the current token.
        idx = GetNextUncertainToken(is_uncertain_saved, &it_unc,
                                    catagorized_tokens.uncertain_indices, idx, unc_tokens);
        if (idx == -1) {
          break;
        }
        const auto& cur_token = tokens_sorted_by_codepoint[idx].token;

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
          RollbackSteps(prev_matched_size - prev_useful_size);
        }

        // Step 2.3. Find if the current token is accepted or rejected.
        bool accepted = true;
        prev_matched_size = prev_useful_size;

        for (int j = prev_useful_size; j < cur_token.size(); ++j) {
          if (!AcceptCodepoint(cur_token[j], false)) {
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

      RollbackSteps(prev_matched_size + 1);

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
    rejected_ids->Reset(init_ctx_->vocab_size);
    int idx_acc = 0;
    if (rejected_indices.size() == 1 && rejected_indices[0] == -1) {
      // When rejected_indices = all_tokens, we can just set rejected_ids = all_tokens - accepted
      SetRejectIdsWithComplement(rejected_ids, accepted_indices, tokens_sorted_by_codepoint);
    } else {
      // Otherwise, rejected_ids = rejected_indices - accepted_indices
      DifferenceWith(&rejected_indices, accepted_indices);
      for (int idx : rejected_indices) {
        rejected_ids->Set(tokens_sorted_by_codepoint[idx].id);
      }
    }
  }

  void Rollback(int num_tokens) {
    CHECK(num_tokens < token_size_history_.size());
    while (num_tokens > 0) {
      int steps = token_size_history_.back();
      RollbackSteps(steps);
      token_size_history_.pop_back();
      --num_tokens;
    }
  }

 private:
  // Set the tokens in tokens_sorted_by_codepoint that not in accepted_indices to rejected_ids.
  void SetRejectIdsWithComplement(DynamicBitSet* rejected_ids,
                                  const std::vector<int32_t>& accepted_indices,
                                  const std::vector<TokenAndId>& tokens_sorted_by_codepoint) {
    auto it_acc = accepted_indices.begin();
    for (int i = 0; i < static_cast<int>(tokens_sorted_by_codepoint.size()); ++i) {
      while (it_acc != accepted_indices.end() && *it_acc < i) {
        ++it_acc;
      }
      if (it_acc != accepted_indices.end() && *it_acc == i) {
        ++it_acc;
        continue;
      }
      rejected_ids->Set(tokens_sorted_by_codepoint[i].id);
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

  std::shared_ptr<GrammarStateInitContext> init_ctx_;
  int max_history_size_;
  std::deque<int> token_size_history_;
};

GrammarStateMatcher::GrammarStateMatcher(std::shared_ptr<GrammarStateInitContext> init_ctx,
                                         int max_rollback_steps)
    : ObjectRef(make_object<GrammarStateMatcherNodeImpl>(init_ctx, max_rollback_steps)) {}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcher")
    .set_body_typed([](BNFGrammar grammar, Optional<Tokenizer> tokenizer, int max_rollback_steps) {
      auto init_ctx = CreateInitContext(
          grammar, tokenizer ? tokenizer.value()->token_table : std::vector<std::string>());
      return GrammarStateMatcher(init_ctx, max_rollback_steps);
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
      matcher->RollbackSteps(accepted_cnt);
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

/*!
 * \brief Find the rejected tokens among all tokens in the tokenizer for the specified
 * GrammarStateMatcher. For test purpose.
 * \param matcher The GrammarStateMatcher to be used.
 * \param grammar The grammar associated to the matcher.
 * \param tokenizer The specified tokenizer.
 * \returns A tuple of rejected token ids.
 */
IntTuple GetRejectedTokenIds(GrammarStateMatcher matcher) {
  DynamicBitSet bitset;
  auto start = std::chrono::high_resolution_clock::now();
  matcher->FindRejectedTokens(&bitset);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "FindRejectedTokens time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us"
            << std::endl;

  std::vector<long> res_vector;
  for (int i = 0; i < bitset.Size(); i++) {
    if (bitset[i]) {
      res_vector.push_back(i);
    }
  }

  auto ret = IntTuple(res_vector);
  return ret;
}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherGetRejectedTokenIds")
    .set_body_typed(GetRejectedTokenIds);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
