/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/grammar/grammar_state_matcher.cc
 */
// #ifndef TVM_LOG_DEBUG
// #define TVM_LOG_DEBUG 1
// #endif
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
    return true;
  }

  void FindNextTokenBitmask(DLTensor* next_token_bitmask) final {
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
    std::vector<bool> unc_tokens_bitset;

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

      if (!is_uncertain_saved) {
        // unc_tokens = all_tokens - accepted_tokens - rejected_tokens
        unc_tokens_bitset.assign(tokens_sorted_by_codepoint.size(), true);
        for (auto idx : catagorized_tokens.accepted_indices) {
          unc_tokens_bitset[idx] = false;
        }
        for (auto idx : catagorized_tokens.rejected_indices) {
          unc_tokens_bitset[idx] = false;
        }
      }

      int iterator_uncertain = -1;

      while (true) {
        // Step 2.1. Find the current token.
        auto idx = GetNextUncertainToken(is_uncertain_saved, &iterator_uncertain,
                                         catagorized_tokens.uncertain_indices, unc_tokens_bitset);
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
    FindNextTokenBitMask(next_token_bitmask, accepted_indices, rejected_indices,
                         tokens_sorted_by_codepoint);
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
  void FindNextTokenBitMask(DLTensor* next_token_bitmask, std::vector<int32_t>& accepted_indices,
                            std::vector<int32_t>& rejected_indices,
                            const std::vector<TokenAndId>& tokens_sorted_by_codepoint) {
    // rejected_ids = Intersect(all_tokens - accepted_indices, rejected_indices)
    DCHECK(next_token_bitmask->dtype.code == kDLUInt && next_token_bitmask->dtype.bits == 32 &&
           next_token_bitmask->data && next_token_bitmask->ndim == 1 && next_token_bitmask->shape);

    BitsetManager next_token_bitset(reinterpret_cast<uint32_t*>(next_token_bitmask->data),
                                    next_token_bitmask->shape[0]);
    next_token_bitset.Reset(init_ctx_->vocab_size, false);

    if (rejected_indices.size() == 1 && rejected_indices[0] == -1) {
      // When rejected_indices = all_tokens, we can just set rejected_ids = all_tokens - accepted
      SetRejectIdsWithComplement(&next_token_bitset, accepted_indices, tokens_sorted_by_codepoint);
    } else {
      // Otherwise, rejected_ids = rejected_indices - accepted_indices
      DifferenceWith(&rejected_indices, accepted_indices);
      for (int idx : rejected_indices) {
        next_token_bitset.Set(tokens_sorted_by_codepoint[idx].id, true);
      }
    }
  }

  // Set the tokens in tokens_sorted_by_codepoint that not in accepted_indices to rejected_ids.
  void SetRejectIdsWithComplement(BitsetManager* rejected_ids,
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
      rejected_ids->Set(tokens_sorted_by_codepoint[i].id, true);
    }
  }

  // If is_uncertain_saved is true, find the next token in uncertain_indices with iterator it_unc.
  // Otherwise, find the next token in unc_tokens with iterative index idx.
  // Return the index of the next token, or -1 if no more token.
  int GetNextUncertainToken(bool is_uncertain_saved, int* iterator_uncertain,
                            const std::vector<int>& uncertain_indices,
                            const std::vector<bool>& uncertain_tokens_bitset) {
    if (is_uncertain_saved) {
      ++*iterator_uncertain;
      if (*iterator_uncertain == uncertain_indices.size()) {
        return -1;
      }
      return uncertain_indices[*iterator_uncertain];
    } else {
      ++*iterator_uncertain;
      while (*iterator_uncertain < uncertain_tokens_bitset.size() &&
             !uncertain_tokens_bitset[*iterator_uncertain]) {
        ++*iterator_uncertain;
      }
      if (*iterator_uncertain == uncertain_tokens_bitset.size()) {
        return -1;
      }
      return *iterator_uncertain;
    }
  }

  friend IntTuple GetRejectedTokenIds(GrammarStateMatcher matcher);

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
    .set_body_typed([](GrammarStateMatcher matcher, int32_t codepoint) {
      auto mutable_node =
          const_cast<GrammarStateMatcherNodeImpl*>(matcher.as<GrammarStateMatcherNodeImpl>());
      return mutable_node->AcceptCodepoint(codepoint);
    });

/*! \brief Check if a matcher can accept the complete string, and then reach the end of the grammar.
 * For test purpose. */
bool MatchCompleteString(GrammarStateMatcher matcher, String str) {
  auto mutable_node =
      const_cast<GrammarStateMatcherNodeImpl*>(matcher.as<GrammarStateMatcherNodeImpl>());
  auto codepoints = Utf8StringToCodepoints(str.c_str());
  int accepted_cnt = 0;
  for (auto codepoint : codepoints) {
    if (!mutable_node->AcceptCodepoint(codepoint, false)) {
      mutable_node->RollbackSteps(accepted_cnt);
      return false;
    }
    ++accepted_cnt;
  }
  return mutable_node->CanReachEnd();
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
  auto vocab_size = matcher.as<GrammarStateMatcherNodeImpl>()->init_ctx_->vocab_size;
  auto bitset_size = BitsetManager::GetBitsetSize(vocab_size);
  auto ndarray = NDArray::Empty(ShapeTuple{static_cast<long>(bitset_size)},
                                DLDataType{kDLUInt, 32, 1}, DLDevice{kDLCPU, 0});
  auto dltensor_manager = ndarray.ToDLPack();
  auto dltensor = ndarray.ToDLPack()->dl_tensor;

  auto start = std::chrono::high_resolution_clock::now();
  matcher->FindNextTokenBitmask(&dltensor);
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "FindRejectedTokens time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us"
            << std::endl;

  auto bitset = BitsetManager(reinterpret_cast<uint32_t*>(dltensor.data), bitset_size);
  std::vector<long> res_vector;
  for (int i = 0; i < vocab_size; i++) {
    if (bitset[i] == 1) {
      res_vector.push_back(i);
    }
  }

  dltensor_manager->deleter(dltensor_manager);

  auto ret = IntTuple(res_vector);
  return ret;
}

TVM_REGISTER_GLOBAL("mlc.serve.GrammarStateMatcherGetRejectedTokenIds")
    .set_body_typed(GetRejectedTokenIds);

}  // namespace serve
}  // namespace llm
}  // namespace mlc
