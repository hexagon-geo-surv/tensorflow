/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/analysis/hlo_reachability.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : words_per_bitset_((instructions.size() + BitSet::kBits - 1) /
                        BitSet::kBits),
      total_words_((instructions.size() + 1 /*for tmp_bit_set_*/) *
                   words_per_bitset_) {
  if (!instructions.empty()) {
    CHECK(instructions[0]->parent() != nullptr)
        << "Instruction must be in a computation.";
    computation_id_ = instructions[0]->parent()->unique_id();
  } else {
    computation_id_ = kComputationIdAbsent;
  }
  uint64_t row = 0;
  uint64_t total_rows = instructions.size() + 1;  // for tmp_bit_set_
  while (row < total_rows) {
    const int rows_to_allocate = std::min(kRowsPerAllocation, total_rows - row);
    size_t words_to_allocate = rows_to_allocate * words_per_bitset_;
    // make_unique initializes the array of words to 0
    bit_storage_.push_back(std::make_unique<BitSet::Word[]>(words_to_allocate));
    row += rows_to_allocate;
  }

  tmp_bit_set_ = BitSetFromIndex(instructions.size());
  int32_t max_local_id = 0;
  for (const HloInstruction* instruction : instructions) {
    max_local_id = std::max(max_local_id, instruction->local_id());
  }
  indices_.resize(max_local_id + 1, kValueAbsent);
  for (size_t i = 0; i < instructions.size(); ++i) {
    BitSetFromIndex(i).Set(i);  // Instructions are reachable from themselves.
    indices_[GetKey(instructions[i])] = i;
  }
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  Index index = GetIndex(instruction);
  BitSet bit_set = BitSetFromIndex(index);
  tmp_bit_set_.CopyBitSet(bit_set);
  SetReachabilityToUnionHelper(inputs, index);
  return bit_set != tmp_bit_set_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  SetReachabilityToUnionHelper(inputs, GetIndex(instruction));
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const Index> input_indices, Index index) {
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const HloInstruction* const> inputs, Index index) {
  absl::InlinedVector<Index, 16> input_indices;
  input_indices.reserve(inputs.size());
  for (const HloInstruction* input : inputs) {
    input_indices.push_back(GetIndex(input));
  }
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const Index> input_indices, Index index) {
  BitSet bit_set = BitSetFromIndex(index);
  // If instruction is part of inputs, don't reset the bit-set.
  if (!absl::c_linear_search(input_indices, index)) {
    bit_set.SetToZero();
  }
  bit_set.Set(index);
  for (Index input_index : input_indices) {
    if (input_index != index) {
      bit_set |= BitSetFromIndex(input_index);
    }
  }
}

void HloReachabilityMap::Replace(const HloInstruction* original,
                                 const HloInstruction* replacement) {
  Key original_key = GetKey(original);
  Key replacement_key = GetKey(replacement);
  if (original_key != replacement_key) {
    DCHECK_LT(original_key, indices_.size());
    if (replacement_key >= indices_.size()) {
      indices_.resize(replacement_key + 1, kValueAbsent);
    }
    indices_[replacement_key] = GetIndex(original);
    indices_[original_key] = kValueAbsent;
  }
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::BuildWithRestrictions(
    const HloComputation* computation,
    absl::FunctionRef<void(const HloInstruction*,
                           std::vector<HloInstruction*>*)>
        add_dependencies) {
  const auto& all = computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(all);

  std::vector<HloInstruction*> inputs;
  for (const HloInstruction* hlo : all) {
    inputs.clear();
    add_dependencies(hlo, &inputs);
    result->FastSetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::Build(
    const HloComputation* computation) {
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(instructions);

  auto get_bit_set = [&](const HloInstruction* instruction) -> BitSet {
    return result->BitSetFromIndex(result->GetIndex(instruction));
  };

  for (const HloInstruction* instruction : instructions) {
    BitSet bit_set = get_bit_set(instruction);

    auto add_dependencies = [&](const HloInstruction* instruction) {
      for (const HloInstruction* operand : instruction->operands()) {
        bit_set |= get_bit_set(operand);
      }
      for (const HloInstruction* predecessor :
           instruction->control_predecessors()) {
        bit_set |= get_bit_set(predecessor);
      }
    };

    add_dependencies(instruction);
  }
  return result;
}

void HloReachabilityMap::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction) {
  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  // Keep track of the number of times an instruction is in the worklist and
  // only process it only if it is the last occurrence. Note that this might
  // still mean that an instruction is processed multiple times.
  absl::flat_hash_map<const HloInstruction*, int64_t> in_worklist;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();
    --in_worklist[item];
    if (in_worklist[item] > 0) {
      continue;
    }

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
        ++in_worklist[user];
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
        ++in_worklist[succ];
      }
    }
  }
}

// Use ptr tagging in `worklist` to check if current instruction is successor of
// left or right.
static constexpr uintptr_t FROM_LEFT_FLAG_MASK = 1;
static constexpr uintptr_t PTR_MASK = ~FROM_LEFT_FLAG_MASK;
static_assert(alignof(HloInstruction) >= 2,
              "HloInstruction must be aligned to at least 2 bytes");
void HloReachabilityMap::UpdateMultipleInstructions(
    const HloInstruction* left, const HloInstruction* right,
    std::vector<uintptr_t>& worklist, std::vector<Index>& indices_to_update) {
  if (left == right) {
    return;
  }
  CHECK(worklist.empty());
  CHECK(indices_to_update.empty());
  DCHECK(IsPresentFast(*left));
  DCHECK(IsPresentFast(*right));
  BitSet left_bit_set = BitSetFromIndex(GetIndex(left));
  BitSet right_bit_set = BitSetFromIndex(GetIndex(right));
  Index left_index = GetIndex(left);
  Index right_index = GetIndex(right);

  absl::flat_hash_set<const HloInstruction*> visited;
  auto add_to_worklist = [&](const HloInstruction* instr,
                             bool from_left) -> void {
    if (visited.insert(instr).second) {
      if (IsPresentFast(*instr)) {
        BitSet bit_set = BitSetFromIndex(GetIndex(instr));
        bool can_skip = false;
        if (from_left) {
          can_skip = bit_set.Get(right_index);
        } else {
          can_skip = bit_set.Get(left_index);
        }
        if (can_skip) {
          return;
        }
      }
      uintptr_t raw_addr = reinterpret_cast<uintptr_t>(instr);
      worklist.push_back(raw_addr | (from_left ? 1 : 0));
    }
  };
  add_to_worklist(left, /*from_left=*/true);
  add_to_worklist(right, /*from_left=*/false);
  if (worklist.empty()) {
    return;
  }
  left_bit_set.GetDifferingWordUnions(right_bit_set, diff_buffer_);
  while (!worklist.empty()) {
    const uintptr_t item_and_from_left = worklist.back();
    worklist.pop_back();

    const bool from_left = (item_and_from_left & FROM_LEFT_FLAG_MASK);
    const HloInstruction* item =
        reinterpret_cast<const HloInstruction*>(item_and_from_left & PTR_MASK);

    if (IsPresentFast(*item)) {
      indices_to_update.push_back(GetIndex(item));
    }
    for (const HloInstruction* succ : item->users()) {
      add_to_worklist(succ, from_left);
    }
    for (const HloInstruction* succ : item->control_successors()) {
      add_to_worklist(succ, from_left);
    }
  }
  for (Index index : indices_to_update) {
    BitSet bit_set = BitSetFromIndex(index);
    bit_set.ApplyWordUnions(diff_buffer_);
  }
  diff_buffer_.clear();
}

void HloReachabilityMap::UpdateMultipleInstructions(
    absl::flat_hash_map<const HloInstruction*,
                        absl::flat_hash_set<const HloInstruction*>>
        to_update) {
  while (!to_update.empty()) {
    auto it = to_update.begin();
    const HloInstruction* instruction = it->first;

    BitSet bit_set = BitSetFromIndex(GetIndex(instruction));
    bool changed = false;
    // NOLINTNEXTLINE the loop aggregation is order independent.
    for (const HloInstruction* operand : it->second) {
      BitSet operand_bit_set = BitSetFromIndex(GetIndex(operand));
      changed |= bit_set.OrUpdate(operand_bit_set);
    }
    to_update.erase(it);
    if (changed) {
      for (const HloInstruction* user : instruction->users()) {
        to_update[user].insert(instruction);
      }
      for (const HloInstruction* succ : instruction->control_successors()) {
        to_update[succ].insert(instruction);
      }
    }
  }
}

}  // namespace xla
