/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/dead_dynamic_update_slice_elimination.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

absl::StatusOr<int64_t> GetConstantAsInt64(const HloInstruction* inst) {
  if (!inst->IsConstant() || !ShapeUtil::IsScalar(inst->shape())) {
    return absl::InvalidArgumentError("Only support scalar constants.");
  }
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<int64_t>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<int64_t> {
        if constexpr (primitive_type_constant == S32 ||
                      primitive_type_constant == U32) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return static_cast<int64_t>(
              inst->literal().GetFirstElement<NativeT>());
        }
        return absl::InvalidArgumentError("Only support 32-bit integers.");
      },
      inst->shape().element_type());
}

absl::StatusOr<std::vector<int64_t>> GetStartIndices(
    const HloInstruction* dus) {
  absl::Span<HloInstruction* const> start_indices_operands =
      absl::MakeSpan(dus->operands()).subspan(2);
  std::vector<int64_t> start_indices;
  for (HloInstruction* operand : start_indices_operands) {
    TF_ASSIGN_OR_RETURN(int64_t start_index, GetConstantAsInt64(operand));
    start_indices.push_back(start_index);
  }
  return start_indices;
}

// If true, the updated elements of the dynamic-update-slice is not accessed
// by the slice user.
bool IsDusUpdateUnused(const std::vector<int64_t>& dus_starts,
                       const Shape& update_shape,
                       const HloInstruction* slice_user) {
  if (slice_user->opcode() != HloOpcode::kSlice) {
    return false;
  }
  // Get Slice ranges
  const auto& slice_starts = slice_user->slice_starts();
  const auto& slice_limits = slice_user->slice_limits();

  // The slice accesses the updated part IFF there is an overlap in *ALL*
  // dimensions. If there is no overlap in any dimension, the slice is safe,
  // i.e., it doesn't access the updated elements.
  for (int dim = 0; dim < update_shape.dimensions().size(); ++dim) {
    int64_t dus_start = dus_starts[dim];
    int64_t dus_limit = dus_start + update_shape.dimensions(dim);
    int64_t slice_start = slice_starts[dim];
    int64_t slice_limit = slice_limits[dim];
    if (dus_start <= slice_limit - 1 && slice_start <= dus_limit - 1) {
      // Limits are exclusive, so minus one for overlap check.
      continue;
    }
    // Disjoint in this dimension, so slice does not overlap with update.
    return true;
  }
  // Overlap in all dimensions, so slice reads updated values.
  return false;
}

// Helper function to process a single DynamicUpdateSlice instruction.
// Returns true if the module was changed.
absl::StatusOr<bool> ProcessDynamicUpdateSlice(HloInstruction* dus,
                                               HloComputation* comp) {
  const auto start_indices_or = GetStartIndices(dus);
  if (!start_indices_or.ok()) {
    // Not a constant start index, cannot simplify.
    return false;
  }
  const std::vector<int64_t>& dus_starts = start_indices_or.value();
  HloInstruction* update_operand = dus->mutable_operand(1);
  if (dus_starts.size() != update_operand->shape().dimensions().size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "DUS start indices size does not match update operand shape "
        "dimensions size.",
        dus->ToString()));
  }

  bool is_dus_update_unused =
      dus->user_count() > 0 &&
      absl::c_all_of(dus->users(), [&](HloInstruction* user) {
        return IsDusUpdateUnused(dus_starts, update_operand->shape(), user);
      });
  VLOG(2) << "  is_dus_update_unused: " << is_dus_update_unused;
  if (is_dus_update_unused) {
    TF_RETURN_IF_ERROR(dus->ReplaceAllUsesWith(dus->mutable_operand(0)));
    TF_RETURN_IF_ERROR(comp->RemoveInstruction(dus));
    return true;  // Changed
  }
  return false;  // Not changed
}

}  // namespace

absl::StatusOr<bool> DeadDynamicUpdateSliceElimination::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  auto computations_range = module->computations(execution_threads);
  std::vector<HloComputation*> computations(computations_range.begin(),
                                            computations_range.end());
  for (auto* comp : computations) {
    auto post_order_instructions = comp->MakeInstructionPostOrder();
    for (auto it = post_order_instructions.rbegin();
         it != post_order_instructions.rend(); ++it) {
      HloInstruction* instruction = *it;
      if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice) {
        continue;
      }
      VLOG(2) << "Processing DUS: " << instruction->ToString();
      TF_ASSIGN_OR_RETURN(bool dus_changed,
                          ProcessDynamicUpdateSlice(instruction, comp));
      if (dus_changed) {
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
