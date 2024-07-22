/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_original_value.h"

#include <optional>

#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {

std::string OriginalValueToString(const OriginalValue& original_value) {
  std::vector<std::string> original_value_str;
  for (const auto& leaf : original_value.leaves()) {
    if (leaf.second) {
      ShapeIndex leaf_shape_index = leaf.first;
      std::string instruction_name = leaf.second->instruction_name;
      ShapeIndex shape_index = leaf.second->shape_index;
      original_value_str.emplace_back(
          "{" + leaf_shape_index.ToString() + ", instruction_name=\"" +
          instruction_name + "\", shape_index=" + shape_index.ToString() + "}");
    }
  }
  return "shape=" + original_value.shape().ToString() + " leaves={" +
         absl::StrJoin(original_value_str, ", ") + "}";
}

}  // namespace xla
