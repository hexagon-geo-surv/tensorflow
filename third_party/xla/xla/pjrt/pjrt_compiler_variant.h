/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_COMPILER_VARIANT_H_
#define XLA_PJRT_PJRT_COMPILER_VARIANT_H_

#include <string>

#include "absl/strings/string_view.h"

namespace xla {

// Represents the PjRt compiler variant to use.
class PjRtCompilerVariant {
 public:
  virtual ~PjRtCompilerVariant() = default;

  // Returns the name of the variant.
  virtual std::string Name() const = 0;

  // Returns true if the variant is the target variant.
  virtual bool IsTargetVariant(absl::string_view target_variant) const = 0;
};

// XLA linked in as a library.
inline constexpr absl::string_view kLinkedVariant = "linked";

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMPILER_VARIANT_H_
