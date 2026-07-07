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

#ifndef XLA_PJRT_TPU_TPU_COMPILER_VARIANT_H_
#define XLA_PJRT_TPU_TPU_COMPILER_VARIANT_H_

#include <memory>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_compiler_variant.h"

namespace xla {

// Concrete implementation of PjRtCompilerVariant for TPU.
class TpuCompilerVariant : public PjRtCompilerVariant {
 public:
  explicit TpuCompilerVariant(absl::string_view variant_name)
      : variant_name_(variant_name) {}

  std::string Name() const override { return variant_name_; }

  bool IsTargetVariant(absl::string_view target_variant) const override {
    return variant_name_ == target_variant;
  }

 private:
  std::string variant_name_;
};

// Returns the TPU compiler variant that should be used.
absl::StatusOr<std::unique_ptr<PjRtCompilerVariant>> PickTpuCompilerVariant();

}  // namespace xla

#endif  // XLA_PJRT_TPU_TPU_COMPILER_VARIANT_H_
