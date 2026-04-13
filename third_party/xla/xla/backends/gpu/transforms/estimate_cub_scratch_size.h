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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_ESTIMATE_CUB_SCRATCH_SIZE_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_ESTIMATE_CUB_SCRATCH_SIZE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla::gpu {

// Updates the scratch size of CUB sort and scan custom calls to match the
// actual scratch size.
//
// For sort, it either invokes the FFI instantiate handler
// to compute the scratch size (requires a device) or, in deviceless mode,
// looks up the scratch size from a bundled lookup table.
//
// It also changes the custom call target to the FFI handler name
// (xla.gpu.ext.cub_sort_keys or xla.gpu.ext.cub_sort_pairs).
class EstimateCubScratchSize : public HloModulePass {
 public:
  struct DevicelessOptions {
    std::string device_name;
    stream_executor::SemanticVersion cub_version;
  };

  // If `deviceless_options` is provided, the pass will runs in deviceless mode.
  explicit EstimateCubScratchSize(
      std::string platform_name,
      std::optional<DevicelessOptions> deviceless_options = std::nullopt)
      : platform_name_(std::move(platform_name)),
        deviceless_options_(std::move(deviceless_options)) {}

  absl::string_view name() const override {
    return "estimate-cub-scratch-size";
  }

 protected:
  absl::Status RunOnSortInstruction(HloCustomCallInstruction* custom_call);
  absl::Status RunOnScanInstruction(HloCustomCallInstruction* custom_call);
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<int64_t> CalculateDevicelessScratchSize(
      HloCustomCallInstruction* custom_call, const Shape& key_shape,
      bool is_pairs, int64_t num_items, int64_t batch_size);

  absl::StatusOr<int64_t> CalculateScratchSpaceWithDevice(
      HloCustomCallInstruction* custom_call, absl::string_view ffi_target,
      const Shape& key_shape, bool is_pairs, int64_t num_items,
      int64_t batch_size);

  std::string platform_name_;
  std::optional<DevicelessOptions> deviceless_options_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_ESTIMATE_CUB_SCRATCH_SIZE_H_
