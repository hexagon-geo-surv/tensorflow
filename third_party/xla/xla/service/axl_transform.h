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

#ifndef XLA_SERVICE_AXL_TRANSFORM_H_
#define XLA_SERVICE_AXL_TRANSFORM_H_

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// An AxlTransform is a user-defined transformation, which is applied at a
// user-specified stage of the XLA compilation pipeline.
// Note: AXL is a permutation of XLA.

class AxlTransformBase {
 public:
  AxlTransformBase() = default;
  virtual ~AxlTransformBase() = default;

  virtual absl::Status Transform();
};

// HloAxlTransform is an AxlTransform that operates on an HLO module. Currently,
// it can only be applied at two stages of the XLA compilation pipeline:
// pre-scheduler and post-scheduler. We expect to expand the stages in the
// future.
class HloAxlTransform : public AxlTransformBase {
 public:
  enum class PipelineStage {
    kPreScheduler = 0,
    kPostScheduler,
  };

  HloAxlTransform() = default;
  using AxlTransformBase::Transform;

  virtual absl::StatusOr<bool> Transform(xla::HloModule* module);
};

// Registers a user's HloAxlTransform implementation to be applied at the
// specified stage (and associated with the given name).
void RegisterHloAxlTransform(HloAxlTransform::PipelineStage stage,
                             absl::string_view name,
                             std::shared_ptr<HloAxlTransform> transform);

// Returns the list of registered HloAxlTransforms for the given stage.
std::vector<std::pair<std::string, std::shared_ptr<HloAxlTransform>>>
GetHloAxlTransforms(HloAxlTransform::PipelineStage stage);

// Clears all registered HloAxlTransforms.
void ClearHloAxlTransforms();

// Applies all registered HloAxlTransforms for the specified stage to the
// given module. Returns an error if any transform fails, otherwise returns true
// if any transform made a change to the module.
absl::StatusOr<bool> ApplyAxlTransformsToModule(
    HloAxlTransform::PipelineStage stage, xla::HloModule* module);

// HloPass that applies all registered HloAxlTransforms for the specified stage.
// HloAxlTransforms which are registered at the same stage, are applied in the
// order in which they were registered.
class ApplyAxlTransforms : public HloModulePass {
 public:
  explicit ApplyAxlTransforms(HloAxlTransform::PipelineStage stage)
      : stage_(stage) {}
  ~ApplyAxlTransforms() override = default;

  absl::string_view name() const override { return "apply-axl-transforms"; }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloAxlTransform::PipelineStage stage_;
};

}  // namespace xla

#endif  // XLA_SERVICE_AXL_TRANSFORM_H_
