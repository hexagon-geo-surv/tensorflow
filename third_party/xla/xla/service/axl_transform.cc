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

#include "xla/service/axl_transform.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/platform/logging.h"

namespace xla {

namespace {
absl::flat_hash_map<
    HloAxlTransform::PipelineStage,
    std::vector<std::pair<std::string, std::shared_ptr<HloAxlTransform>>>>&
GetHloAxlTransformsInternal() {
  static auto* out = new absl::flat_hash_map<
      HloAxlTransform::PipelineStage,
      std::vector<std::pair<std::string, std::shared_ptr<HloAxlTransform>>>>;
  return *out;
}

ABSL_CONST_INIT absl::Mutex transforms_mutex(absl::kConstInit);
}  // namespace
absl::Status AxlTransformBase::Transform() { return absl::OkStatus(); }

absl::StatusOr<bool> HloAxlTransform::Transform(xla::HloModule* module) {
  return false;
}

void RegisterHloAxlTransform(HloAxlTransform::PipelineStage stage,
                             absl::string_view name,
                             std::shared_ptr<HloAxlTransform> transform) {
  absl::MutexLock transforms_lock(&transforms_mutex);
  auto& transforms = GetHloAxlTransformsInternal();
  transforms[stage].emplace_back(
      std::make_pair(std::string(name), std::move(transform)));
}

std::vector<std::pair<std::string, std::shared_ptr<HloAxlTransform>>>
GetHloAxlTransforms(HloAxlTransform::PipelineStage stage) {
  absl::MutexLock transforms_lock(&transforms_mutex);
  auto& transforms = GetHloAxlTransformsInternal();
  return transforms[stage];
}

void ClearHloAxlTransforms() {
  absl::MutexLock transforms_lock(&transforms_mutex);
  GetHloAxlTransformsInternal().clear();
}

absl::StatusOr<bool> ApplyAxlTransformsToModule(
    HloAxlTransform::PipelineStage stage, xla::HloModule* module) {
  std::vector<std::pair<std::string, std::shared_ptr<HloAxlTransform>>>
      transforms;
  {
    absl::MutexLock transforms_lock(&transforms_mutex);
    auto& transforms_map = GetHloAxlTransformsInternal();
    auto it = transforms_map.find(stage);
    if (it == transforms_map.end()) {
      return false;
    }
    transforms = it->second;
  }
  bool changed = false;
  for (auto& pair : transforms) {
    auto status_or_bool = pair.second->Transform(module);
    if (!status_or_bool.status().ok()) {
      return status_or_bool.status();
    }
    changed |= status_or_bool.value();
  }
  return changed;
}

absl::StatusOr<bool> ApplyAxlTransforms::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "ApplyAxlTransforms ENTRY";
  XLA_VLOG_LINES(1, module->ToString());
  auto status_or_bool = ApplyAxlTransformsToModule(stage_, module);
  if (!status_or_bool.status().ok()) {
    return status_or_bool.status();
  }
  bool changed = status_or_bool.value();
  if (changed) {
    HloVerifier verifier(/*layout_sensitive=*/false,
                         /*allow_mixed_precision=*/false);
    auto verifier_status = verifier.Run(module);
    if (!verifier_status.status().ok()) {
      return verifier_status.status();
    }
  }
  VLOG(1) << "ApplyAxlTransforms EXIT";
  XLA_VLOG_LINES(1, module->ToString());
  return changed;
}

}  // namespace xla
