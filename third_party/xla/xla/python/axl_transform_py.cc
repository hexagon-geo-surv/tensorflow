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

#include "nanobind/nanobind.h"
#include "nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/service/axl_transform.h"

namespace xla {

namespace {

class PyCApiAxlTransform : public HloAxlTransform {
 public:
  explicit PyCApiAxlTransform(nanobind::object transform_fn)
      : transform_fn_(std::move(transform_fn)) {}
  ~PyCApiAxlTransform() override = default;

  absl::StatusOr<bool> Transform(xla::HloModule* module) override {
    nanobind::gil_scoped_acquire gil;
    if (!transform_fn_) {
      return absl::InternalError("Python transform function not set.");
    }
    try {
      auto py_result = transform_fn_(module);
      return nanobind::cast<absl::StatusOr<bool>>(py_result);
    } catch (const nanobind::python_error& e) {
      return absl::InternalError(e.what());
    } catch (const std::exception& e) {
      return absl::InternalError(e.what());
    }
  }

 private:
  nanobind::object transform_fn_;
};

void RegisterHloAxlTransformWrapper(HloAxlTransform::PipelineStage stage,
                                    std::string name,
                                    nanobind::object transform_fn) {
  auto transform =
      std::make_shared<PyCApiAxlTransform>(std::move(transform_fn));
  RegisterHloAxlTransform(stage, name, transform);
}
}  // namespace

NB_MODULE(axl_transform, m) {
  nanobind::enum_<HloAxlTransform::PipelineStage>(m, "PipelineStage")
      .value("kPreScheduler", HloAxlTransform::PipelineStage::kPreScheduler)
      .value("kPostScheduler", HloAxlTransform::PipelineStage::kPostScheduler);

  nanobind::class_<HloAxlTransform>(m, "HloAxlTransform");

  m.def("RegisterHloAxlTransform", &RegisterHloAxlTransformWrapper);
}

}  // namespace xla
