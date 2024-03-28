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

#include "xla/python/tools/statusor_caster.h"

#include <Python.h>  // IWYU pragma: keep

#include <cassert>
#include <stdexcept>

#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/nb_defs.h"
// NOTE: The "third_party/pybind11_abseil/status_casters.h" header says
// it's deprecated and that we should import the other headers directly.
#include "pybind11_abseil/import_status_module.h"  // from @pybind11_abseil

namespace nanobind {
namespace google {

nanobind::module_ ImportStatusModule() {
  // TODO(wrengr): Replace with `nb::gil_scoped_acquire` from "nb_misc.h"?
  if (!PyGILState_Check()) {
    // Inlined variant of `pybind11_fail` since nanobind has no analogue.
    assert(!PyErr_Occurred());
    throw std::runtime_error(
        "ImportStatusModule() PyGILState_Check() failure.");
  }
  return nanobind::module_::import_(
      NB_TOSTRING(PYBIND11_ABSEIL_STATUS_MODULE_PATH));
}

}  // namespace google
}  // namespace nanobind
