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

#ifndef XLA_PYTHON_TOOLS_STATUSOR_CASTER_H_
#define XLA_PYTHON_TOOLS_STATUSOR_CASTER_H_

#include "absl/status/statusor.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
// NOTE: The "third_party/pybind11_abseil/status_casters.h" header says
// it's deprecated and that we should import the other headers directly.
#include "pybind11_abseil/status_not_ok_exception.h"  // from @pybind11_abseil

namespace nanobind {
namespace google {

/// Nanobind translation of `pybind11::google::ImportStatusModule()`.
/// Imports the module for Python's version of `absl::Status`.  This
/// function is meant to only be called from an `NB_MODULE` definition;
/// and the Python GIL must be held when calling this function.
///
/// NOTE: Unlike the pybind11 variant, this does not allow implicit casting.
/// The `nanobind::detail::type_caster` class disallows throwing exceptions,
/// so there's no way to define the desired caster.  Instead, client code
/// should explicitly call `ValueOrThrow` to perform the conversion.
nanobind::module_ ImportStatusModule();

/// Explicitly convert non-ok `absl::StatusOr` to Python exceptions.
template <typename T>
inline T ValueOrThrow(absl::StatusOr<T> v) {
  // NOTE: To have the exact same behavior and safety/errors as
  // `pybind11::detail::type_caster<absl::Status>::cast_impl` does,
  // we should call `pybind11::google::internal::CheckStatusModuleImported()`.
  // But alas, the build target which provides that function is marked
  // as having private visibility.
  if (!v.ok()) {
    // As per `pybind11::detail::type_caster<absl::Status>::cast_impl`.
    throw pybind11::google::StatusNotOk(v.status());
    // TODO(wrengr): Do we need to register this exception type with nanobind?
    // Or will nanobind be satisfied by the status library's call to
    // `pybind11::google::internal::RegisterStatusBindings`?  Everything
    // works for our current case of defining a pybind11 extension; but
    // it's unclear if it'll continue to work if/when we switch over to
    // defining a nanobind extension.
  }
  return std::move(v).value();
}

}  // namespace google
}  // namespace nanobind

#endif  // XLA_PYTHON_TOOLS_STATUSOR_CASTER_H_
