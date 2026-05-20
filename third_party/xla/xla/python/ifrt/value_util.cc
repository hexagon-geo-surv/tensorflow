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

#include "xla/python/ifrt/value_util.h"

#include <cstddef>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/tuple.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

std::vector<ArrayRef> ToArrays(absl::Span<const ValueRef> values) {
  std::vector<ArrayRef> arrays;
  arrays.reserve(values.size());
  for (const ValueRef& value : values) {
    arrays.push_back(tsl::FormRef(llvm::cast<Array>(value.get())));
  }
  return arrays;
}

std::vector<ArrayRef> ToArrays(absl::Span<ValueRef> values) {
  std::vector<ArrayRef> arrays;
  arrays.reserve(values.size());
  for (ValueRef& value : values) {
    arrays.push_back(tsl::TakeRef(llvm::cast<Array>(value.release())));
  }
  return arrays;
}

std::vector<ValueRef> ToValues(absl::Span<const ArrayRef> arrays) {
  std::vector<ValueRef> values;
  values.reserve(arrays.size());
  for (const ArrayRef& array : arrays) {
    values.push_back(array);
  }
  return values;
}

std::vector<ValueRef> ToValues(absl::Span<ArrayRef> arrays) {
  std::vector<ValueRef> values;
  values.reserve(arrays.size());
  for (ArrayRef& array : arrays) {
    values.push_back(std::move(array));
  }
  return values;
}

}  // namespace ifrt
}  // namespace xla
