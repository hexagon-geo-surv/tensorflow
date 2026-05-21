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

#ifndef XLA_PYTHON_IFRT_BUNDLE_H_
#define XLA_PYTHON_IFRT_BUNDLE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/layout.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/remap_plan.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

class Client;

class Bundle;
using BundleRef = tsl::RCReference<Bundle>;

// A runtime-managed data structure that represents an ordered list of
// `ValueRef`s.
class Bundle : public tsl::ReferenceCounted<Bundle>,
               public llvm::RTTIExtends<Bundle, llvm::RTTIRoot> {
 public:
  Bundle() = default;

  //// `Value`-like methods.

  virtual Client* client() const = 0;

  // Returns the user context associated with the creation of this array.
  virtual UserContextRef user_context() const = 0;

  // Mode of byte size to calculate.
  enum class ByteSizeMode {
    // Maximum of the sum of byte sizes of arrays placed on the same device and
    // memory kind.
    kMaxPerDeviceAndMemoryKind,
    // Maximum of the sum of byte sizes of arrays placed on the same logical
    // task and memory kind.
    kMaxPerLogicalTaskAndMemoryKind,
    // Total sum of byte sizes of all arrays in the bundle.
    kTotal,
  };

  // Returns the byte size of the bundle. If any value has an unknown size,
  // returns `std::nullopt`.
  virtual absl::StatusOr<std::optional<int64_t>> ByteSize(
      ByteSizeMode mode) const = 0;

  // Returns a future that becomes ready once all values are ready.
  virtual tsl::Future<> GetReadyFuture() const = 0;

  // Deletes all values in this `Bundle`.
  virtual tsl::Future<> Delete() = 0;

  // Returns true if this `Bundle` has been deleted.
  virtual bool IsDeleted() const = 0;

  // Returns a string representation of this `Bundle`.
  virtual std::string DebugString() const = 0;

  //// Bundle-specific methods.

  // Returns the number of values. This is an inexpensive operation.
  virtual int num_values() const = 0;

  // Expands this `Bundle` into `ValueRef`s.
  virtual absl::StatusOr<std::vector<ValueRef>> GetValues(
      ArrayCopySemantics semantics) = 0;

  // Slices a `Bundle` into `Bundle`s. Each output `Bundle` will contain
  // contiguous values of the specified size from the input `Bundle`.
  virtual absl::StatusOr<std::vector<BundleRef>> Slice(
      absl::Span<const int> slice_sizes, ArrayCopySemantics semantics) = 0;

  // Specification for copying a slice of the bundle.
  struct CopySpec {
    // New devices and memory kind for this slice.
    std::optional<DeviceListRef> devices;
    std::optional<MemoryKind> memory_kind;

    // New layouts for each value in this slice.
    //
    // `layouts.empty()` is a shorthand for `nullptr` layouts. Otherwise, the
    // size of the span must match the number of values in this slice.
    absl::Span<const LayoutRef> layouts;

    // Copy semantics for this slice. Aliasing is default.
    ArrayCopySemantics semantics = ArrayCopySemantics::kReuseInput;
  };

  // Returns a new `Bundle` with the values in the specified slices copied.
  //
  // It is equivalent to calling `Slice()`, `Copy()` for each slice, and then
  // `ConcatBundles()`.
  virtual absl::StatusOr<BundleRef> Copy(absl::Span<const int> slice_sizes,
                                         absl::Span<const CopySpec> specs) = 0;

  // Specification for resharding a slice of the bundle.
  //
  // In addition to the fields in `CopySpec`, `ReshardSpec` allows specifying
  // the new sharding spec for the slice.
  struct ReshardSpec : public CopySpec {
    // New shardings for the values in this slice.
    //
    // The size of the span must match the number of values in this slice.
    // sharding spec.
    absl::Span<const ShardingRef> shardings;
  };

  // Returns a new `Bundle` with the values in the specified slices copied.
  //
  // It is equivalent to calling `Slice()`, `Copy()` for each slice, and then
  // `ConcatBundles()`.
  virtual absl::StatusOr<BundleRef> Reshard(
      absl::Span<const int> slice_sizes,
      absl::Span<const ReshardSpec> specs) = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_BUNDLE_H_
