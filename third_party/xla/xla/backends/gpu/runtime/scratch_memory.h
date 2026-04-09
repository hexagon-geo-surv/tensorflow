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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_H_
#define XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_H_

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/tied_ref.h"

namespace xla::gpu {
// Holds a scratch device memory per HLO executable which can be used
// by collective operations such as RaggedAllToAll.
class ScratchMemory {
 public:
  explicit ScratchMemory(
      absl::flat_hash_map<GpuCliqueKey,
                          tsl::TiedRef<stream_executor::MemoryAllocation>>
          memory_allocations,
      absl::flat_hash_map<GpuCliqueKey, tsl::TiedRef<xla::SymmetricMemory>>
          symmetric_memory)
      : memory_allocations_(std::move(memory_allocations)),
        symmetric_memory_(std::move(symmetric_memory)) {}

  absl::StatusOr<stream_executor::DeviceAddressBase> GetScratchMemoryAddress(
      const GpuCliqueKey& clique_key) {
    auto it = memory_allocations_.find(clique_key);
    if (it == memory_allocations_.end()) {
      return absl::NotFoundError(
          absl::StrCat("No scratch memory found for clique key: ", clique_key));
    }
    return it->second.Lock()->address();
  }

  absl::StatusOr<std::shared_ptr<xla::SymmetricMemory>> GetSymmetricMemory(
      const GpuCliqueKey& clique_key) {
    auto it = symmetric_memory_.find(clique_key);
    if (it == symmetric_memory_.end()) {
      return absl::NotFoundError(absl::StrCat(
          "No symmetric memory found for clique key: ", clique_key));
    }
    return it->second.Lock();
  }

 private:
  absl::flat_hash_map<GpuCliqueKey,
                      tsl::TiedRef<stream_executor::MemoryAllocation>>
      memory_allocations_;
  absl::flat_hash_map<GpuCliqueKey, tsl::TiedRef<xla::SymmetricMemory>>
      symmetric_memory_;
};

absl::StatusOr<ScratchMemory> AcquireScratchMemory(
    const CollectiveParams& collective_params,
    const ScratchMemoryRequests& scratch_memory_requests,
    stream_executor::StreamExecutor* executor, CollectiveCliques& cliques);
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SCRATCH_MEMORY_H_
