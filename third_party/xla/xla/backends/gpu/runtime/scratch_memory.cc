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

#include "xla/backends/gpu/runtime/scratch_memory.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"  // gloop
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/scratch_memory_requests.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/symmetric_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/memory_space.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

absl::StatusOr<ScratchMemory> AcquireScratchMemory(
    const CollectiveParams& collective_params,
    const ScratchMemoryRequests& scratch_memory_requests,
    stream_executor::StreamExecutor* executor, CollectiveCliques& cliques) {
  absl::flat_hash_map<GpuCliqueKey,
                      tsl::TiedRef<stream_executor::MemoryAllocation>>
      memory_allocations;

  absl::flat_hash_map<GpuCliqueKey, tsl::TiedRef<xla::SymmetricMemory>>
      symmetric_memories;

  if (scratch_memory_requests.Empty()) {
    return ScratchMemory(std::move(memory_allocations),
                         std::move(symmetric_memories));
  }

  ASSIGN_OR_RETURN(
      std::unique_ptr<stream_executor::MemoryAllocator> collective_allocator,
      executor->CreateMemoryAllocator(
          stream_executor::MemorySpace::kCollective));
  for (const auto& [clique_key, size] :
       scratch_memory_requests.OrderedRequests()) {
    ASSIGN_OR_RETURN(
        std::unique_ptr<stream_executor::MemoryAllocation> memory_allocation,
        collective_allocator->Allocate(size));
    XLA_VLOG_DEVICE(1, executor->device_ordinal())
        << "Allocated scratch memory (address="
        << memory_allocation->address().opaque()
        << "; size=" << memory_allocation->address().size() << ") for clique "
        << clique_key;

    const std::optional<RankId> rank =
        clique_key.rank(collective_params.global_device_id);
    ASSIGN_OR_RETURN(auto* comm, cliques.GetComm(clique_key, rank.value()));
    ASSIGN_OR_RETURN(auto symmetric_memory,
                     comm->CreateSymmetricMemory(memory_allocation->address()));

    ASSIGN_OR_RETURN(auto tied_symmetric_memory,
                     cliques.Tie(clique_key, std::move(symmetric_memory)));

    ASSIGN_OR_RETURN(
        tsl::TiedRef<stream_executor::MemoryAllocation> tied_memory_allocation,
        cliques.Tie(clique_key, std::move(memory_allocation)));

    symmetric_memories[clique_key] = std::move(tied_symmetric_memory);
    memory_allocations[clique_key] = std::move(tied_memory_allocation);
  }

  return ScratchMemory(std::move(memory_allocations),
                       std::move(symmetric_memories));
}

}  // namespace gpu
}  // namespace xla
