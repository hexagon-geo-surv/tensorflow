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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_

#include <cstdint>
#include <list>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/shape_util.h"

namespace xla {
namespace memory_space_assignment {

enum class MemoryTransferDirection {
  kUnsupported,
  kDefaultToAlternate,
  kAlternateToDefault,
};

// REQUIRES:
// * async_copy must be an async copy-start or slice-start instruction.
MemoryTransferDirection GetAsyncCopyDirection(const HloInstruction* async_copy,
                                              int64_t alternate_memory_space);

// This struct is used to track the outstanding async copy instructions and
// the remaining bytes required to be accessed.
struct OutstandingAsyncCopy {
  const HloInstruction* copy_start_inst;
  float remaining_bytes_to_transfer;
  bool operator==(const OutstandingAsyncCopy& other) const {
    return copy_start_inst == other.copy_start_inst &&
           remaining_bytes_to_transfer == other.remaining_bytes_to_transfer;
  }
};

// A wrapper class around runtime simulator.
class RuntimeSimulator {
 public:
  explicit RuntimeSimulator(CostAnalysis* cost_analysis,
                            int64_t alternate_memory_space)
      : cost_analysis_(cost_analysis),
        alternate_memory_space_(alternate_memory_space) {}

  // This constructor is used to inject the outstanding async copy queues for
  // testing purpose.
  explicit RuntimeSimulator(
      CostAnalysis* cost_analysis, int64_t alternate_memory_space,
      const std::list<OutstandingAsyncCopy>& outstanding_read_default_queue,
      const std::list<OutstandingAsyncCopy>& outstanding_write_default_queue)
      : cost_analysis_(cost_analysis),
        alternate_memory_space_(alternate_memory_space),
        outstanding_read_default_queue_(outstanding_read_default_queue),
        outstanding_write_default_queue_(outstanding_write_default_queue) {}

  ~RuntimeSimulator() = default;

  // This function provides a basic estimate without considering the overhead of
  // async copies.
  float SimulateElapsedTimeWithoutAsyncCopies(
      const HloLiveRange& hlo_live_range,
      const AllocationSequence& allocations);

  // This function, compared with SimulateElapsedTimeWithoutAsyncCopies
  // function, provides a more accurate estimated execution time, as it
  // simulates the default memory communication to estimate the overhead of
  // async copies.
  // To simulate the overhead of async copies, we need to maintain two queues to
  // track the memory access requests that read/write the default memory.
  // Specifically, for any async copy instruction (copy-start), we push it to
  // the corresponding queue. When the copy-done instruction is executed, we pop
  // it (and all the previous copy-start instructions) from the queue and
  // calculate the execution time of the async copy. Instead of the copy-done
  // instruction, we also try to drain the queue during computation instruction,
  // in which default memory is not accessed. The most important feature for the
  // memory model is sharing the bandwidth. Specifically, if both queues are not
  // empty, we can only use half of the bandwidth to transfer the request for
  // each of them in parallel.

  // Here is the overview of the algorithm:
  // func SimulateElapsedTime():
  //   read_queue = []
  //   write_queue = []
  //   total_elapsed = 0
  //   for each instruction:
  //     if instruction is copy-start:
  //       if instruction is default to alternate memory:
  //         read_queue.push(instruction)
  //       else if instruction is alternate to default memory:
  //         write_queue.push(instruction)
  //     else if instruction is copy-done:
  //       # pop instruction from the read/write queue and calculate the
  //       # execution time of the async copy
  //       total_elapsed += SimulateAsyncCopyDone()
  //     else if instruction is compute:
  //       # Calculate the execution time of the compute instruction.
  //       total_elapsed += GetInstructionElapsedInAlternateMemory()
  //       # Get time window in which defaul memory is not accessed.
  //       idle_time_window = GetDefaultMemoryBandwidthIdleTime()
  //       # Process read/write queues during the time window.
  //       ProcessAsyncCopyInTimeWindow(idle_time_window)
  //     end if
  //   end for
  //   return total_elapsed
  float SimulateElapsedTime(const HloModule* hlo_module,
                            const HloLiveRange& hlo_live_range,
                            const AllocationSequence& allocations);

  // This is an auxiliary function for simulating the execution
  // time for executing a copy-done instruction. It returns the
  // elapsed time (in seconds) for executing the copy-done instruction.
  //
  // This function also updates the passed in queues as we complete async copies
  // during the simulation.
  //
  // We simulate the shared bandwidth for default-alternate memory access.
  // For example, if the copy-done instruction is a default-write memory
  // process, and there are outstanding default-read memory processes in the
  // outstanding_read_default_queue, then we use half of the bandwidth to
  // process both requests in parallel. Otherwise, we use the full bandwidth to
  // process the default-write request.
  float SimulateAsyncCopyDone(const HloInstruction* copy_done_instruction);

  const std::list<OutstandingAsyncCopy>& GetOutstandingReadDefaultQueue() const;

  const std::list<OutstandingAsyncCopy>& GetOutstandingWriteDefaultQueue()
      const;

  // This is an auxiliary function for simulating the execution
  // time for a compute instruction. It returns the elapsed time (in seconds)
  // for executing the compute instruction.
  //
  // This function first calculates the elapsed time for the compute instruction
  // itself. The elapsed time is independent with any outstanding memory
  // requests.
  //
  // Except returning the elapsed time, this function also updates the
  // outstanding memory requests queue: It calculates the time
  // window in which the compute instruction does not access the default memory,
  // and tries to drain the outstanding memory requests in this time window.
  float SimulateComputeInstruction(
      const HloInstruction* compute_instruction,
      absl::Span<const std::pair<int64_t, ShapeIndex>>
          operands_in_alternate_memory,
      absl::Span<const ShapeIndex> outputs_in_alternate_memory);

  // This is an auxiliary function which simulates the process of draining
  // the memory access queue in a given time window. There are two queues
  // which will share the bandwidth: ```read_queue``` and ```write_queue```
  // which track the memory access requests that read/write the default
  // memory. When both of the queues are not empty, the front requests from
  // both queues equally share the bandwidth. When one of the queue is
  // empty, the other queue can use the full bandwidth.
  void ProcessAsyncCopiesInTimeWindow(float time_window);

 private:
  // This function parses the memory space assignment solution and initializes
  // the maps that record, for each instruction, which outputs and operands are
  // stored in alternate memory. These maps are used to estimate the runtime of
  // the HLO module.
  void InitializeAlternateMemoryMap(const AllocationSequence& allocations);
  const CostAnalysis* cost_analysis_;
  CostAnalysis::Cache cost_analysis_cache_;
  // Members used for memory model simulation
  int64_t alternate_memory_space_;
  std::list<OutstandingAsyncCopy> outstanding_read_default_queue_;
  std::list<OutstandingAsyncCopy> outstanding_write_default_queue_;
  absl::flat_hash_map<const HloInstruction*, std::vector<ShapeIndex>>
      outputs_in_alternate_memory_map_;
  absl::flat_hash_map<const HloInstruction*,
                      std::vector<std::pair<int64_t, ShapeIndex>>>
      operands_in_alternate_memory_map_;
  // This function updates the queue by updating the front request with the
  // processed bytes. If the request is completed (no remaining bytes to
  // process), the function returns the instruction and pop it from the queue.
  // Otherwise, it returns nullptr.
  const HloInstruction* RemoveBytesFromQueueIfNotEmpty(
      std::list<OutstandingAsyncCopy>& async_copy_queue, float processed_bytes);
};
}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
