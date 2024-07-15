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

#include <queue>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_live_range.h"
#include "xla/service/memory_space_assignment/allocation.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"

namespace xla {
namespace memory_space_assignment {

// A wrapper class around runtime simulator.
class RuntimeSimulator {
 public:
  explicit RuntimeSimulator(CostAnalysis* cost_analysis)
      : cost_analysis_(cost_analysis) {}
  virtual ~RuntimeSimulator() = default;
  // This function is used to predict the effectiveness of the memory space
  // assignment solution. Specifically, it returns the estimated execution time
  // (in seconds) of the HLO module for the given memory space assignment (i.e.,
  // ```allocations```).
  float ComputeEstimatedElapsedTime(const HloLiveRange& hlo_live_range,
                                    const AllocationSequence& allocations);

  // This is an auxiliary function which is used for simulating the execution
  // time of a single async copy instruction, with size of
  // ```bytes_to_transfer``` bytes. This async copy instruction has to share the
  // bandwidth with the memory access requests in
  // ```memory_access_queue_to_share_bandwidth```. The bandwidth is shared
  // equally: When the memory_access_queue_to_share_bandwidth is not empty, we
  // can only use half of the bandwidth to transfer the request, and use the
  // other half to transfer the memory requests in the queue. When the queue is
  // drained, we can use the full bandwidth to transfer the request.
  // ```remaining_size_of_buffers``` is a map from the memory access instruction
  // to the remaining size of the buffer. For memory access instructions which
  // are being completed (remaining size is 0), this function also removes them
  // from the map.

  static float SimulateAsyncCopyDone(
      float bytes_to_transfer,
      std::queue<const HloInstruction*>& memory_access_queue_to_share_bandwidth,
      absl::flat_hash_map<const HloInstruction*, float>&
          remaining_size_of_buffers,
      float default_memory_bytes_per_second);

 private:
  const CostAnalysis* cost_analysis_;
  CostAnalysis::Cache cost_analysis_cache_;
};
}  // namespace memory_space_assignment
}  // namespace xla
#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_SIMULATOR_H_
