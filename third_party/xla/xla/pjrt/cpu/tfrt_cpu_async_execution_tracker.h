/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_CPU_TFRT_CPU_ASYNC_EXECUTION_TRACKER_H_
#define XLA_PJRT_CPU_TFRT_CPU_ASYNC_EXECUTION_TRACKER_H_

#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/cpu/cpu_event.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

class TfrtCpuAsyncExecutionTracker;

// RAII wrapper for an async execution.
class TfrtCpuScopedAsyncExecution {
 public:
  TfrtCpuScopedAsyncExecution(TfrtCpuAsyncExecutionTracker* tracker,
                              int32_t launch_id)
      : tracker_(tracker), launch_id_(launch_id) {}
  TfrtCpuScopedAsyncExecution(TfrtCpuScopedAsyncExecution&& other);
  ~TfrtCpuScopedAsyncExecution();

  TfrtCpuScopedAsyncExecution(const TfrtCpuScopedAsyncExecution&) = delete;

  void reset();

 private:
  TfrtCpuAsyncExecutionTracker* tracker_;
  int32_t launch_id_;
};

// Tracks async executions that have not finished yet. Upon destruction, the
// tracker will wait for all async executions to finish to help graceful
// teardown of the runtime state.
class TfrtCpuAsyncExecutionTracker {
 public:
  // Registers a new execution dispatched to a device.
  TfrtCpuScopedAsyncExecution NewAsyncExecution(
      int32_t launch_id, tsl::AsyncValueRef<CpuEvent> execute_event);

  // Removes the execution from the tracker.
  void RemoveAsyncExecution(int32_t launch_id);

  // Sets the state of the execution to an error. Returns true if it succeeds to
  // set the error. Returns false if the execution has been removed or the
  // execute event is already set.
  bool SetError(int32_t launch_id, absl::Status error);

 private:
  absl::Mutex mu_;
  // Maps launch_id to the execute event of async executions that have not
  // finished yet.
  absl::flat_hash_map<int32_t, tsl::AsyncValueRef<CpuEvent>> executions_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_PJRT_CPU_TFRT_CPU_ASYNC_EXECUTION_TRACKER_H_
