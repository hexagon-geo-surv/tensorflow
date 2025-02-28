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

#include "xla/pjrt/cpu/tfrt_cpu_async_execution_tracker.h"

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/cpu/cpu_event.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

TfrtCpuScopedAsyncExecution::TfrtCpuScopedAsyncExecution(
    TfrtCpuScopedAsyncExecution&& other)
    : tracker_(other.tracker_), launch_id_(other.launch_id_) {
  other.tracker_ = nullptr;
}

TfrtCpuScopedAsyncExecution::~TfrtCpuScopedAsyncExecution() { reset(); }

void TfrtCpuScopedAsyncExecution::reset() {
  if (tracker_ != nullptr) {
    tracker_->RemoveAsyncExecution(launch_id_);
    tracker_ = nullptr;
  }
}

TfrtCpuScopedAsyncExecution TfrtCpuAsyncExecutionTracker::NewAsyncExecution(
    int32_t launch_id, tsl::AsyncValueRef<CpuEvent> execute_event) {
  absl::MutexLock lock(&mu_);
  executions_.insert({launch_id, std::move(execute_event)});
  return TfrtCpuScopedAsyncExecution(this, launch_id);
}

void TfrtCpuAsyncExecutionTracker::RemoveAsyncExecution(int32_t launch_id) {
  absl::MutexLock lock(&mu_);
  executions_.erase(launch_id);
}

bool TfrtCpuAsyncExecutionTracker::SetError(int32_t launch_id,
                                            absl::Status error) {
  absl::ReleasableMutexLock lock(&mu_);
  auto it = executions_.find(launch_id);
  if (it != executions_.end() && it->second.IsUnavailable()) {
    tsl::AsyncValueRef<CpuEvent> execute_event = std::move(it->second);
    executions_.erase(it);
    lock.Release();
    execute_event.SetError(std::move(error));
    return true;
  }
  return false;
}

}  // namespace xla
