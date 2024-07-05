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

#ifndef XLA_SERVICE_CPU_RUNTIME_THUNK_UTILS_H_
#define XLA_SERVICE_CPU_RUNTIME_THUNK_UTILS_H_

#include <atomic>
#include <cstdint>

#include "xla/service/cpu/runtime/thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// Keep track of pending tasks and an event that signals completion of the
// operation to the caller.
struct ThunkExecuteState {
  explicit ThunkExecuteState(int64_t parallel_tasks)
      : pending_tasks(parallel_tasks),
        event(tsl::MakeConstructedAsyncValueRef<Thunk::ExecuteEvent>()) {}

  void Notify() {
    if (pending_tasks.load(std::memory_order_relaxed) == 1 ||
        pending_tasks.fetch_sub(1, std::memory_order_relaxed) == 1) {
      event.SetStateConcrete();
    }
  }

  std::atomic<int64_t> pending_tasks;
  tsl::AsyncValueRef<Thunk::ExecuteEvent> event;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_THUNK_UTILS_H_
