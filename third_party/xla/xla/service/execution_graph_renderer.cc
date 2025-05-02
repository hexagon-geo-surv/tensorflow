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

#include "xla/service/execution_graph_renderer.h"

#include <functional>
#include <string>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/execution_graph.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {

absl::Mutex url_renderer_mu(absl::kConstInit);
std::function<absl::StatusOr<std::string>(absl::string_view)>* url_renderer
    ABSL_GUARDED_BY(url_renderer_mu) = nullptr;

absl::Mutex execution_graph_dot_generator_mu(absl::kConstInit);
std::function<absl::StatusOr<std::string>(
    const ExecutionGraph& execution_graph,
    const cpu::ThunkSequence& thunk_sequence)>* execution_graph_dot_generator
    ABSL_GUARDED_BY(execution_graph_dot_generator_mu) = nullptr;

absl::StatusOr<std::string> RenderExecutionGraph(
    const ExecutionGraph& execution_graph,
    const cpu::ThunkSequence& thunk_sequence) {
  std::string rendered_dot;
  {
    absl::MutexLock lock(&execution_graph_dot_generator_mu);
    if (execution_graph_dot_generator == nullptr) {
      return Unavailable(
          "Can't render as URL; no execution graph dot generator was "
          "registered.");
    }
    TF_ASSIGN_OR_RETURN(rendered_dot, (*execution_graph_dot_generator)(
                                          execution_graph, thunk_sequence));
  }
  absl::MutexLock lock(&url_renderer_mu);
  if (url_renderer == nullptr) {
    return Unavailable("Can't render as URL; no URL renderer was registered.");
  }
  return (*url_renderer)(rendered_dot);
}

void RegisterDotGraphToURLRenderer(
    std::function<absl::StatusOr<std::string>(absl::string_view)> renderer) {
  absl::MutexLock lock(&url_renderer_mu);
  if (url_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterGraphToURLRenderer. Last call "
                    "wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
  }
  delete url_renderer;
  url_renderer =
      new std::function<absl::StatusOr<std::string>(absl::string_view)>(
          std::move(renderer));
}

void RegisterExecutionGraphDotGenerator(
    std::function<
        absl::StatusOr<std::string>(const ExecutionGraph& execution_graph,
                                    const cpu::ThunkSequence& thunk_sequence)>
        dot_generator) {
  absl::MutexLock lock(&execution_graph_dot_generator_mu);
  if (execution_graph_dot_generator != nullptr) {
    LOG(WARNING)
        << "Multiple calls to RegisterExecutionGraphDotGenerator. Last call "
           "wins, but because order of initialization in C++ is "
           "nondeterministic, this may not be what you want.";
  }
  delete execution_graph_dot_generator;
  execution_graph_dot_generator = new std::function<absl::StatusOr<std::string>(
      const ExecutionGraph& execution_graph,
      const cpu::ThunkSequence& thunk_sequence)>(std::move(dot_generator));
}

}  // namespace xla
