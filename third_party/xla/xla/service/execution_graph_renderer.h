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

#ifndef XLA_SERVICE_EXECUTION_GRAPH_RENDERER_H_
#define XLA_SERVICE_EXECUTION_GRAPH_RENDERER_H_

#include <functional>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/execution_graph.h"

namespace xla {

// Renders a Thunk Execution Graph as a human-readable visual graph.
absl::StatusOr<std::string> RenderExecutionGraph(
    const ExecutionGraph& execution_graph,
    const cpu::ThunkSequence& thunk_sequence);

// Registers a function which implements rendering a dot graph to a graphviz
// URL.
void RegisterDotGraphToURLRenderer(
    std::function<absl::StatusOr<std::string>(absl::string_view dot)> renderer);

// Registers a function which implements generating a dot file for an execution
// graph with a given thunk sequence.
void RegisterExecutionGraphDotGenerator(
    std::function<
        absl::StatusOr<std::string>(const ExecutionGraph& execution_graph,
                                    const cpu::ThunkSequence& thunk_sequence)>
        dot_generator);

}  // namespace xla

#endif  // XLA_SERVICE_EXECUTION_GRAPH_RENDERER_H_
