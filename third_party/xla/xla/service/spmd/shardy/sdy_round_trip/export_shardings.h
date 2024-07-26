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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_EXPORT_SHARDINGS_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_EXPORT_SHARDINGS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla {
namespace sdy {

// Registers the xla-sdy-round-trip-export-shardings pass.
void registerSdyRoundTripExportShardingsPass();

// Creates the pass that converts the shardings from `kShardingAttr` to
// `kShardingRoundTripAttr` in the HLO frontend attributes and saves the
// mesh symbols as `kMeshesRoundTripAttr` in the module frontend attributes.
//
// If `keepShardings` is true, the shardings attributes are still kept around,
// else they are removed. This is necessary for using native Shardy JAX lowering
// where the backend is Pathways.
std::unique_ptr<mlir::Pass> createSdyRoundTripExportShardingsPass(
    bool keepShardings = true);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_EXPORT_SHARDINGS_H_
