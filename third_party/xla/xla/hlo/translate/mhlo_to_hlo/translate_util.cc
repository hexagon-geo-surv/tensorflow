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

#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/utils/unregistered_attributes.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace mlir {

mlir::FailureOr<xla::Shape> ExtractXlaShape(mlir::Operation* op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(xla::kXlaShape)) {
    return *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  }
  std::vector<xla::Shape> subshapes;
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    subshapes.push_back(xla::TypeToShape(result.getType()));
    if (subshapes.back().element_type() == xla::PRIMITIVE_TYPE_INVALID) {
      return op->emitError() << "result #" << index << " type is not supported";
    }
  }
  if (subshapes.size() > 1) {
    return xla::ShapeUtil::MakeTupleShape(subshapes);
  }
  return subshapes[0];
}

}  // namespace mlir
