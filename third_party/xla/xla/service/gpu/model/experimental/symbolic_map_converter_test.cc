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

#include "xla/service/gpu/model/experimental/symbolic_map_converter.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineMap;
using ::mlir::MLIRContext;

// Helper function to parse an AffineMap from a string.
AffineMap ParseAffineMap(absl::string_view serialized_affine_map,
                         MLIRContext* context) {
  std::string full_affine_map_string =
      absl::StrCat("affine_map<", serialized_affine_map, ">");
  return mlir::cast<mlir::AffineMapAttr>(
             mlir::parseAttribute(full_affine_map_string, context))
      .getValue();
}

TEST(SymbolicMapConverterTest, AffineToSymbolicRoundTrip) {
  MLIRContext mlir_context;
  SymbolicExprContext symbolic_context;

  AffineMap affine_map = ParseAffineMap(
      "(d0, d1)[s0, s1] -> (d0 + s1 * 2, d1 - s0, d0 floordiv 3, d1 mod 4)",
      &mlir_context);

  llvm::SmallVector<SymbolicExpr> symbolic_exprs =
      AffineMapToSymbolicExprs(affine_map, &symbolic_context);

  EXPECT_EQ(symbolic_exprs.size(), 4);

  AffineMap round_trip_map = SymbolicExprsToAffineMap(
      symbolic_exprs, &mlir_context, affine_map.getNumDims(),
      affine_map.getNumSymbols());
  EXPECT_EQ(affine_map, round_trip_map);
}

TEST(SymbolicMapConverterTest, SymbolicToAffineFailure) {
  MLIRContext mlir_context;
  SymbolicExprContext symbolic_context;

  SymbolicExpr d0 = symbolic_context.CreateVariable(0);
  SymbolicExpr c1 = symbolic_context.CreateConstant(1);
  // kMax is not representable in AffineExpr.
  SymbolicExpr max_expr = d0.max(c1);

  AffineMap affine_map =
      SymbolicExprsToAffineMap({max_expr}, &mlir_context, 1, 0);
  EXPECT_FALSE(affine_map);
}

TEST(SymbolicMapConverterTest, SymbolicToAffineNestedFailure) {
  MLIRContext mlir_context;
  SymbolicExprContext symbolic_context;

  SymbolicExpr d0 = symbolic_context.CreateVariable(0);
  SymbolicExpr c1 = symbolic_context.CreateConstant(1);
  SymbolicExpr c2 = symbolic_context.CreateConstant(2);

  // d0 + max(c1, c2). max is not representable in AffineExpr.
  SymbolicExpr nested_max_expr = d0 + c1.max(c2);

  // This should not crash and should return a null AffineMap.
  AffineMap affine_map =
      SymbolicExprsToAffineMap({nested_max_expr}, &mlir_context, 1, 0);
  EXPECT_FALSE(affine_map);
}

struct Interval {
  int64_t lower, upper;

  bool operator==(const Interval& other) const {
    return lower == other.lower && upper == other.upper;
  }
};

TEST(SymbolicMapConverterTest, ConvertAffineConstraintsToSymbolicConstraints) {
  MLIRContext mlir_context;
  SymbolicExprContext symbolic_context;

  mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, &mlir_context);
  mlir::AffineExpr s0 = mlir::getAffineSymbolExpr(0, &mlir_context);
  mlir::AffineExpr c1 = mlir::getAffineConstantExpr(1, &mlir_context);

  llvm::MapVector<mlir::AffineExpr, Interval> affine_constraints;
  affine_constraints[d0 + s0] = {0, 127};
  affine_constraints[s0 * 2] = {0, 63};
  affine_constraints[d0 - c1] = {10, 20};

  llvm::MapVector<SymbolicExpr, Interval> symbolic_constraints =
      ConvertAffineConstraintsToSymbolicConstraints(
          affine_constraints, &symbolic_context, /*num_dims=*/1);

  SymbolicExpr sym_d0 = symbolic_context.CreateVariable(0);
  SymbolicExpr sym_s0 = symbolic_context.CreateVariable(1);
  SymbolicExpr sym_c1 = symbolic_context.CreateConstant(1);

  EXPECT_EQ(symbolic_constraints.size(), 3);
  EXPECT_EQ(symbolic_constraints[sym_d0 + sym_s0], (Interval{0, 127}));
  EXPECT_EQ(symbolic_constraints[sym_s0 * 2], (Interval{0, 63}));
  EXPECT_EQ(symbolic_constraints[sym_d0 - sym_c1], (Interval{10, 20}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
