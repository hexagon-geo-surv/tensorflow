// RUN: emitters_opt %s -split-input-file -xla-simplify-arith | FileCheck %s

module {
  func.func @atan2_simplify(%arg0: f32) -> f32 {
    %cst = arith.constant 1.000000e+00 : f32
    %ret = math.atan2 %arg0, %cst : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_simplify
// CHECK-SAME: (%[[ARG0:.*]]: f32)
// CHECK-NEXT:  %[[RET:.*]] = math.atan %[[ARG0]] : f32
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @atan2_no_simplify_not_one(%arg0: f32) -> f32 {
    %cst = arith.constant 2.000000e+00 : f32
    %ret = math.atan2 %arg0, %cst : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_no_simplify_not_one
// CHECK: math.atan2

// -----

module {
  func.func @atan2_no_simplify_not_constant(%arg0: f32, %arg1: f32) -> f32 {
    %ret = math.atan2 %arg0, %arg1 : f32
    return %ret : f32
  }
}
// CHECK-LABEL: @atan2_no_simplify_not_constant
// CHECK: math.atan2
