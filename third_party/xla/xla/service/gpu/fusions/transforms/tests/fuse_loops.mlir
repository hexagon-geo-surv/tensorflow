// RUN: mlir_fusions_opt -split-input-file %s -xla-gpu-fuse-loops \
// RUN: | FileCheck %s

#indexing_map = #xla_gpu.indexing_map<(d0, d1, d2, d3, d4, d5)[s0, s1] -> 
    (d3 floordiv 30,
    ((d3 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,
    (d3 mod 6) * 32 + d0 mod 32),
  domain: 
    d0 in [0, 127], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 599], d4 in [0, 0], d5 in [0, 0],
    s0 in [0, 7], s1 in [0, 0],
    (d3 mod 6) * 32 + d0 mod 32 in [0, 169],
    is_simplified: true>
#indexing_map1 = #xla_gpu.indexing_map<(d0, d1, d2, d3, d4, d5)[s0, s1] ->
    (0,
    d0 mod 32,
    d0 floordiv 32 + s0 * 4),
  domain:
    d0 in [0, 127], d1 in [0, 0], d2 in [0, 0],
    d3 in [0, 599], d4 in [0, 0], d5 in [0, 0],
    s0 in [0, 7], s1 in [0, 0],
    (d3 mod 6) * 32 + d0 mod 32 in [0, 169],
    is_simplified: true>
func.func @fusion(%arg0: tensor<20x160x170xf32>) -> tensor<1x32x33xf32> {
  %cst = arith.constant dense<0.000000e+00> : vector<8x1xf32>
  %c0 = arith.constant 0 : index
  %thread_id_x = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %thread_id_y = gpu.thread_id  y {xla.range = [0 : index, 0 : index]}
  %thread_id_z = gpu.thread_id  z {xla.range = [0 : index, 0 : index]}
  %block_id_x = gpu.block_id  x {xla.range = [0 : index, 599 : index]}
  %block_id_y = gpu.block_id  y {xla.range = [0 : index, 0 : index]}
  %block_id_z = gpu.block_id  z {xla.range = [0 : index, 0 : index]}
  %shmem = xla_gpu.allocate_shared : tensor<1x32x33xf32>
  %xla_loop = xla_gpu.loop (%thread_id_x, %thread_id_y, %thread_id_z, %block_id_x, %block_id_y, %block_id_z)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map iter_args(%iter = %cst) -> (vector<8x1xf32>) {
    %extracted = tensor.extract %arg0[%ra, %rb, %rc] : tensor<20x160x170xf32>
    %0 = math.exp %extracted : f32
    %1 = vector.insert %0, %iter [%i, %j] : f32 into vector<8x1xf32>
    xla_gpu.yield %1 : vector<8x1xf32>
  }
  %xla_loop_0 = xla_gpu.loop (%thread_id_x, %thread_id_y, %thread_id_z, %block_id_x, %block_id_y, %block_id_z)[%i, %j]
    -> (%ra, %rb, %rc) in #indexing_map1 iter_args(%iter = %shmem) -> (tensor<1x32x33xf32>) {
    %0 = vector.extract %xla_loop[%i, %j] : f32 from vector<8x1xf32>
    %inserted = tensor.insert %0 into %iter[%ra, %rb, %rc] : tensor<1x32x33xf32>
    xla_gpu.yield %inserted : tensor<1x32x33xf32>
  }
  %synced_tensor = xla_gpu.sync_threads %xla_loop_0 : tensor<1x32x33xf32>
  return %synced_tensor : tensor<1x32x33xf32>
}

// CHECK: #[[$FUSED_MAP:.*]] = #xla_gpu.indexing_map<(d0, d1, d2, d3, d4, d5)[s0, s1] -> 
// CHECK-SAME: (d3 floordiv 30, ((d3 floordiv 6) mod 5) * 32 + s0 * 4 + d0 floordiv 32,
// CHECK-SAME: (d3 mod 6) * 32 + d0 mod 32, 0, d0 mod 32, d0 floordiv 32 + s0 * 4),
// CHECK-SAME: domain: d0 in [0, 127], d1 in [0, 0], d2 in [0, 0], d3 in [0, 599],
// CHECK-SAME: d4 in [0, 0], d5 in [0, 0], s0 in [0, 7], s1 in [0, 0], (d3 mod 6) * 32 + d0 mod 32 in [0, 169]

// CHECK: xla_gpu.loop  {{.*}} in #[[$FUSED_MAP]]
// CHECK-NOT: vector.insert
// CHECK-NOT: vector.extract