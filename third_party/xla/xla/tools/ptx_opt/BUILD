load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//xla:lit.bzl", "lit_test_suite")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_binary(
    name = "ptx_opt",
    srcs = ["ptx_opt.cc"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla:debug_options_flags",
        "//xla/service/gpu/llvm_gpu_backend:load_ir_module",
        "//xla/service/gpu/llvm_gpu_backend:nvptx_backend",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:NVPTXCodeGen",  # buildcleaner: keep
        "@llvm-project//llvm:ObjCARC",  # buildcleaner: keep
        "@local_tsl//tsl/platform:platform_port",
    ] + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]),
)

lit_test_suite(
    name = "ptx_opt_tests",
    srcs = glob(["**/*.ll"]),
    args = if_cuda_is_configured([
        "--param=PTX=PTX",
        "--param=GPU=a100_pcie_80",
    ]),
    cfg = "//xla:lit.cfg.py",
    hermetic_cuda_data_dir = "%S/../../../../../cuda_nvcc",
    tools = [
        ":ptx_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)
