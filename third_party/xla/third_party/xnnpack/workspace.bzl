"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "7248c013307b649911bea038012f34cf730a55b461451d83247186dce0d663bd",
        strip_prefix = "XNNPACK-52ac311b528671823943058c626e6627258ae6f7",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/52ac311b528671823943058c626e6627258ae6f7.zip"),
        patch_file = ["//third_party/xnnpack:layering_check_fix.patch"],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
