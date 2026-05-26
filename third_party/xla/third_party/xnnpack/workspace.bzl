"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "4226a9160617e689756ee979b2bf2cd02cf8e1841dba4002e98d608234b9a858",
        strip_prefix = "XNNPACK-4d837f20dfff9177c579e160cad339172c0843e5",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/4d837f20dfff9177c579e160cad339172c0843e5.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
