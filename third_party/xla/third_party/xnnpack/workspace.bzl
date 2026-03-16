"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "c2f660d9874420bda4555212f340c5b845b5c814576460e348da60d96f234f77",
        strip_prefix = "XNNPACK-3ffd5c5261166ec8630e775503f91761364cc6bd",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/3ffd5c5261166ec8630e775503f91761364cc6bd.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
