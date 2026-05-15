"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "e678c224bb78a6beb22f5dddd5c34ea1cde4299facdd7903d915d09b06c346fe",
        strip_prefix = "XNNPACK-ace56b6162087f1926d782d39797a00fb56f2a30",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/ace56b6162087f1926d782d39797a00fb56f2a30.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
