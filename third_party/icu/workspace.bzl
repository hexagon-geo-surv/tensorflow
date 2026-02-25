"""Loads a lightweight subset of the ICU library for Unicode processing."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# NOTE: If you upgrade this, generate the data files by following the
# instructions in third_party/icu/data/BUILD
def repo():
    tf_http_archive(
        name = "icu",
        strip_prefix = "icu",
        sha256 = "d5cf533cf70cd49044d89eda3e74880328eb9426e6fd2b3cc8f9a963d2ad480e",
        urls = tf_mirror_urls("https://github.com/unicode-org/icu/releases/download/release-77-1/icu4c-77_1-src.zip"),
        build_file = "//third_party/icu:icu.BUILD",
        patch_file = ["//third_party/icu:udata.patch"],
        patch_cmds = [
            "rm -f source/common/BUILD.bazel",
            "rm -f source/stubdata/BUILD.bazel",
            "rm -f source/i18n/BUILD.bazel",
            "rm -f source/tools/gennorm2/BUILD.bazel",
            "rm -f source/tools/toolutil/BUILD.bazel",
            "rm -f source/data/unidata/norm2/BUILD.bazel",
        ],
    )
