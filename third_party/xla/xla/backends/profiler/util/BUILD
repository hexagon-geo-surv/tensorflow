load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//xla:internal"],
)

cc_library(
    name = "metadata_registry",
    srcs = ["metadata_registry.cc"],
    hdrs = ["metadata_registry.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)
