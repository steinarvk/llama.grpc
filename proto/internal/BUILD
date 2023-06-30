package(default_visibility = ["//visibility:public"])

load("@rules_proto_grpc//cpp:defs.bzl", "cpp_proto_library")

proto_library(
    name = "llamagrpc_internal_proto",
    srcs = [ "llamagrpc_internal.proto" ],
)

cpp_proto_library(
    name = "llamagrpc_internal_cc_proto",
    protos = [":llamagrpc_internal_proto"],
)