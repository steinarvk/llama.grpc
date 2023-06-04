package(default_visibility = ["//visibility:public"])

load("@rules_proto_grpc//cpp:defs.bzl", "cpp_grpc_library")
load("@rules_proto_grpc//python:defs.bzl", "python_grpc_library")

proto_library(
    name = "llama_proto",
    srcs = [ "llama.proto" ],
)

cpp_grpc_library(
    name = "llama_cc_grpc",
    protos = [":llama_proto"],
)

python_grpc_library(
    name = "llama_py_grpc",
    protos = [":llama_proto"],
)
