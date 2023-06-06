package(default_visibility = ["//visibility:public"])

load("@rules_proto_grpc//python:defs.bzl", "python_proto_library")

proto_library(
    name = "chatbot_proto",
    srcs = [ "chatbot.proto" ],
)

python_proto_library(
    name = "chatbot_py_proto",
    protos = [":chatbot_proto"],
)
