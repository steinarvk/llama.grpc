load("@python_deps//:requirements.bzl", "requirement")

py_binary(
    name = "classify",
    srcs = ["classify.py"],
    deps = [
        "//proto:llama_py_grpc",
        requirement("absl-py")
    ],
)
