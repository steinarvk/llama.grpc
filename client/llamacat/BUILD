load("@python_deps//:requirements.bzl", "requirement")

py_binary(
    name = "llamacat",
    srcs = ["llamacat.py"],
    deps = [
        "//proto:llama_py_grpc",
        requirement("absl-py")
    ],
)
