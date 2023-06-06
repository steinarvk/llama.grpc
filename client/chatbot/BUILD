load("@python_deps//:requirements.bzl", "requirement")

py_binary(
    name = "chatbot",
    srcs = ["chatbot.py"],
    deps = [
        "//proto:llama_py_grpc",
        "//client/chatbot/proto:chatbot_py_proto",
        requirement("absl-py")
    ],
)
