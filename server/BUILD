cc_binary(
    name = "llama.grpc_server",
    srcs = ["main.cc"],
    defines = ["BAZEL_BUILD"],
    deps = [
        "//proto:llama_cc_grpc",
        "@llama.cpp//:llama.cpp",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:initialize",
        "@com_google_absl//absl/strings:str_format",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_reflection",
    ],
)
