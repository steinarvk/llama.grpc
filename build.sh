#!/bin/bash
set -exu

./build-llama.sh

bazelisk build server:llama.grpc_server
./bazel-bin/server/llama.grpc_server --help
