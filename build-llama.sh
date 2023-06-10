#!/bin/bash
set -eux
pushd llama.cpp
rm -rf build.llamacpp
mkdir build.llamacpp
pushd build.llamacpp
cmake ..
make LLAMA_OPENBLAS=1
popd
popd
