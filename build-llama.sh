#!/bin/bash
set -eux
pushd llama.cpp
rm -rf build.llamacpp
mkdir build.llamacpp
pushd build.llamacpp
cmake ..
make
popd
popd
