#!/bin/bash
#
# Example application script that uses cuBLAS
# This demonstrates how compare_fp64.sh can wrap any .sh script
#

echo "Running example CUDA application..."

# Check if the example binary exists
if [ ! -f "./build/example_simple" ] && [ ! -f "./build/Release/example_simple.exe" ]; then
    echo "Error: example_simple not built. Run:"
    echo "  mkdir build && cd build"
    echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build . --config Release"
    exit 1
fi

# Run the appropriate binary
if [ -f "./build/example_simple" ]; then
    ./build/example_simple
elif [ -f "./build/Release/example_simple.exe" ]; then
    ./build/Release/example_simple.exe
fi
