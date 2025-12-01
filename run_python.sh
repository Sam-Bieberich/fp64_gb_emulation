#!/bin/bash
#
# Wrapper script to run the Python example
#

if ! command -v python &> /dev/null; then
    echo "Error: python not found"
    exit 1
fi

python example_simple.py
