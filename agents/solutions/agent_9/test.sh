#!/bin/bash
# Complete test suite for NVPTXCompilerV2

set -e

echo "======================================================================"
echo "NVPTXCompilerV2 Test Suite"
echo "======================================================================"
echo

# Run unit tests
echo "[1/2] Running unit tests..."
echo "----------------------------------------------------------------------"
python3 "$(dirname $0)/unit_tests.py"

if [ $? -ne 0 ]; then
    echo
    echo "✗ Unit tests failed"
    exit 1
fi

echo
echo "[2/2] Running integration tests..."
echo "----------------------------------------------------------------------"
python3 "$(dirname $0)/integration_test.py"

if [ $? -ne 0 ]; then
    echo
    echo "✗ Integration tests failed"
    exit 1
fi

echo
echo "======================================================================"
echo "✓ ALL TESTS PASSED"
echo "======================================================================"
