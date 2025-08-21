#!/bin/bash

# End-to-end test script for PulsatileModelLearning learning pipeline
# This script automates the complete workflow without manual editing

set -e  # Exit on any error

# Generate today's date in YYMMDD format
YYMMDD=$(date +%y%m%d)
echo "========================================"
echo "Starting end-to-end test for date: $YYMMDD"
echo "========================================"

# Set up paths and temporary files
TEMP_CONFIG="scripts/test_corduroy_$YYMMDD.toml"

# Function to cleanup on exit
cleanup() {
    if [ -f "$TEMP_CONFIG" ]; then
        echo "Cleaning up temporary config file: $TEMP_CONFIG"
        rm -f "$TEMP_CONFIG"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

echo ""
echo "Step 1/4: Running classical learning..."
echo "----------------------------------------"
julia --threads=4 --project=PulsatileModelLearning notebooks/learn_classical.jl configs/test_classical.toml

echo ""
echo "Step 2/4: Analyzing classical results..."
echo "----------------------------------------"
julia --threads=4 --project=PulsatileModelLearning notebooks/analyze_learning_results.jl test_classical $YYMMDD simplex

echo ""
echo "Step 3/4: Running corduroy learning..."
echo "--------------------------------------"
echo "Generating temporary corduroy config with date: $YYMMDD"
sed "s/REPLACE_WITH_TEST_CLASSICAL_DATE/$YYMMDD/g" configs/test_corduroy.toml > "$TEMP_CONFIG"

echo "Running corduroy learning with config: $TEMP_CONFIG"
julia --threads=4 --project=PulsatileModelLearning notebooks/learn_corduroy.jl "$TEMP_CONFIG"

echo ""
echo "Step 4/4: Analyzing corduroy results..."
echo "---------------------------------------"
julia --threads=4 --project=PulsatileModelLearning notebooks/analyze_learning_results.jl test_corduroy $YYMMDD corduroy

echo ""
echo "========================================"
echo "✓ End-to-end test completed successfully!"
echo "========================================"
echo ""

