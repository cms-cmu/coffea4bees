#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash coffea4bees/scripts/run-cutflow.sh \
    --input-file "test.coffea" \
    --input-subdir "analysis_test_Run3" \
    --output-base "$OUTPUT_BASE_DIR" \
    --output-filename "test_dump_cutflow.yml" \
    --output-subdir "analysis_cutflow_Run3" \
    --known-cutflow "coffea4bees/analysis/tests/known_Counts_Run3.yml" 
