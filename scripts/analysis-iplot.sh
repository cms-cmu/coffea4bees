#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

display_section_header "Running iPlot test"
run_command python coffea4bees/plots/tests/iPlot_test.py \
    --inputFile $OUTPUT_BASE_DIR/tools_merge_test/test.coffea
