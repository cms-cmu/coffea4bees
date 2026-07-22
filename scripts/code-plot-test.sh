#!/bin/bash
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
INPUT_DIR="$OUTPUT_BASE_DIR/tools_merge_test"
JOB="code_plot_test"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running code plot test"
run_command python coffea4bees/tests/dumpPlotCounts.py \
    --input $INPUT_DIR/test.coffea \
    --output $OUTPUT_DIR/test_dumpPlotCounts.yml
run_command python coffea4bees/tests/plots_test.py \
    --inputFile $INPUT_DIR/test.coffea \
    --known coffea4bees/plots/tests/known_PlotCounts.yml
ls $OUTPUT_DIR/test_dumpPlotCounts.yml

