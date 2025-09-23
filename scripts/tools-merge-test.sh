#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="tools_merge_test"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Merging coffea files"
python src/tools/merge_coffea_files.py \
    -f $OUTPUT_BASE_DIR/analysis_test/test_databkgs.coffea \
        $OUTPUT_BASE_DIR/analysis_signals_test/test_signal.coffea  \
    -o $OUTPUT_DIR/test.coffea

ls $OUTPUT_DIR


