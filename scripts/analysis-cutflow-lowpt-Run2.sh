#!/bin/bash

# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

OUTPUT_DIR=$OUTPUT_BASE_DIR/analysis_cutflow_lowpt_Run2
create_output_directory "$OUTPUT_DIR"

test_all_lowpt_file="$OUTPUT_DIR/test_all_lowpt.coffea"
if [[ ! -f "$test_all_lowpt_file" ]]; then
    display_section_header "Merging coffea files"
    python src/tools/merge_coffea_files.py \
        -f $OUTPUT_BASE_DIR/analysis_test_lowpt_data_Run2/test_data_lowpt.coffea \
            $OUTPUT_BASE_DIR/analysis_test_lowpt_signal_Run2/test_signal_lowpt.coffea  \
        -o $test_all_lowpt_file
fi

# Call the main analysis_test.sh script with Run3-specific parameters
bash coffea4bees/scripts/run-cutflow.sh \
    --input-file "test_all_lowpt.coffea" \
    --input-subdir "analysis_cutflow_lowpt_Run2" \
    --output-base "$OUTPUT_BASE_DIR" \
    --output-filename "test_dump_cutflow.yml" \
    --output-subdir "analysis_cutflow_lowpt_Run2" \
    --known-cutflow "coffea4bees/analysis/tests/known_Counts_lowpt_Run2.yml" 
