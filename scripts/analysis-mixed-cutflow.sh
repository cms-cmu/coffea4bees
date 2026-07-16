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
JOB="analysis_mixed_cutflow"
INPUT_DIR="$OUTPUT_BASE_DIR/analysis_test_mixed"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"


display_section_header "Running dump cutflow test for Bkg and Data"
run_command python coffea4bees/analysis/tests/dumpCutFlow.py \
    --input $INPUT_DIR/testMixedBkg_TT.coffea \
    -o $OUTPUT_DIR/test_dump_MixedBkg_TT.yml
run_command python coffea4bees/analysis/tests/dumpCutFlow.py \
    --input $INPUT_DIR/testMixedBkg_data_3b_for_mixed.coffea \
    -o $OUTPUT_DIR/test_dump_MixedBkg_data_3b_for_mixed.yml
run_command python coffea4bees/analysis/tests/dumpCutFlow.py \
    --input $INPUT_DIR/testMixedData.coffea \
    -o $OUTPUT_DIR/test_dump_MixedData.yml
run_command python coffea4bees/analysis/tests/dumpCutFlow.py \
    --input $INPUT_DIR/testSignal_UL.coffea \
    -o $OUTPUT_DIR/test_dump_Signal.yml
ls $OUTPUT_DIR/test_dump_MixedBkg_TT.yml
ls $OUTPUT_DIR/test_dump_MixedBkg_data_3b_for_mixed.yml
ls $OUTPUT_DIR/test_dump_MixedData.yml
ls $OUTPUT_DIR/test_dump_Signal.yml

display_section_header "Running cutflow test for mixed "
run_command python coffea4bees/analysis/tests/cutflow_test.py \
    --inputFile $INPUT_DIR/testMixedBkg_TT.coffea \
    --knownCounts coffea4bees/analysis/tests/known_Counts_MixedBkg_TT.yml
run_command python coffea4bees/analysis/tests/cutflow_test.py \
    --inputFile $INPUT_DIR/testMixedBkg_data_3b_for_mixed.coffea \
    --knownCounts coffea4bees/analysis/tests/known_Counts_MixedBkg_data_3b_for_mixed.yml
run_command python coffea4bees/analysis/tests/cutflow_test.py \
    --inputFile $INPUT_DIR/testMixedData.coffea \
    --knownCounts coffea4bees/analysis/tests/known_Counts_MixedData.yml
run_command python coffea4bees/analysis/tests/cutflow_test.py \
    --inputFile $INPUT_DIR/testSignal_UL.coffea \
    --knownCounts coffea4bees/analysis/tests/known_Counts_Signal.yml
