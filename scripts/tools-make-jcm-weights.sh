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
JOB="tools_make_jcm_weights"
INPUT_DIR=$OUTPUT_BASE_DIR/tools_merge_test
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

# Fixed-t config: the production default (nominal_jcm_config.yml) floats
# threeTightTagFraction, which the small CI sample cannot constrain and which
# the reference yml files in analysis/tests/ were not produced with.
JCM_CONFIG=coffea4bees/analysis/jcm_tools/metadata/test_jcm_config.yml

display_section_header "Running JCM weights test"
display_section_header "Running ROOT test"
run_command python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
    -o $OUTPUT_DIR/testJCM_ROOT   \
    -r SB --ROOTInputs \
    --jcm_config $JCM_CONFIG \
    --i coffea4bees/analysis/tests/HistsFromROOTFile.coffea

display_section_header "Running Coffea test"
run_command python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
    -o $OUTPUT_DIR/testJCM_Coffea \
    -r SB \
    --jcm_config $JCM_CONFIG \
    -i $INPUT_DIR/test.coffea

display_section_header "Running weights comparison test"
run_command python coffea4bees/analysis/tests/make_weights_test.py \
    --path $OUTPUT_DIR

