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
INPUT_DIR="$OUTPUT_BASE_DIR/analysis_truthStudy_test"
JOB="analysis_truthStudy_plot"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running plotting test"
run_command python  coffea4bees/plots/makePlotsTruthStudy.py \
    $INPUT_DIR/testTruth.coffea \
    -m coffea4bees/plots/metadata/plotsSignal.yml \
    --out ${OUTPUT_DIR}
display_section_header "Checking if pdf files exist"
ls ${OUTPUT_DIR}/RunII/pass4GenBJets00/fourTag/SR/otherGenJet00_pt.pdf

