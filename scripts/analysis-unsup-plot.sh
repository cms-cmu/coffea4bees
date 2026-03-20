#!/bin/bash
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
INPUT_DIR="$OUTPUT_BASE_DIR/analysis_unsup_test"
JOB="analysis_plot_unsup"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running plotting test"
run_command python coffea4bees/plots/makePlots_unsup.py \
    $INPUT_DIR/test_unsup.coffea \
    --doTest   \
    -o $OUTPUT_DIR/ \
    -m coffea4bees/plots/metadata/plotsAll_unsup.yml 

display_section_header "Checking if pdf files exist"
ls $OUTPUT_DIR/RunII/region_SR/mix_v0/v4j_mass.pdf
ls $OUTPUT_DIR/RunII/region_SR_vs_SB/mix_v0/v4j_mass.pdf
ls $OUTPUT_DIR/RunII/region_SR/mix_v0/quadJet_selected_lead_vs_subl_m.pdf
ls $OUTPUT_DIR/RunII/region_SR/data_3b_for_mixed/quadJet_selected_lead_vs_subl_m.pdf

