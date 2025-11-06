#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="analysis_signals_test"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

# Modify the config file
display_section_header "Modifying config"
JOB_CONFIG=coffea4bees/analysis/metadata/HH4b_signals.yml
# JOB_CONFIG=$OUTPUT_DIR/HH4b_signals.yml
# sed -e "s|hist_cuts: .*|hist_cuts: [ passPreSel, passSvB, failSvB ]|" \
    # coffea4bees/analysis/metadata/HH4b_signals.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

# Call the main analysis_test.sh script with Run3-specific parameters
bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "GluGluToHHTo4B_cHHH1 ZZ4b ggZH4b ZH4b" \
    --year "UL17 UL18 UL16_preVFP UL16_postVFP" \
    --output-filename "test_signal.coffea" \
    --output-subdir "$JOB" \
    --config $JOB_CONFIG \
    # --additional-flags "--debug"
