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
JOB="analysis_test_lowpt_data_Run2"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

# Modify the config file
display_section_header "Modifying config"
JOB_CONFIG="coffea4bees/analysis/metadata/HH4b_lowpt_2024_v2.yml"
# JOB_CONFIG=$OUTPUT_DIR/HH4b.yml
# sed -e "s|hist_cuts: .*|hist_cuts: []|" \
    # coffea4bees/analysis/metadata/HH4b.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Running analysis processor for background datasets"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --processor coffea4bees/analysis/processors/processor_HH4b_lowpt.py \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "data" \
    --year "UL17 UL18 UL16_preVFP UL16_postVFP" \
    --friends "coffea4bees/metadata/datasets/2024_v2/friends_HH4b_lowpt.yml" \
    --output-filename "test_data_lowpt.coffea" \
    --output-subdir "$JOB" \
    --config $JOB_CONFIG \
    --dataset-metadata "coffea4bees/metadata/datasets/2024_v2/" \
    # --additional-flags "--slurm --debug"
