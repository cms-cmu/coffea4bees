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
JOB="weights_trigger_analysis"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets/"}
echo "Using datasets file: $DATASETS"


	#-e "s|trigWeight: .*|trigWeight: $DATASETS/trigger_weights_TTTo2L2Nu_friends.json@@trigWeight|" \

display_section_header "Modifying config"
JOB_CONFIG=$OUTPUT_DIR/trigger_weights_HH4b.yml
sed     -e "s|apply_trigWeight: .*|apply_trigWeight: true|" \
    coffea4bees/analysis/metadata/HH4b_signals_Run3.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Running test processor"
time bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "TTTo2L2Nu TTToSemiLeptonic TTToHadronic" \
    --dataset-metadata "$DATASETS" \
    --year "2022_EE 2022_preEE 2023_BPix 2023_preBPix" \
    --output-filename "trigWeight_TTbar_wHLT_newJCM.coffea" \
    --output-subdir "$JOB" \
    --config $JOB_CONFIG \
    --no-test \
    --additional-flags "--condor"

##### Data
##display_section_header "Running test processor"
##time bash coffea4bees/scripts/run-analysis-processor.sh \
##    --output-base "$OUTPUT_BASE_DIR" \
##    --datasets "data" \
##    --dataset-metadata "$DATASETS" \
##    --year "2022_EE 2022_preEE 2023_BPix 2023_preBPix" \
##    --output-filename "trigWeight_data.coffea" \
##    --output-subdir "$JOB" \
##    --config $JOB_CONFIG \
##    --no-test \
##    --additional-flags "--condor"
##
