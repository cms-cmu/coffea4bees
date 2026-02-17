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
JOB="weights_trigger_friendtree_Run3"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
DATASETS=${DATASET_RUN3:-"coffea4bees/metadata/datasets_HH4b_Run3/"}
echo "Using datasets file: $DATASETS"

# Modify the config file
JOB_CONFIG=coffea4bees/analysis/metadata/trigger_weights_Run3.yml
cat $JOB_CONFIG; echo

#    --datasets "TTTo2L2Nu" \
#    --datasets "TTToSemiLeptonic" \
bash coffea4bees/scripts/run-analysis-processor.sh \
    --processor "coffea4bees/analysis/processors/processor_trigger_weights.py" \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "TTToHadronic" \
    --dataset-metadata "$DATASETS" \
    --year "2022_EE 2022_preEE 2023_BPix 2023_preBPix" \
    --output-filename "trigger_weights_TTToHadronic_friends.json" \
    --output-subdir $JOB \
    --config $JOB_CONFIG \
    --no-test \
    --additional-flags "--condor"

