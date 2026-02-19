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
JOB="analysis_test_Run3"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
#DATASETS=${DATASET:-"coffea4bees//metadata/datasets_HH4b_Run3_2025_Run3_skims.yml"}
#DATASETS=${DATASET:-"coffea4bees//metadata/datasets_HH4b_Run3_fourTag_2025_skims.yml"}
DATASETS=${DATASET:-"coffea4bees//metadata/datasets_HH4b_Run3/"}
echo "Using datasets file: $DATASETS"
#

#    --datasets "data TTToHadronic TTToSemiLeptonic TTTo2L2Nu" \

display_section_header "Running test processor"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "data " \
    --dataset-metadata "$DATASETS" \
    --year "2022_EE 2022_preEE 2023_BPix 2023_preBPix" \
    --output-filename "allRun3Data_wSvB.coffea" \
    --output-subdir "$JOB" \
    --config coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml \
    --no-test \
    --additional-flags "--condor"
#    --output-filename "test.coffea" \
