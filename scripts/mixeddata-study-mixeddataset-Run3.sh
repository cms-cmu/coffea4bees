#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets_HH4b_Run2/"}
echo "Using datasets file: $DATASETS"


# Create output directory
JOB="mixeddata_cluster"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"


YEARS="2022_EE 2022_preEE 2023_BPix 2023_preBPix"

display_section_header "Running test processor"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --processor "coffea4bees/analysis/processors/processor_study_mixed_data.py" \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "mixeddata_all" \
    --year "${YEARS}" \
    --output-filename "study_mixed_datasets_all.coffea" \
    --output-subdir "$JOB" \
    --config coffea4bees/analysis/metadata/study_mixed_data.yml \
    --dataset-metadata $DATASETS \
    --no-test 
#    --additional-flags "--debug"


