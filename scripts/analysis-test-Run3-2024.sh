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
JOB="analysis_test_Run3_2024"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
#DATASETS=${DATASET_RUN3:-"coffea4bees//metadata/datasets_HH4b_Run3_2025_Run3_skims.yml"}
DATASETS=${DATASET_RUN3:-"coffea4bees/metadata/datasets_HH4b_Run3/"}
echo "Using datasets file: $DATASETS"

display_section_header "Running test processor"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "data TTTo2L2Nu TTToSemiLeptonic TTToHadronic GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00" \
    --dataset-metadata "$DATASETS" \
    --year "2024" \
    --output-filename "test.coffea" \
    --output-subdir "$JOB" \
    --config coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_2024.yml \
    # --additional-flags "--debug"
