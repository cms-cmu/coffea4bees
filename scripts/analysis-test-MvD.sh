#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse flags
DO_TEST=""
OUTPUT_BASE_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --do-test)
            DO_TEST="1"
            shift
            ;;
        --output-base)
            OUTPUT_BASE_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "${OUTPUT_BASE_ARGS[@]}")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="analysis_MvD"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
DATASETS=${DATASET_RUN3:-"coffea4bees/metadata/datasets_HH4b_Run3/"}
echo "Using datasets file: $DATASETS"

#2022_preEE 2023_BPix 2023_preBPix

display_section_header "Running MvD processor"
if [ -n "$DO_TEST" ]; then
    time bash coffea4bees/scripts/run-analysis-processor.sh \
        --output-base "$OUTPUT_BASE_DIR" \
        --datasets "mixeddata_all data" \
        --dataset-metadata "$DATASETS" \
        --year "2022_EE" \
        --output-filename "test.coffea" \
        --output-subdir "$JOB" \
        --config coffea4bees/analysis/metadata/HH4b_MvD.yml
else
    time bash coffea4bees/scripts/run-analysis-processor.sh \
        --output-base "$OUTPUT_BASE_DIR" \
        --datasets "mixeddata_all data" \
        --dataset-metadata "$DATASETS" \
        --year "2022_EE 2022_preEE 2023_BPix 2023_preBPix" \
        --output-filename "analysis_MvD.coffea" \
        --output-subdir "$JOB" \
        --config coffea4bees/analysis/metadata/HH4b_MvD.yml \
        --no-test \
        --additional-flags "--condor"
fi
