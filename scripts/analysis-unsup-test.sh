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
JOB="analysis_unsup_test"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running test processor"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --processor "coffea4bees/analysis/processors/processor_unsup.py" \
    --output-base "$OUTPUT_BASE_DIR" \
    --dataset-metadata coffea4bees/analysis/metadata/unsup4b.yml \
    --datasets "mixeddata data_3b_for_mixed TTToHadronic TTToSemiLeptonic TTTo2L2Nu" \
    --year "UL17 UL18 UL16_preVFP UL16_postVFP" \
    --output-filename "test_unsup.coffea" \
    --output-subdir "$JOB" \
    --config coffea4bees/analysis/metadata/unsup4b.yml \
    # --additional-flags "--debug"
    