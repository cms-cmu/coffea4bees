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
JOB="mixeddata_cluster"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running test processor"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --processor "coffea4bees/analysis/processors/processor_study_mixed_data.py" \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "mixeddata_all" \
    --year "UL18" \
    --output-filename "test_mixed_datasets_all.coffea" \
    --output-subdir "$JOB" \
    --config coffea4bees/analysis/metadata/study_mixed_data.yml \
    --no-test \
#    --additional-flags "--debug"


