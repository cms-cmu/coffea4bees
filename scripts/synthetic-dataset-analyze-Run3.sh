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
JOB="synthetic_dataset_analyze_Run3"
INPUT_DIR="$OUTPUT_BASE_DIR/synthetic_dataset_make_dataset_Run3"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
echo """
datasets:
  synthetic_data:
    nSamples: 1
    2023_BPix:
      picoAOD:
        files_template:
          - ${INPUT_DIR}/data_2023_BPixD1/picoAOD_seed5.root
          - ${INPUT_DIR}/data_2023_BPixD2/picoAOD_seed5.root
""" > $OUTPUT_DIR/datasets_synthetic_test_Run3.yml
cat $OUTPUT_DIR/datasets_synthetic_test_Run3.yml; echo


display_section_header "Running test processor "
time bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "synthetic_data" \
    --year "2023_BPix" \
    --output-filename "test_synthetic_datasets.coffea" \
    --output-subdir "$JOB" \
    --dataset-metadata "$OUTPUT_DIR/datasets_synthetic_test_Run3.yml" \
    --config coffea4bees/analysis/metadata/HH4b_synthetic_data.yml \
    --no-test
    # --additional-flags "--debug"
