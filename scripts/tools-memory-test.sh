#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

setup_proxy

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/tools_memory_test"
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
DATASETS="coffea4bees/metadata/datasets_HH4b_v1p2.yml"
echo "Using datasets file: $DATASETS"

run_command python src/scripts/memory/memory_test.py \
    --threshold 1600 \
    -o $OUTPUT_DIR/mprofile_ci_test \
    --script runner.py \
        -o test.coffea -t \
        -d GluGluToHHTo4B_cHHH1 \
        -p coffea4bees/analysis/processors/processor_HH4b.py \
        -y UL18 \
        -op ${OUTPUT_DIR} \
        -m $DATASETS \
        -c coffea4bees/analysis/metadata/HH4b_signals.yml
