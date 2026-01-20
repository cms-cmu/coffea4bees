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
JOB="analysis_systematics_test"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"


display_section_header "Modifying HH4b_signals.yml"
JOB_CONFIG=$OUTPUT_DIR/HH4b_signals_modified.yml
sed -e "s|condor_memory: 2GB|chunksize: 10000|g" \
    coffea4bees/analysis/metadata/HH4b_signals.yml \
    > $JOB_CONFIG
cat $JOB_CONFIG; echo

# Call the main run-analysis-processor.sh script
bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --dataset-metadata "coffea4bees/metadata/datasets_HH4b_Run2/" \
    --datasets "GluGluToHHTo4B_cHHH1" \
    --year "UL18" \
    --output-filename "test_systematics.coffea" \
    --output-subdir "$JOB" \
    --config $JOB_CONFIG \
    --additional-flags "--systematics others"
    # --additional-flags "--debug"
