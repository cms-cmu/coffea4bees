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
DATASETS=${DATASET:-"coffea4bees//metadata/datasets_HH4b_Run3_fourTag_2025_skims.yml"}
echo "Using datasets file: $DATASETS"


# Setup proxy if needed
setup_proxy 

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/mixeddata_cluster"
create_output_directory "$OUTPUT_DIR"



display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/make_hemi_library_4b_test.yml"
sed -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
    -e "s|subtract_ttbar.*|subtract_ttbar_with_weights: False|" \
    -e "s|run_SvB.*|run_SvB: False|" \
    coffea4bees/analysis/metadata/make_hemi_library_4b.yml > $JOB_CONFIG
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo

YEARS="2022_EE 2022_preEE 2023_BPix 2023_preBPix"

display_section_header "Running test mixed data clustering"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --processor "coffea4bees/analysis/processors/processor_make_hemi_library.py" \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "data" \
    --year "${YEARS}" \
    --output-filename "test_mixed_datasets_Run3.coffea" \
    --output-subdir "$JOB" \
    --config "$JOB_CONFIG" \
    --dataset-metadata "$DATASETS"
#    --additional-flags "--debug"

#    --year "UL17 UL18 UL16_preVFP UL16_postVFP" \    


