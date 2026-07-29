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

# Setup proxy if needed
setup_proxy 

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/test_mixeddata_make_dataset"
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/declustering_for_test.yml"
sed -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
    -e "s|\#max.*|maxchunks: 1|" \
    -e "s|\#test.*|test_files: 1|" \
    -e "s|workers:.*|workers: 1|" \
    -e "s|chunksize:.*|chunksize: 1000|" \
    -e "s|subtract_ttbar.*|subtract_ttbar_with_weights: True|" \
    coffea4bees/skimmer/metadata/mixeddata.yml > $JOB_CONFIG
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Running test processor make_mixed_data"
cmd=(python runner.py -s \
    -p coffea4bees/skimmer/processor/make_mixed_data.py \
    -c $JOB_CONFIG \
    --friends coffea4bees/metadata/friends_HH4b.yml \
    -y UL18   -d data  \
    -op $OUTPUT_DIR \
    -o picoaod_datasets_mixeddata_test_UL18.yml \
    -m $DATASETS)
time run_command "${cmd[@]}"

ls -R $OUTPUT_DIR
