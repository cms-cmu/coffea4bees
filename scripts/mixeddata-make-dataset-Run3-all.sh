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
DATASETS=${DATASET:-"coffea4bees//metadata/datasets_HH4b_Run3_2025_Run3_skims.yml"}
echo "Using datasets file: $DATASETS"

# Setup proxy if needed
setup_proxy 

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/mixeddata_make_dataset_Run3_all"
create_output_directory "$OUTPUT_DIR"

#    -e "s|\#max.*|maxchunks: 1|" \
#    -e "s|\#test.*|test_files: 1|" \
#    -e "s|workers:.*|workers: 1|" \
#    -e "s|chunksize:.*|chunksize: 1000|" \


display_section_header "Changing metadata"
JOB_CONFIG=coffea4bees/skimmer/metadata/mixeddata_Run3.yml #"$OUTPUT_DIR/declustering_for_test.yml"
#sed -e "s|chunksize:.*|chunksize: 100000|" \
#    coffea4bees/skimmer/metadata/mixeddata.yml > $JOB_CONFIG
#[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo



display_section_header "Running processor make_mixed_data"
cmd=(python runner.py -s \
    -p coffea4bees/skimmer/processor/make_mixed_data.py \
    -c $JOB_CONFIG \
    -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix  -d data  \
    -op $OUTPUT_DIR \
    -o picoaod_datasets_mixeddata_Run3.yml \
    -m $DATASETS \
    --condor)
time run_command "${cmd[@]}"

ls -R $OUTPUT_DIR
