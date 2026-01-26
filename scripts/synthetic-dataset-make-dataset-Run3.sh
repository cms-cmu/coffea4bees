#!/bin/bash
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
# DATASETS=${DATASET_RUN3:-"coffea4bees/metadata/datasets_synthetic_test_Run3.yml"}
DATASETS="coffea4bees/metadata/datasets_synthetic_test_Run3.yml"
echo "Using datasets file: $DATASETS"

# Setup proxy if needed
setup_proxy 

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/synthetic_dataset_make_dataset_Run3"
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/declustering_for_test.yml"
sed -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
    -e "s|\#max.*|maxchunks: 1|" \
    -e "s|\#test.*|test_files: 1|" \
    -e "s|workers:.*|workers: 1|" \
    -e "s|chunksize:.*|chunksize: 1000|" \
    -e "s|subtract_ttbar.*|subtract_ttbar_with_weights: False|" \
    coffea4bees/skimmer/metadata/declustering_noTT_subtraction.yml > $JOB_CONFIG
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Running test processor"
cmd=(python runner.py -s \
    -p coffea4bees/skimmer/processor/make_declustered_data_4b.py \
    -c $JOB_CONFIG \
    -y 2023_BPix  -d data  \
    -op $OUTPUT_DIR \
    -o picoaod_datasets_declustered_test_2023_BPix.yml \
    -m $DATASETS)
time run_command "${cmd[@]}"
# time python runner.py -s -p coffea4bees/skimmer/processor/make_declustered_data_4b.py -c $OUTPUT_DIR/declustering_for_test.yml -y UL18  -d GluGluToHHTo4B_cHHH1 -op $OUTPUT_DIR -o picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_test_UL18.yml -m coffea4bees/metadata/datasets_HH4b_Run2/
ls -R $OUTPUT_DIR


