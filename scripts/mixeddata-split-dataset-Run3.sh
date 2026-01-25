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
OUTPUT_DIR="$OUTPUT_BASE_DIR/mixeddata_split_dataset_Run3"
create_output_directory "$OUTPUT_DIR"

#    -e "s|\#max.*|maxchunks: 1|" \
#    -e "s|\#test.*|test_files: 1|" \
#    -e "s|workers:.*|workers: 1|" \
#    -e "s|chunksize:.*|chunksize: 1000|" \
# -e "s|base_path.*|base_path: $OUTPUT_DIR|" \


for i in {0..0}; do
  echo "Doing splitting job $i"

  display_section_header "Changing metadata"
  JOB_CONFIG="$OUTPUT_DIR/split_mixeddata_v${i}.yml"
  sed -e "s|mixed_subsample.*|mixed_subsample: $i|" \
      -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
      coffea4bees/skimmer/metadata/split_mixeddata_Run3.yml > $JOB_CONFIG
  [[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
  cat $JOB_CONFIG; echo
  display_section_header "Running test processor split_mixed_data job $i"

  cmd=(python runner.py -s \
	    -p coffea4bees/skimmer/processor/split_mixed_data.py \
	    -c $JOB_CONFIG \
	    -y 2022_EE   -d mixeddata_all  \
	    -op $OUTPUT_DIR \
	    -o picoaod_datasets_split_mixeddata_2022_EE.yml \
	    -m $DATASETS)
  time run_command "${cmd[@]}"

  
done




ls -R $OUTPUT_DIR
