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
DATASETS=${DATASET:-"coffea4bees//metadata/datasets_HH4b_Run3/"}
echo "Using datasets file: $DATASETS"

# Setup proxy if needed
setup_proxy 

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/jetDeclustering_make_dataset_Run3"
create_output_directory "$OUTPUT_DIR"

#    -e "s|\#max.*|maxchunks: 1|" \
#    -e "s|\#test.*|test_files: 1|" \
#    -e "s|workers:.*|workers: 1|" \
#    -e "s|chunksize:.*|chunksize: 1000|" \

for i in {0..0}; do
  echo "Doing splitting job $i"

  ### # Test
  ### display_section_header "Changing metadata"
  ### JOB_CONFIG="$OUTPUT_DIR/declustering_v${i}_teset.yml"
  ### sed -e "s|declustering_rand_seed.*|declustering_rand_seed: $i|" \
  ###     -e "s|\#max.*|maxchunks: 1|" \
  ###     -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
  ###     -e "s|\#test.*|test_files: 1|" \
  ###     -e "s|workers:.*|workers: 1|" \
  ###     -e "s|chunksize:.*|chunksize: 1000|" \
  ###     coffea4bees/skimmer/metadata/declustering_Run3.yml > $JOB_CONFIG
  ### cat $JOB_CONFIG; echo
  ### display_section_header "Running test processor make_declustered_data_4b job $i"
  ### 			
  ### cmd=(python runner.py -s \
  ### 	      -p coffea4bees/skimmer/processor/make_declustered_data_4b.py \
  ### 	      -c $JOB_CONFIG \
  ### 	      -y 2022_EE   -d data  \
  ### 	      -op $OUTPUT_DIR \
  ### 	      -o test_picoaod_datasets_syn_2022_EE_subTT_v${i}.yml \
  ### 	      -m $DATASETS \
  ###     )


  # All
  display_section_header "Changing metadata"
  JOB_CONFIG="$OUTPUT_DIR/declustering_v${i}.yml"
  sed -e "s|declustering_rand_seed.*|declustering_rand_seed: $i|" \
      coffea4bees/skimmer/metadata/declustering_Run3.yml > $JOB_CONFIG
  cat $JOB_CONFIG; echo
  display_section_header "Running test processor make_declustered_data_4b job $i"
  			
  cmd=(python runner.py -s \
  	      -p coffea4bees/skimmer/processor/make_declustered_data_4b.py \
  	      -c $JOB_CONFIG \
  	      -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix  -d data  \
  	      -op $OUTPUT_DIR \
  	      -o picoaod_datasets_syn_Run3_subTT_v${i}.yml \
  	      -m $DATASETS \
  	      --condor
      )
  
  time run_command "${cmd[@]}"
done
ls -R $OUTPUT_DIR


