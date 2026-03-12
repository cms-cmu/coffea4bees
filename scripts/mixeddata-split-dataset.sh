#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Setup proxy if needed
setup_proxy 

# Create output directory
OUTPUT_DIR="$OUTPUT_BASE_DIR/test_mixeddata_split_dataset"
create_output_directory "$OUTPUT_DIR"

INPUT_DIR="$OUTPUT_BASE_DIR/test_mixeddata_make_dataset"

display_section_header "Creating metadata"
echo """
datasets:
  mixeddata_all:
    UL18:
      picoAOD:
        A: 
          files:
          - ${INPUT_DIR}/data_UL18A/picoAOD_mixed_all.root
        B: 
          files:
          - ${INPUT_DIR}/data_UL18B/picoAOD_mixed_all.root
        C: 
          files:
          - ${INPUT_DIR}/data_UL18C/picoAOD_mixed_all.root
        D: 
          files:
          - ${INPUT_DIR}/data_UL18D/picoAOD_mixed_all.root

""" > $OUTPUT_DIR/datasets_mixeddata_test.yml; echo



#    -e "s|\#max.*|maxchunks: 1|" \
#    -e "s|\#test.*|test_files: 1|" \
#    -e "s|workers:.*|workers: 1|" \
#    -e "s|chunksize:.*|chunksize: 1000|" \
# -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
			
i=0
echo "Doing splitting job $i"

display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/split_mixeddata_v${i}.yml"
sed -e "s|mixed_subsample.*|mixed_subsample: $i|" \
    -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
    coffea4bees/skimmer/metadata/split_mixeddata.yml > $JOB_CONFIG
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo
display_section_header "Running test processor split_mixed_data job $i"

cmd=(python runner.py -s \
          -p coffea4bees/skimmer/processor/split_mixed_data.py \
          -c $JOB_CONFIG \
          -y UL18   -d mixeddata_all  \
          -op $OUTPUT_DIR \
          -o picoaod_datasets_split_mixeddata_UL18_v$i.yml \
          -m $OUTPUT_DIR/datasets_mixeddata_test.yml)
time run_command "${cmd[@]}"






ls -R $OUTPUT_DIR
