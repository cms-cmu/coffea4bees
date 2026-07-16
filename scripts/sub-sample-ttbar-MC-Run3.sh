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
JOB="sub_sample_MC_Run3"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets/"}
echo "Using datasets file: $DATASETS"

display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/test_sub_sampling_MC_Run3.yml"
sed -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
      coffea4bees/skimmer/metadata/sub_sampling_MC_Run3.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

# 2022_preEE 2023_BPix 2023_preBPix
#            -d  TTTo2L2Nu TTToSemiLeptonic TTToHadronic \
cmd=(python runner.py -s \
	    -p coffea4bees/skimmer/processor/sub_sample_MC.py \
	    -c $JOB_CONFIG \
	    -y 2022_EE  \
            -d TTTo2L2Nu \
	    -op $OUTPUT_DIR \
	    -o picoaod_datasets_sub_sample_TTTo2L2Nu_2022_EE.yml \
	    -m $DATASETS \
	    #	    --condor
      )
time run_command "${cmd[@]}"

