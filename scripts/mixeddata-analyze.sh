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
JOB="mixeddata_analyze"
INPUT_DIR="$OUTPUT_BASE_DIR/test_mixeddata_split_dataset"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"


display_section_header "Creating metadata"
echo """
datasets:
  mixeddata_4b:
    nSamples: 1
    UL18:
      picoAOD:
        files_template:
          - ${INPUT_DIR}/mixeddata_all_UL18A/picoAOD_mixed_vXXX.root
          - ${INPUT_DIR}/mixeddata_all_UL18B/picoAOD_mixed_vXXX.root
          - ${INPUT_DIR}/mixeddata_all_UL18C/picoAOD_mixed_vXXX.root
          - ${INPUT_DIR}/mixeddata_all_UL18D/picoAOD_mixed_vXXX.root

""" > $OUTPUT_DIR/datasets_mixeddata_test.yml; echo


display_section_header "Running test processor"
YEARS="UL18"
python runner.py  -o test_mixeddata.coffea -d    mixeddata_4b  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18  -op $OUTPUT_DIR -m $OUTPUT_DIR/datasets_mixeddata_test.yml -c coffea4bees/analysis/metadata/HH4b_mixed_data_new.yml --weights coffea4bees/metadata/weights_HH4b.yml

#python runner.py  -o histData.coffea -d    data  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18 UL17 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS

