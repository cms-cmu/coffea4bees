#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees//metadata/datasets"}
echo "Using datasets file: $DATASETS"

OUTPUT_DIR="${1:-"output"}/analysis_mixed_new_Run3"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

display_section_header "Running test processor"
YEARS="2022_EE 2022_preEE 2023_BPix 2023_preBPix"
#YEARS="UL18"
python runner.py  -o histMixedDataRun3_pz.coffea -d    mixeddata_4b_pz  -p coffea4bees/analysis/processors/processor_HH4b.py -y $YEARS  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_mixed_data_new.yml --condor

#python runner.py  -o histData.coffea -d    data  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18 UL17 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS

