#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets_HH4b_Run2/"}
echo "Using datasets file: $DATASETS"

OUTPUT_DIR="${1:-"output"}/analysis_mixed_new"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

display_section_header "Running test processor"
YEARS="UL18 UL17 UL16_preVFP UL16_postVFP"
#YEARS="UL18"
python runner.py  -o histMixedData.coffea -d    mixeddata_4b  -p coffea4bees/analysis/processors/processor_HH4b.py -y $YEARS  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_mixed_data_new.yml

#python runner.py  -o histData.coffea -d    data  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18 UL17 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS

