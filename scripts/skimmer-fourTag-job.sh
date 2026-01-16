#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

OUTPUT_DIR="${1:-"output"}/skimmer_fourTag_job"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

display_section_header "Running test processor"
#time python runner.py -s -p coffea4bees/skimmer/processor/skimmer_4b.py -c coffea4bees/skimmer/metadata/HH4b_fourTag.yml -y UL18 UL17 UL16_preVFP UL16_postVFP -d data -op coffea4bees/skimmer/metadata/ -o picoaod_datasets_fourTag_data_Run2.yml -m coffea4bees/metadata/datasets_HH4b.yml
time python runner.py -s -p coffea4bees/skimmer/processor/skimmer_4b.py -c coffea4bees/skimmer/metadata/HH4b_fourTag.yml -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix -d data -op ${OUTPUT_DIR} -o picoaod_datasets_fourTag_data_Run3.yml -m coffea4bees/metadata/datasets_HH4b_Run3/ 
#ls -R skimmer/

