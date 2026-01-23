#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

OUTPUT_DIR="${1:-"output"}/sub_sample_dataset_make_dataset_all"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

display_section_header "Running test processor"
time python runner.py -s -p coffea4bees/skimmer/processor/sub_sample_MC.py -c coffea4bees/skimmer/metadata/sub_sampling_MC.yml -y UL17 UL18 UL16_preVFP UL16_postVFP  -d TTToHadronic TTToSemiLeptonic TTTo2L2Nu -op ${OUTPUT_DIR} -o picoaod_datasets_TT_pseudodata_Run2.yml -m coffea4bees/metadata/datasets_HH4b_Run2/
ls -R ${OUTPUT_DIR}

