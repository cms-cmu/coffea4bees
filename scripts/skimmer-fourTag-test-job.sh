#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

OUTPUT_DIR="${1:-"output"}/skimmer_fourTag_test_job"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi


display_section_header "Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/2024_.*/tmp\//" coffea4bees/skimmer/metadata/HH4b_fourTag.yml > ${OUTPUT_DIR}/tmp_fourTag.yml
else
    sed -e "s#base_.*#base_path: /builds/${CI_PROJECT_PATH}/${OUTPUT_DIR}/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/2024_.*/tmp\//" -e "s/T3_US_FNALLPC/T3_CH_PSI/" coffea4bees/skimmer/metadata/HH4b_fourTag.yml > ${OUTPUT_DIR}/tmp_fourTag.yml
fi
cat ${OUTPUT_DIR}/tmp_fourTag.yml

display_section_header "Running test processor"
python runner.py -s -p coffea4bees/skimmer/processor/skimmer_4b.py -c ${OUTPUT_DIR}/tmp_fourTag.yml -y UL18 -d data -op ${OUTPUT_DIR} -o picoaod_datasets_fourTag_data_test_UL18.yml -m coffea4bees/metadata/datasets_HH4b_Run2/ 

