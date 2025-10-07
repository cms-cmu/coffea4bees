#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

OUTPUT_DIR="${1:-"output"}/sub_sample_dataset_analyze"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ $(hostname) = *fnal* ]]; then
    echo "No changing files"
else
    display_section_header "Modifying previous dataset file (to read local files)"
    ls -lR coffea4bees/skimmer/
    cat coffea4bees/skimmer/metadata/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml
    echo "TEST"
    pwd
    echo ${CI_PROJECT_PATH}
    sed "s|\/builds/$CI_PROJECT_PATH\/python\/||g"  coffea4bees/skimmer/metadata/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml > ${OUTPUT_DIR}/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml
    echo "NEW"
    cat ${OUTPUT_DIR}/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml
fi


# display_section_header "Changing metadata"
# sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/"  coffea4bees/analysis/metadata/HH4b.yml > ${OUTPUT_DIR}/tmp.yml
# cat coffea4bees/analysis/metadata/tmp.yml
#display_section_header "Running test processor"
#time python runner.py -o test_synthetic_data_test.coffea -d data -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18  -op ${OUTPUT_DIR} -c coffea4bees/analysis/metadata/HH4b_synthetic_data.yml -m coffea4bees/metadata/datasets_synthetic_seed17.yml


display_section_header "Running test processor "
# python src/tools/merge_yaml_datasets.py -m coffea4bees/metadata/datasets_HH4b.yml -f ${OUTPUT_DIR}/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml  -o ${OUTPUT_DIR}/datasets_TT_pseudodata_test.yml
# python src/tools/merge_yaml_datasets.py -m coffea4bees/metadata/datasets_synthetic_seed17.yml -f ${OUTPUT_DIR}/picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_Run2_seed17.yml -o ${OUTPUT_DIR}/datasets_synthetic_seed17.yml
#cat ${OUTPUT_DIR}/datasets_synthetic_test.yml
time python runner.py -o test_TT_pseudodata_datasets.coffea -d ps_data_TTToHadronic -p coffea4bees/analysis/processors/processor_HH4b.py -y UL18  -op ${OUTPUT_DIR} -c coffea4bees/analysis/metadata/HH4b_ps_data.yml -m coffea4bees/metadata/datasets_TT_pseudodata_test.yml

