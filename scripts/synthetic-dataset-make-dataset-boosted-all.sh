#!/bin/bash
# Source common functions
source "src/scripts/common.sh"


display_section_header "Checking and creating output/skimmer directory"
OUTPUT_DIR="${1:-"output"}/synthetic_dataset_make_dataset"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

display_section_header "Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/"  coffea4bees/skimmer/metadata/declustering_boosted.yml > $OUTPUT_DIR/declustering_boosted_for_test.yml
elif [[ $(hostname) = *rogue* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/${OUTPUT_DIR}\/#"   coffea4bees/skimmer/metadata/declustering_boosted.yml > $OUTPUT_DIR/declustering_boosted_for_all.yml
else
    sed -e "s#base_.*#base_path: /builds/${CI_PROJECT_PATH}/${OUTPUT_DIR}/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/T3_US_FNALLPC/T3_CH_PSI/"   coffea4bees/skimmer/metadata/declustering_boosted.yml > $OUTPUT_DIR/declustering_boosted_for_test.yml
fi
#cat $OUTPUT_DIR/declustering_boosted_for_test.yml

echo "Hostname"
echo $(hostname)

new_seed=0
echo ${new_seed}
sed -e "s/declustering_rand_seed:.*/declustering_rand_seed: $new_seed/" ${OUTPUT_DIR}/declustering_boosted_for_all.yml > ${OUTPUT_DIR}/declustering_boosted_seed_${new_seed}.yml

display_section_header "Running test processor"
time python runner.py -s -p coffea4bees/skimmer/processor/make_declustered_data_boosted_4b.py -c ${OUTPUT_DIR}/declustering_boosted_seed_${new_seed}.yml -y UL18  -d data GluGluToHHTo4B_cHHH1   -op $OUTPUT_DIR -o picoaod_datasets_declustered_boosted_test_UL18.yml -m coffea4bees/metadata/datasets_HH4b_2024_v2_boosted.yml




ls -R $OUTPUT_DIR


