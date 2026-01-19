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
OUTPUT_DIR="$OUTPUT_BASE_DIR/skimmer_test"
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/HH4b.yml"
sed -e "s|base_path.*|base_path: $OUTPUT_DIR|" \
    -e "s|\#maxchunks.*|maxchunks: 5|" \
    -e "s|\#test.*|test_files: 1|" \
    -e "s|2024_.*|tmp/|" \
    coffea4bees/skimmer/metadata/HH4b.yml > $JOB_CONFIG
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Changing datasets"
#nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/92D0BDF3-91AE-514F-88B5-8F591450B8AD.root"
#sed -e "s#/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" coffea4bees/metadata/datasets_HH4b_Run2/ > $OUTPUT_DIR/datasets_HH4b.yml
#python runner.py -s -p coffea4bees/skimmer/processor/skimmer_4b.py -c $OUTPUT_DIR/datasets_HH4b.yml -y UL18 -d TTToSemiLeptonic -op $OUTPUT_DIR -o picoaod_datasets_TTToSemiLeptonic_UL18.yml -m $OUTPUT_DIR/datasets_HH4b.yml  -t 
nanoAOD_file="root://cmseos.fnal.gov//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/3F95108D-84D2-CD4D-A0D2-324A7D15E691.root"
sed -e "s|/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*|[ '${nanoAOD_file}' ]|" coffea4bees/metadata/datasets_HH4b_Run2/GluGluToHHTo4B.yml > $OUTPUT_DIR/datasets_HH4b.yml

display_section_header "Running test processor"
cmd=(python runner.py -s \
    -p coffea4bees/skimmer/processor/skimmer_4b.py \
    -c $JOB_CONFIG \
    -y UL18 -d GluGluToHHTo4B_cHHH0 \
    -op $OUTPUT_DIR \
    -o picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml \
    -m $OUTPUT_DIR/datasets_HH4b.yml  -t)
run_command "${cmd[@]}"
ls -R $OUTPUT_DIR

