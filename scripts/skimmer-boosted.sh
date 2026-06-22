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
OUTPUT_DIR="$OUTPUT_BASE_DIR/skimmer_boosted"
create_output_directory "$OUTPUT_DIR"

display_section_header "Changing metadata"
JOB_CONFIG="$OUTPUT_DIR/HH4b_boosted.yml"
sed -e "s|base_path: .*|base_path: ${OUTPUT_DIR}|" \
    -e "s|\#max.*|maxchunks: 5|" \
    -e "s|\#test.*|test_files: 1|" \
    -e "s|2024_.*|tmp\/|" \
    -e "s|PFNano|nanoAOD|g" \
    -e "s|picoPFnano_boosted|picoAOD_boosted|g" \
    coffea4bees/skimmer/metadata/HH4b_boosted.yml > $JOB_CONFIG
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Changing datasets"
nanoAOD_file="root://cmseos.fnal.gov//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/3F95108D-84D2-CD4D-A0D2-324A7D15E691.root"

echo """
datasets:
    GluGluToHHTo4B_cHHH0:
        UL18:
            nanoAOD:
                - "${nanoAOD_file}"
            picoAOD:
                count: 1000000
                files:
                    - dummy.root
                outliers:
                    - 334929
                saved_events: 199653
                sumw: 26757.179077148438
                sumw2: 919.0958442687988
                sumw2_diff: 1192990.2674102783
                sumw2_raw: 1193909.4220537543
                sumw_diff: 1453.1521577835083
                sumw_raw: 28210.3267947
                total_events: 1000000
        xs: 0.03077*0.5824**2
""" > ${OUTPUT_DIR}/datasets_HH4b.yml
cat ${OUTPUT_DIR}/datasets_HH4b.yml; echo

display_section_header "Running test processor"
cmd=(python runner.py -s \
    -p coffea4bees/skimmer/processor/skimmer_4b_boosted.py \
    -c $OUTPUT_DIR/HH4b_boosted.yml \
    -y UL18 -d GluGluToHHTo4B_cHHH0 \
    -op $OUTPUT_DIR \
    -o picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml \
    -m $OUTPUT_DIR/datasets_HH4b.yml  -t) 
run_command "${cmd[@]}"

ls -R $OUTPUT_DIR

