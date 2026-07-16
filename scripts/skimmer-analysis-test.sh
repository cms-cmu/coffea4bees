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
JOB="skimmer_analysis_test"
INPUT_DIR=$OUTPUT_BASE_DIR/skimmer_test
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Setting up picoaod dataset file"
PICO_YAML=$OUTPUT_DIR/picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml
cp $INPUT_DIR/picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml $PICO_YAML
# [[ $(hostname) = *runner* ]] && sed -i "s|/builds/${CI_PROJECT_PATH}/||g" $PICO_YAML
cat $PICO_YAML; echo 


display_section_header "Modifying dataset file with skimmer ci output"
run_command python src/tools/merge_yaml_datasets.py \
    -m $INPUT_DIR/datasets_HH4b.yml \
    -f $PICO_YAML \
    -o $OUTPUT_DIR/datasets_HH4b.yml
cat $OUTPUT_DIR/datasets_HH4b.yml; echo

display_section_header "Changing metadata"
JOB_CONFIG=$INPUT_DIR/HH4b.yml
sed -e "s/apply_FvT.*/apply_FvT: false/" \
    -e "s/apply_trig.*/apply_trigWeight: false/" \
    -e "s/run_SvB.*/run_SvB: false/" \
    -e "s/top_reco.*/top_reconstruction: 'fast'/"  \
    coffea4bees/analysis/metadata/HH4b.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Running test processor"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "GluGluToHHTo4B_cHHH0" \
    --year "UL18" \
    --output-filename "test_skimmer.coffea" \
    --output-subdir "$JOB" \
    --dataset-metadata $OUTPUT_DIR/datasets_HH4b.yml \
    --config $JOB_CONFIG \
    --no-test
    # --additional-flags "--debug"
