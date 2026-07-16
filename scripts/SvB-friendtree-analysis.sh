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
JOB="SvB_friendtree_analysis"
INPUT_DIR="$OUTPUT_BASE_DIR/SvB_friendtree"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Modifying config"
JOB_CONFIG=$OUTPUT_DIR/HH4b_signals.yml
sed -e "s|# SvB: \"metadata.*|SvB: $INPUT_DIR/make_friend_SvB.json@@SvB|" \
    -e "s|# SvB_MA: \"metadata.*|SvB_MA: $INPUT_DIR/make_friend_SvB.json@@SvB_MA|" \
    -e "s|SvB   : 'coffea/.*||"  \
    -e "s|SvB_MA: 'coffea/.*||" \
    coffea4bees/analysis/metadata/HH4b_signals.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

display_section_header "Running test processor"
time bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "GluGluToHHTo4B_cHHH1" \
    --year "UL18" \
    --output-filename "test_SvB_friend.coffea" \
    --output-subdir "$JOB" \
    --config $JOB_CONFIG \
    # --additional-flags "--debug"
    