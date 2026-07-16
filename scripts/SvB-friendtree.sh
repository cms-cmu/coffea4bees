#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="SvB_friendtree"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

# Modify the config file
display_section_header "Modifying config"
JOB_CONFIG=$OUTPUT_DIR/HH4b_make_friend_SvB.yml
sed -e "s|/srv/output/tmp/|$OUTPUT_DIR|" \
    coffea4bees/analysis/metadata/HH4b_make_friend_SvB.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "GluGluToHHTo4B_cHHH1" \
    --year "UL18" \
    --output-filename "make_friend_SvB.coffea" \
    --output-subdir $JOB \
    --config $JOB_CONFIG \
    # --additional-flags "--debug"
