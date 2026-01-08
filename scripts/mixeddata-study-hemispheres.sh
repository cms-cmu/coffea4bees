#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

INPUT_DIR="$OUTPUT_BASE_DIR"
JOB="test_study_hemispheres"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB

display_section_header "Changing hemisphere library metadata"
HEMI_LIB="$OUTPUT_DIR/hemisphere_library_for_test.yml"
sed -e "s|output|$INPUT_DIR|" \
    coffea4bees/skimmer/metadata/hemisphere_library_test.yml > $HEMI_LIB
[[ $(hostname) = *runner* ]] && sed -i "s|T3_US_FNALLPC|T3_CH_PSI|" $HEMI_LIB
cat $HEMI_LIB; echo



echo "############### Running hemisphere mixing test"
#python coffea4bees/hemisphere_mixing/tests/test_mixing.py 

python coffea4bees/hemisphere_mixing/study_hemispheres.py --hemifiles $HEMI_LIB  --year UL18  --threshold 10 --output_path $OUTPUT_DIR
