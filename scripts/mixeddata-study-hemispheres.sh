#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

JOB="test_study_hemispheres"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
#!/bin/bash

echo "############### Running hemisphere mixing test"
#python coffea4bees/hemisphere_mixing/tests/test_mixing.py 

python coffea4bees/hemisphere_mixing/study_hemispheres.py --hemifiles coffea4bees/skimmer/metadata/hemisphere_library_test.yml --year UL18  --threshold 10 --output_path $OUTPUT_DIR
