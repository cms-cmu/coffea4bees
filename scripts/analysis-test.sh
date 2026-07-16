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
JOB="analysis_test"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

# Modify the config file
display_section_header "Modifying config"
JOB_CONFIG=$OUTPUT_DIR/HH4b.yml
sed -e "s|hist_cuts: .*|hist_cuts: []|" \
    coffea4bees/analysis/metadata/HH4b.yml > $JOB_CONFIG
cat $JOB_CONFIG; echo

### Temporary fix for CI tests
display_section_header "Temporary Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets"}
# 1. Merge files, add 'datasets:' at the top and indent existing lines by 2 spaces
(cat "$DATASETS/data.yml"; echo; cat "$DATASETS/TT.yml") | \
sed -e 's/^/  /' -e '1s/^/datasets:\n/' > $OUTPUT_DIR/datasets_temp.yml

# 2. Comment out the B block only within UL17 -> picoAOD section
# logic: Find lines inside UL17...UL18 range, then inside picoAOD...UL18 range,
# then matching the B:...C: block, and comment them out (excluding the C: line)
sed -i '/UL17:/,/UL18:/ { 
    /picoAOD:/,/UL18:/ {
        /^[[:space:]]*B:/,/^[[:space:]]*C:/ {
            /^[[:space:]]*C:/! s/^/#/
        }
    }
}' $OUTPUT_DIR/datasets_temp.yml

display_section_header "Running analysis processor for background datasets"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "data TTToHadronic TTToSemiLeptonic TTTo2L2Nu" \
    --year "UL17 UL18 UL16_preVFP UL16_postVFP" \
    --output-filename "test_databkgs.coffea" \
    --output-subdir "$JOB" \
    --config $JOB_CONFIG \
    --dataset-metadata "$OUTPUT_DIR/datasets_temp.yml" \
    # --additional-flags "--slurm --debug"
