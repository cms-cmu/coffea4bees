#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse flags
DO_TEST=""
OUTPUT_BASE_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --do-test)
            DO_TEST="1"
            shift
            ;;
        --output-base)
            OUTPUT_BASE_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "${OUTPUT_BASE_ARGS[@]}")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

# Create output directory
JOB="classifier_inputs_mixeddata_all_Run3"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets/"}
echo "Using datasets file: $DATASETS"

display_section_header "Modifying config"
BASE_CONFIG=coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3.yml
if [ -n "$DO_TEST" ]; then
    JOB_CONFIG=/tmp/${USER}/HH4b_classifier_inputs_Run3_test.yml
    mkdir -p /tmp/${USER}
    sed -e "s|make_classifier_input: .*|make_classifier_input: hists/local|" \
        -e "s|make_friend_FvT_weight: .*|make_friend_FvT_weight: hists/local|" \
        "$BASE_CONFIG" > "$JOB_CONFIG"
    echo "Test mode: using output directory for classifier inputs"
else
    JOB_CONFIG=$BASE_CONFIG
fi
cat $JOB_CONFIG; echo

display_section_header "Running processor"
if [ -n "$DO_TEST" ]; then
    time bash coffea4bees/scripts/run-analysis-processor.sh \
        --output-base "$OUTPUT_BASE_DIR" \
        --datasets "mixeddata_all" \
        --dataset-metadata "$DATASETS" \
        --year "2022_EE 2022_preEE 2023_BPix 2023_preBPix" \
        --output-filename "test_classifier_inputs_mixeddata_all.coffea" \
        --output-subdir "$JOB" \
        --config $JOB_CONFIG \
        # --additional-flags "--friends coffea4bees/metadata/friends_empty.yml"
else
    time bash coffea4bees/scripts/run-analysis-processor.sh \
        --output-base "$OUTPUT_BASE_DIR" \
        --datasets "mixeddata_all" \
        --dataset-metadata "$DATASETS" \
        --year "2022_EE 2022_preEE 2023_BPix 2023_preBPix" \
        --output-filename "classifier_inputs_mixeddata_all.coffea" \
        --output-subdir "$JOB" \
        --config $JOB_CONFIG \
        --no-test \
        --additional-flags "--condor --friends coffea4bees/metadata/friends_empty.yml"
fi
