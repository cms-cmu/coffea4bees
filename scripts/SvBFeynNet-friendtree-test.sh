#!/bin/bash
# Quick local test for SvB_FeynNet friend tree creation.
# Runs in test mode (-t, ~1 chunk) without condor on data/2022_EE.
# Validates that the FeynNet ONNX model loads and the friend tree dump works
# before submitting the full job to condor via Snakefile_SvBFeynNet_friendtrees_Run3.smk.
#
# Usage (from barista root):
#   ./run_container bash coffea4bees/scripts/SvBFeynNet-friendtree-test.sh [--output-base DIR]

source "src/scripts/common.sh"

OUTPUT_BASE_DIR=$(parse_output_base_arg "output/" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory."
    exit 1
fi

JOB="test_SvBFeynNet_friendtree"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running SvB_FeynNet friend tree test (local, test mode)"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "data" \
    --dataset-metadata "coffea4bees/metadata/datasets/" \
    --year "2022_EE" \
    --output-filename "test_SvBFeynNet_friendtree.coffea" \
    --output-subdir "$JOB" \
    --config coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml \
    --friends "coffea4bees/metadata/friends_HH4b.yml"
    # No --condor: runs locally with Dask
    # No --no-test: keeps -t flag (processes ~1 chunk)
