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
INPUT_DIR="$OUTPUT_BASE_DIR/synthetic_dataset_cluster"
JOB="synthetic_dataset_plot"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running test processor"
run_command python coffea4bees/jet_clustering/make_jet_splitting_PDFs.py \
    $INPUT_DIR/test_synthetic_datasets.coffea \
    --doTest  \
    --out $OUTPUT_DIR/jet-splitting-PDFs-test
    
display_section_header "Checking if pdf files exist"
ls $OUTPUT_DIR/jet-splitting-PDFs-test/clustering_pdfs_vs_pT_RunII.yml 
ls $OUTPUT_DIR/jet-splitting-PDFs-test/test_sampling_pt_1b0j_1b0j_mA.pdf 

