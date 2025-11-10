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
JOB="mixeddata_cluster"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running test processor"
bash coffea4bees/scripts/run-analysis-processor.sh \
    --processor "coffea4bees/analysis/processors/processor_make_hemi_library.py" \
    --output-base "$OUTPUT_BASE_DIR" \
    --datasets "data" \
    --year "UL18" \
    --output-filename "test_mixed_datasets.coffea" \
    --output-subdir "$JOB" \
    --config coffea4bees/analysis/metadata/make_hemi_library_4b.yml \
    # --additional-flags "--debug"

#    --year "UL17 UL18 UL16_preVFP UL16_postVFP" \    

### Previous tests
# python runner.py -t -o test_cluster_synthetic_data.coffea -d synthetic_data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/cluster_4b.yml
# python runner.py -t -o test_synthetic_datasets_Run3.coffea -d data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix  -op $OUTPUT_DIR -m coffea4bees/metadata/datasets_HH4b_Run3_fourTag_v3.yml -c coffea4bees/analysis/metadata/cluster_4b_noTTSubtraction.yml

# python runner.py -t -o test_cluster_synthetic_data.coffea -d synthetic_data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/cluster_4b.yml


# time python runner.py  -o cluster_data_Run2.coffea -d data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/cluster_4b_noTTSubtraction.yml
# time python runner.py  -o cluster_data_Run2_noTT.coffea -d data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/cluster_4b.yml
# time python runner.py  -o cluster_synthetic_data_Run2.coffea -d synthetic_data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/cluster_4b_noTTSubtraction.yml
# python runner.py -t -o test_synthetic_datasets_upto6j.coffea -d data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y UL18  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/cluster_and_decluster.yml
# python runner.py  -o test_synthetic_datasets_cluster_2023_preBPix.coffea -d data  -p coffea4bees/analysis/processors/processor_cluster_4b.py -y 2023_preBPix   -op hists/ -m coffea4bees/metadata/datasets_HH4b_Run3_fourTag.yml -c coffea4bees/analysis/metadata/cluster_4b_noTTSubtraction.yml

# python  jet_clustering/make_jet_splitting_PDFs.py $OUTPUT_DIR/test_synthetic_datasets_4j_and_5j.coffea  --out jet_clustering/jet-splitting-PDFs-00-02-00 

