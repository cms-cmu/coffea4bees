#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Parse output base argument
OUTPUT_BASE_DIR=$(parse_output_base_arg "output" "$@")
if [ $? -ne 0 ]; then
    echo "Error parsing output base argument. Use --output-base DIR to specify the output directory. Default DIR=output/"
    exit 1
fi

setup_proxy

DATASETS=${DATASETS:-"coffea4bees/metadata/datasets_HH4b.yml"}

# Create output directory
JOB="analysis_test_mixed"
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

display_section_header "Running test processor"
python runner.py -t -o testMixedBkg_TT.coffea -d   TTTo2L2Nu_for_mixed TTToHadronic_for_mixed TTToSemiLeptonic_for_mixed   -p coffea4bees/analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_nottcheck.yml

python runner.py -t -o testMixedBkg_data_3b_for_mixed_kfold.coffea -d   data_3b_for_mixed  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2017 2018 2016  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_mixed_data.yml

python runner.py -t -o testMixedBkg_data_3b_for_mixed.coffea -d   data_3b_for_mixed  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2017 2018 2016  -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_nottcheck.yml

python runner.py -t -o testMixedData.coffea -d    mixeddata  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2016 2017 2018 -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_nottcheck.yml
python runner.py -t -o testSignals.coffea -d ZH4b ZZ4b  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_nottcheck.yml
python runner.py -t -o testSignals_HH4b.coffea -d GluGluToHHTo4B_cHHH1  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_signals.yml
python src/tools/merge_coffea_files.py -f $OUTPUT_DIR/testSignals_HH4b.coffea $OUTPUT_DIR/testSignals.coffea -o $OUTPUT_DIR/testSignal_UL.coffea
ls $OUTPUT_DIR

display_section_header "Hist --> JSON"

python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_TT.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed_kfold.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedData.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testSignal_UL.coffea
#python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testSignal_preUL.coffea

ls $OUTPUT_DIR

