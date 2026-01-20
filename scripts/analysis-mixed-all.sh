#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets_HH4b_Run2/"}
echo "Using datasets file: $DATASETS"

OUTPUT_DIR="${1:-"output"}/analysis_mixed_all"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

display_section_header "Running test processor"
python runner.py  -o histMixedBkg_TT.coffea -d   TTTo2L2Nu_for_mixed TTToHadronic_for_mixed TTToSemiLeptonic_for_mixed   -p coffea4bees/analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS

python runner.py  -o histMixedBkg_data_3b_for_mixed_kfold.coffea -d   data_3b_for_mixed  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c coffea4bees/analysis/metadata/HH4b_mixed_data.yml

sed -e "s/use_kfold: True/use_kfold: False/" "coffea4bees/analysis/metadata/HH4b_mixed_data.yml" > /tmp/${USER}/HH4b_mixed_data_nokfold.yml
python runner.py  -o histMixedBkg_data_3b_for_mixed.coffea -d   data_3b_for_mixed  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c /tmp/${USER}/HH4b_mixed_data_nokfold.yml

sed -e "s/use_kfold: True/use_kfold: False/" -e "s/use_ZZinSB: False/use_ZZinSB: True/" "coffea4bees/analysis/metadata/HH4b_mixed_data.yml" > /tmp/${USER}/HH4b_mixed_data_ZZinSB.yml
python runner.py  -o histMixedBkg_data_3b_for_mixed_ZZinSB.coffea -d   data_3b_for_mixed  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c /tmp/${USER}/HH4b_mixed_data_ZZinSB.yml

sed -e "s/use_kfold: True/use_kfold: False/" -e "s/use_ZZandZHinSB: False/use_ZZandZHinSB: True/" "coffea4bees/analysis/metadata/HH4b_mixed_data.yml" > /tmp/${USER}/HH4b_mixed_data_ZZandZHinSB.yml
python runner.py  -o histMixedBkg_data_3b_for_mixed_ZZandZHinSB.coffea -d   data_3b_for_mixed  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c /tmp/${USER}/HH4b_mixed_data_ZZandZHinSB.yml

python runner.py  -o histMixedData.coffea -d    mixeddata  -p coffea4bees/analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS

python runner.py  -o histSignal.coffea -d    GluGluToHHTo4B_cHHH1 ZH4b ZZ4b  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS
ls

display_section_header "Hist --> JSON"

python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedData.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_TT.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_kfold.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZinSB.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.coffea
python coffea4bees/stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histSignal.coffea


