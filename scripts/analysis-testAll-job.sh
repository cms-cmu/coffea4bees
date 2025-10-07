#!/bin/bash
# Source common functions
source "src/scripts/common.sh"

# Setup proxy if needed
setup_proxy

display_section_header "Input Datasets"
DATASETS=${DATASET:-"coffea4bees/metadata/datasets_HH4b.yml"}
echo "Using datasets file: $DATASETS"

OUTPUT_DIR="${1:-"output"}/analysis_testAll_job"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi


display_section_header "Running test processor"
#python runner.py -o hist_databkgs.coffea  -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu ZZ4b ZH4b ggZH4b   -p coffea4bees/analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS --condor
#python runner.py -o hist_databkgs.coffea  -d data     -p coffea4bees/analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS --condor
#python runner.py -o hist_TTbkgs.coffea  -d  TTToHadronic TTToSemiLeptonic TTTo2L2Nu    -p coffea4bees/analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS --condor
#python runner.py -o hist_otherSig.coffea  -d  ZZ4b ZH4b ggZH4b    -p coffea4bees/analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS --condor


python runner.py -o hist_signal.coffea -d GluGluToHHTo4B_cHHH1 GluGluToHHTo4B_cHHH0 GluGluToHHTo4B_cHHH2p45 GluGluToHHTo4B_cHHH5  -p coffea4bees/analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op $OUTPUT_DIR -m coffea4bees/metadata/datasets_HH4b_v1p1.yml -c coffea4bees/analysis/metadata/HH4b_systematics.yml --condor

#python src/tools/merge_coffea_files.py -f $OUTPUT_DIR/hist_databkgs.coffea $OUTPUT_DIR/hist_signal.coffea  -o $OUTPUT_DIR/histAll.coffea


#python coffea4bees/analysis/tests/cutflow_test.py   --inputFile ${OUTPUT_DIR}/histAll.coffea --knownCounts coffea4bees/analysis/tests/histAllCounts.yml

# python runner.py -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu ZZ4b ZH4b GluGluToHHTo4B_cHHH1 -c coffea4bees/analysis/metadata/HH4b_noFvT.yml   -p coffea4bees/analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP -o histAll_noFvT.coffea -op hists/


