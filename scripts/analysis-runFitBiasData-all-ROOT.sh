#!/bin/bash
# Source common functions
source "src/scripts/common.sh"


OUTPUT_DIR="${1:-"output"}/analysis_runFitBiasData_all_ROOT"
display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

#  In hist environment
# 

python coffea4bees/stats_analysis/convert_hist_to_json.py --input coffea4bees/analysis/hists/histAll.coffea --output ${OUTPUT_DIR}/histAll.json

## In root envirornment
#

python3 coffea4bees/stats_analysis/convert_json_to_root.py -f ${OUTPUT_DIR}/histAll.json                  --output coffea4bees/analysis/${OUTPUT_DIR}

#
# Make the input with
#
#  python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath coffea4bees/stats_analysis/tests --skip_closure

#
# Test it with
#

#python3 coffea4bees/stats_analysis/runFitBiasData.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath coffea4bees/stats_analysis/fitBiasData/ULHH --bkg_syst_file coffea4bees/stats_analysis/closureFits/ULHH/3bDvTMix4bDvT/SvB_MA/rebin8/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin8.pk
python3 coffea4bees/stats_analysis/runFitBiasData.py  --var SvB_MA_ps_zz   --rebin 8 --outputPath coffea4bees/stats_analysis/fitBiasData/ULHH --bkg_syst_file coffea4bees/stats_analysis/closureFits/ULHH/3bDvTMix4bDvT/SvB_MA/rebin8/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin8.pkl  

