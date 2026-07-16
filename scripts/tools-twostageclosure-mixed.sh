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
JOB="tools_twostageclosure_mixed"
INPUT_DIR=$OUTPUT_BASE_DIR/analysis_test_mixed
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

## In root envirornment

display_section_header "Convert json to root"
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/testMixedData.json --output $OUTPUT_DIR --histos SvB_MA_ps_hh
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/testMixedBkg_TT.json --output $OUTPUT_DIR --histos SvB_MA_ps_hh
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/testMixedBkg_data_3b_for_mixed_kfold.json --output $OUTPUT_DIR --histos SvB_MA_ps_hh
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/testMixedBkg_data_3b_for_mixed.json --output $OUTPUT_DIR --histos SvB_MA_ps_hh
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/testSignal_UL.json                  --output $OUTPUT_DIR --histos SvB_MA_ps_hh


#
# Test it with the
#
display_section_header "Run test runTwoStageClosure"
python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 1 --outputPath $OUTPUT_DIR/testsLocal  --do_CI \
    --input_file_data3b $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed.root \
    --input_file_TT     $OUTPUT_DIR/testMixedBkg_TT.root \
    --input_file_mix    $OUTPUT_DIR/testMixedData.root \
    --input_file_sig    $OUTPUT_DIR/testSignal_UL.root \
    
display_section_header "Run test runTwoStageClosure kfold"
ls -lrt $OUTPUT_DIR/
python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 1 --outputPath $OUTPUT_DIR/testsLocal_kfold/  --do_CI --use_kfold  \
    --input_file_data3b $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed_kfold.root \
    --input_file_TT     $OUTPUT_DIR/testMixedBkg_TT.root \
    --input_file_mix    $OUTPUT_DIR/testMixedData.root \
    --input_file_sig    $OUTPUT_DIR/testSignal_UL.root \


# #python old_make_combine_hists.py -i ./files_HIG-20-011/hists_closure_3bDvTMix4bDvT_SR_weights_newSBDef.root -o HIG-20-011/hist_closure_SvB_MA.root --TDirectory 3bDvTMix4bDvT_v0/hh2018 --var multijet --channel hh2018 -n mj --rebin 10 --systematics ./files_HIG-20-011/closureResults_SvB_MA_hh.pkl
# #python old_make_combine_hists.py -i ./files_HIG-20-011/data2018/hists_j_r.root -o HIG-20-011/hist_SvB_MA.root  -r SR --var SvB_MA_ps_hh --channel hh2018 -n mj --tag three --cut passPreSel --rebin 10 --systematics ./files_HIG-20-011/closureResults_SvB_MA_hh.pkl

#
# Test it with full inputs
#
#

#
# Make the input with
#
#  python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath coffea4bees/stats_analysis/tests --skip_closure

mkdir -p $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/
cp -r coffea4bees/stats_analysis/tests/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.root $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/

python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 1 --outputPath $OUTPUT_DIR --reuse_inputs --do_CI
python3 coffea4bees/stats_analysis/tests/test_runTwoStageClosure.py --knownCounts coffea4bees/stats_analysis/tests/known_twoStageClosure_counts_SvB_MA_ps_hh.yml --output_path $OUTPUT_DIR --inputFile $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.root

python3 coffea4bees/stats_analysis/tests/dumpTwoStageInputs.py --input $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.root   --output $OUTPUT_DIR/test_dump_twoStageClosureInputsCounts.yml

