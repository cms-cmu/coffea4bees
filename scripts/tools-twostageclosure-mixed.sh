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
OUTPUT_DIR=$OUTPUT_BASE_DIR/$JOB
create_output_directory "$OUTPUT_DIR"

## In root envirornment
# Regression test on checked-in closure fixture inputs
mkdir -p $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/
cp -r coffea4bees/stats_analysis/tests/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.root $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/

python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 1 --outputPath $OUTPUT_DIR --years 2016 2017 2018 --reuse_inputs --do_CI --maxBasis 5 --ignore_failures
python3 coffea4bees/stats_analysis/tests/test_runTwoStageClosure.py --knownCounts coffea4bees/stats_analysis/tests/known_twoStageClosure_counts_SvB_MA_ps_hh.yml --output_path $OUTPUT_DIR --inputFile $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.root

python3 coffea4bees/stats_analysis/tests/dumpTwoStageInputs.py --input $OUTPUT_DIR/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.root   --output $OUTPUT_DIR/test_dump_twoStageClosureInputsCounts.yml

