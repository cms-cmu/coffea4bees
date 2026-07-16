#!/bin/bash
# 
# To run:
# (from coffea4bees/)
# ./shell_combine source coffea4bees/scripts//analysis-runTwoStageClosure-all-ROOT.sh hists/ hists/
# (first argument is input directory, second argument is output directory, both from the python directory)
#

return_to_base=false
if [ "$(basename "$PWD")" == "python" ]; then
    echo "You are in the python directory."
else
    return_to_base=true
    display_section_header "Moving to python folder"
    cd coffea4bees/
fi

# Check if the first argument is provided
if [[ $# -lt 1 ]]; then
    # No first argument provided, set a default value
    INPUT_DIR="hists/"
else
    # First argument provided, use the provided value
    INPUT_DIR=$1
fi

# Check if the second argument is provided
if [[ $# -lt 2 ]]; then
    # No second argument provided, set a default value
    OUTPUT_DIR="hists/"
else
    # Second argument provided, use the provided value
    OUTPUT_DIR=$2
fi

echo "The input directory is: $INPUT_DIR"
echo "The output directory is: $OUTPUT_DIR"


display_section_header "Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

#
# In root envirornment
#
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/histMixedData.json --output $OUTPUT_DIR
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/histMixedBkg_TT.json --output $OUTPUT_DIR
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/histMixedBkg_data_3b_for_mixed.json --output $OUTPUT_DIR
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/histMixedBkg_data_3b_for_mixed_kfold.json --output $OUTPUT_DIR
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZinSB.json --output $OUTPUT_DIR
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.json --output $OUTPUT_DIR
python3 coffea4bees/stats_analysis/convert_json_to_root.py -f $INPUT_DIR/histSignal.json --output $OUTPUT_DIR



#
# Make the input with
#
#  python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath coffea4bees/stats_analysis/tests --skip_closure

#
# Test it with
#
# python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin 20 --outputPath $OUTPUT_DIR/closureFits/closureFixTrig --input_file_TT $OUTPUT_DIR/histMixedBkg_TT.root --input_file_mix $OUTPUT_DIR/histMixedData.root --input_file_sig $OUTPUT_DIR/histSignal.root --input_file_data3b $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed.root
# python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zh  --rebin 10 --outputPath $OUTPUT_DIR/closureFits/closureFixTrig --input_file_TT $OUTPUT_DIR/histMixedBkg_TT.root --input_file_mix $OUTPUT_DIR/histMixedData.root --input_file_sig $OUTPUT_DIR/histSignal.root --input_file_data3b $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed.root
# python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zz  --rebin  8 --outputPath $OUTPUT_DIR/closureFits/closureFixTrig --input_file_TT $OUTPUT_DIR/histMixedBkg_TT.root --input_file_mix $OUTPUT_DIR/histMixedData.root --input_file_sig $OUTPUT_DIR/histSignal.root --input_file_data3b $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed.root

for irebin in 20 10 8 5 4 2 1; 
do 
    python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin ${irebin} \
        --outputPath $OUTPUT_DIR/closureFits/ULHH \
        --input_file_TT $OUTPUT_DIR/histMixedBkg_TT.root \
        --input_file_mix $OUTPUT_DIR/histMixedData.root \
        --input_file_sig $OUTPUT_DIR/histSignal.root \
        --input_file_data3b $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed.root 

    python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin ${irebin} \
        --outputPath $OUTPUT_DIR/closureFits/ULHH_kfold --use_kfold \
        --input_file_TT $OUTPUT_DIR/histMixedBkg_TT.root \
        --input_file_mix $OUTPUT_DIR/histMixedData.root \
        --input_file_sig $OUTPUT_DIR/histSignal.root \
        --input_file_data3b $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_kfold.root
    
    python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin ${irebin} \
        --outputPath $OUTPUT_DIR/closureFits/ULHH_ZZinSB --use_ZZinSB \
        --input_file_TT $OUTPUT_DIR/histMixedBkg_TT.root \
        --input_file_mix $OUTPUT_DIR/histMixedData.root \
        --input_file_sig $OUTPUT_DIR/histSignal.root \
        --input_file_data3b $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZinSB.root
    
    python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_hh  --rebin ${irebin} \
            --outputPath $OUTPUT_DIR/closureFits/ULHH_ZZandZHinSB --use_ZZandZHinSB \
            --input_file_TT $OUTPUT_DIR/histMixedBkg_TT.root \
            --input_file_mix $OUTPUT_DIR/histMixedData.root \
            --input_file_sig $OUTPUT_DIR/histSignal.root \
            --input_file_data3b $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root
done 

#python3 coffea4bees/stats_analysis/tests/test_runTwoStageClosure.py --knownCounts coffea4bees/stats_analysis/tests/twoStageClosure_counts_SvB_MA_ps_hh_rebin20.yml --output_path $OUTPUT_DIR/tests


#python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zz  --rebin 8 --outputPath $OUTPUT_DIR/test/ --reuse_inputs --do_CI
#python3 coffea4bees/stats_analysis/runTwoStageClosure.py  --var SvB_MA_ps_zh  --rebin 5 --outputPath $OUTPUT_DIR/tests --reuse_inputs --do_CI

