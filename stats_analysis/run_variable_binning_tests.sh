rebin=$1
# rebin_folder="rebin${rebin}"
rebin_folder="varrebin${rebin}"
var="SvB_MA.ps_hh_fine"

bkg_syst_folder=local_outputs/stat_analysis/optimize_bin/${var//./_}_${rebin_folder}/closureFits/
limits_folder=local_outputs/stat_analysis/optimize_bin/${var//./_}_${rebin_folder}/
# bkg_syst_folder=local_outputs/stat_analysis/ZZandZHinSB/ZZinSB/${var//./_}_${rebin_folder}/closureFits/
# limits_folder=local_outputs/stat_analysis/ZZandZHinSB/ZZinSB/${var//./_}_${rebin_folder}/

echo "Removing old folders ${bkg_syst_folder} and ${limits_folder}"
rm -rf ${bkg_syst_folder}
rm -rf ${limits_folder}

tmp=local_outputs/analysis
tmp_data3b=${tmp}/histMixedBkg_data_3b_for_mixed_kfold.root  ##--use_kfold
# tmp_data3b=output/tmp/histMixedBkg_data_3b_for_mixed_ZZinSB.root  ##--use_ZZinSB
# tmp_data3b=output/tmp/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.root  ##--use_ZZandZHinSB
echo "Running TwoStageClosure"
python3 stats_analysis/runTwoStageClosure.py --var ${var//./_} \
        --outputPath ${bkg_syst_folder} \
        --use_kfold \
        --input_file_TT ${tmp}/histMixedBkg_TT.root \
        --input_file_mix ${tmp}/histMixedData.root \
        --input_file_sig ${tmp}/histAll.root \
        --input_file_data3b ${tmp_data3b} \
        --rebin ${rebin} \
        --variable_binning 

echo "Making combine inputs"
python3 stats_analysis/make_combine_inputs.py --var ${var} \
        -f output/test_coffea4bees/histAll.json \
        --syst_file output/test_coffea4bees/histAll_signals_cHHHX.json  \
        --bkg_syst_file ${bkg_syst_folder}/3bDvTMix4bDvT/SvB_MA/${rebin_folder}/SR/hh/hists_closure_3bDvTMix4bDvT_${var//./_}_${rebin_folder}.pkl \
        --output_dir ${limits_folder} --rebin ${rebin} \
        --metadata stats_analysis/metadata/HH4b.yml \
        --variable_binning

echo "Running combine"
source stats_analysis/run_combine.sh ${limits_folder} --limits