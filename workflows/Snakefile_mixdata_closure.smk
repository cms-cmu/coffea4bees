import os

config = {
    "analysis_container": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest",
    "combine_container": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_11_3_4-combine_v9.1.0-harvester_v2.1.0",
    "output_path": "output/closure_tests/test",
    "datasets": "coffea4bees/metadata/datasets_HH4b_v1.yml",
}

# Define username once for reuse throughout the workflow
USERNAME = os.getenv("USER", "coffea4bees_default")

# Import rule modules
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

include: "helpers/common.smk"

rule final_rule:
    input: 
        # f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/zh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zh_rebin1.pkl",
        # f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin1.pkl",
        f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.pkl",

use rule analysis_processor from analysis as mixed_bkg_tt with:
    output: f"{config['output_path']}/histMixedBkg_TT.coffea"
    params:
        datasets = "TTTo2L2Nu_for_mixed TTToHadronic_for_mixed TTToSemiLeptonic_for_mixed",
        years = "UL17 UL18 UL16_preVFP UL16_postVFP",
        metadata = "coffea4bees/analysis/metadata/HH4b_nottcheck.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config["datasets"],
        blind = False,
        run_performance = False,
        extra_arguments="",
        username = USERNAME
    log: f"{config['output_path']}/logs/analysis_processor_mixed_bkg_tt.log"

use rule analysis_processor from analysis as hist_all with:
    output: f"{config['output_path']}/histAll.coffea"
    params:
        datasets = "data ZZ4b ggZH4b ZH4b",
        years = "UL17 UL18 UL16_preVFP UL16_postVFP",
        metadata = "coffea4bees/analysis/metadata/HH4b.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config["datasets"],
        blind = False,
        run_performance = False,
        extra_arguments="",
        username = USERNAME
    log: f"{config['output_path']}/logs/analysis_processor_hist_all.log"


use rule analysis_processor from analysis as mixed_bkg_data_3b_for_mixed_kfold with:
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed_kfold.coffea"
    params:
        datasets="data_3b_for_mixed",
        years="2016 2017 2018",
        metadata = "coffea4bees/analysis/metadata/HH4b_nottcheck.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config["datasets"],
        blind = False,
        run_performance = False,
        extra_arguments="",
        username = USERNAME
    log: f"{config['output_path']}/logs/analysis_processor_hmixed_bkg_data_3b_for_mixed_kfold.log"

use rule analysis_processor from analysis as mixed_data with:
    output: f"{config['output_path']}/histMixedData.coffea"
    params:
        datasets="mixeddata",
        years="2016 2017 2018",
        metadata = "coffea4bees/analysis/metadata/HH4b_nottcheck.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config["datasets"],
        blind = False,
        run_performance = False,
        extra_arguments="",
        username = USERNAME
    log: f"{config['output_path']}/logs/analysis_processor_mixed_data.log"


use rule analysis_processor from analysis as mixed_bkg_data_3b_for_mixed with:
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.coffea"
    params:
        datasets="data_3b_for_mixed",
        years="2016 2017 2018",
        metadata = "coffea4bees/analysis/metadata/HH4b_mixed_data.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config["datasets"],
        blind = False,
        run_performance = False,
        extra_arguments="",
        username = USERNAME
    log: f"{config['output_path']}/logs/analysis_processor_mixed_bkg_data_3b_for_mixed.log"

use rule convert_hist_to_json from stat_analysis as hist_to_json_histall with:
    input: f"{config['output_path']}/histAll.coffea"
    output: f"{config['output_path']}/histAll.json"
    log: f"{config['output_path']}/logs/hist_to_json_histall.log"

use rule convert_hist_to_json_closure from stat_analysis as hist_to_json_mixed_data with:
    input: f"{config['output_path']}/histMixedData.coffea"
    output: f"{config['output_path']}/histMixedData.json"
    log: f"{config['output_path']}/logs/hist_to_json_mixed_data.log"

use rule convert_hist_to_json_closure from stat_analysis as hist_to_json_mixed_bkg_tt with:
    output: f"{config['output_path']}/histMixedBkg_TT.json"
    input: f"{config['output_path']}/histMixedBkg_TT.coffea"
    log: f"{config['output_path']}/logs/hist_to_json_mixed_bkg_tt.log"

use rule convert_hist_to_json_closure from stat_analysis as hist_to_json_mixed_bkg_data_3b_for_mixed_kfold with:
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed_kfold.json"
    input: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed_kfold.coffea"
    log: f"{config['output_path']}/logs/hist_to_json_mixed_bkg_data_3b_for_mixed_kfold.log"

use rule convert_hist_to_json_closure from stat_analysis as hist_to_json_mixed_bkg_data_3b_for_mixed with:
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.json"
    input: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.coffea"
    log: f"{config['output_path']}/logs/hist_to_json_mixed_bkg_data_3b_for_mixed.log"

use rule convert_json_to_root from stat_analysis as json_histall with:
    input: f"{config['output_path']}/histAll.json"
    output: f"{config['output_path']}/histAll.root"
    log: f"{config['output_path']}/logs/json_histall.log"

use rule convert_json_to_root from stat_analysis as json_to_root_mixed_data with:
    input: f"{config['output_path']}/histMixedData.json"
    output: f"{config['output_path']}/histMixedData.root"
    log: f"{config['output_path']}/logs/json_to_root_mixed_data.log"


use rule convert_json_to_root from stat_analysis as json_to_root_mixed_bkg_tt with:
    input: f"{config['output_path']}/histMixedBkg_TT.json"
    output: f"{config['output_path']}/histMixedBkg_TT.root"
    log: f"{config['output_path']}/logs/json_to_root_mixed_bkg_tt.log"

use rule convert_json_to_root from stat_analysis as json_to_root_mixed_bkg_data_3b_for_mixed_kfold with:
    input: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed_kfold.json"
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed_kfold.root"
    log: f"{config['output_path']}/logs/json_to_root_mixed_bkg_data_3b_for_mixed_kfold.log"

use rule convert_json_to_root from stat_analysis as json_to_root_mixed_bkg_data_3b_for_mixed with:
    input: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.json"
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.root"
    log: f"{config['output_path']}/logs/json_to_root_mixed_bkg_data_3b_for_mixed.log"


use rule run_two_stage_closure from stat_analysis as run_two_stage_closure_hh with:
    input: 
        file_TT = f"{config['output_path']}/histMixedBkg_TT.root",
        file_mix = f"{config['output_path']}/histMixedData.root",
        file_sig = f"{config['output_path']}/histAll.root",
        file_data3b = f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.root"
    output: f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.pkl"
    params:
        outputPath = f"{config['output_path']}/closureFits/ULHH_kfold",
        rebin = "1",
        variable = "SvB_MA_ps_hh",
        extra_arguments = "--use_kfold",
        container_wrapper = './run_container combine'
    log: f"{config['output_path']}/logs/run_two_stage_closure_hh.log"

use rule run_two_stage_closure from stat_analysis as run_two_stage_closure_zz with:
    input: 
        file_TT = f"{config['output_path']}/histMixedBkg_TT.root",
        file_mix = f"{config['output_path']}/histMixedData.root",
        file_sig = f"{config['output_path']}/histAll.root",
        file_data3b = f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.root"
    output: f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin1.pkl"
    params:
        outputPath = f"{config['output_path']}/closureFits/ULHH_kfold",
        rebin = "1",
        variable = "SvB_MA_ps_zz",
        extra_arguments = "",
        container_wrapper = './run_container combine'
    log: f"{config['output_path']}/logs/run_two_stage_closure_zz.log"

use rule run_two_stage_closure from stat_analysis as run_two_stage_closure_zh with:
    input: 
        file_TT = f"{config['output_path']}/histMixedBkg_TT.root",
        file_mix = f"{config['output_path']}/histMixedData.root",
        file_sig = f"{config['output_path']}/histAll.root",
        file_data3b = f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.root"
    output: f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/zh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zh_rebin1.pkl"
    params:
        outputPath = f"{config['output_path']}/closureFits/ULHH_kfold",
        rebin = "1",
        variable = "SvB_MA_ps_zh",
        extra_arguments = "",
        container_wrapper = './run_container combine'
    log: f"{config['output_path']}/logs/run_two_stage_closure_zh.log"
