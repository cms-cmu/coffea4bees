config['output_path'] = "output/boostedVeto"
config['analysis_container'] = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest" 
config['combine_container'] = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_11_3_4-combine_v9.1.0-harvester_v2.1.0"
config['SvB_rebin'] = "1"
# config["variable_binning"] = "--variable_binning"
config["variable_binning"] = ""

rule all:
    input:
        # f"{config['output_path']}/plots/RunII/passPreSel/fourTag/SB/nPVs.pdf",
        f"{config['output_path']}/datacards/SvB_MA_postfitplots_prefit.pdf"

### Including modules
module analysis:
    snakefile: "rules/analysis"
    config: config

module stat_analysis:
    snakefile: "rules/stat_analysis"
    config: config

use rule make_plots from analysis as make_plots_local with:
    input: f"{config['output_path']}/histAll.coffea"
    output: f"{config['output_path']}/plots/RunII/passPreSel/fourTag/SB/nPVs.pdf"
    container: config["analysis_container"]
    params:
        output_dir = f"{config['output_path']}/plots/"

use rule merging_coffea_files from analysis as merge_databkgs with:
    input: expand([f"{config['output_path']}/singlefiles/histsignal__{{sample_signal}}-{{year}}.coffea"], sample_signal=config['dataset_signals'], year=config['year']) + expand([f"{config['output_path']}/singlefiles/hist__{{idat}}-{{iyear}}.coffea"], idat=config['dataset'], iyear=config['year']) + expand([f"{config['output_path']}/singlefiles/histsignalHH4b__GluGluToHHTo4B_cHHH1-{{iyear}}.coffea"], iyear=config['year'])
    output: f"{config['output_path']}/histAll.coffea"
    params:
        output= "histAll.coffea",
        logname= "histAll",
        output_path = config['output_path'],
        run_performance = False


use rule merging_coffea_files from analysis as merge_signals_cHHHX with:
    input: expand([f"{config['output_path']}/singlefiles/histsyst_{{idatsyst}}-{{iyear}}.coffea"], idatsyst=config['dataset_systematics'], iyear=config['year'])
    output: f"{config['output_path']}/histAll_signals_cHHHX.coffea"
    params:
        output= "histAll_signals_cHHHX.coffea",
        logname= "signals_cHHHX",
        output_path = config['output_path'],
        run_performance = False

use rule convert_hist_to_json from stat_analysis as convert_hist_to_json_databkgs with:
    input: f"{config['output_path']}/histAll.coffea"
    output: f"{config['output_path']}/histAll.json"
    params: flag=' '

use rule convert_hist_to_json from stat_analysis as convert_hist_to_json_signals_cHHHX with:
    input: f"{config['output_path']}/histAll_signals_cHHHX.coffea"
    output: f"{config['output_path']}/histAll_signals_cHHHX.json"
    params:
        flag = "-s"

use rule convert_hist_to_json_closure from stat_analysis as convert_hist_to_json_mixedBkg_TT with:
    input: f"{config['output_path']}/histMixedBkg_TT.coffea"
    output: f"{config['output_path']}/histMixedBkg_TT.json"

use rule convert_hist_to_json_closure from stat_analysis as convert_hist_to_json_mixedData with:
    input: f"{config['output_path']}/histMixedData.coffea"
    output: f"{config['output_path']}/histMixedData.json"

use rule convert_hist_to_json_closure from stat_analysis as convert_hist_to_json_mixedBkg_data_3b with:
    input: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.coffea"
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.json"

use rule convert_json_to_root from stat_analysis as convert_json_to_root_databkgs with:
    input: f"{config['output_path']}/histAll.json"
    output: f"{config['output_path']}/histAll.root"

use rule convert_json_to_root from stat_analysis as convert_json_to_root_mixedBkg_TT with:
    input: f"{config['output_path']}/histMixedBkg_TT.json"
    output: f"{config['output_path']}/histMixedBkg_TT.root"

use rule convert_json_to_root from stat_analysis as convert_json_to_root_mixedData with:
    input: f"{config['output_path']}/histMixedData.json"
    output: f"{config['output_path']}/histMixedData.root"

use rule convert_json_to_root from stat_analysis as convert_json_to_root_mixedBkg_data_3b with:
    input: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.json"
    output: f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.root"

use rule run_two_stage_closure from stat_analysis as run_two_stage_closure_local with:
    input: 
        file_TT = f"{config['output_path']}/histMixedBkg_TT.root",
        file_mix = f"{config['output_path']}/histMixedData.root",
        file_sig = f"{config['output_path']}/histAll.root",
        file_data3b = f"{config['output_path']}/histMixedBkg_data_3b_for_mixed.root"
    output: f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin{config['SvB_rebin']}/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin{config['SvB_rebin']}.pkl"
    params:
        outputPath = f"{config['output_path']}/closureFits/ULHH_kfold",
        rebin = {config['SvB_rebin']},
        variable = "SvB_MA_ps_hh",
        variable_binning=f"{config["variable_binning"]}",
        container = config['combine_container']


use rule make_combine_inputs from stat_analysis as make_combine_inputs_local with:
    input:
        injson = f"{config['output_path']}/histAll.json",
        injsonsyst = f"{config['output_path']}/histAll_signals_cHHHX.json",
        bkgsyst = f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin{config['SvB_rebin']}/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin{config['SvB_rebin']}.pkl"
    output:
        f"{config['output_path']}/datacards/shapes.root",
        f"{config['output_path']}/datacards/datacard_HHbb_2018.txt"
    params:
        container = config['combine_container'],
        variable= "SvB_MA.ps_hh_fine",
        rebin={config['SvB_rebin']},
        output_dir=f"{config['output_path']}/datacards/",
        variable_binning=f"{config["variable_binning"]}"

use rule run_combine from stat_analysis as run_combine_local with:
    input: f"{config['output_path']}/datacards/datacard_HHbb_2018.txt"
    params:
        container = config['combine_container'],
        output_dir = f"{config['output_path']}/datacards/"
    output: 
        f"{config['output_path']}/datacards/datacard.root",
        f"{config['output_path']}/datacards/datacard.txt"

use rule run_postfit from stat_analysis as run_postfit_local with:
    input: f"{config['output_path']}/datacards/datacard.root"
    params:
        container = config["combine_container"],
        output_dir = f"{config['output_path']}/datacards/"
    output: f"{config['output_path']}/datacards/SvB_MA_postfitplots_prefit.pdf"

use rule make_syst_plots from stat_analysis as make_syst_plots_local with:
    input: 
        root = f"{config['output_path']}/datacards/shapes.root",
        datacard = f"{config['output_path']}/datacards/datacard.txt"
    params:
        container = config["combine_container"],
        output_dir = f"{config['output_path']}/systs/"
    output: f"{config['output_path']}/datacards/systs/SvB_MA_ps_hh_nominal.pdf"
