_defaults = {
    "analysis_container": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest",
    "input_path": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH",
    "output_path": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH",
    "dataset_location": "",
    'dataset': "",
    "combine_container": "",
    "container_wrapper": "./run_container combine",
    'dataset_systematics': {
        'hh': ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH5'],
        'zz': ['ZZ4b'],
        'zh': ['ZH4b', 'ggZH4b']
    },
    'year': [ 'UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18' ],
    # Each case corresponds to a signal process we want to analyze
    "cases": {
        "zz": {
            "datacard": "datacard_ZZ4b",
            "signallabel": "ZZ_bbbb",
            "othersignal": "ZH_bbbb",
            "workspace": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/datacards/ZZ4b/"
        },
        "zh": {
            "datacard": "datacard_ZH4b",
            "signallabel": "ZH_bbbb",
            "othersignal": "ZZ_bbbb",
            "workspace": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/datacards/ZH4b/"
        },
    }
}
for key, val in _defaults.items():
    config.setdefault(key, val)

import os
import shutil
_roc = config.setdefault('run_on_condor', shutil.which("condor_submit") is not None)
config['run_on_condor'] = str(_roc).lower() not in ('false', '0', 'no')

# Adding specific rules for analysis
module analysis:
    snakefile: "rules/analysis.smk"
    config: config
module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

combine_config = config.copy()
combine_config["output_path"] = os.path.join(config["output_path"], "stat_analysis/")
combine_config["channels"] = {
    "ZZ4b": {
        "signallabel": "ZZ_bbbb",
        "othersignal": "ZH_bbbb",
    },
    "ZH4b": {
        "signallabel": "ZH_bbbb",
        "othersignal": "ZZ_bbbb",
    }
}
combine_config["combine_container"] = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0"

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config

rule all:
    input:
        f"{config['output_path']}/stat_analysis/ZZ4b/limits/datacard_limits__ZZ_bbbb.json",
        f"{config['output_path']}/stat_analysis/ZZ4b/significance/datacard_significance__ZZ_bbbb.log",
        f"{config['output_path']}/stat_analysis/ZZ4b/impacts/datacard_impacts__ZZ_bbbb.pdf",
        f"{config['output_path']}/stat_analysis/ZZ4b/gof/datacard_gof__ZZ_bbbb.pdf",
        f"{config['output_path']}/stat_analysis/ZZ4b/likelihood_scan/datacard_likelihood_scan__ZZ_bbbb.pdf",
        f"{config['output_path']}/stat_analysis/ZH4b/limits/datacard_limits__ZH_bbbb.json",
        f"{config['output_path']}/stat_analysis/ZH4b/significance/datacard_significance__ZH_bbbb.log",
        f"{config['output_path']}/stat_analysis/ZH4b/impacts/datacard_impacts__ZH_bbbb.pdf",
        f"{config['output_path']}/stat_analysis/ZH4b/gof/datacard_gof__ZH_bbbb.pdf",
        f"{config['output_path']}/stat_analysis/ZH4b/likelihood_scan/datacard_likelihood_scan__ZH_bbbb.pdf"

use rule merging_coffea_files from analysis as merge_signals with:
    input:
        input_files=lambda wildcards: expand(
            [f"{config['input_path']}/singlefiles/histsyst_{{idatsyst}}-{{iyear}}.coffea"],
            idatsyst=config['dataset_systematics'][wildcards.key],
            iyear=config['year']
        )
    output:
        output_file=f"{config['output_path']}/histAll_signals_{{key}}.coffea"
    log: f"{config['output_path']}/logs/merge_signals_{{key}}.log"
    params:
        output= f"histAll_signals_{{key}}.coffea",
        logname= f"signals_{{key}}",
        output_path = config['output_path'],
        run_performance = False
    wildcard_constraints:
        key="|".join(config['dataset_systematics'].keys())

use rule convert_hist_to_json from stat_analysis as hist_to_json_signals with:
    input:
        f"{config['output_path']}/histAll_signals_{{key}}.coffea"
    output:
        f"{config['output_path']}/histAll_signals_{{key}}.json"
    params:
        syst_flag = "-s"
    log:
        f"{config['output_path']}/logs/histAll_signals_{{key}}.coffea.log"
    wildcard_constraints:
        key="|".join(config['dataset_systematics'].keys())

use rule run_two_stage_closure from stat_analysis as run_two_stage_closure_local with:
    input: 
        file_TT = f"{config['input_path']}/histMixedBkg_TT.root",
        file_mix = f"{config['input_path']}/histMixedData.root",
        file_sig = f"{config['input_path']}/histAll.root",
        file_data3b = f"{config['input_path']}/histMixedBkg_data_3b_for_mixed.root"
    output: 
        f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/{{key}}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{{key}}_rebin1.pkl"
    params:
        outputPath = f"{config['output_path']}/closureFits/ULHH_kfold",
        rebin = 1,
        variable = "SvB_MA_ps_{key}",
        variable_binning="",
        extra_arguments = "",
        container_wrapper = config['container_wrapper']
    log:
        f"{config['output_path']}/logs/run_two_stage_closure_SvB_MA_ps_{{key}}.log"
    wildcard_constraints:
        key="|".join(config['dataset_systematics'].keys())

# Helper functions
def get_key_for_datacard(datacard):
    # Extract the base datacard name without any potential path components
    datacard_base = os.path.basename(datacard)
    for key, info in config['cases'].items():
        if info['datacard'] == datacard_base:
            return key
    raise ValueError(f"No key found for datacard {datacard}")

use rule make_combine_inputs from stat_analysis as make_combine_inputs_local with:
    input:
        injson = f"{config['output_path']}/histAll.json",
        injsonsyst = lambda wildcards: f"{config['output_path']}/histAll_signals_{get_key_for_datacard(wildcards.datacard)}.json",
        bkgsyst = lambda wildcards: f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/{get_key_for_datacard(wildcards.datacard)}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{get_key_for_datacard(wildcards.datacard)}_rebin1.pkl"
    output:
        f"{config['output_path']}/stat_analysis/{{workspace}}/datacards/{{datacard}}.txt"
    params:
        variable= lambda wildcards: f"SvB_MA.ps_{get_key_for_datacard(wildcards.datacard)}",
        rebin=1,
        output_dir=f"{config['output_path']}/stat_analysis/{{workspace}}/datacards/",
        variable_binning="",
        stat_only="",
        signal=lambda wildcards: f"{get_key_for_datacard(wildcards.datacard).upper()}4b",
        container_wrapper = config['container_wrapper'],
        tag_flags = "",
        syst_file = "",
        metadata = lambda wildcards: f"coffea4bees/stats_analysis/metadata/{get_key_for_datacard(wildcards.datacard).upper()}4b.yml"
    log:
        f"{config['output_path']}/logs/make_combine_inputs_{{workspace}}_{{datacard}}.log"


use rule * from combine

localrules: merge_signals, hist_to_json_signals, run_two_stage_closure_local, make_combine_inputs_local

if not config.get('run_on_condor', True):
    combine_rules = [
        "workspace", "limits", "significance", "likelihood_scan_snapshot",
        "likelihood_scan_chunk", "likelihood_scan", "impacts_initial_fit",
        "impacts_do_fits", "impacts_collect", "split_impacts", "pdf_to_png",
        "gof_data", "gof_toys_chunk", "gof", "fit_diagnostics_bonly", "fit_diagnostics_sb",
        "postfit"
    ]
    workflow._localrules.update(combine_rules)
