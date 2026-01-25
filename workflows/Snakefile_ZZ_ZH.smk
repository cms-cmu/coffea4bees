config = {
    "analysis_container": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest",
    "input_path": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH",
    "output_path": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH",
    "dataset_location": "",
    'dataset': "",
    "combine_container": "",
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
        # "hh": {
        #     "datacard": "datacard_HH4b",
        #     "signallabel": "ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
        #     "othersignal": "ggHH_kl_0_kt_1_13p0TeV_hbbhbb,rggHH_kl_2p45_kt_1_13p0TeV_hbbhbb,rggHH_kl_5_kt_1_13p0TeV_hbbhbb",
        #     "workspace": "hists/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/datacards/HH4b/"
        # },
    }
}

# Include all the combine rules
include: "rules/combine.smk"

# Adding specific rules for analysis
module analysis:
    snakefile: "rules/analysis.smk"
    config: config
module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

def make_outputs(patterns, cases):
    outputs = []
    for pattern in patterns:
        for key, case_info in cases.items():
            # Use the workspace from the case_info directly
            workspace = case_info["workspace"]
            datacard = case_info["datacard"]
            signallabel = case_info["signallabel"]
            
            # Format the pattern with our values
            formatted_pattern = pattern.format(
                workspace=workspace,
                datacard=datacard,
                signallabel=signallabel
            )
            
            outputs.append(formatted_pattern)
    
    return outputs

output_patterns = [
    "{workspace}limits_{datacard}__{signallabel}.txt",
    "{workspace}significance_expected_{datacard}__{signallabel}.txt",
    "{workspace}impacts_combine_{datacard}__{signallabel}_observed.pdf",
    "{workspace}gof_{datacard}__{signallabel}.pdf",
    "{workspace}likelihood_scan_{datacard}__{signallabel}.pdf"
]

rule all:
    input:
        make_outputs(output_patterns, config["cases"])

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

# Override the get_case_param function to work with our dictionary structure
def get_case_param(wildcards, key):
    # For each case in our config
    for case_key, case_info in config["cases"].items():
        # Match by datacard name
        if case_info["datacard"] == wildcards.datacard:
            # If we also need to match workspace, check if it's a substring
            if hasattr(wildcards, 'workspace'):
                # If the workspace directory contains our case workspace directory, it's a match
                if case_info["workspace"].rstrip('/') in wildcards.workspace.rstrip('/'):
                    return case_info[key]
            else:
                # If we don't need to match workspace, just return the value
                return case_info[key]
            
    # If we get here, no match was found
    workspace_val = getattr(wildcards, 'workspace', 'N/A')
    raise ValueError(f"No matching case for datacard={wildcards.datacard}, workspace={workspace_val}")

use rule make_combine_inputs from stat_analysis as make_combine_inputs_local with:
    input:
        injson = f"{config['output_path']}/histAll.json",
        injsonsyst = lambda wildcards: f"{config['output_path']}/histAll_signals_{get_key_for_datacard(wildcards.datacard)}.json",
        bkgsyst = lambda wildcards: f"{config['output_path']}/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/{get_key_for_datacard(wildcards.datacard)}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{get_key_for_datacard(wildcards.datacard)}_rebin1.pkl"
    output:
        f"{config['output_path']}/datacards/{{workspace}}/{{datacard}}.txt"
    params:
        variable= lambda wildcards: f"SvB_MA.ps_{get_key_for_datacard(wildcards.datacard)}",
        rebin=1,
        output_dir=f"{config['output_path']}/datacards/{{workspace}}/",
        variable_binning="",
        stat_only="",
        signal=lambda wildcards: f"{get_key_for_datacard(wildcards.datacard).upper()}4b"
    log:
        f"{config['output_path']}/logs/make_combine_inputs_{{workspace}}_{{datacard}}.log"
