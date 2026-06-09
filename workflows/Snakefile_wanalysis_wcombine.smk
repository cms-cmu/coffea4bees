from datetime import datetime
import os

include: "helpers/common.smk"

import shutil
_roc = config.setdefault('run_on_condor', shutil.which("condor_submit") is not None)
config['run_on_condor'] = str(_roc).lower() not in ('false', '0', 'no')

if config['mode'] == "lowpt":
    config.setdefault('label', "lowpt_wfixedSvB")
    config.setdefault('output_path', "output/lowpt_wfixedSvB/")
    config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_lowpt_2024_v2.yml")
    config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b_lowpt.py")
    config.setdefault('friend_file', "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/friends_HH4b_lowpt.yml")
    config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_lowpt.yml")
    config.setdefault('combine_flags', "--three_tag lowpt_threeTag --four_tag lowpt_fourTag --blind")

elif config['mode'] == "nominal":
    config.setdefault('label', "nominal_wNewSvB")
    config.setdefault('output_path', "output/nominal_wNewSvB/")
    config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
    config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
    config.setdefault('friend_file', "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/friends_HH4b.yml")
    config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
    config.setdefault('combine_flags', "--blind")


config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/")
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('combine_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0")
config.setdefault('container_wrapper', "./run_container combine")
config.setdefault('dataset', ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH5', 'ZH4b', 'ZZ4b', 'ggZH4b'])
config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})
config.setdefault('channels', {
    'HH4b': {
        'signallabel': "ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
        'othersignal': "ggHH_kl_0_kt_1_13p0TeV_hbbhbb ggHH_kl_2p45_kt_1_13p0TeV_hbbhbb ggHH_kl_5_kt_1_13p0TeV_hbbhbb", # qqHH_CV_1_C2V_1_kl_1_13p0TeV_hbbhbb qqHH_CV_m0p758_C2V_1p44_kl_m19p3_13p0TeV_hbbhbb qqHH_CV_m1p6_C2V_2p72_kl_m1p36_13p0TeV_hbbhbb qqHH_CV_1p74_C2V_1p37_kl_14p4_13p0TeV_hbbhbb qqHH_CV_m0p962_C2V_0p959_kl_m1p43_13p0TeV_hbbhbb qqHH_CV_m1p83_C2V_3p57_kl_m3p39_13p0TeV_hbbhbb qqHH_CV_2p12_C2V_3p87_kl_m5p96_13p0TeV_hbbhbb qqHH_CV_m0p012_C2V_0p03_kl_10p2_13p0TeV_hbbhbb qqHH_CV_m1p21_C2V_1p94_kl_m0p94_13p0TeV_hbbhbb",
        'variable': "SvB_MA.ps_hh",
        'signal': "GluGluToHHTo4B_cHHH1"
    }
})

# Derive flat year/era and year lists from year_eras
DATA_YEAR_ERA = [(yr, era) for yr, eras in config['year_eras'].items() for era in eras]
DATA_YEARS = list(config['year_eras'].keys())

# Constrain year wildcard to valid year values (avoids ambiguity with underscores in dataset names)
wildcard_constraints:
    year = "|".join(config['year_eras'].keys())

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

combine_config = config.copy()
combine_config["output_path"] = os.path.join(config["output_path"], "stat_analysis/HH4b/")

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config

rule all_lowpt:
    input:
        f"{config['output_path']}histAll_{config['label']}.coffea",
        f"{config['output_path']}plots_{config['label']}/RunII/region_SB/selJets_n.pdf",
        f"{config['output_path']}stat_analysis/HH4b/limits/datacard_limits__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.json",
        f"{config['output_path']}stat_analysis/HH4b/postfit/datacard_postfit__ggHH_kl_1_kt_1_13p0TeV_hbbhbb.pdf"
    params:
        output_dir = f"{datetime.now().strftime('%Y%m%d')}_{config['label']}/"
    shell:
        """
        echo "Copying results to eos"
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d www/HH4b/Plots/{params.output_dir} -t
        """

rule modify_config_file:
    input: config['analysis_config']
    output:
        f"{config['output_path']}HH4b_{config['label']}_signal.yml"
    shell:
        """
        sed -e 's|apply_FvT: .*|apply_FvT: false|' -e 's|blind:.*|blind: false|' -e 's|plot_ttbar_with_weights: true|plot_ttbar_with_weights: false|' {input} > {output}
        """

use rule analysis_processor from analysis as analysis_data with:
    input: config['analysis_config']
    output: f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{{year}}_{{era}}.coffea"
    log: f"{config['output_path']}logs/analysis_{config['label']}_data__{{year}}_{{era}}.log"
    params:
        datasets = "data",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input[0],
        processor = config['processor'],
        datasets_file = config['dataset_location'],
        blind = True,
        run_performance = False,
        friends = config['friend_file'],
        run_on_condor = lambda wildcards: config['run_on_condor'],
        not_do_proxy = False,
        extra_arguments = lambda wildcards: f'"--era {wildcards.era}"',
        run_container_wrapper = "./run_container",
        dashboard_address = ""

use rule analysis_processor from analysis as analysis_MC with:
    input: f"{config['output_path']}HH4b_{config['label']}_signal.yml"
    output: f"{config['output_path']}singlefiles/histAll_{config['label']}__{{dataset}}__{{year}}.coffea"
    log: f"{config['output_path']}logs/analysis_{config['label']}_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input[0],
        processor = config['processor'],
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = config['friend_file'],
        run_on_condor = lambda wildcards: config['run_on_condor'],
        not_do_proxy = False,
        extra_arguments = "",
        run_container_wrapper = "./run_container",
        dashboard_address = ""

use rule merging_coffea_files from analysis as merging_files with:
    input: [f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA] + expand("{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea", output_path=config['output_path'], dataset=config['dataset'], year=DATA_YEARS)
    output: f"{config['output_path']}histAll_{config['label']}.coffea"
    params:
        run_performance = False
    container: config['analysis_container']
    log: f"{config['output_path']}logs/merging_files.log" 

use rule make_plots from analysis as make_plots with:
    input: f"{config['output_path']}histAll_{config['label']}.coffea"
    output: f"{config['output_path']}plots_{config['label']}/RunII/region_SB/selJets_n.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_{config['label']}/",
        metadata = config['plot_config'],
        extra_arguments = "-s xW ",
        png_cores = 4

use rule convert_hist_to_json from stat_analysis with:
    input: f"{config['output_path']}histAll_{config['label']}.coffea"
    output: f"{config['output_path']}histAll_{config['label']}.json"
    params:
        syst_flag="--histos SvB_MA.ps_hh SvB_MA.ps_hh_fine SvB_MA.ps_zz SvB_MA.ps_zh"
    log: f"{config['output_path']}logs/convert_hist_to_json_{config['label']}.log"


use rule make_combine_inputs from stat_analysis with:
    input:
        injson = f"{config['output_path']}histAll_{config['label']}.json",
        injsonsyst = list([]), 
        bkgsyst = f"reana_outputs/coffea4bees_20250616_af478bd_unblind_boostedVeto/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.pkl"
    output: f"{config['output_path']}stat_analysis/HH4b/datacards/datacard__HH4b.txt"
    params:
        variable= "SvB_MA.ps_hh",
        syst_file = "",
        rebin=1,
        metadata="coffea4bees/stats_analysis/metadata/HH4b.yml",
        output_dir=f"{config['output_path']}stat_analysis/HH4b/datacards/",
        variable_binning="",
        stat_only="--stat_only",
        signal="HH4b",
        tag_flags= config['combine_flags'],
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}logs/make_combine_inputs_HH4b.log"


use rule * from combine

localrules: all_lowpt, modify_config_file, analysis_data, analysis_MC, merging_files, make_plots, convert_hist_to_json, make_combine_inputs

if not config.get('run_on_condor', True) and config.get('run_local_without_container', False):
    combine_rules = [
        "workspace", "limits", "significance", "likelihood_scan_snapshot",
        "likelihood_scan_chunk", "likelihood_scan", "impacts_initial_fit",
        "impacts_do_fits", "impacts_collect", "split_impacts", "pdf_to_png",
        "gof_data", "gof_toys_chunk", "gof", "fit_diagnostics_bonly", "fit_diagnostics_sb",
        "postfit"
    ]
    workflow._localrules.update(combine_rules)