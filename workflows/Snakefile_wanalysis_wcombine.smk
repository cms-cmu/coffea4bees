from datetime import datetime
import os

include: "helpers/common.smk"

import shutil
_roc = config.setdefault('run_on_condor', shutil.which("condor_submit") is not None)
config['run_on_condor'] = False #str(_roc).lower() not in ('false', '0', 'no')

if config['mode'] == "lowpt":
    config.setdefault('label', "lowpt_SvB_LWP")
    config.setdefault('output_path', "output/lowpt_SvB_LWP/")
    config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_lowpt_2024_v2.yml")
    config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b_lowpt.py")
    config.setdefault('friend_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/friends_HH4b_lowpt.yml")
    config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_lowpt.yml")
    config.setdefault('combine_flags', "--three_tag lowpt_threeTag --four_tag lowpt_fourTag --blind")

elif config['mode'] == "nominal":
    config.setdefault('label', "nominal_wNewSvB")
    config.setdefault('output_path', "output/nominal_wNewSvB/")
    config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
    config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
    config.setdefault('friend_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/friends_HH4b.yml")
    config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
    config.setdefault('combine_flags', "--blind")


config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
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
# config.setdefault('combine_outdir', "datacards/HH4b_fine")
config.setdefault('combine_outdir', "datacards/HH4b_fine")
config.setdefault('channels', {
    'HH4b': {
        'signallabel': "ggHH_kl_1_kt_1_13p0TeV_hbbhbb",
        'othersignal': "ggHH_kl_0_kt_1_13p0TeV_hbbhbb ggHH_kl_2p45_kt_1_13p0TeV_hbbhbb ggHH_kl_5_kt_1_13p0TeV_hbbhbb", # qqHH_CV_1_C2V_1_kl_1_13p0TeV_hbbhbb qqHH_CV_m0p758_C2V_1p44_kl_m19p3_13p0TeV_hbbhbb qqHH_CV_m1p6_C2V_2p72_kl_m1p36_13p0TeV_hbbhbb qqHH_CV_1p74_C2V_1p37_kl_14p4_13p0TeV_hbbhbb qqHH_CV_m0p962_C2V_0p959_kl_m1p43_13p0TeV_hbbhbb qqHH_CV_m1p83_C2V_3p57_kl_m3p39_13p0TeV_hbbhbb qqHH_CV_2p12_C2V_3p87_kl_m5p96_13p0TeV_hbbhbb qqHH_CV_m0p012_C2V_0p03_kl_10p2_13p0TeV_hbbhbb qqHH_CV_m1p21_C2V_1p94_kl_m0p94_13p0TeV_hbbhbb",
        'variable': "SvB_MA.ps_hh_fine",
        'signal': "GluGluToHHTo4B_cHHH1",
    },
    'ZZ4b': {
        'signallabel': "ZZ_bbbb",
        'othersignal': "ggHH_kl_1_kt_1_13p0TeV_hbbhbb ZH_bbbb",
        'variable': "SvB_MA.ps_zz",
        'signal': "ZZ4b",
    },
    'ZH4b': {
        'signallabel': "ZH_bbbb",
        'othersignal': "ggHH_kl_1_kt_1_13p0TeV_hbbhbb ZZ_bbbb",
        'variable': "SvB_MA.ps_zh",
        'signal': "ZH4b",
    },
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

combine_config_HH4b = config.copy()
combine_config_HH4b["output_path"] = os.path.join(config["output_path"], "stat_analysis/HH4b/")

combine_config_ZZ4b = config.copy()
combine_config_ZZ4b["output_path"] = os.path.join(config["output_path"], "stat_analysis/ZZ4b/")

combine_config_ZH4b = config.copy()
combine_config_ZH4b["output_path"] = os.path.join(config["output_path"], "stat_analysis/ZH4b/")

module combine_HH4b:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config_HH4b

module combine_ZZ4b:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config_ZZ4b

module combine_ZH4b:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: combine_config_ZH4b

rule all_lowpt:
    input:
        f"{config['output_path']}histAll_{config['label']}.coffea",
        f"{config['output_path']}plots_{config['label']}/RunII/region_SB/selJets_n.pdf",
        f"{config['output_path']}stat_analysis/HH4b/limits/datacard_limits__{config['channels']['HH4b']['signallabel']}.json",
        f"{config['output_path']}stat_analysis/HH4b/postfit/datacard_postfit__{config['channels']['HH4b']['signallabel']}.pdf",
        f"{config['output_path']}stat_analysis/ZZ4b/limits/datacard_limits__{config['channels']['ZZ4b']['signallabel']}.json",
        f"{config['output_path']}stat_analysis/ZZ4b/postfit/datacard_postfit__{config['channels']['ZZ4b']['signallabel']}.pdf",
        f"{config['output_path']}stat_analysis/ZH4b/limits/datacard_limits__{config['channels']['ZH4b']['signallabel']}.json",
        f"{config['output_path']}stat_analysis/ZH4b/postfit/datacard_postfit__{config['channels']['ZH4b']['signallabel']}.pdf",
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
        run_on_condor = True,
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
        run_on_condor = True,
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


CLOSURE_BASE = "reana_outputs/coffea4bees_20250605_0dc846dc_unblinded_ext_ZZZH/closureFits/ULHH_kfold/3bDvTMix4bDvT/SvB_MA/rebin1/SR"

use rule make_combine_inputs from stat_analysis as make_combine_inputs_HH4b with:
    input:
        injson = f"{config['output_path']}histAll_{config['label']}.json",
        injsonsyst = list([]),
        bkgsyst = f"{CLOSURE_BASE}/hh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin1.pkl"
    output: f"{config['output_path']}stat_analysis/HH4b/datacards/datacard__HH4b.txt"
    params:
        variable = f"{config['channels']['HH4b']['variable']}",
        syst_file = "",
        rebin = 1,
        metadata = "coffea4bees/stats_analysis/metadata/HH4b.yml",
        output_dir = f"{config['output_path']}stat_analysis/HH4b/datacards/",
        variable_binning = "",
        stat_only = "--stat_only",
        signal = "HH4b",
        tag_flags = config['combine_flags'],
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}logs/make_combine_inputs_HH4b.log"

use rule make_combine_inputs from stat_analysis as make_combine_inputs_ZZ4b with:
    input:
        injson = f"{config['output_path']}histAll_{config['label']}.json",
        injsonsyst = list([]),
        bkgsyst = f"{CLOSURE_BASE}/zz/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zz_rebin1.pkl"
    output: f"{config['output_path']}stat_analysis/ZZ4b/datacards/datacard__ZZ4b.txt"
    params:
        variable = f"{config['channels']['ZZ4b']['variable']}",
        syst_file = "",
        rebin = 1,
        metadata = "coffea4bees/stats_analysis/metadata/ZZ4b.yml",
        output_dir = f"{config['output_path']}stat_analysis/ZZ4b/datacards/",
        variable_binning = "",
        stat_only = "--stat_only",
        signal = "ZZ4b",
        tag_flags = config['combine_flags'],
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}logs/make_combine_inputs_ZZ4b.log"

use rule make_combine_inputs from stat_analysis as make_combine_inputs_ZH4b with:
    input:
        injson = f"{config['output_path']}histAll_{config['label']}.json",
        injsonsyst = list([]),
        bkgsyst = f"{CLOSURE_BASE}/zh/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_zh_rebin1.pkl"
    output: f"{config['output_path']}stat_analysis/ZH4b/datacards/datacard__ZH4b.txt"
    params:
        variable = f"{config['channels']['ZH4b']['variable']}",
        syst_file = "",
        rebin = 1,
        metadata = "coffea4bees/stats_analysis/metadata/ZH4b.yml",
        output_dir = f"{config['output_path']}stat_analysis/ZH4b/datacards/",
        variable_binning = "",
        stat_only = "--stat_only",
        signal = "ZH4b",
        tag_flags = config['combine_flags'],
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}logs/make_combine_inputs_ZH4b.log"


_HH4b_path = f"{config['output_path']}stat_analysis/HH4b"
_ZZ4b_path = f"{config['output_path']}stat_analysis/ZZ4b"
_ZH4b_path = f"{config['output_path']}stat_analysis/ZH4b"
_log_HH4b  = f"{config['output_path']}logs"
_log_ZZ4b  = f"{config['output_path']}logs"
_log_ZH4b  = f"{config['output_path']}logs"

use rule workspace from combine_HH4b as combine_HH4b_workspace with:
    input: f"{_HH4b_path}/datacards/datacard__HH4b.txt"
    output: f"{_HH4b_path}/workspace/datacard__{{signallabel}}.root"
    log:    f"{_log_HH4b}/workspace_HH4b__{{signallabel}}.log"

use rule workspace from combine_ZZ4b as combine_ZZ4b_workspace with:
    input: f"{_ZZ4b_path}/datacards/datacard__ZZ4b.txt"
    output: f"{_ZZ4b_path}/workspace/datacard__{{signallabel}}.root"
    log:    f"{_log_ZZ4b}/workspace_ZZ4b__{{signallabel}}.log"

use rule workspace from combine_ZH4b as combine_ZH4b_workspace with:
    input: f"{_ZH4b_path}/datacards/datacard__ZH4b.txt"
    output: f"{_ZH4b_path}/workspace/datacard__{{signallabel}}.root"
    log:    f"{_log_ZH4b}/workspace_ZH4b__{{signallabel}}.log"

use rule limits from combine_HH4b as combine_HH4b_limits with:
    input: f"{_HH4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        txt  = f"{_HH4b_path}/limits/datacard_limits__{{signallabel}}.txt",
        json = f"{_HH4b_path}/limits/datacard_limits__{{signallabel}}.json"
    log: f"{_log_HH4b}/limits_HH4b__{{signallabel}}.log"

use rule limits from combine_ZZ4b as combine_ZZ4b_limits with:
    input: f"{_ZZ4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        txt  = f"{_ZZ4b_path}/limits/datacard_limits__{{signallabel}}.txt",
        json = f"{_ZZ4b_path}/limits/datacard_limits__{{signallabel}}.json"
    log: f"{_log_ZZ4b}/limits_ZZ4b__{{signallabel}}.log"

use rule limits from combine_ZH4b as combine_ZH4b_limits with:
    input: f"{_ZH4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        txt  = f"{_ZH4b_path}/limits/datacard_limits__{{signallabel}}.txt",
        json = f"{_ZH4b_path}/limits/datacard_limits__{{signallabel}}.json"
    log: f"{_log_ZH4b}/limits_ZH4b__{{signallabel}}.log"

use rule fit_diagnostics_bonly from combine_HH4b as combine_HH4b_fit_diagnostics_bonly with:
    input: f"{_HH4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        bonly      = f"{_HH4b_path}/postfit/datacard_fitDiagnostics_bonly__{{signallabel}}.root",
        diff_bonly = f"{_HH4b_path}/postfit/datacard_diffNuisances_bonly__{{signallabel}}.root"
    log: f"{_log_HH4b}/fit_diagnostics_bonly_HH4b__{{signallabel}}.log"

use rule fit_diagnostics_bonly from combine_ZZ4b as combine_ZZ4b_fit_diagnostics_bonly with:
    input: f"{_ZZ4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        bonly      = f"{_ZZ4b_path}/postfit/datacard_fitDiagnostics_bonly__{{signallabel}}.root",
        diff_bonly = f"{_ZZ4b_path}/postfit/datacard_diffNuisances_bonly__{{signallabel}}.root"
    log: f"{_log_ZZ4b}/fit_diagnostics_bonly_ZZ4b__{{signallabel}}.log"

use rule fit_diagnostics_bonly from combine_ZH4b as combine_ZH4b_fit_diagnostics_bonly with:
    input: f"{_ZH4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        bonly      = f"{_ZH4b_path}/postfit/datacard_fitDiagnostics_bonly__{{signallabel}}.root",
        diff_bonly = f"{_ZH4b_path}/postfit/datacard_diffNuisances_bonly__{{signallabel}}.root"
    log: f"{_log_ZH4b}/fit_diagnostics_bonly_ZH4b__{{signallabel}}.log"

use rule fit_diagnostics_sb from combine_HH4b as combine_HH4b_fit_diagnostics_sb with:
    input: f"{_HH4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        sb      = f"{_HH4b_path}/postfit/datacard_fitDiagnostics_sb__{{signallabel}}.root",
        diff_sb = f"{_HH4b_path}/postfit/datacard_diffNuisances_sb__{{signallabel}}.root"
    log: f"{_log_HH4b}/fit_diagnostics_sb_HH4b__{{signallabel}}.log"

use rule fit_diagnostics_sb from combine_ZZ4b as combine_ZZ4b_fit_diagnostics_sb with:
    input: f"{_ZZ4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        sb      = f"{_ZZ4b_path}/postfit/datacard_fitDiagnostics_sb__{{signallabel}}.root",
        diff_sb = f"{_ZZ4b_path}/postfit/datacard_diffNuisances_sb__{{signallabel}}.root"
    log: f"{_log_ZZ4b}/fit_diagnostics_sb_ZZ4b__{{signallabel}}.log"

use rule fit_diagnostics_sb from combine_ZH4b as combine_ZH4b_fit_diagnostics_sb with:
    input: f"{_ZH4b_path}/workspace/datacard__{{signallabel}}.root"
    output:
        sb      = f"{_ZH4b_path}/postfit/datacard_fitDiagnostics_sb__{{signallabel}}.root",
        diff_sb = f"{_ZH4b_path}/postfit/datacard_diffNuisances_sb__{{signallabel}}.root"
    log: f"{_log_ZH4b}/fit_diagnostics_sb_ZH4b__{{signallabel}}.log"

use rule postfit from combine_HH4b as combine_HH4b_postfit with:
    input:
        workspace  = f"{_HH4b_path}/workspace/datacard__{{signallabel}}.root",
        fit_result = f"{_HH4b_path}/postfit/datacard_fitDiagnostics_bonly__{{signallabel}}.root"
    output: f"{_HH4b_path}/postfit/datacard_postfit__{{signallabel}}.pdf"
    params:
        signallabel       = "{signallabel}",
        channel           = "HH4b",
        signal            = "{signallabel}",
        ylog              = "--log",
        plot_script       = config.get("postfit_plot_script", "src/stat_analysis/plots/make_postfit_plot.py"),
        metadata_template = lambda wildcards: config.get("metadata_template", "coffea4bees/stats_analysis/metadata/{channel}.yml")
    log: f"{_log_HH4b}/postfit_HH4b__{{signallabel}}.log"

use rule postfit from combine_ZZ4b as combine_ZZ4b_postfit with:
    input:
        workspace  = f"{_ZZ4b_path}/workspace/datacard__{{signallabel}}.root",
        fit_result = f"{_ZZ4b_path}/postfit/datacard_fitDiagnostics_bonly__{{signallabel}}.root"
    output: f"{_ZZ4b_path}/postfit/datacard_postfit__{{signallabel}}.pdf"
    params:
        signallabel       = "{signallabel}",
        channel           = "ZZ4b",
        signal            = "{signallabel}",
        ylog              = "",
        plot_script       = config.get("postfit_plot_script", "src/stat_analysis/plots/make_postfit_plot.py"),
        metadata_template = lambda wildcards: config.get("metadata_template", "coffea4bees/stats_analysis/metadata/{channel}.yml")
    log: f"{_log_ZZ4b}/postfit_ZZ4b__{{signallabel}}.log"

use rule postfit from combine_ZH4b as combine_ZH4b_postfit with:
    input:
        workspace  = f"{_ZH4b_path}/workspace/datacard__{{signallabel}}.root",
        fit_result = f"{_ZH4b_path}/postfit/datacard_fitDiagnostics_bonly__{{signallabel}}.root"
    output: f"{_ZH4b_path}/postfit/datacard_postfit__{{signallabel}}.pdf"
    params:
        signallabel       = "{signallabel}",
        channel           = "ZH4b",
        signal            = "{signallabel}",
        ylog              = "",
        plot_script       = config.get("postfit_plot_script", "src/stat_analysis/plots/make_postfit_plot.py"),
        metadata_template = lambda wildcards: config.get("metadata_template", "coffea4bees/stats_analysis/metadata/{channel}.yml")
    log: f"{_log_ZH4b}/postfit_ZH4b__{{signallabel}}.log"

localrules: all_lowpt, modify_config_file, analysis_data, analysis_MC, merging_files, make_plots, convert_hist_to_json, make_combine_inputs_HH4b, make_combine_inputs_ZZ4b, make_combine_inputs_ZH4b
