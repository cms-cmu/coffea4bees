from datetime import datetime
import os
import shutil

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('friend_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/friends_HH4b.yml")
config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
config.setdefault('combine_flags', "--blind")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
config.setdefault('closure_base', "coffea4bees/stats_analysis/files/HIG-24-010/")


config.setdefault('dataset', ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH5', 'ZH4b', 'ZZ4b', 'ggZH4b'])
config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})
config.setdefault('channels', {})
config.setdefault('combine_outdir', "datacards/HH4b_fine")

### Containers (DO NOT CHANGE unless you know what you are doing)
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('combine_container', "docker://gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0")
config.setdefault('container_wrapper', "./run_container combine")


## for combine in cmslpc
_roc = config.setdefault('run_on_condor', shutil.which("condor_submit") is not None)
config['run_on_condor'] = False


# Derive flat year/era and year lists from year_eras
DATA_YEAR_ERA = [(str(yr), era) for yr, eras in config['year_eras'].items() for era in eras]
DATA_YEARS = [str(y) for y in config['year_eras'].keys()]

# Constrain year wildcard to valid year values
wildcard_constraints:
    year = "|".join([str(y) for y in config['year_eras'].keys()]),
    channel = "|".join(config['channels'].keys()) if config['channels'] else "[a-zA-Z0-9_]+"

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

include: "helpers/common.smk"
module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: config


# Generate all output targets dynamically based on channels configured
targets = [
    f"{config['output_path']}histAll_{config['label']}.coffea",
    f"{config['output_path']}plots_{config['label']}/RunII/region_SB/selJets_n.pdf",
]
for channel, ch_config in config['channels'].items():
    signallabel = ch_config.get('signallabel')
    if signallabel:
        targets.extend([
            f"{config['output_path']}stat_analysis/{channel}/limits/datacard_limits__{signallabel}.json",
            f"{config['output_path']}stat_analysis/{channel}/postfit/datacard_postfit__{signallabel}.pdf",
        ])

rule all_lowpt:
    input: targets
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
    input: 
        runner_script = "runner.py",
        config_file = config['analysis_config'],
        processor_script = config['processor'],
        friend_metadata = config['friend_file'],
        datasets_dir = config['dataset_location']
    output: f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{{year}}_{{era}}.coffea"
    log: f"{config['output_path']}logs/analysis_{config['label']}_data__{{year}}_{{era}}.log"
    params:
        datasets = "data",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        processor = config['processor'],
        datasets_file = config['dataset_location'],
        blind = True,
        run_performance = False,
        friends = config['friend_file'],
        weights = config.get('weights_file', ""),
        run_on_condor = True,
        not_do_proxy = False,
        extra_arguments = lambda wildcards: f'"--era {wildcards.era}"',
        run_container_wrapper = "./run_container",
        dashboard_address = ""

use rule analysis_processor from analysis as analysis_MC with:
    input: 
        runner_script = "runner.py",
        config_file = f"{config['output_path']}HH4b_{config['label']}_signal.yml",
        processor_script = config['processor'],
        friend_metadata = config['friend_file'],
        datasets_dir = config['dataset_location']
    output: f"{config['output_path']}singlefiles/histAll_{config['label']}__{{dataset}}__{{year}}.coffea"
    log: f"{config['output_path']}logs/analysis_{config['label']}_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        processor = config['processor'],
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = config['friend_file'],
        weights = config.get('weights_file', ""),
        run_on_condor = True,
        not_do_proxy = False,
        extra_arguments = "",
        run_container_wrapper = "./run_container",
        dashboard_address = ""

use rule merging_coffea_files from analysis as merging_files with:
    input:
        files = [f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA] + expand("{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea", output_path=config['output_path'], dataset=config['dataset'], year=DATA_YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: f"{config['output_path']}histAll_{config['label']}.coffea"
    params:
        run_performance = False,
        run_container_wrapper = "./run_container"
    container: config['analysis_container']
    log: f"{config['output_path']}logs/merging_files.log" 

use rule make_plots from analysis as make_plots with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea",
        metadata_file = config['plot_config'],
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{config['output_path']}plots_{config['label']}/RunII/region_SB/selJets_n.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_{config['label']}/",
        metadata = config['plot_config'],
        extra_arguments = "-s xW ",
        png_cores = 4,
        run_container_wrapper = "./run_container"

use rule convert_hist_to_json from stat_analysis with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea",
        script = "coffea4bees/stats_analysis/convert_hist_to_json.py"
    output: f"{config['output_path']}histAll_{config['label']}.json"
    params:
        syst_flag=lambda wildcards, input: "--histos " + " ".join([ch_config['variable'].replace(".ps_", "_ps_") for ch_config in config['channels'].values()]) if config['channels'] else ""
    log: f"{config['output_path']}logs/convert_hist_to_json_{config['label']}.log"


use rule make_combine_inputs from stat_analysis as make_combine_inputs_channel with:
    input:
        injson = f"{config['output_path']}histAll_{config['label']}.json",
        injsonsyst = list([]),
        bkgsyst = lambda wildcards: f"{config['closure_base']}/{config['channels'][wildcards.channel]['closure_subdir']}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{config['channels'][wildcards.channel]['closure_subdir']}_rebin1.pkl",
        script = "coffea4bees/stats_analysis/make_combine_inputs.py",
        metadata_file = lambda wildcards: f"coffea4bees/stats_analysis/metadata/{wildcards.channel}.yml"
    output: f"{config['output_path']}stat_analysis/{{channel}}/datacards/datacard__{{channel}}.txt"
    params:
        variable = lambda wildcards: config['channels'][wildcards.channel]['variable'],
        syst_file = "",
        rebin = 1,
        metadata = lambda wildcards: f"coffea4bees/stats_analysis/metadata/{wildcards.channel}.yml",
        output_dir = lambda wildcards: f"{config['output_path']}stat_analysis/{wildcards.channel}/datacards/",
        variable_binning = "",
        stat_only = "--stat_only",
        signal = lambda wildcards: wildcards.channel,
        tag_flags = config['combine_flags'],
        container_wrapper = config['container_wrapper']
    log: f"{config['output_path']}logs/make_combine_inputs_{{channel}}.log"

# Import all rules from combine module
use rule * from combine as *

localrules: all_lowpt, modify_config_file, analysis_data, analysis_MC, merging_files, make_plots, convert_hist_to_json, make_combine_inputs_channel
