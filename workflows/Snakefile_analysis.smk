import os
import shutil

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('friend_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/friends_HH4b.yml")
config.setdefault('weights_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/weights_HH4b.yml")
config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")
config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
config.setdefault('test', False)
config.setdefault('known_counts', "")

config.setdefault('dataset', ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH5', 'ZH4b', 'ZZ4b', 'ggZH4b'])
config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})

### Containers
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")

DATA_YEAR_ERA = [(str(yr), era) for yr, eras in config['year_eras'].items() for era in eras]
DATA_YEARS = [str(y) for y in config['year_eras'].keys()]

original_basename = os.path.basename(config['analysis_config'])
original_name, _ = os.path.splitext(original_basename)
signal_config_path = f"{config['output_path']}{original_name}_signal.yml"

wildcard_constraints:
    year = "|".join([str(y) for y in config['year_eras'].keys()])

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

include: "helpers/common.smk"

def get_analysis_targets(wildcards):
    targets = [
        f"{config['output_path']}histAll_{config['label']}.coffea",
        f"{config['output_path']}plots_{config['label']}/plots_done.txt",
    ]
    if config.get("known_counts"):
        targets.append(f"{config['output_path']}cutflow_validation_{config['label']}.txt")
    return targets

rule all_analysis:
    input: get_analysis_targets

rule all_data:
    input: [f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA]

rule all_signals:
    input: expand("{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea", output_path=config['output_path'], dataset=config['dataset'], year=DATA_YEARS)


rule modify_config_file:
    input: config['analysis_config']
    output: signal_config_path
    run:
        import yaml
        with open(input[0], 'r') as f:
            data = yaml.safe_load(f)
        
        # Apply signal MC modifications at both root level and nested config block
        if 'apply_FvT' in data: data['apply_FvT'] = False
        if 'config' in data and 'apply_FvT' in data['config']: data['config']['apply_FvT'] = False

        if 'blind' in data: data['blind'] = False
        if 'config' in data and 'blind' in data['config']: data['config']['blind'] = False

        if 'plot_ttbar_with_weights' in data: data['plot_ttbar_with_weights'] = False
        if 'config' in data and 'plot_ttbar_with_weights' in data['config']: data['config']['plot_ttbar_with_weights'] = False
        
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

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
        friends = config['friend_file'],
        weights = config['weights_file'],
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            f"--era {wildcards.era}",
            "-t" if config.get("test", False) else "",
            config["additional_parameters"]
        ])),
        run_container_wrapper = "./run_container"

use rule analysis_processor from analysis as analysis_MC with:
    input: 
        runner_script = "runner.py",
        config_file = signal_config_path,
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
        friends = config['friend_file'],
        weights = config['weights_file'],
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config["additional_parameters"]
        ])),
        run_container_wrapper = "./run_container"

use rule merging_coffea_files from analysis as merging_files with:
    input:
        files = [f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA] + expand("{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea", output_path=config['output_path'], dataset=config['dataset'], year=DATA_YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: f"{config['output_path']}histAll_{config['label']}.coffea"
    params:
        run_performance = False,
        run_container_wrapper = "./run_container"
    container: None
    log: f"{config['output_path']}logs/merging_files.log" 

use rule make_plots from analysis with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea",
        metadata_file = config['plot_config'],
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{config['output_path']}plots_{config['label']}/plots_done.txt"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_{config['label']}/",
        metadata = config['plot_config'],
        extra_arguments = "-s xW " + (" ".join([f"--year {y}" for y in DATA_YEARS]) if len(DATA_YEARS) == 1 else ""),
        png_cores = 4,
        run_container_wrapper = "./run_container"
    container: None

use rule check_cutflow from analysis with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea"
    output: f"{config['output_path']}cutflow_validation_{config['label']}.txt"
    log: f"{config['output_path']}logs/cutflow_validation_{config['label']}.log"
    params:
        known_counts = lambda wildcards: config.get("known_counts", ""),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        run_container_wrapper = "./run_container"
    container: None

localrules: modify_config_file, analysis_data, analysis_MC, merging_files, make_plots, check_cutflow
