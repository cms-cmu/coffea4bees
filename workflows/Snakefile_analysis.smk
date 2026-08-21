import os
import shutil

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('friend_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/friends_HH4b.yml")
config.setdefault('weights_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/weights_HH4b.yml")
config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
config.setdefault('test', False)
config.setdefault('known_counts', "")

# Parse boolean for test flag
is_test = config.get('test', False)
if isinstance(is_test, str):
    is_test = is_test.lower() in ("true", "1", "yes")
config['test'] = is_test

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))

python_bin = os.getenv("CONTAINER_PYTHON", "python")
config.setdefault('python_bin', python_bin)

if config.get('test', False) or os.getenv("CI"):
    config.setdefault('additional_parameters', "")
else:
    config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")

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

def get_raw_analysis_config():
    import yaml
    raw = config.get('analysis_config', config.get('analysis', {}))
    if isinstance(raw, str) and os.path.exists(raw):
        with open(raw, 'r') as f:
            raw = yaml.safe_load(f) or {}
    elif not isinstance(raw, dict):
        raw = {}
    
    res = dict(raw)
    for k in ['processor', 'dataset_location', 'friend_file', 'weights_file', 'runner', 'config']:
        if k not in res and k in config:
            res[k] = config[k]
    return res

data_config_path = f"{config['output_path']}analysis_config_data.yml"
signal_config_path = f"{config['output_path']}analysis_config_signal.yml"

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
        f"{config['output_path']}cutflow_validation_{config['label']}.txt",
        f"{config['output_path']}cutflow_{config['label']}.yml",
    ]
    return targets

rule all_analysis:
    input: get_analysis_targets

rule create_analysis_config_data:
    input: workflow.configfiles if workflow.configfiles else []
    output: data_config_path
    run:
        import yaml
        cfg = get_raw_analysis_config()
        if config.get("test", False):
            if 'runner' not in cfg or not isinstance(cfg['runner'], dict):
                cfg['runner'] = {}
            cfg['runner']['condor'] = False
            cfg['runner']['shared_dask'] = False
            cfg['runner']['run_performance'] = False
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule create_analysis_config_signal:
    input: workflow.configfiles if workflow.configfiles else []
    output: signal_config_path
    run:
        import yaml, copy
        cfg = copy.deepcopy(get_raw_analysis_config())
        if 'config' not in cfg or not isinstance(cfg['config'], dict):
            cfg['config'] = {}
        cfg['config']['apply_FvT'] = False
        cfg['config']['blind'] = False
        cfg['config']['plot_ttbar_with_weights'] = False
        if config.get("test", False):
            if 'runner' not in cfg or not isinstance(cfg['runner'], dict):
                cfg['runner'] = {}
            cfg['runner']['condor'] = False
            cfg['runner']['shared_dask'] = False
            cfg['runner']['run_performance'] = False
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

if config.get("test", False):
    # Quick Test / CI Mode: single command for all data years/eras and single command for all signals
    use rule analysis_processor from analysis as analysis_data with:
        input: 
            runner_script = "runner.py",
            config_file = data_config_path
        output: f"{config['output_path']}singlefiles/histAll_{config['label']}_data.coffea"
        log: f"{config['output_path']}logs/analysis_{config['label']}_data.log"
        params:
            datasets = "data",
            years = " ".join(DATA_YEARS),
            config = lambda wildcards, input: input.config_file,
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                "-t",
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper']

    use rule analysis_processor from analysis as analysis_MC with:
        input: 
            runner_script = "runner.py",
            config_file = signal_config_path
        output: f"{config['output_path']}singlefiles/histAll_{config['label']}_signals.coffea"
        log: f"{config['output_path']}logs/analysis_{config['label']}_signals.log"
        params:
            datasets = " ".join(config['dataset']),
            years = " ".join(DATA_YEARS),
            config = lambda wildcards, input: input.config_file,
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                "-t",
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper']

    use rule merging_coffea_files from analysis as merging_files with:
        input:
            files = [
                f"{config['output_path']}singlefiles/histAll_{config['label']}_data.coffea",
                f"{config['output_path']}singlefiles/histAll_{config['label']}_signals.coffea"
            ],
            script = "src/tools/merge_coffea_files.py"
        output: f"{config['output_path']}histAll_{config['label']}.coffea"
        params:
            run_performance = False,
            run_container_wrapper = config['analysis_container_wrapper']
        container: None
        log: f"{config['output_path']}logs/merging_files.log"

    rule all_data:
        input: f"{config['output_path']}singlefiles/histAll_{config['label']}_data.coffea"

    rule all_signals:
        input: f"{config['output_path']}singlefiles/histAll_{config['label']}_signals.coffea"

else:
    # Production Mode: granular split per year/era and dataset for distributed cluster batching
    use rule analysis_processor from analysis as analysis_data with:
        input: 
            runner_script = "runner.py",
            config_file = data_config_path
        output: f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{{year}}_{{era}}.coffea"
        log: f"{config['output_path']}logs/analysis_{config['label']}_data__{{year}}_{{era}}.log"
        params:
            datasets = "data",
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                f"--era {wildcards.era}",
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper']

    use rule analysis_processor from analysis as analysis_MC with:
        input: 
            runner_script = "runner.py",
            config_file = signal_config_path
        output: f"{config['output_path']}singlefiles/histAll_{config['label']}__{{dataset}}__{{year}}.coffea"
        log: f"{config['output_path']}logs/analysis_{config['label']}_{{dataset}}_{{year}}.log"
        params:
            datasets = lambda wildcards: wildcards.dataset,
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper']

    use rule merging_coffea_files from analysis as merging_files with:
        input:
            files = [f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA] + expand("{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea", output_path=config['output_path'], dataset=config['dataset'], year=DATA_YEARS),
            script = "src/tools/merge_coffea_files.py"
        output: f"{config['output_path']}histAll_{config['label']}.coffea"
        params:
            run_performance = False,
            run_container_wrapper = config['analysis_container_wrapper']
        container: None
        log: f"{config['output_path']}logs/merging_files.log"

    rule all_data:
        input: [f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA]

    rule all_signals:
        input: expand("{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea", output_path=config['output_path'], dataset=config['dataset'], year=DATA_YEARS)

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
        extra_arguments = "-s xW --year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
        png_cores = 4,
        run_container_wrapper = config['analysis_container_wrapper']
    container: None

use rule check_cutflow from analysis with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea"
    output:
        validation_txt = f"{config['output_path']}cutflow_validation_{config['label']}.txt",
        cutflow_yml = f"{config['output_path']}cutflow_{config['label']}.yml"
    log: f"{config['output_path']}logs/cutflow_validation_{config['label']}.log"
    params:
        known_counts = lambda wildcards: config.get("known_counts", ""),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = config['analysis_container_wrapper']
    container: None

localrules: create_analysis_config_data, create_analysis_config_signal, analysis_data, analysis_MC, merging_files, make_plots, check_cutflow
