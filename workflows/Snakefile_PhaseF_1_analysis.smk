import os
import shutil

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
config.setdefault('analysis_config', "coffea4bees/analysis/metadata/HH4b_2024_v2.yml")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('friend_file', "coffea4bees/metadata/friends/friends_HH4b.yml")
config.setdefault('weights_file', "coffea4bees/metadata/weights/weights_HH4b.yml")
config.setdefault('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
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

if config.get('additional_parameters') is not None:
    pass
elif config.get('test', False) or os.getenv("CI"):
    config.setdefault('additional_parameters', "")
elif "bridges2" in os.uname().nodename or "psc" in os.uname().nodename or "/ocean/" in os.getcwd():
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

SYNTHETIC_DATA_PREFIXES = ('mixeddata', 'synthetic_data', 'datamixed', 'data_3b_for_mixed')
DATA_DATASETS = [d for d in config['dataset'] if d == 'data' or any(d.startswith(p) for p in SYNTHETIC_DATA_PREFIXES)]
MC_DATASETS = [d for d in config['dataset'] if d not in DATA_DATASETS]

DATA_YEARS = [str(y) for y in config['year_eras'].keys()]
DATA_YEAR_ERA = [(str(yr), era) for yr, eras in config['year_eras'].items() for era in eras] if 'data' in DATA_DATASETS else []
DATA_ERA_FILES = [f"{config['output_path']}singlefiles/histAll_{config['label']}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA]

DATA_NOERA_DATASETS = [d for d in DATA_DATASETS if d != 'data']
DATA_NOERA_FILES = expand(
    "{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea",
    output_path=config['output_path'],
    dataset=DATA_NOERA_DATASETS,
    year=DATA_YEARS
) if DATA_NOERA_DATASETS else []

ALL_DATA_FILES = DATA_ERA_FILES + DATA_NOERA_FILES
MC_FILES = expand(
    "{output_path}singlefiles/histAll_" + config['label'] + "__{dataset}__{year}.coffea",
    output_path=config['output_path'],
    dataset=MC_DATASETS,
    year=DATA_YEARS
) if MC_DATASETS else []

include: "helpers/common.smk"

def get_raw_analysis_config():
    return resolve_config_section(config, primary_key='analysis_config', fallback_keys=['analysis'])

analysis_config_path = f"{config['output_path']}analysis_config.yml"

def get_analysis_config_inputs(wildcards):
    inputs = list(workflow.configfiles) if workflow.configfiles else []
    ds_loc = config.get('analysis_dataset_location', config.get('dataset_location', ''))
    if isinstance(ds_loc, str) and ds_loc.endswith(('.yml', '.yaml')):
        inputs.append(ds_loc)
    return inputs

rule create_analysis_config:
    input: get_analysis_config_inputs
    output: analysis_config_path
    run:
        import yaml, os
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

wildcard_constraints:
    year = "|".join([str(y) for y in config['year_eras'].keys()]),
    dataset = "|".join(DATA_NOERA_DATASETS + MC_DATASETS) if (DATA_NOERA_DATASETS + MC_DATASETS) else "none"

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

def get_analysis_targets(wildcards):
    return [
        f"{config['output_path']}histAll_{config['label']}.coffea",
        f"{config['output_path']}plots_{config['label']}/plots_done.txt",
        f"{config['output_path']}cutflow_validation_{config['label']}.txt",
        f"{config['output_path']}cutflow_{config['label']}.yml",
    ]

rule all_analysis:
    input: get_analysis_targets

if config.get("test", False):
    # Quick Test / CI Mode: single command for all data years/eras and single command for all signals
    DATA_TEST_TARGETS = []
    MC_TEST_TARGETS = []

    if DATA_DATASETS:
        DATA_TEST_TARGETS.append(f"{config['output_path']}singlefiles/histAll_{config['label']}_data.coffea")
        use rule analysis_processor from analysis as analysis_data with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
            output: f"{config['output_path']}singlefiles/histAll_{config['label']}_data.coffea"
            log: f"{config['output_path']}logs/analysis_{config['label']}_data.log"
            params:
                datasets = " ".join(DATA_DATASETS),
                years = " ".join(DATA_YEARS),
                config = lambda wildcards, input: input.config_file,
                extra_arguments = lambda wildcards: " ".join(filter(None, [
                    "-t",
                    config.get("additional_parameters", "")
                ])),
                run_container_wrapper = config['analysis_container_wrapper']

    if MC_DATASETS:
        MC_TEST_TARGETS.append(f"{config['output_path']}singlefiles/histAll_{config['label']}_signals.coffea")
        use rule analysis_processor from analysis as analysis_MC with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
            output: f"{config['output_path']}singlefiles/histAll_{config['label']}_signals.coffea"
            log: f"{config['output_path']}logs/analysis_{config['label']}_signals.log"
            params:
                datasets = " ".join(MC_DATASETS),
                years = " ".join(DATA_YEARS),
                config = lambda wildcards, input: input.config_file,
                extra_arguments = lambda wildcards: " ".join(filter(None, [
                    "-t",
                    config.get("additional_parameters", "")
                ])),
                run_container_wrapper = config['analysis_container_wrapper']

    use rule merging_coffea_files from analysis as merging_files with:
        input:
            files = DATA_TEST_TARGETS + MC_TEST_TARGETS,
            script = "src/tools/merge_coffea_files.py"
        output: f"{config['output_path']}histAll_{config['label']}.coffea"
        params:
            run_performance = False,
            run_container_wrapper = config['analysis_container_wrapper']
        container: None
        log: f"{config['output_path']}logs/merging_files.log"

    rule all_data:
        input: DATA_TEST_TARGETS

    rule all_signals:
        input: MC_TEST_TARGETS

else:
    # Production Mode: granular split per year/era and dataset for distributed cluster batching
    if DATA_YEAR_ERA:
        use rule analysis_processor from analysis as analysis_data with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
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

    if DATA_NOERA_DATASETS or MC_DATASETS:
        use rule analysis_processor from analysis as analysis_dataset with:
            input: 
                runner_script = "runner.py",
                config_file = analysis_config_path
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
            files = ALL_DATA_FILES + MC_FILES,
            script = "src/tools/merge_coffea_files.py"
        output: f"{config['output_path']}histAll_{config['label']}.coffea"
        params:
            run_performance = False,
            run_container_wrapper = config['analysis_container_wrapper']
        container: None
        log: f"{config['output_path']}logs/merging_files.log"

    rule all_data:
        input: ALL_DATA_FILES

    rule all_signals:
        input: MC_FILES

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

localrules: merging_files, make_plots, check_cutflow
