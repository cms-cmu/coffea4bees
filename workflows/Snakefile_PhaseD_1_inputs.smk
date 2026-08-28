# coffea4bees/workflows/Snakefile_PhaseD_1_inputs.smk
# Phase D.1: SvB Classifier Inputs Generation Workflow

import os
import copy
import yaml

# Fallback defaults for backwards compatibility or running direct
config.setdefault('output_path', "output/classifier_inputs_SvB/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
config.setdefault('processor', "coffea4bees/analysis/processors/processor_HH4b.py")
config.setdefault('test', False)

# Parse boolean for test flag
is_test = config.get('test', False)
if isinstance(is_test, str):
    is_test = is_test.lower() in ("true", "1", "yes")
config['test'] = is_test

# Container and python bin
container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))
python_bin = os.getenv("CONTAINER_PYTHON", "python")
config.setdefault('python_bin', python_bin)

if config.get('test', False) or os.getenv("CI"):
    config.setdefault('additional_parameters', "")
else:
    config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")

# Parse datasets
raw_datasets = (
    config.get('classifier_inputs', {}).get('datasets')
    or config.get('classifier_inputs_datasets')
    or config.get('dataset')
    or config.get('datasets', ['ttHbb'])
)
if isinstance(raw_datasets, str):
    raw_datasets = [d.strip() for d in raw_datasets.split(",") if d.strip()]

include_data = "data" in raw_datasets
mc_datasets = [d for d in raw_datasets if d != "data"]
config['dataset'] = raw_datasets

# Parse years and year_eras
DEFAULT_ERAS = {
    'UL16_preVFP':  ['B', 'C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['B', 'C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
    '2022_preEE':   ['B', 'C', 'D'],
    '2022_EE':      ['E', 'F', 'G'],
    '2023_preBPix': ['C01', 'C02', 'C11', 'C12'],
    '2023_BPix':    ['D'],
}

raw_years = config.get('years', config.get('year', None))
if raw_years is not None:
    if isinstance(raw_years, str):
        years = [y.strip() for y in raw_years.split(",") if y.strip()]
    else:
        years = list(raw_years)
elif 'year_eras' in config and isinstance(config['year_eras'], dict):
    years = list(config['year_eras'].keys())
else:
    years = ['UL18']
config['years'] = years
config['year'] = years

if 'year_eras' in config and isinstance(config['year_eras'], dict):
    year_eras = {
        y: config['year_eras'].get(y, DEFAULT_ERAS.get(y, ['A', 'B', 'C', 'D']))
        for y in years
    }
else:
    year_eras = {
        y: DEFAULT_ERAS.get(y, ['A', 'B', 'C', 'D'])
        for y in years
    }

data_year_era_pairs = [
    (str(y), str(era))
    for y, eras in year_eras.items()
    for era in eras
] if include_data else []

is_run2 = any(str(y).startswith("UL") for y in years)

include: "helpers/common.smk"

def get_raw_svb_inputs_config():
    res = resolve_config_section(config, primary_key='classifier_inputs_svb', fallback_keys=['classifier_inputs', 'analysis_config', 'analysis'])
    res.setdefault('processor', config.get('processor', "coffea4bees/analysis/processors/processor_HH4b.py"))
    
    if config.get('classifier_inputs_dataset_location'):
        res['dataset_location'] = config['classifier_inputs_dataset_location']
    elif config.get('dataset_location') and config['dataset_location'].endswith(('.yml', '.yaml')):
        res['dataset_location'] = config['dataset_location']
    elif 'dataset_location' not in res or not res['dataset_location']:
        default_ds_loc = "coffea4bees/metadata/datasets/archive/Run2_2024_v2/" if is_run2 else "coffea4bees/metadata/datasets/"
        res['dataset_location'] = config.get('dataset_location', default_ds_loc)

    res.setdefault('friend_file', config.get('friend_file', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/friends_ttHbb.yml" if is_run2 else "coffea4bees/metadata/datasets/friends_HH4b.yml"))
    res.setdefault('weights_file', config.get('weights_file', "coffea4bees/metadata/weights/weights_HH4b_2024_v2.yml" if is_run2 else "coffea4bees/metadata/weights/weights_HH4b.yml"))

    if 'runner' not in res or not isinstance(res['runner'], dict):
        res['runner'] = copy.deepcopy(config.get('runner', {}))

    if 'config' not in res or not isinstance(res['config'], dict):
        res['config'] = {}
    res['config'].setdefault('make_classifier_input', f"{config['output_path']}classifier_inputs/")
    res['config'].setdefault('apply_FvT', True)
    res['config'].setdefault('fill_histograms', False)

    return res

ci_config_path = f"{config['output_path']}classifier_inputs/classifier_inputs_config.yml"

mc_json_targets = expand(
    f"{config['output_path']}classifier_inputs/classifier_inputs_dataset_{{dataset}}__{{year}}.json",
    dataset=mc_datasets,
    year=years
)

data_json_targets = [
    f"{config['output_path']}classifier_inputs/classifier_inputs_data__{y}_{era}.json"
    for y, era in data_year_era_pairs
]

all_ci_json_files = mc_json_targets + data_json_targets

rule all_svb_classifier_inputs:
    input:
        f"{config['output_path']}classifier_inputs/classifier_inputs_friends.json"

def get_svb_inputs_config_inputs(wildcards):
    inputs = list(workflow.configfiles) if workflow.configfiles else []
    ds_loc = config.get('classifier_inputs_dataset_location', config.get('dataset_location', ''))
    if isinstance(ds_loc, str) and ds_loc.endswith(('.yml', '.yaml')):
        inputs.append(ds_loc)
    return inputs

rule create_svb_inputs_config:
    input: get_svb_inputs_config_inputs
    output: ci_config_path
    run:
        import yaml, os
        cfg = get_raw_svb_inputs_config()
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

if include_data:
    rule svb_inputs_data:
        input:
            runner_script = "runner.py",
            config_file = ci_config_path
        output: f"{config['output_path']}classifier_inputs/classifier_inputs_data__{{year}}_{{era}}.json"
        log: f"{config['output_path']}logs/classifier_inputs_data__{{year}}_{{era}}.log"
        params:
            datasets = "data",
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                f"--era {wildcards.era}",
                "-t" if config.get("test", False) else "",
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper']
        shell:
            """
            {params.run_container_wrapper} python {input.runner_script} \
                -d {params.datasets} \
                -y {params.years} \
                -c {params.config} \
                {params.extra_arguments} \
                2>&1 | tee -a {log}
            """

if mc_datasets:
    rule svb_inputs_mc:
        input:
            runner_script = "runner.py",
            config_file = ci_config_path
        output: f"{config['output_path']}classifier_inputs/classifier_inputs_dataset_{{dataset}}__{{year}}.json"
        log: f"{config['output_path']}logs/classifier_inputs_dataset_{{dataset}}_{{year}}.log"
        params:
            datasets = lambda wildcards: wildcards.dataset,
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                "-t" if config.get("test", False) else "",
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper']
        shell:
            """
            {params.run_container_wrapper} python {input.runner_script} \
                -d {params.datasets} \
                -y {params.years} \
                -c {params.config} \
                {params.extra_arguments} \
                2>&1 | tee -a {log}
            """

rule merge_svb_inputs_friends:
    input: all_ci_json_files
    output: f"{config['output_path']}classifier_inputs/classifier_inputs_friends.json"
    log: f"{config['output_path']}logs/merge_svb_inputs_friends.log"
    run:
        import json, os
        merged = {}
        for fn in input:
            if os.path.exists(fn):
                with open(fn, 'r') as f:
                    data = json.load(f)
                    merged.update(data)
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            json.dump(merged, f, indent=4)
