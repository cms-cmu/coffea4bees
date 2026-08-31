import os
import copy
import yaml

# Fallback defaults for backwards compatibility or running direct
config.setdefault('output_path', "output/skimmer/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('skimmer_config', "coffea4bees/skimmer/metadata/HH4b.yml")
config.setdefault('processor', "coffea4bees/skimmer/processor/skimmer_4b.py")
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
raw_datasets = config.get('dataset', config.get('datasets', ['ttHbb']))
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

skimmer_dataset_location = config.get('skimmer_dataset_location', config.get('dataset_location', \"coffea4bees/metadata/datasets/\"))

def get_raw_skimmer_config():
    skimmer_file = config.get('skimmer_config', "coffea4bees/skimmer/metadata/HH4b.yml")
    base_cfg = {}
    if isinstance(skimmer_file, str) and os.path.exists(skimmer_file):
        with open(skimmer_file, 'r') as f:
            base_cfg = yaml.safe_load(f) or {}

    user_skimmer = config.get('skimmer', {})
    if isinstance(user_skimmer, str) and os.path.exists(user_skimmer):
        with open(user_skimmer, 'r') as f:
            user_skimmer = yaml.safe_load(f) or {}

    res = copy.deepcopy(base_cfg)
    if isinstance(user_skimmer, dict):
        if 'runner' in user_skimmer and isinstance(user_skimmer['runner'], dict):
            res.setdefault('runner', {}).update(user_skimmer['runner'])
        if 'config' in user_skimmer and isinstance(user_skimmer['config'], dict):
            res.setdefault('config', {}).update(user_skimmer['config'])
        for k, v in user_skimmer.items():
            if k not in ('runner', 'config'):
                res[k] = v

    res.setdefault('processor', config.get('processor', "coffea4bees/skimmer/processor/skimmer_4b.py"))
    res['dataset_location'] = skimmer_dataset_location

    if 'runner' not in res or not isinstance(res['runner'], dict):
        res['runner'] = copy.deepcopy(config.get('runner', {}))

    return res

skimmer_config_path = f"{config['output_path']}skimmer/skimmer_config.yml"

mc_targets = expand(
    f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml",
    dataset=mc_datasets,
    year=years
)

data_targets = [
    f"{config['output_path']}skimmer/picoaod_data__{y}_{era}.yml"
    for y, era in data_year_era_pairs
]

all_skim_files = mc_targets + data_targets
modified_datasets_path = f"{config['output_path']}modified_datasets/modified_datasets.yml"

rule all_skims:
    input:
        modified_datasets_path

rule create_skimmer_config:
    input: workflow.configfiles if workflow.configfiles else []
    output: skimmer_config_path
    run:
        import yaml, os
        cfg = get_raw_skimmer_config()
        if config.get("test", False):
            if 'runner' not in cfg or not isinstance(cfg['runner'], dict):
                cfg['runner'] = {}
            cfg['runner']['condor'] = False
            cfg['runner']['shared_dask'] = False
            cfg['runner']['run_performance'] = False
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

if mc_datasets:
    rule skimms_mc:
        input:
            runner_script = "runner.py",
            config_file = skimmer_config_path
        output: f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml"
        log: f"{config['output_path']}logs/skimmer_dataset_{{dataset}}__{{year}}.log"
        params:
            output_dir = f"{config['output_path']}skimmer/",
            dataset_location = config['dataset_location'],
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                "-t" if config.get("test", False) else "",
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper'],
            python_bin = config['python_bin']
        shell:
            """
            set -eo pipefail
            mkdir -p {params.output_dir} $(dirname {log})

            {params.run_container_wrapper} {params.python_bin} runner.py {input.config_file} \
                --processor coffea4bees/skimmer/processor/skimmer_4b.py \
                -s \
                --output $(basename {output}) \
                --output-path {params.output_dir} \
                --datasets {wildcards.dataset} \
                --years {wildcards.year} \
                --metadata {params.dataset_location} \
                {params.extra_arguments} 2>&1 | tee {log}
            """

if include_data:
    rule skimms_data:
        input:
            runner_script = "runner.py",
            config_file = skimmer_config_path
        output: f"{config['output_path']}skimmer/picoaod_data__{{year}}_{{era}}.yml"
        log: f"{config['output_path']}logs/skimmer_data__{{year}}_{{era}}.log"
        params:
            output_dir = f"{config['output_path']}skimmer/",
            dataset_location = config['dataset_location'],
            extra_arguments = lambda wildcards: " ".join(filter(None, [
                "-t" if config.get("test", False) else "",
                config.get("additional_parameters", "")
            ])),
            run_container_wrapper = config['analysis_container_wrapper'],
            python_bin = config['python_bin']
        shell:
            """
            set -eo pipefail
            mkdir -p {params.output_dir} $(dirname {log})

            {params.run_container_wrapper} {params.python_bin} runner.py {input.config_file} \
                --processor coffea4bees/skimmer/processor/skimmer_4b.py \
                -s \
                --output $(basename {output}) \
                --output-path {params.output_dir} \
                --datasets data \
                --years {wildcards.year} \
                --eras {wildcards.era} \
                --metadata {params.dataset_location} \
                {params.extra_arguments} 2>&1 | tee {log}
            """

rule modify_datasets:
    input: all_skim_files
    output: modified_datasets_path
    log: f"{config['output_path']}logs/modify_datasets.log"
    params:
        dataset_location = config['dataset_location'],
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin']
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        echo "Modifying datasets file to point to skimmer outputs" 2>&1 | tee {log}
        {params.run_container_wrapper} {params.python_bin} src/tools/merge_yaml_datasets.py \
            -m {params.dataset_location} \
            -f {input} \
            -o {output} 2>&1 | tee -a {log}
        """
