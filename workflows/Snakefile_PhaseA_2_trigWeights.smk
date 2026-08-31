import os
import copy
import yaml

# Fallback defaults for backwards compatibility or running direct
config.setdefault('output_path', "output/trigger_weights/")
config.setdefault('test', False)

datasets = config.get('dataset', config.get('datasets', ['ttHbb']))
if isinstance(datasets, str):
    # Support comma-separated or single dataset strings
    datasets = [d.strip() for d in datasets.split(",") if d.strip()]
config['dataset'] = datasets

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

is_run2 = any(str(y).startswith("UL") for y in years)

include: "helpers/common.smk"

def get_raw_trigger_weights_config():
    # Read from trigger_weights block or analysis_config block if provided in YAML
    res = resolve_config_section(config, primary_key='trigger_weights', fallback_keys=['analysis_config', 'analysis'])
    res.setdefault('processor', "coffea4bees/analysis/processors/processor_trigger_weights.py")
    
    # Dataset location: prioritize explicit trigger_weights_dataset_location, then config['dataset_location']
    if config.get('trigger_weights_dataset_location'):
        res['dataset_location'] = config['trigger_weights_dataset_location']
    elif config.get('dataset_location') and config['dataset_location'].endswith(('.yml', '.yaml')):
        res['dataset_location'] = config['dataset_location']
    elif 'dataset_location' not in res or not res['dataset_location']:
        default_ds_loc = "coffea4bees/metadata/datasets/archive/Run2_2024_v2/" if is_run2 else "coffea4bees/metadata/datasets/"
        res['dataset_location'] = config.get('dataset_location', default_ds_loc)

    # Runner settings
    if 'runner' not in res or not isinstance(res['runner'], dict):
        res['runner'] = copy.deepcopy(config.get('runner', {}))
    res['runner'].setdefault('write_coffea_output', False)

    # Config block for processor
    if 'config' not in res or not isinstance(res['config'], dict):
        res['config'] = {}
    res['config'].setdefault('make_classifier_input', f"{config['output_path']}trigger_weights/")
    if 'use_vectorized' not in res['config']:
        res['config']['use_vectorized'] = not is_run2
    if 'tagger' not in res['config']:
        res['config']['tagger'] = "DeepJet" if is_run2 else "PNet"

    return res

trig_config_path = f"{config['output_path']}trigger_weights/trigger_weights_config.yml"

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule all_trigger_weights:
    input:
        f"{config['output_path']}trigger_weights/trigger_weights_friends.json"

def get_trigger_weights_config_inputs(wildcards):
    inputs = list(workflow.configfiles) if workflow.configfiles else []
    ds_loc = config.get('trigger_weights_dataset_location', config.get('dataset_location', ''))
    if isinstance(ds_loc, str) and ds_loc.endswith(('.yml', '.yaml')):
        inputs.append(ds_loc)
    return inputs

rule create_trigger_weights_config:
    input: get_trigger_weights_config_inputs
    output: trig_config_path
    run:
        import yaml, os
        cfg = get_raw_trigger_weights_config()
        if config.get("test", False):
            if 'runner' not in cfg or not isinstance(cfg['runner'], dict):
                cfg['runner'] = {}
            cfg['runner']['condor'] = False
            cfg['runner']['shared_dask'] = False
            cfg['runner']['run_performance'] = False
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as analysis_trigger_weights with:
    input:
        runner_script = "runner.py",
        config_file = trig_config_path
    output: f"{config['output_path']}trigger_weights/trigger_weights__{{dataset}}__{{year}}.json"
    log: f"{config['output_path']}logs/analysis_trigger_weights__{{dataset}}__{{year}}.log"
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = config['analysis_container_wrapper']

rule merge_friendtree_json:
    input: expand(f"{config['output_path']}trigger_weights/trigger_weights__{{dataset}}__{{year}}.json", dataset=config['dataset'], year=years)
    output: f"{config['output_path']}trigger_weights/trigger_weights_friends.json"
    log: f"{config['output_path']}logs/merge_friendtree_json.log"
    shell:
        """
        ./run_container python -m src.friendtrees.merge_friend_meta \
            -i {input} \
            -o {output}
        """