import os
import copy
import yaml

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "computeJCM")
config.setdefault('output_path', 'output/computeJCM/')
config.setdefault('test', False)

datasets = config.get('dataset', config.get('datasets', ['data', 'TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic']))
if isinstance(datasets, str):
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

years = list(config['year_eras'].keys()) if 'year_eras' in config and isinstance(config['year_eras'], dict) else config.get('years', config.get('year', ['UL18']))
if isinstance(years, str):
    years = [years]

tag = config.get('tag', "2024_v2")

include: "helpers/common.smk"

def get_raw_jcm_config():
    res = resolve_config_section(config, primary_key='jcm', fallback_keys=['analysis_config', 'analysis'])
    
    if 'config' not in res or not isinstance(res['config'], dict):
        res['config'] = {}
    res['config']['apply_JCM'] = False
    res['config']['apply_FvT'] = False
    res['config']['run_SvB'] = False
    for k in list(res['config'].keys()):
        if k.startswith('SvB') or k == 'FvT':
            res['config'][k] = None

    if config.get("test", False):
        if 'runner' not in res or not isinstance(res['runner'], dict):
            res['runner'] = {}
        res['runner']['condor'] = False
        res['runner']['shared_dask'] = False
        res['runner']['run_performance'] = False

    return res

jcm_config_path = f"{config['output_path']}analysis_config_noJCM.yml"

### Including modules
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule output_computeJCM:
    input:
        f"{config['output_path']}JCM_{tag}/jetCombinatoricModel_SB_{tag}.yml"

rule create_noJCM_config:
    input: workflow.configfiles if workflow.configfiles else []
    output: jcm_config_path
    run:
        import yaml, os
        cfg = get_raw_jcm_config()
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as analysis_noJCM with:
    input:
        runner_script = "runner.py",
        config_file = jcm_config_path
    output: f"{config['output_path']}histAll_NoJCM.coffea"
    log: f"{config['output_path']}logs/analysis_noJCM.log"
    params:
        datasets = " ".join(config['dataset']),
        years = " ".join([str(y) for y in years]),
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = config['analysis_container_wrapper']

use rule make_JCM from analysis as make_new_JCM with:
    input: f"{config['output_path']}histAll_NoJCM.coffea"
    output: f"{config['output_path']}JCM_{tag}/jetCombinatoricModel_SB_{tag}.yml"
    params:
        extra_arguments = config.get('jcm_extra_arguments', ""),
        tag = tag,
        output_dir = f"{config['output_path']}JCM_{tag}/",
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: f"{config['output_path']}logs/make_JCM.log"