# coffea4bees/workflows/Snakefile_PhaseB_1_computeJCM.smk
# Phase B.1: Jet Combinatoric Model Computation Workflow

import os
import copy
import yaml
import re

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "computeJCM")
base_output = config.get('output_path', 'output/computeJCM/')
if not base_output.endswith('/'):
    base_output += '/'
if not base_output.rstrip('/').endswith('computeJCM'):
    JCM_OUTPUT_PATH = os.path.join(base_output, 'computeJCM/')
else:
    JCM_OUTPUT_PATH = base_output

config.setdefault('test', False)

datasets = config.get('jcm_datasets', config.get('jcm', {}).get('datasets', ['data', 'TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic']))
if isinstance(datasets, str):
    datasets = [d.strip() for d in datasets.split(",") if d.strip()]
JCM_DATASETS = datasets

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
    res['friend_file'] = None

    if config.get("test", False):
        if 'runner' not in res or not isinstance(res['runner'], dict):
            res['runner'] = {}
        res['runner']['condor'] = False
        res['runner']['shared_dask'] = False
        res['runner']['run_performance'] = False

    return res

jcm_config_path = f"{JCM_OUTPUT_PATH}analysis_config_noJCM.yml"

### Including modules
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule output_computeJCM:
    input:
        f"{JCM_OUTPUT_PATH}JCM_{tag}/jetCombinatoricModel_SB_{tag}.yml",
        f"{JCM_OUTPUT_PATH}plots_noJCM/plots_done.txt"

DATA_YEAR_ERA = [(str(yr), era) for yr, eras in config['year_eras'].items() for era in eras]
DATA_YEARS = [str(y) for y in config['year_eras'].keys()]
MC_DATASETS = [d for d in JCM_DATASETS if d != 'data']

rule create_noJCM_config:
    input: workflow.configfiles if workflow.configfiles else []
    output: jcm_config_path
    run:
        import yaml, os
        cfg = get_raw_jcm_config()
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as analysis_data_noJCM with:
    input: 
        runner_script = "runner.py",
        config_file = jcm_config_path
    output: f"{JCM_OUTPUT_PATH}singlefiles/hist_data__{{year}}_{{era}}_NoJCM.coffea"
    log: f"{JCM_OUTPUT_PATH}logs/analysis_data__{{year}}_{{era}}.log"
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

use rule analysis_processor from analysis as analysis_MC_noJCM with:
    input: 
        runner_script = "runner.py",
        config_file = jcm_config_path
    output: f"{JCM_OUTPUT_PATH}singlefiles/hist__{{dataset}}__{{year}}_NoJCM.coffea"
    log: f"{JCM_OUTPUT_PATH}logs/analysis__{{dataset}}_{{year}}.log"
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = config['analysis_container_wrapper']

use rule merging_coffea_files from analysis as merge_noJCM with:
    input:
        files = [f"{JCM_OUTPUT_PATH}singlefiles/hist_data__{yr}_{era}_NoJCM.coffea" for yr, era in DATA_YEAR_ERA] + [f"{JCM_OUTPUT_PATH}singlefiles/hist__{ds}__{yr}_NoJCM.coffea" for ds in MC_DATASETS for yr in DATA_YEARS],
        script = "src/tools/merge_coffea_files.py"
    output: f"{JCM_OUTPUT_PATH}histAll_NoJCM.coffea"
    log: f"{JCM_OUTPUT_PATH}logs/merge_noJCM.log"
    params:
        run_performance = False,
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python"),
        input_files = lambda wildcards, input: " ".join([f for f in (input.files if hasattr(input, 'files') else input) if not f.endswith('.py')])

use rule make_JCM from analysis as make_new_JCM with:
    input: f"{JCM_OUTPUT_PATH}histAll_NoJCM.coffea"
    output: f"{JCM_OUTPUT_PATH}JCM_{tag}/jetCombinatoricModel_SB_{tag}.yml"
    params:
        extra_arguments = config.get('jcm_extra_arguments', ""),
        tag = tag,
        output_dir = f"{JCM_OUTPUT_PATH}JCM_{tag}/",
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: f"{JCM_OUTPUT_PATH}logs/make_JCM.log"

use rule make_plots from analysis as make_plots_noJCM with:
    input:
        coffea_file = f"{JCM_OUTPUT_PATH}histAll_NoJCM.coffea",
        metadata_file = config.get('jcm_plot_config', config.get('jcm', {}).get('plot_config', config.get('plot_config', "coffea4bees/plots/metadata/plots_JCM_ttHbb.yml"))),
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{JCM_OUTPUT_PATH}plots_noJCM/plots_done.txt"
    params:
        output_dir = f"{JCM_OUTPUT_PATH}plots_noJCM/",
        metadata = config.get('jcm_plot_config', config.get('jcm', {}).get('plot_config', config.get('plot_config', "coffea4bees/plots/metadata/plots_JCM_ttHbb.yml"))),
        extra_arguments = "-s xW -f png",
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: f"{JCM_OUTPUT_PATH}logs/make_plots_noJCM.log"
