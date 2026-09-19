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
jcm_file_path = f"{JCM_OUTPUT_PATH}JCM_{tag}/jetCombinatoricModel_SB_{tag}.yml"

# Second pass: rerun the same datasets with the freshly fitted JCM applied
# (still no FvT / SvB — those come from Phases C/D), then plot with the NoFvT config.
wjcm_config_path = f"{JCM_OUTPUT_PATH}analysis_config_wJCM.yml"
wjcm_plot_config = config.get('jcm_plot_config', config.get('jcm', {}).get('plot_config', "coffea4bees/plots/metadata/plotsAllNoFvT.yml"))

def get_raw_wjcm_config():
    res = get_raw_jcm_config()
    res['config']['apply_JCM'] = True
    res['config']['JCM_file'] = jcm_file_path
    return res

### Including modules
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# NOTE: input[0] must stay the JCM yml — Snakefile_PhaseB_2 reads rules.output_computeJCM.input[0]
rule output_computeJCM:
    input:
        jcm_file_path,
        f"{JCM_OUTPUT_PATH}histAll_wJCM.coffea",
        f"{JCM_OUTPUT_PATH}plots_wJCM/plots_done.txt"

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
    output: jcm_file_path
    params:
        extra_arguments = config.get('jcm_extra_arguments', ""),
        tag = tag,
        output_dir = f"{JCM_OUTPUT_PATH}JCM_{tag}/",
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: f"{JCM_OUTPUT_PATH}logs/make_JCM.log"

# ---------------------------------------------------------------------------
# Second pass: same datasets, fitted JCM applied (apply_JCM: true, JCM_file -> fit above)
# ---------------------------------------------------------------------------
rule create_wJCM_config:
    input:
        jcm_file = jcm_file_path,
        configfiles = workflow.configfiles if workflow.configfiles else []
    output: wjcm_config_path
    run:
        import yaml, os
        cfg = get_raw_wjcm_config()
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as analysis_data_wJCM with:
    input:
        runner_script = "runner.py",
        config_file = wjcm_config_path,
        jcm_file = jcm_file_path
    output: f"{JCM_OUTPUT_PATH}singlefiles/hist_data__{{year}}_{{era}}_wJCM.coffea"
    log: f"{JCM_OUTPUT_PATH}logs/analysis_data__{{year}}_{{era}}_wJCM.log"
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

use rule analysis_processor from analysis as analysis_MC_wJCM with:
    input:
        runner_script = "runner.py",
        config_file = wjcm_config_path,
        jcm_file = jcm_file_path
    output: f"{JCM_OUTPUT_PATH}singlefiles/hist__{{dataset}}__{{year}}_wJCM.coffea"
    log: f"{JCM_OUTPUT_PATH}logs/analysis__{{dataset}}_{{year}}_wJCM.log"
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = config['analysis_container_wrapper']

use rule merging_coffea_files from analysis as merge_wJCM with:
    input:
        files = [f"{JCM_OUTPUT_PATH}singlefiles/hist_data__{yr}_{era}_wJCM.coffea" for yr, era in DATA_YEAR_ERA] + [f"{JCM_OUTPUT_PATH}singlefiles/hist__{ds}__{yr}_wJCM.coffea" for ds in MC_DATASETS for yr in DATA_YEARS],
        script = "src/tools/merge_coffea_files.py"
    output: f"{JCM_OUTPUT_PATH}histAll_wJCM.coffea"
    log: f"{JCM_OUTPUT_PATH}logs/merge_wJCM.log"
    params:
        run_performance = False,
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python"),
        input_files = lambda wildcards, input: " ".join([f for f in (input.files if hasattr(input, 'files') else input) if not f.endswith('.py')])

use rule make_plots from analysis as make_plots_wJCM with:
    input:
        coffea_file = f"{JCM_OUTPUT_PATH}histAll_wJCM.coffea",
        metadata_file = wjcm_plot_config,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{JCM_OUTPUT_PATH}plots_wJCM/plots_done.txt"
    params:
        output_dir = f"{JCM_OUTPUT_PATH}plots_wJCM/",
        metadata = wjcm_plot_config,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW -f png",
            "--year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: f"{JCM_OUTPUT_PATH}logs/make_plots_wJCM.log"

localrules: create_noJCM_config, create_wJCM_config, merge_noJCM, merge_wJCM, make_new_JCM, make_plots_wJCM
