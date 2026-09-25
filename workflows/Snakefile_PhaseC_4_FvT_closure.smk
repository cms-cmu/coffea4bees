# coffea4bees/workflows/Snakefile_PhaseC_4_FvT_closure.smk
# Phase C.4 (cmslpc): FvT closure check.
#
# After the FvT has been trained and evaluated (C.1-C.3 on falcon), rerun the analysis processor on
# data + ttbar with the JCM *and* the new FvT applied (no SvB), then plot, dump/compare the cutflow
# and build the background-closure table -- the same check Phase B.1 does for the JCM, so the
# 3b -> 4b model is validated before anything (SvB, Phase D) trains on top of it.
#
# ttbar: reused from Phase B.1's wJCM singlefiles by default (fvt_closure.reuse_wJCM_ttbar, jcm_output_path),
#   because the FvT weight only applies to data; set reuse_wJCM_ttbar: false to reprocess MC here.
# Inputs (per roast, from the master config):
#   - JCM:   fvt_closure.JCM_file, default coffea4bees/metadata/weights/JCM/{roast_id}/jetCombinatoricModel_SB_{tag}.yml
#   - FvT:   analysis_config.config.friends.FvT (this roast's Phase C friend; runner.py merges it over friend_file)
# Outputs: {output_path}FvT_closure/{singlefiles/, histAll_FvT_closure.coffea, plots_FvT_closure/ (+index.html),
#           cutflow_FvT_closure.{yml,html}, cutflow_validation_FvT_closure*.txt}
#
# roast: step "C4" (cmslpc) in src/tools/roast.py.

import os
import copy
import yaml

config.setdefault('label', "FvT_closure")
base_output = config.get('output_path', 'output/FvT_closure/')
if not base_output.endswith('/'):
    base_output += '/'
CLOSURE_PATH = base_output if base_output.rstrip('/').endswith('FvT_closure') else os.path.join(base_output, 'FvT_closure/')

config.setdefault('test', False)
is_test = config.get('test', False)
if isinstance(is_test, str):
    is_test = is_test.lower() in ("true", "1", "yes")
config['test'] = is_test

container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('container_wrapper', container_wrapper)
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', container_wrapper))
config.setdefault('python_bin', os.getenv("CONTAINER_PYTHON", "python"))

if config.get('test', False) or os.getenv("CI"):
    config.setdefault('additional_parameters', "")
else:
    config.setdefault('additional_parameters', "--shared-dask --condor --run-performance")

include: "helpers/common.smk"   # resolves {roast_id}, provides resolve_config_section

closure_cfg = config.get('fvt_closure') or {}
if not isinstance(closure_cfg, dict):
    closure_cfg = {}
CLOSURE_OPTION_KEYS = ('datasets', 'plot_config', 'JCM_file', 'known_counts', 'known_counts_test', 'cutflow_list',
                       'reuse_wJCM_ttbar', 'jcm_output_path')

datasets = closure_cfg.get('datasets', ['data', 'TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic'])
if isinstance(datasets, str):
    datasets = [d.strip() for d in datasets.split(",") if d.strip()]
DATA_YEAR_ERA = [(str(yr), era) for yr, eras in config['year_eras'].items() for era in eras] if 'data' in datasets else []
DATA_YEARS = [str(y) for y in config['year_eras'].keys()]
MC_DATASETS = [d for d in datasets if d != 'data']

# The FvT weight is only applied to files with an entry in the FvT friend, i.e. data (see
# event_weights.py: `apply_FvT and "FvT" in event.fields`; rename_FvT_friend returns None for MC).
# MC histograms are therefore identical to Phase B.1's wJCM pass, so by default the ttbar
# singlefiles from there are merged in instead of being reprocessed (saves the 12 MC jobs).
REUSE_WJCM_TTBAR = closure_cfg.get('reuse_wJCM_ttbar', True)
if isinstance(REUSE_WJCM_TTBAR, str):
    REUSE_WJCM_TTBAR = REUSE_WJCM_TTBAR.lower() in ("true", "1", "yes")
JCM_OUTPUT_PATH = closure_cfg.get('jcm_output_path', os.path.join(base_output, 'computeJCM/'))
if not JCM_OUTPUT_PATH.endswith('/'):
    JCM_OUTPUT_PATH += '/'

def mc_singlefile(ds, yr):
    if REUSE_WJCM_TTBAR and ds.startswith('TT'):
        return f"{JCM_OUTPUT_PATH}singlefiles/hist__{ds}__{yr}_wJCM.coffea"
    return f"{CLOSURE_PATH}singlefiles/hist__{ds}__{yr}_FvT_closure.coffea"

tag = config.get('tag', "2024_v2")
jcm_file = closure_cfg.get('JCM_file', f"coffea4bees/metadata/weights/JCM/{config['roast_id']}/jetCombinatoricModel_SB_{tag}.yml")
# A remote JCM (e.g. an earlier production's EOS handoff, for a roast that skips Phase B) is read
# through fsspec by the processor, but is no file Snakemake can see, so it gets no input edge.
jcm_input = [] if "://" in jcm_file else jcm_file
plot_config = closure_cfg.get('plot_config', "coffea4bees/plots/metadata/plotsAll.yml")
CUTFLOW_LIST = closure_cfg.get('cutflow_list', "passJetMult,passPreSel,passDiJetMass,SR_woTrig,SR,SB_woTrig,SB")

def known_cutflow_flag():
    if config.get("test", False):
        f = closure_cfg.get('known_counts_test') or "coffea4bees/analysis/tests/known_Counts_FvT_closure.yml"
    else:
        f = closure_cfg.get('known_counts') or "coffea4bees/analysis/tests/known_fullCounts_FvT_closure.yml"
    return f'--known-cutflow "{f}"' if os.path.exists(f) else '--known-cutflow "none"'

def get_raw_closure_config():
    res = resolve_config_section(config, primary_key='fvt_closure', fallback_keys=['analysis_config', 'analysis'])
    for k in CLOSURE_OPTION_KEYS:      # workflow options, not processor settings
        res.pop(k, None)
    if 'config' not in res or not isinstance(res['config'], dict):
        res['config'] = {}
    res['config']['apply_JCM'] = True
    res['config']['JCM_file'] = jcm_file
    res['config']['apply_FvT'] = True
    res['config']['run_SvB'] = False
    for k in list(res['config'].keys()):
        if k.startswith('SvB'):
            res['config'][k] = None
    if config.get("test", False):
        res.setdefault('runner', {})
        res['runner']['condor'] = False
        res['runner']['shared_dask'] = False
        res['runner']['run_performance'] = False
    return res

closure_config_path = f"{CLOSURE_PATH}analysis_config_FvT_closure.yml"

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# default_target, not position: Snakemake takes the first rule of the top-level Snakefile
# as the default, so adding a rule above this one would silently shrink the DAG to that
# rule's own inputs.
rule all_FvT_closure:
    default_target: True
    input:
        f"{CLOSURE_PATH}histAll_FvT_closure.coffea",
        f"{CLOSURE_PATH}plots_FvT_closure/plots_done.txt",
        f"{CLOSURE_PATH}cutflow_validation_FvT_closure.txt",
        f"{CLOSURE_PATH}cutflow_FvT_closure.html"

rule create_FvT_closure_config:
    input:
        jcm_file = jcm_input,
        configfiles = workflow.configfiles if workflow.configfiles else []
    output: closure_config_path
    run:
        import yaml, os
        cfg = get_raw_closure_config()
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as analysis_data_FvT_closure with:
    input:
        runner_script = "runner.py",
        config_file = closure_config_path
    output: f"{CLOSURE_PATH}singlefiles/hist_data__{{year}}_{{era}}_FvT_closure.coffea"
    log: f"{CLOSURE_PATH}logs/analysis_data__{{year}}_{{era}}_FvT_closure.log"
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

use rule analysis_processor from analysis as analysis_MC_FvT_closure with:
    input:
        runner_script = "runner.py",
        config_file = closure_config_path
    output: f"{CLOSURE_PATH}singlefiles/hist__{{dataset}}__{{year}}_FvT_closure.coffea"
    log: f"{CLOSURE_PATH}logs/analysis__{{dataset}}_{{year}}_FvT_closure.log"
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = config['analysis_container_wrapper']

use rule merging_coffea_files from analysis as merge_FvT_closure with:
    input:
        files = [f"{CLOSURE_PATH}singlefiles/hist_data__{yr}_{era}_FvT_closure.coffea" for yr, era in DATA_YEAR_ERA] + [mc_singlefile(ds, yr) for ds in MC_DATASETS for yr in DATA_YEARS],
        script = "src/tools/merge_coffea_files.py"
    output: f"{CLOSURE_PATH}histAll_FvT_closure.coffea"
    log: f"{CLOSURE_PATH}logs/merge_FvT_closure.log"
    params:
        run_performance = False,
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python"),
        input_files = lambda wildcards, input: " ".join([f for f in (input.files if hasattr(input, 'files') else input) if not f.endswith('.py')])

use rule make_plots from analysis as make_plots_FvT_closure with:
    input:
        coffea_file = f"{CLOSURE_PATH}histAll_FvT_closure.coffea",
        metadata_file = plot_config,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{CLOSURE_PATH}plots_FvT_closure/plots_done.txt"
    params:
        output_dir = f"{CLOSURE_PATH}plots_FvT_closure/",
        metadata = plot_config,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW -f png",
            "--year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log: f"{CLOSURE_PATH}logs/make_plots_FvT_closure.log"

use rule check_cutflow from analysis as check_cutflow_FvT_closure with:
    input:
        coffea_file = f"{CLOSURE_PATH}histAll_FvT_closure.coffea"
    output:
        validation_txt = f"{CLOSURE_PATH}cutflow_validation_FvT_closure.txt",
        cutflow_yml = f"{CLOSURE_PATH}cutflow_FvT_closure.yml"
    log: f"{CLOSURE_PATH}logs/cutflow_validation_FvT_closure.log"
    params:
        known_flag = lambda wildcards: known_cutflow_flag(),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: CUTFLOW_LIST,
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    container: None

# 3b data already carries JCM x FvT, which models multijet + 3b ttbar: Multijet = data 3b
use rule cutflow_closure_table from analysis as FvT_cutflow_closure_table with:
    params:
        title = lambda wildcards: f"{config.get('label', 'FvT_closure')}_cutflow_{wildcards.label}",
        multijet = "data3b",
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")

localrules: create_FvT_closure_config, merge_FvT_closure, make_plots_FvT_closure, check_cutflow_FvT_closure, FvT_cutflow_closure_table
