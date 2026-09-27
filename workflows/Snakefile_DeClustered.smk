# coffea4bees/workflows/Snakefile_DeClustered.smk
# DeClustered (synthetic) dataset production: the top level of the declustered roast. Ports the
# scripts/synthetic-dataset-{cluster,make-dataset,analyze,analyze-cutflow}-Run3*.sh chain, laid
# out like Snakefile_MakeMixedData.smk.
#
#   inputs             fetch the upstream data/ttbar histograms + their runner config (B.1 roast)
#   D.1 cluster        4b data, ttbar subtracted with the upstream FvT -> splitting histograms
#   D.2 PDFs           splitting histograms -> clustering_pdfs_vs_pT_<era>.yml, published to EOS
#   D.3 decluster      4b data re-generated from the PDFs, one replica per seed -> synthetic picoAODs
#                      + the multi-sample dataset YAML (files_template seedXXX, nSamples = n_seeds)
#   D.4 validate       synthetic-data histograms with the upstream config, merged with the upstream
#                      data/ttbar -> cutflow (synthetic 4b next to data 4b, same selection)
#
# Everything this roast consumes comes from other roasts, named under `inputs:` and checked by
# `roast new`: the FvT from the nominal, the histograms from a Phase B.1 roast with the non-tight
# selection (config/nominal_run3_nontight.yml) -- the declustering, like the mixing, is non-tight.
#
# Products are published to `publish_base` on EOS; the dataset YAML under <publish_base>/handoff/
# is what consumer roasts read (runner.py -m accepts root:// URLs). Nothing is installed into the
# checkout: the PDFs in particular go to EOS, not coffea4bees/jet_clustering/, because the condor
# workers get a tarball of the checkout made when the shared dask daemon starts (in D.1), so a PDF
# written into the checkout afterwards would never reach them. Run one step with a target:
# `roast submit <id> --step DeClustered --targets all_D1`.
#
# This file is the ONLY place the config is read and paths are built: the step files include()d
# below use the names defined here and never call config.setdefault themselves.

import os
import copy
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/declustered_run3.yml"

include: "helpers/common.smk"      # {roast_id} substitution

# ── Configuration ─────────────────────────────────────────────────────────────

def _slash(p):
    return p if p.endswith("/") else p + "/"

config.setdefault('test', False)
if isinstance(config['test'], str):
    config['test'] = config['test'].lower() in ("true", "1", "yes")

out = _slash(config['output_path'])
PUB = str(config['publish_base']).rstrip("/")
if config['test']:
    # `roast submit -t` moves output_path to <output_path>_test/; keep the test slice's picoAODs,
    # PDFs and handoff out of the real EOS area too, or the full run is left with stale files.
    PUB = f"{PUB}/test"
HANDOFF = f"{PUB}/handoff"

YEAR_ERAS = {str(y): list(eras) for y, eras in config['year_eras'].items()}
YEARS = list(YEAR_ERAS)

# The declustered data is built with the non-tight four-tag definition, as the mixed data is:
# make_declustered_data_4b.py (a Skimmer4b) has no tight option, so the cluster (D.1) and
# validation (D.4) passes must not use one either, or the PDFs and the check describe a
# different 4b sample from the one declustered.
if config.get('fourTag_use_tight', None) is not False \
        or config['analysis_config']['config'].get('fourTag_use_tight', None) is not False:
    raise ValueError("declustered production requires fourTag_use_tight: false, top level and in "
                     "analysis_config.config")

INPUTS = config.get('inputs') or {}
for _key in ('FvT', 'jcm_hists'):
    if not str(INPUTS.get(_key) or "").startswith("root://"):
        # FvT: the cluster processor falls back to a legacy FvT file next to each picoAOD when no
        # FvT friend is given -- a silent substitute for the upstream roast's classifier.
        raise ValueError(f"inputs.{_key} must be a root:// URL into an upstream roast (see the config)")
FVT = INPUTS['FvT']

# Local copies of the upstream histograms and their runner config: coffea's load() reads local
# files only. D.4 histograms the synthetic data with exactly that config, so synthetic and real
# 4b data are compared like with like -- and it proves the upstream used the non-tight selection.
INPUT_DIR = f"{out}inputs/"
UPSTREAM_HISTS = f"{INPUT_DIR}{os.path.basename(INPUTS['jcm_hists'])}"
UPSTREAM_HIST_CONFIG_URL = f"{os.path.dirname(INPUTS['jcm_hists'])}/analysis_config_noJCM.yml"
UPSTREAM_HIST_CONFIG = f"{INPUT_DIR}analysis_config_noJCM.yml"

CLUSTER = config.get('cluster') or {}
PDFS = config.get('pdfs') or {}
DECL = config.get('declustering') or {}
VAL = config.get('validation') or {}

# PDFs: made here (D.1 + D.2) unless inputs.pdfs points at another declustered roast's published
# set (its <publish_base>/pdfs) -- e.g. to add seeds without re-learning the splittings.
PDF_EXTERNAL = INPUTS.get('pdfs')
if PDF_EXTERNAL and not str(PDF_EXTERNAL).startswith("root://"):
    raise ValueError("inputs.pdfs must be a root:// URL into an upstream declustered roast's pdfs/")
PDF_BASE = str(PDF_EXTERNAL).rstrip("/") if PDF_EXTERNAL else f"{PUB}/pdfs"
PDF_TEMPLATE = f"{PDF_BASE}/clustering_pdfs_vs_pT_XXX.yml"     # XXX -> era, in the processor

# Seeds: one independent replica per seed. runner.py expands the dataset's `files_template`
# over range(nSamples), so the seeds MUST be 0..n_seeds-1 (no gaps, no offset).
N_SEEDS = int(DECL.get('n_seeds', 1))
SEEDS = list(range(N_SEEDS))

# ttbar: the declustering either subtracts ttbar from the 4b data with the upstream FvT (then a
# consumer adds ttbar pseudodata, e.g. the mixeddata roast's ttbar_PSData) or declusters the ttbar
# with the multijet (the current Run 3 `synthetic_data_noTT`). runner.py picks the sample naming
# from the dataset name (src/runner/dataset.py:get_dataset_type: synthetic_data_noTT* -> syn_noTT_v<i>,
# other synthetic_data* -> syn_v<i>), so the name must say which one it is.
SUBTRACT_TT = bool(DECL.get('subtract_ttbar', False))
DATASET_NAME = str(DECL.get('dataset_name', 'synthetic_data_noTT_roast' if not SUBTRACT_TT else 'synthetic_data_subTT'))
if not DATASET_NAME.startswith('synthetic_data') or DATASET_NAME.startswith('synthetic_data_noTT') == SUBTRACT_TT:
    raise ValueError(f"declustering.dataset_name {DATASET_NAME!r} does not match subtract_ttbar={SUBTRACT_TT}: "
                     f"use synthetic_data_noTT<...> without ttbar subtraction, synthetic_data<...> "
                     f"(not _noTT) with it")
if DATASET_NAME in ('synthetic_data', 'synthetic_data_noTT'):
    # load_datasets_metadata refuses a name defined differently in two -m sources, and the
    # default metadata/datasets/synthetic_data.yml already defines these two.
    raise ValueError(f"declustering.dataset_name {DATASET_NAME!r} is taken by metadata/datasets/synthetic_data.yml")
DATASET_URL = f"{HANDOFF}/{DATASET_NAME}.yml"

# Container / runner invocation, as in Snakefile_MakeMixedData.smk (config['test'] parsed above)
_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', _wrapper)
WRAPPER = config['analysis_container_wrapper']
PYTHON = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
CONDOR = "" if (config['test'] or os.getenv("CI")) else "--shared-dask --condor"
TEST_FLAG = "-t" if config['test'] else ""

# Shell prefix for any rule that writes to EOS: roast seeds ./proxy/x509_proxy in the checkout.
# ${VAR:-}, not $VAR: snakemake runs shell blocks under `set -u`.
EOS_PROXY = ('if [ -z "${{X509_USER_PROXY:-}}" ] && [ -f ./proxy/x509_proxy ]; then '
             'export X509_USER_PROXY="$PWD/proxy/x509_proxy"; fi')

def processor_config(section_config, inherit_config=True, **top):
    """A runner config: analysis_config's processor/dataset_location/friend_file/weights_file/
    runner, with `section_config` merged over analysis_config.config (or, inherit_config=False,
    over nothing -- the DeClusterer's Skimmer4b base takes no histogram-pass settings) and `top`
    over the rest. Written with yaml.dump, so there is no sed-patching anywhere in this workflow.
    (The shared analysis_processor rule passes only the config, --datasets, --years and
    extra_arguments to runner.py: processor, friends, weights and condor must all be in here.)"""
    ac = copy.deepcopy(config['analysis_config'])
    cfg = {k: ac[k] for k in ('processor', 'dataset_location', 'friend_file', 'weights_file', 'runner') if k in ac}
    base = ac.get('config', {}) if inherit_config else {}
    cfg['config'] = {**base, **copy.deepcopy(section_config)}
    for k, v in top.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    if config['test']:
        cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
    return cfg

def write_yaml(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# ── Inputs from upstream roasts ───────────────────────────────────────────────

rule fetch_inputs:
    output:
        hists = UPSTREAM_HISTS,
        hist_config = UPSTREAM_HIST_CONFIG
    log: f"{INPUT_DIR}fetch.log"
    params:
        hists = INPUTS['jcm_hists'],
        hist_config = UPSTREAM_HIST_CONFIG_URL
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f "{params.hists}" {output.hists} 2>&1 | tee {log}
        xrdcp -f "{params.hist_config}" {output.hist_config} 2>&1 | tee -a {log}
        for f in {params.hists} {params.hist_config}; do echo "fetched $f" | tee -a {log}; done
        """

# ── Steps ─────────────────────────────────────────────────────────────────────

include: "Snakefile_DeClustered_1_cluster.smk"
include: "Snakefile_DeClustered_2_pdfs.smk"
include: "Snakefile_DeClustered_3_decluster.smk"
include: "Snakefile_DeClustered_4_validate.smk"

# default_target, not position: an included or inserted rule can never steal the default.
rule all_DeClustered:
    default_target: True
    input:
        rules.all_D1.input,
        rules.all_D2.input,
        rules.all_D3.input,
        rules.all_D4.input

localrules: fetch_inputs, all_DeClustered
