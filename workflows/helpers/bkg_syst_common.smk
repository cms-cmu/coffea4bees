# coffea4bees/workflows/helpers/bkg_syst_common.smk
# Shared configuration normalization, environment flags, and path hierarchies
# for ttHbb background systematics workflows (Stages A_1 through F_4).

import os
import sys
import shutil
import yaml
import copy
import re

include: "common.smk"

# ── 1. Configuration Section Resolution ────────────────────────────────────────
phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

# Stage C (FvT) settings: stage_phaseC_configs and src/classifier/workflow/Snakefile read them from
# the top level (eos_base, nominal_classifier_inputs, gpu_*, analyze, ...), as Snakefile_PhaseC.smk
# hoists its `fvt` section. output_path stays the workflow's.
fvt_cfg = resolve_config_section(config, primary_key='phase_e_fvt', fallback_keys=['fvt', 'fvt_classifier'])
for k, v in fvt_cfg.items():
    if k != 'output_path':
        config.setdefault(k, v)

# ── 2. Execution Environment & Container Setup ─────────────────────────────────
config.setdefault('analysis_container', None)
default_container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', default_container_wrapper))
container_wrapper = config['analysis_container_wrapper']
condor_flags = "" if config.get("test", False) else "--shared-dask --condor"
python_bin = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
config.setdefault('python_bin', python_bin)

config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('channel', "ttHbb")
channel = config['channel']

# ── 3. Years & Era Normalization ───────────────────────────────────────────────
raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
if isinstance(raw_years, str):
    YEARS = [str(y).strip() for y in raw_years.split() if str(y).strip()]
else:
    YEARS = [str(y) for y in raw_years]
config['years'] = YEARS

is_run3 = any(('202' in str(y) or 'Run3' in str(y)) for y in YEARS)
config.setdefault('isRun3', is_run3)
run_period = "Run3" if is_run3 else "Run2"

per_year_jcm = bool(config.get('per_year_jcm', (config.get('phaseB_1') or {}).get('per_year_jcm', False)))
config['per_year_jcm'] = per_year_jcm

# ── 4. Subsamples ──────────────────────────────────────────────────────────────
SUB = config.get('subsamples') or {}
# Where the closure samples come from: `split` = 3b mixing (mixeddata_all split with the A_1 JCM),
# `seeds` = 4b mixing (the N seeds of a MakeMixedData 4b-mixing roast's mixeddata_all_<tag>; no JCM)
SUB_SOURCE = SUB.get('source', 'split')
if SUB_SOURCE not in ('split', 'seeds'):
    raise ValueError(f"subsamples.source must be 'split' or 'seeds', got {SUB_SOURCE!r}")

subsample_indices = config.get('subsample_indices', phase_e_cfg.get('subsample_indices', None))
if subsample_indices is not None:
    if isinstance(subsample_indices, str):
        subsample_indices = [int(x) for x in subsample_indices.split()]
    elif isinstance(subsample_indices, int):
        subsample_indices = [subsample_indices]
    SUBSAMPLES = [str(i) for i in subsample_indices]
    N_SUBSAMPLES = len(SUBSAMPLES)
else:
    N_SUBSAMPLES = int(SUB.get('n', config.get('n_subsamples', config.get('n_models', config.get('n_samples', 16)))))
    SUBSAMPLES = [str(i) for i in range(N_SUBSAMPLES)]

config.setdefault('n_subsamples', N_SUBSAMPLES)     # B, C and F read n_subsamples / n_models
TEST_FLAG = "-t" if config.get("test", False) else ""
TTBAR = list(config.get('ttbar') or config.get('ttbar_processes') or [])

# ── 5. Standard Output Directory Hierarchies ──────────────────────────────────
config.setdefault('output_path', "output/ttHbb_bkg_syst/")
out = config['output_path']
if not out.endswith("/"):
    out += "/"
# EOS area for what this workflow writes there (the A_2 subsample picoAODs)
PUB = str(config.get('publish_base', f"{out}publish")).rstrip("/")

out_a1 = f"{out}bkg_syst_A_1_mixed_jcm/"
out_a2 = f"{out}bkg_syst_A_2_make_subsamples/"
out_a3 = f"{out}bkg_syst_A_3_process_subsamples/"
out_b1 = f"{out}bkg_syst_B_1_computeJCM/"
out_c1 = f"{out}bkg_syst_C_1_inputs/"
out_c  = f"{out}bkg_syst_C_FvT/"
out_f1 = f"{out}bkg_syst_F_1_analysis/"
out_f2 = f"{out}bkg_syst_F_2_run_two_stage_closure/"
out_f3 = f"{out}bkg_syst_F_3_stats/"
out_f4 = f"{out}bkg_syst_F_4_stats_mixeddata/"

# ── 6. Inputs: a mixeddata roast + the analysis' nominal roast ─────────────────
# From the MakeMixedData roast (its handoff/ YAMLs): the mixed data and the ttbar pseudodata. The
# mixed-data JCM that splits them into subsamples is fit here (A_1), in the analysis selection.
# From the nominal analysis roast: its B.1 noJCM histograms (A_1, B_1), its Phase F histograms
# (F_2 signal, F_3 / F_4 datacards), its classifier-input manifest (C) and its SvB friend (F_1).
INPUTS = config.get('inputs') or {}
for _key in ('mixeddata_all', 'ttbar_psdata', 'jcm_hists', 'nominal_hists', 'SvB', 'SvB_model'):
    if not INPUTS.get(_key):
        raise ValueError(f"bkg_syst: inputs.{_key} is required (see analysis_ttHbb_bkg_syst.yml)")
MIXED_URL = str(INPUTS['mixeddata_all'])
MIX_NAME = os.path.basename(MIXED_URL).removesuffix(".yml")       # handoff/<dataset key>.yml
PS_URL = str(INPUTS['ttbar_psdata'])
PS_NAME = os.path.basename(PS_URL).removesuffix(".yml")
# Data + ttbar MC noJCM histograms in the analysis selection (the B.1 JCM-fit input), and the runner
# config they were made with (B.1 writes it next to them): A_1 histograms the mixed data with it.
JCM_HISTS_URL = str(INPUTS['jcm_hists'])
JCM_HIST_CONFIG_URL = str(INPUTS.get('analysis_config_noJCM')
                          or f"{os.path.dirname(JCM_HISTS_URL)}/analysis_config_noJCM.yml")
A_INPUT_DIR = f"{out}inputs/"
JCM_HISTS = f"{A_INPUT_DIR}{os.path.basename(JCM_HISTS_URL)}"
JCM_HIST_CONFIG = f"{A_INPUT_DIR}{os.path.basename(JCM_HIST_CONFIG_URL)}"
PS_DATASET = f"{A_INPUT_DIR}{PS_NAME}.yml"
# The nominal Phase F histograms, fetched (coffea's load() reads local files only); F_3 converts
# them to the datacard JSON next to them.
NOMINAL_HISTS_URL = str(INPUTS['nominal_hists'])
config['nominal_coffea'] = f"{A_INPUT_DIR}{os.path.basename(NOMINAL_HISTS_URL)}"
config.setdefault('nominal_json', config['nominal_coffea'].removesuffix(".coffea") + ".json")
# Read in place (fsspec / friend URLs)
if INPUTS.get('classifier_inputs'):
    config['nominal_classifier_inputs'] = str(INPUTS['classifier_inputs'])
config['data_svb_friend'] = str(INPUTS['SvB'])
# ...and the SvB model that made that friend: A_3 evaluates it on the mixed subsamples, so mixed data
# and data are scored by the same SvB in the closure
config['mixed_svb_model'] = str(INPUTS['SvB_model'])
MIXED_DATASET = f"{A_INPUT_DIR}{MIX_NAME}.yml"      # local copy: A_2 reads the seeds' file lists

# The multi-sample closure dataset A_2 assembles: mixeddata_4b (samples mix_v<k>) or
# mixeddata_<tag>_4b (samples mix_<tag>_v<k>) -- the only multi-sample names runner.py knows
# (src/runner/dataset.py:get_dataset_type / mixed_variant_prefix).
SUB_NAME = SUB.get('dataset_name', config.get('multisample_dataset_name', "mixeddata_4b"))
config['multisample_dataset_name'] = SUB_NAME
if SUB_NAME == 'mixeddata_4b':
    SUB_PREFIX = 'mix'
elif (_m := re.fullmatch(r"mixeddata_([A-Za-z0-9]+)_4b", SUB_NAME)) and _m.group(1) != 'noTTSub':
    SUB_PREFIX = f"mix_{_m.group(1)}"
else:
    raise ValueError(f"subsamples.dataset_name {SUB_NAME!r} must be 'mixeddata_4b' or 'mixeddata_<tag>_4b' "
                     f"(runner.py reads any other name as MC)")
MULTISAMPLE_DATASET = f"{out_a2}{SUB_NAME}.yml"
# The dataset metadata C's classifier reads (--metadata): the repo's dataset YAMLs, except any that
# defines SUB_NAME (two do: mixeddata_4b.yml, mixeddata_4b_ttHbb.yml), + MULTISAMPLE_DATASET (A_2)
CLASSIFIER_METADATA = f"{out_a2}classifier_metadata/"
config.setdefault('classifier_metadata', CLASSIFIER_METADATA)

# EOS products of this run, all under publish_base
config.setdefault('classifier_inputs_base', f"{PUB}/classifier_inputs/mixeddata/")
config.setdefault('mixeddata_friend_base', f"{PUB}/friend/mixeddata/")
config.setdefault('eos_base', PUB)                     # C: classifier/ and friend/FvT/
config.setdefault('classifier_inputs_json',
    f"{out_a3}classifier_inputs/classifier_inputs_mixeddata_{channel}.json")
config.setdefault('mixeddata_friend_json', f"{out}coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json")

# Shell prefix for rules that read or write EOS (roast seeds ./proxy/x509_proxy in the checkout).
# Single braces: spliced into shell blocks as {EOS_PROXY}, not re-formatted by snakemake.
EOS_PROXY = ('if [ -z "${X509_USER_PROXY:-}" ] && [ -f ./proxy/x509_proxy ]; then '
             'export X509_USER_PROXY="$PWD/proxy/x509_proxy"; fi')

# EOS handoff for cross-cluster execution (roast): B_1 / A_3 products for C on the GPU host, C's
# friend manifests back for F (bkg_syst_AB_handoff, bkg_syst_C_handoff). handoff.eos_base
_handoff = config.get('handoff') or {}
if not isinstance(_handoff, dict):
    _handoff = {}
HANDOFF_EOS = str(_handoff.get('eos_base') or "").rstrip('/')

def write_yaml(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)

module analysis:
    snakefile: "rules/analysis.smk"     # resolved from the including Snakefile (workflows/)
    config: config
