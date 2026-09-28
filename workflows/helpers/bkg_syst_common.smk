# coffea4bees/workflows/helpers/bkg_syst_common.smk
# Shared configuration normalization, environment flags, and path hierarchies
# for ttHbb background systematics workflows (Stages A_1 through F_4).

import os
import sys
import shutil
import yaml
import copy
import ast as _ast

include: "common.smk"

# ── 1. Configuration Section Resolution ────────────────────────────────────────
phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure', 'mixeddata'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

jcm_cfg = resolve_config_section(config, primary_key='phase_e_jcm', fallback_keys=['jcm', 'phase_e'])
for k, v in jcm_cfg.items():
    config.setdefault(f"jcm_{k}", v)

mixeddata_cfg = resolve_config_section(config, primary_key='mixeddata', fallback_keys=['mixed_data'])
for k, v in mixeddata_cfg.items():
    config.setdefault(k, v)

psdata_cfg = resolve_config_section(config, primary_key='phaseA_2', fallback_keys=['phaseA_4', 'ttbar_psdata', 'psdata', 'ttbar_pseudodata'])
for k, v in psdata_cfg.items():
    if k != 'output_path' and not isinstance(v, dict):
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

# ── 4. Subsamples & Ranks ──────────────────────────────────────────────────────
N_SUBSAMPLES = int(config.get('n_subsamples', config.get('n_models', config.get('n_samples', 15))))
SUBSAMPLES = [str(i) for i in range(N_SUBSAMPLES)]

config.setdefault('default_rank', 0)
_rank_raw = config['default_rank']
if isinstance(_rank_raw, str):
    try:
        _rank = _ast.literal_eval(_rank_raw)
    except (ValueError, SyntaxError):
        _rank = _rank_raw
else:
    _rank = _rank_raw

if isinstance(_rank, (list, tuple)):
    _rank_suffix = f"_rank{int(_rank[0])}_{int(_rank[1])}"
    _rank_tuple = [int(_rank[0]), int(_rank[1])]
else:
    _rank_suffix = f"_rank{int(_rank)}_{int(_rank)}"
    _rank_tuple = [int(_rank), int(_rank)]

config.setdefault('tag', '')
_tag = str(config['tag'])
_tag_suffix = f"_{_tag}" if _tag else ''

# ── 5. Standard Output Directory Hierarchies ──────────────────────────────────
config.setdefault('output_path', "output/ttHbb_bkg_syst/")
out = config['output_path']
if not out.endswith("/"):
    out += "/"

out_a1 = f"{out}bkg_syst_A_1_make_mixeddata/"
out_a2 = f"{out}bkg_syst_A_2_make_ttbar_psdata/"
out_a3 = f"{out}bkg_syst_A_3_make_subsamples/"
out_a4 = f"{out}bkg_syst_A_4_process_subsamples/"
out_b1 = f"{out}bkg_syst_B_1_computeJCM/"
out_c1 = f"{out}bkg_syst_C_1_inputs/"
out_c  = f"{out}bkg_syst_C_FvT/"
out_f1 = f"{out}bkg_syst_F_1_analysis/"
out_f2 = f"{out}bkg_syst_F_2_run_two_stage_closure/"
out_f3 = f"{out}bkg_syst_F_3_stats/"
out_f4 = f"{out}bkg_syst_F_4_stats_mixeddata/"

picoaod_dir = f"{out_a1}make_mixeddata_picoAOD_per_year/"

config.setdefault('base_path',
    f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/{run_period}/{channel}_pz{_rank_suffix}")

# Dataset naming and paths
config.setdefault('dataset_name', f"mixeddata_{channel}{_rank_suffix}")
default_mixeddata_dataset = f"coffea4bees/metadata/datasets/{config.get('mixeddata', {}).get('dataset_name', config.get('dataset_name', 'mixeddata_ttHbb_bkg_syst_rank0_0'))}.yml"
mixeddata_dataset_output = config.get('mixeddata_dataset_output', f"{out}coffea4bees/metadata/datasets/{config.get('mixeddata', {}).get('dataset_name', config.get('dataset_name', 'mixeddata_ttHbb_bkg_syst_rank0_0'))}.yml")
mixeddata_dataset_file = config.get('mixeddata_dataset_file', default_mixeddata_dataset)
config.setdefault('install_path', mixeddata_dataset_output)

# Multi-sample and classifier paths
config.setdefault('multisample_dataset_name', "mixeddata_4b")
config.setdefault('multisample_install_path', f"{out}coffea4bees/metadata/datasets/mixeddata_4b.yml")
config.setdefault('subsample_output_path', f"{out_a3}subsamples/")
config.setdefault('classifier_inputs_base',
    f"root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/{channel}/classifier_inputs/mixeddata/")
config.setdefault('classifier_inputs_json',
    f"{out_a4}classifier_inputs/classifier_inputs_mixeddata_{channel}.json")
config.setdefault('mixeddata_friend_json', f"{out}coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json")

# ── 6. Common Helper Functions ────────────────────────────────────────────────
def get_hemi_stats_file(wildcards):
    year_str = wildcards.year.replace("_preVFP", "").replace("_postVFP", "")
    stats_dir = config.get('hemi_stats_path', 'coffea4bees/skimmer/metadata')
    return f"{stats_dir}/hemi_statistics_{year_str}.yml"

def apply_test_runner_overrides(cfg):
    if config.get("test", False):
        cfg.setdefault("runner", {})
        cfg["runner"]["condor"] = False
        cfg["runner"]["shared_dask"] = False
        cfg["runner"]["workers"] = 2
        cfg["runner"].pop("min_workers", None)
        cfg["runner"].pop("max_workers", None)
        if "chunksize" in config:
            cfg["runner"]["chunksize"] = config["chunksize"]
        elif "chunksize" not in cfg["runner"]:
            cfg["runner"]["chunksize"] = 1000
        if "maxchunks" in config:
            cfg["runner"]["maxchunks"] = config["maxchunks"]
        elif "maxchunks" not in cfg["runner"]:
            cfg["runner"]["maxchunks"] = 1
    return cfg

def rule_exists(rule_name):
    try:
        getattr(rules, rule_name)
        return True
    except Exception:
        return False

def get_multisample_dataset_file(wildcards=None):
    out_file = config.get('multisample_install_path', f"{out}coffea4bees/metadata/datasets/mixeddata_4b.yml")
    repo_file = "coffea4bees/metadata/datasets/mixeddata_4b.yml"
    if rule_exists('build_multisample_registry'):
        return out_file
    if os.path.exists(out_file):
        return out_file
    if os.path.exists(repo_file):
        return repo_file
    return out_file

def get_ttbar_psdata_dataset_file(wildcards=None):
    out_file = config.get('ttbar_psdata_install_path', f"{out}coffea4bees/metadata/datasets/ttbar_PSData_stitched.yml")
    repo_file = "coffea4bees/metadata/datasets/ttbar_PSData_stitched.yml"
    if rule_exists('install_ttbar_psdata_dataset') or rule_exists('create_ttbar_psdata_dataset_yaml'):
        return out_file
    if os.path.exists(out_file):
        return out_file
    if os.path.exists(repo_file):
        return repo_file
    return out_file

def get_mixeddata_dataset_file(wildcards=None):
    ds_name = config.get('mixeddata', {}).get('dataset_name', config.get('dataset_name', 'mixeddata_ttHbb_bkg_syst_rank0_0'))
    out_file = config.get('mixeddata_install_path', f"{out}coffea4bees/metadata/datasets/{ds_name}.yml")
    repo_file = f"coffea4bees/metadata/datasets/{config.get('mixeddata', {}).get('dataset_name', config.get('dataset_name', 'mixeddata_ttHbb_bkg_syst_rank0_0'))}.yml"
    if rule_exists('create_mixeddata_dataset_yaml') or rule_exists('install_mixeddata_dataset_yaml'):
        return out_file
    if os.path.exists(out_file):
        return out_file
    if os.path.exists(repo_file):
        return repo_file
    return out_file
