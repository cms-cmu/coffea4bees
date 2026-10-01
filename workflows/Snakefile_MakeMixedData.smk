# coffea4bees/workflows/Snakefile_MakeMixedData.smk
# Mixed-data dataset production: the top level of the mixeddata roast.
# (Not "Snakefile_MixedData.smk": on a case-insensitive filesystem that is the existing ttHbb
# Snakefile_mixeddata.smk.)
#
#   inputs             fetch the upstream JCM + data/ttbar histograms (non-tight B.1 roast)
#   M.1 hemi library   4b data, ttbar subtracted with the upstream FvT -> hemisphere library + stats
#   M.2 mix            3b data, hemispheres swapped from the library, upstream JCM -> mixeddata_all
#   M.3 validate       mixed-data histograms + upstream data/ttbar -> mixed-data JCM; study
#   M.4 subsample      split mixeddata_all into N disjoint samples (M.3 JCM) -> mixeddata_4b
#   M.5 ttbar psdata   unweighted ttbar pseudodata (one shared sample) -> ttbar_PSData
#   M.6 validation     plots (mixed + ttbar MC vs 4b data, pseudodata, one subsample), cutflow page,
#                      study plots, subsample overlap matrix
#   M.7 signal check   signal MC mixed the same way (3b + 4b), SvB on the fly: does it stay signal-like?
#
# mixing.source: fourTag (config/mixeddata_run3_4bmix.yml) is the 4b-mixing variant: 4b data events
# are mixed instead of 3b x JCM, with their own hemispheres vetoed in the library match, and N seeds
# (random rank among the top k_random neighbours) give the N samples. M.3's JCM fit and M.4's JCM
# splitting drop out: M.2 publishes all seeds as one dataset (mixeddata_all_4bmix, MvD) and M.4
# assembles the per-seed samples + ttbar pseudodata (mixeddata_4bmix_4b, closure).
#
# Everything this roast consumes comes from other roasts, named under `inputs:` and checked by
# `roast new`: the FvT from the nominal, the JCM and its histograms from a Phase B.1 roast with
# the non-tight selection (config/nominal_run3_nontight.yml; in Run 2, the nominal itself).
#
# Products are published to `publish_base` on EOS; the dataset YAMLs under <publish_base>/handoff/
# are what consumer roasts read (runner.py -m accepts root:// URLs). Nothing is installed into the
# checkout. Run one step with a target: `roast submit <id> --step MakeMixedData --targets all_M1`.
#
# This file is the ONLY place the config is read and paths are built: the step files include()d
# below use the names defined here and never call config.setdefault themselves.

import os
import copy
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/mixeddata_run3.yml"

include: "helpers/common.smk"      # {roast_id} substitution

# ── Configuration ─────────────────────────────────────────────────────────────

def _slash(p):
    return p if p.endswith("/") else p + "/"

out = _slash(config['output_path'])
PUB = str(config.get('publish_base', f"{out}publish")).rstrip("/")
HANDOFF = f"{PUB}/handoff"

if 'year_eras' in config:
    YEAR_ERAS = {str(y): list(eras) for y, eras in config['year_eras'].items()}
    YEARS = list(YEAR_ERAS)
else:
    YEARS = [str(y) for y in config.get('years', [])]
    YEAR_ERAS = {y: [] for y in YEARS}
TTBAR = list(config.get('ttbar') or config.get('ttbar_processes') or [])

# Hemisphere-library year keys. Default: one library per data year (UL16_preVFP and UL16_postVFP
# separately; John, 2026-09-27). hemi_library.hemi_year_key: merge_ul16 restores the legacy shared
# UL16 library. The mixer (make_mixed_data.py) is told the same key, so the two cannot disagree.
HEMI_YEAR_KEY = (config.get('hemi_library') or {}).get('hemi_year_key', 'year')
if HEMI_YEAR_KEY not in ('year', 'merge_ul16'):
    raise ValueError(f"hemi_library.hemi_year_key must be 'year' or 'merge_ul16', got {HEMI_YEAR_KEY!r}")

def hemi_year(year):
    """Year key of the hemisphere library / statistics for a data year."""
    if HEMI_YEAR_KEY == 'merge_ul16':
        return year.replace("_preVFP", "").replace("_postVFP", "")
    return year

HEMI_YEARS = list(dict.fromkeys(hemi_year(y) for y in YEARS))

# The mixed data is built with the non-tight four-tag definition (see the config). Refuse to run
# otherwise: every step below would silently build on a different 4b definition.
if config.get('fourTag_use_tight', None) is not False \
        or config['analysis_config']['config'].get('fourTag_use_tight', None) is not False:
    raise ValueError("mixed-data production requires fourTag_use_tight: false, top level and in "
                     "analysis_config.config")

INPUTS = config.get('inputs') or {}
SUBTRACT_TTBAR = (config.get('mixing') or {}).get('subtract_ttbar_with_weights', True)
if SUBTRACT_TTBAR:
    _fvt = str(INPUTS.get('FvT') or "")
    if not _fvt.startswith("root://") and not os.path.exists(_fvt):
        raise ValueError(f"inputs.FvT must be a root:// URL or local path into an upstream roast (see the config)")
    FVT = INPUTS['FvT']
else:
    FVT = INPUTS.get('FvT', None)

for _key in ('JCM', 'jcm_hists'):
    _val = str(INPUTS.get(_key) or "")
    if not _val.startswith("root://") and not os.path.exists(_val):
        raise ValueError(f"inputs.{_key} must be a root:// URL or local path into an upstream roast (see the config)")

# Local copies of the upstream JCM and histograms: jetCombinatoricModel opens the JCM with open()
# (the mixer does so on the submit node, in __init__) and coffea's load() reads local files only.
INPUT_DIR = f"{out}inputs/"
UPSTREAM_JCM = f"{INPUT_DIR}{os.path.basename(INPUTS['JCM'])}"
UPSTREAM_HISTS = f"{INPUT_DIR}{os.path.basename(INPUTS['jcm_hists'])}"
# The runner config those histograms were made with (B.1 writes it next to histAll_NoJCM.coffea):
# M.3 histograms the mixed data with exactly this config, so the mixed-data JCM fit compares like
# with like -- and it proves the upstream really used the non-tight selection.
_upstream_hist_cfg = INPUTS.get('analysis_config_noJCM') or f"{os.path.dirname(INPUTS['jcm_hists'])}/analysis_config_noJCM.yml"
UPSTREAM_HIST_CONFIG_URL = _upstream_hist_cfg
UPSTREAM_HIST_CONFIG = f"{INPUT_DIR}{os.path.basename(_upstream_hist_cfg)}"
MIXED_URL = f"{HANDOFF}/{(config.get('mixing') or {}).get('dataset_name', 'mixeddata_all')}.yml"

# Hemisphere library: built here (M.1) unless inputs.hemilib points at another roast's.
HEMI_EXTERNAL = INPUTS.get('hemilib') or INPUTS.get('hemi_library_yaml')
HEMI_BASE = str(INPUTS.get('hemilib') or "").rstrip("/") if INPUTS.get('hemilib') else f"{PUB}/hemilib"
HEMI_LIB_URL = str(INPUTS.get('hemi_library_yaml') or f"{HEMI_BASE}/hemisphere_library.yml")
HEMI_STATS_URL = str(INPUTS.get('hemi_stats_path') or HEMI_BASE)            # hemi_statistics_<year>.yml live next to the registry

HEMI = config.get('hemi_library') or {}
MIX = config.get('mixing') or {}
MIX_NAME = MIX.get('dataset_name', 'mixeddata_all')

# What is mixed: threeTag (3b data x JCM, the nominal mixed data) or fourTag (4b data; N seeds)
MIX_SOURCE = MIX.get('source', 'threeTag')
if MIX_SOURCE not in ('threeTag', 'fourTag'):
    raise ValueError(f"mixing.source must be 'threeTag' or 'fourTag', got {MIX_SOURCE!r}")
MIX4B = MIX_SOURCE == 'fourTag'

# Samples: 3b mixing splits mixeddata_all into N subsamples with the mixed-data JCM (M.4); 4b mixing
# makes one mixing pass per seed (M.2). Either way subsamples.n is N and subsamples.dataset_name the
# multi-sample dataset.
SUB = config.get('subsamples') or {}
N_SUB = int(SUB.get('n', 16))
SUBSAMPLES = list(range(N_SUB))
SUB_NAME = SUB.get('dataset_name', 'mixeddata_4b')

# runner.py (src/runner/dataset.py:get_dataset_type) decides by name how a dataset is read, and an
# unknown name is MC: mixeddata_all* is one mixed dataset, mixeddata_4b / mixeddata_<tag>_4b a
# multi-sample one with samples mix_v<k> / mix_<tag>_v<k>.
import re as _re
if not MIX_NAME.startswith('mixeddata_all'):
    raise ValueError(f"mixing.dataset_name {MIX_NAME!r} must start with 'mixeddata_all' (runner.py reads any other name as MC)")
if SUB_NAME == 'mixeddata_4b':
    SUB_PREFIX = 'mix'
elif (_m := _re.fullmatch(r"mixeddata_([A-Za-z0-9]+)_4b", SUB_NAME)) and _m.group(1) != 'noTTSub':
    SUB_PREFIX = f"mix_{_m.group(1)}"
else:
    raise ValueError(f"subsamples.dataset_name {SUB_NAME!r} must be 'mixeddata_4b' or 'mixeddata_<tag>_4b' "
                     f"(runner.py reads any other name as MC)")
if MIX4B and SUB_NAME == 'mixeddata_4b':
    raise ValueError("4b mixing needs its own dataset names (e.g. subsamples.dataset_name: mixeddata_4bmix_4b): "
                     "load_datasets_metadata refuses a name defined differently in two -m sources")
PS = config.get('ttbar_psdata') or {}
PS_NAME = PS.get('dataset_name', 'ttbar_PSData')
# M.5's ttbar pseudodata dataset YAML; M.4 folds its files into every subsample (closure pseudo-data
# = mixed subsample + ttbar pseudodata), so it is needed before M.5's own file is included.
PS_DATASET = f"{out}M5/handoff/{PS_NAME}.yml"

# Container / runner invocation, as in Phase B.1
config.setdefault('test', False)
if isinstance(config['test'], str):
    config['test'] = config['test'].lower() in ("true", "1", "yes")
_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', _wrapper)
WRAPPER = config['analysis_container_wrapper']
PYTHON = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
# Processor jobs run on condor through the shared Dask daemon (local only for test / CI). Without
# --condor every runner job processes on the interactive node: 5178c5837 set this to "", and a
# MakeMixedData roast's parallel mixers (4 local workers each, loading the hemisphere library)
# were OOM-killed and took cmslpc338 and cmslpc325 down (2026-09-30).
CONDOR = "" if (config['test'] or os.getenv("CI")) else "--shared-dask --condor"
TEST_FLAG = "-t" if config['test'] else ""

# Shell prefix for any rule that writes to EOS: roast seeds ./proxy/x509_proxy in the checkout.
# ${VAR:-}, not $VAR: snakemake runs shell blocks under `set -u`.
# SINGLE braces: this string is spliced into shell blocks as {EOS_PROXY}, and snakemake does not
# re-format substituted text -- doubled braces (the escape needed when writing it inline in a
# shell block, as Snakefile_PhaseB.smk does) reach bash verbatim as `${{...}}`, a bad substitution
# that killed every publish rule before xrdcp ran.
EOS_PROXY = ('if [ -z "${X509_USER_PROXY:-}" ] && [ -f ./proxy/x509_proxy ]; then '
             'export X509_USER_PROXY="$PWD/proxy/x509_proxy"; fi')

def processor_config(section_config, inherit_config=True, **top):
    """A runner config: analysis_config's processor/dataset_location/friend_file/weights_file/
    runner, with `section_config` merged over analysis_config.config (or, inherit_config=False,
    over nothing -- for the hemisphere-library and mixing processors, which default off the
    histogram-pass settings like apply_btagSF, and the mixer's PicoAOD base takes no **kwargs)
    and `top` over the rest.
    Written with yaml.dump, so there is no sed-patching anywhere in this workflow. (The shared
    analysis_processor rule passes only the config, --datasets, --years and extra_arguments to
    runner.py: processor, friends, weights and condor must all be in here.)"""
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

def check_dataset_yml(path, name, years):
    """Refuse to publish an empty or partial dataset. The skimmer runs with skipbadfiles, so a
    processor error on every chunk becomes an empty registry, runner.py still exits 0, and without
    this the handoff YAML would be published as `<name>: {}` (it was, once: the mixer's JCM
    lookup bug)."""
    with open(path) as f:
        entry = (yaml.safe_load(f) or {}).get(name) or {}
    def nfiles(node):
        if isinstance(node, dict):
            return sum(nfiles(v) for v in node.values())
        return len(node) if isinstance(node, list) else 0
    empty = [y for y in years if not nfiles((entry.get(y) or {}).get('picoAOD'))]
    if empty:
        raise ValueError(f"{path}: dataset {name!r} has no files for {empty} -- the skim failed; "
                         f"see the per-year logs (bad_files) before publishing")

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# ── Inputs from upstream roasts ───────────────────────────────────────────────

rule fetch_inputs:
    output:
        jcm = UPSTREAM_JCM,
        hists = UPSTREAM_HISTS,
        hist_config = UPSTREAM_HIST_CONFIG
    log: f"{INPUT_DIR}fetch.log"
    params:
        jcm = INPUTS['JCM'],
        hists = INPUTS['jcm_hists'],
        hist_config = UPSTREAM_HIST_CONFIG_URL
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p $(dirname {output.jcm}) $(dirname {log})
        fetch_file() {{
            src="$1"; dst="$2"
            if [ "$src" = "$dst" ]; then
                echo "Already staged: $dst" | tee -a {log}
            elif [[ "$src" == root://* ]]; then
                xrdcp -f "$src" "$dst" 2>&1 | tee -a {log}
            elif [ -f "$src" ]; then
                cp -f "$src" "$dst" 2>&1 | tee -a {log}
            else
                echo "Source file not found: $src" | tee -a {log}
                exit 1
            fi
        }}
        fetch_file "{params.jcm}" "{output.jcm}"
        fetch_file "{params.hists}" "{output.hists}"
        fetch_file "{params.hist_config}" "{output.hist_config}"
        for f in "{output.jcm}" "{output.hists}" "{output.hist_config}"; do echo "staged $f" | tee -a {log}; done
        """

# ── Steps ─────────────────────────────────────────────────────────────────────

include: "Snakefile_MakeMixedData_1_hemilib.smk"
include: "Snakefile_MakeMixedData_2_mix.smk"
include: "Snakefile_MakeMixedData_3_validate.smk"
include: "Snakefile_MakeMixedData_4_subsample.smk"
include: "Snakefile_MakeMixedData_5_ttbar_psdata.smk"
include: "Snakefile_MakeMixedData_6_validation.smk"
include: "Snakefile_MakeMixedData_7_signal.smk"

# default_target, not position: an included or inserted rule can never steal the default.
rule all_MakeMixedData:
    default_target: True
    input:
        rules.all_M1.input,
        rules.all_M2.input,
        rules.all_M3.input,
        rules.all_M4.input,
        rules.all_M5.input,
        rules.all_M6.input,
        rules.all_M7.input

localrules: fetch_inputs, all_MakeMixedData
