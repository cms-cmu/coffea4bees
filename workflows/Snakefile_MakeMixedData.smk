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
PUB = str(config['publish_base']).rstrip("/")
HANDOFF = f"{PUB}/handoff"

YEAR_ERAS = {str(y): list(eras) for y, eras in config['year_eras'].items()}
YEARS = list(YEAR_ERAS)
TTBAR = list(config['ttbar'])

def hemi_year(year):
    """Year key of the hemisphere library / statistics. make_mixed_data.py strips _preVFP /
    _postVFP before looking the library up, so both UL16 halves share one UL16 library."""
    return year.replace("_preVFP", "").replace("_postVFP", "")

HEMI_YEARS = list(dict.fromkeys(hemi_year(y) for y in YEARS))

# The mixed data is built with the non-tight four-tag definition (see the config). Refuse to run
# otherwise: every step below would silently build on a different 4b definition.
if config.get('fourTag_use_tight', None) is not False \
        or config['analysis_config']['config'].get('fourTag_use_tight', None) is not False:
    raise ValueError("mixed-data production requires fourTag_use_tight: false, top level and in "
                     "analysis_config.config")

INPUTS = config.get('inputs') or {}
for _key in ('FvT', 'JCM', 'jcm_hists'):
    if not str(INPUTS.get(_key) or "").startswith("root://"):
        # FvT: make_mixed_data.py falls back to a legacy FvT file next to each picoAOD when no FvT
        # friend is given -- a silent substitute for the upstream roast's classifier.
        raise ValueError(f"inputs.{_key} must be a root:// URL into an upstream roast (see the config)")
FVT = INPUTS['FvT']

# Local copies of the upstream JCM and histograms: jetCombinatoricModel opens the JCM with open()
# (the mixer does so on the submit node, in __init__) and coffea's load() reads local files only.
INPUT_DIR = f"{out}inputs/"
UPSTREAM_JCM = f"{INPUT_DIR}{os.path.basename(INPUTS['JCM'])}"
UPSTREAM_HISTS = f"{INPUT_DIR}{os.path.basename(INPUTS['jcm_hists'])}"
# The runner config those histograms were made with (B.1 writes it next to histAll_NoJCM.coffea):
# M.3 histograms the mixed data with exactly this config, so the mixed-data JCM fit compares like
# with like -- and it proves the upstream really used the non-tight selection.
UPSTREAM_HIST_CONFIG_URL = f"{os.path.dirname(INPUTS['jcm_hists'])}/analysis_config_noJCM.yml"
UPSTREAM_HIST_CONFIG = f"{INPUT_DIR}analysis_config_noJCM.yml"
MIXED_URL = f"{HANDOFF}/{(config.get('mixing') or {}).get('dataset_name', 'mixeddata_all')}.yml"

# Hemisphere library: built here (M.1) unless inputs.hemilib points at another roast's.
HEMI_EXTERNAL = INPUTS.get('hemilib')
HEMI_BASE = str(HEMI_EXTERNAL).rstrip("/") if HEMI_EXTERNAL else f"{PUB}/hemilib"
HEMI_LIB_URL = f"{HEMI_BASE}/hemisphere_library.yml"
HEMI_STATS_URL = HEMI_BASE            # hemi_statistics_<year>.yml live next to the registry

HEMI = config.get('hemi_library') or {}
MIX = config.get('mixing') or {}
MIX_NAME = MIX.get('dataset_name', 'mixeddata_all')

# Container / runner invocation, as in Phase B.1
config.setdefault('test', False)
if isinstance(config['test'], str):
    config['test'] = config['test'].lower() in ("true", "1", "yes")
_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', _wrapper)
WRAPPER = config['analysis_container_wrapper']
PYTHON = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
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
        xrdcp -f "{params.jcm}" {output.jcm} 2>&1 | tee {log}
        xrdcp -f "{params.hists}" {output.hists} 2>&1 | tee -a {log}
        xrdcp -f "{params.hist_config}" {output.hist_config} 2>&1 | tee -a {log}
        for f in {params.jcm} {params.hists} {params.hist_config}; do echo "fetched $f" | tee -a {log}; done
        """

# ── Steps ─────────────────────────────────────────────────────────────────────

include: "Snakefile_MakeMixedData_1_hemilib.smk"
include: "Snakefile_MakeMixedData_2_mix.smk"
include: "Snakefile_MakeMixedData_3_validate.smk"
include: "Snakefile_MakeMixedData_4_subsample.smk"
include: "Snakefile_MakeMixedData_5_ttbar_psdata.smk"

# default_target, not position: an included or inserted rule can never steal the default.
rule all_MakeMixedData:
    default_target: True
    input:
        rules.all_M1.input,
        rules.all_M2.input,
        rules.all_M3.input,
        rules.all_M4.input,
        rules.all_M5.input

localrules: fetch_inputs, all_MakeMixedData
