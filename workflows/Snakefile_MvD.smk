# coffea4bees/workflows/Snakefile_MvD.smk
# MvD roast: the mixed-data background model (mixed data x JCM x MvD) and an SvB trained against
# it, through to stat-only limits -- the MvD analogue of nominal Phases B.2/C/D/F. Consumes two
# upstream roasts, named under `inputs:` and checked by `roast new`:
#   nominal roast     data / ttbar / signal classifier inputs + the B.2 config that made them,
#                     the Phase B.1 data + ttbar JCM histograms, the Phase F.1 analysis config
#   mixeddata roast   handoff/mixeddata_all.yml
#
#   V.1 inputs    (cmslpc, this file, --targets all_V1)
#                 mixeddata_all classifier inputs made with the nominal B.2 config, merged with the
#                 nominal manifest; one classifier metadata YAML (committed data/ttbar/signal +
#                 this roast's mixeddata_all); the mixed-data JCM refit with the TIGHT selection
#                 (the mixeddata roast's is non-tight); all published to <PUB>/handoff/
#   V.2 MvD       (falcon, Snakefile_MvD_2_train.smk, config `mvd:`)       -> friend/MvD
#   V.2c closure  (cmslpc, this file, --targets all_V2c): the Phase C.4 analogue -- data vs
#                 mixed x JCM x MvD + TTbar4b_from_MvD, no SvB: plots, cutflow, closure table
#   V.3 SvB       (falcon, Snakefile_MvD_3_svb.smk, config `svb_mvd:`)     -> friend/SvB_MvD
#   V.4 analysis  (cmslpc, this file, --targets all_V4)
#                 processor on data / mixeddata_all (JCM x MvD, TTbar4b_from_MvD) / signal with the
#                 SvB_MvD scores, plots, cutflow, stat-only limits (Phase F.2 rules)
#
# Nothing is installed into the checkout; the steps hand off through EOS. As in
# Snakefile_MakeMixedData.smk, this file is the only place the config is read and paths are built.

import os
import copy
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/mvd_run3.yml"

include: "helpers/common.smk"      # {roast_id} substitution, check_handoff_refs

# ── Configuration ─────────────────────────────────────────────────────────────

def _slash(p):
    return p if p.endswith("/") else p + "/"

out = _slash(config['output_path'])
PUB = str(config['publish_base']).rstrip("/")
HANDOFF = f"{PUB}/handoff"

YEAR_ERAS = {str(y): list(eras) for y, eras in config['year_eras'].items()}
YEARS = list(YEAR_ERAS)

# MvD uses the nominal (tight) four-tag definition (John, 2026-09-27): the mixed data is only the
# 3b-like input, re-selected here with the tight selection, so its JCM is refit (V.1).
if config.get('fourTag_use_tight', None) is not True:
    raise ValueError("the MvD roast requires fourTag_use_tight: true")

INPUTS = config.get('inputs') or {}
for _key in ('mixeddata_all', 'classifier_inputs', 'classifier_inputs_config', 'JCM', 'jcm_hists',
             'analysis_config'):
    if not str(INPUTS.get(_key) or "").startswith("root://"):
        raise ValueError(f"inputs.{_key} must be a root:// URL into an upstream roast (see the config)")

MIX_NAME = 'mixeddata_all'
MIXED_URL = INPUTS['mixeddata_all']

# Local copies of what the processors / merge tools open with plain open() or coffea load().
INPUT_DIR = f"{out}inputs/"
UPSTREAM = {
    'manifest':  (INPUTS['classifier_inputs'],        f"{INPUT_DIR}classifier_inputs_nominal.json"),
    'ci_config': (INPUTS['classifier_inputs_config'], f"{INPUT_DIR}classifier_inputs_config_nominal.yml"),
    'jcm':       (INPUTS['JCM'],                      f"{INPUT_DIR}{os.path.basename(INPUTS['JCM'])}"),
    'hists':     (INPUTS['jcm_hists'],                f"{INPUT_DIR}{os.path.basename(INPUTS['jcm_hists'])}"),
    # the runner config B.1 made those histograms with (written next to them): V.1 histograms the
    # mixed data with exactly this config, so the JCM fit compares like with like
    'hist_config': (f"{os.path.dirname(INPUTS['jcm_hists'])}/analysis_config_noJCM.yml",
                    f"{INPUT_DIR}analysis_config_noJCM.yml"),
    'analysis_config': (INPUTS['analysis_config'],    f"{INPUT_DIR}analysis_config_nominal.yml"),
    'mixed':     (MIXED_URL,                          f"{INPUT_DIR}{MIX_NAME}.yml"),
}

# Products (V.1 publishes; V.2/V.3 read them from EOS through the mvd / svb_mvd blocks' overrides)
MANIFEST_NAME = "classifier_inputs_MvD.json"
METADATA_NAME = "classifier_metadata.yml"
MJ = config.get('mixed_jcm') or {}
JCM_TAG = MJ.get('tag', "mixeddata_tight")
JCM_NAME = f"jetCombinatoricModel_SB_{JCM_TAG}.yml"

# The two falcon steps' friends, read by V.4 (defaults as the mvd / svb_mvd blocks name them)
def _friend(block, default_label):
    blk = config.get(block) or {}
    base = str(blk.get('eos_base') or PUB).rstrip("/")
    return f"{base}/friend/{blk.get('label', default_label)}/result.json@@analysis.0.merged"
MVD_FRIEND = _friend('mvd', 'MvD')
SVB_FRIEND = _friend('svb_mvd', 'SvB_MvD')

# Container / runner invocation, as in Snakefile_MakeMixedData.smk
config.setdefault('test', False)
if isinstance(config['test'], str):
    config['test'] = config['test'].lower() in ("true", "1", "yes")
_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', _wrapper)
WRAPPER = config['analysis_container_wrapper']
PYTHON = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
CONDOR = "" if (config['test'] or os.getenv("CI")) else "--shared-dask --condor"
TEST_FLAG = "-t" if config['test'] else ""

# SINGLE braces: spliced into shell blocks as {EOS_PROXY} (see Snakefile_MakeMixedData.smk).
EOS_PROXY = ('if [ -z "${X509_USER_PROXY:-}" ] && [ -f ./proxy/x509_proxy ]; then '
             'export X509_USER_PROXY="$PWD/proxy/x509_proxy"; fi')

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}

def write_yaml(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)

def require_tight(cfg, what):
    tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
    if tight is not True:
        raise ValueError(f"{what} has fourTag_use_tight={tight!r}; the MvD roast needs the tight selection")

def test_runner(cfg):
    if config['test']:
        cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False, 'run_performance': False})
    return cfg

def mvd_analysis_config(upstream, signal, svb=True, extra=None):
    """The nominal Phase F.1 analysis config (inputs.analysis_config) with the background model
    swapped: FvT -> MvD (V.2), the JCM -> V.1's tight mixed-data fit, and, with svb, SvB_MA -> V.3's
    SvB_MvD. signal: MvD off (the MvD friend covers data + mixed data only; the processor reads
    event.MvD whenever apply_MvD_weight is on). svb=False is V.2c's closure pass (no SvB at all).
    `extra` (a config dict) is merged last."""
    cfg = load_yaml(upstream)
    require_tight(cfg, f"the upstream analysis config ({INPUTS['analysis_config']})")
    # Data / signal from the committed metadata, mixeddata_all from the mixeddata roast. Files, not
    # the directory: runner -m refuses two sources defining one dataset differently, and the
    # directory's mixeddata_all.yml is the legacy sample.
    cfg['dataset_location'] = [p for p in V1_METADATA_FILES if not p.endswith("TT.yml")] + [MIXED_URL]
    c = cfg.setdefault('config', {})
    c['apply_FvT'] = False
    c['plot_ttbar_with_weights'] = False          # TTbar4b_from_d3 needs the FvT
    c['JCM_file'] = MIXED_JCM
    c['run_SvB'] = bool(svb)
    friends = {} if signal else {'MvD': MVD_FRIEND}
    if svb:
        friends['SvB_MA'] = SVB_FRIEND
    else:
        c['SvB_MA'] = None
        c['SvB'] = None
    c['friends'] = friends
    # runner.py applies friends_include to the MERGED set (per-year file + config.friends), so the
    # friends named above must be listed too -- ['trigWeight'] alone silently dropped the MvD and
    # the processor raised "apply_MvD=True but no 'MvD' entry found in friends dict".
    c['friends_include'] = ['trigWeight', *friends]
    c['apply_MvD'] = not signal
    c['apply_MvD_weight'] = not signal
    c['plot_ttbar_with_MvD_weights'] = not signal
    c.update(copy.deepcopy(extra or {}))
    return test_runner(cfg)

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# ── Inputs from upstream roasts ───────────────────────────────────────────────

rule fetch_inputs:
    output: [local for _, local in UPSTREAM.values()]
    log: f"{INPUT_DIR}fetch.log"
    params:
        pairs = " ".join(f'"{url}" {local}' for url, local in UPSTREAM.values())
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        set -- {params.pairs}
        : > {log}
        while [ $# -gt 0 ]; do
            xrdcp -f "$1" "$2" 2>&1 | tee -a {log}
            echo "fetched $1" | tee -a {log}
            shift 2
        done
        """

# ── Steps ─────────────────────────────────────────────────────────────────────

include: "Snakefile_MvD_1_inputs.smk"
include: "Snakefile_MvD_2c_closure.smk"
include: "Snakefile_MvD_4_analysis.smk"

rule all_MvD:
    default_target: True
    input:
        rules.all_V1.input,
        rules.all_V2c.input,
        rules.all_V4.input

localrules: fetch_inputs, all_MvD
