# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_4_closure.smk
#
# Stage C_4: FvT Kinematic, Mixed-Data & ttbar Closure Verification (cmslpc)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Validates the FvT neural network model evaluated in Stage C_3 on subsample v0
# (or configured closure subsample) before proceeding to full Stage F execution.
#
# Executes full kinematic histogramming (m4j, jet pt, eta, dijet masses, etc.)
# over data and ttbar MC with JCM and FvT weights applied (SvB disabled),
# performing essential physics verification:
#   1. FvT Kinematic Closure: Compares 4-tag Data in the Sideband (SB) and
#      4-tag Mixed Data (v0) in SR/SB against the multijet + ttbar background model.
#   2. ttbar Closure Verification: Compares pure ttbar MC against the FvT data-driven
#      predictions from 3b data (TTbar4b_from_d3 and TTbar3b_from_d3).
#   3. Cutflow Validation: Compares event yields across preselection, dijet mass,
#      and signal/sideband regions against reference counts.
#
# TARGET RULES:
#   - `all_bkg_syst_C_4`: Master target producing closure coffea, plots, cutflows,
#     and ttbar comparison tables.
#   - `all_ttbar_MC_vs_d3_bkg_syst`: Produces ttbar MC vs data-driven comparison plots.
#
# INPUTS:
#   - Stage B_1 JCM:  {out_b1}jetCombinatoricModel_SB_mix_v{m}.yml
#   - Stage C_3 FvT:  {out_c}friends/friends_FvT_{mix_name}_v{m}.json
#   - Stage A_3 Mixed Data (v0): {out_a3}histAll_{channel}_mixeddata_v{m}.coffea (optional overlay)
#
# OUTPUTS:
#   - Merged Coffea:    {out_c4}histAll_FvT_closure_v{m}.coffea
#   - Kinematic Plots:  {out_c4}plots_FvT_closure/plots_done.txt
#   - Cutflow Summary:  {out_c4}cutflow_FvT_closure.html, cutflow_validation_FvT_closure.txt
#   - ttbar Validation: {out_c4}plots_ttbar_MC_vs_d3/, plots_ttbar3b_MC_vs_d3/, ttbar_MC_vs_d3_cutflow.html
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc (CPU only, using `./run_container`)
#   - Batch Scheduler: HTCondor / Dask (--cores 4)
# ==============================================================================

import os
import copy
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

roast_id = config.get('roast_id') or config.get('label') or 'nominal'
mix_name = config.get('mix_name', f"{channel}_bkg_syst")

# ── Closure Subsample Resolution ──────────────────────────────────────────────
closure_subsample = str(config.get('closure_subsample', SUBSAMPLES[0] if SUBSAMPLES else '0'))
m0 = closure_subsample

closure_cfg = config.get('fvt_closure') or {}
if not isinstance(closure_cfg, dict):
    closure_cfg = {}

CLOSURE_OPTION_KEYS = (
    'datasets', 'plot_config', 'JCM_file', 'known_counts', 'known_counts_test', 'cutflow_list',
    'reuse_wJCM_ttbar', 'jcm_output_path', 'closure_ttbar', 'ttbar_compare_plot_config',
    'ttbar3b_compare_plot_config', 'wJCM_ttbar_from', 'known_counts_fatal'
)

# ── Datasets & Years ──────────────────────────────────────────────────────────
default_ttbar = config.get('ttbar_processes', ['TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic'])
datasets = closure_cfg.get('datasets', ['data'] + list(default_ttbar))
if isinstance(datasets, str):
    datasets = [d.strip() for d in datasets.split(",") if d.strip()]

DATA_YEARS = [str(y) for y in YEARS]

# Ensure year_eras is well-defined
if 'year_eras' not in config:
    if is_run3:
        config['year_eras'] = {
            '2022_preEE':   ['C', 'D', 'E'],
            '2022_postEE':  ['F', 'G'],
            '2023_preBPix':  ['C', 'D'],
            '2023_postBPix': ['C', 'D'],
        }
    else:
        config['year_eras'] = {
            'UL16_preVFP':  ['C', 'D', 'E', 'F'],
            'UL16_postVFP': ['F', 'G', 'H'],
            'UL17':         ['C', 'D', 'E', 'F'],
            'UL18':         ['A', 'B', 'C', 'D'],
        }

DATA_YEAR_ERA = [
    (str(yr), era)
    for yr in DATA_YEARS
    if yr in config.get('year_eras', {})
    for era in config['year_eras'][yr]
] if 'data' in datasets else []

MC_DATASETS = [d for d in datasets if d != 'data']
TT_DATASETS = [ds for ds in MC_DATASETS if ds.startswith('TT')]

# Optional reuse of Phase B.1 ttbar singlefiles
REUSE_WJCM_TTBAR = closure_cfg.get('reuse_wJCM_ttbar', False)
if isinstance(REUSE_WJCM_TTBAR, str):
    REUSE_WJCM_TTBAR = REUSE_WJCM_TTBAR.lower() in ("true", "1", "yes")

JCM_OUTPUT_PATH = closure_cfg.get('jcm_output_path', out_b1)
if not JCM_OUTPUT_PATH.endswith('/'):
    JCM_OUTPUT_PATH += '/'

WJCM_TTBAR_FROM = closure_cfg.get('wJCM_ttbar_from')
UPSTREAM_TTBAR = f"{out_c4}singlefiles/hist__TTbar_upstream_wJCM.coffea"

def mc_singlefiles():
    if WJCM_TTBAR_FROM and TT_DATASETS:
        other = [f"{out_c4}singlefiles/hist__{ds}__{yr}_v{m0}.coffea"
                 for ds in MC_DATASETS if ds not in TT_DATASETS for yr in DATA_YEARS]
        return [UPSTREAM_TTBAR] + other
    return [mc_singlefile(ds, yr) for ds in MC_DATASETS for yr in DATA_YEARS]

def mc_singlefile(ds, yr):
    if REUSE_WJCM_TTBAR and ds.startswith('TT'):
        cand = f"{JCM_OUTPUT_PATH}singlefiles/hist__{ds}__{yr}_wJCM.coffea"
        if os.path.exists(cand):
            return cand
    return f"{out_c4}singlefiles/hist__{ds}__{yr}_v{m0}.coffea"

# ── Resolve JCM and FvT Friend for Subsample m0 ───────────────────────────────
def _resolve_c4_jcm(year=None):
    filename = f"jetCombinatoricModel_SB_mix_v{m0}_{year}.yml" if year else f"jetCombinatoricModel_SB_mix_v{m0}.yml"
    candidates = [
        os.path.join(out_b1, filename),
        os.path.join(f"coffea4bees/metadata/weights/JCM/{roast_id}", filename),
        os.path.join("coffea4bees/metadata/weights/JCM", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    if HANDOFF_EOS:
        return os.path.join(out_b1, filename)
    return candidates[0]

if per_year_jcm:
    c4_jcm_file = {yr: _resolve_c4_jcm(yr) for yr in DATA_YEARS}
    c4_jcm_input = [p for p in c4_jcm_file.values() if isinstance(p, str) and "://" not in p]
else:
    c4_jcm_file = _resolve_c4_jcm()
    c4_jcm_input = [] if "://" in str(c4_jcm_file) else [c4_jcm_file]

c4_fvt_friend_manifest = f"{out_c}friends/friends_FvT_{mix_name}_v{m0}.json"
# the FvT evaluated on the subsample's synthetic four-tag events (C_3 evaluate_synthetic)
c4_fvt_synthetic_manifest = f"{out_c}friends_synthetic/friends_FvT_{mix_name}_v{m0}.json"

# The closure is checked against the SYNTHETIC four-tag sample the FvT was trained on (process
# mix_v<m0>: the subsample's declustered multijet + ttbar pseudodata), not against the real four-tag
# data -- that check is the nominal workflow's C.4. `plot_config` is a template whose mix_v0 / mix_0
# are set to the closure subsample.
plot_config = closure_cfg.get('plot_config', "coffea4bees/plots/metadata/plotsAll_FvTClosure_synthetic.yml")
c4_plot_config = f"{out_c4}plots_config_FvT_closure_synthetic_v{m0}.yml"
c4_synthetic_config_path = f"{out_c4}synthetic/analysis_config_FvT_closure_synthetic_v{m0}.yml"
c4_synthetic_hist = f"{out_c4}synthetic/histAll_{channel}_mixeddata_v{m0}_FvT.coffea"
c4_closure_hist = f"{out_c4}histAll_FvT_closure_synthetic_v{m0}.coffea"
SYNTHETIC_PROCESS = f"mix_v{m0}"

# Cross-host runs (roast: C on the GPU host, C_4 on cmslpc): fetch the closure subsample's friend
# manifests from the C handoff when this workflow does not make them. A generic {m} pattern, so the
# rule also serves F_1 when both are included (F_1 skips its own copy then).
_c4_fvt_handoff = config.get('fvt_handoff_eos') or globals().get('FVT_HANDOFF_EOS') or HANDOFF_EOS
_c4_rules = {r.name for r in workflow.rules}
if _c4_fvt_handoff and not ({'extract_friend_manifest', 'stage_bkg_syst_friend_manifest'} & _c4_rules):
    rule stage_c4_friend_manifest:
        output:
            f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json"
        wildcard_constraints:
            m = r"\d+"
        params:
            src = f"{_c4_fvt_handoff}/friends/friends_FvT_{mix_name}_v{{m}}.json"
        shell:
            """
            mkdir -p $(dirname {output})
            if command -v xrdcp &>/dev/null; then xrdcp -f '{params.src}' '{output}'; else cp -f '{params.src}' '{output}'; fi
            """
if _c4_fvt_handoff and 'extract_friend_manifest_synthetic' not in _c4_rules:
    rule stage_c4_friend_manifest_synthetic:
        output:
            f"{out_c}friends_synthetic/friends_FvT_{mix_name}_v{{m}}.json"
        wildcard_constraints:
            m = r"\d+"
        params:
            src = f"{_c4_fvt_handoff}/friends_synthetic/friends_FvT_{mix_name}_v{{m}}.json"
        shell:
            """
            mkdir -p $(dirname {output})
            if command -v xrdcp &>/dev/null; then xrdcp -f '{params.src}' '{output}'; else cp -f '{params.src}' '{output}'; fi
            """
CUTFLOW_LIST = closure_cfg.get('cutflow_list', "passJetMult,passPreSel,passDiJetMass,SR_woTrig,SR,SB_woTrig,SB")

ttbar_compare_plot_config = closure_cfg.get('ttbar_compare_plot_config', "coffea4bees/plots/metadata/plotsTTbar_MCvsFromD3.yml")
ttbar3b_compare_plot_config = closure_cfg.get('ttbar3b_compare_plot_config', "coffea4bees/plots/metadata/plotsTTbar3b_MCvsFromD3.yml")

TTBAR_COMPARISON_OUTPUTS = [
    f"{out_c4}plots_ttbar_MC_vs_d3/plots_done.txt",
    f"{out_c4}plots_ttbar3b_MC_vs_d3/plots_done.txt",
    f"{out_c4}ttbar_MC_vs_d3_cutflow.html",
] if any(ds.startswith('TT') for ds in MC_DATASETS) else []

def known_cutflow_flag():
    if config.get("test", False):
        f = closure_cfg.get('known_counts_test') or "coffea4bees/analysis/tests/known_Counts_FvT_closure.yml"
    else:
        f = closure_cfg.get('known_counts') or "coffea4bees/analysis/tests/known_fullCounts_FvT_closure.yml"
    return f'--known-cutflow "{f}"' if os.path.exists(f) else '--known-cutflow "none"'

# ── Staged Analysis Runtime Config Generation ─────────────────────────────────
def get_raw_c4_config():
    res = resolve_config_section(config, primary_key='fvt_closure', fallback_keys=['analysis_config', 'analysis'])
    for k in CLOSURE_OPTION_KEYS:
        res.pop(k, None)
    if 'config' not in res or not isinstance(res['config'], dict):
        res['config'] = {}

    # The four-tag definition: neither `fvt_closure` nor `analysis_config` carries the top-level
    # fourTag_use_tight (Run 3), and without it the processor counts the non-tight four-tag events
    # against the tight-normalised model (Run 3 bkg_syst pilot: data/bkg 1.72).
    if 'fourTag_use_tight' not in res['config'] and 'fourTag_use_tight' in config:
        res['config']['fourTag_use_tight'] = config['fourTag_use_tight']

    # Enable JCM and FvT reweighting
    res['config']['apply_JCM'] = True
    res['config']['JCM_file'] = c4_jcm_file
    res['config']['apply_FvT'] = True
    res['config']['apply_trigWeight'] = True
    res['config']['plot_ttbar_with_weights'] = True
    res['config']['run_SvB'] = False
    res['config']['fill_histograms'] = closure_cfg.get('fill_histograms', 'kinematics+FvT')
    
    # Blinding
    res['config']['blind'] = True

    # Setup friends dict
    friends_dict = res['config'].get('friends', {})
    friends_dict['FvT'] = f"{c4_fvt_friend_manifest}@@FvT"
    if 'trigWeight' not in friends_dict:
        trig_w = config.get('trigweights_file', "coffea4bees/metadata/friends/trigweights_Run2_v2.json@@trigWeight")
        friends_dict['trigWeight'] = trig_w
    res['config']['friends'] = friends_dict

    # Clear SvB entries since SvB is not evaluated here
    for k in list(res['config'].keys()):
        if k.startswith('SvB'):
            res['config'][k] = None

    if config.get("test", False):
        res.setdefault('runner', {})
        res['runner']['condor'] = False
        res['runner']['shared_dask'] = False
        res['runner']['run_performance'] = False
    return res

c4_config_path = f"{out_c4}analysis_config_FvT_closure_v{m0}.yml"

if not globals().get("_ANALYSIS_MODULE_DEFINED", False):
    _ANALYSIS_MODULE_DEFINED = True
    module analysis:
        snakefile: "rules/analysis.smk"
        config: config

# ── Master Stage C_4 Targets ──────────────────────────────────────────────────
rule all_bkg_syst_C_4:
    default_target: True
    input:
        f"{out_c4}histAll_FvT_closure_v{m0}.coffea",
        c4_closure_hist,
        f"{out_c4}plots_FvT_closure/plots_done.txt",
        f"{out_c4}cutflow_validation_FvT_closure.txt",
        f"{out_c4}cutflow_FvT_closure.html",
        TTBAR_COMPARISON_OUTPUTS

rule all_ttbar_MC_vs_d3_bkg_syst:
    input:
        TTBAR_COMPARISON_OUTPUTS

# ── Staging Rules for Cross-Cluster Execution ─────────────────────────────────
_fvt_handoff = config.get('fvt_handoff_eos') or globals().get('FVT_HANDOFF_EOS') or HANDOFF_EOS
_defined_rules = {r.name for r in workflow.rules}
if (_fvt_handoff or HANDOFF_EOS) and 'stage_bkg_syst_c4_friend_manifest' not in _defined_rules and 'extract_friend_manifest' not in _defined_rules:
    rule stage_bkg_syst_c4_friend_manifest:
        output:
            f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json"
        params:
            src = f"{_fvt_handoff}/friends/friends_FvT_{mix_name}_v{{m}}.json"
        shell:
            """
            mkdir -p $(dirname {output})
            if command -v xrdcp &>/dev/null; then
                xrdcp -f '{params.src}' '{output}'
            else
                cp -f '{params.src}' '{output}'
            fi
            """

# ── Config Creation Rule ──────────────────────────────────────────────────────
rule create_bkg_syst_C_4_config:
    input:
        jcm_file = c4_jcm_input,
        fvt_friend = c4_fvt_friend_manifest,
        configfiles = workflow.configfiles if workflow.configfiles else []
    output:
        c4_config_path
    run:
        import yaml, os
        cfg = get_raw_c4_config()
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

# ── The synthetic four-tag side: processor on subsample m0 with its FvT ───────
# As Stage A_3 processes the subsample (same dataset and processor), with C_4's histogram set and
# the FvT evaluated on the synthetic events, so the FvT-output histograms get synthetic entries.
# No JCM (four-tag events) and no trigger weight (synthetic events have none), as A_3.
rule create_bkg_syst_C_4_synthetic_config:
    input:
        fvt_friend = c4_fvt_synthetic_manifest,
        configfiles = workflow.configfiles if workflow.configfiles else []
    output:
        c4_synthetic_config_path
    run:
        import yaml, os
        cfg = get_raw_c4_config()
        c = cfg['config']
        c['apply_JCM'] = False
        c.pop('JCM_file', None)
        c['apply_trigWeight'] = False
        c['plot_ttbar_with_weights'] = False
        c['friends'] = {'FvT': f"{c4_fvt_synthetic_manifest}@@FvT"}
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule analysis_synthetic_bkg_syst_C_4:
    input:
        cfg = c4_synthetic_config_path,
        ds_file = MULTISAMPLE_DATASET,
        fvt_friend = c4_fvt_synthetic_manifest,
    output:
        c4_synthetic_hist
    log:
        f"{out_c4}logs/analysis_synthetic_v{m0}.log"
    params:
        processor = lambda wildcards: get_raw_c4_config().get('processor', f"coffea4bees/analysis/processors/processor_{channel}.py"),
        dataset = f"{SUB_NAME}:{m0}",
        years = " ".join(DATA_YEARS),
        weights_file = config.get('weights_file', f"coffea4bees/metadata/weights/weights_{channel}.yml"),
        output_path = f"{out_c4}synthetic/",
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin),
        extra = lambda wildcards: "" if config.get("test", False) else config.get("additional_parameters", condor_flags),
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_path} $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --metadata {input.ds_file} \
            --datasets {params.dataset} \
            --years {params.years} \
            --weights {params.weights_file} \
            --output-path {params.output_path} \
            --output $(basename {output}) \
            {params.extra} 2>&1 | tee {log}
        test -s {output}
        """

# model (C_4 histAll: 3b data x JCM x FvT, ttbar) + synthetic four-tag in one file for plots and table
use rule merging_coffea_files from analysis as merge_bkg_syst_C_4_synthetic with:
    input:
        files = [f"{out_c4}histAll_FvT_closure_v{m0}.coffea", c4_synthetic_hist],
        script = "src/tools/merge_coffea_files.py"
    output:
        c4_closure_hist
    log:
        f"{out_c4}logs/merge_FvT_closure_synthetic.log"
    params:
        run_performance = False,
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin),
        input_files = lambda wildcards, input: " ".join([f for f in input.files])

rule bkg_syst_C_4_plot_config:
    input:
        plot_config
    output:
        c4_plot_config
    run:
        txt = open(input[0]).read()
        txt = txt.replace("mix_v0", SYNTHETIC_PROCESS).replace("(mix_0)", f"(mix_{m0})")
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            f.write(txt)

# ── Processor Execution Rules ─────────────────────────────────────────────────
use rule analysis_processor from analysis as analysis_data_bkg_syst_C_4 with:
    input:
        runner_script = "runner.py",
        config_file = c4_config_path
    output:
        f"{out_c4}singlefiles/hist_data__{{year}}_{{era}}_v{m0}.coffea"
    log:
        f"{out_c4}logs/analysis_data__{{year}}_{{era}}_v{m0}.log"
    params:
        datasets = "data",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            f"--era {wildcards.era}",
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", condor_flags)
        ])),
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)

use rule analysis_processor from analysis as analysis_MC_bkg_syst_C_4 with:
    input:
        runner_script = "runner.py",
        config_file = c4_config_path
    output:
        f"{out_c4}singlefiles/hist__{{dataset}}__{{year}}_v{m0}.coffea"
    log:
        f"{out_c4}logs/analysis__{{dataset}}_{{year}}_v{m0}.log"
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", condor_flags)
        ])),
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)

rule select_ttbar_upstream_bkg_syst:
    output:
        UPSTREAM_TTBAR
    log:
        f"{out_c4}logs/select_ttbar_upstream.log"
    params:
        url = WJCM_TTBAR_FROM or "",
        processes = " ".join(TT_DATASETS),
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} src/tools/select_processes_coffea.py "{params.url}" \
            -o {output} -p {params.processes} 2>&1 | tee {log}
        """

# ── Merge Coffea Files ────────────────────────────────────────────────────────
use rule merging_coffea_files from analysis as merge_bkg_syst_C_4 with:
    input:
        files = [f"{out_c4}singlefiles/hist_data__{yr}_{era}_v{m0}.coffea" for yr, era in DATA_YEAR_ERA] + mc_singlefiles(),
        script = "src/tools/merge_coffea_files.py"
    output:
        f"{out_c4}histAll_FvT_closure_v{m0}.coffea"
    log:
        f"{out_c4}logs/merge_FvT_closure.log"
    params:
        run_performance = False,
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin),
        input_files = lambda wildcards, input: " ".join([f for f in (input.files if hasattr(input, 'files') else input) if not f.endswith('.py')])

# ── Closure Plots ─────────────────────────────────────────────────────────────
# closure plots: the model against the synthetic four-tag sample (mix_v<m0>), not the real four-tag data
use rule make_plots from analysis as make_plots_bkg_syst_C_4 with:
    input:
        coffea_file = c4_closure_hist,
        metadata_file = c4_plot_config,
        plot_script = "coffea4bees/plots/makePlots.py"
    output:
        f"{out_c4}plots_FvT_closure/plots_done.txt"
    params:
        output_dir = f"{out_c4}plots_FvT_closure/",
        metadata = c4_plot_config,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW -f png",
            "--year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)
    log:
        f"{out_c4}logs/make_plots_FvT_closure.log"

# ── Cutflow Validation ────────────────────────────────────────────────────────
use rule check_cutflow from analysis as check_cutflow_bkg_syst_C_4 with:
    input:
        coffea_file = f"{out_c4}histAll_FvT_closure_v{m0}.coffea"
    output:
        validation_txt = f"{out_c4}cutflow_validation_FvT_closure.txt",
        cutflow_yml = f"{out_c4}cutflow_FvT_closure.yml"
    log:
        f"{out_c4}logs/cutflow_validation_FvT_closure.log"
    params:
        known_flag = lambda wildcards: known_cutflow_flag(),
        fatal = lambda wildcards: "true" if config.get("test", False) else str(closure_cfg.get("known_counts_fatal", False)).lower(),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: CUTFLOW_LIST,
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)
    container: None

# closure table from the merged file: the model columns as before, plus the synthetic four-tag sample
# (mix_v<m0>, the FvT's target) compared with Bkg (--compare)
use rule cutflow_closure_table from analysis as FvT_cutflow_closure_table_bkg_syst with:
    input:
        cutflow_yml = c4_closure_hist,
        validation_txt = f"{out_c4}cutflow_validation_FvT_closure.txt"
    params:
        title = lambda wildcards: f"{config.get('label', 'bkg_syst')}_cutflow_v{m0}_{wildcards.label}",
        multijet = "data3b",
        ttbar = lambda wildcards: closure_cfg.get('closure_ttbar', "TTbar_from_d3"),
        extra_arguments = f"--compare {SYNTHETIC_PROCESS}",
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)

# ── ttbar MC vs Data-Driven Comparisons ───────────────────────────────────────
use rule make_plots from analysis as make_plots_ttbar_MC_vs_d3_bkg_syst with:
    input:
        coffea_file = f"{out_c4}histAll_FvT_closure_v{m0}.coffea",
        metadata_file = ttbar_compare_plot_config,
        plot_script = "coffea4bees/plots/makePlots.py"
    output:
        f"{out_c4}plots_ttbar_MC_vs_d3/plots_done.txt"
    params:
        output_dir = f"{out_c4}plots_ttbar_MC_vs_d3/",
        metadata = ttbar_compare_plot_config,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW -f png",
            "--year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)
    log:
        f"{out_c4}logs/make_plots_ttbar_MC_vs_d3.log"

use rule make_plots from analysis as make_plots_ttbar3b_MC_vs_d3_bkg_syst with:
    input:
        coffea_file = f"{out_c4}histAll_FvT_closure_v{m0}.coffea",
        metadata_file = ttbar3b_compare_plot_config,
        plot_script = "coffea4bees/plots/makePlots.py"
    output:
        f"{out_c4}plots_ttbar3b_MC_vs_d3/plots_done.txt"
    params:
        output_dir = f"{out_c4}plots_ttbar3b_MC_vs_d3/",
        metadata = ttbar3b_compare_plot_config,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW -f png",
            "--year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)
    log:
        f"{out_c4}logs/make_plots_ttbar3b_MC_vs_d3.log"

rule ttbar_MC_vs_d3_cutflow_bkg_syst:
    input:
        cutflow_yml = f"{out_c4}cutflow_FvT_closure.yml"
    output:
        html = f"{out_c4}ttbar_MC_vs_d3_cutflow.html",
        txt = f"{out_c4}ttbar_MC_vs_d3_cutflow.txt"
    log:
        f"{out_c4}logs/ttbar_MC_vs_d3_cutflow.log"
    params:
        title = lambda wildcards: f"{config.get('label', 'bkg_syst')}_v{m0}_ttbar_MC_vs_d3",
        mc = " ".join(ds for ds in MC_DATASETS if ds.startswith('TT')),
        run_container_wrapper = container_wrapper,
        python_bin = lambda wildcards: config.get("python_bin", python_bin)
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        if [ -f src/tools/cutflow_ttbar_compare.py ]; then
            {params.run_container_wrapper} {params.python_bin} src/tools/cutflow_ttbar_compare.py {input.cutflow_yml} \
                -o {output.html} --txt {output.txt} --title {params.title} --mc {params.mc} \
                --estimate TTbar_from_d3 2>&1 | tee {log}
        else
            echo "src/tools/cutflow_ttbar_compare.py not found; skipping" 2>&1 | tee {log}
            echo "<p>ttbar MC vs TTbar_from_d3 table not available</p>" > {output.html}
            cp {log} {output.txt}
        fi
        """

localrules:
    create_bkg_syst_C_4_config, select_ttbar_upstream_bkg_syst, merge_bkg_syst_C_4,
    create_bkg_syst_C_4_synthetic_config, merge_bkg_syst_C_4_synthetic, bkg_syst_C_4_plot_config,
    make_plots_bkg_syst_C_4, check_cutflow_bkg_syst_C_4, FvT_cutflow_closure_table_bkg_syst,
    make_plots_ttbar_MC_vs_d3_bkg_syst, make_plots_ttbar3b_MC_vs_d3_bkg_syst,
    ttbar_MC_vs_d3_cutflow_bkg_syst
