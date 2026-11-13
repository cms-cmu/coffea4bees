# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_4_mvd_closure.smk
#
# Stage C_4 for bkg_model: mvd -- kinematic closure of ONE per-subsample MvD fit against its
# mixed-data target (the FvT C_4's analogue; roast phase BKG_C4_MVD, cmslpc, after BKG_C)
# ==============================================================================
#
# The bkg_syst MvD method trains, per hemisphere-mixed subsample mix_v<m>, an MvD of
#     d4 = mix_v<m> four-tag  vs  mix4 = declustered library (f1_dataset) four-tag x JCM_v<m>  +  t4 = ttbar MC
# and models mix_v<m> as  f1_dataset x JCM_v<m> x MvD_v<m>  +  TTbar4b_from_MvD.  F_2 tests that model
# in SvB_MvD only; this stage checks it in the kinematics (m4j, jets, dijet masses, ...), the
# MvD output and SvB, for the closure subsample m0 (`closure_subsample`, default the first):
#
#   model  : f1_dataset (f1_dataset_location) with JCM_v<m0>, MvD_v<m0> (C_3 friend), trigger
#            weights, SvB from data_svb_friend -- the F_1 model with the one subsample
#   target : mix_v<m0> (Stage A_2 dataset {SUB_NAME}:{m0}) with the MvD evaluated on its own events
#            (C_3 evaluate_synthetic, friends_synthetic/) and its A_3 SvB friend; no JCM / MvD /
#            trigger weight on it (it is not the mixed_all sample, so the processor keeps weight 1)
#   plots  : mix_v<m0> points vs the stack, plotsAll_MvD_roast.yml with data -> mix_v<m0>
#   tables : cutflow dump + cutflow_closure.py (cut x year, mix_v<m0> / Bkg)
#
# No ttbar-MC check: the MvD friend covers the declustered library and the subsample only (the FvT
# C_4 compares ttbar MC with TTbar_from_d3; here TTbar4b_from_MvD is in the stack).
# Options (all optional) under `mvd_closure:`: fill_histograms, plot_config, cutflow_list,
# known_counts.
# ==============================================================================

import os
import copy
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_HH4b_bkg_syst_mvd_declib16_mixed_run3.yml"

include: "helpers/bkg_syst_common.smk"

roast_id = config.get('roast_id') or config.get('label') or 'nominal'
mix_name = config.get('mix_name', f"{channel}_bkg_syst")
m0 = str(config.get('closure_subsample', SUBSAMPLES[0] if SUBSAMPLES else '0'))

if str(config.get('bkg_model', 'fvt')).lower() != 'mvd':
    raise ValueError("Snakefile_bkg_syst_C_4_mvd_closure.smk is the bkg_model: mvd closure; "
                     "use Snakefile_bkg_syst_C_4_closure.smk for the FvT method")
if not config.get('f1_dataset'):
    raise ValueError("bkg_model: mvd needs f1_dataset (the declustered library the MvDs reweight)")

MC4 = config.get('mvd_closure') or {}
out_c4m = f"{out}bkg_syst_C_4_mvd_closure/"
MODEL_DATASET = config['f1_dataset']
TARGET_PROCESS = f"mix_v{m0}"
LABEL = f"MvD_closure_v{m0}"
FILL = MC4.get('fill_histograms', "kinematics+fvt+SvB")
RUN_SVB = bool(config.get('data_svb_friend'))
processor = config.get('analysis_processor') or f"coffea4bees/analysis/processors/processor_{channel}.py"
weights_file = config.get('weights_file', f"coffea4bees/metadata/weights/weights_{channel}.yml")
DATA_YEARS = [str(y) for y in YEARS]

MVD_FRIEND = f"{out_c}friends/friends_FvT_{mix_name}_v{m0}.json"                  # MvD_v<m0> on the library
MVD_SYN_FRIEND = f"{out_c}friends_synthetic/friends_FvT_{mix_name}_v{m0}.json"     # MvD_v<m0> on mix_v<m0>
SVB_SYN_FRIEND = f"{out_a3}histAll_{channel}_mixeddata_v{m0}.json"                # A_3 SvB of mix_v<m0>


def _jcm_v0():
    """JCM_v<m0>: one file, or {year: file} when B_1 fitted per year (as F_1)."""
    def find(fn):
        for d in (out_b1, f"coffea4bees/metadata/weights/JCM/{roast_id}/", "coffea4bees/metadata/weights/JCM/"):
            if os.path.exists(os.path.join(d, fn)):
                return os.path.join(d, fn)
        return os.path.join(out_b1, fn)
    per_year = {y: find(f"jetCombinatoricModel_SB_mix_v{m0}_{y}.yml") for y in DATA_YEARS}
    if all(os.path.exists(p) for p in per_year.values()):
        return per_year
    return find(f"jetCombinatoricModel_SB_mix_v{m0}.yml")

JCM_V0 = _jcm_v0()
JCM_INPUTS = list(JCM_V0.values()) if isinstance(JCM_V0, dict) else [JCM_V0]


def model_config():
    """The F_1 model config (bkg_model: mvd) for the one subsample m0, with kinematic histograms."""
    with open(config.get('analysis_config_mixeddata_bkgs',
                         "coffea4bees/workflows/config/analysis_config_mixeddata_bkgs.yml")) as f:
        cfg = yaml.safe_load(f) or {}
    c = cfg.setdefault('config', {})
    user = (config.get('analysis_config') or {}).get('config', {})
    if isinstance(user, dict):
        c.update(user)
    phase_f = config.get('phase_f_1', config.get('bkg_syst_F_1', config.get('phase_f', {}))) or {}
    if isinstance(phase_f.get('config'), dict):
        c.update(phase_f['config'])
    if isinstance(phase_f.get('runner'), dict):
        cfg.setdefault('runner', {}).update(phase_f['runner'])
    cand = config.get('candidates_selection_cfg') or user.get('candidates_selection_cfg')
    if cand:
        c['candidates_selection_cfg'] = cand
    c['pairing'] = (phase_f.get('config') or {}).get('pairing', user.get('pairing', config.get('pairing', 'nominal')))
    if 'fourTag_use_tight' in config and 'fourTag_use_tight' not in c:
        c['fourTag_use_tight'] = config['fourTag_use_tight']
    c['blind'] = True

    friends = {'trigWeight': config.get('trigweights_file', "coffea4bees/metadata/friends/trigweights_Run2_v2.json@@trigWeight")}
    if RUN_SVB:
        friends['SvB'] = config['data_svb_friend']
        friends['SvB_MA'] = config.get('data_svb_ma_friend', config['data_svb_friend'])
    friends[f"MvD_v{m0}"] = f"{MVD_FRIEND}@@FvT"
    friends['MvD'] = friends[f"MvD_v{m0}"]
    c['friends'] = friends
    c['friends_include'] = list(friends)
    # the F_1 multi-subsample path with the one subsample: weight = JCM_v<m0> x MvD_v<m0>
    c['JCM_file'] = {f"v{m0}": JCM_V0}
    c['subsample_names'] = [f"v{m0}"]
    c['apply_JCM'] = True
    c['apply_trigWeight'] = True
    c['apply_FvT'] = False
    c['plot_ttbar_with_weights'] = False
    c['apply_MvD'] = True
    c['apply_MvD_weight'] = True
    c['plot_ttbar_with_MvD_weights'] = True
    if config.get('mvd_pmix4_floor'):
        c['MvD_pmix4_floor'] = float(config['mvd_pmix4_floor'])
    c['run_SvB'] = RUN_SVB
    c['SvB_MA'] = True if RUN_SVB else None
    if not RUN_SVB:
        c['SvB'] = None
    c['fill_histograms'] = FILL
    if config.get('f1_dataset_location'):
        loc = config['f1_dataset_location']
        cfg['dataset_location'] = list(loc) if isinstance(loc, (list, tuple)) else [loc]
        cfg.pop('datasets_file', None)
    if config.get("test", False):
        cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False, 'run_performance': False})
    return cfg


def target_config():
    """mix_v<m0> with its own MvD evaluation (histograms only: not mixed_all, so weight 1) and SvB."""
    cfg = model_config()
    cfg.pop('dataset_location', None)        # the subsample comes from --metadata (A_2 dataset yml)
    c = cfg['config']
    c['subsample_names'] = []
    c['JCM_file'] = JCM_V0                   # the apply_MvD path needs a JCM object (weight stays 1)
    c['apply_trigWeight'] = False            # synthetic events carry none (as A_3 / the FvT C_4)
    c['plot_ttbar_with_MvD_weights'] = False
    friends = {'MvD': f"{MVD_SYN_FRIEND}@@FvT"}
    if RUN_SVB:
        friends['SvB'] = f"{SVB_SYN_FRIEND}@@SvB_MA"
        friends['SvB_MA'] = f"{SVB_SYN_FRIEND}@@SvB_MA"
    c['friends'] = friends
    c['friends_include'] = list(friends)
    return cfg


def _dump(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


if not globals().get("_ANALYSIS_MODULE_DEFINED", False):
    _ANALYSIS_MODULE_DEFINED = True
    module analysis:
        snakefile: "rules/analysis.smk"
        config: config

MODEL_HIST = f"{out_c4m}model/histAll_{MODEL_DATASET}_v{m0}.coffea"
TARGET_HIST = f"{out_c4m}target/histAll_{TARGET_PROCESS}.coffea"
CLOSURE_HIST = f"{out_c4m}histAll_{LABEL}.coffea"
PLOT_CONFIG = f"{out_c4m}plots_config_{LABEL}.yml"

rule all_bkg_syst_C_4_mvd:
    default_target: True
    input:
        CLOSURE_HIST,
        f"{out_c4m}plots_{LABEL}/plots_done.txt",
        f"{out_c4m}cutflow_{LABEL}.html",

rule create_C_4_mvd_configs:
    input:
        base = config.get('analysis_config_mixeddata_bkgs', "coffea4bees/workflows/config/analysis_config_mixeddata_bkgs.yml"),
        configfiles = workflow.configfiles if workflow.configfiles else []
    output:
        model = f"{out_c4m}model/analysis_config_{LABEL}.yml",
        target = f"{out_c4m}target/analysis_config_{LABEL}.yml"
    run:
        _dump(model_config(), output.model)
        _dump(target_config(), output.target)

# model: the declustered library x JCM_v<m0> x MvD_v<m0> (+ TTbar4b_from_MvD), all years, one condor pass
rule analysis_C_4_mvd_model:
    input:
        cfg = f"{out_c4m}model/analysis_config_{LABEL}.yml",
        jcm = JCM_INPUTS,
        mvd = MVD_FRIEND,
    output:
        MODEL_HIST
    log:
        f"{out_c4m}logs/analysis_model.log"
    params:
        processor = processor,
        dataset = MODEL_DATASET,
        years = " ".join(DATA_YEARS),
        output_path = f"{out_c4m}model/",
        container_wrapper = container_wrapper,
        python_bin = python_bin,
        extra = "" if config.get("test", False) else condor_flags,
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_path} $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output}) \
            {params.extra} 2>&1 | tee {log}
        test -s {output}
        """

# target: the mixed subsample mix_v<m0> (process mix_v<m0>), as Stage A_3 processes it
rule analysis_C_4_mvd_target:
    input:
        cfg = f"{out_c4m}target/analysis_config_{LABEL}.yml",
        ds_file = MULTISAMPLE_DATASET,
        jcm = JCM_INPUTS,
        mvd = MVD_SYN_FRIEND,
        svb = [SVB_SYN_FRIEND] if RUN_SVB else [],
    output:
        TARGET_HIST
    log:
        f"{out_c4m}logs/analysis_target.log"
    params:
        processor = processor,
        dataset = f"{SUB_NAME}:{m0}",
        years = " ".join(DATA_YEARS),
        weights_file = weights_file,
        output_path = f"{out_c4m}target/",
        container_wrapper = container_wrapper,
        python_bin = python_bin,
        extra = "" if config.get("test", False) else condor_flags,
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_path} $(dirname {log})
        test -s {input.mvd} || {{ echo "empty {input.mvd}: run C_3 evaluate_synthetic first" | tee {log}; exit 1; }}
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
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

use rule merging_coffea_files from analysis as merge_C_4_mvd with:
    input:
        files = [MODEL_HIST, TARGET_HIST],
        script = "src/tools/merge_coffea_files.py"
    output:
        CLOSURE_HIST
    log:
        f"{out_c4m}logs/merge.log"
    params:
        run_performance = False,
        run_container_wrapper = container_wrapper,
        python_bin = python_bin,
        input_files = lambda wildcards, input: " ".join(input.files)

# plotsAll_MvD_roast.yml (the MvD roast's data vs mixed x JCM x MvD + TTbar4b_from_MvD) with the
# four-tag data replaced by mix_v<m0>, the multijet by the declustered library, no signal
rule C_4_mvd_plot_config:
    input:
        MC4.get('plot_config', "coffea4bees/plots/metadata/plotsAll_MvD_roast.yml")
    output:
        PLOT_CONFIG
    run:
        with open(input[0]) as f:
            pc = yaml.safe_load(f)
        pc['hists'] = {'data': {**pc['hists']['data'], 'process': TARGET_PROCESS,
                                'label': f"Mixed four-tag (mix_{m0})"}}
        n = 0
        for k, v in pc.get('stack', {}).items():
            if v.get('process') == 'mixeddata_all':
                v['process'] = MODEL_DATASET
                v['label'] = rf"Multijet (declustered $\times$ JCM$_{{{m0}}}$ $\times$ MvD$_{{{m0}}}$)"
                n += 1
            elif v.get('process') == 'TTbar4b_from_MvD':
                v['label'] = rf"$t\bar{{t}}$ (declustered $\times$ JCM$_{{{m0}}}$ $\times$ $p_{{t4}}/p_{{mix4}}$)"
        if not n:
            raise ValueError(f"{input[0]}: no stack entry with process mixeddata_all")
        _dump(pc, output[0])

use rule make_plots from analysis as make_plots_C_4_mvd with:
    input:
        coffea_file = CLOSURE_HIST,
        metadata_file = PLOT_CONFIG,
        plot_script = "coffea4bees/plots/makePlots.py"
    output:
        f"{out_c4m}plots_{LABEL}/plots_done.txt"
    params:
        output_dir = f"{out_c4m}plots_{LABEL}/",
        metadata = PLOT_CONFIG,
        extra_arguments = " ".join(filter(None, [
            "-s xW -f png",
            "--year " + (DATA_YEARS[0] if len(DATA_YEARS) == 1 else ("Run3" if any("202" in y for y in DATA_YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        run_container_wrapper = container_wrapper,
        python_bin = python_bin
    log:
        f"{out_c4m}logs/make_plots.log"

use rule check_cutflow from analysis as check_cutflow_C_4_mvd with:
    input:
        coffea_file = CLOSURE_HIST
    output:
        validation_txt = f"{out_c4m}cutflow_validation_{LABEL}.txt",
        cutflow_yml = f"{out_c4m}cutflow_{LABEL}.yml"
    log:
        f"{out_c4m}logs/cutflow_validation.log"
    params:
        known_flag = lambda wildcards: (f'--known-cutflow "{MC4["known_counts"]}"'
                                        if MC4.get('known_counts') and os.path.exists(MC4['known_counts'])
                                        else '--known-cutflow "none"'),
        fatal = "false",
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: MC4.get('cutflow_list', "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = container_wrapper,
        python_bin = python_bin
    container: None

# cut x year: mix_v<m0> (as "data") vs Bkg = declustered x JCM x MvD (four-tag) + TTbar4b_from_MvD
use rule cutflow_closure_table from analysis as cutflow_closure_C_4_mvd with:
    input:
        cutflow_yml = f"{out_c4m}cutflow_{LABEL}.yml",
        validation_txt = f"{out_c4m}cutflow_validation_{LABEL}.txt"
    output:
        html = f"{out_c4m}cutflow_{LABEL}.html",
        txt = f"{out_c4m}cutflow_{LABEL}_table.txt"
    log:
        f"{out_c4m}logs/cutflow_closure.log"
    params:
        title = f"{config.get('label', 'bkg_syst')}_cutflow_{LABEL}",
        multijet = "mixed4b",
        ttbar = "TTbar4b_from_MvD",
        extra_arguments = f"--multijet-process {MODEL_DATASET} --data {TARGET_PROCESS}",
        run_container_wrapper = container_wrapper,
        python_bin = python_bin

localrules:
    all_bkg_syst_C_4_mvd, create_C_4_mvd_configs, merge_C_4_mvd, C_4_mvd_plot_config,
    make_plots_C_4_mvd, check_cutflow_C_4_mvd, cutflow_closure_C_4_mvd
