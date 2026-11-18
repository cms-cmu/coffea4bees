# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_F_1_analysis.smk
#
# Stage F_1: Data Background Analysis & Mixed Data Closure Verification
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Executes full analysis processing on 3-tag collision data to derive the data-driven
# multijet background model for each of the 16 independent pseudo-experiments
# (mix_0..mix_15). Each subsample evaluation uses its dedicated Jet Combinatoric
# Model (JCM) calibrated in Stage B_1 and trained FvT neural network weights from
# Stage C. The resulting background model distributions are directly compared
# against the 4-tag mixed data pseudo-data to evaluate baseline closure.
#
# MATHEMATICAL FORMULATION:
# The data-driven multijet background prediction in the 4-tag signal region is
# modeled by reweighting 3-tag collision data:
#   N_bkg^{4b}(x) = \sum_{i \in 3b} w_JCM(i) * w_FvT(i; x)
# where:
#   - w_JCM(i) is the event-level combinatoric pseudo-tagging weight promoting
#     3-tag events to the 4-tag phase space (derived in Stage B_1).
#   - w_FvT(i; x) is the kinematics reweighting factor predicted by the FvT
#     neural network (trained in Stage C_2 and evaluated in Stage C_3).
# The closure test evaluates the degree of agreement:
#   R(x) = N_bkg^{4b}(x) / N_mixed^{4b}(x) \approx 1
# across all kinematic observables and control/signal regions.
#
# WORKFLOW EXECUTION PIPELINE / RULES BREAKDOWN:
#   1. Input Preparation & Config Staging (`stage_phaseF_1_configs`):
#      - Reads base dataset configs and stages dedicated per-subsample YAMLs
#        injecting dedicated JCM and FvT friend tree definitions.
#   2. Data Background Processor (`analysis_data_closure`):
#      - Runs `runner.py` with `processor_ttHbb.py` over 3-tag collision data
#        using HTCondor/Dask batching across all analysis years (UL16..UL18).
#      - Produces: `closure_v{m}/histAll_data_v{m}.coffea`
#   3. Mixed Data Symlinking (`link_mixeddata_closure`):
#      - Symlinks Stage A_3 unweighted 4-tag mixed data histograms:
#        `histAll_ttHbb_mixeddata_v{m}.coffea` -> `closure_v{m}/histAll_mixeddata_v{m}.coffea`
#   4. Closure Comparison Plotting (`make_plots_closure`):
#      - Executes `coffea4bees/plots/makePlots.py` to compare Data background model
#        against Mixed Data target distributions across all regions (SR, SB).
#   5. Gallery Generation (`make_gallery_closure`):
#      - Compiles an HTML web gallery (`gallery.html`) for rapid visual inspection.
#
# TARGET RULES:
#   - `all_bkg_syst_F_1`: Master target (generates all closure plots and galleries).
#   - `all_bkg_syst_F_1_hists`: Intermediate target (generates only coffea files).
#   - `closure_v`: Convenience rule for a single subsample (e.g. `closure_v0`).
#
# INPUTS:
#   - Dedicated JCM YAMLs: output/.../bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}.yml
#   - FvT Friend Manifests: output/.../bkg_syst_C_FvT/friends/friends_FvT_{mix_name}_v{m}.json
#   - Stage A_3 Mixed Data: output/.../bkg_syst_A_3_process_subsamples/histAll_ttHbb_mixeddata_v{m}.coffea
#   - Plotting Metadata: coffea4bees/plots/metadata/plots_bkg_syst_closure_ttHbb.yml
#
# OUTPUTS:
#   - Data Background Coffea: output/.../bkg_syst_F_1_analysis/closure_v{m}/histAll_data_v{m}.coffea
#   - Mixed Data Symlink: output/.../bkg_syst_F_1_analysis/closure_v{m}/histAll_mixeddata_v{m}.coffea
#   - Closure Plots: output/.../bkg_syst_F_1_analysis/closure_v{m}/plots/*.png
#   - Summary Gallery: output/.../bkg_syst_F_1_analysis/closure_v{m}/plots/gallery.html
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc (CPU only, using `./run_container`)
#   - Batch Scheduler: HTCondor / Dask (--cores 4)
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

mix_name = config.get('mix_name', f"{channel}_bkg_syst")
roast_id = config.get('roast_id') or config.get('label') or 'nominal'
# bkg_model: fvt (default; 3b data x JCM_v x FvT_v) or mvd (the mixed_all sample of f1_dataset x
# JCM_v x MvD_v: closure of the MvD method, the C stage trains MvD models)
BKG_MODEL = str(config.get('bkg_model', 'fvt')).lower()

def _detect_per_year_jcm():
    if 'per_year_jcm' in config:
        return bool(config['per_year_jcm'])
    if (config.get('phaseB_1', {}) or {}).get('per_year_jcm') is not None:
        return bool(config['phaseB_1']['per_year_jcm'])
    sample_y = YEARS[0] if YEARS else "UL16_preVFP"
    m0 = SUBSAMPLES[0] if SUBSAMPLES else 0
    candidates = [
        f"{out_b1}jetCombinatoricModel_SB_mix_v{m0}_{sample_y}.yml",
        f"coffea4bees/metadata/weights/JCM/jetCombinatoricModel_SB_mix_v{m0}_{sample_y}.yml",
        f"coffea4bees/metadata/weights/JCM/{roast_id}/jetCombinatoricModel_SB_mix_v{m0}_{sample_y}.yml",
    ]
    if any(os.path.exists(p) for p in candidates):
        return True
    if os.path.isdir(out_b1):
        for f in os.listdir(out_b1):
            if f.startswith(f"jetCombinatoricModel_SB_mix_v{m0}_") and f.endswith(".yml"):
                return True
    return False

per_year_jcm = _detect_per_year_jcm()

def _resolve_jcm(m, year=None):
    filename = f"jetCombinatoricModel_SB_mix_v{m}_{year}.yml" if year else f"jetCombinatoricModel_SB_mix_v{m}.yml"
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
    raise FileNotFoundError(f"Could not locate JCM file '{filename}' in {out_b1}, coffea4bees/metadata/weights/JCM/, or EOS handoff")

def get_all_analysis_jcm_inputs(wildcards):
    if per_year_jcm:
        return [_resolve_jcm(m, y) for m in SUBSAMPLES for y in YEARS]
    return [_resolve_jcm(m) for m in SUBSAMPLES]

# ── Stage F_1 Runtime Config Generation ───────────────────────────────────────
rule stage_phaseF_1_config:
    input:
        base_cfg = config.get('analysis_config_mixeddata_bkgs', "coffea4bees/workflows/config/analysis_config_mixeddata_bkgs.yml"),
        jcm = get_all_analysis_jcm_inputs,
        fvt = expand(f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json", m=SUBSAMPLES),
    output:
        cfg = f"{out_f1}analysis_config_mixeddata_bkgs.yml"
    run:
        with open(input.base_cfg, 'r') as f:
            data_cfg = yaml.safe_load(f) or {}

        user_analysis_cfg = (config.get('analysis_config') or {}).get('config', {})
        if isinstance(user_analysis_cfg, dict):
            data_cfg.setdefault('config', {}).update(user_analysis_cfg)

        cand_sel_cfg = (
            config.get('candidates_selection_cfg')
            or user_analysis_cfg.get('candidates_selection_cfg')
            or (config.get('phase_f_1') or {}).get('config', {}).get('candidates_selection_cfg')
        )
        if cand_sel_cfg:
            data_cfg.setdefault('config', {})['candidates_selection_cfg'] = cand_sel_cfg

        phase_f_cfg = config.get('phase_f_1', config.get('bkg_syst_F_1', config.get('phase_f', {})))
        if isinstance(phase_f_cfg.get('config'), dict):
            data_cfg.setdefault('config', {}).update(phase_f_cfg['config'])
        if isinstance(phase_f_cfg.get('runner'), dict):
            data_cfg.setdefault('runner', {}).update(phase_f_cfg['runner'])

        # Enforce blinding: default True
        data_cfg.setdefault('config', {})['blind'] = phase_f_cfg.get('config', {}).get('blind', user_analysis_cfg.get('blind', True))

        fvt_floor = phase_f_cfg.get('config', {}).get('FvT_pd3_floor', phase_f_cfg.get('FvT_pd3_floor', config.get('FvT_pd3_floor', user_analysis_cfg.get('FvT_pd3_floor', 0.01))))
        data_cfg.setdefault('config', {})['FvT_pd3_floor'] = float(fvt_floor)

        default_pairing = 'nominal'
        pairing = phase_f_cfg.get('config', {}).get('pairing', user_analysis_cfg.get('pairing', config.get('pairing', default_pairing)))
        data_cfg.setdefault('config', {})['pairing'] = pairing

        fvt_clamp = phase_f_cfg.get('config', {}).get('FvT_max_clamp', user_analysis_cfg.get('FvT_max_clamp', config.get('FvT_max_clamp', None)))
        if fvt_clamp is not None:
            data_cfg.setdefault('config', {})['FvT_max_clamp'] = float(fvt_clamp)

        jcm_file = {}
        for m in SUBSAMPLES:
            v_name = f"v{m}"
            if per_year_jcm:
                jcm_file[v_name] = {}
                for y in YEARS:
                    jcm_file[v_name][y] = _resolve_jcm(m, y)
            else:
                jcm_file[v_name] = _resolve_jcm(m)

        friends_dict = {
            "trigWeight": config.get('trigweights_file', "coffea4bees/metadata/friends/trigweights_Run2_v2.json@@trigWeight"),
        }
        if 'data_svb_friend' in config:
            friends_dict["SvB"] = config['data_svb_friend']
        if 'data_svb_ma_friend' in config:
            friends_dict["SvB_MA"] = config['data_svb_ma_friend']
        elif 'data_svb_friend' in config and "SvB_MA" not in friends_dict:
            friends_dict["SvB_MA"] = config['data_svb_friend']
        # The per-subsample classifier friends (C_3 manifests, key FvT whatever the model): FvT_v<m>
        # on 3b data (FvT method), or MvD_v<m> on the mixed_all sample (bkg_model: mvd)
        clf = "MvD" if BKG_MODEL == "mvd" else "FvT"
        for m in SUBSAMPLES:
            friends_dict[f"{clf}_v{m}"] = f"{out_c}friends/friends_FvT_{mix_name}_v{m}.json@@FvT"
        friends_dict[clf] = friends_dict[f"{clf}_v{SUBSAMPLES[0]}"]

        data_cfg['config']['JCM_file'] = jcm_file
        data_cfg['config']['subsample_names'] = [f"v{m}" for m in SUBSAMPLES]
        data_cfg['config']['friends'] = friends_dict

        data_cfg['config']['apply_JCM'] = True
        data_cfg['config']['apply_trigWeight'] = True
        # f1_dataset_location: where runner finds f1_dataset (e.g. the declustered library's handoff
        # mixeddata_all_<tag>.yml, a root:// URL; {roast:...} is resolved in the workflow config only)
        if config.get('f1_dataset_location'):
            loc = config['f1_dataset_location']
            data_cfg['dataset_location'] = list(loc) if isinstance(loc, (list, tuple)) else [loc]
            data_cfg.pop('datasets_file', None)
        if BKG_MODEL == "mvd":
            # MvD method: mixed_all four-tag x JCM_v (by nSelJets) x MvD_v, ttbar from p_t4/p_mix4
            data_cfg['config']['apply_FvT'] = False
            data_cfg['config']['plot_ttbar_with_weights'] = False
            data_cfg['config']['apply_MvD'] = True
            data_cfg['config']['apply_MvD_weight'] = True
            data_cfg['config']['plot_ttbar_with_MvD_weights'] = True
            data_cfg['config']['friends_include'] = ['trigWeight', *friends_dict]
            if config.get('mvd_pmix4_floor'):
                data_cfg['config']['MvD_pmix4_floor'] = float(config['mvd_pmix4_floor'])
        else:
            # FvT method: JCM, FvT, trigger weights, and ttbar estimation from 3b data
            data_cfg['config']['apply_FvT'] = True
            data_cfg['config']['plot_ttbar_with_weights'] = True

        os.makedirs(os.path.dirname(output.cfg), exist_ok=True)
        with open(output.cfg, 'w') as f:
            yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

closure_plot_cfg = config.get(
    'closure_plot_config',
    "coffea4bees/plots/metadata/plots_bkg_syst_closure_ttHbb.yml"
)

wildcard_constraints:
    m = r"\d+"

localrules: all_bkg_syst_F_1, all_bkg_syst_F_1_hists, make_plots_closure, make_gallery_closure, stage_phaseF_1_config, validate_f1_normalization, hist_crosscheck_bkg_syst_F_1

# ── Master Target Rule ────────────────────────────────────────────────────────
rule all_bkg_syst_F_1:
    input:
        f"{out_f1}validation_normalization.done",
        f"{out_f1}plots/plots_done.txt",
        f"{out_f1}plots/gallery.html",
        f"{out_f1}hist_crosscheck_bkg_syst_F_1.html"

# ── Histogram Only Target Rule (Before Plotting) ──────────────────────────────
rule all_bkg_syst_F_1_hists:
    input:
        f"{out_f1}histAll_mixeddata_bkgs.coffea"

per_year_jcm = bool(config.get('per_year_jcm', (config.get('phaseB_1', {}) or {}).get('per_year_jcm', False)))

# Staging rules for cross-cluster execution under roast (fetching from EOS handoff if not local).
# Only when F_1 runs without the stage that makes the file (Snakefile_bkg_syst.smk includes B_1 and
# C before F: two rules for one output are ambiguous).
_handoff_in = config.get('input_handoff_eos') or globals().get('INPUT_HANDOFF_EOS') or HANDOFF_EOS
_fvt_handoff = config.get('fvt_handoff_eos') or globals().get('FVT_HANDOFF_EOS') or HANDOFF_EOS
_defined_rules = {r.name for r in workflow.rules}
if (_fvt_handoff or HANDOFF_EOS) and not ({'extract_friend_manifest', 'stage_c4_friend_manifest'} & _defined_rules):
    rule stage_bkg_syst_friend_manifest:
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

if 'make_subsample_jcm_b1' not in _defined_rules:
    rule stage_bkg_syst_jcm:
        output:
            f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}.yml"
        params:
            eos_src = f"{_handoff_in}/JCM/jetCombinatoricModel_SB_mix_v{{m}}.yml" if _handoff_in else "",
            meta_roast = f"coffea4bees/metadata/weights/JCM/{roast_id}/jetCombinatoricModel_SB_mix_v{{m}}.yml",
            meta = f"coffea4bees/metadata/weights/JCM/jetCombinatoricModel_SB_mix_v{{m}}.yml",
        shell:
            """
            mkdir -p $(dirname {output})
            if [ -f '{params.meta_roast}' ]; then
                cp -f '{params.meta_roast}' '{output}'
            elif [ -f '{params.meta}' ]; then
                cp -f '{params.meta}' '{output}'
            elif [ -n '{params.eos_src}' ]; then
                if command -v xrdcp &>/dev/null; then
                    xrdcp -f '{params.eos_src}' '{output}'
                else
                    cp -f '{params.eos_src}' '{output}'
                fi
            else
                echo "ERROR: JCM for v{wildcards.m} not found in metadata or EOS" >&2
                exit 1
            fi
            """

    rule stage_bkg_syst_jcm_per_year:
        output:
            f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml"
        params:
            eos_src = f"{_handoff_in}/JCM/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml" if _handoff_in else "",
            meta_roast = f"coffea4bees/metadata/weights/JCM/{roast_id}/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml",
            meta = f"coffea4bees/metadata/weights/JCM/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml",
        shell:
            """
            mkdir -p $(dirname {output})
            if [ -f '{params.meta_roast}' ]; then
                cp -f '{params.meta_roast}' '{output}'
            elif [ -f '{params.meta}' ]; then
                cp -f '{params.meta}' '{output}'
            elif [ -n '{params.eos_src}' ]; then
                if command -v xrdcp &>/dev/null; then
                    xrdcp -f '{params.eos_src}' '{output}'
                else
                    cp -f '{params.eos_src}' '{output}'
                fi
            else
                echo "ERROR: JCM for v{wildcards.m}_{wildcards.year} not found in metadata or EOS" >&2
                exit 1
            fi
            """


# ── Data Background Model Processor (Single Pass over Collision Data) ──────────
rule analysis_data_closure:
    input:
        cfg = f"{out_f1}analysis_config_mixeddata_bkgs.yml",
        jcm = get_all_analysis_jcm_inputs,
        fvt = expand(f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json", m=SUBSAMPLES),
    output:
        coffea = f"{out_f1}histAll_mixeddata_bkgs.coffea"
    log:
        f"{out_f1}logs/analysis_data.log"
    params:
        processor = config.get('analysis_processor') or f"coffea4bees/analysis/processors/processor_{channel}.py",
        # the background-model sample: 3b data (FvT method) or the mixed_all sample (bkg_model: mvd)
        dataset = config.get('f1_dataset') or "data",
        years = " ".join(YEARS),
        output_path = out_f1,
        condor_flags = condor_flags,
        container_wrapper = container_wrapper,
        python_bin = python_bin
    resources:
        slurm_partition = "work",
        qos = "light",
        mem_mb = 16000,
        cpus_per_task = 8,
        runtime = 240
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

# ── Validate Background Normalization (Hard Abort if Multijet Yield is Unphysical) 
rule validate_f1_normalization:
    input:
        coffea = f"{out_f1}histAll_mixeddata_bkgs.coffea"
    output:
        done = f"{out_f1}validation_normalization.done"
    log:
        f"{out_f1}logs/validate_normalization.log"
    params:
        container_wrapper = container_wrapper,
        python_bin = python_bin,
        # sanity bounds (validate_f1_normalization.py defaults: SB yield [20k, 200k], SR < 800k, FvT/noFvT
        # [0.25, 2.5]) are sample-size dependent: Run 3 2022+2023 HH4b has ~207k SB multijet events, so
        # productions set `f1_validation: {max_sb_yield: ..., ...}` (keys = the script's options)
        bounds = lambda wildcards: " ".join(
            f"--{k.replace('_', '-')} {v}" for k, v in (config.get('f1_validation') or {}).items()
            if k in ('min_sb_yield', 'max_sb_yield', 'max_sr_yield', 'min_ratio', 'max_ratio')),
        # the script checks process `data` three-tag and FvT/noFvT ratios: FvT method only
        skip = "1" if BKG_MODEL == "mvd" else "",
    resources:
        mem_mb = 4000,
        cpus_per_task = 1,
        runtime = 15
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        if [ -n "{params.skip}" ]; then
            echo "bkg_model mvd: FvT-method normalization check skipped" | tee {log} > {output.done}
            exit 0
        fi
        {params.container_wrapper} {params.python_bin} coffea4bees/analysis/tools/validate_f1_normalization.py \
            -i {input.coffea} \
            -o {output.done} {params.bounds} 2>&1 | tee {log}
        """

# ── Closure Comparison & Multi-Subsample Overlay Plots ────────────────────────
rule make_plots_closure:
    input:
        data_coffea = f"{out_f1}histAll_mixeddata_bkgs.coffea",
        norm_valid = f"{out_f1}validation_normalization.done",
        mixed_coffea = expand(f"{out_a3}histAll_{channel}_mixeddata_v{{m}}.coffea", m=SUBSAMPLES),
        plot_cfg = closure_plot_cfg
    output:
        done = f"{out_f1}plots/plots_done.txt"
    log:
        f"{out_f1}logs/make_plots.log"
    params:
        output_dir = f"{out_f1}plots/",
        extra_arguments = " ".join(filter(None, [
            "-s xW",
            "--year " + ("Run3" if any("202" in y for y in YEARS) else "RunII"),
            config.get("plot_extra_arguments", ""),
        ])),
        channel = channel,
        n_models = N_SUBSAMPLES,
        container_wrapper = container_wrapper,
        python_bin = python_bin,
        # the closure plot configs name the FvT-method processes (data threeTag, TTbar4b_from_d3)
        skip = "1" if BKG_MODEL == "mvd" else "",
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_dir} $(dirname {log})
        if [ -n "{params.skip}" ]; then
            echo "bkg_model mvd: F_1 closure plots skipped (FvT-method plot configs); see the F_2 closure fits" | tee {log} > {output.done}
            exit 0
        fi
        {params.container_wrapper} {params.python_bin} coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            {params.extra_arguments} 2>&1 | tee {log}
        {params.container_wrapper} {params.python_bin} coffea4bees/plots/make_subsample_overlay_plots.py \
            -i {input.data_coffea} \
            -o {params.output_dir} \
            --channel {params.channel} \
            --n_subsamples {params.n_models} 2>&1 | tee -a {log}
        touch {output.done}
        """

# ── Summary HTML Gallery ──────────────────────────────────────────────────────
rule make_gallery_closure:
    input:
        plots_done = f"{out_f1}plots/plots_done.txt",
        plot_cfg = closure_plot_cfg
    output:
        gallery = f"{out_f1}plots/gallery.html"
    log:
        f"{out_f1}logs/make_gallery.log"
    params:
        output_dir = f"{out_f1}plots/",
        container_wrapper = container_wrapper,
        python_bin = python_bin
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.gallery}) $(dirname {log})
        if [ -f src/plotting/make_gallery.py ]; then
            {params.container_wrapper} {params.python_bin} src/plotting/make_gallery.py \
                {params.output_dir} -m {input.plot_cfg} --title "Stage_F_1_Mixed-Data_Background_Systematics" -o gallery.html 2>&1 | tee {log}
        else
            touch {output.gallery}
        fi
        """

# ── Histogram Consistency Check vs Stage C_4 ──────────────────────────────────
rule hist_crosscheck_bkg_syst_F_1:
    input:
        data_coffea = f"{out_f1}histAll_mixeddata_bkgs.coffea"
    output:
        html = f"{out_f1}hist_crosscheck_bkg_syst_F_1.html",
        txt = f"{out_f1}hist_crosscheck_bkg_syst_F_1.txt"
    log:
        f"{out_f1}logs/hist_crosscheck.log"
    params:
        reference = f"{out_c4}histAll_FvT_closure_v0.coffea",
        container_wrapper = container_wrapper,
        python_bin = python_bin,
        tolerance = "0.0001",
        variables = "selJets.n,v4j.mass",
        regions = "SB"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.html}) $(dirname {log})
        if [ ! -f src/tools/compare_coffea_hists.py ]; then
            echo "src/tools/compare_coffea_hists.py not found; skipping histogram cross-check" 2>&1 | tee {log}
            echo "<p>histogram cross-check not available</p>" > {output.html}
            cp {log} {output.txt}
        elif [ ! -f "{params.reference}" ]; then
            echo "SKIPPED: reference .coffea file '{params.reference}' not found" 2>&1 | tee {log}
            echo "<p>histogram cross-check skipped: reference file not found</p>" > {output.html}
            cp {log} {output.txt}
        else
            {params.container_wrapper} {params.python_bin} src/tools/compare_coffea_hists.py \
                {input.data_coffea} {params.reference} \
                -o {output.html} --txt {output.txt} --title "Stage_F_1_hist_crosscheck" \
                --label-a Stage_F_1 --label-b Stage_C_4 \
                --variables "{params.variables}" --regions "{params.regions}" \
                --tolerance {params.tolerance} 2>&1 | tee {log}
        fi
        """

