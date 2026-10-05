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
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

mix_name = config.get('mix_name', f"{channel}_bkg_syst")

# ── Stage F_1 Runtime Config Staging ──────────────────────────────────────────
from helpers.stage_configs import stage_phaseF_1_configs
cfg_files = stage_phaseF_1_configs(config, out_f1)

closure_plot_cfg = config.get(
    'closure_plot_config',
    "coffea4bees/plots/metadata/plots_bkg_syst_closure_ttHbb.yml"
)

wildcard_constraints:
    m = r"\d+"

localrules: all_bkg_syst_F_1, all_bkg_syst_F_1_hists, make_plots_closure, make_gallery_closure, stage_bkg_syst_friend_manifest, stage_bkg_syst_jcm, stage_bkg_syst_jcm_per_year

# ── Master Target Rule ────────────────────────────────────────────────────────
rule all_bkg_syst_F_1:
    input:
        f"{out_f1}plots/plots_done.txt",
        f"{out_f1}plots/gallery.html"

# ── Histogram Only Target Rule (Before Plotting) ──────────────────────────────
rule all_bkg_syst_F_1_hists:
    input:
        f"{out_f1}histAll_mixeddata_bkgs.coffea"

per_year_jcm = bool(config.get('per_year_jcm', (config.get('phaseB_1', {}) or {}).get('per_year_jcm', False)))

# Staging rules for cross-cluster execution under roast (fetching from EOS handoff if not local).
# Only when F_1 runs without the stage that makes the file (Snakefile_bkg_syst.smk includes B_1 and
# C before F: two rules for one output are ambiguous).
_defined_rules = {r.name for r in workflow.rules}
if HANDOFF_EOS and 'extract_friend_manifest' not in _defined_rules:
    rule stage_bkg_syst_friend_manifest:
        output:
            f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json"
        params:
            src = f"{HANDOFF_EOS}/friends/friends_FvT_{mix_name}_v{{m}}.json"
        shell:
            """
            mkdir -p $(dirname {output})
            if command -v xrdcp &>/dev/null; then
                xrdcp -f '{params.src}' '{output}'
            else
                cp -f '{params.src}' '{output}'
            fi
            """

if HANDOFF_EOS and 'make_subsample_jcm_b1' not in _defined_rules:
    rule stage_bkg_syst_jcm:
        output:
            f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}.yml"
        params:
            src = f"{HANDOFF_EOS}/JCM/jetCombinatoricModel_SB_mix_v{{m}}.yml"
        shell:
            """
            mkdir -p $(dirname {output})
            if command -v xrdcp &>/dev/null; then
                xrdcp -f '{params.src}' '{output}'
            else
                cp -f '{params.src}' '{output}'
            fi
            """

    rule stage_bkg_syst_jcm_per_year:
        output:
            f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml"
        params:
            src = f"{HANDOFF_EOS}/JCM/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml"
        shell:
            """
            mkdir -p $(dirname {output})
            if command -v xrdcp &>/dev/null; then
                xrdcp -f '{params.src}' '{output}'
            else
                cp -f '{params.src}' '{output}'
            fi
            """

def get_all_analysis_jcm_inputs(wildcards):
    if per_year_jcm:
        return [f"{out_b1}jetCombinatoricModel_SB_mix_v{m}_{y}.yml" for m in SUBSAMPLES for y in YEARS]
    return [f"{out_b1}jetCombinatoricModel_SB_mix_v{m}.yml" for m in SUBSAMPLES]

# ── Data Background Model Processor (Single Pass over Collision Data) ──────────
rule analysis_data_closure:
    input:
        cfg = cfg_files['config'],
        jcm = get_all_analysis_jcm_inputs,
        fvt = expand(f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json", m=SUBSAMPLES),
    output:
        coffea = f"{out_f1}histAll_mixeddata_bkgs.coffea"
    log:
        f"{out_f1}logs/analysis_data.log"
    params:
        processor = config.get('analysis_processor') or f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = "data",
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

# ── Closure Comparison & Multi-Subsample Overlay Plots ────────────────────────
rule make_plots_closure:
    input:
        data_coffea = f"{out_f1}histAll_mixeddata_bkgs.coffea",
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
        python_bin = python_bin
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_dir} $(dirname {log})
        {params.container_wrapper} {params.python_bin} coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            -p 4 \
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
