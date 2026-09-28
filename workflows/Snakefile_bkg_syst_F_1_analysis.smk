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
#      - Symlinks Stage A_4 unweighted 4-tag mixed data histograms:
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
#   - Stage A_4 Mixed Data: output/.../bkg_syst_A_4_process_subsamples/histAll_ttHbb_mixeddata_v{m}.coffea
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

localrules: all_bkg_syst_F_1, all_bkg_syst_F_1_hists, link_mixeddata_closure, make_plots_closure, make_gallery_closure, closure_v

# ── Master Target Rule ────────────────────────────────────────────────────────
rule all_bkg_syst_F_1:
    input:
        expand(f"{out_f1}closure_v{{m}}/plots/plots_done.txt", m=SUBSAMPLES),
        expand(f"{out_f1}closure_v{{m}}/plots/gallery.html", m=SUBSAMPLES)

# ── Histogram Only Target Rule (Before Plotting) ──────────────────────────────
rule all_bkg_syst_F_1_hists:
    input:
        expand(f"{out_f1}closure_v{{m}}/histAll_data_v{{m}}.coffea", m=SUBSAMPLES),
        expand(f"{out_f1}closure_v{{m}}/histAll_mixeddata_v{{m}}.coffea", m=SUBSAMPLES)

# ── Convenience Single-Subsample Rule ─────────────────────────────────────────
rule closure_v:
    input:
        f"{out_f1}closure_v{{m}}/plots/gallery.html"

# ── Data Background Model Processor ───────────────────────────────────────────
rule analysis_data_closure:
    input:
        cfg = lambda wildcards: cfg_files['data_configs'][int(wildcards.m)],
        jcm = f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}.yml",
        fvt = f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json",
    output:
        coffea = f"{out_f1}closure_v{{m}}/histAll_data_v{{m}}.coffea"
    log:
        f"{out_f1}closure_v{{m}}/logs/analysis_data.log"
    params:
        processor = config.get('analysis_processor') or f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = "data",
        years = " ".join(YEARS),
        output_path = f"{out_f1}closure_v{{m}}/",
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

# ── Link Stage A_4 Mixed Data Histograms ──────────────────────────────────────
rule link_mixeddata_closure:
    input:
        f"{out_a4}histAll_{channel}_mixeddata_v{{m}}.coffea"
    output:
        f"{out_f1}closure_v{{m}}/histAll_mixeddata_v{{m}}.coffea"
    shell:
        """
        mkdir -p $(dirname {output})
        ln -sf $(realpath --relative-to=$(dirname {output}) {input}) {output}
        """

# ── Closure Comparison Plots (Mixed Data vs Background Model) ─────────────────
rule make_plots_closure:
    input:
        data_coffea = f"{out_f1}closure_v{{m}}/histAll_data_v{{m}}.coffea",
        mixed_coffea = f"{out_f1}closure_v{{m}}/histAll_mixeddata_v{{m}}.coffea",
        plot_cfg = closure_plot_cfg
    output:
        done = f"{out_f1}closure_v{{m}}/plots/plots_done.txt"
    log:
        f"{out_f1}closure_v{{m}}/logs/make_plots.log"
    params:
        output_dir = f"{out_f1}closure_v{{m}}/plots/",
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW",
            "--year " + ("Run3" if any("202" in y for y in YEARS) else "RunII"),
            config.get("plot_extra_arguments", ""),
        ])),
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
        touch {output.done}
        """

# ── Summary HTML Gallery ──────────────────────────────────────────────────────
rule make_gallery_closure:
    input:
        plots_done = f"{out_f1}closure_v{{m}}/plots/plots_done.txt",
        plot_cfg = closure_plot_cfg
    output:
        gallery = f"{out_f1}closure_v{{m}}/plots/gallery.html"
    log:
        f"{out_f1}closure_v{{m}}/logs/make_gallery.log"
    params:
        output_dir = f"{out_f1}closure_v{{m}}/plots/",
        container_wrapper = container_wrapper,
        python_bin = python_bin
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.gallery}) $(dirname {log})
        if [ -f src/plotting/make_gallery.py ]; then
            {params.container_wrapper} {params.python_bin} src/plotting/make_gallery.py                 {params.output_dir} -m {input.plot_cfg} --title Closure_v{wildcards.m} -o gallery.html 2>&1 | tee {log}
        else
            touch {output.gallery}
        fi
        """
