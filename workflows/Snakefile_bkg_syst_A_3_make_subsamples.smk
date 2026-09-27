# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_3_make_subsamples.smk
#
# Stage A_3: Mixed-Data Subsample Slicing and Multi-Sample Registry Assembly
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Slices the nominal mixed data (produced in Stage A_1) into 15 statistically
# independent pseudo-experiments (v0..v14) using the calibrated mixed-data JCM weights,
# and compiles the multi-sample dataset registry (mixeddata_4b.yml) incorporating
# both the mixed-data subsamples and the sliced ttbar pseudodata (from Stage A_2).
#
# WORKFLOW EXECUTION PIPELINE:
#   1. Slicing Subsamples (MixedDataSplitter):
#      - Runs `runner.py` with `split_mixed_data.py` across 15 pseudo-experiments
#        (v0..v14), partitioning events deterministically by event number and
#        applying the inclusive mixed-data JCM model.
#   2. Build Multi-Sample Registry:
#      - Combines per-subsample picoAOD registries and ttbar_PSData_stitched.yml
#        into a unified multi-sample dataset manifest:
#        output/ttHbb_bkg_syst/coffea4bees/metadata/datasets/mixeddata_4b.yml
#   3. Subsample Correlation & Orthogonality Study:
#      - Runs `processor_study_mixed_data.py` to evaluate event overlap across
#        subsamples and generates the correlation matrix plot.
#
# INPUTS:
#   - Nominal mixed data dataset: coffea4bees/metadata/datasets/mixeddata_...rank0_0.yml (From A_1)
#   - Inclusive mixed data JCM: {out_a1}JCM_2_mixeddata_inclusive/...yml (From A_1)
#   - Stitched ttbar pseudodata dataset: {out}coffea4bees/metadata/datasets/ttbar_PSData_stitched.yml (From A_2)
#
# OUTPUTS:
#   - {out_a3}subsamples/per_subsample/picoaod_datasets_mix_v*.yml
#   - {out}coffea4bees/metadata/datasets/mixeddata_4b.yml (Multi-sample registry)
#   - {out_a3}plots_study_mixeddata/subsample_correlation_matrix.png
#
# USAGE:
#   snakemake -s Snakefile_bkg_syst_A_3_make_subsamples.smk \
#             --configfile config/analysis_ttHbb_bkg_syst.yml -np all_bkg_syst_A_3
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

sub_out = config['subsample_output_path']
mixeddata_jcm_file = f"{out_a1}JCM_2_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml"
psdata_manifest_file = config.get('ttbar_psdata_manifest', f"{out}coffea4bees/metadata/datasets/ttbar_PSData_stitched.yml")

# ── Stage A_3 Runtime Config Staging (Generated into {out_a3}configs/) ────────
from helpers.stage_configs import stage_phaseA_3_configs
cfg_files = stage_phaseA_3_configs(config, out_a3, mixeddata_jcm_file)

localrules: all_bkg_syst_A_3, all_bkg_syst_A_3_alias, all_subsamples, all_study_mixeddata, build_multisample_registry, plot_subsample_correlation

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_bkg_syst_A_3:
    input:
        config['multisample_install_path']

rule all_bkg_syst_A_3_alias:
    input:
        rules.all_bkg_syst_A_3.input

rule all_subsamples:
    input:
        config['multisample_install_path']

rule all_study_mixeddata:
    input:
        f"{out_a3}study_mixeddata_all{_rank_suffix}.coffea",
        f"{out_a3}plots_study_mixeddata/subsample_correlation_matrix.png",

# ── Stage 1: Subsampling Mixed Data (15 Subsamples v0..v14) ────────────────────
rule run_split_mixeddata_per_subsample:
    input:
        ds_ready = get_mixeddata_dataset_file,
        cfg = cfg_files['skimmer_split'],
        jcm_ready = mixeddata_jcm_file,
    output:
        reg = f"{sub_out}picoaod_datasets_mix_v{{v}}.yml",
        done = f"{sub_out}.subsample_v{{v}}.done",
    log:
        f"{sub_out}logs/split_mixeddata_v{{v}}.log"
    params:
        processor = "coffea4bees/skimmer/processor/split_mixed_data.py",
        dataset = config.get('mixeddata', {}).get('dataset_name', config.get('dataset_name', 'mixeddata_ttHbb_bkg_syst_rank0_0')),
        output_path = sub_out,
        base_path = lambda wildcards: f"{config['base_path']}/subsamples/v{wildcards.v}",
        years = " ".join(YEARS),
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.reg}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            -p {params.processor} \
            --metadata {input.ds_ready} \
            -d {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.reg}) \
            --config-overrides mixed_subsample={wildcards.v} base_path={params.base_path} \
            -s {params.condor_flags} 2>&1 | tee {log}
        touch {output.done}
        """

# ── Stage 2: Multi-Sample Registry Generation (incorporating ttbar PSData) ────
rule build_multisample_registry:
    input:
        dones = expand(f"{sub_out}.subsample_v{{v}}.done", v=SUBSAMPLES),
        registries = expand(f"{sub_out}picoaod_datasets_mix_v{{v}}.yml", v=SUBSAMPLES),
        psdata = get_ttbar_psdata_dataset_file,
    output:
        config['multisample_install_path']
    params:
        dataset_name = config['multisample_dataset_name'],
        n_samples = N_SUBSAMPLES,
        years = YEARS,
        psdata_manifest = get_ttbar_psdata_dataset_file,
        python_bin = python_bin,
    log:
        f"{out_a3}logs/build_multisample_registry.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        ps_arg=""
        if [ -n "{params.psdata_manifest}" ] && [ -f "{params.psdata_manifest}" ]; then
            ps_arg="--psdata-manifest {params.psdata_manifest}"
        fi
        {params.python_bin} coffea4bees/hemisphere_mixing/build_multisample_registry.py \
            --registries {input.registries} \
            --output {output} \
            --dataset-name {params.dataset_name} \
            --n-samples {params.n_samples} \
            --years {params.years} \
            $ps_arg 2>&1 | tee {log}
        """

# ── Stage 3: Subsample Correlation & Orthogonality Study ──────────────────────
rule study_mixeddata:
    input:
        ds_ready = get_mixeddata_dataset_file,
        jcm_ready = mixeddata_jcm_file,
        study_cfg = cfg_files['study'],
    output:
        coffea_out = f"{out_a3}study_mixeddata_all{_rank_suffix}.coffea",
    log:
        f"{out_a3}logs/study_mixeddata.log"
    params:
        processor = "coffea4bees/analysis/processors/processor_study_mixed_data.py",
        dataset = config.get('mixeddata', {}).get('dataset_name', config.get('dataset_name', 'mixeddata_ttHbb_bkg_syst_rank0_0')),
        output_path = out_a3,
        years = " ".join(YEARS),
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.study_cfg} \
            --processor {params.processor} \
            --metadata {input.ds_ready} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

rule plot_subsample_correlation:
    input:
        coffea = f"{out_a3}study_mixeddata_all{_rank_suffix}.coffea",
    output:
        plot = f"{out_a3}plots_study_mixeddata/subsample_correlation_matrix.png",
    log:
        f"{out_a3}plots_study_mixeddata/logs/plot_subsample_correlation.log"
    params:
        out_dir = f"{out_a3}plots_study_mixeddata/",
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p {params.out_dir} $(dirname {log})
        {params.container_wrapper} {params.python_bin} scripts/plot_subsample_correlation.py \
            -i {input.coffea} \
            -o {params.out_dir} 2>&1 | tee {log}
        """
