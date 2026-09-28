# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_B_1_computeJCM.smk
#
# Stage B_1: Dedicated Jet Combinatoric Model (JCM) Calibration per Subsample
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Derives 16 dedicated Jet Combinatoric Model (JCM) transfer functions (v0..v15)
# for the ttH(bb) background systematics evaluation.
#
# In the multijet background estimation framework, 3-tag events are reweighted
# into the 4-tag signal region using combinatoric pseudo-tagging probabilities.
# Because each of the 16 pseudo-experiments (mixed data subsamples v0..v14 and
# the nominal dataset) represents an independent statistical realization, a
# dedicated JCM must be fitted for each individual subsample to avoid cross-sample
# leakage and preserve true statistical independence during FvT training.
#
# MATHEMATICAL FORMULATION:
# The JCM parameterizes the probability of promoting untagged jets to b-tags:
#   P(pseudo-tag | n_untagged) = pseudoTagProb * [1 + pairEnhancement * (decay_factor)]
# The transfer function is fitted in the Sideband (SB) region to match:
#   Target: [Data (4b) - ttbar MC (4b)]
#   Model:  Mixed Data (3b / unweighted 4b) * JCM(n_untagged, event_number)
#
# WORKFLOW EXECUTION PIPELINE:
#   1. Input Preparation:
#      - Reads baseline data & stitched ttbar MC histograms: inputs/histAll_NoJCM.coffea
#      - Reads unweighted mixed data histograms for subsample v{m}:
#        output/ttHbb_bkg_syst/bkg_syst_A_4_process_subsamples/histAll_ttHbb_mixeddata_v{m}.coffea
#   2. JCM Parameter Fit (`make_jcm_weights.py`):
#      - Executes `make_jcm_weights.py` per subsample with `--data4bName mix_v{m}`
#        in the Sideband (SB) region, floating the background scale.
#      - Derives:
#        * pseudoTagProb (baseline pseudo-tag probability)
#        * pairEnhancement & pairEnhancementDecay (correlation parameters)
#        * tt4bSF (scale factor for ttbar 4b contribution)
#   3. Model Export:
#      - Writes YAML and TXT parameter tables:
#        `jetCombinatoricModel_SB_mix_v{m}.yml`
#      - Generates validation tables (`JCM_validation_SB_mix_v{m}.txt`) and
#        yield projection reports (`JCM_expected_yields_SB_mix_v{m}.txt`).
#   4. Single-Subsample Verification & Closure:
#      - Runs `processor_ttHbb.py` on subsample v0 using its dedicated JCM.
#      - Executes `makePlots.py` to produce data-vs-model closure distributions
#        in the SR and SB regions (`plots_subsample_v0_closure/`).
#
# INPUTS:
#   - Baseline Data/MC Coffea: output/ttHbb_bkg_syst/inputs/histAll_NoJCM.coffea
#   - Subsample Coffea: output/ttHbb_bkg_syst/bkg_syst_A_4_process_subsamples/histAll_ttHbb_mixeddata_v{m}.coffea
#   - Fit Configuration: coffea4bees/analysis/jcm_tools/metadata/ttHbb_subsample_jcm_config.yml
#   - Plotting Metadata: coffea4bees/plots/metadata/plots_JCM_ttHbb.yml
#
# OUTPUTS:
#   - Dedicated JCM YAMLs: output/ttHbb_bkg_syst/bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}.yml
#   - Diagnostic Plots: output/ttHbb_bkg_syst/bkg_syst_B_1_computeJCM/plots_v{m}/selJets_noJCM_n.png
#   - Validation Closure: output/ttHbb_bkg_syst/bkg_syst_B_1_computeJCM/test_v0_closure/plots_subsample_v0_closure/
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc (CPU only, using `./run_container`)
#   - Batch Scheduler: HTCondor / Dask (--cores 4)
# ==============================================================================

import os
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# ── Stage B_1 Static Configs & Resolution ─────────────────────────────────────
phaseB_1 = config.get('phaseB_1', {})
jcm_section = phaseB_1.get('jcm', {})
val_section = phaseB_1.get('validation', {})

fit_cfg_path = jcm_section.get('config', "coffea4bees/analysis/jcm_tools/metadata/ttHbb_subsample_jcm_config.yml")
jcm_plot_cfg_path = jcm_section.get('plot_config', "coffea4bees/plots/metadata/plots_JCM_ttHbb.yml")
jcm_region = jcm_section.get('region', "SB")
jcm_year = jcm_section.get('year', "RunII")

val_subsample = val_section.get('subsample', 0)
val_proc_cfg = val_section.get('processor_config', "coffea4bees/workflows/config/analysis_config_bkg_syst_test_subsample.yml")
val_plot_cfg = val_section.get('plot_config', "coffea4bees/plots/metadata/plots_subsample_v0_closure_ttHbb.yml")

DATA_NOJCM_INPUT = jcm_section.get('data_coffea', config.get('data_nojcm_coffea', "inputs/histAll_NoJCM.coffea"))
if not DATA_NOJCM_INPUT.startswith("/") and not DATA_NOJCM_INPUT.startswith("output/"):
    jcm_input_coffea = os.path.join(out, DATA_NOJCM_INPUT)
else:
    jcm_input_coffea = DATA_NOJCM_INPUT

localrules: all_bkg_syst_B_1, prepare_data_noJCM_b1, make_subsample_jcm_b1, test_plots_subsample_closure, test_v0_jcm_closure, all_test_subsample_closure

rule all_bkg_syst_B_1:
    input:
        expand(f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}.yml", m=range(N_SUBSAMPLES)),
        expand(f"{out_b1}plots_v{{m}}/selJets_noJCM_n.png", m=range(N_SUBSAMPLES)),
        expand(f"{out_b1}plots_v{{m}}/tagJets_noJCM_n.png", m=range(N_SUBSAMPLES)),
        f"{out_b1}test_v{val_subsample}_closure/plots/plots_done.txt"

rule prepare_data_noJCM_b1:
    input:
        lambda wildcards: DATA_NOJCM_INPUT if os.path.exists(DATA_NOJCM_INPUT) else jcm_input_coffea
    output:
        f"{out_b1}histAll_NoJCM_data.coffea"
    shell:
        """
        mkdir -p $(dirname {output})
        if [ "{input}" != "{output}" ]; then
            ln -sf $(realpath --relative-to=$(dirname {output}) {input}) {output}
        fi
        """

def get_subsample_jcm_inputs_b1(wildcards):
    subsample_coffea = f"{out_a4}histAll_{channel}_mixeddata_v{wildcards.m}.coffea"
    data_coffea = (
        config.get('dummy_jcm_file', "coffea4bees/metadata/weights/JCM/jetCombinatoricModel_SB_dummy.yml")
        if config.get('test', False)
        else f"{out_b1}histAll_NoJCM_data.coffea"
    )
    return {
        "data_coffea": data_coffea,
        "subsample_coffea": subsample_coffea,
        "fit_cfg": fit_cfg_path,
        "plot_cfg": jcm_plot_cfg_path,
    }

rule make_subsample_jcm_b1:
    input:
        unpack(get_subsample_jcm_inputs_b1)
    output:
        jcm_yaml = f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}.yml",
        seljets_plot = f"{out_b1}plots_v{{m}}/selJets_noJCM_n.png",
        tagjets_plot = f"{out_b1}plots_v{{m}}/tagJets_noJCM_n.png",
    log:
        f"{out_b1}logs/make_jcm_v{{m}}.log"
    params:
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
        test_mode = config.get('test', False),
        dummy_jcm = config.get('dummy_jcm_file', "coffea4bees/metadata/weights/JCM/jetCombinatoricModel_SB_dummy.yml"),
        region = jcm_region,
        year = jcm_year,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.jcm_yaml})/plots_v{wildcards.m} $(dirname {log})
        if [ "{params.test_mode}" = "True" ] || [ "{params.test_mode}" = "true" ]; then
            echo "Test mode: deploying dummy JCM {params.dummy_jcm} -> {output.jcm_yaml}" > {log}
            cp {params.dummy_jcm} {output.jcm_yaml}
            touch {output.seljets_plot} {output.tagjets_plot}
        else
            {params.container_wrapper} {params.python_bin} coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
                -i {input.data_coffea} {input.subsample_coffea} \
                --jcm_config {input.fit_cfg} \
                -m {input.plot_cfg} \
                --combine_input_files \
                -w mix_v{wildcards.m} \
                --data4bName mix_v{wildcards.m} \
                -r {params.region} \
                -o $(dirname {output.jcm_yaml})/plots_v{wildcards.m}/ \
                --year {params.year} 2>&1 | tee {log}
            cp $(dirname {output.jcm_yaml})/plots_v{wildcards.m}/jetCombinatoricModel_{params.region}_mix_v{wildcards.m}.yml {output.jcm_yaml}
        fi
        """

# ── Verification: Processor Test on a Single Subsample with Calibrated JCM ────
rule test_processor_subsample_with_jcm:
    input:
        ds_file = get_multisample_dataset_file,
        jcm = f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}.yml",
        proc_cfg = val_proc_cfg,
    output:
        coffea = f"{out_b1}test_v{{m}}_closure/histAll_{channel}_mixeddata_v{{m}}_with_JCM.coffea",
    log:
        f"{out_b1}test_v{{m}}_closure/logs/test_processor_v{{m}}.log"
    params:
        processor = config.get('analysis_processor') or (config.get('analysis_config', {}) or {}).get('processor') or f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = f"{config['multisample_dataset_name']}:{{m}}",
        output_path = f"{out_b1}test_v{{m}}_closure/",
        years = " ".join(YEARS),
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.proc_cfg} \
            --processor {params.processor} \
            --metadata {input.ds_file} \
            --datasets {params.dataset} \
            --years {params.years} \
            --config-override apply_JCM=True JCM_file={input.jcm} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

rule test_plots_subsample_closure:
    input:
        data_coffea = f"{out_b1}histAll_NoJCM_data.coffea",
        subsample_coffea = f"{out_b1}test_v{{m}}_closure/histAll_{channel}_mixeddata_v{{m}}_with_JCM.coffea",
        plot_cfg = val_plot_cfg,
    output:
        done = f"{out_b1}test_v{{m}}_closure/plots/plots_done.txt",
    log:
        f"{out_b1}test_v{{m}}_closure/logs/make_plots_closure.log"
    params:
        output_dir = f"{out_b1}test_v{{m}}_closure/plots/",
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW",
            "--year " + ("Run3" if any("202" in y for y in YEARS) else "RunII"),
            config.get("plot_extra_arguments", ""),
        ])),
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_dir} $(dirname {log})
        echo "Making closure plots" 2>&1 | tee {log}
        {params.container_wrapper} {params.python_bin} coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.subsample_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            {params.extra_arguments} 2>&1 | tee -a {log}
        if [ -f src/plotting/make_gallery.py ]; then
            echo "Making gallery" 2>&1 | tee -a {log}
            {params.container_wrapper} {params.python_bin} src/plotting/make_gallery.py \
                {params.output_dir} -m {input.plot_cfg} --title "$(basename {params.output_dir})" 2>&1 | tee -a {log}
        fi
        touch {output.done}
        """

rule test_v0_jcm_closure:
    input:
        f"{out_b1}test_v0_closure/plots/plots_done.txt"

rule all_test_subsample_closure:
    input:
        expand(f"{out_b1}test_v{{m}}_closure/plots/plots_done.txt", m=range(N_SUBSAMPLES))
