import os

include: "rules/run3_variants.smk"

config.setdefault('datasets', ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data'])
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

if config["mode"] == "nominal":
    config.setdefault('output_path', "output/Run3/")
    config.setdefault('jcm_install_path', "coffea4bees/analysis/weights/JCM/Run3/jetCombinatoricModel_SB_.yml")
    # FvT training reuses the existing committed Run3 classifier inputs JSON
    config.setdefault('classifier_inputs_install_path', "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_Run3.json")

elif config["mode"] == "quadjet_run2":
    config.setdefault('output_path', "output/Run3_quadjet_run2/")
    config.setdefault('jcm_install_path', "coffea4bees/analysis/weights/JCM/Run3/jetCombinatoricModel_SB_quadjet_run2.yml")
    # FvT training for quadjet_run2 reuses the MvD-produced classifier inputs JSON
    config.setdefault('classifier_inputs_install_path', "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3_quadjet_run2.json")

out = config['output_path']

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

config.setdefault('jcm_config', "coffea4bees/analysis/jcm_tools/metadata/nominal_jcm_config.yml")
config.setdefault('plots_metadata', "coffea4bees/plots/metadata/plotsAll.yml")

rule all:
    input:
        f"{out}histAll_Run3{config['label']}.coffea",
        f"{out}jcm/jetCombinatoricModel_SB_.yml",
        f"{out}plots_wJCM/plots_done.txt"

rule all_histograms:
    """Remake all histogram merges and plots. Does not require JCM refit."""
    input:
        f"{out}histAll_Run3{config['label']}.coffea",
        f"{out}histAll_wJCM{config['label']}.coffea",
        f"{out}plots_wJCM/plots_done.txt"

rule all_with_training:
    input:
        rules.all.input,
        expand(f"{out}{{classifier}}/evaluate.done", classifier=["FvT"]),
        expand(f"{out}{{classifier}}/analyze.done",  classifier=["FvT"]),
        f"{out}plots_FvT/plots_done.txt"

rule all_FvT_histograms:
    """Remake FvT-weighted histograms/plots. Requires evaluate.done."""
    input:
        f"{out}histAll_FvT{config['label']}.coffea",
        f"{out}plots_FvT/plots_done.txt",

# ── Histograms ────────────────────────────────────────────────────────────────
# Use __ (double underscore) as separator between dataset and year to avoid
# ambiguous wildcard matching, since both dataset names and years contain _.
# SvB friend trees are reused from the MvD workflow output.

# SvB_MA friend tree pointer. Defaults to the Run3-trained SvB produced by
# Snakefile_Run3_SvB_training.smk on this same EOS base (mtime 2026-05-12).
# For A/B testing against the legacy Run2-trained SvB_MA scores, override:
#   --config svb_ma_path='coffea4bees/metadata/datasets_HH4b_Run3/SvBfriend_mixeddata_data.json@@SvB_MA'
config.setdefault('svb_ma_path',
    f"{config['eos_base']}/friend/SvB/result.json@@analysis.0.merged")

rule create_friends_wSvB:
    input:
        friends_yml    = "coffea4bees/metadata/friends_HH4b.yml",
        svb_json       = "coffea4bees/metadata/datasets_HH4b_Run3/SvBfriend_mixeddata_data.json",
        feynet_json    = "coffea4bees/metadata/datasets_HH4b_Run3/SvBFeynNetfriend_mixeddata_data.json",
    output: f"{out}friends_wSvB.yml"
    params:
        svb_ma_path = config['svb_ma_path']
    shell:
        """
        sed \
            -e 's|    SvB:.*|    SvB: "{input.svb_json}@@SvB"|' \
            -e 's|    SvB_MA:.*|    SvB_MA: {params.svb_ma_path}|' \
            -e 's|    SvB_FeynNet:.*|    SvB_FeynNet: "{input.feynet_json}@@SvB_FeynNet"|' \
            {input.friends_yml} > {output}
        echo "Patched friends:"
        grep -E "SvB" {output}
        """

rule create_histogram_config_wSvB:
    input:
        config_file = config['histogram_config']
    output: f"{out}histogram_config_wSvB.yml"
    params:
        # Pass --config enable_SvB_FeynNet_comparison=true to also fill the
        # 2D/3D SvB_MA vs SvB_FeynNet comparison histograms.
        svb_fn_cmp = str(config.get('enable_SvB_FeynNet_comparison', False)).lower(),
        hemi_diag  = str(config.get('compute_hemi_mixing_diagnostics', False)).lower()
    shell:
        """
        sed \
            -e 's|  run_SvB:.*|  run_SvB: true|' \
            -e 's|  run_SvB_FeynNet_comparison:.*|  run_SvB_FeynNet_comparison: {params.svb_fn_cmp}|' \
            -e 's|  compute_hemi_mixing_diagnostics:.*|  compute_hemi_mixing_diagnostics: {params.hemi_diag}|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "run_SvB|compute_hemi_mixing_diagnostics" {output}
        """

use rule analysis_processor from analysis as make_histograms with:
    input:
        config_file  = f"{out}histogram_config_wSvB.yml",
        friends_file = f"{out}friends_wSvB.yml",
    output: f"{out}histograms/hist_{{dataset}}__{{year}}.coffea"
    log: f"{out}logs/hist_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = "{year}",
        config = lambda wildcards, input: input.config_file,
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = lambda wildcards, input: input.friends_file,
        run_on_condor = config['run_on_condor'],
        extra_arguments = "",
        run_container_wrapper = "./run_container",
        dashboard_address = 0

use rule merging_coffea_files from analysis as merge_histograms with:
    input:
        expand(
            "{out}histograms/hist_{dataset}__{year}.coffea",
            out=out,
            dataset=config['datasets'],
            year=config['years']
        )
    output: f"{out}histAll_Run3{config['label']}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False
    log: f"{out}logs/merge_histograms.log"

# ── JCM fitting ───────────────────────────────────────────────────────────────

rule make_JCM_Run3:
    input: ancient(f"{out}histAll_Run3{config['label']}.coffea")
    output: f"{out}jcm/jetCombinatoricModel_SB_.yml"
    container: config['analysis_container']
    params:
        output_dir = f"{out}jcm/",
        jcm_config = config['jcm_config'],
        extra_arguments = "",
    log: f"{out}logs/make_JCM_Run3.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        echo "Computing JCM for Run3 nominal" 2>&1 | tee -a {log}
        python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
            -o {params.output_dir} \
            -i {input} \
            -r SB \
            --jcm_config {params.jcm_config} \
            {params.extra_arguments} 2>&1 | tee -a {log}
        ls {params.output_dir} 2>&1 | tee -a {log}
        """

# ── Histograms with fitted JCM (re-run all data + TT) ─────────────────────────
# JCM scales threeTag → fourTag-like, so all samples with threeTag content
# (data, TT) must be re-run with the new JCM applied.

rule create_histogram_config_wJCM:
    input:
        jcm_file = f"{out}jcm/jetCombinatoricModel_SB_.yml",
        config_file = config['histogram_config']
    output: f"{out}histogram_config_wJCM.yml"
    params:
        svb_fn_cmp = str(config.get('enable_SvB_FeynNet_comparison', False)).lower(),
        hemi_diag  = str(config.get('compute_hemi_mixing_diagnostics', False)).lower()
    shell:
        """
        sed \
            -e 's|  run_SvB:.*|  run_SvB: true|' \
            -e 's|  run_SvB_FeynNet_comparison:.*|  run_SvB_FeynNet_comparison: {params.svb_fn_cmp}|' \
            -e 's|  JCM_file.*|  JCM_file: {input.jcm_file}|' \
            -e 's|  compute_hemi_mixing_diagnostics:.*|  compute_hemi_mixing_diagnostics: {params.hemi_diag}|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "run_SvB|JCM_file|compute_hemi_mixing_diagnostics" {output}
        """

use rule analysis_processor from analysis as make_histograms_wJCM with:
    input:
        config_file  = f"{out}histogram_config_wJCM.yml",
        friends_file = f"{out}friends_wSvB.yml",
    output: f"{out}histograms_wJCM/hist_{{dataset}}__{{year}}.coffea"
    log: f"{out}logs/hist_wJCM_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = "{year}",
        config = lambda wildcards, input: input.config_file,
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = lambda wildcards, input: input.friends_file,
        run_on_condor = config['run_on_condor'],
        extra_arguments = "",
        run_container_wrapper = "./run_container",
        dashboard_address = 0

use rule merging_coffea_files from analysis as merge_histograms_wJCM with:
    input:
        expand(
            "{out}histograms_wJCM/hist_{dataset}__{year}.coffea",
            out=out,
            dataset=config['datasets'],
            year=config['years']
        )
    output: f"{out}histAll_wJCM{config['label']}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False
    log: f"{out}logs/merge_histograms_wJCM.log"

rule make_plots_wJCM:
    input: f"{out}histAll_wJCM{config['label']}.coffea"
    output: f"{out}plots_wJCM/plots_done.txt"
    container: config['analysis_container']
    params:
        output_dir = f"{out}plots_wJCM/",
        metadata = config['plots_metadata'],
        extra_arguments = "",
    log: f"{out}logs/make_plots_wJCM.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        echo "Making plots with fitted JCM" 2>&1 | tee -a {log}
        python coffea4bees/plots/makePlots.py {input} -o {params.output_dir} -m {params.metadata} {params.extra_arguments} 2>&1 | tee -a {log}
        python src/plotting/pb_pdf_to_png.py -r -j 4 {params.output_dir} 2>&1 | tee -a {log}
        touch {output}
        """

# ── Histograms with FvT weights (depends on evaluate) ────────────────────────
# Re-runs all data + TT samples with apply_FvT=true. The FvT friend tree
# produced by classifier evaluate (on EOS) reweights threeTag events into
# the 4b-like multijet background.

rule create_friends_FvT:
    input:
        friends_yml = "coffea4bees/metadata/friends_HH4b.yml",
        svb_json    = "coffea4bees/metadata/datasets_HH4b_Run3/SvBfriend_mixeddata_data.json",
        feynet_json = "coffea4bees/metadata/datasets_HH4b_Run3/SvBFeynNetfriend_mixeddata_data.json",
    output: f"{out}friends_FvT.yml"
    params:
        fvt_path    = f"{config['eos_base']}/friend/FvT/result.json@@analysis.0.merged",
        svb_ma_path = config['svb_ma_path']
    shell:
        """
        sed \
            -e 's|    FvT:.*|    FvT: {params.fvt_path}|' \
            -e 's|    SvB:.*|    SvB: "{input.svb_json}@@SvB"|' \
            -e 's|    SvB_MA:.*|    SvB_MA: {params.svb_ma_path}|' \
            -e 's|    SvB_FeynNet:.*|    SvB_FeynNet: "{input.feynet_json}@@SvB_FeynNet"|' \
            {input.friends_yml} > {output}
        echo "Patched friends:"
        grep -E "FvT|SvB" {output}
        """

rule create_histogram_config_FvT:
    input:
        jcm_file    = config['jcm_install_path'],
        config_file = config['histogram_config']
    output: f"{out}histogram_config_FvT.yml"
    params:
        svb_fn_cmp = str(config.get('enable_SvB_FeynNet_comparison', False)).lower(),
        hemi_diag  = str(config.get('compute_hemi_mixing_diagnostics', False)).lower()
    shell:
        """
        sed \
            -e 's|  run_SvB:.*|  run_SvB: true|' \
            -e 's|  run_SvB_FeynNet_comparison:.*|  run_SvB_FeynNet_comparison: {params.svb_fn_cmp}|' \
            -e 's|  JCM_file.*|  JCM_file: {input.jcm_file}|' \
            -e 's|  apply_FvT:.*|  apply_FvT: true|' \
            -e 's|  plot_ttbar_with_weights.*|  plot_ttbar_with_weights: true|' \
            -e 's|  compute_hemi_mixing_diagnostics:.*|  compute_hemi_mixing_diagnostics: {params.hemi_diag}|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "run_SvB|JCM_file|apply_FvT|plot_ttbar_with_weights|compute_hemi_mixing_diagnostics" {output}
        """

# Only data is re-run with FvT — TT estimate comes from 3b data × FvT.d3_to_t4
# (plot_ttbar_with_weights=true), so re-running TT MC samples is redundant.
use rule analysis_processor from analysis as make_histograms_FvT with:
    input:
        config_file   = f"{out}histogram_config_FvT.yml",
        friends_file  = f"{out}friends_FvT.yml",
        evaluate_done = ancient(expand(f"{out}{{classifier}}/evaluate.done", classifier=["FvT"])),
    output: f"{out}histograms_FvT/hist_data__{{year}}.coffea"
    log: f"{out}logs/hist_FvT_data__{{year}}.log"
    params:
        datasets              = "data",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = lambda wildcards, input: input.friends_file,
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

use rule merging_coffea_files from analysis as merge_histograms_FvT with:
    input:
        expand(
            "{out}histograms_FvT/hist_data__{year}.coffea",
            out=out,
            year=config['years']
        )
    output: f"{out}histAll_FvT{config['label']}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False
    log: f"{out}logs/merge_histograms_FvT.log"

rule make_plots_FvT:
    input: f"{out}histAll_FvT{config['label']}.coffea"
    output: f"{out}plots_FvT/plots_done.txt"
    container: config['analysis_container']
    params:
        output_dir      = f"{out}plots_FvT/",
        metadata        = config['plots_metadata'],
        extra_arguments = "",
    log: f"{out}logs/make_plots_FvT.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        echo "Making plots with FvT weights" 2>&1 | tee -a {log}
        python coffea4bees/plots/makePlots.py {input} -o {params.output_dir} -m {params.metadata} {params.extra_arguments} 2>&1 | tee -a {log}
        python src/plotting/pb_pdf_to_png.py -r -j 4 {params.output_dir} 2>&1 | tee -a {log}
        touch {output}
        """

# ── Classifier training ────────────────────────────────────────────────────────
# Install the fitted JCM to the path expected by train.yml. Commit the file
# to git after running to version the inputs.

rule install_JCM:
    input: f"{out}jcm/jetCombinatoricModel_SB_.yml"
    output: config['jcm_install_path']
    shell:
        """
        mkdir -p $(dirname {output})
        cp {input} {output}
        echo "Installed JCM to {output} — commit this file to git to version it."
        """

module training:
    snakefile: "Snakefile_Run3_training.smk"
    config: config

use rule create_train_yml    from training
use rule create_evaluate_yml from training
use rule train               from training
use rule analyze             from training
use rule evaluate            from training
