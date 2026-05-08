import os

include: "rules/run3_variants.smk"

# Top-level mixed-data dataset name. Default 'mixeddata_all' preserves the
# legacy MvD workflow. Override (e.g. --config dataset_name=mixeddata_all_rank0)
# to point the whole pipeline at a different mixed-data sample. The matching
# dataset yaml must exist under coffea4bees/metadata/datasets_HH4b_Run3/, and
# the JCM yaml's data3bName field is sed-patched at fit time (see
# create_jcm_config rule below) so its committed value doesn't have to match.
config.setdefault('dataset_name', 'mixeddata_all')

config.setdefault('datasets', ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data', config['dataset_name']])
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

# Build a tag combining mode and dataset_name so install paths don't collide
# between (mode, rank) runs. Empty for legacy nominal + 'mixeddata_all' so
# historical install paths are preserved unchanged.
_mode_tag    = "" if config['mode'] == "nominal" else config['mode']
_dsn_tag     = "" if config['dataset_name'] == "mixeddata_all" \
               else config['dataset_name'].removeprefix("mixeddata_all_") or config['dataset_name']
_install_tag = "_".join(filter(None, [_mode_tag, _dsn_tag]))

# When running on a non-legacy dataset_name (e.g. mixeddata_all_rank0),
# default to reusing the legacy data/TT classifier inputs instead of
# regenerating them. data/TT inputs don't change with rank, so this saves
# 16 condor jobs per rank-suffixed run. Override with --config
# reuse_legacy_classifier_inputs=False to force a full regenerate.
if _dsn_tag:
    config.setdefault('reuse_legacy_classifier_inputs', True)

if config["mode"] == "nominal":
    _output_subdir = f"Run3_MvD{'_' + _dsn_tag if _dsn_tag else ''}"
elif config["mode"] == "quadjet_run2":
    _output_subdir = f"Run3_MvD_quadjet_run2{'_' + _dsn_tag if _dsn_tag else ''}"

config.setdefault('output_path', f"output/{_output_subdir}/")
config.setdefault('jcm_install_path',
    f"coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_{_install_tag}.yml")
config.setdefault('classifier_inputs_install_path',
    f"coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3"
    f"{'_' + _install_tag if _install_tag else ''}.json")

out = config['output_path']

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# Classifier-inputs rules now live in their own Snakefile.
# Module is declared here so its outputs are visible to `rule all`; the
# corresponding `use rule` statements are placed below `rule all` so the
# wildcard-bearing `classifier_inputs` rule is not picked as the default target.
module classifier_inputs:
    snakefile: "Snakefile_classifier_inputs_Run3.smk"
    config: config

config.setdefault('jcm_config', "coffea4bees/analysis/jcm_tools/metadata/mixeddata_all_config_Run3.yml")
config.setdefault('plots_metadata', "coffea4bees/plots/metadata/plotsAll_MvD.yml")

# SvB / SvB_FeynNet friend JSONs are mode-independent (one fit covers both
# modes) but rank-dependent — rank0 events have different UUIDs than rank1
# / legacy. Defaults pull from the rank-suffixed install paths produced by
# Snakefile_SvB_friendtrees_Run3.smk and Snakefile_SvBFeynNet_friendtrees_Run3.smk.
_svb_install_tag = f"_{_dsn_tag}" if _dsn_tag else ""
config.setdefault('svb_friend_json',
    f"coffea4bees/metadata/datasets_HH4b_Run3/SvBfriend_mixeddata_data{_svb_install_tag}.json")
config.setdefault('feynet_friend_json',
    f"coffea4bees/metadata/datasets_HH4b_Run3/SvBFeynNetfriend_mixeddata_data{_svb_install_tag}.json")

rule all:
    input:
        f"{out}histAll_Run3MvD{config['label']}.coffea",
        config['classifier_inputs_install_path'],
        f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml",
        f"{out}plots_wJCM/plots_done.txt"

rule all_histograms:
    """Remake all histogram merges and plots. Does not require JCM refit or classifier_inputs."""
    input:
        f"{out}histAll_Run3MvD{config['label']}.coffea",
        f"{out}histAll_mixeddata_wJCM{config['label']}.coffea",
        f"{out}plots_wJCM/plots_done.txt",
        f"{out}histAll_MvD{config['label']}.coffea",
        f"{out}plots_MvD/plots_done.txt",

rule all_MvD_histograms:
    """Remake MvD-weighted and wJCM histograms/plots. Does not require JCM refit or training."""
    input:
        f"{out}histAll_mixeddata_wJCM{config['label']}.coffea",
        f"{out}plots_wJCM/plots_done.txt",
        f"{out}histAll_MvD{config['label']}.coffea",
        f"{out}plots_MvD/plots_done.txt",

rule all_with_training:
    input:
        rules.all.input,
        expand(f"{out}{{classifier}}/evaluate.done", classifier=["MvD"]),
        expand(f"{out}{{classifier}}/analyze.done",  classifier=["MvD"]),
        f"{out}plots_MvD/plots_done.txt"

# Re-export classifier-inputs rules from the module after `rule all` so
# the wildcard `classifier_inputs` rule isn't picked as default target.
use rule classifier_inputs            from classifier_inputs
use rule merge_json_classifier_inputs from classifier_inputs
use rule install_classifier_inputs    from classifier_inputs

# ── Histograms ────────────────────────────────────────────────────────────────
# Use __ (double underscore) as separator between dataset and year to avoid
# ambiguous wildcard matching, since both dataset names and years contain _.

rule create_friends_wSvB:
    input:
        friends_yml    = "coffea4bees/metadata/friends_HH4b.yml",
        svb_json       = config['svb_friend_json'],
        feynet_json    = config['feynet_friend_json'],
    output: f"{out}friends_wSvB.yml"
    shell:
        """
        sed \
            -e 's|    SvB:.*|    SvB: "{input.svb_json}@@SvB"|' \
            -e 's|    SvB_MA:.*|    SvB_MA: "{input.svb_json}@@SvB_MA"|' \
            -e 's|    SvB_FeynNet:.*|    SvB_FeynNet: "{input.feynet_json}@@SvB_FeynNet"|' \
            {input.friends_yml} > {output}
        echo "Patched friends:"
        grep -E "SvB" {output}
        """

rule create_histogram_config_wSvB:
    input:
        config_file = config['histogram_config']
    output: f"{out}histogram_config_wSvB.yml"
    shell:
        """
        sed \
            -e 's|  run_SvB.*|  run_SvB: true|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "run_SvB" {output}
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
    output: f"{out}histAll_Run3MvD{config['label']}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False
    log: f"{out}logs/merge_histograms.log"

# ── JCM fitting ───────────────────────────────────────────────────────────────

rule create_jcm_config:
    """Patch the JCM config so data3bName matches our configured dataset_name.
    Committed yaml hardcodes 'mixeddata_all'; this rule lets us point the fit
    at any rank/variant without editing the source."""
    input:  config['jcm_config']
    output: f"{out}jcm_config_patched.yml"
    params:
        dataset_name = config['dataset_name']
    shell:
        """
        sed -e 's|data3bName:.*|data3bName: {params.dataset_name}|' \
            {input} > {output}
        echo "Patched JCM config:"
        grep -E "data3bName" {output}
        """

rule make_JCM_Run3MvD:
    input:
        hist       = ancient(f"{out}histAll_Run3MvD{config['label']}.coffea"),
        jcm_config = f"{out}jcm_config_patched.yml",
    output: f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml"
    container: config['analysis_container']
    params:
        output_dir = f"{out}jcm_for_mixed_all/",
        extra_arguments = "",
    log: f"{out}logs/make_JCM_Run3MvD.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        echo "Computing JCM for mixed_all Run3MvD" 2>&1 | tee -a {log}
        python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
            -o {params.output_dir} \
            -i {input.hist} \
            -r SB \
            --jcm_config {input.jcm_config} \
            {params.extra_arguments} 2>&1 | tee -a {log}
        ls {params.output_dir} 2>&1 | tee -a {log}
        """

# ── Histograms with fitted JCM (mixeddata_all only) ───────────────────────────

rule create_histogram_config_wJCM:
    input:
        jcm_file = f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml",
        config_file = config['histogram_config']
    output: f"{out}histogram_config_wJCM.yml"
    shell:
        """
        sed \
            -e 's|  run_SvB.*|  run_SvB: true|' \
            -e 's|  JCM_file.*|  JCM_file: {input.jcm_file}|' \
            -e 's|  apply_MvD_weight.*|  apply_MvD_weight: false|' \
            -e 's|  apply_MvD:[^_].*|  apply_MvD: true|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "run_SvB|JCM_file|apply_MvD" {output}
        """

use rule analysis_processor from analysis as make_histograms_mixeddata_wJCM with:
    input:
        config_file  = f"{out}histogram_config_wJCM.yml",
        friends_file = f"{out}friends_wSvB.yml",
    output: f"{out}histograms_wJCM/hist_{config['dataset_name']}__{{year}}.coffea"
    log: f"{out}logs/hist_wJCM_{config['dataset_name']}__{{year}}.log"
    params:
        datasets = config['dataset_name'],
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

use rule merging_coffea_files from analysis as merge_histograms_mixeddata_wJCM with:
    input:
        expand(
            "{out}histograms_wJCM/hist_{dataset_name}__{year}.coffea",
            out=out,
            dataset_name=config['dataset_name'],
            year=config['years']
        ),
        expand(
            "{out}histograms/hist_{dataset}__{year}.coffea",
            out=out,
            dataset=['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data'],
            year=config['years']
        )
    output: f"{out}histAll_mixeddata_wJCM{config['label']}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False
    log: f"{out}logs/merge_histograms_mixeddata_wJCM.log"

rule make_plots_wJCM:
    input: f"{out}histAll_mixeddata_wJCM{config['label']}.coffea"
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

# ── Classifier training ────────────────────────────────────────────────────────
# Install the fitted JCM to the path expected by train.yml. Commit both this
# file and the classifier inputs JSON (installed by the imported module) to git
# after running, to version the inputs.

rule install_JCM:
    input: f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml"
    output: config['jcm_install_path']
    shell:
        """
        mkdir -p $(dirname {output})
        cp {input} {output}
        echo "Installed JCM to {output} — commit this file to git to version it."
        """

# ── Histograms with MvD weights (depends on evaluate) ─────────────────────────
# Re-runs only data and mixeddata_all with apply_MvD=true, apply_MvD_weight=true.
# plot_ttbar_with_MvD_weights is safe to set globally — it is guarded by
# isMixedDataAll in the processor, so data processing is unaffected.
# TTbar histograms are reused from the nominal step (MvD weights don't apply).

rule create_friends_MvD:
    input:
        friends_yml = "coffea4bees/metadata/friends_HH4b.yml",
        svb_json    = config['svb_friend_json'],
        feynet_json = config['feynet_friend_json'],
    output: f"{out}friends_MvD.yml"
    params:
        mvd_path = f"{config['eos_base']}/friend/MvD/result.json@@analysis.0.merged"
    shell:
        """
        sed \
            -e 's|    MvD:.*|    MvD: {params.mvd_path}|' \
            -e 's|    SvB:.*|    SvB: "{input.svb_json}@@SvB"|' \
            -e 's|    SvB_MA:.*|    SvB_MA: "{input.svb_json}@@SvB_MA"|' \
            -e 's|    SvB_FeynNet:.*|    SvB_FeynNet: "{input.feynet_json}@@SvB_FeynNet"|' \
            {input.friends_yml} > {output}
        echo "Patched friends:"
        grep -E "MvD|SvB" {output}
        """

rule create_histogram_config_MvD:
    input:
        jcm_file    = config['jcm_install_path'],
        config_file = config['histogram_config']
    output: f"{out}histogram_config_MvD.yml"
    shell:
        """
        sed \
            -e 's|  run_SvB.*|  run_SvB: true|' \
            -e 's|  JCM_file.*|  JCM_file: {input.jcm_file}|' \
            -e 's|  apply_MvD_weight.*|  apply_MvD_weight: true\\n  plot_ttbar_with_MvD_weights: true|' \
            -e 's|  apply_MvD:[^_].*|  apply_MvD: true|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "run_SvB|JCM_file|apply_MvD|plot_ttbar_with_MvD" {output}
        """

use rule analysis_processor from analysis as make_histograms_data_MvD with:
    input:
        config_file   = f"{out}histogram_config_MvD.yml",
        friends_file  = f"{out}friends_MvD.yml",
        evaluate_done = ancient(expand(f"{out}{{classifier}}/evaluate.done", classifier=["MvD"])),
    output: f"{out}histograms_MvD/hist_data__{{year}}.coffea"
    log: f"{out}logs/hist_MvD_data__{{year}}.log"
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

use rule analysis_processor from analysis as make_histograms_mixeddata_MvD with:
    input:
        config_file   = f"{out}histogram_config_MvD.yml",
        friends_file  = f"{out}friends_MvD.yml",
        evaluate_done = ancient(expand(f"{out}{{classifier}}/evaluate.done", classifier=["MvD"])),
    output: f"{out}histograms_MvD/hist_{config['dataset_name']}__{{year}}.coffea"
    log: f"{out}logs/hist_MvD_{config['dataset_name']}__{{year}}.log"
    params:
        datasets              = config['dataset_name'],
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

use rule merging_coffea_files from analysis as merge_histograms_MvD with:
    input:
        expand(
            "{out}histograms_MvD/hist_data__{year}.coffea",
            out=out,
            year=config['years']
        ),
        expand(
            "{out}histograms_MvD/hist_{dataset_name}__{year}.coffea",
            out=out,
            dataset_name=config['dataset_name'],
            year=config['years']
        ),
        expand(
            "{out}histograms/hist_{dataset}__{year}.coffea",
            out=out,
            dataset=['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu'],
            year=config['years']
        )
    output: f"{out}histAll_MvD{config['label']}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False
    log: f"{out}logs/merge_histograms_MvD.log"

rule make_plots_MvD:
    input: f"{out}histAll_MvD{config['label']}.coffea"
    output: f"{out}plots_MvD/plots_done.txt"
    container: config['analysis_container']
    params:
        output_dir      = f"{out}plots_MvD/",
        metadata        = config['plots_metadata'],
        extra_arguments = "",
    log: f"{out}logs/make_plots_MvD.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        echo "Making plots with MvD weights" 2>&1 | tee -a {log}
        python coffea4bees/plots/makePlots.py {input} -o {params.output_dir} -m {params.metadata} {params.extra_arguments} 2>&1 | tee -a {log}
        python src/plotting/pb_pdf_to_png.py -r -j 4 {params.output_dir} 2>&1 | tee -a {log}
        touch {output}
        """


module training:
    snakefile: "Snakefile_Run3MvD_training.smk"
    config: config

use rule create_train_yml    from training
use rule create_evaluate_yml from training
use rule train               from training
use rule analyze             from training
use rule evaluate            from training
