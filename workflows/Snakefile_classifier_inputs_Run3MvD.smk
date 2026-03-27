import os

config.setdefault('mode', 'nominal')
config.setdefault('output_path', "output/Run3_MvD/")

config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('datasets', ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data', 'mixeddata_all'])
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

if config["mode"] == "nominal":
    config.setdefault('label', '')   # appended to output_path, e.g. '_quadjet_run2'
    config.setdefault('classifier_config', "coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3.yml")
    config.setdefault('histogram_config', "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml")

    config.setdefault('jcm_install_path', "coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_.yml")
    config.setdefault('classifier_inputs_install_path', "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3.json")
    config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_v2")

elif config["mode"] == "quadjet_run2":
    config.setdefault('classifier_config', "coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3_quadjet_run2.yml")
    config.setdefault('histogram_config', "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml")
    config.setdefault('label', '_quadjet_run2')

    config.setdefault('jcm_install_path', "coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_quadjet_run2.yml")
    config.setdefault('classifier_inputs_install_path', "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3_quadjet_run2.json")
    config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_quadjet_run2")



else:
    print(f"Mode {config['mode']} Not Recognized!")
    import sys
    sys.exit(-1)

# Effective output path incorporates the label so variants don't overwrite nominal outputs
out = config['output_path'].rstrip('/') + config['label'] + '/'

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

config.setdefault('jcm_config', "coffea4bees/analysis/jcm_tools/metadata/mixeddata_all_config_Run3.yml")
config.setdefault('plots_metadata', "coffea4bees/plots/metadata/plotsAll_MvD.yml")

rule all:
    input:
        f"{out}histAll_Run3MvD{config['label']}.coffea",
        f"{out}classifier_inputs_Run3MvD{config['label']}.json",
        f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml",
        f"{out}plots_wJCM/plots_done.txt"

rule all_with_training:
    input:
        rules.all.input,
        expand("output/Run3_MvD/{classifier}/evaluate.done", classifier=["MvD"]),
        expand("output/Run3_MvD/{classifier}/analyze.done",  classifier=["MvD"])

# ── Histograms ────────────────────────────────────────────────────────────────
# Use __ (double underscore) as separator between dataset and year to avoid
# ambiguous wildcard matching, since both dataset names and years contain _.

use rule analysis_processor from analysis as make_histograms with:
    output: f"{out}histograms/hist_{{dataset}}__{{year}}.coffea"
    log: f"{out}logs/hist_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = "{year}",
        config = config['histogram_config'],
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor = True,
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

rule make_JCM_Run3MvD:
    input: f"{out}histAll_Run3MvD{config['label']}.coffea"
    output: f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml"
    container: config['analysis_container']
    params:
        output_dir = f"{out}jcm_for_mixed_all/",
        jcm_config = config['jcm_config'],
        extra_arguments = "",
    log: f"{out}logs/make_JCM_Run3MvD.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        echo "Computing JCM for mixed_all Run3MvD" 2>&1 | tee -a {log}
        python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
            -o {params.output_dir} \
            -i {input} \
            -r SB \
            --jcm_config {params.jcm_config} \
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
            -e 's|  JCM_file.*|  JCM_file: {input.jcm_file}|' \
            -e 's|  apply_MvD_weight.*|  apply_MvD_weight: false|' \
            -e 's|  apply_MvD:[^_].*|  apply_MvD: true|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "JCM_file|apply_MvD" {output}
        """

use rule analysis_processor from analysis as make_histograms_mixeddata_wJCM with:
    input: f"{out}histogram_config_wJCM.yml"
    output: f"{out}histograms_wJCM/hist_mixeddata_all__{{year}}.coffea"
    log: f"{out}logs/hist_wJCM_mixeddata_all__{{year}}.log"
    params:
        datasets = "mixeddata_all",
        years = "{year}",
        config = lambda wildcards, input: input[0],
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor = True,
        extra_arguments = "",
        run_container_wrapper = "./run_container",
        dashboard_address = 0

use rule merging_coffea_files from analysis as merge_histograms_mixeddata_wJCM with:
    input:
        expand(
            "{out}histograms_wJCM/hist_mixeddata_all__{year}.coffea",
            out=out,
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

# ── Classifier inputs ─────────────────────────────────────────────────────────

use rule analysis_processor from analysis as classifier_inputs with:
    output: f"{out}classifier_inputs/classifier_inputs_{{dataset}}__{{year}}.coffea"
    log: f"{out}logs/classifier_inputs_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = "{year}",
        config = config['classifier_config'],
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor = True,
        extra_arguments = "",
        run_container_wrapper = "./run_container",
        dashboard_address = 0

rule merge_json_classifier_inputs:
    input:
        expand(
            "{out}classifier_inputs/classifier_inputs_{dataset}__{year}.coffea",
            out=out,
            dataset=config['datasets'],
            year=config['years']
        )
    output: f"{out}classifier_inputs_Run3MvD{config['label']}.json"
    log: f"{out}logs/merge_json_classifier_inputs.log"
    container: config["analysis_container"]
    shell:
        """
        echo "Merging JSON classifier input files" 2>&1 | tee -a {log}
        python -m src.friendtrees.merge_friend_meta -i $(echo {input} | sed 's/\\.coffea/.json/g') -o {output} 2>&1 | tee -a {log}
        """

# ── Classifier training ────────────────────────────────────────────────────────
# Install the fitted JCM and classifier inputs JSON to the paths expected by
# train.yml. Commit both files to git after running to version the inputs.

rule install_JCM:
    input: f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml"
    output: config['jcm_install_path']
    shell:
        """
        mkdir -p $(dirname {output})
        cp {input} {output}
        echo "Installed JCM to {output} — commit this file to git to version it."
        """

rule install_classifier_inputs:
    input: f"{out}classifier_inputs_Run3MvD{config['label']}.json"
    output: config['classifier_inputs_install_path']
    shell:
        """
        mkdir -p $(dirname {output})
        cp {input} {output}
        echo "Installed classifier inputs JSON to {output} — commit this file to git to version it."
        """

module training:
    snakefile: "Snakefile_classifier_training_Run3MvD.smk"
    config: config

use rule create_train_yml from training
use rule train            from training
use rule analyze          from training
use rule evaluate         from training
