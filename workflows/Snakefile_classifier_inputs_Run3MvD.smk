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
elif config["mode"] == "quadjet_run2":
    config.setdefault('classifier_config', "coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3_quadjet_run2.yml")
    config.setdefault('histogram_config', "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml")
    config.setdefault('label', '_quadjet_run2')   # appended to output_path, e.g. '_quadjet_run2'
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

rule all:
    input:
        f"{out}histAll_Run3MvD{config['label']}.coffea",
        f"{out}classifier_inputs_Run3MvD{config['label']}.json",
        f"{out}jcm_for_mixed_all/jetCombinatoricModel_SB_.yml"

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
