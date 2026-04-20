import os

config.setdefault('mode', 'nominal')

config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('datasets', ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data'])
config.setdefault('years', ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

if config["mode"] == "nominal":
    config.setdefault('label', '')
    config.setdefault('output_path', "output/Run3/")
    config.setdefault('histogram_config', "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml")

elif config["mode"] == "quadjet_run2":
    config.setdefault('label', '_quadjet_run2')
    config.setdefault('output_path', "output/Run3_quadjet_run2/")
    config.setdefault('histogram_config', "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml")

else:
    print(f"Mode {config['mode']} Not Recognized!")
    import sys
    sys.exit(-1)

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

# ── Histograms ────────────────────────────────────────────────────────────────
# Use __ (double underscore) as separator between dataset and year to avoid
# ambiguous wildcard matching, since both dataset names and years contain _.
# SvB friend trees are reused from the MvD workflow output.

rule create_friends_wSvB:
    input:
        friends_yml    = "coffea4bees/metadata/friends_HH4b.yml",
        svb_json       = "coffea4bees/metadata/datasets_HH4b_Run3/SvBfriend_mixeddata_data.json",
        feynet_json    = "coffea4bees/metadata/datasets_HH4b_Run3/SvBFeynNetfriend_mixeddata_data.json",
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
    shell:
        """
        sed \
            -e 's|  run_SvB.*|  run_SvB: true|' \
            -e 's|  JCM_file.*|  JCM_file: {input.jcm_file}|' \
            {input.config_file} > {output}
        echo "Patched config:"
        grep -E "run_SvB|JCM_file" {output}
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
        run_on_condor = True,
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
