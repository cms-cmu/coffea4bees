# Dedicated Snakemake workflow for MixedData processing, data vs mixeddata comparison plots, and unblinded Combine statistical analysis.
import os

configfile: "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

config.setdefault('output_path', 'output/ttHbb/')
config.setdefault('label', 'ttHbb_mixeddata')
config.setdefault('analysis_container_wrapper', "./run_container")
config.setdefault('stats_container_wrapper', "./run_container combine")
config.setdefault('container_wrapper', "./run_container combine")

MIXEDDATA_YEARS = ['2016', '2017', '2018']
SIGNAL_YEARS = ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18']

include: "helpers/common.smk"

original_config = workflow.configfiles[0] if workflow.configfiles else "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

wildcard_constraints:
    myear = "|".join(MIXEDDATA_YEARS)

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

include: "Snakefile_stats.smk"

rule all:
    input:
        f"{config['output_path']}plots_mixeddata_vs_data/plots_done.txt",
        f"{config['output_path']}plots_mixeddata/plots_done.txt",
        f"{config['output_path']}stat_analysis/ttHbb_mixeddata/limits/datacard_limits__ttHbb.json",
        f"{config['output_path']}stat_analysis/ttHbb_mixeddata/postfit/datacard_postfit__ttHbb.pdf",
        f"{config['output_path']}stat_analysis/ttHbb_mixeddata/significance/datacard_significance__ttHbb.log",
        f"{config['output_path']}stat_analysis/ttHbb_mixeddata/likelihood_scan/datacard_likelihood_scan__ttHbb.pdf",

ROOT_DIR = os.getcwd()

use rule make_plots from analysis as make_plots_data_vs_mixeddata with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea",
        metadata_file = lambda wildcards: os.path.join(ROOT_DIR, "coffea4bees/plots/metadata/plots_mixeddata_vs_data.yml"),
        plot_script = lambda wildcards: os.path.join(ROOT_DIR, "coffea4bees/plots/makePlots.py")
    output: f"{config['output_path']}plots_mixeddata_vs_data/plots_done.txt"
    log: f"{config['output_path']}logs/make_plots_mixeddata_vs_data.log"
    params:
        output_dir = f"{config['output_path']}plots_mixeddata_vs_data/",
        metadata = "coffea4bees/plots/metadata/plots_mixeddata_vs_data.yml",
        extra_arguments = "-s xW --year RunII",
        png_cores = 4,
        run_container_wrapper = config['analysis_container_wrapper']
    container: None

use rule make_plots from analysis as make_plots_mixeddata with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea",
        metadata_file = lambda wildcards: os.path.join(ROOT_DIR, "coffea4bees/plots/metadata/plotsAll_mixeddata.yml"),
        plot_script = lambda wildcards: os.path.join(ROOT_DIR, "coffea4bees/plots/makePlots.py")
    output: f"{config['output_path']}plots_mixeddata/plots_done.txt"
    log: f"{config['output_path']}logs/make_plots_mixeddata.log"
    params:
        output_dir = f"{config['output_path']}plots_mixeddata/",
        metadata = "coffea4bees/plots/metadata/plotsAll_mixeddata.yml",
        extra_arguments = "-s xW --year RunII",
        png_cores = 4,
        run_container_wrapper = config['analysis_container_wrapper']
    container: None

localrules: all, analysis_mixeddata, merging_files, make_plots_data_vs_mixeddata, make_plots_mixeddata