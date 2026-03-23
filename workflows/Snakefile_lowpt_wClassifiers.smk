from datetime import datetime
import os

config.setdefault('output_path', "output/lowpt/")
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['GluGluToHHTo4B_cHHH1'])
config.setdefault('eras', ['A', 'B', 'C', 'D'])
config.setdefault('year', 'UL18')
config.setdefault('eos_path', f"{datetime.now().strftime('%Y%m%d')}_lowpt_test")

temp_label = "wFvT"

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule all_lowpt:
    input:
        f"{config['output_path']}histAll_lowpt_{temp_label}.coffea",
        f"{config['output_path']}plots_lowpt_{temp_label}/RunII/passPreSel/region_SB/nPVs.pdf"
    shell:
        """
        echo "Copying results to eos"
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d www/HH4b/Plots/{config[eos_path]}/ -t
        """

rule modify_config_file:
    input:
        config_file = "coffea4bees/analysis/metadata/HH4b_lowpt_2024_v2.yml"
    output:
        f"{config['output_path']}HH4b_lowpt_2024_v2_signal.yml"
    shell:
        """
        sed -e 's|apply_FvT: .*|apply_FvT: false|' -e 's|plot_ttbar_with_weights: true|plot_ttbar_with_weights: false|' {input.config_file} > {output}
        """

rule analysis_lowpt_data:
    input: "coffea4bees/analysis/metadata/HH4b_lowpt_2024_v2.yml"
    output: f"{config['output_path']}histAll_lowpt_{temp_label}_data{{era}}.coffea"
    params:
        datasets = "data",
        years = config['year'],
        metadata = input[0],
        processor = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        era = lambda wildcards: wildcards.era,
        datasets_file = config['dataset_location'],
        output = f"histAll_lowpt_{temp_label}_data{{era}}.coffea",
        friends = "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/friends_HH4b_lowpt.yml",
    log: f"{config['output_path']}logs/analysis_lowpt_{temp_label}_data_{{era}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor {params.processor} \
            --config {params.metadata} \
            --dataset-metadata {params.datasets_file} \
            --year {params.years} \
            --datasets {params.datasets} \
            --output-base {config[output_path]} \
            --output-filename {params.output} \
            --friends {params.friends} \
            --condor \
            --no-test \
            --log {log} \
            --additional-flags "--era {params.era}"
        """

use rule analysis_lowpt_data as analysis_lowpt_MC with:
    input: f"{config['output_path']}HH4b_lowpt_2024_v2_signal.yml"
    output: f"{config['output_path']}histAll_lowpt_{temp_label}__{{dataset}}.coffea"
    params:
        datasets = "{dataset}",
        years = config['year'],
        metadata = lambda wildcards, input: input[0],
        processor = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        datasets_file = config['dataset_location'],
        friends = "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/friends_HH4b_lowpt.yml",
        output = f"histAll_lowpt_{temp_label}__{{dataset}}.coffea",
        era = "A"
    log: f"{config['output_path']}logs/analysis_lowpt_{temp_label}_{{dataset}}.log"

use rule merging_coffea_files from analysis as merging_lowpt_files with:
    input: expand("{output_path}histAll_lowpt_" + temp_label + "_data{era}.coffea", output_path=config['output_path'], temp_label=temp_label, era=config['eras']) + expand("{output_path}histAll_lowpt_" + temp_label + "__{dataset}.coffea", output_path=config['output_path'], temp_label=temp_label, dataset=config['dataset'])
    output: f"{config['output_path']}histAll_lowpt_{temp_label}.coffea"
    params:
        run_performance = False
    container: config['analysis_container']
    log: f"{config['output_path']}logs/merging_lowpt_files.log" 

use rule make_plots from analysis as make_plots_lowpt with:
    input: f"{config['output_path']}histAll_lowpt_{temp_label}.coffea"
    output: f"{config['output_path']}plots_lowpt_{temp_label}/RunII/passPreSel/region_SB/nPVs.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_lowpt_{temp_label}/",
        metadata = "coffea4bees/plots/metadata/plotsAll_lowpt.yml",
        extra_arguments = "-s xW "

