from datetime import datetime
import os

config.setdefault('output_path', "output/lowpt/")
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH5', 'ZH4b', 'ZZ4b', 'ggZH4b'])
config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})

# Derive flat year list from year_eras keys
DATA_YEAR_ERA = [(yr, era) for yr, eras in config['year_eras'].items() for era in eras]
config.setdefault('eos_path', f"{datetime.now().strftime('%Y%m%d')}_lowpt_test")

temp_label = "wNominalSvB"

# Constrain year wildcard to valid year values (avoids ambiguity with underscores in dataset names)
wildcard_constraints:
    year = "|".join(config['year_eras'].keys())

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule all_lowpt:
    input:
        f"{config['output_path']}histAll_lowpt_{temp_label}.coffea",
        f"{config['output_path']}plots_lowpt_{temp_label}/RunII/region_SB/selJets_lowpt_n.pdf"
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

use rule analysis_processor from analysis as analysis_lowpt_data with:
    input: "coffea4bees/analysis/metadata/HH4b_lowpt_2024_v2.yml"
    output: f"{config['output_path']}singlefiles/histAll_lowpt_{temp_label}_data__{{year}}_{{era}}.coffea"
    log: f"{config['output_path']}logs/analysis_lowpt_{temp_label}_data__{{year}}_{{era}}.log"
    params:
        datasets = "data",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input[0],
        processor = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/friends_HH4b_lowpt.yml",
        run_on_condor = True,
        extra_arguments = lambda wildcards: f'"--era {wildcards.era}"',
        run_container_wrapper = "./run_container"

use rule analysis_processor from analysis as analysis_lowpt_MC with:
    input: f"{config['output_path']}HH4b_lowpt_2024_v2_signal.yml"
    output: f"{config['output_path']}singlefiles/histAll_lowpt_{temp_label}__{{dataset}}__{{year}}.coffea"
    log: f"{config['output_path']}logs/analysis_lowpt_{temp_label}_{{dataset}}__{{year}}.log"
    params:
        datasets = "{dataset}",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input[0],
        processor = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        friends = "coffea4bees/metadata/datasets_HH4b_Run2/2024_v2/friends_HH4b_lowpt.yml",
        run_on_condor = True,
        extra_arguments = "",
        run_container_wrapper = "./run_container"

use rule merging_coffea_files from analysis as merging_lowpt_files with:
    input: [f"{config['output_path']}singlefiles/histAll_lowpt_{temp_label}_data__{yr}_{era}.coffea" for yr, era in DATA_YEAR_ERA] + expand("{output_path}singlefiles/histAll_lowpt_" + temp_label + "__{dataset}__{year}.coffea", output_path=config['output_path'], dataset=config['dataset'], year=config['year_eras'].keys())
    output: f"{config['output_path']}histAll_lowpt_{temp_label}.coffea"
    params:
        run_performance = False
    container: config['analysis_container']
    log: f"{config['output_path']}logs/merging_lowpt_files.log" 

use rule make_plots from analysis as make_plots_lowpt with:
    input: f"{config['output_path']}histAll_lowpt_{temp_label}.coffea"
    output: f"{config['output_path']}plots_lowpt_{temp_label}/RunII/region_SB/selJets_lowpt_n.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_lowpt_{temp_label}/",
        metadata = "coffea4bees/plots/metadata/plotsAll_lowpt.yml",
        extra_arguments = "-s xW "

