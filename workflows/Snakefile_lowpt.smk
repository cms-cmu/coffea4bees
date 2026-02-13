import os
username = os.getenv("USER")

config.setdefault('output_path', "output/lowpt/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run2/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['data', 'TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic'])
config.setdefault('year', 'UL18')
config.setdefault('eos_path', '20260213_lowpt_test')

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# # Include the base JCM computation rules with the correct config
# module computeJCM:
#     snakefile: "Snakefile_computeJCM.smk"
#     config: config

# use rule * from computeJCM

rule all_lowpt:
    input:
        f"{config['output_path']}histAll_lowpt.coffea",
        f"{config['output_path']}plots/RunII/passPreSel/region_SB/nPVs.pdf",
        f"{config['output_path']}plots_noJCMlowpt/RunII/passPreSel/region_SB/nPVs.pdf",
        # f"{config['output_path']}JCM_lowpt_2024_v2/jetCombinatoricModel_SB_2024_v2.yml"
    # default_target: True
    shell:
        """
        echo "Copying results to eos"
        bash src/tools/copy_files_to_cernbox.sh -s {config['output_path']} -d www/HH4b/{config['eos_path']}/
        """

use rule analysis_processor from analysis as analysis_nojcm_lowpt with:
    output: f"{config['output_path']}histAll_lowpt_noJCM.coffea"
    params:
        datasets = config['dataset'],
        years = config['year'],
        metadata = "coffea4bees/analysis/metadata/HH4b_noJCM.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        extra_arguments = "--condor",
        username = username
    log: f"{config['output_path']}logs/analysis_nojcm_lowpt.log"

use rule make_plots from analysis as make_plots_noJCMlowpt with:
    input: f"{config['output_path']}histAll_lowpt_noJCM.coffea"
    output: f"{config['output_path']}plots_noJCMlowpt/RunII/passPreSel/region_SB/nPVs.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_noJCMlowpt/",
        metadata = "coffea4bees/plots/metadata/plotsAll_lowpt.yml",
        extra_arguments = "-s xW "

use rule make_JCM from analysis as make_new_JCM_lowpt with:
    input: f"{config['output_path']}histAll_lowpt_noJCM.coffea"
    output: f"{config['output_path']}JCM_lowpt_2024_v2/jetCombinatoricModel_SB_2024_v2.yml"
    params:
        extra_arguments = "--lowpt -m coffea4bees/plots/metadata/plotsJCM_lowpt.yml --jcm_config coffea4bees/analysis/jcm_tools/metadata/lowpt_jcm_config.yml",
        tag = "2024_v2",
        output_dir = f"{config['output_path']}/JCM_lowpt_2024_v2/",
    log: f"{config['output_path']}logs/make_new_JCM_lowpt.log"

rule create_metadata_lowpt:
    input: f"{config['output_path']}JCM_lowpt_2024_v2/jetCombinatoricModel_SB_2024_v2.yml"
    output: f"{config['output_path']}HH4b_wlowptJCM.yml"
    shell:
        """
        echo "Modifying metadata file to include new JCM"
        sed -e 's|  JCM_file.*|  JCM_file: {input}|' -e 's|#apply_JCM_lowpt|apply_JCM_lowpt|' coffea4bees/analysis/metadata/HH4b_lowpt_2024_v2.yml > {output}
        cat {output}
        """

use rule analysis_processor from analysis as analysis_lowpt with:
    input: f"{config['output_path']}HH4b_wlowptJCM.yml"
    output: f"{config['output_path']}histAll_lowpt_{{dataset}}.coffea"
    params:
        datasets = "{dataset}",
        years = config['year'],
        metadata = lambda wildcards, input: input[0],
        processor = "coffea4bees/analysis/processors/processor_HH4b_lowpt.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        extra_arguments = "--condor",
        username = username
    log: f"{config['output_path']}logs/analysis_lowpt{{dataset}}.log"

use rule merging_coffea_files from analysis as merging_lowpt_files with:
    input: expand(f"{config['output_path']}histAll_lowpt_{{dataset}}.coffea", dataset=config['dataset'])
    output: f"{config['output_path']}histAll_lowpt.coffea"
    params:
        run_performance = False
    log: f"{config['output_path']}logs/merging_lowpt_files.log" 

use rule make_plots from analysis as make_plots with:
    input: f"{config['output_path']}histAll_lowpt.coffea"
    output: f"{config['output_path']}plots/RunII/passPreSel/region_SB/nPVs.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots/",
        metadata = "coffea4bees/plots/metadata/plotsAll_lowpt.yml",
        extra_arguments = "-s xW "
