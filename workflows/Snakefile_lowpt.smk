import os
username = os.getenv("USER")
from datetime import datetime

config.setdefault('output_path', "output/lowpt_friendtrees_LWP/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['data', 'TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic', 'GluGluToHHTo4B_cHHH1', 'GluGluToHHTo4B_cHHH0', 'GluGluToHHTo4B_cHHH2p45', 'GluGluToHHTo4B_cHHH5', 'ZH4b', 'ZZ4b', 'ggZH4b'])
config.setdefault('year', [ 'UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
config.setdefault('classifier_path', 'root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/lowpt_LWP/')
config.setdefault('eos_path', f'Plots/{datetime.now().strftime("%Y%m%d")}_lowpt_friendtrees_LWP/')

# Import analysis module
module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all_lowpt:
    input:
        f"{config['output_path']}plots_noJCMlowpt/RunII/region_SB/nPVs.pdf",
        f"{config['output_path']}plots_wlowptJCM/RunII/region_SB/nPVs.pdf",
        f"{config['output_path']}classifier_inputs_lowpt_wlowptJCM.json"
    # default_target: True
    shell:
        """
        echo "Copying results to eos"
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d www/HH4b/{config[eos_path]}/ -t
        """

rule create_noJCM_config_lowpt:
    input: "coffea4bees/analysis/metadata/HH4b_noJCM.yml"
    output: f"{config['output_path']}HH4b_noJCM_lowpt.yml"
    shell:
        """
        sed -e 's|processor_HH4b.py|processor_HH4b_lowpt.py|' \
            -e 's|friends_HH4b.yml|friends_HH4b_lowpt.yml|' \
            -e 's|coffea4bees/metadata/datasets/|coffea4bees/metadata/datasets/archive/Run2_2024_v2/|' \
            {input} > {output}
        """

use rule analysis_processor from analysis as analysis_nojcm_lowpt with:
    input: f"{config['output_path']}HH4b_noJCM_lowpt.yml"
    output: f"{config['output_path']}histAll_lowpt_noJCM.coffea"
    log: f"{config['output_path']}logs/analysis_nojcm_lowpt.log"
    params:
        datasets = config['dataset'],
        years = config['year'],
        config = lambda wildcards, input: input[0],
        run_container_wrapper = "./run_container"

use rule make_plots from analysis as make_plots_noJCMlowpt with:
    input: f"{config['output_path']}histAll_lowpt_noJCM.coffea"
    output: f"{config['output_path']}plots_noJCMlowpt/RunII/region_SB/nPVs.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_noJCMlowpt/",
        metadata = "coffea4bees/plots/metadata/plotsAll_lowpt_noFvT.yml",
        extra_arguments = "-s xW "

use rule make_JCM from analysis as make_new_JCM_lowpt with:
    input: f"{config['output_path']}histAll_lowpt_noJCM.coffea"
    output: f"{config['output_path']}JCM_lowpt_2024_v2/jetCombinatoricModel_SB_2024_v2.yml"
    params:
        extra_arguments = "--lowpt -m coffea4bees/plots/metadata/plotsJCM_lowpt.yml --jcm_config coffea4bees/analysis/jcm_tools/metadata/lowpt_jcm_config.yml",
        tag = "2024_v2",
        output_dir = f"{config['output_path']}/JCM_lowpt_2024_v2/"
    log: f"{config['output_path']}logs/make_new_JCM_lowpt.log"

rule create_metadata_lowpt:
    input: 
        jcm_file = f"{config['output_path']}JCM_lowpt_2024_v2/jetCombinatoricModel_SB_2024_v2.yml",
        config_file = "coffea4bees/analysis/metadata/HH4b_lowpt_classifier_inputs.yml"
    output: f"{config['output_path']}HH4b_wlowptJCM.yml"
    shell:
        """
        echo "Modifying metadata file to include new JCM"
        sed -e 's|  JCM_file.*|  JCM_file: {input.jcm_file}|' \
            -e 's|make_classifier_input: .*|make_classifier_input: {config[classifier_path]}|' \
            {input.config_file} > {output}
        cat {output}
        """

use rule analysis_processor from analysis as analysis_lowpt_classifier_inputs with:
    input: f"{config['output_path']}HH4b_wlowptJCM.yml"
    output: f"{config['output_path']}histAll_lowpt_wlowptJCM_{{dataset}}__{{year}}.coffea"
    log: f"{config['output_path']}logs/analysis_lowpt_wlowptJCM_{{dataset}}__{{year}}.log"
    container: ""
    params:
        datasets = "{dataset}",
        years = "{year}",
        config = lambda wildcards, input: input[0],
        extra_arguments = "--shared-dask",
        run_container_wrapper = "./run_container"

use rule merging_coffea_files from analysis as merging_lowpt_files with:
    input: expand(f"{config['output_path']}histAll_lowpt_wlowptJCM_{{dataset}}__{{year}}.coffea", dataset=config['dataset'], year=config['year'])
    output: f"{config['output_path']}histAll_lowpt_wlowptJCM.coffea"
    container: config["analysis_container"]
    params:
        run_performance = False
    log: f"{config['output_path']}logs/merging_lowpt_files.log" 

use rule make_plots from analysis as make_plots_wlowptJCM with:
    input: f"{config['output_path']}histAll_lowpt_wlowptJCM.coffea"
    output: f"{config['output_path']}plots_wlowptJCM/RunII/region_SB/nPVs.pdf"
    log: f"{config['output_path']}logs/make_plots.log"
    params:
        output_dir = f"{config['output_path']}plots_wlowptJCM/",
        metadata = "coffea4bees/plots/metadata/plotsAll_lowpt_noFvT.yml",
        extra_arguments = "-s xW "

rule merge_json_friend_files:
    input: expand(f"{config['output_path']}histAll_lowpt_wlowptJCM_{{dataset}}__{{year}}.coffea", dataset=config['dataset'], year=config['year'])
    output: f"{config['output_path']}classifier_inputs_lowpt_wlowptJCM.json"
    log: f"{config['output_path']}logs/merge_json_friend_files.log"
    container: config["analysis_container"]
    shell:
        """
        echo "Merging JSON friend files for classifier input"
        python -m src.friendtrees.merge_friend_meta -i $(echo {input} | sed 's/\.coffea/.json/g') -o {output} 2>&1 | tee -a {log}
        """
