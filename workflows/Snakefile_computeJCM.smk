import os
username = os.getenv("USER")

# Use provided config or fall back to defaults
config.setdefault('output_path', 'output/computeJCM/')
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['data', 'TTToSemiLeptonic', 'TTTo2L2Nu', 'TTToHadronic'])
config.setdefault('year', 'UL18')

rule output_computeJCM:
    input:
        f"{config['output_path']}JCM_2024_v2/jetCombinatoricModel_SB_2024_v2.yml"

### Including modules
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule create_noJCM_config:
    input: "coffea4bees/analysis/metadata/HH4b_noJCM.yml"
    output: f"{config['output_path']}analysis_config_noJCM.yml"
    params:
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        dataset_location = config['dataset_location'],
        friend_file = "coffea4bees/metadata/friends/friends_HH4b.yml"
    run:
        import yaml
        with open(input[0], 'r') as f:
            cfg = yaml.safe_load(f) or {}
        cfg['processor'] = params.processor
        cfg['dataset_location'] = params.dataset_location
        cfg['friend_file'] = params.friend_file
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as analysis_noJCM with:
    input: f"{config['output_path']}analysis_config_noJCM.yml"
    output: f"{config['output_path']}histAll_NoJCM.coffea"
    params:
        datasets = config['dataset'],
        years = config['year'],
        config = lambda wildcards, input: input[0]
    log: f"{config['output_path']}logs/analysis_all.log"


use rule make_JCM from analysis as make_new_JCM with:
    input: f"{config['output_path']}histAll_NoJCM.coffea"
    output: f"{config['output_path']}JCM_2024_v2/jetCombinatoricModel_SB_2024_v2.yml"
    params:
        extra_arguments = "",
        tag = "2024_v2",
        output_dir = f"{config['output_path']}JCM_2024_v2/"
    log: f"{config['output_path']}logs/make_JCM.log"