import os
username = os.getenv("USER")

# Use provided config or fall back to defaults
config.setdefault('output_path', 'output/computeJCM/')
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run2/")
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

use rule analysis_processor from analysis as analysis_noJCM with:
    output: f"{config['output_path']}histAll_NoJCM.coffea"
    params:
        datasets = config['dataset'],
        years = config['year'],
        metadata = "coffea4bees/analysis/metadata/HH4b_noJCM.yml",
        processor = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file = config['dataset_location'],
        blind = False,
        run_performance = False,
        extra_arguments = "",
        username = username
    log: f"{config['output_path']}logs/analysis_all.log"


use rule make_JCM from analysis as make_new_JCM with:
    input: f"{config['output_path']}histAll_NoJCM.coffea"
    output: f"{config['output_path']}JCM_2024_v2/jetCombinatoricModel_SB_2024_v2.yml"
    params:
        extra_arguments = "",
        tag = "2024_v2",
        output_dir = f"{config['output_path']}JCM_2024_v2/"
    log: f"{config['output_path']}logs/make_JCM.log"