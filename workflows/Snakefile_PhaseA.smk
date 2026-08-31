import os
import copy
import yaml

# Global configuration defaults
config.setdefault('output_path', "output/skim_trigWeights/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
config.setdefault('dataset', ['ttHbb'])
config.setdefault('year', ['UL18'])
config.setdefault('test', False)

# Save original baseline dataset location for skimmer
original_dataset_location = config.get('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
modified_datasets_file = f"{config['output_path']}modified_datasets/modified_datasets.yml"

# Top target rule
rule all_skim_trigWeights:
    input:
        modified_datasets_file,
        f"{config['output_path']}trigger_weights/trigger_weights_friends.json"

# 1. Include skimmer subworkflow
include: "Snakefile_PhaseA_1_skimmer.smk"

# 2. Point trigger_weights dataset_location to the modified_datasets.yml produced by skimmer
config['trigger_weights_dataset_location'] = modified_datasets_file

# 3. Include trigger weights subworkflow
include: "Snakefile_PhaseA_2_trigWeights.smk"