# coffea4bees/workflows/Snakefile_PhaseC_2_train.smk
# Phase C.2: FvT Classifier Training Workflow

import os
from datetime import datetime

# Fallback default configuration
config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/nominal_sel")
config.setdefault('plot_base', "root://eosuser.cern.ch//eos/user/a/algomez/www/HH4b/Plots")
config.setdefault('classifier_config_paths', "coffea4bees")
config.setdefault('wfs_base', "coffea4bees/classifier/config/workflows/HH4b_2024_v2/FvT")
config.setdefault('label', "FvT_nominal")
config.setdefault('output_dir', f"output/{config['label']}/")
config.setdefault('plot_inputs', False)
config.setdefault('plot_weights', False)
config.setdefault('evaluate', True)

config.setdefault('model', f"{config['eos_base']}/classifier/{config['label']}")
config.setdefault('friend', f"{config['eos_base']}/friend/{config['label']}")
config.setdefault('train_template', f"model: {config['eos_base']}/classifier/{config['label']}")
config.setdefault('eval_template', f"model: {config['eos_base']}/classifier/{config['label']}, FvT: {config['eos_base']}/friend/{config['label']}")
config.setdefault('metadata', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/classifier_inputs_nominal.json@@HCR_input")

# Include generic classifier workflow
include: "../../src/classifier/workflow/Snakefile"

rule all_fvt_training:
    input:
        rules.all.input
