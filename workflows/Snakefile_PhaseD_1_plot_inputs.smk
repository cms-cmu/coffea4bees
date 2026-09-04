# coffea4bees/workflows/Snakefile_PhaseD_1_plot_inputs.smk
# Phase D.1: SvB Classifier Diagnostic Plotting Workflow (plot_inputs & plot_weights)

import os
import yaml

include: "helpers/common.smk"

svb_cfg = resolve_config_section(config, primary_key='svb', fallback_keys=['svb_classifier', 'classifier'])
for k, v in svb_cfg.items():
    config.setdefault(k, v)

config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/nominal_sel")
config.setdefault('plot_base', "root://eosuser.cern.ch//eos/user/a/algomez/www/HH4b/Plots")
config.setdefault('classifier_config_paths', "coffea4bees")
config.setdefault('wfs_base', "coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB")
config.setdefault('label', "SvB_nominal")
config.setdefault('output_dir', f"output/{config['label']}/")
config.setdefault('plot_inputs', True)
config.setdefault('plot_weights', True)
config.setdefault('evaluate', False)

config.setdefault('model', f"{config['eos_base']}/classifier/{config['label']}")
config.setdefault('friend', f"{config['eos_base']}/friend/{config['label']}")
config.setdefault('train_template', f"model: {config['eos_base']}/classifier/{config['label']}, FvT: {config['eos_base']}/friend/FvT_nominal")
config.setdefault('metadata', "coffea4bees/metadata/datasets/classifier_inputs_nominal.json@@HCR_input")

if 'classifier_setting' in config:
    output_dir = config['output_dir'].rstrip('/')
    common_path = f"{output_dir}/common.yml"
    os.makedirs(output_dir, exist_ok=True)
    with open(common_path, 'w') as f:
        yaml.dump({'setting': config['classifier_setting']}, f, default_flow_style=False)
    config['common'] = common_path

include: "../../src/classifier/workflow/Snakefile"

rule all_svb_plot_inputs:
    input:
        f"{OUTPUT_DIR}/plot_inputs_raw.done",
        f"{OUTPUT_DIR}/plot_inputs_dataprep.done",
        f"{OUTPUT_DIR}/plot_weights.done"
