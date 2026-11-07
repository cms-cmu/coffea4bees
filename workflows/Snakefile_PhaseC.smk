# coffea4bees/workflows/Snakefile_PhaseC.smk
# Phase C: Master Coordinator Workflow for FvT Classifier Pipeline

import os
import yaml

include: "helpers/common.smk"

fvt_cfg = resolve_config_section(config, primary_key='fvt', fallback_keys=['fvt_classifier', 'classifier'])
for k, v in fvt_cfg.items():
    config[k] = v

config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/nominal_sel")
config.setdefault('plot_base', "root://eosuser.cern.ch//eos/user/a/algomez/www/HH4b/Plots")
config.setdefault('classifier_config_paths', "coffea4bees")
config.setdefault('wfs_base', "coffea4bees/classifier/config/workflows/HH4b_2024_v2/FvT")
config.setdefault('label', "FvT_nominal")
config.setdefault('output_dir', f"output/{config['label']}/")
config.setdefault('plot_inputs', True)
config.setdefault('plot_weights', True)
config.setdefault('evaluate', True)

config.setdefault('model', f"{config['eos_base']}/classifier/{config['label']}")
config.setdefault('friend', f"{config['eos_base']}/friend/{config['label']}")
config.setdefault('train_template', f"model: {config['eos_base']}/classifier/{config['label']}")
config.setdefault('eval_template', f"model: {config['eos_base']}/classifier/{config['label']}, FvT: {config['eos_base']}/friend/{config['label']}")
config.setdefault('metadata', "coffea4bees/metadata/datasets/classifier_inputs_nominal.json@@HCR_input")

output_dir = config['output_dir'].rstrip('/')
os.makedirs(output_dir, exist_ok=True)

if 'classifier_setting' in config:
    common_path = f"{output_dir}/common.yml"
    with open(common_path, 'w') as f:
        yaml.dump({'setting': config['classifier_setting']}, f, default_flow_style=False)
    config['common'] = common_path

if 'train_workflow' in config or 'eval_workflow' in config:
    wfs_dir = f"{output_dir}/wfs"
    os.makedirs(wfs_dir, exist_ok=True)
    if 'train_workflow' in config:
        with open(f"{wfs_dir}/train.yml", 'w') as f:
            yaml.dump(config['train_workflow'], f, default_flow_style=False)
    if 'eval_workflow' in config:
        with open(f"{wfs_dir}/evaluate.yml", 'w') as f:
            yaml.dump(config['eval_workflow'], f, default_flow_style=False)
    config['wfs_base'] = wfs_dir
elif config.get('workflow_overrides') or config.get('workflow_inserts'):
    # Keep the checked-in templates, override options by flag (e.g. the roast's JCM / manifest)
    # and insert ones they lack (e.g. a wider --architecture)
    config['wfs_base'] = write_workflow_overrides(config['wfs_base'], config.get('workflow_overrides'),
                                                  f"{output_dir}/wfs", log=print,
                                                  inserts=config.get('workflow_inserts'))

include: "../../src/classifier/workflow/Snakefile"

# ── Training Summary Dashboard & Quality Gate ─────────────────────────────────
localrules: phaseC_training_summary, phaseC_quality_gate

rule phaseC_training_summary:
    input:
        rules.all.input
    output:
        html = f"{output_dir}/training_summary.html"
    params:
        models_dir = output_dir,
        title = lambda wildcards: f"FvT Training Summary — {config.get('roast_id', config.get('label', 'FvT_nominal'))}"
    shell:
        """
        python3 src/classifier/training_summary.py \
            --models-dir {params.models_dir} \
            --output {output.html} \
            --title "{params.title}"
        """

rule phaseC_quality_gate:
    input:
        rules.all.input,
        summary = f"{output_dir}/training_summary.html",
    output:
        token = f"{output_dir}/quality_gate.done"
    params:
        models_dir = output_dir,
        max_chi2 = config.get('fvt_max_chi2_ndf', config.get('max_chi2_ndf', 5.0)),
        metric = config.get('fvt_quality_metric', 'chi2_ndf_fvt'),
    shell:
        """
        python3 src/classifier/check_fvt_quality.py \
            --models-dir {params.models_dir} \
            --max-chi2 {params.max_chi2} \
            --metric {params.metric} \
            --output-token {output.token}
        """

# default_target, not position: Snakemake takes the first rule of the top-level Snakefile
# as the default, so adding a rule above this one would silently shrink the DAG to that
# rule's own inputs.
rule all_PhaseC:
    default_target: True
    input:
        rules.all.input,
        f"{output_dir}/training_summary.html",
        f"{output_dir}/quality_gate.done"
