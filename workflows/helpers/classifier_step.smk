# coffea4bees/workflows/helpers/classifier_step.smk
# Set up ONE classifier training/evaluation (src/classifier/workflow/Snakefile) from a config
# block, the way Snakefile_PhaseC.smk / Snakefile_PhaseD.smk do for `fvt:` / `svb:`. Used by
# the MvD roast's falcon steps (Snakefile_MvD_2_train.smk: `mvd:`, Snakefile_MvD_3_svb.smk:
# `svb_mvd:`). The generic Snakefile reads its settings from top-level config globals, so each
# step is its own Snakefile run and includes it exactly once.
#
# Block keys (all as in nominal_run3.yml's fvt/svb): eos_base, plot_base, classifier_config_paths,
# wfs_base, label, output_dir, plot_inputs, plot_weights, evaluate, model, friend, train_template,
# eval_template, metadata, workflow_overrides, workflow_inserts, and workflow_modules (module
# renames, e.g. HCR.SvB.Background -> HCR.SvB.BackgroundMixed; see write_workflow_overrides).

import os
import yaml

include: "common.smk"


def setup_classifier_step(key):
    """Copy config[key] to top level, fill the generic Snakefile's required keys, write
    common.yml from classifier_setting and the overridden templates. Raises if the block is
    missing: a default label/eos_base would train into some other production's area."""
    if not isinstance(config.get(key), dict):
        raise ValueError(f"config block `{key}:` is required for this step")
    blk = resolve_config_section(config, primary_key=key, inherit_keys=[])
    for k, v in blk.items():
        config[k] = v
    for k in ('eos_base', 'plot_base', 'wfs_base', 'label'):
        if not config.get(k):
            raise ValueError(f"`{key}.{k}` is required")
    config.setdefault('classifier_config_paths', "coffea4bees")
    config.setdefault('output_dir', f"output/{config['label']}/")
    config.setdefault('plot_inputs', False)
    config.setdefault('plot_weights', False)
    config.setdefault('evaluate', True)
    config.setdefault('model', "{eos_base}/classifier/{label}")
    config.setdefault('friend', "{eos_base}/friend/{label}")

    output_dir = config['output_dir'].rstrip('/')
    os.makedirs(output_dir, exist_ok=True)
    if 'classifier_setting' in config:
        common_path = f"{output_dir}/common.yml"
        with open(common_path, 'w') as f:
            yaml.dump({'setting': config['classifier_setting']}, f, default_flow_style=False)
        config['common'] = common_path
    if config.get('workflow_overrides') or config.get('workflow_inserts') or config.get('workflow_modules'):
        config['wfs_base'] = write_workflow_overrides(config['wfs_base'], config.get('workflow_overrides'),
                                                      f"{output_dir}/wfs", log=print,
                                                      inserts=config.get('workflow_inserts'),
                                                      modules=config.get('workflow_modules'))
