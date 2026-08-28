# coffea4bees/workflows/Snakefile_PhaseD_3_evaluate.smk
# Phase D.3: SvB Classifier Evaluation Workflow
# Supports full evaluation or single-dataset evaluation via --config dataset=<dataset_name>

import os
import copy
import yaml
from datetime import datetime

include: "helpers/common.smk"

# Resolve SvB configuration block from master config or fallback keys
svb_cfg = resolve_config_section(config, primary_key='svb', fallback_keys=['svb_classifier', 'classifier'])
for k, v in svb_cfg.items():
    config.setdefault(k, v)

# Fallback default configuration
config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/nominal_sel")
config.setdefault('plot_base', "root://eosuser.cern.ch//eos/user/a/algomez/www/HH4b/Plots")
config.setdefault('classifier_config_paths', "coffea4bees")
config.setdefault('wfs_base', "coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB")
config.setdefault('label', "SvB_nominal")
config.setdefault('output_dir', f"output/{config['label']}/")
config.setdefault('model', f"{config['eos_base']}/classifier/{config['label']}")
config.setdefault('friend', f"{config['eos_base']}/friend/{config['label']}")
config.setdefault('eval_template', f"model: {config['eos_base']}/classifier/{config['label']}, SvB: {config['eos_base']}/friend/{config['label']}")

CLASSIFIER = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest"
INIT = "set -e && set +u && { [ -f /entrypoint.sh ] && source /entrypoint.sh || true; } && set -u && export PYTHONUNBUFFERED=1"

OUTPUT_DIR = config["output_dir"].rstrip("/")
LABEL = config["label"]
EOS_BASE = config["eos_base"]
EVAL_TEMPLATE = config["eval_template"].format(eos_base=EOS_BASE, label=LABEL)
WFS_BASE = config["wfs_base"]
CLASSIFIER_CONFIG_PATHS = config["classifier_config_paths"]

# Handle classifier_setting embedded in master config
if 'classifier_setting' in config:
    common_path = f"{OUTPUT_DIR}/common.yml"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(common_path, 'w') as f:
        yaml.dump({'setting': config['classifier_setting']}, f, default_flow_style=False)
    COMMON_PATH = common_path
else:
    COMMON_PATH = config.get("common", f"{WFS_BASE}/../common.yml")

target_datasets = config.get('dataset', config.get('datasets', None))
if target_datasets and isinstance(target_datasets, str):
    target_datasets = [d.strip() for d in target_datasets.split(",") if d.strip()]

if target_datasets:
    # Single or selected dataset evaluation targets
    dataset_targets = [f"{OUTPUT_DIR}/evaluate_{d}.done" for d in target_datasets]

    rule all_svb_evaluation:
        input: dataset_targets

    rule evaluate_single_dataset_svb:
        output:
            flag = f"{OUTPUT_DIR}/evaluate_{{dataset}}.done",
        log:
            f"{OUTPUT_DIR}/logs/evaluate_{{dataset}}.log",
        container: CLASSIFIER
        resources:
            runtime = 240,
            mem_mb  = 64000,
            gres    = "mps:25",
            qos     = "light",
        threads: 8
        params:
            init                    = INIT,
            classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
            wfs_base                = WFS_BASE,
            template_str            = EVAL_TEMPLATE,
            common                  = COMMON_PATH,
            dataset                 = "{dataset}"
        shell:
            """
            {params.init} && \
            PORT=$(shuf -i 10200-10300 -n 1) && \
            CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
            python -m src.classifier.task.main \
                template "{{{params.template_str}}}" {params.wfs_base}/evaluate.yml \
                -from {params.common} \
                -setting dataset "{params.dataset}" \
                -setting Monitor "address: '127.0.0.1:$PORT'" \
                -flag debug \
                2>&1 | tee -a {log}
            touch {output.flag}
            """
else:
    rule all_svb_evaluation:
        input: f"{OUTPUT_DIR}/evaluate.done"

    rule evaluate_all_svb:
        output:
            flag = f"{OUTPUT_DIR}/evaluate.done",
        log:
            f"{OUTPUT_DIR}/evaluate.log",
        container: CLASSIFIER
        resources:
            runtime = 240,
            mem_mb  = 64000,
            gres    = "mps:25",
            qos     = "light",
        threads: 8
        params:
            init                    = INIT,
            classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
            wfs_base                = WFS_BASE,
            template_str            = EVAL_TEMPLATE,
            common                  = COMMON_PATH,
        shell:
            """
            {params.init} && \
            PORT=$(shuf -i 10200-10300 -n 1) && \
            CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
            python -m src.classifier.task.main \
                template "{{{params.template_str}}}" {params.wfs_base}/evaluate.yml \
                -from {params.common} \
                -setting Monitor "address: '127.0.0.1:$PORT'" \
                -flag debug \
                2>&1 | tee -a {log}
            touch {output.flag}
            """
