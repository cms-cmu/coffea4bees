import os

config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/ttHbb_Run3")
config.setdefault('output_path', "output/ttHbb_Run3_evaluation/")

BASE = config['eos_base']
out = config['output_path']
WFS_BASE = "coffea4bees/classifier/config/workflows/ttHbb_Run3"
CLASSIFIER_CONFIG_PATHS = "coffea4bees"

CLASSIFIER_GPU = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest"
INIT = "set -e && set +u && source /entrypoint.sh && set -u && export PYTHONUNBUFFERED=1"

FvT_MODEL = "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/ttHbb_v2/classifier/FvT_ttHbb_v2"
FvT_FRIEND = f"{BASE}/friend/FvT"

SvB_MODEL = "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/ttHbb_v2/classifier/SvB_ttHbb_v2"
SvB_FRIEND = f"{BASE}/friend/SvB"

CLASSIFIERS = {
    "FvT": {
        "model": FvT_MODEL,
        "friend": FvT_FRIEND,
        "eval_template": f"model: {FvT_MODEL}, FvT: {FvT_FRIEND}",
    },
    "SvB": {
        "model": SvB_MODEL,
        "friend": SvB_FRIEND,
        "eval_template": f"model: {SvB_MODEL}, SvB: {SvB_FRIEND}",
    },
}

TARGETS = list(CLASSIFIERS.keys())

rule all:
    input:
        expand(f"{out}{{classifier}}/evaluate.done", classifier=TARGETS)

rule evaluate:
    input:
        eval_yml = lambda wc: f"{WFS_BASE}/{wc.classifier}/evaluate.yml",
        common_yml = f"{WFS_BASE}/common.yml",
    output:
        flag = f"{out}{{classifier}}/evaluate.done",
    log:
        f"{out}{{classifier}}/evaluate.log",
    container: CLASSIFIER_GPU
    resources:
        runtime = 240,
        mem_mb = 16000,
        gres = "mps:25",
        slurm_partition = "work",
    threads: 4
    params:
        init = INIT,
        classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
        wfs_base = WFS_BASE,
        template_str = lambda wc: CLASSIFIERS[wc.classifier]["eval_template"],
    shell:
        """
        mkdir -p $(dirname {output.flag}) $(dirname {log})
        {params.init} && \
        PORT=$(shuf -i 10000-60000 -n 1) && \
        CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
        python -m src.classifier.task.main \
            template "{{{params.template_str}}}" {input.eval_yml} \
            -from {params.wfs_base}/common.yml \
            -setting Monitor "address: '127.0.0.1:$PORT'" \
            2>&1 | tee -a {log}
        touch {output.flag}
        """
