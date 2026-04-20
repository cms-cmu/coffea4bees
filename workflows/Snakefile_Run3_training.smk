from datetime import datetime
DATE = datetime.now().strftime("%Y%m%d")

##### change these vars #####
config.setdefault('lpc_user',  "jda102")
config.setdefault('cern_user', "j/johnda")
config.setdefault('eos_base',  f"root://cmseos.fnal.gov//store/user/{config['lpc_user']}/HH4b_Run3_v2")
#############################

BASE     = config['eos_base']
CERNUSER = config['cern_user']

WFS_BASE = "coffea4bees/classifier/config/workflows/HH4b_Run3"
CLASSIFIER_CONFIG_PATHS = "coffea4bees"

# Container images
CLASSIFIER_GPU = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest"
CLASSIFIER_CPU = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_cpu_latest"

# Entrypoint sourced inside the container before running commands
INIT = "set -e && set +u && source /entrypoint.sh && set -u && export PYTHONUNBUFFERED=1"

# Per-classifier configuration.
# train.yml and evaluate.yml live under {WFS_BASE}/{classifier}/
FvT_MODEL  = f"{BASE}/classifier/FvT"
FvT_FRIEND = f"{BASE}/friend/FvT"
CLASSIFIERS = {
    "FvT": {
        "model":          FvT_MODEL,
        "friend":         FvT_FRIEND,
        "train_template": f"model: {FvT_MODEL}",
        "eval_template":  f"model: {FvT_MODEL}, FvT: {FvT_FRIEND}",
    },
}

# Select a single classifier via: snakemake --config classifier=FvT
CLASSIFIER = config.get("classifier", None)
if CLASSIFIER and CLASSIFIER not in CLASSIFIERS:
    raise ValueError(f"Unknown classifier '{CLASSIFIER}'. Choose from: {list(CLASSIFIERS.keys())}")
TARGETS = [CLASSIFIER] if CLASSIFIER else list(CLASSIFIERS.keys())

# Inputs produced by Snakefile_Run3.smk and installed to git.
# These paths are also hardcoded in train.yml.
config.setdefault('jcm_install_path', "coffea4bees/analysis/weights/JCM/Run3/jetCombinatoricModel_SB_.yml")
config.setdefault('classifier_inputs_install_path', "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_Run3.json")

config.setdefault('output_path', "output/Run3/")
out = config['output_path']
LABEL = config.get('label', '')

TRAIN_YML_TEMPLATE = f"{WFS_BASE}/FvT/train.yml"

rule create_train_yml:
    input:
        template = TRAIN_YML_TEMPLATE,
        jcm      = config['jcm_install_path'],
        json     = config['classifier_inputs_install_path'],
    output: f"{out}train.yml"
    shell:
        """
        sed \
            -e 's|--JCM-weight.*|--JCM-weight "" {input.jcm}@@JCM_weights|' \
            -e 's|--friends.*|--friends "" {input.json}@@HCR_input|' \
            {input.template} > {output}
        echo "Patched train.yml:"
        grep -E "JCM-weight|friends" {output}
        """


rule all_training:
    input:
        expand(f"{out}{{classifier}}/evaluate.done", classifier=TARGETS),
        expand(f"{out}{{classifier}}/analyze.done",  classifier=TARGETS),


rule train:
    input:
        train_yml = f"{out}train.yml",
    output:
        flag = f"{out}{{classifier}}/train.done",
    log:
        f"{out}{{classifier}}/train.log",
    container: CLASSIFIER_GPU
    resources:
        runtime = 240,
        mem_mb  = 32000,
        gres    = "mps:50",
    threads: 4
    params:
        init                    = INIT,
        classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
        wfs_base                = WFS_BASE,
        template_str            = lambda wc: CLASSIFIERS[wc.classifier]["train_template"],
        wfs                     = lambda wc: f"{WFS_BASE}/{wc.classifier}",
    shell:
        """
        {params.init} && \
        PORT=$(shuf -i 10000-60000 -n 1) && \
        CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
        ./src/pyml.py \
            template "{{{params.template_str}}}" {input.train_yml} \
            -from {params.wfs_base}/common.yml \
            -setting Monitor "address: '127.0.0.1:$PORT'" \
            -flag debug \
            2>&1 | tee -a {log}
        touch {output.flag}
        """


rule analyze:
    input:
        f"{out}{{classifier}}/train.done",
    output:
        flag = f"{out}{{classifier}}/analyze.done",
    log:
        f"{out}{{classifier}}/analyze.log",
    container: CLASSIFIER_CPU
    resources:
        runtime = 60,
        mem_mb  = 8000,
    threads: 1
    params:
        init                    = INIT,
        classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
        model                   = lambda wc: CLASSIFIERS[wc.classifier]["model"],
        report                  = lambda wc: wc.classifier,
        plot                    = lambda wc: f"root://eosuser.cern.ch//eos/user/{CERNUSER}/www/HH4b/Plots/{DATE}_{wc.classifier}_Run3{LABEL}",
    shell:
        """
        mkdir -p proxy
        if ! voms-proxy-info --file ./proxy/x509_proxy --exists -valid 1:00 &>/dev/null; then
            voms-proxy-init -voms cms -valid 192:00 -out ./proxy/x509_proxy
        fi
        export X509_USER_PROXY="$PWD/proxy/x509_proxy"
        {params.init} && \
        PORT=$(shuf -i 10000-60000 -n 1) && \
        CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
        ./src/pyml.py analyze \
            --results {params.model}/result.json \
            -analysis HCR.LossROC \
            -setting IO "output: {params.plot}" \
            -setting IO "report: {params.report}" \
            -setting Monitor "address: '127.0.0.1:$PORT'" \
            2>&1 | tee -a {log}
        touch {output.flag}
        """


rule evaluate:
    input:
        f"{out}{{classifier}}/train.done",
    output:
        flag = f"{out}{{classifier}}/evaluate.done",
    log:
        f"{out}{{classifier}}/evaluate.log",
    container: CLASSIFIER_GPU
    resources:
        runtime = 240,
        mem_mb  = 32000,
        gres    = "mps:50",
    threads: 4
    params:
        init                    = INIT,
        classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
        wfs_base                = WFS_BASE,
        template_str            = lambda wc: CLASSIFIERS[wc.classifier]["eval_template"],
        wfs                     = lambda wc: f"{WFS_BASE}/{wc.classifier}",
    shell:
        """
        {params.init} && \
        PORT=$(shuf -i 10000-60000 -n 1) && \
        CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
        ./src/pyml.py \
            template "{{{params.template_str}}}" {params.wfs}/evaluate.yml \
            -from {params.wfs_base}/common.yml \
            -setting Monitor "address: '127.0.0.1:$PORT'" \
            -flag debug \
            2>&1 | tee -a {log}
        touch {output.flag}
        """
