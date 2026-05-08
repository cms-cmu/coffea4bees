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

# Container images. The default GPU image targets recent CUDA (good for
# rogue / lxplus). cmslpcgpu* nodes have older P100s (sm60) and need a
# Chuyuan-built variant; the lpc_gpu snakemake profile overrides this knob.
config.setdefault('classifier_gpu_tag', 'classifier_latest')
config.setdefault('classifier_cpu_tag', 'classifier_cpu_latest')
CLASSIFIER_GPU = f"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:{config['classifier_gpu_tag']}"
CLASSIFIER_CPU = f"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:{config['classifier_cpu_tag']}"

# Entrypoint sourced inside the container before running commands
INIT = "set -e && set +u && source /entrypoint.sh && set -u && export PYTHONUNBUFFERED=1"

# Per-classifier configuration.
# train.yml and evaluate.yml live under {WFS_BASE}/{classifier}/
MvD_MODEL  = f"{BASE}/classifier/MvD"
MvD_FRIEND = f"{BASE}/friend/MvD"
CLASSIFIERS = {
    "MvD": {
        "model":          MvD_MODEL,
        "friend":         MvD_FRIEND,
        "train_template": f"model: {MvD_MODEL}",
        "eval_template":  f"model: {MvD_MODEL}, MvD: {MvD_FRIEND}",
    },
}

# Select a single classifier via: snakemake --config classifier=MvD
CLASSIFIER = config.get("classifier", None)
if CLASSIFIER and CLASSIFIER not in CLASSIFIERS:
    raise ValueError(f"Unknown classifier '{CLASSIFIER}'. Choose from: {list(CLASSIFIERS.keys())}")
TARGETS = [CLASSIFIER] if CLASSIFIER else list(CLASSIFIERS.keys())

# Inputs produced by Snakefile_classifier_inputs_Run3MvD.smk and installed to git.
# These paths are also hardcoded in train.yml.
config.setdefault('jcm_install_path', "coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_.yml")
config.setdefault('classifier_inputs_install_path', "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3.json")

config.setdefault('output_path', "output/Run3_MvD/")
config.setdefault('dataset_name', 'mixeddata_all')
out = config['output_path']

TRAIN_YML_TEMPLATE    = f"{WFS_BASE}/MvD/train.yml"
EVALUATE_YML_TEMPLATE = f"{WFS_BASE}/MvD/evaluate.yml"

# Default metadata path. Legacy 'mixeddata_all' uses the archive yaml (which
# has both data/TT/mixeddata_all entries under a `datasets:` wrapper). For
# non-legacy dataset_name (e.g. mixeddata_all_rank0), use the parent dir
# in dir-mode — the merger picks up the rank-suffixed yaml at top level
# alongside data.yml and TT.yml. Don't use parent dir for legacy because
# the parent's mixeddata_all.yml has top-key `mixeddata_all_noTT`, which
# wouldn't match the trainer's `mixeddata_all` lookup.
LEGACY_METADATA = "coffea4bees/metadata/datasets_HH4b_Run3/archive/datasets_HH4b_Run3_2025_Run3_skims"
RANK_METADATA   = "coffea4bees/metadata/datasets_HH4b_Run3/"
_metadata_path  = LEGACY_METADATA if config['dataset_name'] == 'mixeddata_all' else RANK_METADATA


rule create_train_yml:
    """Patch the committed train.yml template with installed JCM / friend
    paths, inject --data-mixed-all-name so the trainer reads from the
    configured dataset_name, and redirect --metadata to a path that has
    that dataset_name registered."""
    input:
        template = TRAIN_YML_TEMPLATE,
        jcm      = config['jcm_install_path'],
        json     = config['classifier_inputs_install_path'],
    output: f"{out}train.yml"
    params:
        dataset_name  = config['dataset_name'],
        metadata_path = _metadata_path,
    shell:
        """
        sed \
            -e 's|--JCM-weight.*|--JCM-weight "source:mixed_all" {input.jcm}@@JCM_weights|' \
            -e 's|--friends.*|--friends "" {input.json}@@HCR_input|' \
            -e 's|--metadata coffea4bees/metadata/datasets_HH4b_Run3/archive/datasets_HH4b_Run3_2025_Run3_skims|--metadata {params.metadata_path}|' \
            -e '/--data-source mixed_all detector/a\\      - --data-mixed-all-name {params.dataset_name}' \
            {input.template} > {output}
        echo "Patched train.yml:"
        grep -E "JCM-weight|friends|data-mixed-all-name|metadata" {output}
        """


rule create_evaluate_yml:
    """Patch the committed evaluate.yml template to inject
    --data-mixed-all-name so evaluation reads from the same dataset as
    training, and redirect --metadata likewise."""
    input:
        template = EVALUATE_YML_TEMPLATE,
    output: f"{out}evaluate.yml"
    params:
        dataset_name  = config['dataset_name'],
        metadata_path = _metadata_path,
    shell:
        """
        sed \
            -e 's|--metadata coffea4bees/metadata/datasets_HH4b_Run3/archive/datasets_HH4b_Run3_2025_Run3_skims|--metadata {params.metadata_path}|' \
            -e '/--data-source detector mixed_all/a\\      - --data-mixed-all-name {params.dataset_name}' \
            {input.template} > {output}
        echo "Patched evaluate.yml:"
        grep -E "data-source|data-mixed-all-name|metadata" {output}
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
        plot                    = lambda wc: f"root://eosuser.cern.ch//eos/user/{CERNUSER}/www/HH4b/Plots/{DATE}_{wc.classifier}_Run3MvD",
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
        train_done   = f"{out}{{classifier}}/train.done",
        evaluate_yml = f"{out}evaluate.yml",
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
    shell:
        """
        {params.init} && \
        PORT=$(shuf -i 10000-60000 -n 1) && \
        CLASSIFIER_CONFIG_PATHS={params.classifier_config_paths} \
        ./src/pyml.py \
            template "{{{params.template_str}}}" {input.evaluate_yml} \
            -from {params.wfs_base}/common.yml \
            -setting Monitor "address: '127.0.0.1:$PORT'" \
            -flag debug \
            2>&1 | tee -a {log}
        touch {output.flag}
        """
