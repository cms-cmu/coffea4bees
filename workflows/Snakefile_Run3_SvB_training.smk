"""Snakefile: Run3 SvB training (quadjet_run2 inputs).

Adapted from Snakefile_Run3MvD_training.smk and Snakefile_Run3_training.smk.

SvB is structurally heavier than FvT/MvD because:
  * train.yml uses two `dataset` modules (HCR.SvB.Background + HCR.SvB.Signal)
  * The Background module needs the FvT result.json as an additional friend
    tree (the "label:data" friend) to apply FvT weights to the 3b→4b
    background projection during training.

The FvT result.json is treated as a *soft* dependency — Snakemake does not
re-train FvT. The path defaults to ${EOS}/friend/FvT which is whatever
the most recent FvT-evaluate step on the same EOS base produced.

Usage:
    ./run_container snakemake \\
        --snakefile coffea4bees/workflows/Snakefile_Run3_SvB_training.smk \\
        --cores 1 --resources gres=mps:25 \\
        output/Run3_quadjet_run2/SvB/train.done

Override paths via --config when needed, e.g. to point at a different EOS
prefix or a non-quadjet_run2 JCM/inputs combination.
"""
from datetime import datetime
DATE = datetime.now().strftime("%Y%m%d")

##### change these vars #####
config.setdefault('lpc_user',  "jda102")
config.setdefault('cern_user', "j/johnda")
config.setdefault('eos_base',
    f"root://cmseos.fnal.gov//store/user/{config['lpc_user']}/HH4b_Run3_quadjet_run2")
#############################

BASE     = config['eos_base']
CERNUSER = config['cern_user']

WFS_BASE = "coffea4bees/classifier/config/workflows/HH4b_Run3"
CLASSIFIER_CONFIG_PATHS = "coffea4bees"

# Container images. Same auto-detect logic as the other Run3 training
# Snakefiles: cmslpcgpu* nodes have older P100s (sm60) and need a
# Chuyuan-built variant; everyone else defaults to the recent CUDA image.
import socket as _socket
_default_gpu_tag = (
    'classifier_lpc_latest' if 'cmslpcgpu' in _socket.gethostname()
    else 'classifier_latest'
)
config.setdefault('classifier_gpu_tag', _default_gpu_tag)
config.setdefault('classifier_cpu_tag', 'classifier_cpu_latest')
CLASSIFIER_GPU = f"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:{config['classifier_gpu_tag']}"
CLASSIFIER_CPU = f"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:{config['classifier_cpu_tag']}"

# Entrypoint sourced inside the container before running commands
INIT = "set -e && set +u && source /entrypoint.sh && set -u && export PYTHONUNBUFFERED=1"

# Per-variant configuration.
# `svb_variant` selects the study variant: it names the WFS config subdir
# (e.g. HH4b_Run3/SvB_SRSB), the model/friend EOS subdir, and the plot/report
# name. Default 'SvB' reproduces the nominal training exactly. The FvT friend
# stays at the shared nominal base regardless of variant.
config.setdefault('svb_variant', 'SvB')
VARIANT = config['svb_variant']

SvB_MODEL  = f"{BASE}/classifier/{VARIANT}"
SvB_FRIEND = f"{BASE}/friend/{VARIANT}"
FvT_FRIEND = f"{BASE}/friend/FvT"      # soft reference — assumed pre-existing
CLASSIFIERS = {
    VARIANT: {
        "model":          SvB_MODEL,
        "friend":         SvB_FRIEND,
        # SvB train.yml needs both {model} and {FvT}.
        "train_template": f"model: {SvB_MODEL}, FvT: {FvT_FRIEND}",
        "eval_template":  f"model: {SvB_MODEL}, SvB: {SvB_FRIEND}",
    },
}

# Single-target Snakefile; classifier=VARIANT always. Kept structurally
# consistent with the FvT/MvD Snakefiles so it's easy to extend later.
CLASSIFIER = config.get("classifier", VARIANT)
if CLASSIFIER not in CLASSIFIERS:
    raise ValueError(f"Unknown classifier '{CLASSIFIER}'. Choose from: {list(CLASSIFIERS.keys())}")
TARGETS = [CLASSIFIER]

# Inputs (already installed in the repo).
# Defaults follow the quadjet_run2 mode of Snakefile_Run3.smk.
config.setdefault('jcm_install_path',
    "coffea4bees/analysis/weights/JCM/Run3/jetCombinatoricModel_SB_quadjet_run2.yml")
config.setdefault('classifier_inputs_install_path',
    # SvB-specific JSON — has HH4b signal entries in addition to data+TT.
    # Produced by Snakefile_classifier_inputs_Run3.smk with
    #   dataset_name=GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00,
    #   reuse_legacy_classifier_inputs=true.
    "coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_SvB_Run3_quadjet_run2.json")

config.setdefault('output_path', "output/Run3_quadjet_run2/")
out = config['output_path']
LABEL = config.get('label', '')

TRAIN_YML_TEMPLATE    = f"{WFS_BASE}/{VARIANT}/train.yml"
EVALUATE_YML_TEMPLATE = f"{WFS_BASE}/{VARIANT}/evaluate.yml"


rule create_train_yml:
    """Patch the committed SvB train.yml template with the installed JCM
    weight path and the two friend-tree paths.

    Unlike the FvT/MvD Snakefiles which only have one `--friends` line,
    SvB train.yml has TWO: the HCR_input friends (the classifier-inputs
    JSON, applied to both Background and Signal datasets) and the
    label:data friend (the FvT result.json, applied only to the
    Background dataset to provide FvT weights). The two are anchored
    on distinct trailing keywords (HCR_input vs analysis.0.merged) so
    the sed patterns don't collide.
    """
    input:
        template = TRAIN_YML_TEMPLATE,
        jcm      = config['jcm_install_path'],
        json     = config['classifier_inputs_install_path'],
    output: f"{out}{VARIANT}_train.yml"
    shell:
        """
        sed \
            -e 's|^      - --JCM-weight.*|      - --JCM-weight "" {input.jcm}@@JCM_weights|' \
            -e 's|^      - --friends "" .*HCR_input.*|      - --friends "" {input.json}@@HCR_input|' \
            {input.template} > {output}
        echo "Patched SvB train.yml:"
        grep -E "JCM-weight|friends" {output}
        """


rule create_evaluate_yml:
    """Patch the committed SvB evaluate.yml template with the installed
    classifier-inputs JSON path."""
    input:
        template = EVALUATE_YML_TEMPLATE,
        json     = config['classifier_inputs_install_path'],
    output: f"{out}{VARIANT}_evaluate.yml"
    shell:
        """
        sed \
            -e 's|^      - --friends "" .*HCR_input.*|      - --friends "" {input.json}@@HCR_input|' \
            {input.template} > {output}
        echo "Patched SvB evaluate.yml:"
        grep -E "friends" {output}
        """


rule all_training:
    input:
        expand(f"{out}{{classifier}}/evaluate.done", classifier=TARGETS),
        expand(f"{out}{{classifier}}/analyze.done",  classifier=TARGETS),


rule train:
    input:
        train_yml = f"{out}{VARIANT}_train.yml",
    output:
        flag = f"{out}{{classifier}}/train.done",
    log:
        f"{out}{{classifier}}/train.log",
    container: CLASSIFIER_GPU
    resources:
        # mem/runtime sized for the SR+SB variant: the sideband keeps many more
        # events than SR-only, so peak RSS exceeds the old 32 GB cgroup cap and
        # the job swap-thrashes to a standstill (same failure mode the eval rule
        # hit — see SvB-Run3 notes). 56 GB + 8 h matches the eval-rule tuning.
        # The nominal SR-only training fits in 32 GB/52 min; these are upper
        # bounds with margin, the scheduler only charges what is used.
        # 100x nets (SR+SB esp.) are slow at the capped batch -> 16h; others 8h.
        runtime = 960 if "100x" in VARIANT else 480,
        mem_mb  = 56000,
        # MPS = fraction of the A100 (mps:N = N% SMs). The admin reaper kills a
        # job whose actual GPU SM usage exceeds its mps:N request. The wide 30x
        # net (n_features=54) peaks above 30% and got reaped at mps:30, so wide
        # nets need mps:50 (medium's per-job max) for headroom; smaller nets fit
        # mps:30. (mps:50 also runs ~faster: 50% SMs vs 30%.)
        gres    = "mps:50" if ("30x" in VARIANT or "100x" in VARIANT) else "mps:30",
        # >25 MPS/job requires the 'medium' QoS (CMS-CMU SLURM guide); jobs left
        # on the default 'light' QoS get reaped by the admin (CANCELLED by 0 at
        # ~5 min). NB: medium MaxWall is 8h, so the 100x variant (runtime 960=16h)
        # would be DenyOnLimit-rejected here — 100x needs a separate QoS/time plan.
        qos     = "medium",
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
        train_done = f"{out}{{classifier}}/train.done",
        eval_yml   = f"{out}{VARIANT}_evaluate.yml",
    output:
        flag = f"{out}{{classifier}}/evaluate.done",
    log:
        f"{out}{{classifier}}/evaluate.log",
    container: CLASSIFIER_GPU
    resources:
        # SvB eval with max-evaluators=1 does 3 sequential passes (one per fold);
        # 4h SLURM limit isn't enough. Bumped to 8h. Memory sized to fit rogue01's
        # free RAM (58 GB) without swap; max-evaluators bumped back to 2 to
        # complete in ~3-4h while staying under the 56 GB cap.
        runtime = 480,
        mem_mb  = 56000,
        gres    = "mps:25",
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
            template "{{{params.template_str}}}" {input.eval_yml} \
            -from {params.wfs_base}/common.yml \
            -setting Monitor "address: '127.0.0.1:$PORT'" \
            2>&1 | tee -a {log}
        touch {output.flag}
        """
