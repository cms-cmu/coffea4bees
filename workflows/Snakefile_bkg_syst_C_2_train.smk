# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_2_train.smk
#
# Stage C_2: FvT Neural Network Training across 16 Subsamples (GPU on falcon / bridges2)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Trains 16 independent FvT (Four-versus-Three tag) PyTorch neural networks
# (mix_0..mix_15) for the ttH(bb) background systematics evaluation.
#
# Each model is trained on a dedicated pseudo-experiment realization to evaluate
# background shape and normalization uncertainties. To ensure strict statistical
# independence and physical accuracy:
#   1. Dedicated JCM: Each training mix_v{m} reweights 3-tag collision data
#      using its OWN dedicated JCM transfer function derived in Stage B_1:
#      output/ttHbb_bkg_syst/bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}.yml
#   2. Full 4b Target (`mixeddata_4b:{m}`): Each 4b target contains both the
#      hemisphere-mixed multijet subsample AND the sliced pseudotagged ttbar MC
#      (subsample + PSttbar from Stage A_2 / A_3).
#   3. Full Detector Sample: Contains Data 3b, ttbar MC 4b, and ttbar MC 3b
#      (`--data-source detector mixed --no-detector-4b`), allowing the classifier
#      to separate multijet QCD from ttbar and derive both the multijet reweighting
#      and ttbar fraction simultaneously.
#
# NEURAL NETWORK ARCHITECTURE & TRAINING DETAILS:
#   - Framework: PyTorch with HCR (Hierarchical Classifier for Resonances).
#   - Inputs: Jet kinematics (pT, eta, phi, mass, deepJet/PNet b-tag scores),
#     candidate quadjet pairings, and event-level variables.
#   - Training Schedule: EarlyStopStep with validation benchmarking.
#   - Precision: Mixed precision (bf16 or fp16).
#   - Hardware Allocation: 1 GPU per training instance (SLURM partition: GPU-shared / work).
#
# INPUTS:
#   - Dedicated JCMs: {out_b1}jetCombinatoricModel_SB_mix_v{m}.yml (From B_1)
#   - Detector Inputs: {out_c1}inputs/classifier_inputs_ttHbb.json (From C_1)
#   - Mixed Inputs: {out_c1}inputs/classifier_inputs_mixeddata_ttHbb_v{m}.json (From C_1)
#
# OUTPUTS:
#   - Trained Model Checkpoints: {out_c}models/mix_{m}/model.pt
#   - Completion Tokens: {out_c}models/mix_{m}/train.done
#
# EXECUTION ENVIRONMENT:
#   - Cluster: falcon (GPU partition work/light) or bridges2 (GPU-shared)
#   - Container: /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest
# ==============================================================================

import os
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# ── Stage C Configuration ─────────────────────────────────────────────────────
fvt_cfg = resolve_config_section(config, primary_key='phase_e_fvt', fallback_keys=['fvt', 'fvt_classifier'])
for k, v in fvt_cfg.items():
    if k != 'output_path':
        config[k] = v

n_models = int(config.get('n_models', config.get('n_subsamples', 16)))
MIX_INDICES = list(range(n_models))

out_c = f"{out}bkg_syst_C_FvT/"
os.makedirs(out_c, exist_ok=True)
models_dir = f"{out_c}models/"
os.makedirs(models_dir, exist_ok=True)
configs_dir = f"{out_c}configs/"
os.makedirs(configs_dir, exist_ok=True)

# Dedicated JCM path template from Stage B_1
out_b1 = f"{out}bkg_syst_B_1_computeJCM/"
config.setdefault('jcm_template', os.path.join(out_b1, "jetCombinatoricModel_SB_mix_v{m}.yml"))

# Inputs from Stage C_1
out_c1 = f"{out}bkg_syst_C_1_inputs/inputs/"
nominal_ci_json = os.path.join(out_c1, "classifier_inputs_ttHbb.json")
mixed_ci_template = os.path.join(out_c1, "classifier_inputs_mixeddata_ttHbb_v{m}.json")

# Container & SLURM Resources
CLASSIFIER_CONTAINER = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest"
INIT = "set -e && set +u && { [ -f /entrypoint.sh ] && source /entrypoint.sh || true; } && set -u && export PYTHONUNBUFFERED=1 && { ulimit -n 65536 2>/dev/null || true; }"

is_bridges2 = any(k in os.uname().nodename for k in ['bridges2', 'psc', 'ocean']) or os.path.exists('/ocean/projects/phy260026p')
default_gpu_partition = "GPU-shared" if is_bridges2 else "work"
default_gpu_qos = "" if is_bridges2 else "light"
default_gpu_gres = "gpu:1" if is_bridges2 else "mps:25"
default_gpu_mem = 22000 if is_bridges2 else 16000
default_gpu_account = "phy260026p" if is_bridges2 else ""

localrules: all_bkg_syst_C_2, create_fvt_train_config

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_bkg_syst_C_2:
    input:
        expand(f"{models_dir}mix_{{m}}/train.done", m=MIX_INDICES)

# ── Rule: Generate Dynamic Subsample Training Configuration ───────────────────
rule create_fvt_train_config:
    input:
        jcm_file = lambda w: config['jcm_template'].format(m=w.m),
    output:
        f"{configs_dir}train_mix_{{m}}.yml"
    params:
        mix = "{m}",
        out_model_dir = lambda w: f"{models_dir}mix_{w.m}/",
        eos_base = config.get('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/Run2"),
        epochs = config.get('epochs', 10),
        batch_size = config.get('batch_size', 1024),
        kfolds = max(2, int(config.get('kfolds', 3))),
        precision = config.get('precision', "fp16"),
        multisample_ds = config.get('multisample_dataset_name', 'mixeddata_4b'),
    run:
        m = wildcards.m
        mixed_ci = mixed_ci_template.format(m=m)
        jcm_path = input.jcm_file
        
        # Dataset configuration incorporating dedicated JCM, mixeddata_4b (subsample+PSttbar), and detector 3b+ttbar
        dataset_options = [
            "--metadata coffea4bees/metadata/datasets/",
            "--max-workers 8",
            "--data-source detector mixed",
            "--no-detector-4b",
            f"--data-mixed-name {params.multisample_ds}",
            f"--data-mixed-samples {m}",
            f'--JCM-weight "" {jcm_path}@@JCM_weights',
            f'--friends "" {nominal_ci_json}@@HCR_input {mixed_ci}@@HCR_input',
        ]
        
        train_cfg = {
            "main": {
                "module": "train",
                "option": [
                    "--max-loaders 2",
                    "--max-trainers 2",
                    "--device cuda cpu",
                ]
            },
            "data": [
                {
                    "module": "HCR.FvT.TrainBaseline",
                    "option": dataset_options
                }
            ],
            "model": [
                {
                    "module": "HCR.FvT.baseline.Train",
                    "option": [
                        f"--kfolds {params.kfolds}",
                        "--kfold-seed FvT random",
                        "--training EarlyStopStep",
                        {"epoch": params.epochs, "bs_init": params.batch_size},
                        f"--precision {params.precision}",
                    ]
                }
            ],
            "setting": [
                {
                    "module": "IO",
                    "option": [
                        {"output": os.path.abspath(params.out_model_dir)}
                    ]
                },
                {
                    "module": "Monitor",
                    "option": [
                        {"address": f"barista-monitor-mix{m}"}
                    ]
                }
            ]
        }
        
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(train_cfg, f, default_flow_style=False)

# ── Rule: Train FvT Model on GPU ──────────────────────────────────────────────
rule train_fvt_subsample:
    input:
        train_cfg = f"{configs_dir}train_mix_{{m}}.yml",
        jcm_file = lambda w: config['jcm_template'].format(m=w.m),
    output:
        flag = f"{models_dir}mix_{{m}}/train.done",
    log:
        f"{models_dir}mix_{{m}}/logs/train.log"
    container: CLASSIFIER_CONTAINER
    resources:
        slurm_partition = config.get('gpu_partition', default_gpu_partition),
        qos = config.get('gpu_qos', default_gpu_qos),
        mem_mb = config.get('gpu_mem', default_gpu_mem),
        cpus_per_task = 4,
        gres = config.get('gpu_gres', default_gpu_gres),
        slurm_account = config.get('slurm_account', default_gpu_account),
        runtime = 480,
    params:
        init = INIT,
        out_model_dir = f"{models_dir}mix_{{m}}/",
    shell:
        """
        {params.init} && \
        mkdir -p {params.out_model_dir} $(dirname {log}) && \
        set +u && \
        if [ -z "$X509_USER_PROXY" ] && [ -f ./proxy/x509_proxy ]; then \
            export X509_USER_PROXY="$PWD/proxy/x509_proxy"; \
        fi && \
        set -u && \
        CLASSIFIER_CONFIG_PATHS=coffea4bees python -m src.classifier.task.main \
            from {input.train_cfg} \
            -flag debug 2>&1 | tee {log}
        test ${{PIPESTATUS[0]}} -eq 0
        touch {output.flag}
        """
