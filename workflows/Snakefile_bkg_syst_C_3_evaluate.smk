# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_3_evaluate.smk
#
# Stage C_3: High-Throughput FvT Model Evaluation & Friend Tree Generation (GPU)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Evaluates the 16 trained FvT models (mix_0..mix_15) over 3-tag collision data
# to produce per-event reweighting friend trees for the ttH(bb) background
# systematics pipeline.
#
# FRIEND TREE SCHEMA & BRANCHES:
# The evaluation computes neural network classifier predictions for each event:
#   - `FvT`: Primary multijet transfer weight (ratio of 4b to 3b probability).
#   - `d3_to_t4`: Probability transfer score mapping 3b multijet to 4b ttbar.
#   - `d3_to_t3`: Probability transfer score mapping 3b multijet to 3b ttbar.
#
# EVALUATION PIPELINE:
#   1. Configuration Staging:
#      - Dynamically generates evaluation YAML per model using high-throughput
#        batching (`batch_eval: 65536`) to maximize GPU tensor core utilization.
#   2. Batched Inference (`src.classifier.task.main`):
#      - Evaluates the trained checkpoint over 3b Data across Run 2 eras.
#      - Streams resulting branches directly into ROOT friend trees on EOS:
#        {eos_base}/friend/FvT/{mix_name}_v{m}/
#   3. Friend Manifest Extraction (`extract_friend_manifest.py`):
#      - Parses the evaluation `result.json` and produces a clean JSON manifest:
#        {out_c}friends/friends_FvT_{mix_name}_v{m}.json
#      - This JSON is subsequently used by coffea processors in Stage C_4 and
#        Stage F to attach the friend tree during histogram filling.
#
# INPUTS:
#   - Trained Model Checkpoint: {out_c}models/mix_{m}/train.done (From C_2)
#   - Evaluation Workflow Template: {out_c}configs/evaluate_mix_{m}.yml
#
# OUTPUTS:
#   - Friend Tree ROOT Files on EOS: {eos_base}/friend/FvT/{mix_name}_v{m}/*.root
#   - Friend Manifest JSON: {out_c}friends/friends_FvT_{mix_name}_v{m}.json
#   - Completion Tokens: {out_c}models/mix_{m}/evaluate.done
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

mix_name = config.get('mix_name', "ttHbb_bkg_syst")
out_c = f"{out}bkg_syst_C_FvT/"
os.makedirs(out_c, exist_ok=True)
models_dir = f"{out_c}models/"
os.makedirs(models_dir, exist_ok=True)
friends_dir = f"{out_c}friends/"
os.makedirs(friends_dir, exist_ok=True)
configs_dir = f"{out_c}configs/"
os.makedirs(configs_dir, exist_ok=True)

eos_base = config.get('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/Run2")

out_c1 = f"{out}bkg_syst_C_1_inputs/inputs/"
nominal_ci_json = config.get('nominal_classifier_inputs', os.path.join(out_c1, "classifier_inputs_ttHbb_stitched.json"))
if not os.path.exists(nominal_ci_json) and os.path.exists(os.path.join(out_c1, "classifier_inputs_ttHbb_stitched.json")):
    nominal_ci_json = os.path.join(out_c1, "classifier_inputs_ttHbb_stitched.json")

RUN2_ERAS = {
    "UL16_preVFP": ["C", "D", "E", "F"],
    "UL16_postVFP": ["F", "G", "H"],
    "UL17": ["B", "C", "D", "E", "F"],
    "UL18": ["A", "B", "C", "D"],
}
RUN2_YEARS = ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]
_config_years = config.get('years', None)
if _config_years is not None:
    if isinstance(_config_years, str):
        _config_years = [_config_years]
    RUN2_YEARS = [y for y in RUN2_YEARS if y in _config_years]
    RUN2_ERAS = {y: RUN2_ERAS[y] for y in RUN2_YEARS if y in RUN2_ERAS}

# Container & SLURM Resources
CLASSIFIER_CONTAINER = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest"
INIT = "set -e && set +u && { [ -f /entrypoint.sh ] && source /entrypoint.sh || true; } && set -u && export PYTHONUNBUFFERED=1 && { ulimit -n 65536 2>/dev/null || true; }"

is_bridges2 = any(k in os.uname().nodename for k in ['bridges2', 'psc', 'ocean']) or os.path.exists('/ocean/projects/phy260026p')
default_eval_partition = "GPU-shared" if is_bridges2 else "work"
default_eval_qos = "" if is_bridges2 else "light"
default_eval_gres = "gpu:1" if is_bridges2 else "mps:25"
default_eval_mem = 22000 if is_bridges2 else 24000
default_eval_account = "phy260026p" if is_bridges2 else ""

localrules: all_bkg_syst_C_3, create_fvt_eval_config

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_bkg_syst_C_3:
    input:
        expand(f"{models_dir}mix_{{m}}/evaluate.done", m=MIX_INDICES),
        expand(f"{friends_dir}friends_FvT_{mix_name}_v{{m}}.json", m=MIX_INDICES)

# ── Rule: Generate Dynamic Subsample Evaluation Configuration ─────────────────
rule create_fvt_eval_config:
    input:
        train_done = f"{models_dir}mix_{{m}}/train.done"
    output:
        f"{configs_dir}evaluate_mix_{{m}}.yml"
    params:
        mix = "{m}",
        eos_base = eos_base,
        mix_name = mix_name,
        batch_eval = config.get('batch_eval', 65536),
        precision = config.get('precision', 'fp16'),
    run:
        m = wildcards.m
        eval_cfg = {
            "main": {
                "module": "evaluate",
                "option": [
                    "--max-evaluators 3",
                    "--device cuda cpu",
                ]
            },
            "dataset": [
                {
                    "module": "HCR.FvT.Eval",
                    "option": [
                        "--metadata coffea4bees/metadata/datasets/",
                        "--max-workers 4",
                        "--data-source detector",
                        f'--friends "" {nominal_ci_json}@@HCR_input',
                    ]
                }
            ],
            "model": [
                {
                    "module": "HCR.FvT.baseline.Eval",
                    "option": [
                        "--models",
                        "Final",
                        os.path.abspath(f"{models_dir}mix_{m}/result.json")
                    ]
                }
            ],
            "analysis": [
                {
                    "module": "kfold.Merge",
                    "option": [
                        "--name FvT",
                        "--step 100000",
                        "--workers 4",
                        "--clean"
                    ]
                }
            ],
            "setting": [
                {
                    "module": "IO",
                    "option": [
                        {"output": f"{params.eos_base}/friend/FvT/{params.mix_name}_v{m}/"}
                    ]
                },
                {
                    "module": "Monitor",
                    "option": [
                        {"enable": False}
                    ]
                },
                {
                    "module": "cms.CollisionData",
                    "option": [
                        {
                            "eras": RUN2_ERAS,
                            "years": RUN2_YEARS,
                        }
                    ]
                },
                {
                    "module": "HCR.InputBranch",
                    "option": [
                        {"feature_ancillary": ["xW", "nSelJets", "xbW", "year"]}
                    ]
                },
                {
                    "module": "ROOT",
                    "option": [
                        {"friend_allow_missing": False}
                    ]
                },
                {
                    "module": "ml.Training",
                    "option": [
                        {"precision": params.precision}
                    ]
                },
                {
                    "module": "ml.DataLoader",
                    "option": [
                        {"batch_eval": params.batch_eval},
                        {"num_workers": 4},
                        {"optimize_sliceable_dataset": True}
                    ]
                }
            ]
        }
        
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], 'w') as f:
            yaml.dump(eval_cfg, f, default_flow_style=False)

# ── Rule: Evaluate FvT Model on GPU & Extract Friend Manifest ─────────────────
rule evaluate_fvt_subsample:
    input:
        eval_cfg = f"{configs_dir}evaluate_mix_{{m}}.yml",
        train_done = f"{models_dir}mix_{{m}}/train.done",
    output:
        flag = f"{models_dir}mix_{{m}}/evaluate.done",
        json = f"{friends_dir}friends_FvT_{mix_name}_v{{m}}.json",
    log:
        f"{models_dir}mix_{{m}}/logs/evaluate.log"
    container: CLASSIFIER_CONTAINER
    resources:
        slurm_partition = config.get('eval_partition', default_eval_partition),
        qos = config.get('eval_qos', default_eval_qos),
        mem_mb = config.get('eval_mem', default_eval_mem),
        cpus_per_task = 4,
        gres = config.get('eval_gres', default_eval_gres),
        slurm_account = config.get('slurm_account', default_eval_account),
        runtime = 240,
    params:
        init = INIT,
        fvt_friend_dir = lambda wildcards: f"{eos_base}/friend/FvT/{mix_name}_v{wildcards.m}",
        eos_base = eos_base,
    shell:
        """
        {params.init} && \
        mkdir -p $(dirname {output.flag}) $(dirname {output.json}) $(dirname {log}) && \
        set +u && \
        if [ -z "$X509_USER_PROXY" ] && [ -f ./proxy/x509_proxy ]; then \
            export X509_USER_PROXY="$PWD/proxy/x509_proxy"; \
        fi && \
        set -u && \
        CLASSIFIER_CONFIG_PATHS=coffea4bees python -m src.classifier.task.main \
            from {input.eval_cfg} \
            -flag debug 2>&1 | tee {log}
        test ${{PIPESTATUS[0]}} -eq 0

        # Copy result.json and extract friend manifest
        TMP_RES=$(mktemp -p . --suffix=.json)
        if [[ "{params.eos_base}" == root://* ]]; then
            xrdcp -f '{params.fvt_friend_dir}/result.json' "$TMP_RES"
        else
            cp -f '{params.fvt_friend_dir}/result.json' "$TMP_RES"
        fi
        python coffea4bees/scripts/extract_friend_manifest.py "$TMP_RES" "{output.json}" 2>&1 | tee -a {log}
        test ${{PIPESTATUS[0]}} -eq 0
        rm -f "$TMP_RES"
        touch {output.flag}
        """
