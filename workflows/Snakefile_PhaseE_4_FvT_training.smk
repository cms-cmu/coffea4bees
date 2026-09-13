# coffea4bees/workflows/Snakefile_PhaseE_2_FvT_training.smk
# Phase E_2: FvT Classifier Pipeline for Mixed Data (Option A - falcon GPU)
#
# Trains 15 FvT models (v0..v14) and evaluates FvT friend trees on 3-tag data with optimized GPU hyperparameters.
# All configurations are self-contained in workflows/config/ without depending on classifier/config/

import os
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

include: "helpers/common.smk"

# Resolve phase_e and FvT config block
fvt_cfg = resolve_config_section(config, primary_key='phase_e_fvt', fallback_keys=['fvt_classifier', 'phase_e', 'fvt'])
for k, v in fvt_cfg.items():
    if k != 'output_path':
        config[k] = v

config.setdefault('channel', "ttHbb")
config.setdefault('n_models', 16)
config.setdefault('mix_name', "3bDvTMix4bDvT")
config.setdefault('eos_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/ttHbb")
base_output_path = config.get('output_path', "output/ttHbb_mixeddata_closure/")
if not base_output_path.endswith('/'):
    base_output_path += '/'
fvt_sub = fvt_cfg.get('output_path', "FvT_training/")
if fvt_sub.startswith("output/"):
    fvt_out = os.path.join(base_output_path, "FvT_training/")
else:
    fvt_out = os.path.join(base_output_path, fvt_sub)
if not fvt_out.endswith('/'):
    fvt_out += '/'
config.setdefault('precision', "bf16")
config.setdefault('disable_benchmark', True)
# Early stopping: on by default for PhaseE so each of the 16 trainings costs only the
# epochs it needs. EarlyStopStep requires validation benchmarks, so it also forces
# Monitor on and benchmarks enabled. Set training_schedule: FixedStep to opt out.
config.setdefault('jcm_template',
    "output/ttHbb_mixeddata_stitched_closure/JCM_subsamples/jetCombinatoricModel_SB_mix_v{m}.yml")
config.setdefault('training_schedule', "EarlyStopStep")
config.setdefault('early_stop', {})
_es = config['early_stop'] or {}
config['early_stop_opts'] = (
    {f"es_{k}": v for k, v in _es.items()}
    if config['training_schedule'] == "EarlyStopStep" else {}
)
if config['training_schedule'] == "EarlyStopStep":
    config['disable_benchmark'] = False
config.setdefault('batch_eval', 65536)
# Classifier-input friend trees. The nominal (detector) inputs used at evaluation time MUST be
# dumped with the same processor/phase space as the mixed subsamples used for training, otherwise
# the model is evaluated outside the region it was trained on.
config.setdefault('nominal_classifier_inputs',
    "coffea4bees/metadata/datasets/classifier_inputs_ttHbb_stitched.json")
config.setdefault('mixed_classifier_inputs',
    "coffea4bees/metadata/datasets/classifier_inputs_mixeddata_ttHbb.json")

# Run 2 CollisionData metadata
RUN2_ERAS = {
    "UL16_preVFP": ["C", "D", "E", "F"],
    "UL16_postVFP": ["F", "G", "H"],
    "UL17": ["B", "C", "D", "E", "F"],
    "UL18": ["A", "B", "C", "D"],
}
RUN2_YEARS = ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]

MIX_INDICES = list(range(int(config['n_models'])))
out = fvt_out
os.makedirs(out, exist_ok=True)

localrules: all_PhaseE_4, create_fvt_train_config, create_fvt_eval_config

rule all_PhaseE_4:
    input:
        expand(f"{out}models/mix_{{m}}/train.done", m=MIX_INDICES),
        expand(f"{out}friends/friends_FvT_{config['mix_name']}_v{{m}}.json", m=MIX_INDICES)

# Dynamic creation of train configuration per mixed subsample
def _optional_cache(wildcards):
    # No rule produces these caches; treat them as optional so training can run
    # directly from ROOT (the path used for the validated v0/v1 trainings).
    p = f"{out}cache_3b/cache_v{wildcards.m}/result.json"
    return [p] if os.path.exists(p) else []

rule create_fvt_train_config:
    input:
        cache_file = _optional_cache
    output:
        f"{out}configs/train_mix_{{m}}.yml"
    params:
        mix = "{m}",
        mix_name = config['mix_name'],
        eos_base = config['eos_base'],
        epochs = config.get('epochs', 10),
        batch_size = config.get('batch_size', 1024),
        kfolds = config.get('kfolds', 3),
        offset = config.get('kfold_offset', 0),
        precision = config['precision'],
        disable_benchmark = config['disable_benchmark'],
        training_schedule = config['training_schedule'],
        early_stop_opts = config['early_stop_opts'],
        jcm_file = lambda w: config['jcm_template'].format(m=w.m),
    run:
        out_path = os.path.abspath(str(output[0]))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        m = int(params.mix)
        # Mirrors coffea4bees/classifier/config/workflows/ttHbb_mixeddata_stitched/FvT/train_v1.yml,
        # the recipe validated by the v1 closure test.
        # Per-subsample registry. The merged 16-subsample file makes the loader
        # return 0 events ("Dataset loaded 0 events"); the validated v1 recipe pairs
        # classifier_inputs_mixeddata_ttHbb_v<N>.json with --data-mixed-samples <N>.
        # Generate the per-subsample files with tmp/split_ci.py.
        mixed_ci = config['mixed_classifier_inputs'].replace(".json", f"_v{m}.json")
        # IMPORTANT: each option must be ONE packed string, exactly as in the
        # validated train_v1.yml. Splitting a multi-arg option such as
        # "--JCM-weight" (nargs=2), "--friends" or "--data-source" across separate
        # YAML list items breaks argument grouping: the mixed dataset then resolves
        # no friends and the loader dies with "Dataset loaded 0 events".
        dataset_cfg = [
            {
                "module": "HCR.FvT.TrainBaseline",
                "option": [
                    "--metadata coffea4bees/metadata/datasets/",
                    "--max-workers 20",
                    "--data-source detector mixed",
                    "--no-detector-4b",
                    "--data-mixed-name mixeddata_4b",
                    f"--data-mixed-samples {m}",
                    f'--JCM-weight "" {params.jcm_file}@@JCM_weights',
                    f'--friends "" {config["nominal_classifier_inputs"]}@@HCR_input {mixed_ci}@@HCR_input',
                ]
            },
        ]
        if input.cache_file:
            dataset_cfg.append(
                {"module": "cache", "option": ["--input", os.path.abspath(str(input.cache_file[0]))]}
            )

        train_cfg = {
            "main": {
                "module": "train",
                "option": [
                    "--max-loaders 2",
                    "--max-trainers 3"
                ]
            },
            "model": [
                {
                    "module": "HCR.FvT.baseline.Train",
                    "option": [
                        f"--kfolds {params.kfolds}",
                        "--kfold-seed FvT random",
                        f"--kfold-seed-offsets {params.offset}",
                        f"--training {params.training_schedule}",
                        {"epoch": params.epochs, "bs_init": params.batch_size,
                         **params.early_stop_opts},
                        # v1 used FixedStep for finetuning; keep it identical.
                        "--finetuning FixedStep",
                        {"epoch": 1, "bs_init": 16384},
                    ]
                }
            ],
            "dataset": dataset_cfg,
            "setting": [
                {
                    "module": "IO",
                    "option": [
                        {"output": f"{params.eos_base}/classifier/FvT/{params.mix_name}_v{params.mix}/"}
                    ]
                },
                {
                    "module": "Monitor",
                    "option": [
                        {"enable": config['training_schedule'] == "EarlyStopStep"},
                        # Default address ":10200" is a FIXED TCP port, so concurrent
                        # trainings on one node collide ("Address already in use").
                        # A non "ip:port" string selects a UNIX socket whose path gets a
                        # uuid4 appended (process.pipe_address), unique per job.
                        {"address": f"barista-monitor-mix{params.mix}"},
                    ]
                },
                {
                    "module": "cms.MC_TTbar",
                    "option": [
                        {"datasets": [
                            "TTToSemiLeptonic_stitched",
                            "TTToHadronic_stitched",
                            "TTTo2L2Nu_stitched",
                        ]}
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
                        {"friend_allow_missing": True}
                    ]
                },
                {
                    "module": "ml.DataLoader",
                    "option": [
                        {"optimize_sliceable_dataset": True}
                    ]
                },
                {
                    "module": "ml.Training",
                    "option": [
                        {"precision": params.precision},
                        {"disable_benchmark": params.disable_benchmark}
                    ]
                }
            ]
        }
        with open(out_path, "w") as f:
            # width=10000 prevents yaml from line-wrapping long option strings such as
            # --friends and --JCM-weight (nargs=2+). A wrapped string is split across
            # YAML list items, which breaks argument grouping: only the first path is
            # seen and the dataset loads 0 events. See Snakefile comment lines 107-110.
            yaml.dump(train_cfg, f, default_flow_style=False, width=10000)

# Train FvT on falcon GPU
rule train_fvt_mixed_model:
    input:
        f"{out}configs/train_mix_{{m}}.yml"
    output:
        f"{out}models/mix_{{m}}/train.done"
    log:
        f"{out}logs/train_mix_{{m}}.log"
    params:
        mix = "{m}",
        mix_name = config['mix_name'],
        eos_base = config['eos_base'],
    resources:
        slurm_partition = "work",
        qos = "light",
        mem_mb = 16000,
        cpus_per_task = 4,
        gres = "mps:25",
        runtime = 720,
    retries: 3
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        CLASSIFIER_CONFIG_PATHS=coffea4bees ./run_container classifier python -m src.classifier.task.main from {input} 2>&1 | tee {log}
        touch {output}
        """

# Dynamic creation of evaluate configuration per mixed subsample
rule create_fvt_eval_config:
    input:
        f"{out}models/mix_{{m}}/train.done"
    output:
        f"{out}configs/eval_mix_{{m}}.yml"
    params:
        mix = "{m}",
        mix_name = config['mix_name'],
        eos_base = config['eos_base'],
        precision = config['precision'],
        batch_eval = config['batch_eval'],
    run:
        out_path = os.path.abspath(str(output[0]))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        eval_cfg = {
            "main": {
                "module": "evaluate",
                "option": [
                    "--max-evaluators 3",
                    "--device cuda cpu"
                ]
            },
            "dataset": [
                {
                    "module": "HCR.FvT.Eval",
                    "option": [
                        "--metadata", "coffea4bees/metadata/datasets/",
                        "--max-workers 4",
                        "--data-source", "detector",
                        "--friends", "", f"{config['nominal_classifier_inputs']}@@HCR_input"
                    ]
                }
            ],
            "model": [
                {
                    "module": "HCR.FvT.baseline.Eval",
                    "option": [
                        "--models",
                        "Final",
                        f"{params.eos_base}/classifier/FvT/{params.mix_name}_v{params.mix}/result.json"
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
                        {"output": f"{params.eos_base}/friend/FvT/{params.mix_name}_v{params.mix}/"}
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
        with open(out_path, "w") as f:
            yaml.dump(eval_cfg, f, default_flow_style=False)

# Evaluate FvT model on 3-tag data (Option A on falcon GPU)
rule evaluate_fvt_mixed_model:
    input:
        cfg = f"{out}configs/eval_mix_{{m}}.yml",
        done = f"{out}models/mix_{{m}}/train.done",
    output:
        f"{out}friends/friends_FvT_{config['mix_name']}_v{{m}}.json"
    log:
        f"{out}logs/eval_mix_{{m}}.log"
    params:
        eos_base = config['eos_base'],
        mix_name = config['mix_name'],
        mix = "{m}",
    resources:
        slurm_partition = "work",
        qos = "light",
        mem_mb = 24000,
        cpus_per_task = 4,
        gres = "mps:25",
    retries: 3
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        CLASSIFIER_CONFIG_PATHS=coffea4bees ./run_container classifier python -m src.classifier.task.main from {input.cfg} 2>&1 | tee {log}
        TMP_RES=$(mktemp --suffix=.json)
        X509_USER_PROXY=$(pwd)/proxy/x509_proxy xrdcp -f '{params.eos_base}/friend/FvT/{params.mix_name}_v{params.mix}/result.json' "$TMP_RES"
        python3 -c "import json; data=json.load(open('$TMP_RES')); merged=data['analysis'][0]['merged']; json.dump({{'FvT': merged}}, open('{output}', 'w'))"
        rm -f "$TMP_RES"
        """
