import os
import copy
import yaml

def _deep_merge(base, overlay):
    """Recursively merge overlay dictionary into base dictionary."""
    if not overlay or not isinstance(overlay, dict):
        return base
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

def _dump_yaml(data, output_file):
    """Write dictionary to YAML file only if content changed, preserving mtime."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                if f.read() == new_content:
                    return output_file
        except Exception:
            pass
    with open(output_file, 'w') as f:
        f.write(new_content)
    return output_file

def stage_phaseA_3_configs(config, out_a3):
    """
    Generate unified runtime config for Stage A_3 (subsample processing: classifier inputs +
    JCM histograms + SvB friend trees) into {out_a3}configs/.
    Only non-default parameters specified in config['phaseA_3'] are merged on top.
    """
    phaseA_3_cfg = config.get('phaseA_3') or {}
    channel = config.get('channel', 'ttHbb')
    configs_dir = os.path.join(out_a3, "configs")
    default_processor = config.get('analysis_processor') or (config.get('analysis_config', {}) or {}).get('processor') or f"coffea4bees/analysis/processors/processor_{channel}.py"

    classifier_inputs_base = config['classifier_inputs_base']     # set by bkg_syst_common
    friend_base = config['mixeddata_friend_base']

    process_subsamples_cfg = {
        "processor": default_processor,
        "weights": f"coffea4bees/metadata/weights/weights_{channel}.yml",
        "dataset_location": "coffea4bees/metadata/datasets/",
        "runner": {
            "workers": 4,
            "min_workers": 25,
            "max_workers": 200,
            "worker_memory": "6GB",
            "friend_base": friend_base,
            "write_coffea_output": True,
        },
        "config": {
            "blind": False,
            "apply_FvT": False,
            "apply_JCM": False,
            "apply_trigWeight": False,
            "apply_btagSF": True,
            "apply_boosted_veto": False,
            "run_SvB": True,
            # the nominal roast's SvB model (inputs.SvB_model), not weights_<channel>.yml's
            "SvB_MA": [{"path": config['mixed_svb_model'], "name": config.get('mixed_svb_model_name', "Final")}],
            "top_reconstruction": "fast",
            "fill_histograms": True,
            "hist_cuts": ["pass_nSelJets_gt6"],
            "make_classifier_input": classifier_inputs_base,
            "make_friend_SvB": friend_base,
        }
    }
    if isinstance(phaseA_3_cfg, dict):
        if 'runner' in phaseA_3_cfg and isinstance(phaseA_3_cfg['runner'], dict):
            _deep_merge(process_subsamples_cfg['runner'], phaseA_3_cfg['runner'])
        if 'config' in phaseA_3_cfg and isinstance(phaseA_3_cfg['config'], dict):
            _deep_merge(process_subsamples_cfg['config'], phaseA_3_cfg['config'])
        for k in ['worker_memory', 'workers', 'min_workers', 'max_workers']:
            if k in phaseA_3_cfg:
                process_subsamples_cfg['runner'][k] = phaseA_3_cfg[k]
        for k, v in phaseA_3_cfg.items():
            if k not in ['runner', 'config', 'worker_memory', 'workers', 'min_workers', 'max_workers'] and not isinstance(v, dict):
                process_subsamples_cfg['config'][k] = v

    process_subsamples_file = _dump_yaml(process_subsamples_cfg, os.path.join(configs_dir, f"process_subsamples_mixeddata_{channel}.yml"))

    return {
        "process_subsamples": process_subsamples_file,
    }


def stage_phaseC_configs(config, out_c):
    """
    Generate concrete per-subsample classifier workflow configs into {out_c}models/mix_{m}/.
    Writes:
      - {out_c}models/mix_{m}/wfs/train.yml
      - {out_c}models/mix_{m}/wfs/evaluate.yml
      - {out_c}models/mix_{m}/common.yml
    Also registers train_templates and eval_templates into config for barista's generic Snakefile.
    """
    channel = config.get('channel', 'ttHbb')
    out = config.get('output_path', 'output/ttHbb_bkg_syst/')
    if not out.endswith('/'):
        out += '/'
    n_models = int(config.get('n_models', config.get('n_subsamples', 16)))
    eos_base = config['eos_base']                                 # set by bkg_syst_common
    mix_name = config.get("mix_name", "ttHbb_bkg_syst")

    nominal_ci = config.get(
        'nominal_classifier_inputs',
        'coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json'
    )

    mixed_ci_template = config.get(
        'mixed_classifier_inputs_template',
        f"{out}bkg_syst_A_3_process_subsamples/histAll_{channel}_mixeddata_v{{m}}.json"
    )

    jcm_template = config.get(
        'jcm_template',
        f"{out}bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{{m}}.yml"
    )

    raw_train_wf = config.get("fvt_train_workflow", config.get("train_workflow", {}))
    raw_eval_wf = config.get("fvt_eval_workflow", config.get("eval_workflow", {}))

    common_cfg = config.get("classifier_setting", [
        {
            "module": "ml.Training",
            "option": [{"precision": config.get("precision", "fp16")}]
        }
    ])
    if isinstance(common_cfg, list):
        common_data = {"setting": common_cfg}
    elif isinstance(common_cfg, dict):
        common_data = common_cfg if "setting" in common_cfg else {"setting": common_cfg}
    else:
        common_data = {"setting": []}

    train_templates = config.setdefault("train_templates", {})
    eval_templates = config.setdefault("eval_templates", {})

    staged_configs = {}

    def _format_structure(obj, mapping):
        if isinstance(obj, str):
            res = obj
            for k, v in mapping.items():
                res = res.replace(f"{{{k}}}", str(v))
            return res
        elif isinstance(obj, dict):
            return {k: _format_structure(v, mapping) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_format_structure(elem, mapping) for elem in obj]
        return obj

    for m in range(n_models):
        model_dir = os.path.join(out_c, f"models/mix_{m}")
        wfs_dir = os.path.join(model_dir, "wfs")
        os.makedirs(wfs_dir, exist_ok=True)

        mapping = {
            "mix": str(m),
            "jcm": jcm_template.format(m=m),
            "mixed_ci": mixed_ci_template.format(m=m),
            "nominal_ci": nominal_ci,
        }

        train_file = os.path.join(wfs_dir, "train.yml")
        eval_file = os.path.join(wfs_dir, "evaluate.yml")
        common_file = os.path.join(model_dir, "common.yml")

        _dump_yaml(_format_structure(raw_train_wf, mapping), train_file)
        _dump_yaml(_format_structure(raw_eval_wf, mapping), eval_file)
        _dump_yaml(common_data, common_file)

        train_templates[model_dir] = f"model: {eos_base}/classifier/{mix_name}_v{m}"
        eval_templates[model_dir] = f"model: {eos_base}/classifier/{mix_name}_v{m}, FvT: {eos_base}/friend/FvT/{mix_name}_v{m}"

        staged_configs[m] = {
            "train": train_file,
            "eval": eval_file,
            "common": common_file,
        }

    return staged_configs


def stage_phaseF_1_configs(config, out_f1):
    """
    Generate all effective runtime configs for Stage F_1 into {out_f1}closure_v{m}/.
    Creates concrete, self-contained analysis_config_data.yml for each subsample m,
    eliminating runtime dynamic YAML generation in Snakemake run: blocks.
    """
    phase_f_cfg = config.get('phase_f_1', config.get('bkg_syst_F_1', config.get('phase_f', {})))
    channel = config.get('channel', 'ttHbb')
    n_subsamples = int(config.get('n_subsamples', config.get('n_models', 16)))
    out = config.get('output_path', 'output/ttHbb_bkg_syst/')
    if not out.endswith('/'):
        out += '/'
    mix_name = config.get('mix_name', f"{channel}_bkg_syst")
    is_test = config.get('test', False)

    data_configs = {}
    for m in range(n_subsamples):
        closure_dir = os.path.join(out_f1, f"closure_v{m}")
        jcm_file = os.path.join(out, f"bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}.yml")
        fvt_friend = os.path.join(out, f"bkg_syst_C_FvT/friends/friends_FvT_{mix_name}_v{m}.json@@FvT")

        runner_dict = {
            "workers": 4,
            "condor": True,
            "shared_dask": True,
            "run_performance": True,
            "worker_memory": "4GB",
            "dataset_location": config.get('dataset_location', "coffea4bees/metadata/datasets/"),
            "datasets_file": "coffea4bees/metadata/datasets/data.yml",
            "weights_file": config.get('weights_file', f"coffea4bees/metadata/weights/weights_{channel}.yml"),
        }
        if is_test:
            runner_dict["condor"] = False
            runner_dict["shared_dask"] = False
            runner_dict["workers"] = 2
            runner_dict["chunksize"] = config.get("chunksize", 1000)
            runner_dict["maxchunks"] = config.get("maxchunks", 1)

        data_cfg = {
            "processor": config.get('analysis_processor') or f"coffea4bees/analysis/processors/processor_{channel}.py",
            "weights_file": config.get('weights_file', f"coffea4bees/metadata/weights/weights_{channel}.yml"),
            "runner": runner_dict,
            "config": {
                "blind": False,
                "apply_FvT": True,
                "apply_JCM": True,
                "JCM_file": jcm_file,
                "friends": {
                    "trigWeight": config.get('trigweights_file', "coffea4bees/metadata/friends/trigweights_Run2_v2.json@@trigWeight"),
                    "SvB_MA": config['data_svb_friend'],          # inputs.SvB (bkg_syst_common)
                    "FvT": fvt_friend,
                },
                "apply_trigWeight": True,
                "apply_btagSF": True,
                "apply_boosted_veto": False,
                "run_SvB": True,
                "SvB_MA": True,
                "top_reconstruction": "fast",
                "plot_ttbar_with_weights": True,
                "candidates_selection_cfg": f"coffea4bees/analysis/metadata/candidates_selection_thresholds_{channel}.yml",
                "fill_histograms": True,
                "hist_cuts": ["pass_nSelJets_gt6"],
            }
        }
        if 'runner' in phase_f_cfg and isinstance(phase_f_cfg['runner'], dict):
            _deep_merge(data_cfg['runner'], phase_f_cfg['runner'])
        if 'config' in phase_f_cfg and isinstance(phase_f_cfg['config'], dict):
            _deep_merge(data_cfg['config'], phase_f_cfg['config'])

        cfg_path = _dump_yaml(data_cfg, os.path.join(closure_dir, "analysis_config_data.yml"))
        data_configs[m] = cfg_path

    return {
        "data_configs": data_configs,
    }

