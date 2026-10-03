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

def stage_phaseA_3_configs(config, out_a3, subsamples):
    """
    Generate the runtime configs for Stage A_3 (subsample processing: classifier inputs + JCM
    histograms + SvB friend trees) into {out_a3}configs/, one per subsample.
    Only non-default parameters specified in config['phaseA_3'] are merged on top.

    One per subsample because every subsample contains the SAME ttbar pseudodata files: with one
    friend directory for all of them, the 16 parallel jobs wrote, merged and cleaned up friend
    trees of the same pseudodata files in the same place (one job's cleanup removed another's
    files: "Unable to remove .../TTToSemiLeptonic_UL18/HCR_input_..._picoAOD_PSData.root").
    Each subsample now has its own <base>/v<k>/ (and the pseudodata friends exist once per
    subsample -- small).
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
            # the A_3 pass (SvB on the fly + friend dumps) was OOM-killed at 6 GB
            "worker_memory": config.get('worker_memory', "8GB"),
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

    files = {}
    for v in subsamples:
        cfg = copy.deepcopy(process_subsamples_cfg)
        # per-subsample friend directories (see the docstring); friend_base is a runner key, so it
        # cannot go through runner.py --config-overrides
        cfg['runner']['friend_base'] = f"{cfg['runner']['friend_base'].rstrip('/')}/v{v}/"
        for key in ('make_classifier_input', 'make_friend_SvB'):
            if cfg['config'].get(key):
                cfg['config'][key] = f"{cfg['config'][key].rstrip('/')}/v{v}/"
        files[str(v)] = _dump_yaml(cfg, os.path.join(configs_dir, f"process_subsamples_mixeddata_{channel}_v{v}.yml"))

    return {
        "process_subsamples": files,
    }


def build_classifier_metadata(repo_dir, dataset_file, out_dir, name):
    """C's classifier merges every YAML of its --metadata directory and looks the mixed samples up
    by name: copy the repo's dataset YAMLs except those defining `name`, plus dataset_file.
    Returns the skipped file names."""
    import glob, shutil
    os.makedirs(out_dir, exist_ok=True)
    skipped = []
    for path in sorted(glob.glob(os.path.join(repo_dir, "*.yml"))):
        with open(path) as f:
            keys = set(yaml.safe_load(f) or {})
        if name in keys:
            skipped.append(os.path.basename(path))
            continue
        shutil.copy(path, out_dir)
    shutil.copy(dataset_file, out_dir)
    return skipped


def _fetch_classifier_metadata(config, handoff_eos):
    """On the GPU host (C run from the EOS handoff, no local A_2), rebuild A_2's classifier metadata
    directory from the repo YAMLs + the multi-sample dataset the AB handoff published."""
    import subprocess, tempfile
    meta_dir = config['classifier_metadata']
    if os.path.isdir(meta_dir) or not handoff_eos:
        return
    name = config.get('multisample_dataset_name', "mixeddata_4b")
    src = f"{handoff_eos}/datasets/{name}.yml"
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, f"{name}.yml")
        cmd = ["xrdcp", "-f", src, dst] if src.startswith("root://") else ["cp", "-f", src, dst]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as e:
            # dry runs without EOS access: C's train jobs fail on the missing directory instead
            print(f"WARNING: could not fetch {src} for {meta_dir}: {e}")
            return
        build_classifier_metadata(config.get('dataset_location', "coffea4bees/metadata/datasets/"),
                                  dst, meta_dir, name)


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
    phase_c_cfg = config.get('phase_e_fvt', config.get('phaseC', config.get('phase_c', {}))) or {}
    n_models = int(config.get('n_models', config.get('n_subsamples', phase_c_cfg.get('n_models', 16))))
    eos_base = config['eos_base']                                 # set by bkg_syst_common
    mix_name = config.get("mix_name", "ttHbb_bkg_syst")
    per_year_jcm = bool(config.get('per_year_jcm', (config.get('phaseB_1', {}) or {}).get('per_year_jcm', False)))
    raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
    if isinstance(raw_years, str):
        years = [str(y).strip() for y in raw_years.split() if str(y).strip()]
    else:
        years = [str(y) for y in raw_years]

    nominal_ci = phase_c_cfg.get(
        'nominal_classifier_inputs',
        config.get('nominal_classifier_inputs', 'coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json')
    )

    handoff_eos = (config.get('handoff') or {}).get('eos_base')
    if handoff_eos:
        handoff_eos = str(handoff_eos).rstrip('/')

    default_subsample_dir = "bkg_syst_A_3_process_subsamples"

    local_ci_sample = f"{out}{default_subsample_dir}/histAll_{channel}_mixeddata_v0.json"
    if not os.path.exists(local_ci_sample):     # A ran elsewhere (the AB handoff)
        _fetch_classifier_metadata(config, handoff_eos)
    if handoff_eos and not os.path.exists(local_ci_sample) and not phase_c_cfg.get('mixed_classifier_inputs_template') and 'mixed_classifier_inputs_template' not in config:
        mixed_ci_template = f"{handoff_eos}/classifier_inputs/histAll_{channel}_mixeddata_v{{m}}.json"
    else:
        mixed_ci_template = phase_c_cfg.get(
            'mixed_classifier_inputs_template',
            config.get(
                'mixed_classifier_inputs_template',
                f"{out}{default_subsample_dir}/histAll_{channel}_mixeddata_v{{m}}.json"
            )
        )

    local_jcm_sample = f"{out}bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v0.yml"
    if handoff_eos and not os.path.exists(local_jcm_sample) and not phase_c_cfg.get('jcm_template') and 'jcm_template' not in config:
        jcm_template = f"{handoff_eos}/JCM/jetCombinatoricModel_SB_mix_v{{m}}.yml"
    else:
        jcm_template = phase_c_cfg.get(
            'jcm_template',
            config.get(
                'jcm_template',
                f"{out}bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{{m}}.yml"
            )
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

        model_eos = f"{eos_base}/classifier/{mix_name}_v{m}"
        fvt_eos = f"{eos_base}/friend/FvT/{mix_name}_v{m}"

        mapping = {
            "mix": str(m),
            # A_2's metadata directory: this run's multi-sample dataset is the only one named mixed_name
            "metadata": config['classifier_metadata'],
            "mixed_name": config.get('multisample_dataset_name', "mixeddata_4b"),
            "jcm": jcm_template.format(m=m),
            "mixed_ci": mixed_ci_template.format(m=m),
            "nominal_ci": nominal_ci,
            "model": model_eos,
            "FvT": fvt_eos,
        }
        for y in years:
            local_jcm_y = os.path.join(out, f"bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}_{y}.yml")
            if handoff_eos and not os.path.exists(local_jcm_y):
                mapping[f"jcm_{y}"] = f"{handoff_eos}/JCM/jetCombinatoricModel_SB_mix_v{m}_{y}.yml"
            else:
                mapping[f"jcm_{y}"] = local_jcm_y

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
    Generate the consolidated runtime config for Stage F_1 into {out_f1}.
    Creates concrete, self-contained analysis_config_mixeddata_bkgs.yml evaluating
    all 16 subsamples simultaneously in a single pass over 3-tag collision data.
    """
    phase_f_cfg = config.get('phase_f_1', config.get('bkg_syst_F_1', config.get('phase_f', {})))
    channel = config.get('channel', 'ttHbb')
    n_subsamples = int(config.get('n_subsamples', config.get('n_models', 16)))
    out = config.get('output_path', config.get('out', 'output/ttHbb_bkg_syst/'))
    if not out.endswith('/'):
        out += '/'
    mix_name = config.get('mix_name', f"{channel}_bkg_syst")
    is_test = config.get('test', False)
    per_year_jcm = bool(config.get('per_year_jcm', (config.get('phaseB_1', {}) or {}).get('per_year_jcm', False)))
    raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
    if isinstance(raw_years, str):
        years = [str(y).strip() for y in raw_years.split() if str(y).strip()]
    else:
        years = [str(y) for y in raw_years]

    subsample_names = [f"v{m}" for m in range(n_subsamples)]

    handoff_eos = (config.get('handoff') or {}).get('eos_base')
    if handoff_eos:
        handoff_eos = str(handoff_eos).rstrip('/')

    # Build multi-subsample JCM dictionary
    jcm_file = {}
    for m in range(n_subsamples):
        v_name = f"v{m}"
        if per_year_jcm:
            jcm_file[v_name] = {}
            for y in years:
                loc_jcm = os.path.join(out, f"bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}_{y}.yml")
                if os.path.exists(loc_jcm) or not handoff_eos:
                    jcm_file[v_name][y] = loc_jcm
                else:
                    jcm_file[v_name][y] = f"{handoff_eos}/JCM/jetCombinatoricModel_SB_mix_v{m}_{y}.yml"
        else:
            loc_jcm = os.path.join(out, f"bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}.yml")
            if os.path.exists(loc_jcm) or not handoff_eos:
                jcm_file[v_name] = loc_jcm
            else:
                jcm_file[v_name] = f"{handoff_eos}/JCM/jetCombinatoricModel_SB_mix_v{m}.yml"

    # Build friends dictionary
    friends_dict = {
        "trigWeight": config.get('trigweights_file', "coffea4bees/metadata/friends/trigweights_Run2_v2.json@@trigWeight"),
        "SvB_MA": config['data_svb_friend'],                      # inputs.SvB (bkg_syst_common)
    }
    for m in range(n_subsamples):
        loc_friend = os.path.join(out, f"bkg_syst_C_FvT/friends/friends_FvT_{mix_name}_v{m}.json")
        if os.path.exists(loc_friend) or not handoff_eos:
            friends_dict[f"FvT_v{m}"] = f"{loc_friend}@@FvT"
        else:
            friends_dict[f"FvT_v{m}"] = f"{handoff_eos}/friends/friends_FvT_{mix_name}_v{m}.json@@FvT"
    friends_dict["FvT"] = friends_dict["FvT_v0"]

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
            "subsample_names": subsample_names,
            "friends": friends_dict,
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

    cfg_path = _dump_yaml(data_cfg, os.path.join(out_f1, "analysis_config_mixeddata_bkgs.yml"))

    return {
        "config": cfg_path,
        "data_configs": {0: cfg_path},
    }

