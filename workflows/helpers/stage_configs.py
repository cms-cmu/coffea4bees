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

def stage_phaseA_1_configs(config, out_a1, jcm_model_file):
    """
    Generate all effective runtime configs for Stage A_1 into {out_a1}configs/.
    Only non-default parameters specified in config['phaseA_1'] are merged on top.
    """
    phaseA_1 = config.get('phaseA_1', config.get('bkg_syst_A_1', {}))
    channel = config.get('channel', 'ttHbb')
    dataset_name = config.get('dataset_name', f"mixeddata_{channel}_rank0_0")
    configs_dir = os.path.join(out_a1, "configs")

    # 1. Mixed-data Skimmer Config
    skimmer_cfg = {
        "runner": {
            "workers": 4,
            "min_workers": 1,
            "max_workers": 200,
            "chunksize": 100000,
            "picosize": 100000,
            "basketsize": 10000,
            "class_name": "HemiMixer",
            "data_tier": "picoAOD",
            "condor_cores": 1,
            "worker_memory": "4GB",
            "condor_transfer_input_files": ["src", "coffea4bees/"],
            "allowlist_sites": ["T2_US_Nebraska", "T2_US_Purdue", "T3_US_FNALLPC", "T3_US_NotreDame"],
        },
        "config": {
            "base_path": config.get('base_path', f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/Run2/{channel}_pz_rank0_0"),
            "subtract_ttbar_with_weights": False,
            "apply_JCM": True,
            "JCM_file": str(jcm_model_file),
            "hemi_library_yaml": "coffea4bees/skimmer/metadata/hemisphere_library_noTT.yml",
            "hemi_stats_path": "coffea4bees/skimmer/metadata",
            "use_boost_corrected_matching": False,
            "use_topk_matching": False,
            "k_neighbors": 10,
            "collision_mode": "retry",
            "default_rank": 0,
            "step": 100000,
            "skip_collections": ["notCanJet", "canJet0", "canJet1", "canJet2", "canJet3"],
            "skip_branches": [
                "btagWeight_.*", "HHSR", "ZZSR", "ZHSR", "SR", "st", "d01TruthMatch",
                "nMuon_selected", "fourTag", "threeTag", "xW", "leadStM", "nSelJets",
                "d02TruthMatch", "d12TruthMatch", "truthMatch", "pseudoTagWeight",
                "ttbarWeight", "nIsoMuons", "xt", "weight", "aveAbsEtaOth", "xWbW",
                "nAllNotCanJets", "dRjjOther", "xWt", "nPSTJets", "passXWt",
                "d03TruthMatch", "xbW", "mcPseudoTagWeight", "d23TruthMatch", "m4j",
                "sublStM", "dRjjClose", "aveAbsEta", "stNotCan", "SB",
                "selectedViewTruthMatch", "d13TruthMatch",
            ]
        }
    }
    if 'skimmer' in phaseA_1 and isinstance(phaseA_1['skimmer'], dict):
        for k, v in phaseA_1['skimmer'].items():
            if k == 'runner' and isinstance(v, dict):
                _deep_merge(skimmer_cfg['runner'], v)
            elif k == 'config' and isinstance(v, dict):
                _deep_merge(skimmer_cfg['config'], v)
            elif k in skimmer_cfg['runner']:
                skimmer_cfg['runner'][k] = copy.deepcopy(v)
            else:
                skimmer_cfg['config'][k] = copy.deepcopy(v)

    skimmer_file = _dump_yaml(skimmer_cfg, os.path.join(configs_dir, "mixeddata_skimmer_config.yml"))

    # 2. Analysis Config (Unit Weights)
    default_processor = config.get('analysis_processor') or (config.get('analysis_config', {}) or {}).get('processor') or f"coffea4bees/analysis/processors/processor_{channel}.py"
    analysis_unit_cfg = {
        "processor": default_processor,
        "runner": {
            "workers": 4,
            "worker_memory": "4GB",
            "condor": True,
            "shared_dask": True,
            "run_performance": False,
        },
        "config": {
            "blind": False,
            "apply_FvT": True,
            "apply_JCM": False,
            "apply_trigWeight": True,
            "apply_btagSF": True,
            "apply_boosted_veto": False,
            "run_SvB": False,
            "top_reconstruction": "fast",
            "fill_histograms": True,
            "hist_cuts": [],
        }
    }
    analysis_overrides = phaseA_1.get('analysis', phaseA_1.get('analysis_all', {}))
    if isinstance(analysis_overrides, dict):
        for k, v in analysis_overrides.items():
            if k == 'runner' and isinstance(v, dict):
                _deep_merge(analysis_unit_cfg['runner'], v)
            elif k == 'config' and isinstance(v, dict):
                _deep_merge(analysis_unit_cfg['config'], v)
            elif k in analysis_unit_cfg['runner']:
                analysis_unit_cfg['runner'][k] = copy.deepcopy(v)
            else:
                analysis_unit_cfg['config'][k] = copy.deepcopy(v)

    analysis_unit_file = _dump_yaml(analysis_unit_cfg, os.path.join(configs_dir, "analysis_config_mixeddata_all.yml"))

    # 3. Mixed-Data JCM Derivation Config
    jcm_fit_cfg = {
        "data4bName": "data",
        "taglabel4b": "fourTag",
        "data3bName": dataset_name,
        "taglabel3b": "fourTag",
        "taglabel3b_tt": "threeTag",
        "selJets": "selJets_noJCM.n",
        "tagJets": "tagJets_noJCM.n",
        "subtract3bTT": False,
        "ignoreTT": False,
        "ttbarProcesses": [],
        "float_t": False,
    }
    if 'jcm' in phaseA_1 and isinstance(phaseA_1['jcm'], dict):
        _deep_merge(jcm_fit_cfg, phaseA_1['jcm'])
    jcm_fit_file = _dump_yaml(jcm_fit_cfg, os.path.join(configs_dir, "jcm_mixeddata_config.yml"))

    # JCM Plot Config
    mix_process_list = [f"{dataset_name}{e}" for e in ["A", "B", "C", "D", "E", "F", "G", "H"]]
    jcm_plot_cfg = {
        "hists": {
            "data": {
                "process": "data",
                "tag": "fourTag",
                "year": "RunII",
                "label": "Four-tag Data",
                "edgecolor": "k",
                "fillcolor": "k",
            },
            "JCM": {
                "process": "JCM",
                "tag": "fourTag",
                "year": "RunII",
                "label": "JCM fit",
                "edgecolor": "r",
                "fillcolor": "r",
                "histtype": "step",
                "scalefactor": 1,
            }
        },
        "stack": {
            "MultiJet": {
                "process": mix_process_list,
                "tag": "fourTag",
                "year": "RunII",
                "fillcolor": "orange",
                "edgecolor": "k",
                "label": "Multijet (mixed)",
                "scalefactor": 1.0,
            }
        },
        "ratios": {
            "dataToBkg": {
                "numerator": {"type": "hists", "key": "data"},
                "denominator": {"type": "stack"},
                "uncertianty": "nominal",
                "color": "k",
                "marker": "o",
            },
            "dataToJCM": {
                "numerator": {"type": "hists", "key": "data"},
                "denominator": {"type": "hists", "key": "JCM"},
                "uncertianty": "nominal",
                "color": "r",
                "marker": "o",
            }
        },
        "codes": {
            "region": {"SR": 2, "SB": 1, "other": 0},
            "tag": {"threeTag": 3, "fourTag": 4, "other": 0},
        },
        "doRatio": 1,
    }
    if 'jcm_plots' in phaseA_1 and isinstance(phaseA_1['jcm_plots'], dict):
        _deep_merge(jcm_plot_cfg, phaseA_1['jcm_plots'])
    jcm_plot_file = _dump_yaml(jcm_plot_cfg, os.path.join(configs_dir, "plots_metadata_mixeddata_JCM.yml"))

    # 4. Analysis Config (with Calibrated Mixed-Data JCM)
    analysis_jcm_cfg = copy.deepcopy(analysis_unit_cfg)
    analysis_jcm_cfg["config"]["apply_JCM"] = True
    jcm_calibrated_file = os.path.join(out_a1, "JCM_2_mixeddata_inclusive", f"jetCombinatoricModel_inclusive_{channel}_mixeddata.yml")
    analysis_jcm_cfg["config"]["JCM_file"] = jcm_calibrated_file
    if 'analysis_with_jcm' in phaseA_1 and isinstance(phaseA_1['analysis_with_jcm'], dict):
        _deep_merge(analysis_jcm_cfg, phaseA_1['analysis_with_jcm'])
    analysis_jcm_file = _dump_yaml(analysis_jcm_cfg, os.path.join(configs_dir, "analysis_config_mixeddata_all_with_mixeddata_JCM.yml"))

    # 5. Closure Plots Config
    closure_plots_cfg = {
        "hists": {
            "data": {
                "process": "data",
                "tag": "fourTag",
                "label": "Four-tag data",
                "edgecolor": "k",
                "fillcolor": "k",
            }
        },
        "stack": {
            "mixeddata": {
                "process": mix_process_list,
                "tag": "fourTag",
                "fillcolor": "orange",
                "edgecolor": "k",
                "label": "Mixed data (JCM weighted)",
            }
        },
        "ratios": {
            "dataToStack": {
                "numerator": {"type": "hists", "key": "data"},
                "denominator": {"type": "stack"},
                "uncertianty": "nominal",
                "color": "k",
                "marker": "o",
            }
        },
        "doRatio": 1,
        "categories": ["inclusive"],
        "regions": ["sum", "SR", "SB"],
    }
    if 'closure_plots' in phaseA_1 and isinstance(phaseA_1['closure_plots'], dict):
        _deep_merge(closure_plots_cfg, phaseA_1['closure_plots'])
    closure_plots_file = _dump_yaml(closure_plots_cfg, os.path.join(configs_dir, "plots_metadata_mixeddata_closure.yml"))

    # 6. Comparison Plots vs Data
    vs_data_plots_cfg = copy.deepcopy(closure_plots_cfg)
    vs_data_plots_cfg["stack"]["mixeddata"]["label"] = "Mixed data (all)"
    if 'vs_data_plots' in phaseA_1 and isinstance(phaseA_1['vs_data_plots'], dict):
        _deep_merge(vs_data_plots_cfg, phaseA_1['vs_data_plots'])
    vs_data_plots_file = _dump_yaml(vs_data_plots_cfg, os.path.join(configs_dir, "plots_metadata_mixeddata_vs_data.yml"))

    return {
        "skimmer": skimmer_file,
        "analysis_unit": analysis_unit_file,
        "jcm_fit": jcm_fit_file,
        "jcm_plots": jcm_plot_file,
        "analysis_jcm": analysis_jcm_file,
        "closure_plots": closure_plots_file,
        "vs_data_plots": vs_data_plots_file,
    }


def stage_phaseA_2_configs(config, out_a2):
    """
    Generate all effective runtime configs for Stage A_2 (TTbar Pseudodata Slicing) into {out_a2}configs/.
    Loads base config from coffea4bees/skimmer/metadata/sub_sampling_MC.yml and overlays phaseA_2 parameters.
    """
    phaseA_2 = config.get('phaseA_2', config.get('phaseA_4', config.get('ttbar_psdata', {})))
    configs_dir = os.path.join(out_a2, "configs")
    base_file = phaseA_2.get('base_config', "coffea4bees/skimmer/metadata/sub_sampling_MC.yml")
    with open(base_file, 'r') as f:
        skimmer_cfg = yaml.safe_load(f) or {}

    datasets_file = phaseA_2.get('datasets_file', config.get('datasets_file', "coffea4bees/metadata/datasets/TT_stitched.yml"))
    base_path = phaseA_2.get('base_path', config.get('base_path', "root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/Run2/ttbar_PSData_stitched"))
    seed = int(phaseA_2.get('seed', 5))

    skimmer_cfg.setdefault('runner', {})['datasets_file'] = str(datasets_file)
    skimmer_cfg.setdefault('config', {})['base_path'] = str(base_path)
    skimmer_cfg['config']['sub_sampling_rand_seed'] = seed
    skimmer_cfg['config']['apply_trigWeight'] = True
    skimmer_cfg['config']['require_trigWeight'] = True

    if 'skimmer' in phaseA_2 and isinstance(phaseA_2['skimmer'], dict):
        for k, v in phaseA_2['skimmer'].items():
            if k == 'runner' and isinstance(v, dict):
                _deep_merge(skimmer_cfg['runner'], v)
            elif k == 'config' and isinstance(v, dict):
                _deep_merge(skimmer_cfg['config'], v)
            elif k in skimmer_cfg['runner']:
                skimmer_cfg['runner'][k] = copy.deepcopy(v)
            else:
                skimmer_cfg['config'][k] = copy.deepcopy(v)
    elif isinstance(phaseA_2, dict):
        if 'worker_memory' in phaseA_2:
            skimmer_cfg['runner']['worker_memory'] = str(phaseA_2['worker_memory'])
        if 'chunksize' in phaseA_2:
            skimmer_cfg['runner']['chunksize'] = int(phaseA_2['chunksize'])
            skimmer_cfg['config']['step'] = int(phaseA_2['chunksize'])
        if 'picosize' in phaseA_2:
            skimmer_cfg['runner']['picosize'] = int(phaseA_2['picosize'])

    skimmer_file = _dump_yaml(skimmer_cfg, os.path.join(configs_dir, "skimmer_subsample_mc_ttbar_psdata.yml"))

    return {
        "skimmer": skimmer_file,
    }


def stage_phaseA_3_configs(config, out_a3, mixeddata_jcm_file):
    """
    Generate all effective runtime configs for Stage A_3 (Subsample Splitting) into {out_a3}configs/.
    Loads base config from coffea4bees/skimmer/metadata/split_mixeddata.yml and overlays phaseA_3 parameters.
    """
    phaseA_3 = config.get('phaseA_3', config.get('phaseA_2', config.get('bkg_syst_A_3', {})))
    channel = config.get('channel', 'ttHbb')
    configs_dir = os.path.join(out_a3, "configs")
    default_processor = config.get('analysis_processor') or (config.get('analysis_config', {}) or {}).get('processor') or f"coffea4bees/analysis/processors/processor_{channel}.py"
    n_subsamples = int(config.get('n_subsamples', config.get('n_models', config.get('n_samples', 16))))

    # 1. Split Mixed-Data Skimmer Config (MixedDataSplitter)
    base_file = phaseA_3.get('base_config', "coffea4bees/skimmer/metadata/split_mixeddata.yml")
    with open(base_file, 'r') as f:
        skimmer_split_cfg = yaml.safe_load(f) or {}

    skimmer_split_cfg.setdefault('config', {})['apply_JCM'] = True
    skimmer_split_cfg['config']['JCM_file'] = str(mixeddata_jcm_file)
    skimmer_split_cfg['config']['n_subsamples'] = n_subsamples

    if 'skimmer' in phaseA_3 and isinstance(phaseA_3['skimmer'], dict):
        for k, v in phaseA_3['skimmer'].items():
            if k in ['runner', 'config'] and isinstance(v, dict):
                _deep_merge(skimmer_split_cfg.setdefault(k, {}), v)
            elif k in skimmer_split_cfg.setdefault('runner', {}):
                skimmer_split_cfg['runner'][k] = copy.deepcopy(v)
            else:
                skimmer_split_cfg['config'][k] = copy.deepcopy(v)
    skimmer_split_file = _dump_yaml(skimmer_split_cfg, os.path.join(configs_dir, f"skimmer_split_mixeddata_{channel}.yml"))

    # 2. Study Mixed Data Config
    study_cfg = {
        "runner": {
            "workers": 4,
            "worker_memory": "4GB",
            "condor": True,
            "shared_dask": True,
            "run_performance": False,
        },
        "config": {
            "apply_JCM": True,
            "JCM_file": str(mixeddata_jcm_file),
        }
    }
    if 'study' in phaseA_3 and isinstance(phaseA_3['study'], dict):
        _deep_merge(study_cfg, phaseA_3['study'])
    study_file = _dump_yaml(study_cfg, os.path.join(configs_dir, f"study_mixeddata_{channel}.yml"))

    # 3. Subsample Unit Analysis Config
    analysis_unit_cfg = {
        "processor": default_processor,
        "runner": {
            "workers": 4,
            "worker_memory": "4GB",
            "condor": True,
            "shared_dask": True,
            "run_performance": False,
        },
        "config": {
            "blind": False,
            "apply_FvT": False,
            "apply_JCM": False,
            "apply_trigWeight": False,
            "apply_btagSF": True,
            "apply_boosted_veto": False,
            "run_SvB": False,
            "top_reconstruction": "fast",
            "fill_histograms": True,
            "hist_cuts": ["pass_nSelJets_gt6"],
        }
    }
    if 'analysis' in phaseA_3 and isinstance(phaseA_3['analysis'], dict):
        for k, v in phaseA_3['analysis'].items():
            if k == 'runner' and isinstance(v, dict):
                _deep_merge(analysis_unit_cfg['runner'], v)
            elif k == 'config' and isinstance(v, dict):
                _deep_merge(analysis_unit_cfg['config'], v)
            elif k in analysis_unit_cfg['config']:
                analysis_unit_cfg['config'][k] = copy.deepcopy(v)
            else:
                analysis_unit_cfg[k] = copy.deepcopy(v)
    analysis_unit_file = _dump_yaml(analysis_unit_cfg, os.path.join(configs_dir, f"analysis_mixeddata_subsample_unit.yml"))

    # 4. Plots v0 Closure Config (All inclusive)
    plots_closure_cfg = {
        "hists": {
            "data": {
                "process": "data",
                "tag": "fourTag",
                "label": "Four-tag data",
                "edgecolor": "k",
                "fillcolor": "k",
            }
        },
        "stack": {
            "mixeddata": {
                "process": "mix_v0",
                "tag": "fourTag",
                "fillcolor": "#FFDF7Fff",
                "edgecolor": "k",
                "label": "Mixed data v0 (4b)",
            }
        },
        "ratios": {
            "dataToStack": {
                "numerator": {"type": "hists", "key": "data"},
                "denominator": {"type": "stack"},
                "uncertianty": "nominal",
                "color": "k",
                "marker": "o",
            }
        },
        "doRatio": 1,
        "categories": ["inclusive"],
        "regions": ["SR", "SB"],
    }
    if 'closure_plots' in phaseA_3 and isinstance(phaseA_3['closure_plots'], dict):
        _deep_merge(plots_closure_cfg, phaseA_3['closure_plots'])
    plots_closure_file = _dump_yaml(plots_closure_cfg, os.path.join(configs_dir, f"plots_v0_closure_{channel}.yml"))

    # 5. Plots v0 vs mixeddata_all Config (All inclusive)
    plots_vs_all_cfg = {
        "hists": {
            "subsample_v0": {
                "process": "mix_v0",
                "tag": "fourTag",
                "label": "Mixed data v0 (unweighted)",
                "edgecolor": "k",
                "fillcolor": "k",
            }
        },
        "stack": {
            "mixeddata_all": {
                "process": f"mixeddata_{channel}",
                "tag": "fourTag",
                "fillcolor": "orange",
                "edgecolor": "k",
                "label": "mixeddata_all * JCM",
            }
        },
        "ratios": {
            "v0ToAll": {
                "numerator": {"type": "hists", "key": "subsample_v0"},
                "denominator": {"type": "stack"},
                "uncertianty": "nominal",
                "color": "k",
                "marker": "o",
            }
        },
        "doRatio": 1,
        "categories": ["inclusive"],
        "regions": ["sum", "SR", "SB"],
    }
    if 'correlation_plots' in phaseA_3 and isinstance(phaseA_3['correlation_plots'], dict):
        _deep_merge(plots_vs_all_cfg, phaseA_3['correlation_plots'])
    plots_vs_all_file = _dump_yaml(plots_vs_all_cfg, os.path.join(configs_dir, f"plots_v0_vs_mixeddata_all_{channel}.yml"))

    return {
        "skimmer_split": skimmer_split_file,
        "study": study_file,
        "analysis_unit": analysis_unit_file,
        "plots_closure": plots_closure_file,
        "plots_vs_all": plots_vs_all_file,
    }


def stage_phaseA_4_configs(config, out_a4):
    """
    Generate unified runtime config for Stage A_4 (subsample processing: classifier inputs +
    JCM histograms + SvB friend trees) into {out_a4}configs/.
    Only non-default parameters specified in config['phaseA_4'] are merged on top.
    """
    phaseA_4 = config.get('phaseA_4', config.get('phaseA_3', config.get('bkg_syst_A_4', {})))
    channel = config.get('channel', 'ttHbb')
    configs_dir = os.path.join(out_a4, "configs")
    default_processor = config.get('analysis_processor') or (config.get('analysis_config', {}) or {}).get('processor') or f"coffea4bees/analysis/processors/processor_{channel}.py"

    classifier_inputs_base = config.get(
        'classifier_inputs_base',
        f"root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/{channel}_stitched/classifier_inputs/mixeddata/"
    )
    friend_base = config.get(
        'mixeddata_friend_base',
        f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/friends/{channel}/"
    )

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
            "SvB_MA": True,
            "top_reconstruction": "fast",
            "fill_histograms": True,
            "hist_cuts": ["pass_nSelJets_gt6"],
            "make_classifier_input": classifier_inputs_base,
            "make_friend_SvB": friend_base,
        }
    }
    if isinstance(phaseA_4, dict):
        if 'runner' in phaseA_4 and isinstance(phaseA_4['runner'], dict):
            _deep_merge(process_subsamples_cfg['runner'], phaseA_4['runner'])
        if 'config' in phaseA_4 and isinstance(phaseA_4['config'], dict):
            _deep_merge(process_subsamples_cfg['config'], phaseA_4['config'])
        for k in ['worker_memory', 'workers', 'min_workers', 'max_workers']:
            if k in phaseA_4:
                process_subsamples_cfg['runner'][k] = phaseA_4[k]
        for k, v in phaseA_4.items():
            if k not in ['runner', 'config', 'worker_memory', 'workers', 'min_workers', 'max_workers'] and not isinstance(v, dict):
                process_subsamples_cfg['config'][k] = v

    process_subsamples_file = _dump_yaml(process_subsamples_cfg, os.path.join(configs_dir, f"process_subsamples_mixeddata_{channel}.yml"))

    return {
        "process_subsamples": process_subsamples_file,
    }


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
        fvt_friend = os.path.join(out, f"bkg_syst_C_1_FvT/friends/friends_FvT_{mix_name}_v{m}.json@@FvT")

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
                    "SvB_MA": config.get('data_svb_friend', f"root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/{channel}_stitched/friend/SvB_{channel}_stitched/result.json@@analysis.0.merged"),
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

