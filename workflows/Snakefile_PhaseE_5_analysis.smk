# coffea4bees/workflows/Snakefile_PhaseE_3_analysis.smk
# Phase E_3: Analysis Histogramming & Comparison Plots for Mixed Data

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_mixeddata_closure/")
config.setdefault('channel', "ttHbb")
channel = config['channel']

raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
if isinstance(raw_years, str):
    YEARS = [str(y).strip() for y in raw_years.split() if str(y).strip()]
else:
    YEARS = [str(y) for y in raw_years]
config['years'] = YEARS
default_container_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', config.get('container_wrapper', default_container_wrapper))
condor_flags = "" if config.get("test", False) else "--shared-dask --condor"

def apply_test_runner_overrides(cfg):
    if config.get("test", False):
        cfg.setdefault("runner", {})
        cfg["runner"]["condor"] = False
        cfg["runner"]["shared_dask"] = False
        cfg["runner"]["workers"] = 2
        cfg["runner"].pop("min_workers", None)
        cfg["runner"].pop("max_workers", None)
        if "chunksize" in config:
            cfg["runner"]["chunksize"] = config["chunksize"]
        elif "chunksize" not in cfg["runner"]:
            cfg["runner"]["chunksize"] = 1000
        if "maxchunks" in config:
            cfg["runner"]["maxchunks"] = config["maxchunks"]
        elif "maxchunks" not in cfg["runner"]:
            cfg["runner"]["maxchunks"] = 1
    return cfg

python_bin = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
config.setdefault('python_bin', python_bin)

out = config['output_path']
if not out.endswith("/"):
    out += "/"
os.makedirs(out, exist_ok=True)

if "_SNAKEFILE_EVAL_FRIENDS_INCLUDED" not in globals():
    _SNAKEFILE_EVAL_FRIENDS_INCLUDED = True
    include: "Snakefile_eval_friends_mixeddata_ttHbb.smk"

wildcard_constraints:
    v = r"\d+",
    mode = "(2class|4class)",

localrules: all_PhaseE_5, all_closure_hists, closure_test_subsample, closure_test_subsample_mode, create_closure_data_config, create_closure_data_config_mode, create_closure_mixeddata_config, create_closure_mixeddata_config_mode, create_closure_plot_config, create_closure_plot_config_mode, check_cutflow_mixeddata

rule all_PhaseE_5:
    input:
        f"{out}histAll_{config['label']}.coffea",
        f"{out}plots_comparison/plots_done.txt",
        f"{out}plots_analysis/plots_done.txt"

rule all_closure_hists:
    input:
        expand(f"{out}closure_v{{v}}/histAll_data_v{{v}}.coffea", v=range(int(config.get('n_models', 16)))),
        expand(f"{out}closure_v{{v}}/histAll_mixeddata_v{{v}}.coffea", v=range(int(config.get('n_models', 16)))),

# ── Subsample Closure Test Master Target ──────────────────────────────────────
rule closure_test_subsample_mode:
    input:
        f"{out}closure_v{{v}}_{{mode}}/plots/plots_done.txt"

rule closure_test_subsample:
    input:
        f"{out}closure_v{{v}}/plots/plots_done.txt"

# ── Histogramming ─────────────────────────────────────────────────────────────
rule run_analysis_mixeddata:
    input:
        friend_json = config.get('mixeddata_friend_json', "coffea4bees/metadata/friends/friends_ttHbb_mixeddata_4b.json"),
        dataset_yaml = config.get('multisample_install_path', config.get('datasets_file', "coffea4bees/metadata/datasets/mixeddata_4b.yml")),
    output:
        f"{out}histAll_{config['label']}.coffea"
    log:
        f"{out}logs/analysis_{config['label']}.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        config_file = config.get('analysis_config_file', workflow.configfiles[0] if workflow.configfiles else "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"),
        datasets = "mixeddata_4b",
        output_path = out,
        output_name = f"histAll_{config['label']}.coffea",
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
    resources:
        slurm_partition = "work",
        qos = "light",
        mem_mb = 32000,
        cpus_per_task = 16,
        runtime = 240,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {params.config_file} \
            --processor {params.processor} \
            --datasets {params.datasets} \
            --output-path {params.output_path} \
            --output {params.output_name} 2>&1 | tee {log}
        """

rule check_cutflow_mixeddata:
    input:
        coffea_file = f"{out}histAll_{config['label']}.coffea"
    output:
        validation_txt = f"{out}cutflow_validation_{config['label']}.txt",
        cutflow_yml = f"{out}cutflow_{config['label']}.yml"
    log:
        f"{out}logs/cutflow_validation_{config['label']}.log"
    params:
        known_counts = lambda wildcards: config.get("known_counts", ""),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = config.get('analysis_container_wrapper', ""),
        python_bin = config.get('python_bin', "python")
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.validation_txt}) $(dirname {log})
        echo "Running cutflow analysis and verification for {input.coffea_file}" > {log}
        {params.run_container_wrapper} bash coffea4bees/scripts/run-cutflow.sh \
            --input-file "{input.coffea_file}" \
            --output-file "{output.cutflow_yml}" \
            $([ -n "{params.known_counts}" ] && [ "{params.known_counts}" != "none" ] && [ -f "{params.known_counts}" ] && echo "--known-cutflow {params.known_counts}") \
            --error-threshold "{params.error_threshold}" \
            --cutflow-list "{params.cutflow_list}" \
            --python-bin "{params.python_bin}" 2>&1 | tee -a {log}
        touch {output.validation_txt}
        """

# ── Subsample Closure: Step 5 Data 3b (JCM * FvT) and Subsample 4b ─────────────
rule create_closure_data_config:
    input:
        jcm = f"{out}JCM_subsamples/jetCombinatoricModel_sum_mix_v{{v}}.yml",
        fvt = f"{out}FvT_training/friends/friends_FvT_{config.get('mix_name', '3bDvTMix4bDvT')}_v{{v}}.json",
    output:
        cfg = f"{out}closure_v{{v}}/analysis_config_data.yml",
        friends = f"{out}closure_v{{v}}/friends_data.yml",
        weights = f"{out}closure_v{{v}}/weights_data.yml",
    params:
        channel = channel,
    run:
        import yaml
        os.makedirs(os.path.dirname(output.cfg), exist_ok=True)
        friends_dict = {
            "friends": {
                y: {
                    "trigWeight": "coffea4bees/metadata/datasets/trigweights_2024_v2.json@@trigWeight",
                    "FvT": f"{input.fvt}@@FvT",
                    "SvB_MA": f"root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/{params.channel}_v3/friend/SvB_{params.channel}_v3/result.json@@analysis.0.merged"
                } for y in YEARS
            }
        }
        with open(output.friends, 'w') as f:
            yaml.dump(friends_dict, f, default_flow_style=False)

        weights_dict = {
            "weights": {
                y: {
                    "JCM_file": str(input.jcm)
                } for y in YEARS
            }
        }
        with open(output.weights, 'w') as f:
            yaml.dump(weights_dict, f, default_flow_style=False)

        cfg = {
            "weights_file": output.weights,
            "friend_file": output.friends,
            "runner": {
                "workers": 4,
                "worker_memory": "6GB",
                "condor": True,
                "shared_dask": True,
                "run_performance": True,
                "dataset_location": "coffea4bees/metadata/datasets/",
                "datasets_file": "coffea4bees/metadata/datasets/data.yml",
                "friend_file": output.friends,
                "weights_file": output.weights,
            },
            "config": {
                "blind": False,
                "apply_FvT": True,
                "apply_JCM": True,
                "JCM_file": str(input.jcm),
                "apply_trigWeight": True,
                "apply_btagSF": True,
                "apply_boosted_veto": False,
                "run_SvB": True,
                "SvB_MA": True,
                "top_reconstruction": "fast",
                "candidates_selection_cfg": f"coffea4bees/analysis/metadata/candidates_selection_thresholds_{params.channel}.yml",
                "fill_histograms": True,
                "hist_cuts": ["pass_nSelJets_gt6", "fail_nSelJets_le6"],
            }
        }
        cfg = apply_test_runner_overrides(cfg)
        with open(output.cfg, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule run_closure_data:
    input:
        cfg = f"{out}closure_v{{v}}/analysis_config_data.yml",
        friends = f"{out}closure_v{{v}}/friends_data.yml",
        weights = f"{out}closure_v{{v}}/weights_data.yml",
        jcm = f"{out}JCM_subsamples/jetCombinatoricModel_sum_mix_v{{v}}.yml",
        fvt = f"{out}FvT_training/friends/friends_FvT_{config.get('mix_name', '3bDvTMix4bDvT')}_v{{v}}.json",
    output:
        coffea_out = f"{out}closure_v{{v}}/histAll_data_v{{v}}.coffea",
    log:
        f"{out}closure_v{{v}}/logs/run_closure_data.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = "data",
        output_path = f"{out}closure_v{{v}}/",
        years = " ".join(YEARS),
        channel = channel,
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --friends {input.friends} \
            --weights {input.weights} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

rule create_closure_mixeddata_config:
    input:
        friend_json = config.get('mixeddata_friend_json', f"coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json"),
        dataset_file = config.get('multisample_install_path', config.get('datasets_file', "coffea4bees/metadata/datasets/mixeddata_4b.yml")),
    output:
        cfg = f"{out}closure_v{{v}}/analysis_config_mixeddata.yml",
        friends = f"{out}closure_v{{v}}/friends_mixeddata.yml",
    params:
        channel = channel,
        friend_json = config.get('mixeddata_friend_json', f"coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json"),
        dataset_file = config.get('multisample_install_path', config.get('datasets_file', "coffea4bees/metadata/datasets/mixeddata_4b.yml")),
    run:
        import yaml
        os.makedirs(os.path.dirname(output.cfg), exist_ok=True)
        friends_dict = {
            "friends": {
                y: {
                    "SvB_MA": f"{params.friend_json}@@SvB_MA"
                } for y in YEARS
            }
        }
        with open(output.friends, 'w') as f:
            yaml.dump(friends_dict, f, default_flow_style=False)

        cfg = {
            "weights_file": f"coffea4bees/metadata/weights/weights_{params.channel}.yml",
            "friend_file": output.friends,
            "runner": {
                "workers": 4,
                "worker_memory": "4GB",
                "condor": True,
                "shared_dask": True,
                "run_performance": True,
                "dataset_location": "coffea4bees/metadata/datasets/",
                "datasets_file": str(params.dataset_file),
                "friend_file": output.friends,
                "weights_file": f"coffea4bees/metadata/weights/weights_{params.channel}.yml",
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
                "candidates_selection_cfg": f"coffea4bees/analysis/metadata/candidates_selection_thresholds_{params.channel}.yml",
                "fill_histograms": True,
                "hist_cuts": ["pass_nSelJets_gt6", "fail_nSelJets_le6"],
            }
        }
        cfg = apply_test_runner_overrides(cfg)
        with open(output.cfg, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule run_closure_mixeddata:
    input:
        cfg = f"{out}closure_v{{v}}/analysis_config_mixeddata.yml",
        friends = f"{out}closure_v{{v}}/friends_mixeddata.yml",
        friends_json = config.get('mixeddata_friend_json', f"coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json"),
    output:
        coffea_out = f"{out}closure_v{{v}}/histAll_mixeddata_v{{v}}.coffea",
    log:
        f"{out}closure_v{{v}}/logs/run_closure_mixeddata.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = lambda wildcards: f"mixeddata_4b:{wildcards.v}",
        output_path = f"{out}closure_v{{v}}/",
        years = " ".join(YEARS),
        channel = channel,
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --friends {input.friends} \
            --weights coffea4bees/metadata/weights/weights_{params.channel}.yml \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

# ── Subsample Closure: Step 6 Comparison Plot (Data 3b vs Subsample 4b) ───────
rule create_closure_plot_config:
    output:
        f"{out}closure_v{{v}}/plots_metadata.yml"
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        pcfg = {
            "hists": {
                "mixeddata": {
                    "process": f"mix_v{wildcards.v}",
                    "tag": "fourTag",
                    "label": f"Mixed data v{wildcards.v} (4b)",
                    "edgecolor": "k",
                    "fillcolor": "k",
                }
            },
            "stack": {
                "multijet": {
                    "process": "data",
                    "tag": "threeTag",
                    "fillcolor": "#f5a742",
                    "edgecolor": "k",
                    "label": "Data 3b (JCM * FvT)",
                }
            },
            "ratios": {
                "mixedToModel": {
                    "numerator": {
                        "type": "hists",
                        "key": "mixeddata",
                    },
                    "denominator": {
                        "type": "stack",
                    },
                    "uncertianty": "nominal",
                    "color": "k",
                    "marker": "o",
                }
            },
            "doRatio": 1,
            "categories": ["inclusive", "pass_nSelJets_gt6", "fail_nSelJets_le6"],
            "regions": ["SR", "SB"],
        }
        with open(output[0], 'w') as f:
            yaml.dump(pcfg, f, default_flow_style=False)

rule make_plots_closure:
    input:
        data_coffea = f"{out}closure_v{{v}}/histAll_data_v{{v}}.coffea",
        mixed_coffea = f"{out}closure_v{{v}}/histAll_mixeddata_v{{v}}.coffea",
        plot_cfg = f"{out}closure_v{{v}}/plots_metadata.yml",
    output:
        f"{out}closure_v{{v}}/plots/plots_done.txt"
    log:
        f"{out}closure_v{{v}}/logs/make_plots.log"
    params:
        plot_script = "coffea4bees/plots/makePlots.py",
        output_dir = f"{out}closure_v{{v}}/plots/",
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {params.plot_script} \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year RunII \
            -p 1 2>&1 | tee {log}
        touch {output}
        """

# ── Subsample Closure with Explicit FvT Mode (e.g. 2class, 4class) ────────────
rule create_closure_data_config_mode:
    input:
        jcm = f"{out}JCM_subsamples/jetCombinatoricModel_sum_mix_v{{v}}.yml",
        fvt = f"{out}FvT_{{mode}}/friends/friends_FvT_{config.get('mix_name', '3bDvTMix4bDvT')}_{{mode}}_v{{v}}.json",
    output:
        cfg = f"{out}closure_v{{v}}_{{mode}}/analysis_config_data.yml",
        friends = f"{out}closure_v{{v}}_{{mode}}/friends_data.yml",
        weights = f"{out}closure_v{{v}}_{{mode}}/weights_data.yml",
    params:
        channel = channel,
    run:
        import yaml
        os.makedirs(os.path.dirname(output.cfg), exist_ok=True)
        friends_dict = {
            "friends": {
                y: {
                    "trigWeight": "coffea4bees/metadata/datasets/trigweights_2024_v2.json@@trigWeight",
                    "FvT": f"{input.fvt}@@FvT",
                    "SvB_MA": f"root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/{params.channel}_v3/friend/SvB_{params.channel}_v3/result.json@@analysis.0.merged"
                } for y in YEARS
            }
        }
        with open(output.friends, 'w') as f:
            yaml.dump(friends_dict, f, default_flow_style=False)

        weights_dict = {
            "weights": {
                y: {
                    "JCM_file": str(input.jcm)
                } for y in YEARS
            }
        }
        with open(output.weights, 'w') as f:
            yaml.dump(weights_dict, f, default_flow_style=False)

        cfg = {
            "weights_file": output.weights,
            "friend_file": output.friends,
            "runner": {
                "workers": 4,
                "worker_memory": "4GB",
                "condor": True,
                "shared_dask": True,
                "run_performance": True,
                "dataset_location": "coffea4bees/metadata/datasets/",
                "datasets_file": "coffea4bees/metadata/datasets/data.yml",
                "friend_file": output.friends,
                "weights_file": output.weights,
            },
            "config": {
                "blind": False,
                "apply_FvT": True,
                "apply_JCM": True,
                "JCM_file": str(input.jcm),
                "apply_trigWeight": True,
                "apply_btagSF": True,
                "apply_boosted_veto": False,
                "run_SvB": True,
                "SvB_MA": True,
                "top_reconstruction": "fast",
                "candidates_selection_cfg": f"coffea4bees/analysis/metadata/candidates_selection_thresholds_{params.channel}.yml",
                "fill_histograms": True,
                "hist_cuts": ["pass_nSelJets_gt6", "fail_nSelJets_le6"],
            }
        }
        cfg = apply_test_runner_overrides(cfg)
        with open(output.cfg, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule run_closure_data_mode:
    input:
        cfg = f"{out}closure_v{{v}}_{{mode}}/analysis_config_data.yml",
        friends = f"{out}closure_v{{v}}_{{mode}}/friends_data.yml",
        weights = f"{out}closure_v{{v}}_{{mode}}/weights_data.yml",
        jcm = f"{out}JCM_subsamples/jetCombinatoricModel_sum_mix_v{{v}}.yml",
        fvt = f"{out}FvT_{{mode}}/friends/friends_FvT_{config.get('mix_name', '3bDvTMix4bDvT')}_{{mode}}_v{{v}}.json",
    output:
        coffea_out = f"{out}closure_v{{v}}_{{mode}}/histAll_data_v{{v}}.coffea",
    log:
        f"{out}closure_v{{v}}_{{mode}}/logs/run_closure_data.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = "data",
        output_path = f"{out}closure_v{{v}}_{{mode}}/",
        years = " ".join(YEARS),
        channel = channel,
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --friends {input.friends} \
            --weights {input.weights} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

rule create_closure_mixeddata_config_mode:
    output:
        cfg = f"{out}closure_v{{v}}_{{mode}}/analysis_config_mixeddata.yml",
        friends = f"{out}closure_v{{v}}_{{mode}}/friends_mixeddata.yml",
    params:
        channel = channel,
        friend_json = config.get('mixeddata_friend_json', f"coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json"),
        dataset_file = config.get('multisample_install_path', config.get('datasets_file', "coffea4bees/metadata/datasets/mixeddata_4b.yml")),
    run:
        import yaml
        os.makedirs(os.path.dirname(output.cfg), exist_ok=True)
        friends_dict = {
            "friends": {
                y: {
                    "SvB_MA": f"{params.friend_json}@@SvB_MA"
                } for y in YEARS
            }
        }
        with open(output.friends, 'w') as f:
            yaml.dump(friends_dict, f, default_flow_style=False)

        cfg = {
            "weights_file": f"coffea4bees/metadata/weights/weights_{params.channel}.yml",
            "friend_file": output.friends,
            "runner": {
                "workers": 4,
                "worker_memory": "4GB",
                "condor": True,
                "shared_dask": True,
                "run_performance": True,
                "dataset_location": config.get('dataset_location', "coffea4bees/metadata/datasets/"),
                "datasets_file": str(params.dataset_file),
                "friend_file": output.friends,
                "weights_file": f"coffea4bees/metadata/weights/weights_{params.channel}.yml",
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
                "candidates_selection_cfg": f"coffea4bees/analysis/metadata/candidates_selection_thresholds_{params.channel}.yml",
                "fill_histograms": True,
                "hist_cuts": ["pass_nSelJets_gt6", "fail_nSelJets_le6"],
            }
        }
        cfg = apply_test_runner_overrides(cfg)
        with open(output.cfg, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule run_closure_mixeddata_mode:
    input:
        cfg = f"{out}closure_v{{v}}_{{mode}}/analysis_config_mixeddata.yml",
        friends = f"{out}closure_v{{v}}_{{mode}}/friends_mixeddata.yml",
        friends_json = config.get('mixeddata_friend_json', f"coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json"),
    output:
        coffea_out = f"{out}closure_v{{v}}_{{mode}}/histAll_mixeddata_v{{v}}.coffea",
    log:
        f"{out}closure_v{{v}}_{{mode}}/logs/run_closure_mixeddata.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = lambda wildcards: f"mixeddata_4b:{wildcards.v}",
        output_path = f"{out}closure_v{{v}}_{{mode}}/",
        years = " ".join(YEARS),
        channel = channel,
        container_wrapper = config['analysis_container_wrapper'],
        condor_flags = condor_flags,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --friends {input.friends} \
            --weights coffea4bees/metadata/weights/weights_{params.channel}.yml \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

rule create_closure_plot_config_mode:
    output:
        f"{out}closure_v{{v}}_{{mode}}/plots_metadata.yml"
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        pcfg = {
            "hists": {
                "mixeddata": {
                    "process": f"mix_v{wildcards.v}",
                    "tag": "fourTag",
                    "label": f"Mixed data v{wildcards.v} (4b)",
                    "edgecolor": "k",
                    "fillcolor": "k",
                }
            },
            "stack": {
                "multijet": {
                    "process": "data",
                    "tag": "threeTag",
                    "fillcolor": "#f5a742",
                    "edgecolor": "k",
                    "label": "Data 3b (JCM * FvT)",
                }
            },
            "ratios": {
                "mixedToModel": {
                    "numerator": {
                        "type": "hists",
                        "key": "mixeddata",
                    },
                    "denominator": {
                        "type": "stack",
                    },
                    "uncertianty": "nominal",
                    "color": "k",
                    "marker": "o",
                }
            },
            "doRatio": 1,
            "categories": ["inclusive", "pass_nSelJets_gt6", "fail_nSelJets_le6"],
            "regions": ["SR", "SB"],
        }
        with open(output[0], 'w') as f:
            yaml.dump(pcfg, f, default_flow_style=False)

rule make_plots_closure_mode:
    input:
        data_coffea = f"{out}closure_v{{v}}_{{mode}}/histAll_data_v{{v}}.coffea",
        mixed_coffea = f"{out}closure_v{{v}}_{{mode}}/histAll_mixeddata_v{{v}}.coffea",
        plot_cfg = f"{out}closure_v{{v}}_{{mode}}/plots_metadata.yml",
    output:
        f"{out}closure_v{{v}}_{{mode}}/plots/plots_done.txt"
    log:
        f"{out}closure_v{{v}}_{{mode}}/logs/make_plots.log"
    params:
        plot_script = "coffea4bees/plots/makePlots.py",
        output_dir = f"{out}closure_v{{v}}_{{mode}}/plots/",
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {params.plot_script} \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year RunII \
            -p 1 2>&1 | tee {log}
        touch {output}
        """

# ── Comparison Plots ──────────────────────────────────────────────────────────
rule make_plots_comparison_mixeddata:
    input:
        f"{out}histAll_{config['label']}.coffea"
    output:
        f"{out}plots_comparison/plots_done.txt"
    log:
        f"{out}logs/plots_comparison.log"
    params:
        plot_script = "coffea4bees/plots/makePlots.py",
        plot_config = "coffea4bees/plots/metadata/plots_mixeddata_vs_data.yml",
        output_dir = f"{out}plots_comparison/",
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {params.plot_script} {input} \
            -o {params.output_dir} \
            -m {params.plot_config} \
            --year RunII 2>&1 | tee {log}
        touch {output}
        """

rule make_plots_analysis_mixeddata:
    input:
        f"{out}histAll_{config['label']}.coffea"
    output:
        f"{out}plots_analysis/plots_done.txt"
    log:
        f"{out}logs/plots_analysis.log"
    params:
        plot_script = "coffea4bees/plots/makePlots.py",
        plot_config = "coffea4bees/plots/metadata/plotsAll_ttHbb_mixeddata.yml",
        output_dir = f"{out}plots_analysis/",
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {params.plot_script} {input} \
            -o {params.output_dir} \
            -m {params.plot_config} \
            --year RunII 2>&1 | tee {log}
        touch {output}
        """
