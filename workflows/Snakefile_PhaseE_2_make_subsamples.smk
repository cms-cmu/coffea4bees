# coffea4bees/workflows/Snakefile_PhaseE_1b_make_subsamples.smk
# Phase E_1b: Mixed-Data Subsample Slicing, Registry, and Subsample Closure Verification
#
# Slices mixeddata_all into 15 statistically independent pseudo-experiments (v0..v14)
# using the calibrated JCM weights, compiles the multi-sample registry (mixeddata_4b.yml),
# and performs automated closure verification of subsample v0 against 4b Data and mixeddata_all.

import os
import shutil
import ast as _ast

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

include: "helpers/common.smk"

# Resolve phase_e configuration block if present
phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure', 'mixeddata'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

# Resolve phase_e_jcm configuration block
jcm_cfg = resolve_config_section(config, primary_key='phase_e_jcm', fallback_keys=['jcm', 'phase_e'])
for k, v in jcm_cfg.items():
    config.setdefault(f"jcm_{k}", v)

# Resolve mixeddata configuration block
mixeddata_cfg = resolve_config_section(config, primary_key='mixeddata', fallback_keys=['mixed_data'])
for k, v in mixeddata_cfg.items():
    config.setdefault(k, v)

# Container and general configuration
config.setdefault('analysis_container',
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/")
config.setdefault('channel', "ttHbb")
channel = config['channel']

# Parse years and determine run period (Run 2 vs Run 3)
raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
if isinstance(raw_years, str):
    YEARS = [str(y).strip() for y in raw_years.split() if str(y).strip()]
else:
    YEARS = [str(y) for y in raw_years]
config['years'] = YEARS

is_run3 = any(('202' in str(y) or 'Run3' in str(y)) for y in YEARS)
config.setdefault('isRun3', is_run3)
run_period = "Run3" if is_run3 else "Run2"

# Rank configuration (0 = nearest neighbor rank [0,0])
config.setdefault('default_rank', 0)
_rank_raw = config['default_rank']
if isinstance(_rank_raw, str):
    try:
        _rank = _ast.literal_eval(_rank_raw)
    except (ValueError, SyntaxError):
        _rank = _rank_raw
else:
    _rank = _rank_raw

if isinstance(_rank, (list, tuple)):
    _rank_suffix = f"_rank{int(_rank[0])}_{int(_rank[1])}"
    _rank_tuple = [int(_rank[0]), int(_rank[1])]
else:
    _rank_suffix = f"_rank{int(_rank)}_{int(_rank)}"
    _rank_tuple = [int(_rank), int(_rank)]

config.setdefault('tag', '')
_tag = str(config['tag'])
_tag_suffix = f"_{_tag}" if _tag else ''

# Output paths
config.setdefault('output_path', f"output/ttHbb_mixeddata_closure/")
config.setdefault('base_path',
    f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/{run_period}/{channel}_pz{_rank_suffix}")

# Dataset naming
config.setdefault('dataset_name', f"mixeddata_{channel}{_rank_suffix}")
config.setdefault('install_path', f"coffea4bees/metadata/datasets/mixeddata_{channel}{_rank_suffix}.yml")

# Subsampling configuration (16 datasets v0..v15)
config.setdefault('n_subsamples', 16)
config.setdefault('multisample_dataset_name', "mixeddata_4b")
config.setdefault('multisample_install_path', "coffea4bees/metadata/datasets/mixeddata_4b.yml")
config.setdefault('subsample_output_path', f"output/{channel}_mixeddata_subsamples/")
N_SUBSAMPLES = int(config['n_subsamples'])
SUBSAMPLES = [str(i) for i in range(N_SUBSAMPLES)]

# Classifier inputs configuration on EOS
config.setdefault('classifier_inputs_base',
    f"root://cmseos.fnal.gov//store/user/algomez/XX4b/2024_v2/{channel}/classifier_inputs/mixeddata/")
config.setdefault('classifier_inputs_json',
    f"coffea4bees/metadata/datasets/classifier_inputs_mixeddata_{channel}.json")

SVB_FRIEND_JSON = f"coffea4bees/metadata/friends/friends_{channel}_mixeddata_4b.json"

out = config['output_path']
if not out.endswith("/"):
    out += "/"
sub_out = config['subsample_output_path']
jcm_input_coffea = jcm_cfg.get('input_coffea', config.get('jcm_input_coffea', f"{out}inputs/histAll_NoJCM.coffea"))
mixeddata_jcm_file = f"{out}JCM_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml"

localrules: all_PhaseE_2, all_PhaseE_1b, all_subsamples, all_subsample_jcm, all_classifier_inputs_mixeddata, all_classifier_inputs_subsamples, all_friends_mixeddata, all_study_mixeddata, all_subsample_closure, prepare_data_noJCM, create_subsample_config, build_multisample_registry, create_noJCM_subsamples_config, create_subsample_jcm_config, make_subsample_jcm, create_study_mixeddata_config, plot_subsample_correlation, create_analysis_config_subsample, create_plot_config_v0_closure, make_plots_v0_closure, create_plot_config_v0_vs_mixeddata_all, make_plots_v0_vs_mixeddata_all, create_classifier_inputs_config_mixeddata, create_classifier_inputs_config_subsample, update_classifier_inputs_subsample_json, merge_all_classifier_inputs_subsamples_json, create_eval_config, merge_friends_json

# ── Default Master Target (Full Phase E2 End-to-End) ───────────────────────────
rule all_PhaseE_2:
    input:
        config['multisample_install_path'],
        expand(f"{out}JCM_subsamples/jetCombinatoricModel_SB_mix_v{{m}}.yml", m=range(N_SUBSAMPLES)),
        config['classifier_inputs_json'],
        SVB_FRIEND_JSON,

rule all_PhaseE_1b:
    input:
        rules.all_PhaseE_2.input

# ── Sub-Target Aliases ─────────────────────────────────────────────────────────
rule all_subsamples:
    input:
        config['multisample_install_path']

rule all_subsample_jcm:
    input:
        expand(f"{out}JCM_subsamples/jetCombinatoricModel_SB_mix_v{{m}}.yml", m=range(N_SUBSAMPLES))

rule all_classifier_inputs_mixeddata:
    input:
        config['classifier_inputs_json']

rule all_classifier_inputs_subsamples:
    input:
        f"{out}classifier_inputs/merge_all_subsamples.done"

rule all_friends_mixeddata:
    input:
        SVB_FRIEND_JSON

rule all_study_mixeddata:
    input:
        f"{out}study_mixeddata_all{_rank_suffix}.coffea",
        f"{out}plots_study_mixeddata/subsample_correlation_matrix.png",

rule all_subsample_closure:
    input:
        f"{out}plots_v0_closure/plots_done.txt",
        f"{out}plots_v0_vs_mixeddata_all/plots_done.txt",

# ── Stage 1: Subsampling Mixed Data (15 Subsamples v0..v14) ────────────────────
rule create_subsample_config:
    input:
        jcm_file = mixeddata_jcm_file,
    output:
        f"{sub_out}configs/split_mixeddata_v{{v}}.yml"
    params:
        base_path = lambda wildcards: f"{config['base_path']}/subsamples/v{wildcards.v}",
        v = "{v}",
        n_subsamples = N_SUBSAMPLES,
        ds_file = config['install_path'],
        dataset_location = config['dataset_location'],
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "runner": {
                "workers": 4,
                "min_workers": 1,
                "max_workers": 200,
                "chunksize": 100000,
                "picosize": 100000,
                "basketsize": 10000,
                "class_name": "MixedDataSplitter",
                "data_tier": "picoAOD",
                "condor_cores": 1,
                "worker_memory": "4GB",
                "condor_transfer_input_files": ["src", "coffea4bees/"],
                "allowlist_sites": ["T2_US_Nebraska", "T2_US_Purdue", "T3_US_FNALLPC", "T3_US_NotreDame"],
                "datasets_file": params.ds_file,
                "dataset_location": params.dataset_location,
            },
            "config": {
                "base_path": str(params.base_path),
                "apply_JCM": True,
                "JCM_file": str(input.jcm_file),
                "mixed_subsample": int(params.v),
                "n_subsamples": int(params.n_subsamples),
                "step": 100000,
                "skip_collections": None,
                "skip_branches": None,
            }
        }
        with open(output[0], 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule run_split_mixeddata_per_subsample:
    input:
        ds_ready = config['install_path'],
        cfg = f"{sub_out}configs/split_mixeddata_v{{v}}.yml",
        jcm_ready = mixeddata_jcm_file,
    output:
        reg = f"{sub_out}per_subsample/picoaod_datasets_mix_v{{v}}.yml",
        done = f"{sub_out}.subsample_v{{v}}.done",
    log:
        f"{sub_out}logs/split_mixeddata_v{{v}}.log"
    params:
        processor = "coffea4bees/skimmer/processor/split_mixed_data.py",
        dataset = config['dataset_name'],
        output_path = f"{sub_out}per_subsample/",
        years = " ".join(YEARS),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.reg}) $(dirname {log})
        ./run_container python runner.py {input.cfg} \
            -p {params.processor} \
            -d {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.reg}) \
            -s --shared-dask --condor 2>&1 | tee {log}
        touch {output.done}
        """

# ── Stage 2: Multi-Sample Registry Generation ─────────────────────────────────
rule build_multisample_registry:
    input:
        dones = expand(f"{sub_out}.subsample_v{{v}}.done", v=SUBSAMPLES),
        registries = expand(f"{sub_out}per_subsample/picoaod_datasets_mix_v{{v}}.yml", v=SUBSAMPLES),
    output:
        config['multisample_install_path']
    params:
        dataset_name = config['multisample_dataset_name'],
        n_samples = N_SUBSAMPLES,
        years = YEARS,
    run:
        import yaml, re
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)

        multisample_data = {
            params.dataset_name: {
                "nSamples": int(params.n_samples),
                "xs": {"Run2": 1, "Run3": 1}
            }
        }
        era_files = {}
        curr_era = None
        in_files = False
        # Only the first per-subsample registry is parsed: every subsample produces the same set
        # of era/file layouts, and the filenames are templated to vXXX below. If v0 happens to be
        # the one still split into chunks, the template collapse above plus the membership test
        # keep the emitted list unique.
        with open(input.registries[0], 'r') as f:
            for line in f:
                if not line.startswith(' ') and ':' in line:
                    curr_era = line.split(':')[0].strip()
                    era_files[curr_era] = []
                    in_files = False
                elif curr_era and line.strip().startswith('files:'):
                    in_files = True
                elif in_files:
                    stripped = line.strip()
                    if stripped.startswith('- '):
                        era_files[curr_era].append(stripped[2:].strip())
                    elif stripped and not stripped.startswith('#'):
                        in_files = False

        for year in params.years:
            multisample_data[params.dataset_name][year] = {"picoAOD": {"files_template": []}}
            for era_key, files in era_files.items():
                if f"_{year}" in era_key:
                    for fp in files:
                        t = re.sub(r'/subsamples/v\d+/', '/subsamples/vXXX/', fp)
                        # Collapse the per-subsample picoAOD name to the vXXX template. Pre-hadd
                        # chunked files (picoAOD_mixed_v<N>.chunk0.root, .chunk1.root, ...) all map
                        # onto the same picoAOD_mixed_vXXX.root target, so the membership test below
                        # is what keeps them from being emitted once per chunk.
                        t = re.sub(r'_v\d+(\.chunk\d+)?\.root', r'_vXXX.root', t)
                        if t not in multisample_data[params.dataset_name][year]["picoAOD"]["files_template"]:
                            multisample_data[params.dataset_name][year]["picoAOD"]["files_template"].append(t)

        print(f"Building multisample registry with {len(params.years)} years")
        with open(output[0], 'w') as f:
            yaml.dump(multisample_data, f, default_flow_style=False)

# ── Stage 2b: Data 3b and Subsample noJCM Histogramming ────────────────────────
DATA_NOJCM_INPUT = config.get('data_nojcm_coffea', jcm_input_coffea)

rule prepare_data_noJCM:
    input:
        DATA_NOJCM_INPUT
    output:
        f"{out}JCM_subsamples/histAll_NoJCM_data.coffea"
    shell:
        """
        mkdir -p $(dirname {output})
        ln -sf $(readlink -f {input}) {output}
        """

rule create_noJCM_subsamples_config:
    input:
        ds_file = config['multisample_install_path']
    output:
        f"{out}JCM_subsamples/analysis_config_subsamples_noJCM.yml"
    params:
        dataset_location = config['dataset_location']
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "runner": {
                "workers": 4,
                "worker_memory": "4GB",
                "condor": True,
                "shared_dask": True,
                "run_performance": True,
                "datasets_file": input.ds_file,
                "dataset_location": params.dataset_location,
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
                "hist_cuts": ["pass_nSelJets_gt6", "fail_nSelJets_le6"],
            }
        }
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule run_noJCM_subsamples:
    input:
        ds_ready = config['multisample_install_path'],
        analysis_cfg = f"{out}JCM_subsamples/analysis_config_subsamples_noJCM.yml",
    output:
        coffea_out = f"{out}JCM_subsamples/histAll_subsamples_noJCM.coffea",
    log:
        f"{out}JCM_subsamples/logs/run_noJCM_subsamples.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = config['multisample_dataset_name'],
        output_path = f"{out}JCM_subsamples/",
        years = " ".join(YEARS),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        ./run_container python runner.py {input.analysis_cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            --shared-dask --condor 2>&1 | tee {log}
        """

# ── Stage 2c: Dedicated JCM Fits for Each Subsample (m=0..15) ─────────────────
rule create_subsample_jcm_config:
    output:
        f"{out}JCM_subsamples/configs/config_v{{m}}.yml"
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        fit_cfg = {
            "data4bName": f"mix_v{wildcards.m}",
            "taglabel4b": "fourTag",
            "data3bName": "data",
            "taglabel3b": "threeTag",
            "taglabel3b_tt": "threeTag",
            "selJets": "selJets.n",
            "tagJets": "tagJets.n",
            "ignoreTT": False,
            "subtract3bTT": True,
            "ttbarProcesses": [
                "TTTo2L2Nu_stitched",
                "TTToSemiLeptonic_stitched",
                "TTToHadronic_stitched",
            ],
        }
        with open(output[0], "w") as f:
            yaml.dump(fit_cfg, f, default_flow_style=False)

rule make_subsample_jcm:
    input:
        data_coffea = f"{out}JCM_subsamples/histAll_NoJCM_data.coffea",
        subsample_coffea = f"{out}classifier_inputs/histAll_{channel}_mixeddata_v{{m}}.coffea",
        fit_cfg = f"{out}JCM_subsamples/configs/config_v{{m}}.yml",
    output:
        f"{out}JCM_subsamples/jetCombinatoricModel_SB_mix_v{{m}}.yml"
    log:
        f"{out}JCM_subsamples/logs/make_jcm_v{{m}}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        ./run_container python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
            -i {input.data_coffea} {input.subsample_coffea} \
            --jcm_config {input.fit_cfg} \
            -w mix_v{wildcards.m} \
            -r SB \
            -o $(dirname {output})/ \
            --no-plots \
            --year RunII 2>&1 | tee {log}
        """

# ── Stage 3: Subsample Correlation & Orthogonality Study ──────────────────────
rule create_study_mixeddata_config:
    output:
        f"{out}study_mixeddata_config.yml"
    params:
        ds_file = config['install_path'],
        dataset_location = config['dataset_location'],
        jcm_file = mixeddata_jcm_file,
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "runner": {
                "workers": 4,
                "worker_memory": "4GB",
                "condor": True,
                "shared_dask": True,
                "run_performance": True,
                "datasets_file": params.ds_file,
                "dataset_location": params.dataset_location,
            },
            "config": {
                "apply_JCM": True,
                "JCM_file": params.jcm_file,
            }
        }
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule study_mixeddata:
    input:
        ds_ready = config['install_path'],
        jcm_ready = mixeddata_jcm_file,
        study_cfg = f"{out}study_mixeddata_config.yml",
    output:
        coffea_out = f"{out}study_mixeddata_all{_rank_suffix}.coffea",
    log:
        f"{out}logs/study_mixeddata.log"
    params:
        processor = "coffea4bees/analysis/processors/processor_study_mixed_data.py",
        dataset = config['dataset_name'],
        output_path = out,
        years = " ".join(YEARS),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        ./run_container python runner.py {input.study_cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            --shared-dask --condor 2>&1 | tee {log}
        """

rule plot_subsample_correlation:
    input:
        coffea = f"{out}study_mixeddata_all{_rank_suffix}.coffea",
    output:
        plot = f"{out}plots_study_mixeddata/subsample_correlation_matrix.png",
    log:
        f"{out}plots_study_mixeddata/logs/plot_subsample_correlation.log"
    params:
        out_dir = f"{out}plots_study_mixeddata/",
    shell:
        """
        set -eo pipefail
        mkdir -p {params.out_dir} $(dirname {log})
        ./run_container python scripts/plot_subsample_correlation.py \
            -i {input.coffea} \
            -o {params.out_dir} 2>&1 | tee {log}
        """

# ── Stage 4: Subsample Processing (Unweighted Unit Weights) ────────────────────
rule create_analysis_config_subsample:
    input:
        ds_file = config['multisample_install_path'],
    output:
        f"{out}analysis_config_subsample_v{{v}}.yml"
    params:
        dataset_location = config['dataset_location'],
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "runner": {
                "workers": 4,
                "worker_memory": "4GB",
                "condor": True,
                "shared_dask": True,
                "run_performance": True,
                "datasets_file": input.ds_file,
                "dataset_location": params.dataset_location,
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
                "hist_cuts": ["pass_nSelJets_gt6", "fail_nSelJets_le6"],
            }
        }
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule run_analysis_subsample:
    input:
        ds_ready = config['multisample_install_path'],
        analysis_cfg = f"{out}analysis_config_subsample_v{{v}}.yml",
    output:
        coffea_out = f"{out}histAll_{channel}_mixeddata_v{{v}}.coffea",
    log:
        f"{out}logs/run_analysis_subsample_v{{v}}.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = lambda wildcards: f"{config['multisample_dataset_name']}:{wildcards.v}",
        output_path = out,
        years = " ".join(YEARS),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        ./run_container python runner.py {input.analysis_cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            --shared-dask --condor 2>&1 | tee {log}
        """

# ── Stage 5: Subsample v0 Closure Validation Plots ────────────────────────────
rule create_plot_config_v0_closure:
    output:
        f"{out}plots_metadata_v0_closure.yml"
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        pcfg = {
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
                "TTbar": {
                    "process": ["TTToHadronic", "TTToSemiLeptonic", "TTTo2L2Nu"],
                    "tag": "fourTag",
                    "fillcolor": "#85D1FBff",
                    "edgecolor": "k",
                    "label": "TTbar",
                },
                "mixeddata": {
                    "process": "mix_v0",
                    "tag": "fourTag",
                    "fillcolor": "orange",
                    "edgecolor": "k",
                    "label": "Mixed data v0 (unweighted)",
                }
            },
            "ratios": {
                "dataToStack": {
                    "numerator": {
                        "type": "hists",
                        "key": "data",
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
            "regions": ["sum", "SR", "SB"],
        }
        with open(output[0], "w") as f:
            yaml.dump(pcfg, f, default_flow_style=False)

rule make_plots_v0_closure:
    input:
        data_coffea = jcm_input_coffea,
        subsample_coffea = f"{out}histAll_{channel}_mixeddata_v0.coffea",
        plot_cfg = f"{out}plots_metadata_v0_closure.yml",
    output:
        done = f"{out}plots_v0_closure/plots_done.txt",
    log:
        f"{out}logs/make_plots_v0_closure.log"
    params:
        output_dir = f"{out}plots_v0_closure/",
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        ./run_container python coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.subsample_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year RunII \
            -p 1 2>&1 | tee {log}
        touch {output.done}
        """

rule create_plot_config_v0_vs_mixeddata_all:
    output:
        f"{out}plots_metadata_v0_vs_mixeddata_all.yml"
    params:
        dataset = config['dataset_name'],
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        pcfg = {
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
                    "process": [f"{params.dataset}{e}" for e in ["A", "B", "C", "D", "E", "F", "G", "H"]] + [params.dataset],
                    "tag": "fourTag",
                    "fillcolor": "orange",
                    "edgecolor": "k",
                    "label": "mixeddata_all * JCM",
                }
            },
            "ratios": {
                "v0ToAll": {
                    "numerator": {
                        "type": "hists",
                        "key": "subsample_v0",
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
            "regions": ["sum", "SR", "SB"],
        }
        with open(output[0], "w") as f:
            yaml.dump(pcfg, f, default_flow_style=False)

rule make_plots_v0_vs_mixeddata_all:
    input:
        mixed_coffea = f"{out}histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea",
        subsample_coffea = f"{out}histAll_{channel}_mixeddata_v0.coffea",
        plot_cfg = f"{out}plots_metadata_v0_vs_mixeddata_all.yml",
    output:
        done = f"{out}plots_v0_vs_mixeddata_all/plots_done.txt",
    log:
        f"{out}logs/make_plots_v0_vs_mixeddata_all.log"
    params:
        output_dir = f"{out}plots_v0_vs_mixeddata_all/",
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        ./run_container python coffea4bees/plots/makePlots.py \
            {input.mixed_coffea} {input.subsample_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year RunII \
            -p 1 2>&1 | tee {log}
        touch {output.done}
        """

# ── Stage 6: Classifier Inputs Production for All 15 Subsamples ───────────────
rule create_classifier_inputs_config_mixeddata:
    output:
        f"{out}classifier_inputs_config_mixeddata.yml"
    params:
        channel = channel,
        inputs_base = config['classifier_inputs_base'],
        ds_file = config['multisample_install_path'],
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "runner": {
                "workers": 4,
                # Adaptive scaling starts from min_workers and ramps lazily; with a
                # shared cluster and several concurrent subsample jobs it stalled at
                # ~3 workers while 184 tasks sat queued. Ask for a real floor.
                "min_workers": 25,
                "max_workers": 200,
                "dataset_location": "coffea4bees/metadata/datasets/",
                "datasets_file": params.ds_file,
            },
            "config": {
                "blind": False,
                "apply_FvT": False,
                "apply_JCM": False,
                "apply_trigWeight": False,
                "apply_btagSF": True,
                "run_SvB": False,
                "top_reconstruction": "fast",
                "fill_histograms": True,
                "make_classifier_input": params.inputs_base,
            }
        }
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule make_classifier_inputs_mixeddata:
    input:
        analysis_cfg = f"{out}classifier_inputs_config_mixeddata.yml",
        ds_ready = config['multisample_install_path'],
    output:
        coffea_out = f"{out}histAll_{channel}_mixeddata__{config['multisample_dataset_name']}.coffea",
        json_out = config['classifier_inputs_json'],
    log:
        f"{out}logs/make_classifier_inputs_{channel}_mixeddata.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = config['multisample_dataset_name'],
        output_path = out,
        years = " ".join(YEARS),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        ./run_container python runner.py {input.analysis_cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            --shared-dask --condor 2>&1 | tee {log}
        """

rule create_classifier_inputs_config_subsample:
    output:
        f"{out}classifier_inputs_config_subsample_v{{v}}.yml"
    params:
        channel = channel,
        inputs_base = config['classifier_inputs_base'],
        ds_file = config['multisample_install_path'],
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "runner": {
                "workers": 4,
                # Adaptive scaling starts from min_workers and ramps lazily; with a
                # shared cluster and several concurrent subsample jobs it stalled at
                # ~3 workers while 184 tasks sat queued. Ask for a real floor.
                "min_workers": 25,
                "max_workers": 200,
                "dataset_location": "coffea4bees/metadata/datasets/",
                "datasets_file": params.ds_file,
            },
            "config": {
                "blind": False,
                "apply_FvT": False,
                "apply_JCM": False,
                "apply_trigWeight": False,
                "apply_btagSF": True,
                "run_SvB": False,
                "top_reconstruction": "fast",
                "fill_histograms": True,
                "make_classifier_input": params.inputs_base,
            }
        }
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule make_classifier_inputs_subsample:
    input:
        analysis_cfg = f"{out}classifier_inputs_config_subsample_v{{v}}.yml",
        ds_ready = config['multisample_install_path'],
    output:
        coffea_out = f"{out}classifier_inputs/histAll_{channel}_mixeddata_v{{v}}.coffea",
        json_out = f"{out}classifier_inputs/histAll_{channel}_mixeddata_v{{v}}.json",
    log:
        f"{out}logs/make_classifier_inputs_{channel}_mixeddata_v{{v}}.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = lambda wildcards: f"{config['multisample_dataset_name']}:{wildcards.v}",
        output_path = f"{out}classifier_inputs/",
        years = " ".join(YEARS),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea_out}) $(dirname {log})
        ./run_container python runner.py {input.analysis_cfg} \
            --processor {params.processor} \
            --datasets {params.dataset} \
            --years {params.years} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea_out}) \
            --shared-dask --condor 2>&1 | tee {log}
        """

rule update_classifier_inputs_subsample_json:
    input:
        json_in = f"{out}classifier_inputs/histAll_{channel}_mixeddata_v{{v}}.json",
    output:
        done = f"{out}classifier_inputs/update_json_v{{v}}.done"
    params:
        target_json = config['classifier_inputs_json'],
        v = "{v}",
    run:
        import json, os, re
        with open(input.json_in) as f:
            new_data = json.load(f)

        target = params.target_json
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            try:
                with open(target) as f:
                    curr = json.load(f)
            except Exception:
                curr = {"HCR_input": {"name": "HCR_input", "branches": [], "data": []}}
        else:
            curr = {"HCR_input": {"name": "HCR_input", "branches": [], "data": []}}

        if "HCR_input" in new_data:
            new_branches = new_data["HCR_input"].get("branches", [])
            new_entries = new_data["HCR_input"].get("data", [])

            # Filter out existing entries for this subsample if re-running.
            # The subsample identity lives in the friend-tree path (.../mixeddata/mix_v<N>_<era>/...),
            # not in the source picoAOD path: shared ttbar PSData sources carry no _v<N> marker at
            # all, so matching on e[0]["path"] alone never evicts them and they accumulate on
            # every re-run. Match the friend path first, and fall back to the source path
            # (covering both picoAOD_mixed_v<N>.root and the pre-hadd picoAOD_mixed_v<N>.chunk*.root).
            _v = str(params.v)
            _friend_re = re.compile(rf"/mix_v{_v}_")
            _source_re = re.compile(rf"_v{_v}(\.chunk\d+)?\.root$|/v{_v}/")

            def _is_this_subsample(entry):
                try:
                    friend_path = entry[1][0]["chunk"]["path"]
                except (IndexError, KeyError, TypeError):
                    friend_path = ""
                if _friend_re.search(friend_path):
                    return True
                return bool(_source_re.search(entry[0]["path"]))

            old_entries = [e for e in curr.get("HCR_input", {}).get("data", []) if not _is_this_subsample(e)]
            curr["HCR_input"]["branches"] = list(set(curr.get("HCR_input", {}).get("branches", []) + new_branches))
            curr["HCR_input"]["data"] = old_entries + new_entries

        with open(target, "w") as f:
            json.dump(curr, f, indent=2)

        with open(output.done, "w") as f:
            f.write("done\n")

rule merge_all_classifier_inputs_subsamples_json:
    input:
        jsons = expand(f"{out}classifier_inputs/histAll_{{channel}}_mixeddata_v{{v}}.json", channel=[channel], v=range(N_SUBSAMPLES))
    output:
        done = f"{out}classifier_inputs/merge_all_subsamples.done"
    params:
        target_json = config['classifier_inputs_json'],
        n_samples = N_SUBSAMPLES,
    run:
        import json, os, re
        target = params.target_json
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            try:
                with open(target) as f:
                    curr = json.load(f)
            except Exception:
                curr = {"HCR_input": {"name": "HCR_input", "branches": [], "data": []}}
        else:
            curr = {"HCR_input": {"name": "HCR_input", "branches": [], "data": []}}

        all_branches = set(curr.get("HCR_input", {}).get("branches", []))
        # Keep non-mixeddata entries. As in update_classifier_inputs_subsample_json, the
        # subsample identity is carried by the friend-tree path (.../mixeddata/mix_v<N>_<era>/...);
        # shared ttbar PSData sources have no _v<N> marker in their source path, so a source-only
        # match leaves them behind and they pile up each time this rule re-runs.
        _friend_any_re = re.compile(r"/mix_v\d+_")
        _source_any_re = re.compile(r"_v\d+(\.chunk\d+)?\.root$|/v\d+/")

        def _is_any_subsample(entry):
            try:
                friend_path = entry[1][0]["chunk"]["path"]
            except (IndexError, KeyError, TypeError):
                friend_path = ""
            if _friend_any_re.search(friend_path):
                return True
            return bool(_source_any_re.search(entry[0]["path"]))

        all_entries = [e for e in curr.get("HCR_input", {}).get("data", []) if not _is_any_subsample(e)]

        for jf in input.jsons:
            if not os.path.exists(jf):
                continue
            with open(jf) as f:
                d = json.load(f)
            if "HCR_input" in d:
                all_branches.update(d["HCR_input"].get("branches", []))
                for entry in d["HCR_input"].get("data", []):
                    all_entries.append(entry)

        # Guard against the same (source file, friend chunk) pair being listed twice, which
        # would double-count those events in training.
        _seen = set()
        _deduped = []
        for entry in all_entries:
            try:
                key = (entry[0]["path"], entry[1][0]["chunk"]["path"],
                       entry[1][0].get("start"), entry[1][0].get("stop"))
            except (IndexError, KeyError, TypeError):
                key = None
            if key is not None:
                if key in _seen:
                    continue
                _seen.add(key)
            _deduped.append(entry)
        all_entries = _deduped

        curr["HCR_input"]["branches"] = sorted(list(all_branches))
        curr["HCR_input"]["data"] = all_entries

        with open(target, "w") as f:
            json.dump(curr, f, indent=2)

        with open(output.done, "w") as f:
            f.write("done\n")


# ── Stage 7: SvB Friend Trees Evaluation (4 Years Distributed & Merged) ───────
if "_SNAKEFILE_EVAL_FRIENDS_INCLUDED" not in globals():
    _SNAKEFILE_EVAL_FRIENDS_INCLUDED = True
    include: "Snakefile_eval_friends_mixeddata_ttHbb.smk"

