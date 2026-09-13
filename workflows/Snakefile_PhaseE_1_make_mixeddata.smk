# coffea4bees/workflows/Snakefile_PhaseE_1_make_mixeddata.smk
# Phase E_1: Mixed-Data Production & Inclusive JCM Calibration for Run 2 & Run 3
#
# Generates base mixed data at rank (0,0), derives the inclusive Jet Combinatoric Model (JCM)
# against 4b Data minus ttbar, evaluates mixed data with the calibrated JCM, and generates
# complete closure validation plots.

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
config.setdefault('analysis_container', None)
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
config.setdefault('output_path', f"output/ttHbb_mixeddata_stitched_closure/")
config.setdefault('base_path',
    f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/{run_period}/{channel}_pz{_rank_suffix}")

# Dataset naming
config.setdefault('dataset_name', f"mixeddata_{channel}{_rank_suffix}")
config.setdefault('install_path', f"coffea4bees/metadata/datasets/mixeddata_{channel}{_rank_suffix}.yml")

out = config['output_path']
if not out.endswith("/"):
    out += "/"

# ── JCM Inclusive Fit Setup (Phase B unmixed JCM for mixing) ───────────────────
module analysis:
    snakefile: "rules/analysis.smk"
    config: config

jcm_source_coffea = jcm_cfg.get('source_coffea', config.get('jcm_source_coffea', "output/ttHbb_stitched/computeJCM/histAll_NoJCM.coffea"))
jcm_input_coffea  = jcm_cfg.get('input_coffea', config.get('jcm_input_coffea', f"{out}inputs/histAll_NoJCM.coffea"))
jcm_out_dir       = jcm_cfg.get('output_dir', config.get('jcm_output_dir', f"{out}JCM_inclusive/"))
if not jcm_out_dir.endswith('/'):
    jcm_out_dir += '/'
jcm_tag           = jcm_cfg.get('tag', config.get('tag', "ttHbb_stitched_inclusive"))
jcm_region        = jcm_cfg.get('region', config.get('jcm_region', "inclusive"))
jcm_model_file    = f"{jcm_out_dir}jetCombinatoricModel_{jcm_region}_{jcm_tag}.yml"

localrules: all_PhaseE_1, all_JCM_inclusive, stage_input_coffea, make_new_JCM, all_mixeddata_all, create_skimmer_config, merge_mixeddata_registries, install_mixeddata_dataset, compare_mixeddata_vs_data, create_analysis_config_mixeddata_all, create_plot_config_mixeddata_vs_data, make_plots_mixeddata_vs_data, create_mixeddata_jcm_config, create_mixeddata_jcm_plot_config, make_mixeddata_JCM, all_mixeddata_JCM, create_analysis_config_mixeddata_all_with_mixeddata_JCM, create_plot_config_mixeddata_closure, make_plots_mixeddata_closure, all_mixeddata_closure, all_run_mixeddata_all_with_mixeddata_JCM

# ── Default Master Target (Full Phase E1 End-to-End) ───────────────────────────
rule all_PhaseE_1:
    input:
        jcm_model_file,
        config['install_path'],
        f"{out}histAll_{channel}_mixeddata_all.coffea",
        f"{out}JCM_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
        f"{out}histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea",
        f"{out}plots_mixeddata_closure/plots_done.txt",

# ── Sub-Target Aliases ─────────────────────────────────────────────────────────
rule all_JCM_inclusive:
    input:
        jcm_model_file

rule all_mixeddata_all:
    input:
        config['install_path']

rule all_mixeddata_JCM:
    input:
        f"{out}JCM_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml"

rule all_run_mixeddata_all_with_mixeddata_JCM:
    input:
        f"{out}histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea"

rule all_mixeddata_closure:
    input:
        f"{out}plots_mixeddata_closure/plots_done.txt"

rule compare_mixeddata_vs_data:
    input:
        f"{out}plots_mixeddata_vs_data/plots_done.txt"

# ── Stage 1: Initial JCM for Mixing ───────────────────────────────────────────
rule stage_input_coffea:
    input:
        jcm_source_coffea
    output:
        jcm_input_coffea
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output})
        if [ ! -f "{output}" ]; then
            echo "Staging copy of {input} -> {output} (non-destructive)"
            cp "{input}" "{output}"
        fi
        """

use rule make_JCM from analysis as make_new_JCM with:
    input:
        jcm_input_coffea
    output:
        jcm_model_file
    params:
        extra_arguments = config.get('jcm_extra_arguments', jcm_cfg.get('extra_arguments', "--jcm_config coffea4bees/analysis/jcm_tools/metadata/ttHbb_jcm_config.yml")),
        tag = jcm_tag,
        region = jcm_region,
        output_dir = jcm_out_dir,
        run_container_wrapper = config.get('analysis_container_wrapper', config.get('container_wrapper', './run_container')),
        python_bin = lambda wildcards: config.get("python_bin", "python")
    log:
        f"{jcm_out_dir}logs/make_JCM.log"

# ── Stage 2: Hemisphere Mixing Production ──────────────────────────────────────
rule create_skimmer_config:
    input:
        jcm_model = jcm_model_file
    output:
        f"{out}mixeddata_skimmer_config.yml"
    params:
        base_path = config['base_path'],
        subtract_ttbar = config.get('subtract_ttbar_with_weights', True),
        hemi_lib = config.get('hemi_library_yaml', 'coffea4bees/skimmer/metadata/hemisphere_library_noTT.yml'),
        hemi_stats = config.get('hemi_stats_path', 'coffea4bees/skimmer/metadata'),
        use_boost = config.get('use_boost_corrected_matching', True),
        default_rank = config.get('default_rank', 0),
        worker_memory = config.get('worker_memory', '6GB'),
        chunksize = config.get('chunksize', 100000),
        picosize = config.get('picosize', 100000),
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        skimmer_cfg = {
            "runner": {
                "workers": 4,
                "min_workers": 1,
                "max_workers": 200,
                "chunksize": int(params.chunksize),
                "picosize": int(params.picosize),
                "basketsize": 10000,
                "class_name": "HemiMixer",
                "data_tier": "picoAOD",
                "condor_cores": 1,
                "worker_memory": str(params.worker_memory),
                "condor_transfer_input_files": ["src", "coffea4bees/"],
                "allowlist_sites": ["T2_US_Nebraska", "T2_US_Purdue", "T3_US_FNALLPC", "T3_US_NotreDame"],
            },
            "config": {
                "base_path": str(params.base_path),
                "subtract_ttbar_with_weights": bool(params.subtract_ttbar),
                "apply_JCM": True,
                "JCM_file": str(input.jcm_model),
                "hemi_library_yaml": str(params.hemi_lib),
                "hemi_stats_path": str(params.hemi_stats),
                "use_boost_corrected_matching": bool(params.use_boost),
                "use_topk_matching": True,
                "k_neighbors": 10,
                "collision_mode": "retry",
                "default_rank": params.default_rank,
                "step": int(params.chunksize),
                "skip_collections": ["notCanJet", "canJet0", "canJet1", "canJet2", "canJet3"],
                "skip_branches": [
                    "btagWeight_.*", "HHSR", "ZZSR", "ZHSR", "SR", "st", "d01TruthMatch",
                    "nMuon_selected", "fourTag", "threeTag", "xW", "leadStM", "nSelJets",
                    "d02TruthMatch", "d12TruthMatch", "truthMatch", "pseudoTagWeight",
                    "ttbarWeight", "nIsoMuons", "xt", "weight", "aveAbsEtaOth", "xWbW",
                    "nAllNotCanJets", "dRjjOther", "xWt", "nPSTJets", "passXWt",
                    "d03TruthMatch", "xbW", "mcPseudoTagWeight", "d23TruthMatch", "m4j",
                    "sublStM", "dRjjClose", "aveAbsEta", "stNotCan", "SB",
                    "selectedViewTruthMatch", "d13TruthMatch"
                ]
            }
        }
        with open(output[0], 'w') as f:
            yaml.dump(skimmer_cfg, f, default_flow_style=False)

def get_hemi_stats_file(wildcards):
    year_str = wildcards.year.replace("_preVFP", "").replace("_postVFP", "")
    stats_dir = config.get('hemi_stats_path', 'coffea4bees/skimmer/metadata')
    return f"{stats_dir}/hemi_statistics_{year_str}.yml"

rule make_mixed_data_picoAOD_per_year:
    input:
        config_file = f"{out}mixeddata_skimmer_config.yml",
        hemi_lib = config.get('hemi_library_yaml', 'coffea4bees/skimmer/metadata/hemisphere_library_noTT.yml'),
        hemi_stats = get_hemi_stats_file,
    output:
        reg = f"{out}per_year/picoaod_datasets_{config['dataset_name']}__{{year}}.yml",
        done = f"{out}.make_mixed_data_{{year}}.done",
    log:
        f"{out}logs/make_mixeddata__{{year}}.log"
    params:
        friends = config.get('friends_file', 'coffea4bees/metadata/friends/friends_ttHbb.yml'),
        processor = "coffea4bees/skimmer/processor/make_mixed_data.py",
        dataset = "data",
        output_path = f"{out}per_year/",
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.reg}) $(dirname {log})
        ./run_container python runner.py {input.config_file} \
            -p {params.processor} \
            -d {params.dataset} \
            --friends {params.friends} \
            --years {wildcards.year} \
            --output-path {params.output_path} \
            --output $(basename {output.reg}) \
            -s --shared-dask --condor 2>&1 | tee {log}
        touch {output.done}
        """

rule merge_mixeddata_registries:
    input:
        expand(f"{out}per_year/picoaod_datasets_{config['dataset_name']}__{{year}}.yml", year=YEARS)
    output:
        f"{out}picoaod_datasets_{config['dataset_name']}_combined.yml"
    log:
        f"{out}logs/merge_mixeddata_registries.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        python coffea4bees/workflows/scripts/merge_mixeddata_registries.py {input} {output} 2>&1 | tee -a {log}
        """

rule install_mixeddata_dataset:
    input:
        f"{out}picoaod_datasets_{config['dataset_name']}_combined.yml"
    output:
        config['install_path']
    log:
        f"{out}logs/install_mixeddata_dataset.log"
    params:
        name = config['dataset_name']
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        python src/tools/make_dataset_yml.py -i {input} -o {output} -n {params.name} 2>&1 | tee -a {log}
        echo "Installed {output} (dataset name: {params.name})" 2>&1 | tee -a {log}
        """

# ── Stage 3: Process Raw Mixed Data (Fill Unweighted Distributions) ───────────
rule create_analysis_config_mixeddata_all:
    output:
        f"{out}analysis_config_mixeddata_all.yml"
    params:
        ds_file = config['install_path'],
        dataset_location = config['dataset_location'],
        jcm_file = jcm_model_file,
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
                "blind": False,
                "apply_FvT": False,
                "apply_JCM": True,
                "JCM_file": params.jcm_file,
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

rule run_analysis_mixeddata_all:
    input:
        ds_ready = config['install_path'],
        jcm_ready = jcm_model_file,
        analysis_cfg = f"{out}analysis_config_mixeddata_all.yml",
    output:
        coffea_out = f"{out}histAll_{channel}_mixeddata_all.coffea",
    log:
        f"{out}logs/run_analysis_mixeddata_all.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = config['dataset_name'],
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

# ── Stage 4: Mixed-Data Inclusive JCM Derivation ──────────────────────────────
rule create_mixeddata_jcm_config:
    output:
        f"{out}jcm_mixeddata_config.yml"
    params:
        dataset_name = config['dataset_name'],
        ttbar_processes = config.get('ttbar_processes', ['TTTo2L2Nu_stitched', 'TTToSemiLeptonic_stitched', 'TTToHadronic_stitched']),
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        cfg = {
            "data4bName": "data",
            "taglabel4b": "fourTag",
            "data3bName": params.dataset_name,
            "taglabel3b": "fourTag",
            "taglabel3b_tt": "threeTag",
            "selJets": "selJets_noJCM.n",
            "tagJets": "tagJets_noJCM.n",
            "subtract3bTT": False,
            "ignoreTT": not bool(config.get('subtract_ttbar_with_weights', True)),
            "ttbarProcesses": list(params.ttbar_processes),
            "float_t": True,
        }
        with open(output[0], "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

rule create_mixeddata_jcm_plot_config:
    output:
        f"{out}plots_metadata_mixeddata_JCM.yml"
    params:
        dataset = config['dataset_name'],
        subtract_ttbar = config.get('subtract_ttbar_with_weights', True),
        ttbar_processes = config.get('ttbar_processes', ['TTTo2L2Nu_stitched', 'TTToSemiLeptonic_stitched', 'TTToHadronic_stitched']),
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        pcfg = {
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
            "stack": {},
            "ratios": {
                "dataToBkg": {
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
                },
                "dataToJCM": {
                    "numerator": {
                        "type": "hists",
                        "key": "data",
                    },
                    "denominator": {
                        "type": "hists",
                        "key": "JCM",
                    },
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
        if params.subtract_ttbar:
            for idx, proc in enumerate(params.ttbar_processes):
                pcfg["stack"][proc] = {
                    "process": proc,
                    "tag": "fourTag",
                    "year": "RunII",
                    "fillcolor": "#85D1FBff",
                    "edgecolor": "k" if idx == len(params.ttbar_processes) - 1 else "#85D1FBff",
                    "label": "TTbar" if idx == len(params.ttbar_processes) - 1 else "None",
                }
        pcfg["stack"]["MultiJet"] = {
            "process": params.dataset,
            "tag": "fourTag",
            "year": "RunII",
            "fillcolor": "orange",
            "edgecolor": "k",
            "label": "Multijet (mixed)",
            "scalefactor": 1.0,
        }
        with open(output[0], "w") as f:
            yaml.dump(pcfg, f, default_flow_style=False)

rule make_mixeddata_JCM:
    input:
        data_tt_coffea = jcm_input_coffea,
        mixed_coffea = f"{out}histAll_{channel}_mixeddata_all.coffea",
        jcm_cfg = f"{out}jcm_mixeddata_config.yml",
        plot_cfg = f"{out}plots_metadata_mixeddata_JCM.yml",
    output:
        model_file = f"{out}JCM_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
        seljets_plot = f"{out}JCM_mixeddata_inclusive/selJets_noJCM_n.png",
        tagjets_plot = f"{out}JCM_mixeddata_inclusive/tagJets_noJCM_n.png",
    log:
        f"{out}JCM_mixeddata_inclusive/logs/make_mixeddata_JCM.log"
    params:
        output_dir = f"{out}JCM_mixeddata_inclusive/",
        region = "inclusive",
        tag = f"{channel}_mixeddata",
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR $(dirname {output.model_file}) $(dirname {log})
        
        echo "Computing JCM for ttbar + mixeddata vs data 4b inclusive" 2>&1 | tee {log}
        ./run_container env PYTHONPATH=. python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
            -o {params.output_dir} \
            -r {params.region} \
            -i {input.data_tt_coffea} {input.mixed_coffea} \
            --jcm_config {input.jcm_cfg} \
            -m {input.plot_cfg} \
            -w {params.tag} 2>&1 | tee -a {log}
        ls -la {params.output_dir}
        """

# ── Stage 5: Evaluate Mixed Data with Calibrated JCM ──────────────────────────
rule create_analysis_config_mixeddata_all_with_mixeddata_JCM:
    input:
        jcm_file = f"{out}JCM_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
    output:
        f"{out}analysis_config_mixeddata_all_with_mixeddata_JCM.yml"
    params:
        ds_file = config['install_path'],
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
                "datasets_file": params.ds_file,
                "dataset_location": params.dataset_location,
            },
            "config": {
                "blind": False,
                "apply_FvT": False,
                "apply_JCM": True,
                "JCM_file": input.jcm_file,
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

rule run_analysis_mixeddata_all_with_mixeddata_JCM:
    input:
        ds_ready = config['install_path'],
        jcm_ready = f"{out}JCM_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
        analysis_cfg = f"{out}analysis_config_mixeddata_all_with_mixeddata_JCM.yml",
    output:
        coffea_out = f"{out}histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea",
    log:
        f"{out}logs/run_analysis_mixeddata_all_with_mixeddata_JCM.log"
    params:
        processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = config['dataset_name'],
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

# ── Stage 6: Mixed-Data Closure Plots (Full Stack vs Data 4b) ─────────────────
rule create_plot_config_mixeddata_closure:
    input:
        jcm_file = f"{out}JCM_mixeddata_inclusive/jetCombinatoricModel_inclusive_{channel}_mixeddata.yml",
    output:
        f"{out}plots_metadata_mixeddata_closure.yml"
    params:
        dataset = config['dataset_name'],
        ttbar_processes = config.get('ttbar_processes', ['TTTo2L2Nu_stitched', 'TTToSemiLeptonic_stitched', 'TTToHadronic_stitched']),
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(input.jcm_file, "r") as f:
            jcm_data = yaml.safe_load(f)
        mu_qcd = float(jcm_data.get("mu_qcd", 0.1251))

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
                "mixeddata": {
                    "process": [f"{params.dataset}{e}" for e in ["A", "B", "C", "D", "E", "F", "G", "H"]] + [params.dataset],
                    "tag": "fourTag",
                    "fillcolor": "orange",
                    "edgecolor": "k",
                    "label": "Mixed data (JCM weighted)",
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
        if config.get('subtract_ttbar_with_weights', True):
            pcfg["stack"]["TTbar"] = {
                "process": list(params.ttbar_processes),
                "tag": "fourTag",
                "fillcolor": "#85D1FBff",
                "edgecolor": "k",
                "label": "TTbar",
            }
        with open(output[0], "w") as f:
            yaml.dump(pcfg, f, default_flow_style=False)

rule make_plots_mixeddata_closure:
    input:
        data_coffea = jcm_input_coffea,
        mixed_coffea = f"{out}histAll_{channel}_mixeddata_all_with_mixeddata_JCM.coffea",
        plot_cfg = f"{out}plots_metadata_mixeddata_closure.yml",
    output:
        done = f"{out}plots_mixeddata_closure/plots_done.txt",
    log:
        f"{out}logs/make_plots_mixeddata_closure.log"
    params:
        output_dir = f"{out}plots_mixeddata_closure/",
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        ./run_container python coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year RunII \
            -p 1 2>&1 | tee {log}
        touch {output.done}
        """

# ── Optional: Pre-JCM Comparison Plots vs Data 4b ─────────────────────────────
rule create_plot_config_mixeddata_vs_data:
    output:
        f"{out}plots_metadata_mixeddata_vs_data.yml"
    params:
        dataset = config['dataset_name'],
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
                "mixeddata": {
                    "process": [f"{params.dataset}{e}" for e in ["A", "B", "C", "D", "E", "F", "G", "H"]] + [params.dataset],
                    "tag": "fourTag",
                    "fillcolor": "orange",
                    "edgecolor": "k",
                    "label": "Mixed data (all)",
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

rule make_plots_mixeddata_vs_data:
    input:
        data_coffea = jcm_input_coffea,
        mixed_coffea = f"{out}histAll_{channel}_mixeddata_all.coffea",
        plot_cfg = f"{out}plots_metadata_mixeddata_vs_data.yml",
    output:
        done = f"{out}plots_mixeddata_vs_data/plots_done.txt",
    log:
        f"{out}logs/make_plots_mixeddata_vs_data.log"
    params:
        output_dir = f"{out}plots_mixeddata_vs_data/",
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.done}) $(dirname {log})
        ./run_container python coffea4bees/plots/makePlots.py \
            {input.data_coffea} {input.mixed_coffea} \
            -o {params.output_dir} \
            -m {input.plot_cfg} \
            --combine_input_files \
            --year RunII 2>&1 | tee {log}
        touch {output.done}
        """
