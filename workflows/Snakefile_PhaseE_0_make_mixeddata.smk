# coffea4bees/workflows/Snakefile_PhaseE_0_make_mixeddata.smk
# Phase E_0: Generic Mixed-Data Production for Run 2 & Run 3
#
# Generates base mixed data at rank (0,0), splits into 15 subsample datasets (v0..v14),
# and provides independent targets for validation (study), JCM fitting, and friend trees.

import os
import shutil
import ast as _ast

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb.yml"

include: "helpers/common.smk"

# Resolve phase_e configuration block if present
phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure', 'mixeddata'])
for k, v in phase_e_cfg.items():
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
_full_tag_suffix = f"{_tag_suffix}{_rank_suffix}"

# Physics knobs: pz boost, ttbar subtraction with FvT, and ttHbb JCM
config.setdefault('use_boost_corrected_matching', True)
config.setdefault('subtract_ttbar_with_weights', True)

# Output paths
config.setdefault('output_path', f"output/{run_period}_mixeddata{_full_tag_suffix}/")
out = config['output_path'].rstrip("/") + "/"
config.setdefault('base_path', f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/mixeddata_{run_period}{_full_tag_suffix}")
eos_base = config['base_path'].rstrip("/")

# Base mixed-data dataset names
REGISTRY_BASE = f"picoaod_datasets_mixeddata_{run_period}_noTT_pz{_full_tag_suffix}.yml"
config.setdefault('dataset_name', f"mixeddata_all{_full_tag_suffix}")
config.setdefault('install_path', f"coffea4bees/metadata/datasets/mixeddata_all{_full_tag_suffix}.yml")
config.setdefault('multisample_dataset_name', "mixeddata_4b")
config.setdefault('multisample_install_path', "coffea4bees/metadata/datasets/mixeddata_4b.yml")

# Subsamples (15 by default for closure studies)
N_SAMPLES = int(config.get('n_samples', 15))
SAMPLES = [f"v{i}" for i in range(N_SAMPLES)]

# Defaults per run period
if is_run3:
    config.setdefault('skimmer_config', "coffea4bees/skimmer/metadata/mixeddata_Run3.yml")
    config.setdefault('hemi_lib', "coffea4bees/skimmer/metadata/hemisphere_library_Run3_noTT_pt25.yml")
    config.setdefault('hemi_stats_dir', "coffea4bees/skimmer/metadata/hemi_statistics_noTT_pt25")
    config.setdefault('study_config', "coffea4bees/analysis/metadata/study_mixed_data_Run3.yml")
    config.setdefault('jcm_config', "coffea4bees/analysis/jcm_tools/metadata/mixeddata_all_config_Run3.yml")
    config.setdefault('jcm_file', "coffea4bees/metadata/weights/JCM/Run3/jetCombinatoricModel_SB_v2.yml")
    config.setdefault('friends_file', f"coffea4bees/metadata/friends/friends_{channel}.yml")
    config.setdefault('weights_file', f"coffea4bees/metadata/weights/weights_{channel}.yml")
else:
    config.setdefault('skimmer_config', "coffea4bees/skimmer/metadata/mixeddata.yml")
    config.setdefault('hemi_lib', "coffea4bees/skimmer/metadata/hemisphere_library_noTT.yml")
    config.setdefault('hemi_stats_dir', "coffea4bees/skimmer/metadata")
    config.setdefault('study_config', "coffea4bees/analysis/metadata/study_mixed_data.yml")
    config.setdefault('jcm_config', "coffea4bees/analysis/jcm_tools/metadata/mixeddata_all_config.yml")
    config.setdefault('jcm_file', "coffea4bees/metadata/weights/JCM/Run2_ttHbb/jetCombinatoricModel_SB_2024_v2.yml" if channel == "ttHbb" else "coffea4bees/metadata/weights/JCM/jetCombinatoricModel_SB_v2.yml")
    config.setdefault('friends_file', f"coffea4bees/metadata/friends/friends_{channel}.yml")
    config.setdefault('weights_file', f"coffea4bees/metadata/weights/weights_{channel}.yml")

config['friends_file'] = f"coffea4bees/metadata/friends/friends_{channel}.yml"
config['weights_file'] = f"coffea4bees/metadata/weights/weights_{channel}.yml"

config.setdefault('split_skimmer_config', "coffea4bees/skimmer/metadata/split_mixeddata.yml")
config.setdefault('worker_memory', '8GB')
config.setdefault('chunksize', 100000)
config.setdefault('mode', 'nominal')
config.setdefault('float_t', True)

_roc = config.setdefault('run_on_condor', shutil.which("condor_submit") is not None)
run_on_condor = str(_roc).lower() not in ('false', '0', 'no')
config['run_on_condor'] = run_on_condor

SVB_FRIEND_JSON = f"coffea4bees/metadata/friends/friends_{channel}_mixeddata{_full_tag_suffix}.json" if not is_run3 else f"coffea4bees/metadata/friends/data_SvBfriend{_full_tag_suffix}.json"
FEYNET_FRIEND_JSON = f"coffea4bees/metadata/friends/SvBFeynNetfriend_mixeddata_data{_full_tag_suffix}.json"

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

# Target rule: default Phase E_0 target creates the 15-subsample dataset and the study coffea
rule all_PhaseE_0:
    input:
        config['multisample_install_path'],
        f"{out}study_mixeddata_all{_full_tag_suffix}.coffea"

# Specific modular target rules
rule make_base_mixeddata:
    input: config['install_path']

rule split_mixeddata_all:
    input: config['multisample_install_path']

rule study_mixeddata_all:
    input: f"{out}study_mixeddata_all{_full_tag_suffix}.coffea"

rule fit_jcm_all:
    input: f"{out}jcm_{config['mode']}{_full_tag_suffix}/jetCombinatoricModel_SB_.yml"

rule make_friends_all:
    input:
        [SVB_FRIEND_JSON, FEYNET_FRIEND_JSON] if is_run3 else [SVB_FRIEND_JSON]


# ==============================================================================
# 1. Base Mixed-Data Production (Rank 0,0)
# ==============================================================================

rule patch_skimmer_config:
    input: config['skimmer_config']
    output: f"{out}mixeddata_{run_period}.yml"
    params:
        base_path      = eos_base,
        hemi_library   = config['hemi_lib'],
        hemi_stats     = config['hemi_stats_dir'],
        worker_memory  = config['worker_memory'],
        chunksize      = config['chunksize'],
        default_rank   = _rank_tuple,
        boost_match    = bool(config['use_boost_corrected_matching']),
        subtract_ttbar = bool(config['subtract_ttbar_with_weights']),
    run:
        import yaml
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(input[0], "r") as f:
            cfg = yaml.safe_load(f) or {}
        cfg_runner = cfg.setdefault("runner", {})
        cfg_runner["worker_memory"] = params.worker_memory
        cfg_runner["chunksize"] = params.chunksize
        cfg_runner["picosize"] = params.chunksize
        cfg_sec = cfg.setdefault("config", {})
        cfg_sec["base_path"] = params.base_path
        cfg_sec["default_rank"] = params.default_rank
        cfg_sec["hemi_library_yaml"] = params.hemi_library
        cfg_sec["hemi_stats_path"] = params.hemi_stats
        cfg_sec["step"] = params.chunksize
        cfg_sec["subtract_ttbar_with_weights"] = params.subtract_ttbar
        cfg_sec["use_boost_corrected_matching"] = params.boost_match
        with open(output[0], "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)

def get_hemi_stats_file(wildcards):
    year = wildcards.year
    if is_run3:
        return f"{config['hemi_stats_dir']}/hemi_statistics_{year}.yml"
    year_str = year.replace("_preVFP", "").replace("_postVFP", "")
    return f"{config['hemi_stats_dir']}/hemi_statistics_{year_str}.yml"

use rule analysis_processor from analysis as make_mixeddata with:
    input:
        config_file = f"{out}mixeddata_{run_period}.yml",
        hemi_lib    = config['hemi_lib'],
        hemi_stats  = get_hemi_stats_file,
    output: f"{out}per_year/picoaod_datasets_mixeddata_{run_period}_noTT_pz__{{year}}.yml"
    log:    f"{out}logs/make_mixeddata__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets              = "data",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/skimmer/processor/make_mixed_data.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config['friends_file'],
        run_on_condor         = config['run_on_condor'],
        weights               = config['weights_file'],
        extra_arguments       = f"-s -p coffea4bees/skimmer/processor/make_mixed_data.py --friends {config['friends_file']} --weights {config['weights_file']} --shared-dask --condor" if config['run_on_condor'] else f"-s -p coffea4bees/skimmer/processor/make_mixed_data.py --friends {config['friends_file']} --weights {config['weights_file']}",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

rule merge_mixeddata_registries:
    input:
        expand(f"{out}per_year/picoaod_datasets_mixeddata_{run_period}_noTT_pz__{{year}}.yml", year=YEARS)
    output: f"{out}{REGISTRY_BASE}"
    log: f"{out}logs/merge_mixeddata_registries.log"
    shell:
        """
        python coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input} {output} 2>&1 | tee -a {log}
        """

rule install_mixeddata_dataset:
    input:  f"{out}{REGISTRY_BASE}"
    output: config['install_path']
    container: None
    params:
        name = config['dataset_name']
    log: f"{out}logs/install_mixeddata_dataset.log"
    shell:
        """
        mkdir -p $(dirname {output})
        ./run_container python src/tools/make_dataset_yml.py \
            -i {input} \
            -o {output} \
            -n {params.name} 2>&1 | tee -a {log}
        echo "Installed base mixed dataset: {output}" 2>&1 | tee -a {log}
        """


# ==============================================================================
# 2. 15-Subsample Splitting (v0..v14) from Base Rank (0,0) Mixed Data
# ==============================================================================

rule patch_split_skimmer_config:
    input: config['split_skimmer_config']
    output: f"{out}configs/split_mixeddata__{{sample}}.yml"
    params:
        eos_base      = f"{eos_base}_subsamples",
        worker_memory = config['worker_memory'],
        chunksize     = config['chunksize'],
        jcm_file      = config['jcm_file'],
    run:
        import yaml
        sample_id = wildcards.sample
        sub_idx = int(sample_id.replace("v", "")) if sample_id.replace("v", "").isdigit() else 0
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(input[0], "r") as f:
            cfg = yaml.safe_load(f) or {}
        cfg_runner = cfg.setdefault("runner", {})
        cfg_runner["worker_memory"] = params.worker_memory
        cfg_runner["chunksize"] = params.chunksize
        cfg_runner["picosize"] = params.chunksize
        cfg_sec = cfg.setdefault("config", {})
        cfg_sec["base_path"] = f"{params.eos_base}/{sample_id}"
        cfg_sec["mixed_subsample"] = sub_idx
        cfg_sec["n_subsamples"] = 16
        cfg_sec["JCM_file"] = params.jcm_file
        cfg_sec["apply_JCM"] = True
        with open(output[0], "w") as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)

use rule analysis_processor from analysis as split_mixeddata with:
    input:
        config_file = f"{out}configs/split_mixeddata__{{sample}}.yml",
        dataset_yml = config['install_path'],
    output: f"{out}per_year/picoaod_datasets_split__{{year}}__{{sample}}.yml"
    log:    f"{out}logs/split_mixeddata__{{year}}__{{sample}}.log"
    wildcard_constraints:
        year   = "|".join(YEARS),
        sample = "v[0-9]+"
    params:
        datasets              = config['dataset_name'],
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/skimmer/processor/split_mixed_data.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config['friends_file'],
        run_on_condor         = config['run_on_condor'],
        weights               = config['weights_file'],
        extra_arguments       = f"-s -p coffea4bees/skimmer/processor/split_mixed_data.py --friends {config['friends_file']} --weights {config['weights_file']} --shared-dask --condor" if config['run_on_condor'] else f"-s -p coffea4bees/skimmer/processor/split_mixed_data.py --friends {config['friends_file']} --weights {config['weights_file']}",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

rule merge_split_registries:
    input:
        expand(f"{out}per_year/picoaod_datasets_split__{{year}}__{{sample}}.yml", year=YEARS, sample="{sample}")
    output: f"{out}registries/picoaod_datasets_mixeddata__{{sample}}.yml"
    log: f"{out}logs/merge_split_registries__{{sample}}.log"
    wildcard_constraints:
        sample = "v[0-9]+"
    shell:
        """
        python coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input} {output} 2>&1 | tee -a {log}
        """

rule combine_subsample_datasets:
    input:
        expand(f"{out}registries/picoaod_datasets_mixeddata__{{sample}}.yml", sample=SAMPLES)
    output: config['multisample_install_path']
    log: f"{out}logs/combine_subsample_datasets.log"
    params:
        name = config['multisample_dataset_name']
    shell:
        """
        mkdir -p $(dirname {output})
        python coffea4bees/workflows/scripts/combine_mixeddata_datasets.py \
            --registries {input} \
            --output {output} \
            --dataset-name {params.name} 2>&1 | tee -a {log}
        echo "Installed multi-sample mixed dataset: {output}" 2>&1 | tee -a {log}
        """


# ==============================================================================
# 3. Validation: Study Mixed-Data Processor
# ==============================================================================

use rule analysis_processor from analysis as study_mixeddata with:
    input:
        config_file = config['study_config'],
        dataset_yml = config['install_path'],
    output: f"{out}study_mixeddata/study_mixeddata__{{year}}.coffea"
    log:    f"{out}logs/study_mixeddata__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets              = config['dataset_name'],
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/analysis/processors/processor_study_mixed_data.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config['friends_file'],
        run_on_condor         = config['run_on_condor'],
        weights               = config['weights_file'],
        extra_arguments       = f"-p coffea4bees/analysis/processors/processor_study_mixed_data.py --friends {config['friends_file']} --weights {config['weights_file']} --shared-dask --condor" if config['run_on_condor'] else f"-p coffea4bees/analysis/processors/processor_study_mixed_data.py --friends {config['friends_file']} --weights {config['weights_file']}",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

use rule merging_coffea_files from analysis as merge_study_mixeddata with:
    input:
        expand(f"{out}study_mixeddata/study_mixeddata__{{year}}.coffea", year=YEARS)
    output: f"{out}study_mixeddata_all{_full_tag_suffix}.coffea"
    container: None
    params:
        run_performance = False,
        run_container_wrapper = "./run_container"
    log: f"{out}logs/merge_study_mixeddata.log"


# ==============================================================================
# 4. Histograms & JCM Model Fitting
# ==============================================================================

SHARED_HISTS_OUT = f"output/{run_period}_mixeddata_shared_{config['mode']}{_tag_suffix}/"
SHARED_DATASETS  = ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data']

config.setdefault('histogram_config',
    "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml" if is_run3
    else "coffea4bees/analysis/metadata/HH4b_run_fastTopReco.yml")

rule create_histogram_config:
    input:  config['histogram_config']
    output: f"{SHARED_HISTS_OUT}histogram_config.yml"
    shell:
        """
        mkdir -p $(dirname {output})
        sed -e 's|  fourTag_use_tight:.*|  fourTag_use_tight: false|' \
            -e 's|  worker_memory:.*|  worker_memory: 8GB|' \
            -e 's|processor:.*|processor: coffea4bees/analysis/processors/processor_ttHbb.py|' \
            {input} > {output}
        grep -q "fourTag_use_tight" {output} || echo "  fourTag_use_tight: false" >> {output}
        grep -q "processor:" {output} || echo "processor: coffea4bees/analysis/processors/processor_ttHbb.py" >> {output}
        """

use rule analysis_processor from analysis as make_histograms_shared with:
    input:
        config_file = f"{SHARED_HISTS_OUT}histogram_config.yml",
    output: f"{SHARED_HISTS_OUT}histograms/hist_{{dataset}}__{{year}}.coffea"
    log:    f"{SHARED_HISTS_OUT}logs/hist_{{dataset}}__{{year}}.log"
    wildcard_constraints:
        dataset = "|".join(SHARED_DATASETS),
        year    = "|".join(YEARS),
    params:
        datasets              = "{dataset}",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = f"coffea4bees/analysis/processors/processor_{channel}.py" if os.path.exists(f"coffea4bees/analysis/processors/processor_{channel}.py") else "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config['friends_file'],
        run_on_condor         = config['run_on_condor'],
        weights               = config['weights_file'],
        extra_arguments       = "--shared-dask --condor" if config['run_on_condor'] else "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

use rule analysis_processor from analysis as make_histograms_mixeddata with:
    input:
        config_file = f"{SHARED_HISTS_OUT}histogram_config.yml",
        dataset_yml = config['install_path'],
    output: f"{out}histograms/hist_{{dataset}}__{{year}}.coffea"
    log:    f"{out}logs/hist_{{dataset}}__{{year}}.log"
    wildcard_constraints:
        dataset = config['dataset_name'],
        year    = "|".join(YEARS),
    params:
        datasets              = "{dataset}",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = f"coffea4bees/analysis/processors/processor_{channel}.py" if os.path.exists(f"coffea4bees/analysis/processors/processor_{channel}.py") else "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = config['friends_file'],
        run_on_condor         = config['run_on_condor'],
        weights               = config['weights_file'],
        extra_arguments       = "--shared-dask --condor" if config['run_on_condor'] else "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

use rule merging_coffea_files from analysis as merge_histograms with:
    input:
        expand(f"{SHARED_HISTS_OUT}histograms/hist_{{dataset}}__{{year}}.coffea", dataset=SHARED_DATASETS, year=YEARS),
        expand(f"{out}histograms/hist_{{dataset_name}}__{{year}}.coffea", dataset_name=config['dataset_name'], year=YEARS),
    output: f"{out}histAll_{config['mode']}{_full_tag_suffix}.coffea"
    container: None
    params:
        run_performance = False,
        run_container_wrapper = "./run_container"
    log: f"{out}logs/merge_histograms.log"

rule create_jcm_config:
    input:  config['jcm_config']
    output: f"{out}jcm_config_{config['mode']}.yml"
    params:
        dataset_name = config['dataset_name'],
        float_t      = "true" if str(config['float_t']).lower() in ('true', '1', 'yes') else "false",
    shell:
        """
        mkdir -p $(dirname {output})
        sed -e 's|data3bName:.*|data3bName: {params.dataset_name}|' \
            -e 's|^float_t:.*|float_t: {params.float_t}|' \
            {input} > {output}
        grep -q "^float_t:" {output} || printf '\nfloat_t: %s\n' "{params.float_t}" >> {output}
        """

rule fit_JCM:
    input:
        hist       = ancient(f"{out}histAll_{config['mode']}{_full_tag_suffix}.coffea"),
        jcm_config = f"{out}jcm_config_{config['mode']}.yml",
    output: f"{out}jcm_{config['mode']}{_full_tag_suffix}/jetCombinatoricModel_SB_.yml"
    container: None
    params:
        output_dir = f"{out}jcm_{config['mode']}{_full_tag_suffix}/",
        label      = f"{config['mode']}, rank={_rank_tuple}",
    log: f"{out}logs/fit_JCM.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR
        ./run_container env PYTHONPATH=. python coffea4bees/analysis/jcm_tools/make_jcm_weights.py \
            -o {params.output_dir} \
            -i {input.hist} \
            -r SB \
            --jcm_config {input.jcm_config} 2>&1 | tee -a {log}
        """


# ==============================================================================
# 5. Friend Trees (SvB & SvBFeynNet)
# ==============================================================================

if is_run3:
    module svb_friends:
        snakefile: "Snakefile_SvB_friendtrees_Run3.smk"
        config: config

    module feynet_friends:
        snakefile: "Snakefile_SvBFeynNet_friendtrees_Run3.smk"
        config: config

    use rule make_SvB_friendtrees_mixeddata  from svb_friends
    use rule make_SvB_friendtrees_HH         from svb_friends
    use rule merge_SvB_friendtrees_mixeddata from svb_friends
    use rule install_SvB_friend_json         from svb_friends

    use rule make_SvBFeynNet_friendtrees_data       from feynet_friends
    use rule make_SvBFeynNet_friendtrees_mixeddata  from feynet_friends
    use rule make_SvBFeynNet_friendtrees_ttbar      from feynet_friends
    use rule make_SvBFeynNet_friendtrees_HH         from feynet_friends
    use rule merge_SvBFeynNet_friendtrees           from feynet_friends
    use rule install_SvBFeynNet_friend_json         from feynet_friends
else:
    rule create_eval_config:
        output: f"{out}eval_config.yml"
        params:
            friend_base = f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/friends/{channel}/",
            weights_file = config['weights_file'],
        run:
            import yaml
            os.makedirs(os.path.dirname(output[0]), exist_ok=True)
            cfg = {
                "runner": {
                    "workers": 4,
                    "friend_base": params.friend_base,
                },
                "weights": params.weights_file,
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
                    "fill_histograms": False,
                    "make_friend_SvB": params.friend_base,
                }
            }
            with open(output[0], "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

    rule make_SvB_friendtrees_mixeddata_Run2:
        input:
            eval_cfg = f"{out}eval_config.yml",
            ds_ready = ancient(config['multisample_install_path']),
        output:
            f"{out}friends_json/friends_{{dataset}}__{{year}}.json"
        log:
            f"{out}logs/friends_{{dataset}}__{{year}}.log"
        params:
            processor = f"coffea4bees/analysis/processors/processor_{channel}.py",
            output_path = f"{out}friends_json/",
        shell:
            """
            set -eo pipefail
            mkdir -p $(dirname {output}) $(dirname {log})
            ./run_container python runner.py {input.eval_cfg} \
                --processor {params.processor} \
                --datasets {wildcards.dataset} \
                --years {wildcards.year} \
                --output-path {params.output_path} \
                --output $(basename {output} .json).coffea \
                --shared-dask --condor 2>&1 | tee {log}
            """

    rule merge_SvB_friendtrees_Run2:
        input:
            expand(f"{out}friends_json/friends_{{dataset}}__{{year}}.json",
                   dataset=[config['multisample_dataset_name']],
                   year=YEARS)
        output:
            SVB_FRIEND_JSON
        log:
            f"{out}logs/merge_SvB_friendtrees_Run2.log"
        shell:
            """
            set -eo pipefail
            mkdir -p $(dirname {output}) $(dirname {log})
            ./run_container python -m src.friendtrees.merge_friend_meta -i {input} -o {output} 2>&1 | tee {log}
            """

