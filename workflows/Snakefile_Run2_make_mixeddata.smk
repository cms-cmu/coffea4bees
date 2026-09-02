"""Mixed-data picoAOD production for Run 2 (per-year, parallel).

Runs runner.py -s on coffea4bees/skimmer/processor/make_mixed_data.py
with one HTCondor submission per Run 2 year (UL16_preVFP, UL16_postVFP, UL17, UL18),
merges the per-year picoAOD registries into a combined registry,
installs the validation dataset YAML, and runs processor_study_mixed_data.py
to produce validation histograms for comparison against nominal mixeddata_4b.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \
        --snakefile coffea4bees/workflows/Snakefile_Run2_make_mixeddata.smk \
        --cores 4
"""

import shutil
import ast as _ast

config.setdefault('analysis_container',
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',
    "coffea4bees/metadata/datasets/")
config.setdefault('years',
    ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
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
else:
    _rank_suffix = f"_rank{int(_rank)}_{int(_rank)}"

config.setdefault('output_path', f"output/Run2_mixeddata{_rank_suffix}/")
config.setdefault('base_path',
    f"root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata_Run2{_rank_suffix}")
config.setdefault('dataset_name', f"mixeddata_Run2{_rank_suffix}")
config.setdefault('install_path',
    f"coffea4bees/metadata/datasets/mixeddata_Run2{_rank_suffix}.yml")

_roc = config.setdefault('run_on_condor',
    shutil.which("condor_submit") is not None)
config['run_on_condor'] = str(_roc).lower() not in ('false', '0', 'no')

out         = config['output_path']
SKIMMER_CFG = "coffea4bees/skimmer/metadata/mixeddata.yml"
HEMI_LIB    = config.setdefault('hemi_lib',
    "coffea4bees/skimmer/metadata/hemisphere_library_noTT.yml")
HEMI_STATS_DIR = config.setdefault('hemi_stats_dir',
    "coffea4bees/skimmer/metadata")
STUDY_CFG   = "coffea4bees/analysis/metadata/study_mixed_data.yml"
REGISTRY    = f"picoaod_datasets_mixeddata_Run2{_rank_suffix}.yml"

module analysis:
    snakefile: "rules/analysis.smk"
    config: config

rule all:
    input:
        f"{out}study_mixeddata_all{_rank_suffix}.coffea",

config.setdefault('worker_memory', '6GB')
config.setdefault('chunksize', 100000)

rule patch_skimmer_config:
    input:  SKIMMER_CFG
    output: f"{out}mixeddata_Run2.yml"
    params:
        base_path     = config['base_path'],
        hemi_library  = HEMI_LIB,
        hemi_stats    = HEMI_STATS_DIR,
        worker_memory = config['worker_memory'],
        chunksize     = config['chunksize'],
        default_rank  = (f"[{_rank[0]}, {_rank[1]}]"
                         if isinstance(_rank, (list, tuple))
                         else str(int(_rank))),
    shell:
        """
        mkdir -p $(dirname {output})
        sed -e 's|  base_path:.*|  base_path: {params.base_path}|' \
            -e 's|  default_rank:.*|  default_rank: {params.default_rank}|' \
            -e 's|  hemi_library_yaml:.*|  hemi_library_yaml: {params.hemi_library}|' \
            -e 's|  hemi_stats_path:.*|  hemi_stats_path: {params.hemi_stats}|' \
            -e 's|  worker_memory:.*|  worker_memory: {params.worker_memory}|' \
            -e 's|  chunksize:.*|  chunksize: {params.chunksize}|' \
            -e 's|  step:.*|  step: {params.chunksize}|' \
            {input} > {output}
        echo "Patched Run 2 skimmer config:"
        grep -E "base_path|default_rank|hemi_library_yaml|hemi_stats_path|worker_memory|chunksize|step" {output}
        """

def get_hemi_stats_file(wildcards):
    year = wildcards.year
    year_str = year.replace("_preVFP", "").replace("_postVFP", "")
    return f"{HEMI_STATS_DIR}/hemi_statistics_{year_str}.yml"

use rule analysis_processor from analysis as make_mixeddata with:
    input:
        config_file = f"{out}mixeddata_Run2.yml",
        hemi_lib    = HEMI_LIB,
        hemi_stats  = get_hemi_stats_file,
    output: f"{out}per_year/picoaod_datasets_mixeddata_Run2__{{year}}.yml"
    log:    f"{out}logs/make_mixeddata__{{year}}.log"
    wildcard_constraints:
        year = "|".join(config['years'])
    params:
        datasets              = "data",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/skimmer/processor/make_mixed_data.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = "coffea4bees/metadata/friends/friends_HH4b.yml",
        run_on_condor         = config['run_on_condor'],
        weights               = config.get('weights_file', "coffea4bees/metadata/weights/weights_HH4b.yml"),
        extra_arguments       = "-s -p coffea4bees/skimmer/processor/make_mixed_data.py --shared-dask --condor" if config['run_on_condor'] else "-s -p coffea4bees/skimmer/processor/make_mixed_data.py",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

rule merge_mixeddata_registries:
    input:
        expand(
            f"{out}per_year/picoaod_datasets_mixeddata_Run2__{{year}}.yml",
            year=config['years'],
        )
    output: f"{out}{REGISTRY}"
    log: f"{out}logs/merge_mixeddata_registries.log"
    shell:
        """
        python coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input} {output} 2>&1 | tee -a {log}
        """

rule install_mixeddata_dataset:
    input:  f"{out}{REGISTRY}"
    output: config['install_path']
    container: config['analysis_container']
    params:
        name = config['dataset_name']
    log: f"{out}logs/install_mixeddata_dataset.log"
    shell:
        """
        mkdir -p $(dirname {output})
        python src/tools/make_dataset_yml.py \
            -i {input} \
            -o {output} \
            -n {params.name} 2>&1 | tee -a {log}
        echo "Installed {output} (dataset name: {params.name})" 2>&1 | tee -a {log}
        """

use rule analysis_processor from analysis as study_mixeddata with:
    input:
        config_file = STUDY_CFG,
        dataset_yml = config['install_path'],
    output: f"{out}study_mixeddata/study_mixeddata__{{year}}.coffea"
    log:    f"{out}logs/study_mixeddata__{{year}}.log"
    wildcard_constraints:
        year = "|".join(config['years'])
    params:
        datasets              = config['dataset_name'],
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/analysis/processors/processor_study_mixed_data.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = "coffea4bees/metadata/friends/friends_HH4b.yml",
        run_on_condor         = config['run_on_condor'],
        weights               = config.get('weights_file', "coffea4bees/metadata/weights/weights_HH4b.yml"),
        extra_arguments       = "-p coffea4bees/analysis/processors/processor_study_mixed_data.py --shared-dask --condor" if config['run_on_condor'] else "-p coffea4bees/analysis/processors/processor_study_mixed_data.py",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0

use rule merging_coffea_files from analysis as merge_study_mixeddata with:
    input:
        expand(
            f"{out}study_mixeddata/study_mixeddata__{{year}}.coffea",
            year=config['years'],
        )
    output: f"{out}study_mixeddata_all{_rank_suffix}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False,
        run_container_wrapper = ""
    log: f"{out}logs/merge_study_mixeddata.log"
