"""Mixed-data picoAOD production for Run3 (per-year, parallel).

Wraps coffea4bees/scripts/mixeddata-make-dataset-Run3-all.sh as a Snakemake
workflow. Runs runner.py -s on coffea4bees/skimmer/processor/make_mixed_data.py
with one HTCondor submission per Run3 year, then dict-merges the per-year
picoAOD-registry YAMLs into a single combined registry matching the
historical script output.

The mixed-data step is independent of the quadjet variant, so this Snakefile
does not include rules/run3_variants.smk and has no `mode` config.

Three upstream artifacts are consumed but not produced here (they are
committed to git and rarely change):
  - coffea4bees/analysis/weights/JCM/Run3/jetCombinatoricModel_SB_v2.yml
      produced by JCM fitting (see Snakefile_Run3.smk:make_JCM_Run3 +
      install_JCM, fed from histAll_Run3.coffea)
  - coffea4bees/skimmer/metadata/hemisphere_library_Run3_noTT.yml
      produced by coffea4bees/scripts/mixeddata-cluster-Run3-all.sh
      (processor_make_hemi_library.py over Run3 data)
  - coffea4bees/skimmer/metadata/hemi_statistics_noTT/hemi_statistics_{year}.yml
      produced by coffea4bees/hemisphere_mixing/study_hemispheres.py
      (consumes the hemisphere library above; runs after it)
They are declared as `input:` so the DAG fails fast if any is missing and
so wiring their producer rules later requires no edits here.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \\
        --snakefile coffea4bees/workflows/Snakefile_Run3_make_mixeddata.smk \\
        --cores 4
"""

import shutil

config.setdefault('analysis_container',
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',
    "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('years',
    ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])
config.setdefault('default_rank', 0)  # 0 = nearest neighbor; >0 selects further neighbors

# Auto-namespace local outputs and EOS picoAOD location by rank to keep
# concurrent rank runs from clobbering each other. The _rank{N} suffix is
# always applied so directory names visibly self-document the rank used.
# An explicit --config output_path=... or base_path=... still wins.
_rank        = int(config['default_rank'])
_rank_suffix = f"_rank{_rank}"
config.setdefault('output_path', f"output/Run3_mixeddata{_rank_suffix}/")
config.setdefault('base_path',
    f"root://cmseos.fnal.gov//store/user/jda102/XX4b/mixed_data_all_noTT_pz{_rank_suffix}")
config.setdefault('dataset_name',  f"mixeddata_all{_rank_suffix}")
config.setdefault('install_path',
    f"coffea4bees/metadata/datasets_HH4b_Run3/mixeddata_all{_rank_suffix}.yml")

# Histogram-making variant. The skim/study upstream is variant-independent, but
# the histograms feeding JCM differ by quadjet strategy. 'nominal' uses the
# Run2-style quadjet selection; 'quadjet_run2' uses the Run3 reproduction of
# Run2 quadjet+tight definitions. fourTag_use_tight is force-disabled by the
# create_histogram_config rule below in BOTH modes.
config.setdefault('mode', 'nominal')
if config['mode'] == 'nominal':
    config.setdefault('histogram_config',
        "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml")
elif config['mode'] == 'quadjet_run2':
    config.setdefault('histogram_config',
        "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml")
else:
    raise ValueError(f"Unknown mode: {config['mode']!r}; expected 'nominal' or 'quadjet_run2'")
config.setdefault('histogram_datasets',
    ['TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu', 'data', config['dataset_name']])

# Auto-detect HTCondor; override with --config run_on_condor=True/False.
_roc = config.setdefault('run_on_condor',
    shutil.which("condor_submit") is not None)
config['run_on_condor'] = str(_roc).lower() not in ('false', '0', 'no')

out         = config['output_path']
SKIMMER_CFG    = "coffea4bees/skimmer/metadata/mixeddata_Run3.yml"
JCM_FILE       = "coffea4bees/analysis/weights/JCM/Run3/jetCombinatoricModel_SB_v2.yml"
HEMI_LIB       = "coffea4bees/skimmer/metadata/hemisphere_library_Run3_noTT.yml"
HEMI_STATS_DIR = "coffea4bees/skimmer/metadata/hemi_statistics_noTT"
STUDY_CFG      = "coffea4bees/analysis/metadata/study_mixed_data_Run3.yml"
REGISTRY       = "picoaod_datasets_mixeddata_Run3_noTT_pz.yml"

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input: f"{out}study_mixeddata_all{_rank_suffix}.coffea"


rule patch_skimmer_config:
    """Inject configured fields (base_path, default_rank) into the skimmer
    config. Other fields are passed through unchanged."""
    input:  SKIMMER_CFG
    output: f"{out}mixeddata_Run3.yml"
    params:
        base_path    = config['base_path'],
        default_rank = config['default_rank']
    shell:
        """
        sed -e 's|  base_path:.*|  base_path: {params.base_path}|' \
            -e 's|  default_rank:.*|  default_rank: {params.default_rank}|' \
            {input} > {output}
        echo "Patched skimmer config:"
        grep -E "base_path|default_rank" {output}
        """


use rule analysis_processor from analysis as make_mixeddata with:
    input:
        config_file = f"{out}mixeddata_Run3.yml",
        jcm         = JCM_FILE,
        hemi_lib    = HEMI_LIB,
        hemi_stats  = f"{HEMI_STATS_DIR}/hemi_statistics_{{year}}.yml",
    output: f"{out}per_year/picoaod_datasets_mixeddata_Run3_noTT_pz__{{year}}.yml"
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
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "-s",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


rule merge_mixeddata_registries:
    """Dict-merge per-year picoAOD-registry YAMLs into one.

    Each year's YAML is keyed by per-year dataset names (e.g.
    data_2022_EEE), so top-level keys never collide. Asserts no overlap to
    catch accidental reruns clobbering each other.
    """
    input:
        expand(
            f"{out}per_year/picoaod_datasets_mixeddata_Run3_noTT_pz__{{year}}.yml",
            year=config['years'],
        )
    output: f"{out}{REGISTRY}"
    container: config['analysis_container']
    log: f"{out}logs/merge_mixeddata_registries.log"
    shell:
        """
        python -c '
import sys, yaml
out = {{}}
for f in sys.argv[1:-1]:
    with open(f) as fh:
        d = yaml.full_load(fh) or {{}}
    overlap = set(out) & set(d)
    if overlap:
        raise SystemExit(f"Key collision merging {{f}}: {{overlap}}")
    out.update(d)
with open(sys.argv[-1], "w") as fh:
    yaml.dump(out, fh, default_flow_style=False)
print(f"Merged {{len(sys.argv)-2}} files -> {{sys.argv[-1]}} ({{len(out)}} datasets)")
' {input} {output} 2>&1 | tee -a {log}
        """


rule install_mixeddata_dataset:
    """Convert the merged registry into a master datasets-style YAML
    (top-level dataset name → year → picoAOD → era → files) and install it
    under coffea4bees/metadata/datasets_HH4b_Run3/. Commit the installed
    file to git to version the dataset for downstream `-d` consumers."""
    input:  f"{out}{REGISTRY}"
    output: config['install_path']
    container: config['analysis_container']
    params:
        name = config['dataset_name']
    log: f"{out}logs/install_mixeddata_dataset.log"
    shell:
        """
        mkdir -p $(dirname {output})
        python src/tools/make_dataset_yml.py \\
            -i {input} \\
            -o {output} \\
            -n {params.name} 2>&1 | tee -a {log}
        echo "Installed {output} (dataset name: {params.name}) — commit this file to git to version it." 2>&1 | tee -a {log}
        """


# ── Study mixed-data ──────────────────────────────────────────────────────────
# Wraps coffea4bees/scripts/mixeddata-study-mixeddataset-Run3.sh. Runs
# processor_study_mixed_data.py over the just-installed mixeddata_all_rank{N}
# dataset, per-year, then merges into a single coffea.

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
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "",
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
        run_performance = False
    log: f"{out}logs/merge_study_mixeddata.log"


# ── Histograms (data + TTbar + mixeddata_all_rank{N}) for JCM fitting ─────────
# Per-(dataset, year) histogram jobs, then a merge into a single coffea. The
# input config is sed-patched to force fourTag_use_tight: false regardless of
# whether the chosen mode's yaml ships with it true (quadjet_run2) or absent
# (nominal). See `mode` config knob above.

rule all_histograms:
    """Make per-dataset/year histograms and merge them. Prerequisite for JCM fit."""
    input: f"{out}histAll_{config['mode']}{_rank_suffix}.coffea"


rule create_histogram_config:
    """Patch the histogram config to (1) force fourTag_use_tight: false in
    both modes, and (2) bump worker_memory to give the dask merge enough
    headroom on cmslpc (the committed yaml ships 5GB, ~equal to the
    submit-slot cap, which OOMs during merge). Done as a patch so the
    committed yaml — shared with Snakefile_Run3.smk — stays unchanged."""
    input:  config['histogram_config']
    output: f"{out}histogram_config_{config['mode']}.yml"
    shell:
        """
        sed -e 's|  fourTag_use_tight:.*|  fourTag_use_tight: false|' \
            -e 's|  worker_memory:.*|  worker_memory: 8GB|' \
            {input} > {output}
        grep -q "fourTag_use_tight" {output} \
            || echo "  fourTag_use_tight: false" >> {output}
        echo "Patched histogram config:"
        grep -E "fourTag_use_tight|worker_memory" {output}
        """


use rule analysis_processor from analysis as make_histograms with:
    input:
        config_file = f"{out}histogram_config_{config['mode']}.yml",
        dataset_yml = config['install_path'],
    output: f"{out}histograms/hist_{{dataset}}__{{year}}.coffea"
    log:    f"{out}logs/hist_{{dataset}}__{{year}}.log"
    wildcard_constraints:
        dataset = "|".join(config['histogram_datasets']),
        year    = "|".join(config['years']),
    params:
        datasets              = "{dataset}",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


use rule merging_coffea_files from analysis as merge_histograms with:
    input:
        expand(
            f"{out}histograms/hist_{{dataset}}__{{year}}.coffea",
            dataset=config['histogram_datasets'],
            year=config['years'],
        )
    output: f"{out}histAll_{config['mode']}{_rank_suffix}.coffea"
    container: config['analysis_container']
    params:
        run_performance = False
    log: f"{out}logs/merge_histograms.log"
