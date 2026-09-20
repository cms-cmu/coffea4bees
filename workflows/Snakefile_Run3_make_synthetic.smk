"""Synthetic-dataset (jet-declustering) production for Run3, end-to-end.

Builds the synthetic 4b background model from real Run3 data by learning the
jet-splitting kinematics and re-generating (declustering) jets from them. Wraps
the three historical scripts as one Snakemake DAG:

  scripts/synthetic-dataset-cluster-Run3-all.sh        -> cluster + make PDFs
  scripts/synthetic-dataset-make-dataset-Run3-all.sh   -> decluster (make picoAODs)
  scripts/synthetic-dataset-analyze-Run3-all.sh         -> analyze (histograms)

Stages (all HTCondor-parallel per year; PDFs join cluster -> decluster):

  1.  cluster       processor_cluster_4b.py over `data`, per year (cluster_4b_Run3.yml,
                    subtract_ttbar_with_weights: true) -> per-year splitting-hist coffea
  1m. merge         -> {SHARED}synthetic_datasets_Run3_nott.coffea
  1b. make_pdfs     make_jet_splitting_PDFs.py over the merged coffea, --years Run3
                    -> {pdf_dir}clustering_pdfs_vs_pT_{year}.yml  (one per era)
  2.  make_dataset  make_declustered_data_4b.py (runner -s) per (seed, year), reading
                    the PDFs above -> per-(seed,year) picoAOD registry on EOS
  2m. merge         per-seed registries (years never collide) -> registry_seed{N}.yml
  3.  install       registries -> files_template dataset YAML (installed under
                    metadata/datasets_HH4b_Run3/), nSamples = n_seeds
  4.  analyze       processor_HH4b.py over the installed synthetic dataset, per year
  4m. merge         -> {out}histDeClusteredDataRun3_noTT.coffea

Cluster + PDFs are seed-independent, so they live in a shared (tag-keyed but
rank/seed-independent) dir that every n_seeds run reuses.

2023 pt threshold: all three processors read
coffea4bees/analysis/metadata/object_selection_thresholds.yml, whose
`era_overrides: "2023"` sets the selected-jet pt_min to 25 GeV for the 2023
eras. It is declared as an `input:` on every processing stage so the DAG fails
fast if it is ever missing and the pt25 selection is guaranteed to be picked up.

Seeds: the analyze step reads all seeds through runner.py's multi-sample path,
which expands the `files_template` placeholder XXX over range(nSamples). So the
seeds MUST be the contiguous set 0..n_seeds-1; they are derived here as
range(n_seeds) and n_seeds becomes nSamples in the installed dataset.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \\
        --snakefile coffea4bees/workflows/Snakefile_Run3_make_synthetic.smk \\
        --cores 4
    # multiple seeds:
    #   ... --config n_seeds=5
"""

import shutil

config.setdefault('analysis_container',
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',
    "coffea4bees/metadata/datasets/")
config.setdefault('years',
    ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

# Number of declustering seeds. Each seed is an independent synthetic replica
# (declustering_rand_seed = 0, 1, ...); the analyze step reads all of them via
# nSamples. n_seeds MUST stay a contiguous count (seeds = 0..n_seeds-1) for the
# runner.py XXX->range(nSamples) expansion to line up. Override: --config n_seeds=5.
config.setdefault('n_seeds', 1)
n_seeds = int(config['n_seeds'])
SEEDS = list(range(n_seeds))

# Optional campaign tag folded into the namespace suffix, mirroring the
# mixeddata Snakefile. A tag implies a different object selection / PDF build,
# so tagged runs land on their own EOS/dataset/output paths. Empty by default.
config.setdefault('tag', '')
_tag = str(config['tag'])
_tag_suffix = f"_{_tag}" if _tag else ''

# Local-output / dataset-name suffix. n_seeds is encoded so variants with a
# different sample count self-document and never clobber each other (or the
# committed synthetic_data.yml, which is dataset `synthetic_data_noTT`).
_suffix = f"{_tag_suffix}_ns{n_seeds}"

config.setdefault('output_path', f"output/Run3_synthetic{_suffix}/")
# EOS picoAOD location. The seed is in the filename (picoAOD_seed{N}.root), so
# seeds share this directory; only the tag namespaces it.
config.setdefault('base_path',
    f"root://cmseos.fnal.gov//store/user/jda102/XX4b/DeClustered_noTT{_tag_suffix}")
config.setdefault('dataset_name', f"synthetic_data_noTT{_suffix}")
config.setdefault('install_path',
    f"coffea4bees/metadata/datasets/synthetic_data_noTT{_suffix}.yml")

# Auto-detect HTCondor; override with --config run_on_condor=True/False.
_roc = config.setdefault('run_on_condor',
    shutil.which("condor_submit") is not None)
config['run_on_condor'] = str(_roc).lower() not in ('false', '0', 'no')

out    = config['output_path']
# Cluster + PDFs are seed-independent: shared dir keyed by tag only, reused by
# every n_seeds run.
SHARED = f"output/Run3_synthetic_shared{_tag_suffix}/"
# The generated clustering PDFs must live INSIDE coffea4bees/ (not output/),
# because the declustering runs on HTCondor and only `src` + `coffea4bees` are
# shipped to the workers (condor_transfer_input_files in declustering_Run3.yml).
# A PDF path under output/ is invisible on the worker -> FileNotFoundError. The
# `-synthetic{tag}` suffix keeps it namespaced so the committed
# jet-splitting-PDFs-* dirs are never clobbered. (Generated files show as
# untracked in git; that's expected.)
PDF_DIR = f"coffea4bees/jet_clustering/jet-splitting-PDFs-synthetic{_tag_suffix}/"

CLUSTER_CFG   = "coffea4bees/analysis/metadata/cluster_4b_Run3.yml"
DECLUSTER_CFG = "coffea4bees/skimmer/metadata/declustering_Run3.yml"
ANALYZE_CFG   = config.get("analyze_cfg", "coffea4bees/analysis/metadata/HH4b_mixed_data_new.yml")
# Object-selection thresholds (carries the 2023 pt25 era_override). Declared as
# an input on every processing stage so the DAG fails fast if it is missing.
OBJ_SEL       = "coffea4bees/analysis/metadata/object_selection_thresholds.yml"
FRIENDS       = config.get("friends_yml", "coffea4bees/metadata/friends/friends_HH4b.yml")

# make_jet_splitting_PDFs.py --years Run3 writes clustering_pdfs_vs_pT_{era}.yml
# for exactly these eras (same set as config['years']).
PDF_FILES = [f"{PDF_DIR}clustering_pdfs_vs_pT_{y}.yml" for y in config['years']]

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input:
        f"{out}histDeClusteredDataRun3_noTT.coffea",
        config['install_path'],


# ── Stage 1: cluster (learn splittings) ───────────────────────────────────────
# processor_cluster_4b.py over `data`, per year, with TT subtraction. Output is
# the splitting-histogram coffea used to build the PDFs. Seed-independent ->
# shared dir.

use rule analysis_processor from analysis as make_cluster with:
    input:
        config_file = CLUSTER_CFG,
        obj_sel     = OBJ_SEL,
    output: f"{SHARED}cluster/synthetic_datasets_Run3_nott__{{year}}.coffea"
    log:    f"{SHARED}logs/cluster__{{year}}.log"
    wildcard_constraints:
        year = "|".join(config['years'])
    params:
        datasets              = "data",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/analysis/processors/processor_cluster_4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = FRIENDS,
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


use rule merging_coffea_files from analysis as merge_cluster with:
    input:
        expand(
            f"{SHARED}cluster/synthetic_datasets_Run3_nott__{{year}}.coffea",
            year=config['years'],
        )
    output: f"{SHARED}synthetic_datasets_Run3_nott.coffea"
    # No container: directive here. Under `--profile lpc`
    # (software-deployment-method: apptainer) a container: directive makes
    # snakemake enter apptainer for the rule; then run_container_wrapper's
    # ./run_container tries to enter apptainer AGAIN and aborts ("Apptainer is
    # not installed") because apptainer isn't available inside the container.
    # Dropping container: (like the analysis_processor jobs, which have none)
    # lets ./run_container provide the single container with the full env.
    container: None
    params:
        run_performance = False,
        run_container_wrapper = "./run_container"
    log: f"{SHARED}logs/merge_cluster.log"


# ── Stage 1b: make clustering PDFs ────────────────────────────────────────────
# Distill the merged splitting histograms into per-era PDF templates. Written to
# a workflow-local dir so the committed jet-splitting-PDFs-*/ stays untouched;
# the declustering config is patched (below) to point the declusterer here.

rule make_pdfs:
    input:
        hist   = f"{SHARED}synthetic_datasets_Run3_nott.coffea",
        obj_sel = OBJ_SEL,
    output: PDF_FILES
    params:
        pdf_dir = PDF_DIR,
    log: f"{SHARED}logs/make_pdfs.log"
    shell:
        """
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR {params.pdf_dir}
        echo "Making jet-splitting PDFs -> {params.pdf_dir}" 2>&1 | tee -a {log}
        ./run_container env PYTHONPATH=. python \
            coffea4bees/jet_clustering/make_jet_splitting_PDFs.py \
            {input.hist} --out {params.pdf_dir} --years Run3 2>&1 | tee -a {log}
        ls {params.pdf_dir} 2>&1 | tee -a {log}
        """


# ── Stage 2: make dataset (decluster) ─────────────────────────────────────────
# Patch the declustering config per seed (rand_seed + workflow-local PDF path +
# EOS base_path), then run make_declustered_data_4b.py per (seed, year).

rule patch_decluster_config:
    input:
        cfg  = DECLUSTER_CFG,
        pdfs = PDF_FILES,
    output: f"{out}configs/declustering_seed{{seed}}.yml"
    wildcard_constraints:
        seed = "|".join(str(s) for s in SEEDS)
    params:
        base_path = config['base_path'],
        pdf_file  = f"{PDF_DIR}clustering_pdfs_vs_pT_XXX.yml",
    shell:
        """
        sed -e 's|  declustering_rand_seed:.*|  declustering_rand_seed: {wildcards.seed}|' \
            -e 's|  clustering_pdfs_file:.*|  clustering_pdfs_file: "{params.pdf_file}"|' \
            -e 's|  base_path:.*|  base_path: {params.base_path}|' \
            {input.cfg} > {output}
        echo "Patched declustering config (seed {wildcards.seed}):"
        grep -E "declustering_rand_seed|clustering_pdfs_file|base_path" {output}
        """


use rule analysis_processor from analysis as make_synthetic with:
    input:
        config_file = f"{out}configs/declustering_seed{{seed}}.yml",
        obj_sel     = OBJ_SEL,
    output: f"{out}per_seed/seed{{seed}}/reg__{{year}}.yml"
    log:    f"{out}logs/make_synthetic__seed{{seed}}__{{year}}.log"
    wildcard_constraints:
        seed = "|".join(str(s) for s in SEEDS),
        year = "|".join(config['years'])
    params:
        datasets              = "data",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/skimmer/processor/make_declustered_data_4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = FRIENDS,
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "-s",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


rule merge_synthetic_registries:
    """Dict-merge a seed's per-year registries into one. Per-year keys (e.g.
    data_2022_EEE) never collide across years."""
    input:
        expand(
            f"{out}per_seed/seed{{{{seed}}}}/reg__{{year}}.yml",
            year=config['years'],
        )
    output: f"{out}registry_seed{{seed}}.yml"
    wildcard_constraints:
        seed = "|".join(str(s) for s in SEEDS)
    log: f"{out}logs/merge_registries__seed{{seed}}.log"
    shell:
        """
        python coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input} {output} 2>&1 | tee -a {log}
        """


# ── Stage 3: install dataset ──────────────────────────────────────────────────
# Convert the per-seed registries into a files_template dataset YAML with
# nSamples = n_seeds, matching metadata/datasets_HH4b_Run3/synthetic_data.yml.

rule install_synthetic_dataset:
    input:
        expand(f"{out}registry_seed{{seed}}.yml", seed=SEEDS)
    output: config['install_path']
    params:
        name = config['dataset_name'],
    log: f"{out}logs/install_synthetic_dataset.log"
    shell:
        """
        mkdir -p $(dirname {output})
        python coffea4bees/workflows/scripts/install_synthetic_dataset.py \
            -n {params.name} \
            -o {output} \
            {input} 2>&1 | tee -a {log}
        echo "Installed {output} (dataset {params.name}) — commit to git to version it." 2>&1 | tee -a {log}
        """


# ── Stage 4: analyze ──────────────────────────────────────────────────────────
# processor_HH4b.py over the installed synthetic dataset (multi-sample path
# expands nSamples seeds), per year, then merge.

use rule analysis_processor from analysis as make_analyze with:
    input:
        config_file = ANALYZE_CFG,
        dataset_yml = config['install_path'],
        obj_sel     = OBJ_SEL,
    output: f"{out}analyze/hist__{{year}}.coffea"
    log:    f"{out}logs/analyze__{{year}}.log"
    wildcard_constraints:
        year = "|".join(config['years'])
    params:
        datasets              = config['dataset_name'],
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/analysis/processors/processor_HH4b.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        friends               = FRIENDS,
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


use rule merging_coffea_files from analysis as merge_analyze with:
    input:
        expand(
            f"{out}analyze/hist__{{year}}.coffea",
            year=config['years'],
        )
    output: f"{out}histDeClusteredDataRun3_noTT.coffea"
    # See merge_cluster: drop container: so ./run_container isn't nested inside
    # the profile's apptainer.
    container: None
    params:
        run_performance = False,
        run_container_wrapper = "./run_container"
    log: f"{out}logs/merge_analyze.log"
