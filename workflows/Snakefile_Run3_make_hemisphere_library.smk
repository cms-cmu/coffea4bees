"""Hemisphere-library production for Run3 (per-year, parallel).

Formalizes the two bash scripts that build the hemisphere library consumed by
the mixed-data workflow:
  - coffea4bees/scripts/mixeddata-cluster-Run3-all.sh
      (processor_make_hemi_library.py over Run3 data -> hemisphere ROOT files
       on EOS + per-year registry)
  - coffea4bees/scripts/mixeddata-study-hemispheres.sh
      (study_hemispheres.py over that library -> hemi_statistics_{year}.yml)

DAG:
    patch_hemi_config
        v
    cluster_hemi (per year, --datasets data, HTCondor)   -> per-year .coffea
        v                                                   (carries per-dataset
    build_year_library (regroup per-dataset -> per-year)    hemisphere file lists)
        |---------> merge_hemi_library -> hemisphere_library_Run3_noTT_{tag}.yml
        v
    study_hemispheres (per year, --threshold N) -> hemi_statistics_{year}.yml

The two installed artifacts (the registry YAML and the hemi_statistics dir) are
exactly the inputs the mixed-data workflow declares as committed prerequisites
(Snakefile_Run3_make_mixeddata.smk). Run this workflow deliberately to rebuild
the library; it never triggers, and is never triggered by, the mixed-data graph.

No processor change is needed to pick up a new jet-pT threshold:
processor_make_hemi_library inherits HH4bBaseProcessor, which defaults
object_selection_cfg to object_selection_thresholds.yml and threads it through
apply_4b_selection. Era-specific thresholds (e.g. 25 GeV in 2023, 30 GeV in
2022) are resolved from event.metadata['year'] inside apply_4b_selection.

Output namespacing is controlled by `tag` (default 'pt25') so a rebuilt library
never overwrites the production `hemiLib_noTT` library. Override per run with
--config tag=<name>.

Usage:
    snakemake --profile software/snakemake/profiles/lpc \\
        --snakefile coffea4bees/workflows/Snakefile_Run3_make_hemisphere_library.smk \\
        --cores 4
"""

import shutil

config.setdefault('analysis_container',
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',
    "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('years',
    ['2022_EE', '2022_preEE', '2023_BPix', '2023_preBPix'])

# Output namespace. Keeps a rebuilt library from clobbering production
# (hemiLib_noTT / hemisphere_library_Run3_noTT.yml). Override with --config tag=...
config.setdefault('tag', 'pt25')
_tag = config['tag']

# Minimum per-category count for study_hemispheres.py (hemi_statistics binning).
config.setdefault('study_threshold', 1000)

# EOS location for the hemisphere ROOT files (patched into the hemi-library
# config below). Rank-independent; tag-namespaced.
config.setdefault('hemi_base_path',
    f"root://cmseos.fnal.gov//store/user/jda102/XX4b/hemiLib_noTT_{_tag}")

# Installed (committed-location) artifacts — the two inputs make_mixeddata wants.
config.setdefault('install_library',
    f"coffea4bees/skimmer/metadata/hemisphere_library_Run3_noTT_{_tag}.yml")
config.setdefault('install_stats_dir',
    f"coffea4bees/skimmer/metadata/hemi_statistics_noTT_{_tag}")

config.setdefault('output_path', f"output/Run3_make_hemilib_{_tag}/")

# Auto-detect HTCondor; override with --config run_on_condor=True/False.
_roc = config.setdefault('run_on_condor',
    shutil.which("condor_submit") is not None)
config['run_on_condor'] = str(_roc).lower() not in ('false', '0', 'no')

out          = config['output_path']
HEMI_CFG     = "coffea4bees/analysis/metadata/make_hemi_library_4b.yml"
INSTALL_LIB  = config['install_library']
STATS_DIR    = config['install_stats_dir']

module analysis:
    snakefile: "rules/analysis.smk"
    config: config


rule all:
    input:
        INSTALL_LIB,
        expand(f"{STATS_DIR}/hemi_statistics_{{year}}.yml", year=config['years']),


rule patch_hemi_config:
    """Patch the hemisphere-library config to point base_path at the
    tag-namespaced EOS location, and force the proven cluster settings
    (subtract_ttbar_with_weights: True, run_SvB: False). The committed yaml is
    left untouched."""
    input:  HEMI_CFG
    output: f"{out}make_hemi_library_4b.yml"
    params:
        base_path = config['hemi_base_path'],
    shell:
        """
        sed -e 's|^  base_path:.*|  base_path: {params.base_path}|' \
            -e 's|^  subtract_ttbar.*|  subtract_ttbar_with_weights: true|' \
            -e 's|^  run_SvB.*|  run_SvB: false|' \
            {input} > {output}
        echo "Patched hemi-library config:"
        grep -E "base_path|subtract_ttbar|run_SvB" {output}
        """


# ── Cluster: build hemisphere ROOT files + per-year .coffea ────────────────────
# Mirrors mixeddata-cluster-Run3-all.sh but one year per HTCondor submission for
# parallelism. Runs in analysis mode (no -s): processor_make_hemi_library writes
# the hemisphere ROOT files to EOS via dump_friend_trees and returns per-dataset
# {files, source} metadata, which lands in the saved .coffea.
use rule analysis_processor from analysis as cluster_hemi with:
    input:
        config_file = f"{out}make_hemi_library_4b.yml",
    output: f"{out}cluster/hemi__{{year}}.coffea"
    log:    f"{out}logs/cluster_hemi__{{year}}.log"
    wildcard_constraints:
        year = "|".join(config['years'])
    params:
        datasets              = "data",
        years                 = "{year}",
        config                = lambda wildcards, input: input.config_file,
        processor             = "coffea4bees/analysis/processors/processor_make_hemi_library.py",
        datasets_file         = config['dataset_location'],
        blind                 = False,
        run_performance       = False,
        # Must be non-empty: run_container drops empty-string args, so an empty
        # --friends would swallow the following --year token and break parsing.
        # This is the script's own default, matching the original cluster bash
        # (which passed no --friends and thus used this default).
        friends               = "coffea4bees/metadata/friends_HH4b.yml",
        run_on_condor         = config['run_on_condor'],
        extra_arguments       = "",
        run_container_wrapper = "./run_container",
        dashboard_address     = 0


rule build_year_library:
    """Regroup the per-dataset hemisphere file lists from the cluster .coffea
    into a single {year: [files]} entry. Because each cluster job processes one
    year, every dataset key in the .coffea (data_2022_EEE, data_2022_EEF, ...)
    belongs to that year, so we just concatenate all their file lists under the
    year key — no era->year string parsing needed."""
    input:  f"{out}cluster/hemi__{{year}}.coffea"
    output: f"{out}per_year/hemisphere_library__{{year}}.yml"
    wildcard_constraints:
        year = "|".join(config['years'])
    log: f"{out}logs/build_year_library__{{year}}.log"
    shell:
        """
        ./run_container python coffea4bees/workflows/scripts/regroup_hemi_library.py \
            {wildcards.year} {input} {output} 2>&1 | tee -a {log}
        """


rule merge_hemi_library:
    """Merge the per-year library entries into the installed registry YAML
    (year -> [files]). Keys are distinct years, so no collision is possible;
    assert anyway to catch accidental reruns."""
    input:
        expand(f"{out}per_year/hemisphere_library__{{year}}.yml", year=config['years'])
    output: INSTALL_LIB
    log: f"{out}logs/merge_hemi_library.log"
    shell:
        """
        mkdir -p $(dirname {output})
        ./run_container python coffea4bees/workflows/scripts/merge_hemi_registries.py \
            {input} {output} 2>&1 | tee -a {log}
        echo "Installed {output} — commit to git to version the library." 2>&1 | tee -a {log}
        """


rule study_hemispheres:
    """Run study_hemispheres.py over one year's hemisphere library to produce
    hemi_statistics_{year}.yml (the per-category count/range statistics consumed
    by the mixed-data sampling). Reads the per-year library directly so years
    run in parallel."""
    input:  f"{out}per_year/hemisphere_library__{{year}}.yml"
    output: f"{STATS_DIR}/hemi_statistics_{{year}}.yml"
    wildcard_constraints:
        year = "|".join(config['years'])
    params:
        threshold = config['study_threshold'],
        stats_dir = STATS_DIR,
    log: f"{out}logs/study_hemispheres__{{year}}.log"
    shell:
        """
        mkdir -p {params.stats_dir}
        ./run_container python coffea4bees/hemisphere_mixing/study_hemispheres.py \
            --hemifiles {input} \
            --year {wildcards.year} \
            --threshold {params.threshold} \
            --output_path {params.stats_dir} 2>&1 | tee -a {log}
        ls -l {output} 2>&1 | tee -a {log}
        """
