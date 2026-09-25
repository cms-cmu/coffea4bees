# coffea4bees/workflows/Snakefile_MakeMixedData_1_hemilib.smk
# M.1: hemisphere library from 4b data, ttbar subtracted with the upstream roast's FvT
# (rand > FvT.d4_to_t4), plus the per-category statistics the mixing samples from.
#
#   M1_cluster (per data year, condor)  processor_make_hemi_library: hemisphere ROOT files are
#                                        written straight to <HEMI_BASE>/ by dump_friend_trees
#   M1_regroup (per data year)          per-dataset file lists -> {hemi_year: [files]}
#   M1_merge                            -> hemisphere_library.yml (UL16_pre/postVFP share UL16)
#   M1_study (per hemi year)            study_hemispheres.py -> hemi_statistics_<year>.yml
#   M1_publish                          registry + statistics -> <HEMI_BASE>/ on EOS
#
# The mixing jobs (M.2) run on condor workers and read the registry and statistics from EOS
# through fsspec (hemisphere_mixing/mixing_helpers.py), so nothing is installed into the checkout.
# With inputs.hemilib set, M.2 reads another roast's library and none of this runs.

M1_OUT = f"{out}M1/"
M1_CONFIG = f"{M1_OUT}make_hemi_library.yml"
M1_REGISTRY = f"{M1_OUT}hemisphere_library.yml"
M1_STATS = f"{M1_OUT}stats/"
M1_PUBLISHED = f"{M1_OUT}published.done"
M1_DONE = [] if HEMI_EXTERNAL else [M1_PUBLISHED]

rule M1_config:
    input: HEMI.get('config_template', "coffea4bees/analysis/metadata/make_hemi_library_4b.yml")
    output: M1_CONFIG
    run:
        with open(input[0]) as f:
            tmpl = yaml.safe_load(f) or {}
        cfg = processor_config(
            {**(tmpl.get('config') or {}),
             'base_path': f"{HEMI_BASE}/",
             'subtract_ttbar_with_weights': True,     # turns apply_FvT on (processor_HH4b)
             'run_SvB': False,
             'fourTag_use_tight': False,              # the library IS the 4b hemispheres
             'friends': {'FvT': FVT},                 # merged over friend_file by runner.py
             'friends_include': ['FvT']},             # data only: no trigWeight needed
            inherit_config=False,
            processor="coffea4bees/analysis/processors/processor_make_hemi_library.py",
            runner=tmpl.get('runner') or {})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M1_cluster with:
    input:
        runner_script = "runner.py",
        config_file = M1_CONFIG
    output: f"{M1_OUT}cluster/hemi__{{year}}.coffea"
    log: f"{M1_OUT}logs/cluster__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = "data",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule M1_regroup:
    """Every dataset key in one year's cluster output (data_2022_EEE, ...) belongs to that year:
    concatenate their hemisphere-file lists under the library's year key."""
    input: f"{M1_OUT}cluster/hemi__{{year}}.coffea"
    output: f"{M1_OUT}per_year/hemisphere_library__{{year}}.yml"
    log: f"{M1_OUT}logs/regroup__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        key = lambda wildcards: hemi_year(wildcards.year)
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/regroup_hemi_library.py \
            {params.key} {input} {output} 2>&1 | tee {log}
        """

rule M1_merge:
    """One registry, {hemi_year: [files]}. Unlike merge_hemi_registries.py, a repeated key is
    expected (UL16_preVFP and UL16_postVFP both feed UL16) and its lists are concatenated; a
    file appearing twice is an error (a rerun clobbering another year)."""
    input: expand(f"{M1_OUT}per_year/hemisphere_library__{{year}}.yml", year=YEARS)
    output: M1_REGISTRY
    run:
        merged = {}
        for path in input:
            with open(path) as f:
                for key, files in (yaml.full_load(f) or {}).items():
                    files = [files] if isinstance(files, str) else list(files)
                    dup = set(merged.get(key, [])) & set(files)
                    if dup:
                        raise ValueError(f"{path}: hemisphere files already listed under {key}: {sorted(dup)[:3]}")
                    merged.setdefault(key, []).extend(files)
        missing = [y for y in HEMI_YEARS if not merged.get(y)]
        if missing:
            raise ValueError(f"hemisphere library has no files for {missing}")
        write_yaml(output[0], merged)

rule M1_study:
    input: M1_REGISTRY
    output: f"{M1_STATS}hemi_statistics_{{hyear}}.yml"
    log: f"{M1_OUT}logs/study__{{hyear}}.log"
    wildcard_constraints:
        hyear = "|".join(HEMI_YEARS)
    params:
        threshold = HEMI.get('study_threshold', 1000)
    shell:
        """
        mkdir -p {M1_STATS}
        {WRAPPER} {PYTHON} coffea4bees/hemisphere_mixing/study_hemispheres.py \
            --hemifiles {input} --year {wildcards.hyear} --threshold {params.threshold} \
            --output_path {M1_STATS} 2>&1 | tee {log}
        ls -l {output} 2>&1 | tee -a {log}
        """

rule M1_publish:
    input:
        registry = M1_REGISTRY,
        stats = expand(f"{M1_STATS}hemi_statistics_{{hyear}}.yml", hyear=HEMI_YEARS)
    output: M1_PUBLISHED
    log: f"{M1_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        for f in {input.registry} {input.stats}; do
            xrdcp -f -p "$f" "{HEMI_BASE}/$(basename $f)" 2>&1 | tee -a {log}
            echo "published $f -> {HEMI_BASE}/$(basename $f)" | tee -a {log}
        done
        date > {output}
        """

rule all_M1:
    input: M1_DONE

localrules: M1_config, M1_regroup, M1_merge, M1_study, M1_publish, all_M1
