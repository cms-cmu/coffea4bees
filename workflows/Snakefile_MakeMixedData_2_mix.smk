# coffea4bees/workflows/Snakefile_MakeMixedData_2_mix.smk
# M.2: mixing. make_mixed_data.py (runner -s) takes 3b data events (ttbar subtracted with the
# upstream FvT, rand > FvT.d3_to_t3), applies the upstream non-tight JCM (inputs.JCM) as pseudo-tag weights and replaces both
# hemispheres with their nearest neighbours in the M.1 library.
#
#   M2_config                         skimmer config (mixing.skimmer_template + this roast's values)
#   M2_mix (per data year, condor)    picoAODs -> <PUB>/picoAOD/<name>/, per-year registry
#   M2_merge                          -> one registry
#   M2_dataset_yml                    registry -> dataset YAML keyed `mixeddata_all`
#   M2_publish                        -> <PUB>/handoff/mixeddata_all.yml   (what MvD reads)
#
# 4b mixing (mixing.source: fourTag): 4b data events, no JCM, ttbar subtracted with FvT.d4_to_t4,
# their own hemispheres vetoed in the library match, one pass per seed s = 0..N-1 (rank drawn
# uniformly from the top mixing.k_random neighbours) -> <PUB>/picoAOD/<name>/v<s>/.
#   M2_config (per s) / M2_mix (per s, year) / M2_merge (per s) -> per-seed registries (M.4 reads them)
#   M2_dataset_yml                    all seeds -> ONE dataset (e.g. mixeddata_all_4bmix): N x the
#                                     statistics for MvD, correlated through the shared 4b events

M2_OUT = f"{out}M2/"
# M.7 re-points this config at the signal (4b mixing: seed 0's)
M2_CONFIG = f"{M2_OUT}configs/make_mixed_data_v0.yml" if MIX4B else f"{M2_OUT}make_mixed_data.yml"
M2_REGISTRY = f"{M2_OUT}picoaod_datasets_{MIX_NAME}.yml"
# 4b mixing: one cleaned registry per seed (M.4 builds the multi-sample dataset from them)
M2_SEED_REGISTRIES = [f"{M2_OUT}per_seed/picoaod_datasets_{MIX_NAME}_v{s}.yml" for s in SUBSAMPLES]
M2_DATASET = f"{M2_OUT}handoff/{MIX_NAME}.yml"
M2_PUBLISHED = f"{M2_OUT}published.done"

def _rank(r):
    """int, or [rp, rn] for independent per-side ranks (a string from --config is parsed)."""
    if isinstance(r, str):
        r = yaml.safe_load(r)
    return [int(x) for x in r] if isinstance(r, (list, tuple)) else int(r)

def _m2_config(template, jcm, dst, seed=None):
    """The mixer's runner config. seed=None: 3b mixing; seed=s: 4b mixing, seed s."""
    with open(template) as f:
        tmpl = yaml.safe_load(f) or {}
    step = int(MIX.get('chunksize', 100000))
    runner = {**(tmpl.get('runner') or {}),
              'worker_memory': MIX.get('worker_memory', '8GB'), 'chunksize': step}
    section = {**(tmpl.get('config') or {}),
               'base_path': f"{PUB}/picoAOD/{MIX_NAME}",
               'step': step,
               # A path, not `true`: `JCM_file: true` makes make_mixed_data.py read the per-year
               # JCM from the weights file instead (the old workflow's rule input was only a DAG
               # edge, so a new fit there was silently ignored).
               'apply_JCM': True,
               'JCM_file': jcm,
               'subtract_ttbar_with_weights': SUBTRACT_TTBAR,
               'hemi_library_yaml': HEMI_LIB_URL,     # read by the condor workers, via fsspec
               'hemi_stats_path': HEMI_STATS_URL,
               'default_rank': _rank(MIX.get('default_rank', config.get('default_rank', 0))),
               'use_topk_matching': bool(MIX.get('use_topk_matching', True)),
               'k_neighbors': int(MIX.get('k_neighbors', 10)),
               'collision_mode': MIX.get('collision_mode', 'retry'),
               'use_boost_corrected_matching': bool(MIX.get('use_boost_corrected_matching', True)),
               'hemi_year_key': HEMI_YEAR_KEY}
    if MIX.get('boost_acceptance_eta') is not None:
        # skip match candidates whose z boost would move a jet across the tracker acceptance
        section['boost_acceptance_eta'] = float(MIX['boost_acceptance_eta'])
    if seed is not None:
        # 4b data, real tags: no pseudo-tags. The mixer takes the FvT variable from mix_tags
        # (d4_to_t4) and vetoes each event's own hemispheres, which are in the library.
        section.pop('JCM_file')
        section.update({'apply_JCM': False,
                        'mix_tags': 'fourTag',
                        'exclude_source_event': True,
                        'rank_selection': 'random',
                        'k_random': int(MIX.get('k_random', 10)),
                        'mixing_seed': int(seed),
                        'base_path': f"{PUB}/picoAOD/{MIX_NAME}/v{seed}"})
        # ONE output file per (seed, era), so every seed templates to the same file set in M.4
        # (the M.4 subsample lesson: 100k-event chunks gave different file counts per sample)
        runner['picosize'] = int(MIX.get('picosize', 10**9))
    if SUBTRACT_TTBAR and FVT:
        section['friends'] = {'FvT': FVT}
        section['friends_include'] = ['FvT']
    cfg = processor_config(section, inherit_config=False,
                           processor="coffea4bees/skimmer/processor/make_mixed_data.py",
                           runner=runner)
    if seed is None and (not isinstance(cfg['config']['JCM_file'], str) or cfg['config']['JCM_file'] != jcm):
        raise ValueError(f"mixing config JCM_file is {cfg['config']['JCM_file']!r}, not the upstream JCM {jcm}")
    write_yaml(dst, cfg)

if not MIX4B:
    rule M2_config:
        input:
            template = MIX.get('skimmer_template', "coffea4bees/skimmer/metadata/mixeddata_Run3.yml"),
            jcm = UPSTREAM_JCM,
        output: M2_CONFIG
        run:
            _m2_config(input.template, input.jcm, output[0])

    use rule analysis_processor from analysis as M2_mix with:
        input:
            runner_script = "runner.py",
            config_file = M2_CONFIG,
            jcm = UPSTREAM_JCM,
            hemilib = M1_DONE,
        output: f"{M2_OUT}per_year/picoaod_datasets_{MIX_NAME}__{{year}}.yml"
        log: f"{M2_OUT}logs/mix__{{year}}.log"
        wildcard_constraints:
            year = "|".join(YEARS)
        params:
            datasets = "data",
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, CONDOR])),
            run_container_wrapper = WRAPPER,
            python_bin = PYTHON

    rule M2_merge:
        input: expand(f"{M2_OUT}per_year/picoaod_datasets_{MIX_NAME}__{{year}}.yml", year=YEARS)
        output: M2_REGISTRY
        log: f"{M2_OUT}logs/merge.log"
        shell:
            """
            {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
                {input} {output} 2>&1 | tee {log}
            """

    rule M2_dataset_yml:
        input: M2_REGISTRY
        output: M2_DATASET
        log: f"{M2_OUT}logs/dataset_yml.log"
        shell:
            """
            mkdir -p $(dirname {output})
            {WRAPPER} {PYTHON} src/tools/make_dataset_yml.py -i {input} -o {output} -n {MIX_NAME} 2>&1 | tee {log}
            """

else:
    rule M2_config:
        input:
            template = MIX.get('skimmer_template', "coffea4bees/skimmer/metadata/mixeddata_Run3.yml"),
        output: f"{M2_OUT}configs/make_mixed_data_v{{s}}.yml"
        wildcard_constraints:
            s = r"\d+"
        run:
            _m2_config(input.template, None, output[0], seed=int(wildcards.s))

    use rule analysis_processor from analysis as M2_mix with:
        input:
            runner_script = "runner.py",
            config_file = f"{M2_OUT}configs/make_mixed_data_v{{s}}.yml",
            hemilib = M1_DONE,
        output: f"{M2_OUT}per_year/picoaod_datasets_{MIX_NAME}_v{{s}}__{{year}}.yml"
        log: f"{M2_OUT}logs/mix_v{{s}}__{{year}}.log"
        wildcard_constraints:
            s = r"\d+",
            year = "|".join(YEARS)
        params:
            datasets = "data",
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, CONDOR])),
            run_container_wrapper = WRAPPER,
            python_bin = PYTHON

    rule M2_merge:
        input: expand(f"{M2_OUT}per_year/picoaod_datasets_{MIX_NAME}_v{{{{s}}}}__{{year}}.yml", year=YEARS)
        output: f"{M2_OUT}per_seed/picoaod_datasets_{MIX_NAME}_v{{s}}.yml"
        log: f"{M2_OUT}logs/merge_v{{s}}.log"
        wildcard_constraints:
            s = r"\d+"
        shell:
            """
            {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
                {input} {output} 2>&1 | tee {log}
            """

    rule M2_dataset_yml:
        """Every seed's files under one dataset key. Each seed alone must cover every year
        (checked here; M2_check then checks the union)."""
        input: M2_SEED_REGISTRIES
        output: M2_DATASET
        run:
            from src.tools.make_dataset_yml import parse_dataset_key
            entry = {}
            for s_, path in zip(SUBSAMPLES, input):
                with open(path) as f:
                    registry = yaml.safe_load(f) or {}
                years_s = set()
                for key, ds in registry.items():
                    year, era = parse_dataset_key(key)
                    if year is None:
                        raise ValueError(f"{path}: cannot tell the year of registry key {key!r}")
                    files = (ds or {}).get('files') or []
                    if not files:
                        continue
                    years_s.add(year)
                    pico = entry.setdefault(year, {'picoAOD': {}})['picoAOD']
                    target = pico.setdefault(era, {'files': []}) if era else pico.setdefault('files', [])
                    (target['files'] if era else target).extend(files)
                missing = [y for y in YEARS if y not in years_s]
                if missing:
                    raise ValueError(f"{path}: seed {s_} has no mixed files for {missing}")
            for year in entry.values():
                pico = year['picoAOD']
                for k, v in pico.items():
                    if k == 'files':
                        pico[k] = sorted(set(v))
                    else:
                        v['files'] = sorted(set(v['files']))
            write_yaml(output[0], {MIX_NAME: entry})

rule M2_check:
    """Every year must have mixed files: an all-bad-files skim otherwise publishes `{}`."""
    input: M2_DATASET
    output: f"{M2_OUT}dataset_checked.done"
    run:
        check_dataset_yml(input[0], MIX_NAME, YEARS)
        with open(output[0], "w") as f:
            f.write("ok\n")

rule M2_publish:
    input:
        dataset = M2_DATASET,
        checked = f"{M2_OUT}dataset_checked.done"
    output: M2_PUBLISHED
    log: f"{M2_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f -p {input.dataset} "{HANDOFF}/$(basename {input.dataset})" 2>&1 | tee {log}
        echo "published {input.dataset} -> {HANDOFF}/$(basename {input.dataset})" | tee -a {log}
        date > {output}
        """

rule all_M2:
    input: M2_PUBLISHED

localrules: M2_config, M2_merge, M2_dataset_yml, M2_check, M2_publish, all_M2
