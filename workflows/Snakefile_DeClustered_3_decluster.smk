# coffea4bees/workflows/Snakefile_DeClustered_3_decluster.smk
# D.3: decluster. make_declustered_data_4b.py (runner -s) re-clusters every 4b data event into
# its splitting tree and re-generates each splitting from the D.2 PDFs, seeded by
# declustering_rand_seed: every seed is an independent synthetic replica of the 4b data.
# declustering.method: library -- each splitting is instead replaced by a real one from D.1's
# splitting library, a random one of each jet's nearest neighbours drawn from the seed
# (library.selection random; rank: the seed is the neighbour rank) (picoAOD_lib_seed<s>).
# (scripts/synthetic-dataset-make-dataset-Run3-all.sh, config skimmer/metadata/declustering_Run3.yml)

#
#   D3_config (per seed)              skimmer config (declustering.skimmer_template + this roast's values)
#   D3_decluster (per seed x year, condor)
#                                     picoAODs -> <PUB>/picoAOD/<name>/<dataset>/picoAOD_seed<s>*.root,
#                                     per-(seed, year) registry
#   D3_check (per seed x year)        refuse a registry with lost chunks/eras (runner.py exits 0);
#                                     moved to *.incomplete so a rerun redoes that job
#   D3_merge (per seed)               -> one clean registry per seed (numpy tags dropped)
#   D3_dataset_yml                    -> one multi-sample dataset: <multijet name>, nSamples: n_seeds,
#                                        per-year files_template with seedXXX (runner expands XXX)
#   D3_combined_yml (subtract_ttbar)  -> <name>: the same + the ttbar pseudodata files, per year
#                                        (as M.4 builds mixeddata_4b: pseudo-data = multijet + ttbar)
#   D3_all_yml (all_dataset_name)     -> <all name> (mixeddata_all_*): every seed's multijet files as
#                                        ONE sample, per year and era -- the MvD background model, as
#                                        the 4b mixing's mixeddata_all_4bmix
#   D3_publish                        -> <PUB>/handoff/<multijet name>.yml, <name>.yml (+ <all name>.yml)
#                                        (consumers read <name>; D.4/D.5 the multijet one; MvD <all name>)

D3_OUT = f"{out}D3/"
D3_MJ_DATASET = f"{D3_OUT}handoff/{MJ_NAME}.yml"
D3_DATASET = f"{D3_OUT}handoff/{DATASET_NAME}.yml"
D3_ALL_DATASET = f"{D3_OUT}handoff/{ALL_NAME}.yml" if ALL_NAME else None
D3_HANDOFFS = list(dict.fromkeys(filter(None, [D3_MJ_DATASET, D3_DATASET, D3_ALL_DATASET])))
D3_PUBLISHED = f"{D3_OUT}published.done"
# what the declustering reads: the D.2 PDFs, or D.1's splitting library
D3_SOURCE_DONE = LIB_DONE if LIBRARY else D2_DONE

rule D3_config:
    input:
        template = DECL.get('skimmer_template', "coffea4bees/skimmer/metadata/declustering_Run3.yml"),
        source = D3_SOURCE_DONE
    output: f"{D3_OUT}configs/declustering_seed{{seed}}.yml"
    wildcard_constraints:
        seed = r"\d+"
    run:
        with open(input.template) as f:
            tmpl = yaml.safe_load(f) or {}
        runner = {**(tmpl.get('runner') or {})}
        for k in ('worker_memory', 'chunksize'):
            if k in DECL:
                runner[k] = DECL[k]
        # the shared Dask daemon's workers get the code tarball of the job that STARTS it. Without
        # D.1 (inputs.splitting_library) that is D.3, and the Run 2 template's list lacks
        # coffea4bees/skimmer: every chunk failed "No module named coffea4bees.skimmer"
        # (declib16_run2). Ship all of the code, as D.1 does.
        runner['condor_transfer_input_files'] = ['src', 'coffea4bees']
        section = {**(tmpl.get('config') or {}),
                   'base_path': f"{PUB}/picoAOD/{MJ_NAME}",
                   'clustering_pdfs_file': PDF_TEMPLATE,   # read by the condor workers, via fsspec
                   'declustering_rand_seed': int(wildcards.seed),
                   'subtract_ttbar_with_weights': SUBTRACT_TT}
        if LIBRARY:
            section.update({'clustering_pdfs_file': "None",
                            'declustering_method': 'library',
                            'clustering_library_file': LIB_REGISTRY_URL,   # {year: [files]}, via fsspec
                            **{f"library_{k}": LIB_OPTS[k]
                               for k in ('carry_fields', 'min_entries', 'scale_pt', 'boost_z', 'selection', 'k_neighbors', 'max_distance', 'mass_match_weight', 'cache_size')
                               if k in LIB_OPTS}})
        for k in ('b_pt_threshold', 'dr_threshold', 'max_jet_retry', 'max_event_retry'):
            if k in DECL:
                section[k] = DECL[k]
        if SUBTRACT_TT:
            section.update({'friends': {'FvT': FVT}, 'friends_include': ['FvT']})
        else:
            section['friends_include'] = []            # reads no friend at all
        cfg = processor_config(section, inherit_config=False,
                               processor="coffea4bees/skimmer/processor/make_declustered_data_4b.py",
                               runner=runner)
        if cfg['runner'].get('class_name') != 'DeClusterer':
            raise ValueError(f"{input.template}: runner.class_name is {cfg['runner'].get('class_name')!r}, not DeClusterer")
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as D3_decluster with:
    input:
        runner_script = "runner.py",
        config_file = f"{D3_OUT}configs/declustering_seed{{seed}}.yml",
        source = D3_SOURCE_DONE
    output: f"{D3_OUT}per_seed/seed{{seed}}/picoaod_datasets__{{year}}.yml"
    log: f"{D3_OUT}logs/decluster__seed{{seed}}__{{year}}.log"
    wildcard_constraints:
        seed = r"\d+",
        year = "|".join(YEARS)
    params:
        datasets = "data",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule D3_check:
    """runner.py exits 0 when the skim loses chunks (it then skips the merge and writes a registry
    with no usable files). Refuse that here, per (seed, year): the incomplete registry is moved to
    *.incomplete, so the next run (roast resume) re-runs just that job."""
    input: f"{D3_OUT}per_seed/seed{{seed}}/picoaod_datasets__{{year}}.yml"
    output: f"{D3_OUT}per_seed/seed{{seed}}/complete__{{year}}.ok"
    log: f"{D3_OUT}logs/check__seed{{seed}}__{{year}}.log"
    wildcard_constraints:
        seed = r"\d+",
        year = "|".join(YEARS)
    params:
        expect = lambda wildcards: f"{wildcards.year}:{','.join(YEAR_ERAS[wildcards.year])}",
        test = "--test" if config['test'] else ""
    shell:
        """
        set -o pipefail
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/check_skim_registry.py \
            {input} {output} --expect {params.expect} {params.test} 2>&1 | tee {log}
        """

rule D3_merge:
    """Per-year registry keys (data_2022_EEE, ...) never collide across years; the script
    refuses an overlap (a rerun clobbering another year) and drops runner's numpy tags."""
    input:
        registries = expand(f"{D3_OUT}per_seed/seed{{{{seed}}}}/picoaod_datasets__{{year}}.yml", year=YEARS),
        checked = expand(f"{D3_OUT}per_seed/seed{{{{seed}}}}/complete__{{year}}.ok", year=YEARS)
    output: f"{D3_OUT}per_seed/registry_seed{{seed}}.yml"
    log: f"{D3_OUT}logs/merge__seed{{seed}}.log"
    wildcard_constraints:
        seed = r"\d+"
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input.registries} {output} 2>&1 | tee {log}
        """

rule D3_dataset_yml:
    """One dataset key, per-year `files_template` lists with the seed replaced by XXX. Every seed
    must produce the same set of templates -- checked, not assumed: the seedXXX expansion would
    otherwise read files that do not exist, or silently skip ones that do."""
    input: expand(f"{D3_OUT}per_seed/registry_seed{{seed}}.yml", seed=SEEDS)
    output: D3_MJ_DATASET
    run:
        import re
        from src.tools.make_dataset_yml import parse_dataset_key
        per_seed = []
        for s, path in zip(SEEDS, input):
            with open(path) as f:
                registry = yaml.safe_load(f) or {}
            templates, eras = {}, set()
            for key, entry in registry.items():
                year, era = parse_dataset_key(key)
                if year is None:
                    raise ValueError(f"{path}: cannot tell the year of registry key {key!r}")
                if (entry or {}).get('files'):
                    eras.add((year, era))
                for fp in (entry or {}).get('files') or []:
                    # keep .chunkN: the files on disk are picoAOD[_lib]_seed<s>.chunk<k>.root
                    t = re.sub(rf'/{PICO_PREFIX}{s}((\.chunk\d+)?\.root)$', rf'/{PICO_PREFIX}XXX\1', fp)
                    if 'XXX' not in t:
                        raise ValueError(f"{path}: {fp} does not carry seed {s}")
                    templates.setdefault(year, set()).add(t)
            # the skimmer runs with skipbadfiles: a broken era still exits 0, with no files
            missing_eras = [f"{y}{e}" for y, es in YEAR_ERAS.items() for e in es if (y, e) not in eras]
            if missing_eras and not config['test']:
                raise ValueError(f"{path}: no declustered files for eras {missing_eras}")
            per_seed.append(templates)
        for s, templates in zip(SEEDS[1:], per_seed[1:]):
            if templates != per_seed[0]:
                only_s = sorted(set().union(*templates.values()) - set().union(*per_seed[0].values()))[:3]
                only_0 = sorted(set().union(*per_seed[0].values()) - set().union(*templates.values()))[:3]
                raise ValueError(f"seed {s} files differ from seed 0 once templated (e.g. only seed {s}: "
                                 f"{only_s}, only seed 0: {only_0})")
        missing = [y for y in YEARS if y not in per_seed[0]]
        if missing:
            raise ValueError(f"no declustered files for {missing}")
        dataset = {'nSamples': N_SEEDS, 'xs': {'Run2': 1, 'Run3': 1}}
        for year in YEARS:
            dataset[year] = {'picoAOD': {'files_template': sorted(per_seed[0][year])}}
        write_yaml(output[0], {MJ_NAME: dataset})

if SUBTRACT_TT:
    rule D3_combined_yml:
        """The consumer dataset: every seed's multijet files plus the (one, shared) ttbar pseudodata,
        per year -- the pseudodata files carry no XXX, so every seed reads the same ones."""
        input:
            multijet = D3_MJ_DATASET,
            psdata = PS_DATASET
        output: D3_DATASET
        run:
            with open(input.multijet) as f:
                mj = yaml.safe_load(f)[MJ_NAME]
            with open(input.psdata) as f:
                ps = (yaml.safe_load(f) or {}).get(PS_NAME) or {}
            dataset = {k: v for k, v in mj.items() if k not in YEARS}
            for year in YEARS:
                ps_files = ((ps.get(year) or {}).get('picoAOD') or {}).get('files') or []
                if not ps_files:
                    raise ValueError(f"{input.psdata}: no {PS_NAME} files for {year}")
                if any('XXX' in p for p in ps_files):
                    raise ValueError(f"{input.psdata}: pseudodata file names contain XXX")
                dataset[year] = {'picoAOD': {'files_template':
                                             list(mj[year]['picoAOD']['files_template']) + sorted(ps_files)}}
            write_yaml(output[0], {DATASET_NAME: dataset})

    localrules: D3_combined_yml

if ALL_NAME:
    rule D3_all_yml:
        """The MvD background model: every seed's multijet files under one dataset key, per year and
        era (the layout runner.py's single-sample data path reads), as M2_dataset_yml does for the 4b
        mixing. Each seed alone must cover every year and era."""
        input: expand(f"{D3_OUT}per_seed/registry_seed{{seed}}.yml", seed=SEEDS)
        output: D3_ALL_DATASET
        run:
            from src.tools.make_dataset_yml import parse_dataset_key
            entry = {}
            for s, path in zip(SEEDS, input):
                with open(path) as f:
                    registry = yaml.safe_load(f) or {}
                eras_s = set()
                for key, ds in registry.items():
                    year, era = parse_dataset_key(key)
                    if year is None:
                        raise ValueError(f"{path}: cannot tell the year of registry key {key!r}")
                    files = (ds or {}).get('files') or []
                    if not files:
                        continue
                    eras_s.add((year, era))
                    pico = entry.setdefault(year, {'picoAOD': {}})['picoAOD']
                    target = pico.setdefault(era, {'files': []})['files'] if era else pico.setdefault('files', [])
                    target.extend(files)
                missing = [f"{y}{e}" for y, es in YEAR_ERAS.items() for e in es if (y, e) not in eras_s]
                if missing and not config['test']:
                    raise ValueError(f"{path}: seed {s} has no declustered files for {missing}")
            if missing_years := [y for y in YEARS if y not in entry]:
                raise ValueError(f"no declustered files for {missing_years}")
            for year in entry.values():
                for k, v in year['picoAOD'].items():
                    if k == 'files':
                        year['picoAOD'][k] = sorted(set(v))
                    else:
                        v['files'] = sorted(set(v['files']))
            write_yaml(output[0], {ALL_NAME: {y: entry[y] for y in YEARS}})

    localrules: D3_all_yml

rule D3_publish:
    input: D3_HANDOFFS
    output: D3_PUBLISHED
    log: f"{D3_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        for f in {input}; do
            xrdcp -f -p "$f" "{HANDOFF}/$(basename $f)" 2>&1 | tee -a {log}
            echo "published $f -> {HANDOFF}/$(basename $f)" | tee -a {log}
        done
        date > {output}
        """

rule all_D3:
    input: D3_PUBLISHED

localrules: D3_config, D3_check, D3_merge, D3_dataset_yml, D3_publish, all_D3
