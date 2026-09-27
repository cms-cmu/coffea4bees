# coffea4bees/workflows/Snakefile_DeClustered_3_decluster.smk
# D.3: decluster. make_declustered_data_4b.py (runner -s) re-clusters every 4b data event into
# its splitting tree and re-generates each splitting from the D.2 PDFs, seeded by
# declustering_rand_seed: every seed is an independent synthetic replica of the 4b data.
# (scripts/synthetic-dataset-make-dataset-Run3-all.sh, config skimmer/metadata/declustering_Run3.yml)
#
#   D3_config (per seed)              skimmer config (declustering.skimmer_template + this roast's values)
#   D3_decluster (per seed x year, condor)
#                                     picoAODs -> <PUB>/picoAOD/<name>/<dataset>/picoAOD_seed<s>*.root,
#                                     per-(seed, year) registry
#   D3_merge (per seed)               -> one clean registry per seed (numpy tags dropped)
#   D3_dataset_yml                    -> one multi-sample dataset: <name>, nSamples: n_seeds, per-year
#                                        files_template with seedXXX (runner expands XXX)
#   D3_publish                        -> <PUB>/handoff/<name>.yml   (what consumers read)

D3_OUT = f"{out}D3/"
D3_DATASET = f"{D3_OUT}handoff/{DATASET_NAME}.yml"
D3_PUBLISHED = f"{D3_OUT}published.done"

rule D3_config:
    input:
        template = DECL.get('skimmer_template', "coffea4bees/skimmer/metadata/declustering_Run3.yml"),
        pdfs = D2_DONE
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
        section = {**(tmpl.get('config') or {}),
                   'base_path': f"{PUB}/picoAOD/{DATASET_NAME}",
                   'clustering_pdfs_file': PDF_TEMPLATE,   # read by the condor workers, via fsspec
                   'declustering_rand_seed': int(wildcards.seed),
                   'subtract_ttbar_with_weights': SUBTRACT_TT}
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
        pdfs = D2_DONE
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

rule D3_merge:
    """Per-year registry keys (data_2022_EEE, ...) never collide across years; the script
    refuses an overlap (a rerun clobbering another year) and drops runner's numpy tags."""
    input: expand(f"{D3_OUT}per_seed/seed{{{{seed}}}}/picoaod_datasets__{{year}}.yml", year=YEARS)
    output: f"{D3_OUT}per_seed/registry_seed{{seed}}.yml"
    log: f"{D3_OUT}logs/merge__seed{{seed}}.log"
    wildcard_constraints:
        seed = r"\d+"
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input} {output} 2>&1 | tee {log}
        """

rule D3_dataset_yml:
    """One dataset key, per-year `files_template` lists with the seed replaced by XXX. Every seed
    must produce the same set of templates -- checked, not assumed: the seedXXX expansion would
    otherwise read files that do not exist, or silently skip ones that do."""
    input: expand(f"{D3_OUT}per_seed/registry_seed{{seed}}.yml", seed=SEEDS)
    output: D3_DATASET
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
                    # keep .chunkN: the files on disk are picoAOD_seed<s>.chunk<k>.root
                    t = re.sub(rf'/picoAOD_seed{s}((\.chunk\d+)?\.root)$', r'/picoAOD_seedXXX\1', fp)
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
        write_yaml(output[0], {DATASET_NAME: dataset})

rule D3_publish:
    input: D3_DATASET
    output: D3_PUBLISHED
    log: f"{D3_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f -p {input} "{HANDOFF}/$(basename {input})" 2>&1 | tee {log}
        echo "published {input} -> {HANDOFF}/$(basename {input})" | tee -a {log}
        date > {output}
        """

rule all_D3:
    input: D3_PUBLISHED

localrules: D3_config, D3_merge, D3_dataset_yml, D3_publish, all_D3
