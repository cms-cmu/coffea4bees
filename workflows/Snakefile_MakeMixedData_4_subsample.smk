# coffea4bees/workflows/Snakefile_MakeMixedData_4_subsample.smk
# M.4: split mixeddata_all into N statistically independent pseudo-experiments.
#
# split_mixed_data.py gives each mixed 4b event a pseudo-tag weight w from the M.3 mixed-data JCM
# and puts it in subsample v when one per-event uniform number lies in [v*w, (v+1)*w)
# (mixing_helpers.assign_mixed_subsamples): N unweighted, disjoint samples as long as N*w <= 1.
# Events with N*w > 1 wrap to (event+v) % 9 and are shared -- M.3's study counts them.
#
#   M4_split_config (per v)           skimmer config (subsample.split_template + this roast's values)
#   M4_split (per v, condor)          picoAODs -> <PUB>/picoAOD/subsamples/v<v>/, registry
#   M4_clean (per v)                  registry -> plain YAML (runner leaves numpy tags in it)
#   M4_dataset_yml                    -> one multi-sample dataset: `mixeddata_4b`, nSamples: N,
#                                        per-year files_template with vXXX (runner expands XXX)
#   M4_publish                        -> <PUB>/handoff/mixeddata_4b.yml   (what the closure reads)

M4_OUT = f"{out}M4/"
SUB = config.get('subsamples') or {}
N_SUB = int(SUB.get('n', 16))
SUBSAMPLES = list(range(N_SUB))
SUB_NAME = SUB.get('dataset_name', 'mixeddata_4b')
M4_DATASET = f"{M4_OUT}handoff/{SUB_NAME}.yml"
M4_PUBLISHED = f"{M4_OUT}published.done"

rule M4_split_config:
    input:
        template = SUB.get('split_template', "coffea4bees/skimmer/metadata/split_mixeddata_Run3.yml"),
        jcm = MIXED_JCM
    output: f"{M4_OUT}configs/split_v{{v}}.yml"
    wildcard_constraints:
        v = r"\d+"
    run:
        with open(input.template) as f:
            tmpl = yaml.safe_load(f) or {}
        section = {**(tmpl.get('config') or {}),
                   'base_path': f"{PUB}/picoAOD/subsamples/v{wildcards.v}",
                   'apply_JCM': True,
                   'JCM_file': input.jcm,          # not the template's *_splitting.txt
                   'mixed_subsample': int(wildcards.v),
                   'n_subsamples': N_SUB}
        cfg = processor_config(section, inherit_config=False,
                               processor="coffea4bees/skimmer/processor/split_mixed_data.py",
                               dataset_location=[MIXED_URL],
                               runner=tmpl.get('runner') or {})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M4_split with:
    input:
        runner_script = "runner.py",
        config_file = f"{M4_OUT}configs/split_v{{v}}.yml",
        published = M2_PUBLISHED
    output: f"{M4_OUT}per_subsample/picoaod_datasets_{MIX_NAME}_v{{v}}.yml"
    log: f"{M4_OUT}logs/split_v{{v}}.log"
    wildcard_constraints:
        v = r"\d+"
    params:
        datasets = MIX_NAME,
        years = " ".join(YEARS),
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule M4_clean:
    input: f"{M4_OUT}per_subsample/picoaod_datasets_{MIX_NAME}_v{{v}}.yml"
    output: f"{M4_OUT}per_subsample/clean_v{{v}}.yml"
    log: f"{M4_OUT}logs/clean_v{{v}}.log"
    wildcard_constraints:
        v = r"\d+"
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/merge_mixeddata_registries.py \
            {input} {output} 2>&1 | tee {log}
        """

rule M4_dataset_yml:
    """One dataset key, per-year `files_template` lists with the subsample index replaced by XXX.
    Every subsample must produce the same set of templates -- checked, not assumed (E.2 parsed
    only v0's registry, line by line)."""
    input: expand(f"{M4_OUT}per_subsample/clean_v{{v}}.yml", v=SUBSAMPLES)
    output: M4_DATASET
    run:
        import re
        from src.tools.make_dataset_yml import parse_dataset_key
        per_v = []
        for v, path in zip(SUBSAMPLES, input):
            with open(path) as f:
                registry = yaml.safe_load(f) or {}
            templates = {}
            for key, entry in registry.items():
                year, _era = parse_dataset_key(key)
                if year is None:
                    raise ValueError(f"{path}: cannot tell the year of registry key {key!r}")
                for fp in (entry or {}).get('files') or []:
                    t = re.sub(rf'/subsamples/v{v}/', '/subsamples/vXXX/', fp)
                    # keep .chunkN: the files on disk are picoAOD_mixed_v<v>.chunk<k>.root (as in the
                    # existing mixeddata_4b.yml); E.2 dropped it, pointing at files that don't exist
                    t = re.sub(rf'_v{v}((\.chunk\d+)?\.root)$', r'_vXXX\1', t)
                    if 'XXX' not in t:
                        raise ValueError(f"{path}: {fp} does not carry subsample index v{v}")
                    templates.setdefault(year, set()).add(t)
            per_v.append(templates)
        for v, templates in zip(SUBSAMPLES[1:], per_v[1:]):
            if templates != per_v[0]:
                only_v = sorted(set().union(*templates.values()) - set().union(*per_v[0].values()))[:3]
                only_0 = sorted(set().union(*per_v[0].values()) - set().union(*templates.values()))[:3]
                raise ValueError(f"subsample v{v} files differ from v0 once templated (e.g. only v{v}: "
                                 f"{only_v}, only v0: {only_0}); the vXXX expansion would read "
                                 f"missing files or skip existing ones")
        missing = [y for y in YEARS if y not in per_v[0]]
        if missing:
            raise ValueError(f"no subsample files for {missing}")
        dataset = {'nSamples': N_SUB, 'xs': {'Run2': 1, 'Run3': 1}}
        for year in YEARS:
            dataset[year] = {'picoAOD': {'files_template': sorted(per_v[0][year])}}
        write_yaml(output[0], {SUB_NAME: dataset})

rule M4_publish:
    input: M4_DATASET
    output: M4_PUBLISHED
    log: f"{M4_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f -p {input} "{HANDOFF}/$(basename {input})" 2>&1 | tee {log}
        echo "published {input} -> {HANDOFF}/$(basename {input})" | tee -a {log}
        date > {output}
        """

rule all_M4:
    input: M4_PUBLISHED

localrules: M4_split_config, M4_clean, M4_dataset_yml, M4_publish, all_M4
