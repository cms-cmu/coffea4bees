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

M2_OUT = f"{out}M2/"
M2_CONFIG = f"{M2_OUT}make_mixed_data.yml"
M2_REGISTRY = f"{M2_OUT}picoaod_datasets_{MIX_NAME}.yml"
M2_DATASET = f"{M2_OUT}handoff/{MIX_NAME}.yml"
M2_PUBLISHED = f"{M2_OUT}published.done"

def _rank(r):
    """int, or [rp, rn] for independent per-side ranks (a string from --config is parsed)."""
    if isinstance(r, str):
        r = yaml.safe_load(r)
    return [int(x) for x in r] if isinstance(r, (list, tuple)) else int(r)

rule M2_config:
    input:
        template = MIX.get('skimmer_template', "coffea4bees/skimmer/metadata/mixeddata_Run3.yml"),
        jcm = UPSTREAM_JCM,
    output: M2_CONFIG
    run:
        with open(input.template) as f:
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
                   'JCM_file': input.jcm,
                   'subtract_ttbar_with_weights': True,
                   'hemi_library_yaml': HEMI_LIB_URL,     # read by the condor workers, via fsspec
                   'hemi_stats_path': HEMI_STATS_URL,
                   'default_rank': _rank(MIX.get('default_rank', 0)),
                   'use_topk_matching': bool(MIX.get('use_topk_matching', True)),
                   'k_neighbors': int(MIX.get('k_neighbors', 10)),
                   'collision_mode': MIX.get('collision_mode', 'retry'),
                   'use_boost_corrected_matching': bool(MIX.get('use_boost_corrected_matching', True)),
                   'friends': {'FvT': FVT},
                   'friends_include': ['FvT']}
        cfg = processor_config(section, inherit_config=False,
                               processor="coffea4bees/skimmer/processor/make_mixed_data.py",
                               runner=runner)
        if not isinstance(cfg['config']['JCM_file'], str) or cfg['config']['JCM_file'] != input.jcm:
            raise ValueError(f"mixing config JCM_file is {cfg['config']['JCM_file']!r}, not the upstream JCM {input.jcm}")
        write_yaml(output[0], cfg)

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

rule M2_publish:
    input: M2_DATASET
    output: M2_PUBLISHED
    log: f"{M2_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f -p {input} "{HANDOFF}/$(basename {input})" 2>&1 | tee {log}
        echo "published {input} -> {HANDOFF}/$(basename {input})" | tee -a {log}
        date > {output}
        """

rule all_M2:
    input: M2_PUBLISHED

localrules: M2_config, M2_merge, M2_dataset_yml, M2_publish, all_M2
