# coffea4bees/workflows/Snakefile_DeClustered_1_cluster.smk
# D.1: learn the jet splittings. processor_cluster_4b.py takes 4b data events (ttbar subtracted
# with the upstream FvT, rand > FvT.d4_to_t4), clusters the four candidate jets into their
# splitting tree and histograms the splitting variables per splitting type and pT bin.
# (formerly scripts/synthetic-dataset-cluster-Run3-all.sh; config analysis/metadata/cluster_4b_Run3.yml,
#  Run 2 cluster_4b.yml)
#
#   D1_config                         runner config (cluster.config_template + this roast's values)
#   D1_cluster (per data year, condor) -> splitting histograms, one coffea per year
#   D1_merge                          -> splittings.coffea (D.2 reads all eras from one file)
#
# declustering.method: library -- D1_cluster also writes the splitting library (one ROOT file per
# chunk, <LIB_BASE>/<dataset>/), and
#   D1_library_regroup (per year)     the chunk files listed in that year's coffea -> {year: [files]}
#   D1_library_merge                  -> one registry for all years
#   D1_library_publish                -> <LIB_BASE>/splitting_library.yml (what D.3 reads)
#   D1_library_summary (per year)     -> D1/library/summary/: rows per exact splitting type and the
#                                        lookup group each resolves to (summarize_splitting_library.py)
#
# With inputs.pdfs set (pdf method), D.3 declusters with another roast's PDFs and none of this runs.
# D1_merge (the histograms' only consumer is D.2) runs only when the PDFs are made (MAKE_PDFS).

D1_OUT = f"{out}D1/"
D1_CONFIG = f"{D1_OUT}cluster_4b.yml"
D1_MERGED = f"{D1_OUT}splittings.coffea"
D1_LIB_REGISTRY = f"{D1_OUT}library/splitting_library.yml"
D1_LIB_PUBLISHED = f"{D1_OUT}library/published.done"
LIB_DONE = [D1_LIB_PUBLISHED] if BUILD_LIBRARY else []            # inputs.splitting_library: nothing to wait for
D1_LIB_SUMMARIES = [f"{D1_OUT}library/summary/splitting_library_summary_{y}.yml" for y in YEARS] if BUILD_LIBRARY else []

rule D1_config:
    input: CLUSTER.get('config_template', "coffea4bees/analysis/metadata/cluster_4b_Run3.yml")
    output: D1_CONFIG
    run:
        with open(input[0]) as f:
            tmpl = yaml.safe_load(f) or {}
        cfg = processor_config(
            {**(tmpl.get('config') or {}),
             'clustering_pdfs_file': "None",          # learn the splittings, don't sample them
             'subtract_ttbar_with_weights': True,     # turns apply_FvT on (processor_HH4b)
             'run_SvB': False,
             'fourTag_use_tight': False,              # the 4b sample the DeClusterer re-generates
             'friends': {'FvT': FVT},                 # merged over friend_file by runner.py
             'friends_include': ['FvT'],              # data only: no trigWeight needed
             **({'splitting_library_base_path': LIB_BASE,
                 'splitting_library_carry_fields': list(LIB_OPTS.get('carry_fields', ['btagScore']))}
                if BUILD_LIBRARY else {})},
            # the script ran cluster_4b_Run3.yml alone, on the processor defaults: not the
            # histogram-pass settings (top reconstruction, btagSF, ...) of analysis_config.config
            inherit_config=False,
            processor="coffea4bees/analysis/processors/processor_cluster_4b.py",
            runner={**(tmpl.get('runner') or {}),
                    **{k: CLUSTER[k] for k in ('worker_memory', 'chunksize') if k in CLUSTER}})
        if not cfg['config'].get('subtract_ttbar_with_weights'):
            raise ValueError("D.1 must subtract ttbar: the PDFs describe the multijet splittings")
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as D1_cluster with:
    input:
        runner_script = "runner.py",
        config_file = D1_CONFIG
    output: f"{D1_OUT}cluster/splittings__{{year}}.coffea"
    log: f"{D1_OUT}logs/cluster__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = "data",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as D1_merge with:
    input:
        files = expand(f"{D1_OUT}cluster/splittings__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: D1_MERGED
    log: f"{D1_OUT}logs/merge.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

rule D1_library_regroup:
    """Every dataset key in one year's cluster output belongs to that year (same layout as the
    hemisphere library's, hence the same script)."""
    input: f"{D1_OUT}cluster/splittings__{{year}}.coffea"
    output: f"{D1_OUT}library/per_year/splitting_library__{{year}}.yml"
    log: f"{D1_OUT}logs/library_regroup__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/regroup_hemi_library.py \
            {wildcards.year} {input} {output} 2>&1 | tee {log}
        """

rule D1_library_merge:
    input: expand(f"{D1_OUT}library/per_year/splitting_library__{{year}}.yml", year=YEARS)
    output: D1_LIB_REGISTRY
    run:
        merged = {}
        for path in input:
            with open(path) as f:
                for key, files in (yaml.safe_load(f) or {}).items():
                    files = [files] if isinstance(files, str) else list(files)
                    if key in merged:
                        raise ValueError(f"{path}: splitting library year {key} listed twice")
                    if not all(str(p).startswith("root://") for p in files):
                        # the condor workers read the library; a local path would not reach them
                        raise ValueError(f"{path}: splitting library files are not on EOS: {files[:2]}")
                    merged[key] = files
        missing = [y for y in YEARS if not merged.get(y)]
        if missing:
            raise ValueError(f"splitting library has no files for {missing}")
        write_yaml(output[0], merged)

rule D1_library_publish:
    input: D1_LIB_REGISTRY
    output: D1_LIB_PUBLISHED
    log: f"{D1_OUT}logs/library_publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f -p {input} "{LIB_REGISTRY_URL}" 2>&1 | tee {log}
        echo "published {input} -> {LIB_REGISTRY_URL}" | tee -a {log}
        date > {output}
        """

rule D1_library_summary:
    input: D1_LIB_REGISTRY
    output:
        yml = f"{D1_OUT}library/summary/splitting_library_summary_{{year}}.yml",
        txt = f"{D1_OUT}library/summary/splitting_library_summary_{{year}}.txt"
    log: f"{D1_OUT}logs/library_summary__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        min_entries = int(LIB_OPTS.get('min_entries', 10))
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        {WRAPPER} {PYTHON} coffea4bees/jet_clustering/summarize_splitting_library.py {input} {wildcards.year} \
            -o $(dirname {output.yml}) --min-entries {params.min_entries} 2>&1 | tee {log}
        """

rule all_D1:
    input: ([D1_MERGED] if MAKE_PDFS else []) + LIB_DONE + D1_LIB_SUMMARIES

localrules: D1_config, D1_merge, D1_library_regroup, D1_library_merge, D1_library_publish, D1_library_summary, all_D1
