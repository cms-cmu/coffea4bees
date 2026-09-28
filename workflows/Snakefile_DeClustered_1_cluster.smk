# coffea4bees/workflows/Snakefile_DeClustered_1_cluster.smk
# D.1: learn the jet splittings. processor_cluster_4b.py takes 4b data events (ttbar subtracted
# with the upstream FvT, rand > FvT.d4_to_t4), clusters the four candidate jets into their
# splitting tree and histograms the splitting variables per splitting type and pT bin.
# (scripts/synthetic-dataset-cluster-Run3-all.sh, config analysis/metadata/cluster_4b_Run3.yml)
#
#   D1_config                         runner config (cluster.config_template + this roast's values)
#   D1_cluster (per data year, condor) -> splitting histograms, one coffea per year
#   D1_merge                          -> splittings.coffea (D.2 reads all eras from one file)
#
# With inputs.pdfs set, D.3 declusters with another roast's PDFs and none of this runs.

D1_OUT = f"{out}D1/"
D1_CONFIG = f"{D1_OUT}cluster_4b.yml"
D1_MERGED = f"{D1_OUT}splittings.coffea"

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
             'friends_include': ['FvT']},             # data only: no trigWeight needed
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

rule all_D1:
    input: [] if PDF_EXTERNAL else [D1_MERGED]

localrules: D1_config, D1_merge, all_D1
