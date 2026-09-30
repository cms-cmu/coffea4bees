# coffea4bees/workflows/Snakefile_DeClustered_6_signal.smk
# D.6: signal scrambling check. The declustered sample is made from the 4b data, which contains some
# signal; the declustering must wash out resonances (a H->bb pair is re-made from real QCD
# splittings / PDF samples, so its mass should no longer peak). Decluster signal MC with the same
# method, library / PDFs and seed as D.3, histogram it next to the original MC with D.4's analysis
# config with the analysis SvB (inputs.SvB_model) evaluated on the fly, and report how much of the
# Higgs-candidate peak -- and of the high-SvB tail -- survives.
#
#   D6_config                         skimmer config: D.3's, on signal MC (no ttbar subtraction,
#                                     trigWeight friend), picoAODs -> <PUB>/picoAOD/signal_declustered
#   D6_decluster (per year, condor)   -> per-year registry
#   D6_dataset_yml                    -> synthetic_mc_<signal>: the declustered files + the original
#                                        sample's normalisation (sumw, xs, ...)
#   D6_hist_config                    D.4's upstream analysis config, reading the original signal
#                                     metadata and the D6 dataset yml
#   D6_hists (per year, condor)       processor_HH4b over the original and the declustered signal
#   D6_merge_hists                    -> histAll_signal_check.coffea
#   D6_report                         -> signal_check.{txt,yml} + mass overlays (plots/)
#
# signal_check: {enabled: false} turns it off; datasets: the signal samples (default: the SM ggHH,
# the only Run 3 signal covering 2022 + 2023).

SIG = config.get('signal_check') or {}
SIG_ENABLED = bool(SIG.get('enabled', True))
SIG_DATASETS = list(SIG.get('datasets', ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"]))
SIG_SEED = int(SIG.get('seed', 0))
# max_chunks: decluster only this many chunks per sample-year (the fractions / shapes the check
# reports need tens of thousands of 4b events, not the whole ~3M-event 4b sample); None: all.
SIG_MAX_CHUNKS = SIG.get('max_chunks')
SIG_NAMES = {d: f"synthetic_mc_{d}" for d in SIG_DATASETS}   # 'synthetic_mc_*': runner type mc, own process
SVB_MODEL = INPUTS.get('SvB_model')
if SIG_ENABLED and SVB_MODEL and not str(SVB_MODEL).startswith("root://"):
    raise ValueError("inputs.SvB_model must be a root:// URL to an upstream roast's classifier result.json")

D6_OUT = f"{out}D6/"
D6_CONFIG = f"{D6_OUT}configs/declustering_signal.yml"
D6_DATASET = f"{D6_OUT}datasets/signal_declustered.yml"
D6_HIST_CONFIG = f"{D6_OUT}analysis_config_signal.yml"
D6_HISTALL = f"{D6_OUT}histAll_signal_check.coffea"
D6_REPORT = f"{D6_OUT}signal_check.txt"

rule D6_config:
    input:
        template = DECL.get('skimmer_template', "coffea4bees/skimmer/metadata/declustering_Run3.yml"),
        source = D3_SOURCE_DONE
    output: D6_CONFIG
    run:
        with open(input.template) as f:
            tmpl = yaml.safe_load(f) or {}
        runner = {**(tmpl.get('runner') or {})}
        for k in ('worker_memory', 'chunksize'):
            if k in DECL:
                runner[k] = DECL[k]
        # Signal MC passes the 4b selection ~10x more often than data, so a data-sized chunk means
        # ~10x the events to decluster per task (4+ GB workers): smaller chunks.
        runner['chunksize'] = int(SIG.get('chunksize', 5000))
        if config['test']:
            runner['workers'] = 1            # local test on an interactive node: one worker
        elif SIG_MAX_CHUNKS:
            runner['maxchunks'] = int(SIG_MAX_CHUNKS)
        section = {**(tmpl.get('config') or {}),
                   'base_path': f"{PUB}/picoAOD/signal_declustered",
                   'clustering_pdfs_file': PDF_TEMPLATE,
                   'declustering_rand_seed': SIG_SEED,
                   'subtract_ttbar_with_weights': False,       # signal MC: nothing to subtract
                   'friends_include': ['trigWeight'],           # the GluGlu MC trigger weights ...
                   'require_trigWeight': False}                 # ... which Run 3 ggF lacks: unit weights
                                                                # (a shape test; the analysis skips them too)
        for k in ('b_pt_threshold', 'dr_threshold', 'max_jet_retry', 'max_event_retry'):
            if k in DECL:
                section[k] = DECL[k]
        if LIBRARY:
            section.update({'clustering_pdfs_file': "None",
                            'declustering_method': 'library',
                            'clustering_library_file': LIB_REGISTRY_URL,
                            **{f"library_{k}": LIB_OPTS[k]
                               for k in ('carry_fields', 'min_entries', 'scale_pt', 'boost_z', 'selection', 'k_neighbors', 'max_distance', 'mass_match_weight')
                               if k in LIB_OPTS}})
        cfg = processor_config(section, inherit_config=False,
                               processor="coffea4bees/skimmer/processor/make_declustered_data_4b.py",
                               runner=runner)
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as D6_decluster with:
    input:
        runner_script = "runner.py",
        config_file = D6_CONFIG,
        source = D3_SOURCE_DONE
    output: f"{D6_OUT}per_year/picoaod_datasets__{{year}}.yml"
    log: f"{D6_OUT}logs/decluster__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_DATASETS),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule D6_dataset_yml:
    """synthetic_mc_<signal>: per year, the declustered files of that sample (registry keys are
    <signal>_<year>) and the ORIGINAL sample's normalisation -- the declustering keeps every event
    and its generator weight, so sumw / count / xs carry over unchanged."""
    input:
        registries = expand(f"{D6_OUT}per_year/picoaod_datasets__{{year}}.yml", year=YEARS),
        merge_script = "coffea4bees/workflows/scripts/merge_mixeddata_registries.py"
    output: D6_DATASET
    log: f"{D6_OUT}logs/dataset_yml.log"
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.merge_script} {input.registries} {D6_OUT}datasets/registry.yml 2>&1 | tee {log}
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/declustered_signal_check.py dataset \
            --registry {D6_OUT}datasets/registry.yml --metadata {config[analysis_config][dataset_location]} \
            --signals {SIG_DATASETS} --years {YEARS} --prefix synthetic_mc_ -o {output} 2>&1 | tee -a {log}
        """

rule D6_hist_config:
    input: UPSTREAM_HIST_CONFIG
    output: D6_HIST_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        base = config['analysis_config']['dataset_location']
        cfg['dataset_location'] = [base, D6_DATASET]
        cfg.get('runner', {}).pop('dataset_location', None)
        # no Run 3 trigger-weight friend covers the ggF signal (as in the nominal roast's signal passes)
        cfg.setdefault('config', {})['require_trigWeight'] = False
        if SVB_MODEL:
            # on-the-fly SvB (no friend trees exist for the declustered events): the same model
            # for the original and the declustered signal
            cfg.setdefault('config', {}).update({'run_SvB': True, 'SvB': None,
                                                 'SvB_MA': [{'path': SVB_MODEL, 'name': 'Final'}]})
        if config['test']:
            cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as D6_hists with:
    input:
        runner_script = "runner.py",
        config_file = D6_HIST_CONFIG,
        dataset = D6_DATASET
    output: f"{D6_OUT}singlefiles/hist__signal__{{year}}.coffea"
    log: f"{D6_OUT}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_DATASETS + list(SIG_NAMES.values())),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as D6_merge_hists with:
    input:
        files = expand(f"{D6_OUT}singlefiles/hist__signal__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: D6_HISTALL
    log: f"{D6_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

rule D6_report:
    input: D6_HISTALL
    output:
        txt = D6_REPORT,
        yml = f"{D6_OUT}signal_check.yml"
    log: f"{D6_OUT}logs/report.log"
    params:
        pairs = " ".join(f"{d}:{n}" for d, n in SIG_NAMES.items()),
        # a subsample is normalised to the full sample's sumw: its yields are not comparable
        subsampled = "--subsampled" if (SIG_MAX_CHUNKS or config['test']) else ""
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"; mkdir -p $MPLCONFIGDIR
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/declustered_signal_check.py report \
            --hists {input} --pairs {params.pairs} -o {D6_OUT} {params.subsampled} 2>&1 | tee {log}
        """

rule all_D6:
    input: [D6_REPORT] if SIG_ENABLED else []

localrules: D6_config, D6_dataset_yml, D6_hist_config, D6_merge_hists, D6_report, all_D6
