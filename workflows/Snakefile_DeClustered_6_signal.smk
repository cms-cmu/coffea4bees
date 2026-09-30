# coffea4bees/workflows/Snakefile_DeClustered_6_signal.smk
# D.6: signal check (as MakeMixedData M.7). The declustered sample is made from the 4b data, which
# contains some signal; declustering must wash out resonances -- each splitting is re-made from a
# real QCD splitting (library) or the PDFs, so a H->bb pair should no longer peak. Otherwise
# signal leaks into the synthetic background model.
#
# Signal MC is declustered with D.3's method, library / PDFs and seed (thinned by event % N).
# Everything is histogrammed with D.4's upstream (non-tight) config and the analysis SvB
# (inputs.SvB_model, the nominal roast's Phase D) evaluated on the fly:
#   the signal          4b; declustered (synthetic_mc_<signal>)
#   the declustered     this roast's multijet handoff (or inputs.declustered_data), seed
#   data                signal_check.seed, thinned by event % N (weights x N)
# Two galleries (make_plots): plots_signal/ (the signal curves) and plots_declustered/ (the
# declustered data with both signals overlaid). The report (M.7's mixeddata_signal_check.py)
# gives the numbers: how much of the high-SvB tail and the Higgs-candidate peak survives.
#
#   D6_config                         skimmer config: D.3's, on signal MC (event_subsample, no ttbar
#                                     subtraction, trigWeight friend, unit weights if absent)
#                                     picoAODs -> <PUB>/picoAOD/signal_declustered
#   D6_decluster (per year, condor)   -> per-year registry
#   D6_dataset_yml                    -> synthetic_mc_<signal>: the declustered files + the original
#                                        sample's normalisation (sumw / subsample, xs, ...)
#   D6_hist_config_{signal,declustered}  D.4's upstream config, SvB on the fly, unblinded
#   D6_hists_{signal,declustered} (per year, condor)
#   D6_merge_hists                    -> histAll_signal_check.coffea
#   D6_plot_config_{signal,declustered} + D6_plots_{signal,declustered} -> plots_*/ (makePlots + gallery)
#   D6_report                         -> signal_check.{txt,yml}
#
# signal_check: {enabled: false} turns it off; datasets: the signal samples (default: the SM ggHH,
# the only Run 3 signal covering 2022 + 2023); subsample: decluster only signal events with
# event % N == 0 (default 10; 1 = all); declustered_data_subsample: the same for the declustered-
# data SvB pass (default 10). The name synthetic_mc_* gives isSyntheticMC in processor_config: MC
# weights, no JEC. It must not contain "mix" (-> mixed data, unit weights).

SIG = config.get('signal_check') or {}
SIG_ENABLED = bool(SIG.get('enabled', True))
SIG_DATASETS = list(SIG.get('datasets', ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"]))
SIG_SEED = int(SIG.get('seed', 0))
SIG_NAMES = {d: f"synthetic_mc_{d}" for d in SIG_DATASETS}
# Decluster only events with event % N == 0: the check measures fractions and shapes (the full ggHH
# 4b sample costs as much CPU as declustering all the data). The dataset's sumw is divided by N.
SIG_SUBSAMPLE = int(SIG.get('subsample', 10))
# The same thinning for the declustered-data SvB pass (millions of events, SvB on the fly);
# processor_HH4b weights the kept events by N.
DATA_SUBSAMPLE = int(SIG.get('declustered_data_subsample', 10))
for _k, _n in (('subsample', SIG_SUBSAMPLE), ('declustered_data_subsample', DATA_SUBSAMPLE)):
    if _n < 1:
        raise ValueError(f"signal_check.{_k} must be >= 1, got {_n}")
SVB_MODEL = INPUTS.get('SvB_model')
if SIG_ENABLED and not str(SVB_MODEL or "").startswith("root://"):
    raise ValueError("signal_check needs inputs.SvB_model: a root:// URL to an upstream roast's SvB result.json")
# The declustered data to compare with: this roast's multijet handoff (after D.3), or another
# roast's (inputs.declustered_data, e.g. a D.6-only roast on an existing library).
DECL_DATA_URL = INPUTS.get('declustered_data') or MJ_URL
DECL_DATA_NAME = str(SIG.get('declustered_data_name', MJ_NAME))
DECL_DATA_DONE = [] if INPUTS.get('declustered_data') else [D3_PUBLISHED]
DECL_DATA_PROCESS = f"{'syn_noTT' if DECL_DATA_NAME.startswith('synthetic_data_noTT') else 'syn'}_v{SIG_SEED}"
PLOT_ERA = 'Run3' if any('202' in y for y in YEARS) else 'RunII'

D6_OUT = f"{out}D6/"
D6_CONFIG = f"{D6_OUT}configs/declustering_signal.yml"
D6_DATASET = f"{D6_OUT}datasets/signal_declustered.yml"
D6_HIST_CONFIG = {k: f"{D6_OUT}analysis_config_{k}.yml" for k in ("signal", "declustered")}
D6_HISTALL = f"{D6_OUT}histAll_signal_check.coffea"
D6_REPORT = f"{D6_OUT}signal_check.txt"
D6_PLOTS = ("signal", "declustered")

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
        # Signal MC passes the 4b selection ~10x more often than data: 5000 events per chunk
        # (4+ GB workers otherwise), times the event % N thinning (only 1/N is declustered).
        runner['chunksize'] = int(SIG.get('chunksize', 5000 * SIG_SUBSAMPLE))
        if config['test']:
            runner['workers'] = 1            # local test on an interactive node: one worker
        section = {**(tmpl.get('config') or {}),
                   'base_path': f"{PUB}/picoAOD/signal_declustered",
                   'clustering_pdfs_file': PDF_TEMPLATE,
                   'declustering_rand_seed': SIG_SEED,
                   'event_subsample': SIG_SUBSAMPLE,           # decluster only event % N == 0
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
    """synthetic_mc_<signal>: per year, the declustered files of that sample and the ORIGINAL
    sample's normalisation (the declustering keeps each event's generator weight), sumw divided by
    signal_check.subsample for the event % N thinning, so the yields compare like with like."""
    input:
        registries = expand(f"{D6_OUT}per_year/picoaod_datasets__{{year}}.yml", year=YEARS),
        merge_script = "coffea4bees/workflows/scripts/merge_mixeddata_registries.py",
        script = "coffea4bees/workflows/scripts/mixeddata_signal_check.py"
    output: D6_DATASET
    log: f"{D6_OUT}logs/dataset_yml.log"
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.merge_script} {input.registries} {D6_OUT}datasets/registry.yml 2>&1 | tee {log}
        {WRAPPER} {PYTHON} {input.script} dataset \
            --registry {D6_OUT}datasets/registry.yml --metadata {config[analysis_config][dataset_location]} \
            --signals {SIG_DATASETS} --years {YEARS} --prefix synthetic_mc_ --subsample {SIG_SUBSAMPLE} \
            -o {output} 2>&1 | tee -a {log}
        """

def _d6_hist_config(src, dst, datasets, subsample=1):
    """D.4's upstream (non-tight) analysis config with the analysis SvB on the fly, unblinded,
    thinned by event % subsample (processor_HH4b weights the kept events by N)."""
    with open(src) as f:
        cfg = yaml.safe_load(f) or {}
    tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
    if tight is not False:
        raise ValueError(f"upstream histogram config has fourTag_use_tight={tight!r}; need the non-tight selection")
    cfg['dataset_location'] = list(datasets)
    cfg.get('runner', {}).pop('dataset_location', None)
    c = cfg.setdefault('config', {})
    # No SvB friend covers the declustered events: evaluate the analysis SvB on the fly. A friend
    # takes precedence over the classifier (load_SvB), so drop them (FvT, a 3b-data weight, too);
    # the MC needs only the trigger weights. Four-tag only: no JCM.
    c['friends'] = {k: v for k, v in (c.get('friends') or {}).items()
                    if not (k.startswith('SvB') or k.startswith('FvT'))}
    c.pop('JCM_file', None)
    c.update({'run_SvB': True, 'SvB': None,
              'SvB_MA': [{'path': SVB_MODEL, 'name': 'Final'}],
              'apply_FvT': False,
              'apply_JCM': False,
              'friends_include': ['trigWeight'],
              'require_trigWeight': False,    # Run 3 ggF has no trigger-weight friend
              'blind': False,
              'event_subsample': subsample})
    if config['test']:
        cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
    write_yaml(dst, cfg)

rule D6_hist_config_signal:
    input: UPSTREAM_HIST_CONFIG
    output: D6_HIST_CONFIG['signal']
    run:
        _d6_hist_config(input[0], output[0], [config['analysis_config']['dataset_location'], D6_DATASET])

rule D6_hist_config_declustered:
    input: UPSTREAM_HIST_CONFIG
    output: D6_HIST_CONFIG['declustered']
    run:
        _d6_hist_config(input[0], output[0], [DECL_DATA_URL], subsample=DATA_SUBSAMPLE)

use rule analysis_processor from analysis as D6_hists_signal with:
    input:
        runner_script = "runner.py",
        config_file = D6_HIST_CONFIG['signal'],
        dataset = D6_DATASET
    output: f"{D6_OUT}singlefiles/hist__signal__{{year}}.coffea"
    log: f"{D6_OUT}logs/hists_signal__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_DATASETS + list(SIG_NAMES.values())),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule analysis_processor from analysis as D6_hists_declustered with:
    input:
        runner_script = "runner.py",
        config_file = D6_HIST_CONFIG['declustered'],
        published = DECL_DATA_DONE
    output: f"{D6_OUT}singlefiles/hist__{DECL_DATA_NAME}__{{year}}.coffea"
    log: f"{D6_OUT}logs/hists_declustered__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = DECL_DATA_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as D6_merge_hists with:
    input:
        files = expand(f"{D6_OUT}singlefiles/hist__signal__{{year}}.coffea", year=YEARS)
                + expand(f"{D6_OUT}singlefiles/hist__{DECL_DATA_NAME}__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: D6_HISTALL
    log: f"{D6_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

def _plot_entry(process, tag, label, color, linestyle="solid", scale=1, histtype="step"):
    return {'process': list(process), 'tag': tag, 'label': label, 'edgecolor': color, 'fillcolor': color,
            'histtype': histtype, 'linestyle': linestyle, 'scalefactor': scale}

_SUMMARY = ['SvB_MA.ps_hh', 'SvB_MA.ps', 'quadJet_selected.lead.mass', 'quadJet_selected.subl.mass',
            'quadJet_selected.xHH', 'm4j', 'v4j.mass']

rule D6_plot_config_signal:
    """The signal curves, no data or background: the 4b signal and the declustered signal."""
    output: f"{D6_OUT}plots_signal.yml"
    run:
        cfg = {'hists': {
                   'HH4b_4b': _plot_entry(SIG_DATASETS, 'fourTag', "4b signal", "#e42536"),
                   'HH4b_declustered': _plot_entry(SIG_NAMES.values(), 'fourTag', "declustered signal", "#1f77b4", "dashed")},
               'doRatio': 0,
               'summary': _SUMMARY}
        write_yaml(output[0], cfg)

rule D6_plot_config_declustered:
    """The declustered data (the synthetic background model) with the 4b signal and the declustered
    signal overlaid (x100, the scale of the analysis signal plots); the ratio panel is each (x100)
    signal / the declustered data."""
    output: f"{D6_OUT}plots_declustered.yml"
    run:
        cfg = {'hists': {
                   'HH4b_4b': _plot_entry(SIG_DATASETS, 'fourTag', "4b signal (x100)", "#e42536", "solid", 100),
                   'HH4b_declustered': _plot_entry(SIG_NAMES.values(), 'fourTag', "declustered signal (x100)",
                                                   "#1f77b4", "dashed", 100)},
               'stack': {
                   'DeclusteredData': {'process': DECL_DATA_PROCESS, 'tag': 'fourTag', 'fillcolor': "#FFDF7Fff",
                                       'edgecolor': 'k', 'label': "Declustered data"}},
               'ratios': {
                   'sig4bToDeclustered': {'numerator': {'type': 'hists', 'key': 'HH4b_4b'},
                                          'denominator': {'type': 'stack'},
                                          'uncertianty': 'nominal', 'color': "#e42536", 'marker': "s"},
                   'declSigToDeclustered': {'numerator': {'type': 'hists', 'key': 'HH4b_declustered'},
                                            'denominator': {'type': 'stack'},
                                            'uncertianty': 'nominal', 'color': "#1f77b4", 'marker': "o"}},
               'doRatio': 1,
               'summary': _SUMMARY}
        write_yaml(output[0], cfg)

use rule make_plots from analysis as D6_plots_signal with:
    input:
        coffea_file = D6_HISTALL,
        metadata_file = f"{D6_OUT}plots_signal.yml",
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{D6_OUT}plots_signal/plots_done.txt"
    params:
        output_dir = f"{D6_OUT}plots_signal/",
        metadata = f"{D6_OUT}plots_signal.yml",
        extra_arguments = f"-s xW -f png --year {PLOT_ERA}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{D6_OUT}logs/plots_signal.log"

use rule make_plots from analysis as D6_plots_declustered with:
    input:
        coffea_file = D6_HISTALL,
        metadata_file = f"{D6_OUT}plots_declustered.yml",
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{D6_OUT}plots_declustered/plots_done.txt"
    params:
        output_dir = f"{D6_OUT}plots_declustered/",
        metadata = f"{D6_OUT}plots_declustered.yml",
        extra_arguments = f"-s xW -f png --year {PLOT_ERA}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{D6_OUT}logs/plots_declustered.log"

rule D6_report:
    input:
        hists = D6_HISTALL,
        script = "coffea4bees/workflows/scripts/mixeddata_signal_check.py"
    output:
        txt = D6_REPORT,
        yml = f"{D6_OUT}signal_check.yml"
    log: f"{D6_OUT}logs/report.log"
    params:
        # original:(no 3b origin):declustered process names
        triples = " ".join(f"{d}:-:{SIG_NAMES[d]}" for d in SIG_DATASETS)
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.script} report --label declustered --title "D.6 signal check" \
            --hists {input.hists} --samples {params.triples} -o {D6_OUT} 2>&1 | tee {log}
        """

rule all_D6:
    input: [D6_REPORT] + [f"{D6_OUT}plots_{k}/plots_done.txt" for k in D6_PLOTS] if SIG_ENABLED else []

localrules: D6_config, D6_dataset_yml, D6_hist_config_signal, D6_hist_config_declustered, D6_merge_hists,
            D6_plot_config_signal, D6_plot_config_declustered, D6_plots_signal, D6_plots_declustered,
            D6_report, all_D6
