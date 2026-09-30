# coffea4bees/workflows/Snakefile_MakeMixedData_7_signal.smk
# M.7: signal check. The data the mixed sample is made from contains some signal: the 3b events
# that are mixed, and the 4b events whose hemispheres fill the library. Mixing must break the
# HH correlations, since each hemisphere is replaced by a library hemisphere from a different
# event. Otherwise signal leaks into the mixed background model, and so into MvD and the SvB fit.
#
# Signal MC is mixed with M.2's settings and M.1's library (every preselected event, thinned by
# event % N): 3b events get pseudo-tags from the M.2 JCM, as in the data; 4b events keep their
# real tags; each mixed event records its origin. Everything is histogrammed with this roast's
# upstream (M.3) config and the analysis SvB (inputs.SvB_model, the nominal roast's Phase D)
# evaluated on the fly:
#   the signal     4b; 3b x M.2 JCM; mixed from 3b (synthetic_mc_3b_*) x the M.2 JCM on the input
#                  event's untagged loose jets -- the weight a signal event in the 3b data carries
#                  before it is mixed; mixed from 4b (synthetic_mc_4b_*), no JCM
#   mixeddata_all  x M.3's mixed-data JCM (as M.6), thinned by event % N (weights x N)
# Two galleries (make_plots): plots_signal/ (the four signal curves) and plots_mixeddata/
# (mixeddata_all with the mixed 3b signal overlaid). The report gives the numbers: how much of the
# high-SvB tail and the Higgs-candidate peak survives.
#
#   M7_config                        skimmer config: M.2's, on signal MC (mix_tags threeTag_fourTag,
#                                    event_subsample, no ttbar subtraction, trigWeight friend)
#                                    picoAODs -> <PUB>/picoAOD/signal_mixed
#   M7_mix (per year, condor)        -> per-year registry
#   M7_dataset_yml                   -> synthetic_mc_{3b,4b}_<signal>: both the same mixed files + the
#                                       original sample's normalisation (sumw / subsample, xs, ...)
#   M7_hist_config_signal            upstream config: SvB on the fly, the M.2 JCM, unblinded
#   M7_hist_config_mixeddata         upstream config on mixeddata_all: the M.3 JCM (as M.6), SvB on
#                                    the fly, event_subsample
#   M7_hists_signal, M7_hists_mixeddata (per year, condor)
#   M7_merge_hists                   -> histAll_signal_check.coffea
#   M7_plot_config_{signal,mixeddata} + M7_plots_{signal,mixeddata}  -> plots_*/ (makePlots + gallery)
#   M7_report                        -> signal_check.{txt,yml}
#
# 4b mixing: only the 4b signal events are mixed (mix_tags fourTag, no JCM), with seed 0's settings;
# the report's "original" is then compared through its four-tag histograms.
#
# signal_check: {enabled: false} turns it off; datasets: the signal samples.
# The name synthetic_mc_* gives isSyntheticMC in processor_config: MC weights with the b-tag SF
# stored at mixing time (CMSbtag), no JEC, no truth matching. It must not contain "mix", which
# would make it mixed data (unit weights).

SIG = config.get('signal_check') or {}
SIG_ENABLED = bool(SIG.get('enabled', True))
SIG_DATASETS = list(SIG.get('datasets', ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"]))
# One set of mixed files, two datasets by the input event's tag (processor_HH4b keeps each origin
# and weights the 3b one by the JCM). The names must contain synthetic_mc and not "mix".
SIG_3B = {d: f"synthetic_mc_3b_{d}" for d in SIG_DATASETS}
SIG_4B = {d: f"synthetic_mc_4b_{d}" for d in SIG_DATASETS}
# Mix only events with event % N == 0: the check measures fractions and shapes, and 1/10 of the
# signal still leaves ~5k Run 3 events in SR at SvB > 0.8 (~1.5% on the surviving fraction) at
# ~1/10 the mixing and histogramming cost. The dataset's sumw is divided by N. 1: every event.
SIG_SUBSAMPLE = int(SIG.get('subsample', 10))
# The same thinning for the mixeddata_all SvB pass (tens of millions of events, SvB on the fly);
# processor_HH4b weights the kept events by N.
MIX_SUBSAMPLE = int(SIG.get('mixeddata_subsample', 10))
for _k, _n in (('subsample', SIG_SUBSAMPLE), ('mixeddata_subsample', MIX_SUBSAMPLE)):
    if _n < 1:
        raise ValueError(f"signal_check.{_k} must be >= 1, got {_n}")
SVB_MODEL = INPUTS.get('SvB_model')
if SIG_ENABLED and not str(SVB_MODEL or "").startswith("root://"):
    raise ValueError("signal_check needs inputs.SvB_model: a root:// URL to an upstream roast's SvB result.json")
PLOT_ERA = 'Run3' if any('202' in y for y in YEARS) else 'RunII'

M7_OUT = f"{out}M7/"
M7_CONFIG = f"{M7_OUT}make_mixed_signal.yml"
M7_DATASET = f"{M7_OUT}datasets/signal_mixed.yml"
M7_HIST_CONFIG = {k: f"{M7_OUT}analysis_config_{k}.yml" for k in ("signal", "mixeddata")}
M7_HISTALL = f"{M7_OUT}histAll_signal_check.coffea"
M7_REPORT = f"{M7_OUT}signal_check.txt"
M7_PLOTS = ("signal", "mixeddata")

rule M7_config:
    input:
        template = MIX.get('skimmer_template', "coffea4bees/skimmer/metadata/mixeddata_Run3.yml"),
        jcm = UPSTREAM_JCM,
        m2 = M2_CONFIG
    output: M7_CONFIG
    run:
        # M.2's mixing config (same library, matching, ranks and JCM), re-pointed at the signal
        with open(input.m2) as f:
            cfg = yaml.safe_load(f) or {}
        step = int(SIG.get('chunksize', 20000))
        cfg['runner'] = {**cfg.get('runner', {}), 'chunksize': step,
                         'worker_memory': SIG.get('worker_memory', MIX.get('worker_memory', '8GB'))}
        cfg['config'].update({
            'base_path': f"{PUB}/picoAOD/signal_mixed",
            'step': step,
            # all preselected signal, 3b and 4b; 4b mixing: the 4b events (no JCM)
            'mix_tags': 'fourTag' if MIX4B else 'threeTag_fourTag',
            'subtract_ttbar_with_weights': False, # signal MC: nothing to subtract
            'friends': {},
            'friends_include': ['trigWeight'],    # the GluGlu MC trigger weights ...
            'require_trigWeight': False,          # ... which Run 3 ggF lacks: unit weights (shape test)
        })
        if MIX4B:
            cfg['config']['exclude_source_event'] = False   # signal MC is not in the library
        if config['test']:
            cfg['runner'].update({'condor': False, 'shared_dask': False, 'workers': 1})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M7_mix with:
    input:
        runner_script = "runner.py",
        config_file = M7_CONFIG,
        jcm = UPSTREAM_JCM,
        hemilib = M1_DONE,
    output: f"{M7_OUT}per_year/picoaod_datasets_signal_mixed__{{year}}.yml"
    log: f"{M7_OUT}logs/mix__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_DATASETS),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, ["-s", TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule M7_dataset_yml:
    """synthetic_mc_{3b,4b}_<signal>: per year, the mixed files of that sample and the ORIGINAL sample's
    normalisation. Mixing keeps each event's generator weight, but it drops the events that are
    not preselected and the unresolved hemisphere collisions. So the sumw is the original one,
    divided by signal_check.subsample for the event % N thinning, and the yields compare like
    with like."""
    input:
        registries = expand(f"{M7_OUT}per_year/picoaod_datasets_signal_mixed__{{year}}.yml", year=YEARS),
        merge_script = "coffea4bees/workflows/scripts/merge_mixeddata_registries.py",
        script = "coffea4bees/workflows/scripts/mixeddata_signal_check.py"
    output: M7_DATASET
    log: f"{M7_OUT}logs/dataset_yml.log"
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.merge_script} {input.registries} {M7_OUT}datasets/registry.yml 2>&1 | tee {log}
        {WRAPPER} {PYTHON} {input.script} dataset \
            --registry {M7_OUT}datasets/registry.yml --metadata {config[analysis_config][dataset_location]} \
            --signals {SIG_DATASETS} --years {YEARS} --prefix synthetic_mc_3b_ synthetic_mc_4b_ --subsample {SIG_SUBSAMPLE} \
            -o {output} 2>&1 | tee -a {log}
        """

def _m7_hist_config(src, dst, datasets, jcm_file, mixed_data=False, subsample=1):
    """This roast's upstream (M.3) analysis config with the analysis SvB on the fly.
    Signal: the M.2 JCM on the 3b events (and on the 3b-origin mixed signal, processor_HH4b).
    mixed_data: M.6's mixeddata_all weighting (the M.3 JCM on the four-tag mixed events via the
    apply_MvD branch, no MvD friend), thinned by event % subsample."""
    with open(src) as f:
        cfg = yaml.safe_load(f) or {}
    tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
    if tight is not False:
        raise ValueError(f"upstream histogram config has fourTag_use_tight={tight!r}; need the non-tight selection")
    cfg['dataset_location'] = list(datasets)
    cfg.get('runner', {}).pop('dataset_location', None)
    c = cfg.setdefault('config', {})
    # No SvB friend covers the mixed events: evaluate the analysis SvB on the fly. A friend takes
    # precedence over the classifier (load_SvB), so drop them (FvT, a 3b-data weight, too); the
    # MC needs only the trigger weights.
    c['friends'] = {k: v for k, v in (c.get('friends') or {}).items()
                    if not (k.startswith('SvB') or k.startswith('FvT'))}
    c.update({'run_SvB': True, 'SvB': None,
              'SvB_MA': [{'path': SVB_MODEL, 'name': 'Final'}],
              'apply_FvT': False,
              'apply_JCM': True, 'JCM_file': jcm_file,
              'friends_include': ['trigWeight'],
              'require_trigWeight': False,    # Run 3 ggF has no trigger-weight friend
              'blind': False,                 # a blinded synthetic_mc lost its SR SvB tail
              'event_subsample': subsample})
    if mixed_data:
        c.update({'apply_MvD': True, 'apply_MvD_weight': False})
    if config['test']:
        cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
    write_yaml(dst, cfg)

rule M7_hist_config_signal:
    input:
        hist_config = UPSTREAM_HIST_CONFIG,
        jcm = UPSTREAM_JCM
    output: M7_HIST_CONFIG['signal']
    run:
        _m7_hist_config(input.hist_config, output[0],
                        [config['analysis_config']['dataset_location'], M7_DATASET], input.jcm)

rule M7_hist_config_mixeddata:
    input:
        hist_config = UPSTREAM_HIST_CONFIG,
        jcm = MIXED_JCM
    output: M7_HIST_CONFIG['mixeddata']
    run:
        _m7_hist_config(input.hist_config, output[0], [MIXED_URL], input.jcm,
                        mixed_data=True, subsample=MIX_SUBSAMPLE)

use rule analysis_processor from analysis as M7_hists_signal with:
    input:
        runner_script = "runner.py",
        config_file = M7_HIST_CONFIG['signal'],
        dataset = M7_DATASET,
        jcm = UPSTREAM_JCM
    output: f"{M7_OUT}singlefiles/hist__signal__{{year}}.coffea"
    log: f"{M7_OUT}logs/hists_signal__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_DATASETS + list(SIG_3B.values()) + list(SIG_4B.values())),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule analysis_processor from analysis as M7_hists_mixeddata with:
    input:
        runner_script = "runner.py",
        config_file = M7_HIST_CONFIG['mixeddata'],
        published = M2_PUBLISHED,
        jcm = MIXED_JCM
    output: f"{M7_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea"
    log: f"{M7_OUT}logs/hists_mixeddata__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = MIX_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as M7_merge_hists with:
    input:
        files = expand(f"{M7_OUT}singlefiles/hist__signal__{{year}}.coffea", year=YEARS)
                + expand(f"{M7_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: M7_HISTALL
    log: f"{M7_OUT}logs/merge_hists.log"
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

rule M7_plot_config_signal:
    """The four signal curves, no data or background: 4b signal, 3b signal x M.2 JCM, and the mixed
    signal from 3b (x the same JCM) and from 4b input events. No ratio panel."""
    output: f"{M7_OUT}plots_signal.yml"
    run:
        cfg = {'hists': {
                   'HH4b_4b': _plot_entry(SIG_DATASETS, 'fourTag', "4b signal", "#e42536"),
                   'HH4b_3b': _plot_entry(SIG_DATASETS, 'threeTag', "3b signal x JCM", "#e42536", "dashed"),
                   'HH4b_mixed3b': _plot_entry(SIG_3B.values(), 'fourTag', "mixed 3b signal x JCM", "#1f77b4", "dashed"),
                   'HH4b_mixed4b': _plot_entry(SIG_4B.values(), 'fourTag', "mixed 4b signal", "#1f77b4")},
               'doRatio': 0,
               'summary': _SUMMARY}
        write_yaml(output[0], cfg)

rule M7_plot_config_mixeddata:
    """mixeddata_all x the mixed-data JCM (the mixed background model) with the unmixed 4b signal and
    the mixed 3b signal x the M.2 JCM overlaid (x100, the scale of the analysis signal plots); the
    ratio panel is each (x100) signal / mixed data."""
    output: f"{M7_OUT}plots_mixeddata.yml"
    run:
        cfg = {'hists': {
                   'HH4b_4b': _plot_entry(SIG_DATASETS, 'fourTag', "4b signal (x100)", "#e42536", "solid", 100),
                   'HH4b_mixed3b': _plot_entry(SIG_3B.values(), 'fourTag', "mixed 3b signal x JCM (x100)",
                                               "#1f77b4", "dashed", 100)},
               'stack': {
                   'MixedData': {'process': MIX_NAME, 'tag': 'fourTag', 'fillcolor': "#FFDF7Fff",
                                 'edgecolor': 'k', 'label': "Mixed data x JCM"}},
               'ratios': {
                   'sig4bToMixed': {'numerator': {'type': 'hists', 'key': 'HH4b_4b'},
                                    'denominator': {'type': 'stack'},
                                    'uncertianty': 'nominal', 'color': "#e42536", 'marker': "s"},
                   'mixed3bToMixed': {'numerator': {'type': 'hists', 'key': 'HH4b_mixed3b'},
                                      'denominator': {'type': 'stack'},
                                      'uncertianty': 'nominal', 'color': "#1f77b4", 'marker': "o"}},
               'doRatio': 1,
               'summary': _SUMMARY}
        write_yaml(output[0], cfg)

use rule make_plots from analysis as M7_plots_signal with:
    input:
        coffea_file = M7_HISTALL,
        metadata_file = f"{M7_OUT}plots_signal.yml",
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{M7_OUT}plots_signal/plots_done.txt"
    params:
        output_dir = f"{M7_OUT}plots_signal/",
        metadata = f"{M7_OUT}plots_signal.yml",
        extra_arguments = f"-s xW -f png --year {PLOT_ERA}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{M7_OUT}logs/plots_signal.log"

use rule make_plots from analysis as M7_plots_mixeddata with:
    input:
        coffea_file = M7_HISTALL,
        metadata_file = f"{M7_OUT}plots_mixeddata.yml",
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{M7_OUT}plots_mixeddata/plots_done.txt"
    params:
        output_dir = f"{M7_OUT}plots_mixeddata/",
        metadata = f"{M7_OUT}plots_mixeddata.yml",
        extra_arguments = f"-s xW -f png --year {PLOT_ERA}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{M7_OUT}logs/plots_mixeddata.log"

rule M7_report:
    input:
        hists = M7_HISTALL,
        script = "coffea4bees/workflows/scripts/mixeddata_signal_check.py"
    output:
        txt = M7_REPORT,
        yml = f"{M7_OUT}signal_check.yml"
    log: f"{M7_OUT}logs/report.log"
    params:
        # original:mixed-3b:mixed-4b process names
        triples = " ".join(f"{d}:{SIG_3B[d]}:{SIG_4B[d]}" for d in SIG_DATASETS)
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.script} report \
            --hists {input.hists} --samples {params.triples} -o {M7_OUT} 2>&1 | tee {log}
        """

rule all_M7:
    input: [M7_REPORT] + [f"{M7_OUT}plots_{k}/plots_done.txt" for k in M7_PLOTS] if SIG_ENABLED else []

localrules: M7_config, M7_dataset_yml, M7_hist_config_signal, M7_hist_config_mixeddata, M7_merge_hists,
            M7_plot_config_signal, M7_plot_config_mixeddata, M7_plots_signal, M7_plots_mixeddata,
            M7_report, all_M7
