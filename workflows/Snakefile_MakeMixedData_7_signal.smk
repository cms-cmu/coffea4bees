# coffea4bees/workflows/Snakefile_MakeMixedData_7_signal.smk
# M.7: signal check. The data the mixed sample is made from contains some signal: the 3b events
# that are mixed, and the 4b events whose hemispheres fill the library. Mixing must break the
# HH correlations, since each hemisphere is replaced by a library hemisphere from a different
# event. Otherwise signal leaks into the mixed background model, and so into MvD and the SvB fit.
#
# Signal MC is mixed with M.2's settings and M.1's library. Every preselected event is mixed:
# 3b events get pseudo-tags from the upstream JCM, as in the data; 4b events keep their real
# tags. The mixed signal is then histogrammed with the nominal production's Phase F analysis
# config (inputs.phaseF_hists; SvB = inputs.SvB_model evaluated on the fly, the model behind F's
# SvB friend) and merged into F's histAll, so it is plotted next to the nominal F data,
# background and original signal with F's plot config plus a mixed-signal entry (make_plots +
# the standard gallery). The report gives the numbers: how much of the high-SvB tail and the
# Higgs-candidate peak survives.
#
#   M7_config                        skimmer config: M.2's, on signal MC (mix_tags threeTag_fourTag,
#                                    no ttbar subtraction, trigWeight friend, unit weights if absent)
#                                    picoAODs -> <PUB>/picoAOD/signal_mixed
#   M7_mix (per year, condor)        -> per-year registry
#   M7_dataset_yml                   -> synthetic_mc_<signal>: the mixed files + the original sample's
#                                       normalisation (sumw, xs, ...)
#   M7_fetch_F                       F's histAll + the analysis_config.yml next to it
#   M7_hist_config                   F's analysis config, pointed at the mixed signal: SvB and FvT
#                                    friends dropped, SvB_MA = inputs.SvB_model on the fly, unblinded
#   M7_hists (per year, condor)      processor_HH4b over the mixed signal
#   M7_merge_hists                   F's histAll + the mixed signal -> histAll_signal_check.coffea
#   M7_plot_config + M7_plots        F's plot config + HH4b mixed -> makePlots gallery (plots/)
#   M7_report                        -> signal_check.{txt,yml}
#
# signal_check: {enabled: false} turns it off; datasets: the signal samples; plot_config: the F
# plot config (default: plotsAll_ttbarWeights.yml, Phase F's).
# The name synthetic_mc_* gives isSyntheticMC in processor_config: MC weights with the b-tag SF
# stored at mixing time (CMSbtag), no JEC, no truth matching. It must not contain "mix", which
# would make it mixed data (unit weights).

SIG = config.get('signal_check') or {}
SIG_ENABLED = bool(SIG.get('enabled', True))
SIG_DATASETS = list(SIG.get('datasets', ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"]))
SIG_NAMES = {d: f"synthetic_mc_{d}" for d in SIG_DATASETS}
SVB_MODEL = INPUTS.get('SvB_model')
if SIG_ENABLED and not str(SVB_MODEL or "").startswith("root://"):
    raise ValueError("signal_check needs inputs.SvB_model: a root:// URL to an upstream roast's SvB result.json")

PHASEF_HISTS = INPUTS.get('phaseF_hists')
if SIG_ENABLED and not str(PHASEF_HISTS or "").startswith("root://"):
    raise ValueError("signal_check needs inputs.phaseF_hists: a root:// URL to the nominal roast's "
                     "Phase F histAll_<label>.coffea (its analysis_config.yml must sit next to it)")
SIG_PLOT_CONFIG = SIG.get('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")

M7_OUT = f"{out}M7/"
M7_CONFIG = f"{M7_OUT}make_mixed_signal.yml"
M7_DATASET = f"{M7_OUT}datasets/signal_mixed.yml"
M7_HIST_CONFIG = f"{M7_OUT}analysis_config_signal.yml"
M7_HISTALL = f"{M7_OUT}histAll_signal_check.coffea"
M7_REPORT = f"{M7_OUT}signal_check.txt"
M7_F_HISTS = f"{M7_OUT}inputs/{os.path.basename(str(PHASEF_HISTS))}"
M7_F_CONFIG = f"{M7_OUT}inputs/analysis_config_phaseF.yml"
M7_PLOT_CONFIG = f"{M7_OUT}plots_signal_check.yml"

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
            'mix_tags': 'threeTag_fourTag',       # all preselected signal, 3b and 4b
            'subtract_ttbar_with_weights': False, # signal MC: nothing to subtract
            'friends': {},
            'friends_include': ['trigWeight'],    # the GluGlu MC trigger weights ...
            'require_trigWeight': False,          # ... which Run 3 ggF lacks: unit weights (shape test)
        })
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
    """synthetic_mc_<signal>: per year, the mixed files of that sample and the ORIGINAL sample's
    normalisation. Mixing keeps each event's generator weight, but it drops the events that are
    not preselected and the unresolved hemisphere collisions. So the sumw is the original one,
    and the yields compare like with like."""
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
            --signals {SIG_DATASETS} --years {YEARS} --prefix synthetic_mc_ -o {output} 2>&1 | tee -a {log}
        """

rule M7_fetch_F:
    output:
        hists = M7_F_HISTS,
        config = M7_F_CONFIG
    log: f"{M7_OUT}logs/fetch_F.log"
    params:
        hists = PHASEF_HISTS,
        config = f"{os.path.dirname(str(PHASEF_HISTS))}/analysis_config.yml"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        xrdcp -f "{params.hists}" {output.hists} 2>&1 | tee {log}
        xrdcp -f "{params.config}" {output.config} 2>&1 | tee -a {log}
        """

rule M7_hist_config:
    input: M7_F_CONFIG
    output: M7_HIST_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        cfg['dataset_location'] = [config['analysis_config']['dataset_location'], M7_DATASET]
        cfg.get('runner', {}).pop('dataset_location', None)
        c = cfg.setdefault('config', {})
        # No SvB friend covers the mixed events: evaluate the analysis SvB on the fly (the model F's
        # SvB_MA friend was made from). A friend takes precedence over the classifier (load_SvB), so
        # drop them; FvT (a 3b-data weight) does not cover the mixed events either.
        c['friends'] = {k: v for k, v in (c.get('friends') or {}).items()
                        if not (k.startswith('SvB') or k.startswith('FvT'))}
        # F's JCM (a 3b-data weight) is installed only in the nominal roast's checkout; not needed.
        c.pop('JCM_file', None)
        c.update({'run_SvB': True, 'SvB': None,
                  'SvB_MA': [{'path': SVB_MODEL, 'name': 'Final'}],
                  'apply_FvT': False,
                  'apply_JCM': False,
                  # F's friends_include has FvT and SvB_MA, which runner.py would fill from
                  # friends_HH4b.yml (overriding the on-the-fly SvB); the MC needs only trigWeight
                  'friends_include': ['trigWeight'],
                  'require_trigWeight': False,    # Run 3 ggF has no trigger-weight friend
                  'blind': False})                # no data here; a blinded synthetic_mc lost its SR SvB tail
        if config['test']:
            cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M7_hists with:
    input:
        runner_script = "runner.py",
        config_file = M7_HIST_CONFIG,
        dataset = M7_DATASET
    output: f"{M7_OUT}singlefiles/hist__signal_mixed__{{year}}.coffea"
    log: f"{M7_OUT}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_NAMES.values()),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as M7_merge_hists with:
    input:
        files = [M7_F_HISTS] + expand(f"{M7_OUT}singlefiles/hist__signal_mixed__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: M7_HISTALL
    log: f"{M7_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

rule M7_plot_config:
    """F's plot config + the mixed signal, drawn like F's HH4b (same scale factor) but dashed."""
    input: SIG_PLOT_CONFIG
    output: M7_PLOT_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        hists = cfg.setdefault('hists', {})
        ref = dict(hists.get('HH4b') or {})
        scale = ref.get('scalefactor', 100)
        hists['HH4b_mixed'] = {'process': list(SIG_NAMES.values()), 'tag': 'fourTag',
                               'label': f"HH4b mixed (x{scale:g})", 'edgecolor': "#1f77b4",
                               'fillcolor': "#1f77b4", 'histtype': 'step', 'linestyle': 'dashed',
                               'scalefactor': scale}
        write_yaml(output[0], cfg)

use rule make_plots from analysis as M7_plots with:
    input:
        coffea_file = M7_HISTALL,
        metadata_file = M7_PLOT_CONFIG,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{M7_OUT}plots/plots_done.txt"
    params:
        output_dir = f"{M7_OUT}plots/",
        metadata = M7_PLOT_CONFIG,
        extra_arguments = f"-s xW -f png --year {'Run3' if any('202' in y for y in YEARS) else 'RunII'}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{M7_OUT}logs/plots.log"

rule M7_report:
    input:
        hists = M7_HISTALL,
        script = "coffea4bees/workflows/scripts/mixeddata_signal_check.py"
    output:
        txt = M7_REPORT,
        yml = f"{M7_OUT}signal_check.yml"
    log: f"{M7_OUT}logs/report.log"
    params:
        pairs = " ".join(f"{d}:{n}" for d, n in SIG_NAMES.items())
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.script} report \
            --hists {input.hists} --pairs {params.pairs} -o {M7_OUT} 2>&1 | tee {log}
        """

rule all_M7:
    input: [M7_REPORT, f"{M7_OUT}plots/plots_done.txt"] if SIG_ENABLED else []

localrules: M7_config, M7_dataset_yml, M7_fetch_F, M7_hist_config, M7_merge_hists, M7_plot_config,
            M7_plots, M7_report, all_M7
