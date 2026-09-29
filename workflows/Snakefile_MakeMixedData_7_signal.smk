# coffea4bees/workflows/Snakefile_MakeMixedData_7_signal.smk
# M.7: signal check. The data the mixed sample is made from contains some signal: the 3b events
# that are mixed, and the 4b events whose hemispheres fill the library. Mixing must break the
# HH correlations, since each hemisphere is replaced by a library hemisphere from a different
# event. Otherwise signal leaks into the mixed background model, and so into MvD and the SvB fit.
#
# Signal MC is mixed with M.2's settings and M.1's library. Every preselected event is mixed:
# 3b events get pseudo-tags from the upstream JCM, as in the data; 4b events keep their real
# tags. Both the original and the mixed signal are then histogrammed with M.3's config, with
# the analysis SvB (inputs.SvB_model) evaluated on the fly on both. The report shows how much
# of the high-SvB tail and the Higgs-candidate peak survives.
#
#   M7_config                        skimmer config: M.2's, on signal MC (mix_tags threeTag_fourTag,
#                                    no ttbar subtraction, trigWeight friend, unit weights if absent)
#                                    picoAODs -> <PUB>/picoAOD/signal_mixed
#   M7_mix (per year, condor)        -> per-year registry
#   M7_dataset_yml                   -> synthetic_mc_<signal>: the mixed files + the original sample's
#                                       normalisation (sumw, xs, ...)
#   M7_hist_config                   the upstream noJCM analysis config (as M.3), SvB friends
#                                    dropped, SvB_MA = inputs.SvB_model on the fly
#   M7_hists (per year, condor)      processor_HH4b over the original and the mixed signal
#   M7_merge_hists                   -> histAll_signal_check.coffea
#   M7_report                        -> signal_check.{txt,yml} + overlays (plots/, index.html)
#
# signal_check: {enabled: false} turns it off; datasets: the signal samples.
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

M7_OUT = f"{out}M7/"
M7_CONFIG = f"{M7_OUT}make_mixed_signal.yml"
M7_DATASET = f"{M7_OUT}datasets/signal_mixed.yml"
M7_HIST_CONFIG = f"{M7_OUT}analysis_config_signal.yml"
M7_HISTALL = f"{M7_OUT}histAll_signal_check.coffea"
M7_REPORT = f"{M7_OUT}signal_check.txt"

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

rule M7_hist_config:
    input: UPSTREAM_HIST_CONFIG
    output: M7_HIST_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
        if tight is not False:
            raise ValueError(f"upstream histogram config has fourTag_use_tight={tight!r}; need the non-tight selection")
        cfg['dataset_location'] = [config['analysis_config']['dataset_location'], M7_DATASET]
        cfg.get('runner', {}).pop('dataset_location', None)
        c = cfg.setdefault('config', {})
        # No SvB friend covers the mixed events: evaluate the analysis SvB on the fly, the same
        # model on both samples. A friend takes precedence over the classifier (load_SvB), so
        # drop them all.
        c['friends'] = {k: v for k, v in (c.get('friends') or {}).items() if not k.startswith('SvB')}
        c.update({'run_SvB': True, 'SvB': None,
                  'SvB_MA': [{'path': SVB_MODEL, 'name': 'Final'}],
                  'require_trigWeight': False})   # Run 3 ggF has no trigger-weight friend
        if config['test']:
            cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as M7_hists with:
    input:
        runner_script = "runner.py",
        config_file = M7_HIST_CONFIG,
        dataset = M7_DATASET
    output: f"{M7_OUT}singlefiles/hist__signal__{{year}}.coffea"
    log: f"{M7_OUT}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_DATASETS + list(SIG_NAMES.values())),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as M7_merge_hists with:
    input:
        files = expand(f"{M7_OUT}singlefiles/hist__signal__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: M7_HISTALL
    log: f"{M7_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

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
        export MPLCONFIGDIR="/tmp/matplotlib"; mkdir -p $MPLCONFIGDIR
        {WRAPPER} {PYTHON} {input.script} report \
            --hists {input.hists} --pairs {params.pairs} -o {M7_OUT} 2>&1 | tee {log}
        """

rule all_M7:
    input: [M7_REPORT] if SIG_ENABLED else []

localrules: M7_config, M7_dataset_yml, M7_hist_config, M7_merge_hists, M7_report, all_M7
