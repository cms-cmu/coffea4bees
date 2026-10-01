# coffea4bees/workflows/Snakefile_DeClustered_6_signal.smk
# D.6: signal check (as MakeMixedData M.7). The declustered sample is made from the 4b data, which
# contains some signal; declustering must wash out resonances -- each splitting is re-made from a
# real QCD splitting (library) or the PDFs, so a H->bb pair should no longer peak. Otherwise
# signal leaks into the synthetic background model.
#
# Signal MC is declustered with D.3's method, library / PDFs and seed, histogrammed with the nominal
# production's Phase F analysis config (inputs.phaseF_hists; SvB = inputs.SvB_model evaluated on the
# fly, the model behind F's SvB friend) and merged into F's histAll, so it is plotted next to the
# nominal F data, background and original signal with F's plot config plus a declustered-signal
# entry (make_plots + the standard gallery). The report (the M.7 tool, mixeddata_signal_check.py)
# gives the numbers: how much of the high-SvB tail and the Higgs-candidate peak survives.
#
#   D6_config                         skimmer config: D.3's, on signal MC (no ttbar subtraction,
#                                     trigWeight friend, unit weights if absent)
#                                     picoAODs -> <PUB>/picoAOD/signal_declustered
#   D6_decluster (per year, condor)   -> per-year registry
#   D6_dataset_yml                    -> synthetic_mc_<signal>: the declustered files + the original
#                                        sample's normalisation, scaled to the events declustered
#                                        when signal_check.max_chunks subsamples
#   D6_fetch_F                        F's histAll + the analysis_config.yml next to it
#   D6_hist_config                    F's analysis config, pointed at the declustered signal: SvB and
#                                     FvT friends dropped, SvB_MA = inputs.SvB_model on the fly, unblinded
#   D6_hists (per year, condor)       processor_HH4b over the declustered signal
#   D6_merge_hists                    F's histAll + the declustered signal -> histAll_signal_check.coffea
#   D6_plot_config + D6_plots         F's plot config + HH4b declustered -> makePlots gallery (plots/)
#   D6_report                         -> signal_check.{txt,yml}
#
# signal_check: {enabled: false} turns it off; datasets: the signal samples (default: the SM ggHH,
# the only Run 3 signal covering 2022 + 2023); plot_config: the F plot config (default
# plotsAll_ttbarWeights.yml, Phase F's). The name synthetic_mc_* gives isSyntheticMC in
# processor_config: MC weights, no JEC. It must not contain "mix" (-> mixed data, unit weights).

SIG = config.get('signal_check') or {}
SIG_ENABLED = bool(SIG.get('enabled', True))
SIG_DATASETS = list(SIG.get('datasets', ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"]))
SIG_SEED = int(SIG.get('seed', 0))
# max_chunks: decluster only this many chunks per sample-year (the check needs tens of thousands of
# 4b events, not the whole ~3M-event 4b sample, which costs as much CPU as all the data); None: all.
SIG_MAX_CHUNKS = SIG.get('max_chunks')
SIG_NAMES = {d: f"synthetic_mc_{d}" for d in SIG_DATASETS}
SVB_MODEL = INPUTS.get('SvB_model')
if SIG_ENABLED and not str(SVB_MODEL or "").startswith("root://"):
    raise ValueError("signal_check needs inputs.SvB_model: a root:// URL to an upstream roast's SvB result.json")
PHASEF_HISTS = INPUTS.get('phaseF_hists')
if SIG_ENABLED and not str(PHASEF_HISTS or "").startswith("root://"):
    raise ValueError("signal_check needs inputs.phaseF_hists: a root:// URL to the nominal roast's "
                     "Phase F histAll_<label>.coffea (its analysis_config.yml must sit next to it)")
SIG_PLOT_CONFIG = SIG.get('plot_config', "coffea4bees/plots/metadata/plotsAll_ttbarWeights.yml")

D6_OUT = f"{out}D6/"
D6_CONFIG = f"{D6_OUT}configs/declustering_signal.yml"
D6_DATASET = f"{D6_OUT}datasets/signal_declustered.yml"
D6_HIST_CONFIG = f"{D6_OUT}analysis_config_signal.yml"
D6_HISTALL = f"{D6_OUT}histAll_signal_check.coffea"
D6_REPORT = f"{D6_OUT}signal_check.txt"
D6_F_HISTS = f"{D6_OUT}inputs/{os.path.basename(str(PHASEF_HISTS))}"
D6_F_CONFIG = f"{D6_OUT}inputs/analysis_config_phaseF.yml"
D6_PLOT_CONFIG = f"{D6_OUT}plots_signal_check.yml"

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
    """synthetic_mc_<signal>: per year, the declustered files of that sample and the ORIGINAL
    sample's normalisation (the declustering keeps every event and its generator weight). With
    max_chunks only part of the sample is declustered: sumw / sumw2 are then scaled by the events
    processed over the sample's (the registry's total_events / the original count; the powheg
    weights are ~constant), so the yields still compare like with like."""
    input:
        registries = expand(f"{D6_OUT}per_year/picoaod_datasets__{{year}}.yml", year=YEARS),
        merge_script = "coffea4bees/workflows/scripts/merge_mixeddata_registries.py",
        script = "coffea4bees/workflows/scripts/mixeddata_signal_check.py"
    output: D6_DATASET
    log: f"{D6_OUT}logs/dataset_yml.log"
    params:
        full = f"{D6_OUT}datasets/signal_declustered_fullnorm.yml"
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.merge_script} {input.registries} {D6_OUT}datasets/registry.yml 2>&1 | tee {log}
        {WRAPPER} {PYTHON} {input.script} dataset \
            --registry {D6_OUT}datasets/registry.yml --metadata {config[analysis_config][dataset_location]} \
            --signals {SIG_DATASETS} --years {YEARS} --prefix synthetic_mc_ -o {params.full} 2>&1 | tee -a {log}
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/declustered_signal_norm.py \
            {params.full} {D6_OUT}datasets/registry.yml {output} 2>&1 | tee -a {log}
        """

rule D6_fetch_F:
    output:
        hists = D6_F_HISTS,
        config = D6_F_CONFIG
    log: f"{D6_OUT}logs/fetch_F.log"
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

rule D6_hist_config:
    input: D6_F_CONFIG
    output: D6_HIST_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        cfg['dataset_location'] = [config['analysis_config']['dataset_location'], D6_DATASET]
        cfg.get('runner', {}).pop('dataset_location', None)
        c = cfg.setdefault('config', {})
        # No SvB friend covers the declustered events: evaluate the analysis SvB on the fly (the model
        # F's SvB_MA friend was made from). A friend takes precedence over the classifier (load_SvB),
        # so drop them; FvT (a 3b-data weight) does not cover them either.
        c['friends'] = {k: v for k, v in (c.get('friends') or {}).items()
                        if not (k.startswith('SvB') or k.startswith('FvT'))}
        c.pop('JCM_file', None)          # F's JCM (3b-data weight) lives in the nominal checkout only
        c.update({'run_SvB': True, 'SvB': None,
                  'SvB_MA': [{'path': SVB_MODEL, 'name': 'Final'}],
                  'apply_FvT': False,
                  'apply_JCM': False,
                  'friends_include': ['trigWeight'],   # the MC needs only trigWeight (not F's FvT / SvB_MA)
                  'require_trigWeight': False,         # Run 3 ggF has no trigger-weight friend
                  'blind': False})
        if config['test']:
            cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
        write_yaml(output[0], cfg)

use rule analysis_processor from analysis as D6_hists with:
    input:
        runner_script = "runner.py",
        config_file = D6_HIST_CONFIG,
        dataset = D6_DATASET
    output: f"{D6_OUT}singlefiles/hist__signal_declustered__{{year}}.coffea"
    log: f"{D6_OUT}logs/hists__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = " ".join(SIG_NAMES.values()),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as D6_merge_hists with:
    input:
        files = [D6_F_HISTS] + expand(f"{D6_OUT}singlefiles/hist__signal_declustered__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: D6_HISTALL
    log: f"{D6_OUT}logs/merge_hists.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

rule D6_plot_config:
    """F's plot config + the declustered signal, drawn like F's HH4b (same scale factor) but dashed."""
    input: SIG_PLOT_CONFIG
    output: D6_PLOT_CONFIG
    run:
        with open(input[0]) as f:
            cfg = yaml.safe_load(f) or {}
        hists = cfg.setdefault('hists', {})
        ref = dict(hists.get('HH4b') or {})
        scale = ref.get('scalefactor', 100)
        hists['HH4b_declustered'] = {'process': list(SIG_NAMES.values()), 'tag': 'fourTag',
                                     'label': f"HH4b declustered (x{scale:g})", 'edgecolor': "#1f77b4",
                                     'fillcolor': "#1f77b4", 'histtype': 'step', 'linestyle': 'dashed',
                                     'scalefactor': scale}
        write_yaml(output[0], cfg)

use rule make_plots from analysis as D6_plots with:
    input:
        coffea_file = D6_HISTALL,
        metadata_file = D6_PLOT_CONFIG,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{D6_OUT}plots/plots_done.txt"
    params:
        output_dir = f"{D6_OUT}plots/",
        metadata = D6_PLOT_CONFIG,
        extra_arguments = f"-s xW -f png --year {'Run3' if any('202' in y for y in YEARS) else 'RunII'}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{D6_OUT}logs/plots.log"

rule D6_report:
    input:
        hists = D6_HISTALL,
        script = "coffea4bees/workflows/scripts/mixeddata_signal_check.py"
    output:
        txt = D6_REPORT,
        yml = f"{D6_OUT}signal_check.yml"
    log: f"{D6_OUT}logs/report.log"
    params:
        pairs = " ".join(f"{d}:{n}" for d, n in SIG_NAMES.items())
    shell:
        """
        set -eo pipefail
        {WRAPPER} {PYTHON} {input.script} report \
            --hists {input.hists} --pairs {params.pairs} -o {D6_OUT} 2>&1 | tee {log}
        """

rule all_D6:
    input: [D6_REPORT, f"{D6_OUT}plots/plots_done.txt"] if SIG_ENABLED else []

localrules: D6_config, D6_dataset_yml, D6_fetch_F, D6_hist_config, D6_merge_hists, D6_plot_config,
            D6_plots, D6_report, all_D6
