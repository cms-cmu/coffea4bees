# coffea4bees/workflows/Snakefile_MakeMixedData_6_validation.smk
# M.6: validation plots of what M.1-M.5 produced. Adds outputs only; nothing downstream reads them.
#
# One plot set, three comparisons (coffea4bees/plots/metadata/plotsMixedData_validation.yml):
#   stack  = multijet (mixeddata_all x mixed-data JCM from M.3) + ttbar MC
#   points = 4b data                                  -> vs the stack
#   line   = ttbar pseudodata (M.5)                   -> vs the stack's ttbar MC part
#   line   = one mixeddata_4b subsample k (M.4; its files include the ttbar pseudodata)
#                                                     -> vs the stack
#
#   M6_config_mixed / M6_hists_mixed (per year)   processor_HH4b over mixeddata_all, mixed JCM applied
#   M6_config_closure / M6_hists_closure (per yr) mixeddata_4b (--samples k) + ttbar_PSData, no JCM
#                                                 (both unit-weight); read from the EOS handoff
#   M6_merge                                      + the upstream data / ttbar MC histograms (fetched)
#   M6_cutflow + M6_cutflow_page                  four-tag cutflow dump + closure table (the shared
#                                                 cutflow_closure_table rule)
#   M6_plot_config + M6_plots                     makePlots gallery (SR / SB, ratios)
#   M6_study                                      study plots + subsample overlap matrix (from M.3)
#                                                 + study/index.html
#
# 4b mixing: multijet = mixeddata_all_4bmix (all N seeds, unit weight) scaled by 1/N in the plots and
# the cutflow table; the subsample line is one seed (mix_4bmix_v<k>, + ttbar pseudodata). M6_study
# reads the per-seed picoAODs: seed-overlap matrix (same replacement hemispheres), rank / distance
# distributions, and the self-match check (a mixed hemisphere from its own event must never occur).

import hashlib

M6_OUT = f"{out}M6/"
VAL = config.get('validation') or {}
# "a random subsample": drawn once per roast (stable across reruns), or pinned in the config
VAL_SUB = int(VAL['subsample']) if VAL.get('subsample') is not None else \
    int(hashlib.md5(config['roast_id'].encode()).hexdigest(), 16) % N_SUB
SUB_URL = f"{HANDOFF}/{SUB_NAME}.yml"
SVB_SUB_URL = str(SUB_EXTERNAL) if SUB_EXTERNAL else SUB_URL     # what the SvB bootstrap reads
PS_URL = f"{HANDOFF}/{PS_NAME}.yml"
M6_HISTALL = f"{M6_OUT}histAll_validation.coffea"
M6_PLOT_CONFIG = f"{M6_OUT}plotsMixedData_validation.yml"
PLOT_YEAR = "Run3" if any("202" in y for y in YEARS) else "RunII"

def _m6_hist_config(src, dst, datasets_urls, jcm_file):
    with open(src) as f:
        cfg = yaml.safe_load(f) or {}
    tight = (cfg.get('config') or {}).get('fourTag_use_tight', False)
    if tight is not False:
        raise ValueError(f"upstream histogram config has fourTag_use_tight={tight!r}; need the non-tight selection")
    cfg['dataset_location'] = list(datasets_urls)
    cfg.get('runner', {}).pop('dataset_location', None)
    cfg.setdefault('config', {})
    if jcm_file:
        cfg['config']['apply_JCM'] = True
        cfg['config']['JCM_file'] = jcm_file
        # The JCM-on-4b-mixed-events weight lives only in add_pseudotagweights' apply_MvD branch
        # (event_weights.py); with apply_MvD_weight false no MvD friend is loaded
        # (processor_HH4b load_MvD). Without it the mixed events keep unit weight (~10x data).
        cfg['config']['apply_MvD'] = True
        cfg['config']['apply_MvD_weight'] = False
    else:
        cfg['config']['apply_JCM'] = False      # subsamples + pseudodata are already unit-weight
        cfg['config'].pop('JCM_file', None)
    if config['test']:
        cfg.setdefault('runner', {}).update({'condor': False, 'shared_dask': False})
    write_yaml(dst, cfg)

rule M6_config_mixed:
    input:
        hist_config = UPSTREAM_HIST_CONFIG,
        jcm = [] if MIX4B else MIXED_JCM
    output: f"{M6_OUT}analysis_config_mixedJCM.yml"
    run:
        # 4b mixing: unit-weight four-tag events, no JCM
        _m6_hist_config(input.hist_config, output[0], [MIXED_URL], None if MIX4B else input.jcm)

rule M6_config_closure:
    input: UPSTREAM_HIST_CONFIG
    output: f"{M6_OUT}analysis_config_closure.yml"
    run:
        _m6_hist_config(input[0], output[0], [SUB_URL, PS_URL], None)

use rule analysis_processor from analysis as M6_hists_mixed with:
    input:
        runner_script = "runner.py",
        config_file = f"{M6_OUT}analysis_config_mixedJCM.yml",
        published = M2_PUBLISHED
    output: f"{M6_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea"
    log: f"{M6_OUT}logs/hists_mixed__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = MIX_NAME,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule analysis_processor from analysis as M6_hists_closure with:
    input:
        runner_script = "runner.py",
        config_file = f"{M6_OUT}analysis_config_closure.yml",
        subsamples = M4_PUBLISHED,
        psdata = M5_PUBLISHED
    output: f"{M6_OUT}singlefiles/hist__closure_v{VAL_SUB}__{{year}}.coffea"
    log: f"{M6_OUT}logs/hists_closure__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    params:
        datasets = f"{SUB_NAME} {PS_NAME}",
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [f"--samples {VAL_SUB}", TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as M6_merge with:
    input:
        files = [UPSTREAM_HISTS]
                + expand(f"{M6_OUT}singlefiles/hist__{MIX_NAME}__{{year}}.coffea", year=YEARS)
                + expand(f"{M6_OUT}singlefiles/hist__closure_v{VAL_SUB}__{{year}}.coffea", year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: M6_HISTALL
    log: f"{M6_OUT}logs/merge.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

use rule check_cutflow from analysis as M6_cutflow with:
    input:
        coffea_file = M6_HISTALL
    output:
        validation_txt = f"{M6_OUT}cutflow_validation_dump.txt",
        cutflow_yml = f"{M6_OUT}cutflow_validation.yml"
    log: f"{M6_OUT}logs/cutflow.log"
    params:
        known_flag = '--known-cutflow "none"',
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

use rule cutflow_closure_table from analysis as M6_cutflow_page with:
    # the shared closure table (src/tools/cutflow_closure.py, as Phases B / C.4 / F and DeClustered
    # D.5): Multijet = mixeddata_all x mixed JCM as is, ttbar pseudodata vs tt 4b MC, and subsample
    # k (mixed + ttbar pseudodata) vs Bkg
    input:
        cutflow_yml = f"{M6_OUT}cutflow_validation.yml",
        validation_txt = f"{M6_OUT}cutflow_validation_dump.txt"
    output:
        html = f"{M6_OUT}cutflow_validation.html",
        txt = f"{M6_OUT}cutflow_validation.txt"
    log: f"{M6_OUT}logs/cutflow_page.log"
    params:
        title = f"{config.get('label', 'mixeddata')}_validation_v{VAL_SUB}",
        multijet = "sample4b",            # Multijet column = the four-tag sample --multijet-process
        ttbar = " ".join(TTBAR),
        extra_arguments = (f"--multijet-process {MIX_NAME} --pseudodata {PS_NAME} --compare {SUB_PREFIX}_v{VAL_SUB}"
                           + (f" --multijet-scale {1.0 / N_SUB!r}" if MIX4B else "")),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule M6_plot_config:
    input: VAL.get('plot_template', "coffea4bees/plots/metadata/plotsMixedData_validation.yml")
    output: M6_PLOT_CONFIG
    run:
        with open(input[0]) as f:
            text = f.read()
        if "mix_vK" not in text:
            raise ValueError(f"{input[0]}: no `mix_vK` placeholder for the subsample")
        text = text.replace("mix_vK", f"{SUB_PREFIX}_v{VAL_SUB}").replace("subsample K", f"subsample v{VAL_SUB}")
        if MIX4B:
            # the multijet is the union of the N seeds: 1/N of it is one sample's worth
            plot_cfg = yaml.safe_load(text)
            mj = plot_cfg['stack']['MultiJet']
            mj.update({'process': MIX_NAME, 'scalefactor': 1.0 / N_SUB,
                       'label': f"Mixed 4b data (mean of {N_SUB} seeds)"})
            plot_cfg['hists']['subsample']['label'] = (f"{SUB_NAME} seed v{VAL_SUB} "
                                                       "(mixed + $t\\bar{t}$ pseudodata)")
            text = yaml.dump(plot_cfg, default_flow_style=False, sort_keys=False)
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        with open(output[0], "w") as f:
            f.write(text)

use rule make_plots from analysis as M6_plots with:
    input:
        coffea_file = M6_HISTALL,
        metadata_file = M6_PLOT_CONFIG,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{M6_OUT}plots/plots_done.txt"
    params:
        output_dir = f"{M6_OUT}plots/",
        metadata = M6_PLOT_CONFIG,
        extra_arguments = f"-s xW -f png --year {PLOT_YEAR}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{M6_OUT}logs/plots.log"

if not MIX4B:
    rule M6_study:
        input: M3_STUDY
        output:
            matrix = f"{M6_OUT}study/subsample_overlap_matrix.png",
            summary = f"{M6_OUT}study/summary.yml",
            index = f"{M6_OUT}study/index.html"
        log: f"{M6_OUT}logs/study.log"
        shell:
            """
            {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/mixeddata_validation_report.py study \
                {input} {M6_OUT}study --n-subsamples {N_SUB} 2>&1 | tee {log}
            """
else:
    rule M6_study:
        input:
            registries = M2_SEED_REGISTRIES,
            published = M2_PUBLISHED
        output:
            matrix = f"{M6_OUT}study/seed_overlap_matrix.png",
            summary = f"{M6_OUT}study/summary.yml",
            index = f"{M6_OUT}study/index.html"
        log: f"{M6_OUT}logs/study.log"
        params:
            years = " ".join(VAL.get('seed_study_years') or YEARS)
        shell:
            """
            set -eo pipefail
            {EOS_PROXY}
            {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/mixeddata_validation_report.py seeds \
                {input.registries} --outdir {M6_OUT}study --years {params.years} 2>&1 | tee {log}
            """

# SvB Poisson bootstrap of the N subsamples' average (as DeClustered D5_svb_*): the analysis SvB on
# the fly on every sample of the M.4 dataset (<SUB_PREFIX>_v<k>, unit weight), the four-tag SR events'
# (run, lumi, event, SvB_MA.ps) + the two library hemispheres each was mixed from dumped; then one
# Poisson(1) weight per source hemisphere, (w1-1)(w2-1)+1 per mixed event (John's 2025 recipe), and
# the toy-to-toy variance of the N-sample mean SvB histogram -> N_eff (workflows/scripts/svb_bootstrap.py
# --weights hemi). The ttbar pseudodata in each sample (run == 1) is left out.
MIX_SVB_BOOT = bool(VAL.get('svb_bootstrap', True)) and str(INPUTS.get('SvB_model') or "").startswith("root://")
M6_SVB_OUT = f"{M6_OUT}svb_bootstrap/"
if MIX_SVB_BOOT:
    rule M6_svb_config:
        input: UPSTREAM_HIST_CONFIG
        output: f"{M6_SVB_OUT}analysis_config_svb_dump.yml"
        run:
            _m7_hist_config(input[0], output[0], [SVB_SUB_URL], None)   # SvB on the fly, unblinded, unit weight
            with open(output[0]) as f:
                cfg = yaml.safe_load(f)
            cfg['config'].update({'dump_SvB_in_SR': True, 'dump_hemi_sources': True, 'fill_histograms': False})
            write_yaml(output[0], cfg)

    use rule analysis_processor from analysis as M6_svb_dump with:
        input:
            runner_script = "runner.py",
            config_file = f"{M6_SVB_OUT}analysis_config_svb_dump.yml",
            # this roast's M.4 dataset (+ the M.5 pseudodata its files include), or another roast's
            published = [] if SUB_EXTERNAL else [M4_PUBLISHED, M5_PUBLISHED]
        output: f"{M6_SVB_OUT}dumps/svb_dump__{{year}}.coffea"
        log: f"{M6_SVB_OUT}logs/dump__{{year}}.log"
        wildcard_constraints:
            year = "|".join(YEARS)
        params:
            datasets = SUB_NAME,
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
            run_container_wrapper = WRAPPER,
            python_bin = PYTHON

    rule M6_svb_bootstrap:
        input: expand(f"{M6_SVB_OUT}dumps/svb_dump__{{year}}.coffea", year=YEARS)
        output:
            index = f"{M6_SVB_OUT}index.html",
            summary = f"{M6_SVB_OUT}summary.yml"
        log: f"{M6_SVB_OUT}logs/bootstrap.log"
        params:
            toys = int(VAL.get('svb_bootstrap_toys', 30)),
            # one token: run_container re-quotes the arguments
            title = f"{config.get('label', 'mixeddata')}_SvB_MA_ps_SR_fourTag_{N_SUB}samples"
        shell:
            """
            set -o pipefail
            {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/svb_bootstrap.py {input} \\
                --outdir {M6_SVB_OUT} --prefix {SUB_PREFIX} --weights hemi --toys {params.toys} \\
                --title {params.title} 2>&1 | tee {log}
            """

    rule all_M6_svb:
        input: f"{M6_SVB_OUT}index.html"

    localrules: M6_svb_config, M6_svb_bootstrap, all_M6_svb

rule all_M6:
    input:
        f"{M6_OUT}plots/plots_done.txt",
        f"{M6_OUT}cutflow_validation.html",
        f"{M6_OUT}study/index.html",
        [f"{M6_SVB_OUT}index.html"] if MIX_SVB_BOOT else []

localrules: M6_config_mixed, M6_config_closure, M6_merge, M6_cutflow, M6_cutflow_page,
            M6_plot_config, M6_plots, M6_study, all_M6
