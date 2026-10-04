# coffea4bees/workflows/Snakefile_DeClustered_5_monitoring.smk
# D.5: monitoring of what D.1-D.4 produced, after MakeMixedData's M.6. Adds outputs only;
# nothing downstream reads them. Reuses D.4's merged histograms (no extra condor pass).
#
# One plot set (coffea4bees/plots/metadata/plotsDeClustered_validation.yml), all four-tag:
#   points = 4b data                 vs the stack
#   stack  = declustered multijet, seed k + ttbar MC
#   line   = ttbar pseudodata        vs the stack's ttbar MC part
# (subtract_ttbar: false: stack = declustered data incl. ttbar; ttbar MC a reference line, no
#  pseudodata)
#
#   D5_cutflow_page                  closure table (the shared cutflow_closure_table rule): data 4b vs
#                                    declustered seed k + ttbar MC, pseudodata vs ttbar MC
#   D5_plot_config + D5_plots        makePlots gallery (SR / SB, ratios)
#   D5_pdf_page                      index of D.2's PDFs + sampling-test plots, per era
#   D5_seed_study (library, >= 2 seeds)  seed overlaps (same library draw / identical event), single-
#                                    candidate lookups, effective number of independent replicas

import hashlib

D5_OUT = f"{out}D5/"
# "a random seed": drawn once per roast (stable across reruns), or pinned in the config
VAL_SEED = int(VAL['seed']) if VAL.get('seed') is not None else \
    int(hashlib.md5(config['roast_id'].encode()).hexdigest(), 16) % N_SEEDS
if VAL_SEED not in SEEDS:
    raise ValueError(f"validation.seed {VAL_SEED} is not one of the declustering seeds {SEEDS}")
SYN_PREFIX = "syn_noTT" if MJ_NAME.startswith("synthetic_data_noTT") else "syn"
SYN_PROCESS = f"{SYN_PREFIX}_v{VAL_SEED}"
# With several seeds the validation shows their MEAN (all seeds' histograms / cutflows summed, scaled
# by 1/n_seeds), as M.6 does for the 4b mixing; one seed otherwise.
SYN_ALL = [f"{SYN_PREFIX}_v{s}" for s in SEEDS]
SYN_TAG = f"mean of {N_SEEDS} seeds" if N_SEEDS > 1 else f"seed {VAL_SEED}"
D5_PLOT_CONFIG = f"{D5_OUT}plotsDeClustered_validation.yml"
PLOT_YEAR = "Run3" if any("202" in y for y in YEARS) else "RunII"

use rule cutflow_closure_table from analysis as D5_cutflow_page with:
    # the shared closure table (src/tools/cutflow_closure.py, as Phases B / C.4 / F), with the
    # declustered sample as the Multijet column and the ttbar pseudodata vs tt 4b MC
    input:
        cutflow_yml = f"{D4_OUT}cutflow_declustered.yml",
        validation_txt = f"{D4_OUT}cutflow_validation_declustered.txt"
    output:
        html = f"{D5_OUT}cutflow_monitoring.html",
        txt = f"{D5_OUT}cutflow_monitoring.txt"
    log: f"{D5_OUT}logs/cutflow_page.log"
    params:
        title = (f"{config.get('label', 'declustered')}_declustered_mean{N_SEEDS}seeds" if N_SEEDS > 1
                 else f"{config.get('label', 'declustered')}_declustered_seed{VAL_SEED}"),
        multijet = "sample4b",            # Multijet column = the four-tag sample --multijet-process
        ttbar = " ".join(TTBAR),
        extra_arguments = " ".join(["--multijet-process", *SYN_ALL, "--multijet-scale", repr(1.0 / N_SEEDS)]
                                   + (["--pseudodata", PS_NAME] if SUBTRACT_TT else [])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule D5_plot_config:
    input: VAL.get('plot_template', "coffea4bees/plots/metadata/plotsDeClustered_validation.yml")
    output: D5_PLOT_CONFIG
    run:
        with open(input[0]) as f:
            text = f.read()
        if "syn_vK" not in text:
            raise ValueError(f"{input[0]}: no `syn_vK` placeholder for the declustered sample")
        cfg = yaml.safe_load(text.replace("syn_vK", SYN_PROCESS).replace("seed K", SYN_TAG))
        stack = cfg.get('stack') or {}
        if N_SEEDS > 1 and 'MultiJet' in stack:
            stack['MultiJet'].update({'process': list(SYN_ALL), 'scalefactor': 1.0 / N_SEEDS})
        hists = cfg.setdefault('hists', {})
        if 'psdata' in hists:
            if SUBTRACT_TT:
                hists['psdata']['process'] = PS_NAME
            else:
                hists.pop('psdata')
                (cfg.get('ratios') or {}).pop('psdataToTTbar', None)
        if 'TTbar' in stack:
            stack['TTbar']['process'] = list(TTBAR)
            if not SUBTRACT_TT:
                # the declustered sample already models the ttbar: keep TT MC as a reference line
                tt = stack.pop('TTbar')
                cfg.setdefault('hists', {})['TTbar'] = {**tt, 'histtype': 'step',
                                                        'label': tt.get('label', 'ttbar MC') + " (in the declustered data)"}
                if 'MultiJet' in stack:
                    stack['MultiJet']['label'] = f"Declustered 4b data incl. $t\\bar{{t}}$ ({SYN_TAG})"
        write_yaml(output[0], cfg)

use rule make_plots from analysis as D5_plots with:
    input:
        coffea_file = D4_HISTALL,
        metadata_file = D5_PLOT_CONFIG,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{D5_OUT}plots/plots_done.txt"
    params:
        output_dir = f"{D5_OUT}plots/",
        metadata = D5_PLOT_CONFIG,
        extra_arguments = f"-s xW -f png --year {PLOT_YEAR}",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    log: f"{D5_OUT}logs/plots.log"

rule D5_pdf_page:
    input: D2_PDFS
    output: f"{D2_OUT}index.html"
    log: f"{D5_OUT}logs/pdf_page.log"
    params:
        era_dirs = lambda wildcards, input: " ".join(os.path.dirname(p) for p in input)
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/declustered_validation_report.py pdfs \
            {D2_OUT} {params.era_dirs} 2>&1 | tee {log}
        """

# Seed study (library method, >= 2 seeds): how often two seeds draw the same library splitting for
# the same event (Jet_lib_index), identical events, the single-candidate lookups (same in every seed),
# and the effective number of independent replicas per cut and per bin (D.4 cutflow / histograms:
# rho = 1 - Var_seeds / <N>, N_eff = n / (1 + (n-1) rho)). Reads only a few branches of the per-seed
# picoAODs, for validation.seed_study_years (default all). See declustered_seed_study.py.
SEED_STUDY = LIBRARY and N_SEEDS >= 2 and bool(VAL.get('seed_study', True))
if SEED_STUDY:
    rule D5_seed_study:
        input:
            registries = expand(f"{D3_OUT}per_seed/registry_seed{{seed}}.yml", seed=SEEDS),
            published = D3_PUBLISHED,
            cutflow = f"{D4_OUT}cutflow_declustered.yml",
            hists = D4_HISTALL
        output:
            index = f"{D5_OUT}seed_study/index.html",
            summary = f"{D5_OUT}seed_study/summary.yml"
        log: f"{D5_OUT}logs/seed_study.log"
        params:
            years = " ".join(VAL.get('seed_study_years') or YEARS)
        shell:
            """
            set -eo pipefail
            {EOS_PROXY}
            {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/declustered_seed_study.py \
                {input.registries} --outdir $(dirname {output.index}) --years {params.years} \
                --cutflow {input.cutflow} --hists {input.hists} --process-prefix {SYN_PREFIX} 2>&1 | tee {log}
            """

    localrules: D5_seed_study

# SvB Poisson bootstrap (library, >= 2 seeds, an upstream SvB model): the analysis SvB evaluated on
# the fly on every seed of the multijet sample (D.6's config recipe), the (run, lumi, event, SvB_MA.ps)
# of the four-tag SR events dumped (processor_HH4b dump_SvB_in_SR); then each INPUT event gets 30
# Poisson(1) weights shared by its declustered versions, and the toy-to-toy variance of the n-seed
# mean SvB histogram gives its statistical power: N_eff = <N>/Var (workflows/scripts/svb_bootstrap.py).
SVB_BOOT = LIBRARY and N_SEEDS >= 2 and bool(VAL.get('svb_bootstrap', True)) \
    and str(INPUTS.get('SvB_model') or "").startswith("root://")
SVB_BOOT_OUT = f"{D5_OUT}svb_bootstrap/"
if SVB_BOOT:
    rule D5_svb_config:
        input: UPSTREAM_HIST_CONFIG
        output: f"{SVB_BOOT_OUT}analysis_config_svb_dump.yml"
        run:
            _d6_hist_config(input[0], output[0], [MJ_URL])     # SvB on the fly, unblinded, no FvT/JCM
            with open(output[0]) as f:
                cfg = yaml.safe_load(f)
            cfg['config'].update({'dump_SvB_in_SR': True, 'fill_histograms': False})
            write_yaml(output[0], cfg)

    use rule analysis_processor from analysis as D5_svb_dump with:
        input:
            runner_script = "runner.py",
            config_file = f"{SVB_BOOT_OUT}analysis_config_svb_dump.yml",
            published = D3_PUBLISHED
        output: f"{SVB_BOOT_OUT}dumps/svb_dump__{{year}}.coffea"
        log: f"{SVB_BOOT_OUT}logs/dump__{{year}}.log"
        wildcard_constraints:
            year = "|".join(YEARS)
        params:
            datasets = MJ_NAME,
            years = lambda wildcards: wildcards.year,
            config = lambda wildcards, input: input.config_file,
            extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
            run_container_wrapper = WRAPPER,
            python_bin = PYTHON

    rule D5_svb_bootstrap:
        input: expand(f"{SVB_BOOT_OUT}dumps/svb_dump__{{year}}.coffea", year=YEARS)
        output:
            index = f"{SVB_BOOT_OUT}index.html",
            summary = f"{SVB_BOOT_OUT}summary.yml"
        log: f"{SVB_BOOT_OUT}logs/bootstrap.log"
        params:
            toys = int(VAL.get('svb_bootstrap_toys', 30)),
            title = f"{config.get('label', 'declustered')}: SvB_MA ps (SR, four-tag), {N_SEEDS} seeds"
        shell:
            """
            set -o pipefail
            {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/svb_bootstrap.py {input} \\
                --outdir {SVB_BOOT_OUT} --prefix {SYN_PREFIX} --toys {params.toys} \\
                --title "{params.title}" 2>&1 | tee {log}
            """

    localrules: D5_svb_config, D5_svb_bootstrap

rule all_D5:
    input:
        f"{D5_OUT}plots/plots_done.txt",
        f"{D5_OUT}cutflow_monitoring.html",
        [f"{D2_OUT}index.html"] if MAKE_PDFS else [],
        [f"{D5_OUT}seed_study/index.html"] if SEED_STUDY else [],
        [f"{SVB_BOOT_OUT}index.html"] if SVB_BOOT else []

localrules: D5_cutflow_page, D5_plot_config, D5_plots, D5_pdf_page, all_D5
