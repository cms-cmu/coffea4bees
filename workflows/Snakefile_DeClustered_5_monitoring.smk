# coffea4bees/workflows/Snakefile_DeClustered_5_monitoring.smk
# D.5: monitoring of what D.1-D.4 produced, after MakeMixedData's M.6. Adds outputs only;
# nothing downstream reads them. Reuses D.4's merged histograms (no extra condor pass).
#
# One plot set (coffea4bees/plots/metadata/plotsDeClustered_validation.yml), all four-tag:
#   points = 4b data                 vs the stack
#   stack  = declustered data, seed k (+ ttbar MC when the declustering subtracted ttbar;
#            otherwise ttbar MC is a reference line: the declustered sample already contains it)
#
#   D5_cutflow_page                  cutflow page: data 4b vs model (seed k and the mean over seeds,
#                                    seed-to-seed rms)
#   D5_plot_config + D5_plots        makePlots gallery (SR / SB, ratios)
#   D5_pdf_page                      index of D.2's PDFs + sampling-test plots, per era

import hashlib

D5_OUT = f"{out}D5/"
# "a random seed": drawn once per roast (stable across reruns), or pinned in the config
VAL_SEED = int(VAL['seed']) if VAL.get('seed') is not None else \
    int(hashlib.md5(config['roast_id'].encode()).hexdigest(), 16) % N_SEEDS
if VAL_SEED not in SEEDS:
    raise ValueError(f"validation.seed {VAL_SEED} is not one of the declustering seeds {SEEDS}")
SYN_PREFIX = "syn_noTT" if DATASET_NAME.startswith("synthetic_data_noTT") else "syn"
SYN_PROCESS = f"{SYN_PREFIX}_v{VAL_SEED}"
D5_PLOT_CONFIG = f"{D5_OUT}plotsDeClustered_validation.yml"
PLOT_YEAR = "Run3" if any("202" in y for y in YEARS) else "RunII"

rule D5_cutflow_page:
    input: f"{D4_OUT}cutflow_declustered.yml"
    output:
        html = f"{D5_OUT}cutflow_monitoring.html",
        txt = f"{D5_OUT}cutflow_monitoring.txt"
    log: f"{D5_OUT}logs/cutflow_page.log"
    params:
        ttbar_flag = "" if SUBTRACT_TT else "--ttbar-in-sample"
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/declustered_validation_report.py cutflow \
            {input} {D5_OUT} --prefix {SYN_PREFIX} --seed {VAL_SEED} --n-seeds {N_SEEDS} \
            {params.ttbar_flag} 2>&1 | tee {log}
        """

rule D5_plot_config:
    input: VAL.get('plot_template', "coffea4bees/plots/metadata/plotsDeClustered_validation.yml")
    output: D5_PLOT_CONFIG
    run:
        with open(input[0]) as f:
            text = f.read()
        if "syn_vK" not in text:
            raise ValueError(f"{input[0]}: no `syn_vK` placeholder for the declustered sample")
        cfg = yaml.safe_load(text.replace("syn_vK", SYN_PROCESS).replace("seed K", f"seed {VAL_SEED}"))
        stack = cfg.get('stack') or {}
        if 'TTbar' in stack:
            stack['TTbar']['process'] = list(TTBAR)
            if not SUBTRACT_TT:
                # the declustered sample already models the ttbar: keep TT MC as a reference line
                tt = stack.pop('TTbar')
                cfg.setdefault('hists', {})['TTbar'] = {**tt, 'histtype': 'step',
                                                        'label': tt.get('label', 'ttbar MC') + " (in the declustered data)"}
                if 'MultiJet' in stack:
                    stack['MultiJet']['label'] = f"Declustered 4b data incl. $t\\bar{{t}}$ (seed {VAL_SEED})"
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

rule all_D5:
    input:
        f"{D5_OUT}plots/plots_done.txt",
        f"{D5_OUT}cutflow_monitoring.html",
        [] if PDF_EXTERNAL else [f"{D2_OUT}index.html"]

localrules: D5_cutflow_page, D5_plot_config, D5_plots, D5_pdf_page, all_D5
