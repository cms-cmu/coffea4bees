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
#   M6_cutflow + M6_cutflow_page                  four-tag cutflow dump + HTML page
#   M6_plot_config + M6_plots                     makePlots gallery (SR / SB, ratios)
#   M6_study                                      study plots + subsample overlap matrix (from M.3)
#                                                 + study/index.html

import hashlib

M6_OUT = f"{out}M6/"
VAL = config.get('validation') or {}
# "a random subsample": drawn once per roast (stable across reruns), or pinned in the config
VAL_SUB = int(VAL['subsample']) if VAL.get('subsample') is not None else \
    int(hashlib.md5(config['roast_id'].encode()).hexdigest(), 16) % N_SUB
SUB_URL = f"{HANDOFF}/{SUB_NAME}.yml"
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
        jcm = MIXED_JCM
    output: f"{M6_OUT}analysis_config_mixedJCM.yml"
    run:
        _m6_hist_config(input.hist_config, output[0], [MIXED_URL], input.jcm)

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

rule M6_cutflow_page:
    input: f"{M6_OUT}cutflow_validation.yml"
    output:
        html = f"{M6_OUT}cutflow_validation.html",
        txt = f"{M6_OUT}cutflow_validation.txt"
    log: f"{M6_OUT}logs/cutflow_page.log"
    shell:
        """
        {WRAPPER} {PYTHON} coffea4bees/workflows/scripts/mixeddata_validation_report.py cutflow \
            {input} {M6_OUT} --subsample {VAL_SUB} 2>&1 | tee {log}
        """

rule M6_plot_config:
    input: VAL.get('plot_template', "coffea4bees/plots/metadata/plotsMixedData_validation.yml")
    output: M6_PLOT_CONFIG
    run:
        with open(input[0]) as f:
            text = f.read()
        if "mix_vK" not in text:
            raise ValueError(f"{input[0]}: no `mix_vK` placeholder for the subsample")
        text = text.replace("mix_vK", f"mix_v{VAL_SUB}").replace("subsample K", f"subsample v{VAL_SUB}")
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

rule all_M6:
    input:
        f"{M6_OUT}plots/plots_done.txt",
        f"{M6_OUT}cutflow_validation.html",
        f"{M6_OUT}study/index.html"

localrules: M6_config_mixed, M6_config_closure, M6_merge, M6_cutflow, M6_cutflow_page,
            M6_plot_config, M6_plots, M6_study, all_M6
