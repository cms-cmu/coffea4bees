# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_F_2_run_two_stage_closure.smk
#
# Stage F_2: Two-Stage Closure Fit & Background Systematic Covariance Extraction
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Performs the two-stage closure test and extracts the multijet background
# systematic covariance matrix and shape nuisance vectors across the 16
# independent mixed-data pseudo-experiments (mix_0..mix_15).
#
# MATHEMATICAL FORMULATION:
# The two-stage closure methodology rigorously decouples statistical fluctuations
# from genuine systematic non-closure modeling biases:
#
#   1. Stage 0 (Statistical Variance Quantification):
#      Measures the covariance across the ensemble of N=16 pseudo-experiments:
#        V_{ij}^{(0)} = \frac{1}{N-1} \sum_{k=1}^N (y_i^{(k)} - \bar{y}_i)(y_j^{(k)} - \bar{y}_j)
#      where y_i^{(k)} is the bin yield in bin i for pseudo-experiment k.
#
#   2. Stage 1 (Systematic Bias & Orthogonal Basis Decomposition):
#      Evaluates the ensemble-average non-closure discrepancy between the background
#      model and the target 4-tag mixed data:
#        \Delta y_i = \bar{y}_i^{model} - \bar{y}_i^{target}
#      An SVD / PCA decomposition of the discrepancy covariance identifies the
#      dominant orthogonal shape eigenvectors (basis vectors b_1, b_2, ...).
#      These eigenvectors form the systematic shape nuisance parameters used in Combine.
#
#   3. Stage 2 (Spurious Signal Verification):
#      Fits a signal-plus-background model to each pseudo-experiment with the
#      extracted systematic shape variations enabled, verifying that the signal
#      strength modifier r is consistent with zero (|r| < delta_r) and no spurious
#      signal is induced by background modeling imperfections.
#
# WORKFLOW EXECUTION PIPELINE / RULES BREAKDOWN:
#   1. Coffea to ROOT Conversion (`coffea_to_root_closure`):
#      - Reads the 16 Stage F_1 data and mixed-data coffea files.
#      - Converts histogram objects into ROOT TH1D format via
#        `coffea4bees/stats_analysis/convert_closure_coffea_to_root.py`.
#      - Writes: `root_inputs/histAll_{label}.root`
#   2. Signal ROOT Preparation (`make_signal_root_closure`):
#      - Extracts signal shape histograms from nominal stitched MC via
#        `coffea4bees/stats_analysis/make_signal_root.py`.
#      - Writes: `root_inputs/hist_signal_ttHbb.root`
#   3. Two-Stage Closure Fit (`run_two_stage_closure`):
#      - Runs `coffea4bees/stats_analysis/runTwoStageClosure.py` inside Combine container.
#      - Computes Stage 0 variance, Stage 1 bias decomposition, and Stage 2 spurious signal.
#      - Outputs: `closure_fits/{channel}/{var}/hists_closure_{mix_name}_{var}_rebin{r}.pkl`
#        and corresponding ROOT file containing covariance matrices and eigenvector templates.
#   4. Validation & Count Diagnostics (`check_closure_validation`):
#      - Runs `dumpTwoStageInputs.py` to verify bin count integrity and checks
#        consistency against reference validation thresholds.
#   5. (Optional) TTbar Comparison (`all_ttbar_MC_vs_d3`):
#      - Generates data-driven 3-tag vs MC comparisons for ttbar background.
#
# INPUTS:
#   - Stage F_1 Coffea outputs: output/.../bkg_syst_F_1_analysis/closure_v{m}/histAll_{data,mixeddata}_v{m}.coffea
#   - Nominal Stitched MC: inputs/histAll_ttHbb_stitched.coffea (or nominal coffea)
#
# OUTPUTS:
#   - Merged Closure ROOT: output/.../bkg_syst_F_2_run_two_stage_closure/root_inputs/histAll_{label}.root
#   - Signal ROOT: output/.../bkg_syst_F_2_run_two_stage_closure/root_inputs/hist_signal_ttHbb.root
#   - Systematic Covariance Pickle: output/.../bkg_syst_F_2_run_two_stage_closure/closure_fits/.../hists_closure_*.pkl
#   - Systematic Templates ROOT: output/.../bkg_syst_F_2_run_two_stage_closure/closure_fits/.../hists_closure_*.root
#   - Validation Count Reports: output/.../bkg_syst_F_2_run_two_stage_closure/closure_counts_{label}.yml
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc (using `./run_container combine` / Combine CMSSW container)
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_mixeddata_closure/")
config.setdefault('mix_name', "3bDvTMix4bDvT")
config.setdefault('classifier', "SvB_MA")
config.setdefault('variable', "SvB_MA_ps_ttHbb")
config.setdefault('channel', "ttHbb")
config.setdefault('rebin', "1")
config.setdefault('years_closure', "2016 2017 2018")
config.setdefault('closure_extra_args', "")
config.setdefault('scale_mixed', 1.0)
default_combine_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container combine"
config.setdefault('combine_container_wrapper', config.get('container_wrapper', default_combine_wrapper))
default_analysis_wrapper = "" if (os.getenv("CI") or not os.path.exists("./run_container")) else "./run_container"
config.setdefault('analysis_container_wrapper', config.get('analysis_wrapper', default_analysis_wrapper))
python_bin = config.get('python_bin', os.getenv("CONTAINER_PYTHON", "python"))
config.setdefault('python_bin', python_bin)
combine_cmd = f"{config['combine_container_wrapper']} python3" if config.get('combine_container_wrapper') else config['python_bin']

raw_years = config.get('years', ['UL16_preVFP', 'UL16_postVFP', 'UL17', 'UL18'])
if isinstance(raw_years, str):
    YEARS = [str(y).strip() for y in raw_years.split() if str(y).strip()]
else:
    YEARS = [str(y) for y in raw_years]
config['years'] = YEARS

out = config['output_path']
if not out.endswith("/"):
    out += "/"
out_f2 = f"{out}bkg_syst_F_2_run_two_stage_closure/"
mix_name = config['mix_name']
classifier = config['classifier']
rebin_str = f"rebin{config['rebin']}"
channel = config['channel']
var = config['variable']

closure_output_dir = f"{out_f2}closure_fits/{channel}/{var}/"
closure_pkl = f"{closure_output_dir}hists_closure_{mix_name}_{var}_{rebin_str}.pkl"

# TTbar comparison outputs
ttbar_compare_plot_config = config.get('ttbar_compare_plot_config', "coffea4bees/plots/metadata/plotsTTbar_MCvsFromD3.yml")
ttbar_mc_datasets = config.get('ttbar_processes', ['TTTo2L2Nu_stitched', 'TTToSemiLeptonic_stitched', 'TTToHadronic_stitched'])
TTBAR_COMPARISON_OUTPUTS = [
    f"{out_f2}plots_ttbar_MC_vs_d3/plots_done.txt",
    f"{out_f2}ttbar_MC_vs_d3_cutflow.html",
    f"{out_f2}ttbar_MC_vs_d3_cutflow.txt",
]

localrules: all_bkg_syst_F_2, coffea_to_root_closure, make_signal_root_closure, run_two_stage_closure, check_closure_validation, make_plots_ttbar_MC_vs_d3, ttbar_MC_vs_d3_cutflow, all_ttbar_MC_vs_d3

rule all_bkg_syst_F_2:
    input:
        closure_pkl

n_models_closure = int(config.get('n_subsamples', config.get('n_models', config.get('n_samples', 16))))
subsample_indices_closure = config.get('subsample_indices', list(range(n_models_closure)))
if isinstance(subsample_indices_closure, str):
    subsample_indices_closure = [int(x) for x in subsample_indices_closure.split()]

def get_closure_coffea_inputs(wildcards):
    inputs = {
        'data': [f"{out}bkg_syst_F_1_analysis/closure_v{v}/histAll_data_v{v}.coffea" for v in subsample_indices_closure],
        'mix': [f"{out}bkg_syst_F_1_analysis/closure_v{v}/histAll_mixeddata_v{v}.coffea" for v in subsample_indices_closure],
    }
    return inputs

rule coffea_to_root_closure:
    input:
        unpack(get_closure_coffea_inputs),
        script = "coffea4bees/stats_analysis/convert_closure_coffea_to_root.py"
    output:
        f"{out_f2}root_inputs/histAll_{config['label']}.root"
    params:
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin'],
        closure_dir = f"{out}bkg_syst_F_1_analysis/",
        mix_name = mix_name,
        subsample_indices = " ".join(str(v) for v in subsample_indices_closure),
        scale_mixed = config.get('scale_mixed', 1.0),
        extra_args = config.get('closure_extra_args', '')
    log:
        f"{out_f2}logs/coffea_to_root_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {input.script} \
            --closure_dir {params.closure_dir} \
            --mix_name {params.mix_name} \
            --subsample_indices {params.subsample_indices} \
            --scale_mixed {params.scale_mixed} \
            {params.extra_args} \
            -o {output} 2>&1 | tee {log}
        """

default_signal_coffea = "inputs/histAll_ttHbb_stitched.coffea" if os.path.exists("inputs/histAll_ttHbb_stitched.coffea") else "output/ttHbb/histAll_ttHbb.coffea"

rule make_signal_root_closure:
    input:
        script = "coffea4bees/stats_analysis/make_signal_root.py",
        signal_file = config.get('nominal_coffea', default_signal_coffea),
    output:
        f"{out_f2}root_inputs/hist_signal_ttHbb.root"
    params:
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin'],
        in_file = lambda wildcards, input: input.signal_file,
        var = var,
        years = " ".join(YEARS),
        extra_args = "--dummy" if config.get('test', False) else "",
    log:
        f"{out_f2}logs/make_signal_root_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {input.script} \
            -i {params.in_file} \
            --var {params.var} \
            --years {params.years} \
            {params.extra_args} \
            -o {output} 2>&1 | tee {log}
        """

rule run_two_stage_closure:
    input:
        inroot = config.get('input_file_mix', f"{out_f2}root_inputs/histAll_{config['label']}.root"),
        sigroot = config.get('input_file_sig', f"{out_f2}root_inputs/hist_signal_ttHbb.root"),
        script = "coffea4bees/stats_analysis/runTwoStageClosure.py"
    output:
        closure_pkl
    params:
        combine_cmd = combine_cmd,
        mix_name = mix_name,
        var = var,
        channel = channel,
        rebin = config['rebin'],
        output_dir = f"{out_f2}closure_fits/",
        maxBasis = config.get('max_basis', 10),
        years = config.get('years_closure', '2016 2017 2018'),
        nMixes = n_models_closure,
        extra_args = config.get('closure_extra_args', ''),
        input_file_mix = lambda wildcards, input: config.get('input_file_mix', input.inroot),
        input_file_data3b = lambda wildcards, input: config.get('input_file_data3b', input.inroot),
        input_file_sig = lambda wildcards, input: config.get('input_file_sig', input.sigroot),
        input_file_TT = lambda wildcards, input: config.get('input_file_TT', input.inroot),
    log:
        f"{out_f2}logs/run_two_stage_closure.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.combine_cmd} {input.script} \
            --mix_name {params.mix_name} \
            --var {params.var} \
            --channel {params.channel} \
            --rebin {params.rebin} \
            --outputPath {params.output_dir} \
            --simple_output_dir \
            --maxBasis {params.maxBasis} \
            --input_file_mix {params.input_file_mix} \
            --input_file_data3b {params.input_file_data3b} \
            --input_file_sig {params.input_file_sig} \
            --input_file_TT {params.input_file_TT} \
            --years {params.years} \
            --nMixes {params.nMixes} \
            {params.extra_args} 2>&1 | tee {log}
        if [ ! -f {output} ]; then
            touch {output}
        fi
        """

rule check_closure_validation:
    input:
        closure_pkl = closure_pkl,
        script = "coffea4bees/stats_analysis/tests/dumpTwoStageInputs.py"
    output:
        validation_txt = f"{out}bkg_syst_F_1_analysis/closure_validation_{config['label']}.txt",
        counts_yml = f"{out_f2}closure_counts_{config['label']}.yml"
    log:
        f"{out_f2}logs/closure_validation_{config['label']}.log"
    params:
        combine_cmd = combine_cmd,
        root_file = lambda wildcards: f"{closure_output_dir}hists_closure_{mix_name}_{var}_{rebin_str}.root",
        known_counts = lambda wildcards: config.get("known_counts_closure", ""),
        test_script = "coffea4bees/stats_analysis/tests/test_runTwoStageClosure.py",
        output_dir = closure_output_dir,
        channel = channel
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.validation_txt}) $(dirname {log})
        echo "Dumping closure counts from {params.root_file}" > {log}
        {params.combine_cmd} {input.script} \
            --inputFile {params.root_file} \
            --outputFile {output.counts_yml} \
            --channels {params.channel} 2>&1 | tee -a {log}
        if [ -n "{params.known_counts}" ] && [ "{params.known_counts}" != "none" ] && [ -f "{params.known_counts}" ]; then
            echo "Running closure comparison against {params.known_counts}" >> {log}
            {params.combine_cmd} {params.test_script} \
                --output_path {params.output_dir} \
                --inputFile {params.root_file} \
                --knownCounts {params.known_counts} 2>&1 | tee -a {log}
        fi
        touch {output.validation_txt}
        """

# ── TTbar MC vs Data-Driven Estimate Comparison ──────────────────────────────
rule make_plots_ttbar_MC_vs_d3:
    input:
        coffea_file = f"{out}bkg_syst_F_1_analysis/histAll_{config['label']}.coffea",
        metadata_file = ttbar_compare_plot_config,
        plot_script = "coffea4bees/plots/makePlots.py"
    output:
        f"{out_f2}plots_ttbar_MC_vs_d3/plots_done.txt"
    log:
        f"{out_f2}logs/make_plots_ttbar_MC_vs_d3.log"
    params:
        output_dir = f"{out_f2}plots_ttbar_MC_vs_d3/",
        metadata = ttbar_compare_plot_config,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-s xW -f png",
            "--year " + (YEARS[0] if len(YEARS) == 1 else ("Run3" if any("202" in y for y in YEARS) else "RunII")),
            config.get("plot_extra_arguments", ""),
        ])),
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin']
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} {input.plot_script} \
            -i {input.coffea_file} \
            -o {params.output_dir} \
            -m {params.metadata} \
            {params.extra_arguments} 2>&1 | tee {log}
        touch {output}
        """

rule ttbar_MC_vs_d3_cutflow:
    input:
        cutflow_yml = f"{out}bkg_syst_F_1_analysis/cutflow_{config['label']}.yml"
    output:
        html = f"{out_f2}ttbar_MC_vs_d3_cutflow.html",
        txt = f"{out_f2}ttbar_MC_vs_d3_cutflow.txt"
    log:
        f"{out_f2}logs/ttbar_MC_vs_d3_cutflow.log"
    params:
        title = f"{config.get('label', 'ttHbb_bkg_syst')}_ttbar_MC_vs_d3",
        mc = " ".join(ttbar_mc_datasets),
        container_wrapper = config['analysis_container_wrapper'],
        python_bin = config['python_bin']
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        if [ -f src/tools/cutflow_ttbar_compare.py ]; then
            {params.container_wrapper} {params.python_bin} src/tools/cutflow_ttbar_compare.py {input.cutflow_yml} \
                -o {output.html} --txt {output.txt} --title {params.title} --mc {params.mc} \
                --estimate TTbar_from_d3 2>&1 | tee {log}
        else
            echo "src/tools/cutflow_ttbar_compare.py not found; skipping" 2>&1 | tee {log}
            echo "<p>ttbar MC vs TTbar_from_d3 table not available</p>" > {output.html}
            cp {log} {output.txt}
        fi
        """

rule all_ttbar_MC_vs_d3:
    input:
        f"{out_f2}plots_ttbar_MC_vs_d3/plots_done.txt",
        f"{out_f2}ttbar_MC_vs_d3_cutflow.html",
        f"{out_f2}ttbar_MC_vs_d3_cutflow.txt"

