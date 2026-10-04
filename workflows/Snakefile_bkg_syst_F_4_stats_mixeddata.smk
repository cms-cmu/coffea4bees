# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_F_4_stats_mixeddata.smk
#
# Stage F_4: Unblinded Combine Statistical Interpretation on Average Mixed Data
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Validates the complete statistical inference pipeline without unblinding real
# collision data in the Signal Region by utilizing the ensemble-average of the
# 4-tag mixed-data subsamples as a realistic pseudo-data observation.
# This unblinded mock analysis allows end-to-end verification of background closure,
# goodness-of-fit (GoF), signal injection/recovery, and likelihood profile behavior.
#
# STATISTICAL METHODOLOGY:
#   - Pseudo-Data Construction:
#     The unblinded observation in the 4-tag signal and control regions is constructed
#     by computing the mean yield across the N=16 independent mixed-data subsamples:
#       \bar{N}_{mixed}(x) = \frac{1}{N} \sum_{k=0}^{N-1} N_{mixed}^{(k)}(x)
#     with variance scaled by 1/N^2:
#       \sigma^2(\bar{N}_{mixed}(x)) = \frac{1}{N^2} \sum_{k=0}^{N-1} \sigma^2(N_{mixed}^{(k)}(x))
#   - Unblinded Combine Verification:
#     Executes Combine routines with `--unblind` using the average mixed data as
#     the observed count. Tests that:
#       * Goodness-of-Fit (Saturated model test) yields a p-value consistent with expectation.
#       * The fitted signal strength modifier is consistent with zero (r = 0 within uncertainties)
#         in the absence of injected signal.
#       * Systematic nuisance pulls and constraints behave normally.
#
# WORKFLOW EXECUTION PIPELINE / RULES BREAKDOWN:
#   1. Average Mixed Data Construction (`make_mixeddata_ave_json`):
#      - Reads nominal JSON template and Stage F_1 mixed-data coffea files across
#        all subsamples via `coffea4bees/stats_analysis/make_mixeddata_ave_json.py`.
#      - Injects the average mixed data yields as the 4-tag 'data' observation.
#      - Writes: `histAll_ttHbb_mixeddata_ave.json`
#   2. Unblinded Combine Inputs Generation (`make_combine_inputs_mixeddata`):
#      - Executes `make_combine_inputs.py` with the average mixed-data JSON as input,
#        incorporating Stage F_2 background systematic covariance matrices.
#      - Produces unblinded datacards: `stat_analysis_unblinded_mixeddata/{channel}/datacards/datacard__{channel}.txt`
#   3. Unblinded Statistical Inference (`combine.smk` & `stat_analysis.smk`):
#      - Runs Combine routines with unblinding:
#        * Unblinded limits: `datacard_limits__{signal}.json`
#        * Unblinded postfit distributions: `datacard_postfit__{signal}.pdf`
#        * Unblinded significance: `datacard_significance__{signal}.json`
#        * Unblinded likelihood scans: `datacard_likelihood_scan__{signal}.pdf`
#        * Unblinded nuisance impacts: `datacard_impacts__{signal}.pdf`
#        * Goodness-of-fit test: `datacard_gof__{signal}.pdf`
#
# TARGET RULE:
#   - `all_bkg_syst_F_4`: Master target for all unblinded pseudo-data statistical outputs.
#
# INPUTS:
#   - Nominal JSON: inputs/histAll_ttHbb_stitched.json (or nominal path)
#   - Stage A_3 Mixed Data Coffea: output/.../bkg_syst_A_3_process_subsamples/histAll_{channel}_mixeddata_v{m}.coffea
#   - Stage F_2 Systematic Pickle: output/.../bkg_syst_F_2_run_two_stage_closure/closure_fits/.../hists_closure_*.pkl
#
# OUTPUTS:
#   - Average Mixed JSON: output/.../bkg_syst_F_4_stats_mixeddata/histAll_ttHbb_mixeddata_ave.json
#   - Unblinded Datacards: output/.../bkg_syst_F_4_stats_mixeddata/stat_analysis_unblinded_mixeddata/{channel}/...
#   - Unblinded Limits, Postfit, Significance, Likelihood Scans, Impacts, GoF Plots
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc (using `./run_container combine` / Combine CMSSW container)
# ==============================================================================

import os
import sys

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_mixeddata_closure/")
config.setdefault('phase_f_output_path', "output/ttHbb/")
config.setdefault('phase_f_label', "ttHbb")

phase_f_out = config.get('phase_f_output_path', "output/ttHbb/")
if not phase_f_out.endswith("/"):
    phase_f_out += "/"
phase_f_lbl = config.get('phase_f_label', "ttHbb")

out = config['output_path']
if not out.endswith("/"):
    out += "/"
out_f4 = f"{out}bkg_syst_F_4_stats_mixeddata/"
if 'out_a3' not in globals():     # standalone: no bkg_syst_common
    out_a3 = f"{out}bkg_syst_A_3_process_subsamples/"
a3_channel = config.get('channel', 'ttHbb')
nominal_json = config['nominal_json']           # written by F_3 from inputs.nominal_hists
ave_mixeddata_json = config.get('ave_mixeddata_json', f"{out_f4}histAll_ttHbb_mixeddata_ave.json")

config.setdefault('make_combine_inputs', {})
config['make_combine_inputs'].setdefault('rebin', 1)
config['make_combine_inputs'].setdefault('variable_binning', "")
config['make_combine_inputs'].setdefault('stat_only', "--stat_only")
config['make_combine_inputs'].setdefault('bkgsyst', "")
config['make_combine_inputs'].setdefault('syst_file', "")
config['make_combine_inputs'].setdefault('metadata_template', config.get('metadata_template', "coffea4bees/stats_analysis/metadata/{channel}.yml"))
config['make_combine_inputs'].setdefault('multijet_process', "data")
config['make_combine_inputs'].setdefault('tt_processes', ["TTTo", "TTbar4b_from_d3"])

# Likelihood scan defaults
config.setdefault('likelihood_scan_points', 40)
config.setdefault('likelihood_scan_split_size', 10)
config.setdefault('likelihood_scan_r_min', 0)
config.setdefault('likelihood_scan_r_max', 2)
config.setdefault('likelihood_scan_y_cut', 30)
config.setdefault('likelihood_scan_y_max', 30)

config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})
config.setdefault('channels', {})
config.setdefault('combine_outdir', "datacards/")

### Containers
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('combine_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0")
config.setdefault('container_wrapper', "./run_container combine")
config.setdefault('stats_container_wrapper', config.get('container_wrapper', "./run_container combine"))

# Closure systematic path resolution
def get_bkgsyst_for_channel(channel, rebin=None):
    ch_config = config.get('channels', {}).get(channel, {})
    if rebin is None:
        rebin_val = str(config.get('rebin', '12'))
    else:
        rebin_val = str(rebin)
    rebin_str = f"rebin{rebin_val}"
    if 'bkgsyst' in ch_config and ch_config['bkgsyst']:
        import re as re_mod
        return re_mod.sub(r'rebin\d+', rebin_str, ch_config['bkgsyst'])
    global_bkgsyst = config.get('make_combine_inputs', {}).get('bkgsyst') or config.get('bkgsyst')
    if global_bkgsyst:
        return global_bkgsyst.format(
            output_path=out,
            channel=channel,
            closure_subdir=ch_config.get('closure_subdir', channel)
        )
    closure_subdir = ch_config.get('closure_subdir', config.get('channel', 'ttHbb'))
    mix_name = config.get('mix_name', '3bDvTMix4bDvT')
    var = ch_config.get('closure_var', config.get('variable', 'SvB_MA_ps_ttHbb'))
    classifier = config.get('classifier', 'SvB_MA')
    region = get_region_for_channel(channel)
    return f"{out}bkg_syst_F_2_run_two_stage_closure/closure_fits/{mix_name}/{classifier}/{rebin_str}/{region}/{closure_subdir}/hists_closure_{mix_name}_{var}_{rebin_str}.pkl"

for ch_name, ch_config in config.get('channels', {}).items():
    ch_config.setdefault('bkgsyst', get_bkgsyst_for_channel(ch_name))

# Filter channels applicable for mixeddata (variables present in mixeddata ntuples)
mixeddata_channels = [
    ch for ch, ch_cfg in config.get('channels', {}).items()
    if ch_cfg.get('variable') in ["SvB_MA.ps_ttHbb", "SvB_MA.ps_ttHbb_gt6"]
]

wildcard_constraints:
    channel = "|".join(mixeddata_channels) if mixeddata_channels else "[a-zA-Z0-9_]+",
    rebin = r"\d+"

def get_stat_only_flag(channel=None):
    if channel and channel in config.get('channels', {}):
        ch_val = config['channels'][channel].get('stat_only', None)
        if ch_val is not None:
            if isinstance(ch_val, bool):
                return '--stat_only' if ch_val else ''
            if str(ch_val).lower() in ['true', '1', '--stat_only', '--stat-only']:
                return '--stat_only'
            if str(ch_val).lower() in ['false', '0', 'none', '']:
                return ''
            return '--stat_only' if str(ch_val) == '--stat-only' else str(ch_val)
        if '_stat_only' in channel or channel.endswith('_stat'):
            return '--stat_only'
    val = config.get('make_combine_inputs', {}).get('stat_only', '--stat_only')
    if isinstance(val, bool):
        return '--stat_only' if val else ''
    if val in ['--stat_only', '']:
        return val
    if str(val).lower() in ['true', '1', '--stat_only', '--stat-only']:
        return '--stat_only'
    if str(val).lower() in ['false', '0', 'none', '']:
        return ''
    return '--stat_only' if str(val) == '--stat-only' else str(val)

def get_region_for_channel(channel):
    ch_config = config.get('channels', {}).get(channel, {})
    if 'region' in ch_config:
        return ch_config['region']
    return 'SR'

module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: config

def get_combine_targets_F_4(wildcards):
    checkpoint_output = checkpoints.evaluate_closure_candidates.get().output[0]
    import json
    try:
        with open(checkpoint_output, "r") as f:
            data = json.load(f)
        passing_rebins = data.get("passing_rebins", [])
    except Exception as e:
        print(f"[Snakemake] Error reading {checkpoint_output}: {e}")
        passing_rebins = []

    if not passing_rebins:
        print("[Snakemake] Notice: No candidate rebin scheme passed the two-stage closure test. Stage F_4 Mixed Data Combine targets will be empty.")
        return []

    targets = []
    for r in passing_rebins:
        for channel in mixeddata_channels:
            sig = config['channels'][channel].get('signallabel')
            if not sig:
                continue
            ch_dir = f"{out_f4}stat_analysis_unblinded_mixeddata_rebin{r}/{channel}/"
            targets.extend([
                f"{ch_dir}limits/datacard_limits__{sig}.json",
                f"{ch_dir}postfit/datacard_postfit__{sig}.pdf",
                f"{ch_dir}significance/datacard_significance__{sig}.log",
                f"{ch_dir}significance/datacard_significance__{sig}.json",
                f"{ch_dir}likelihood_scan/datacard_likelihood_scan__{sig}.pdf",
                f"{ch_dir}impacts/datacard_impacts__{sig}.pdf",
                f"{ch_dir}gof/datacard_gof__{sig}.pdf",
            ])
    return targets

rule all_bkg_syst_F_4:
    input:
        get_combine_targets_F_4

n_models_f4 = int(config.get('n_subsamples', config.get('n_models', config.get('n_samples', 16))))
subsample_indices_f4 = config.get('subsample_indices', list(range(n_models_f4)))
if isinstance(subsample_indices_f4, str):
    subsample_indices_f4 = [int(x) for x in subsample_indices_f4.split()]

rule make_mixeddata_ave_json:
    input:
        nominal_json = nominal_json,
        # the unweighted 4b mixed subsamples (A_3); the single-pass F_1 makes no per-subsample copies
        subsample_files = [f"{out_a3}histAll_{a3_channel}_mixeddata_v{v}.coffea" for v in subsample_indices_f4]
    output:
        ave_json = ave_mixeddata_json
    params:
        closure_dir = out_a3,
        file_template = f"histAll_{a3_channel}_mixeddata_v{{v}}.coffea",
        script = "coffea4bees/stats_analysis/make_mixeddata_ave_json.py",
        subsamples = " ".join(str(v) for v in subsample_indices_f4),
        container_wrapper = "./run_container"
    container: None
    log: f"{out}logs/make_mixeddata_ave_json.log"
    shell:
        """
        mkdir -p $(dirname {output.ave_json})
        mkdir -p $(dirname {log})
        {params.container_wrapper} python3 {params.script} \
            -i {input.nominal_json} \
            -c {params.closure_dir} \
            --file_template '{params.file_template}' \
            --subsamples {params.subsamples} \
            -o {output.ave_json} > {log} 2>&1
        """

use rule make_combine_inputs from stat_analysis as make_combine_inputs_mixeddata with:
    input:
        injson = ave_mixeddata_json,
        injsonsyst = list([]),
        bkgsyst = lambda wildcards: get_bkgsyst_for_channel(wildcards.channel, wildcards.rebin),
        script = "coffea4bees/stats_analysis/make_combine_inputs.py",
        metadata_file = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0])
    output: f"{out_f4}stat_analysis_unblinded_mixeddata_rebin{{rebin}}/{{channel}}/datacards/datacard__{{channel}}.txt"
    params:
        variable = lambda wildcards: config['channels'][wildcards.channel]['variable'],
        syst_file = lambda wildcards, input: f"--syst_file {config['make_combine_inputs']['syst_file']}" if config['make_combine_inputs']['syst_file'] else "",
        rebin = lambda wildcards, input: wildcards.rebin,
        metadata = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0]),
        output_dir = lambda wildcards: f"{out_f4}stat_analysis_unblinded_mixeddata_rebin{wildcards.rebin}/{wildcards.channel}/datacards/",
        variable_binning = lambda wildcards, input: config['make_combine_inputs']['variable_binning'],
        stat_only = lambda wildcards, input: get_stat_only_flag(wildcards.channel),
        signal = lambda wildcards: wildcards.channel,
        tag_flags = lambda wildcards: (
            "--three_tag threeTag --four_tag fourTag --region SR "
            + (f"--cut {config['channels'][wildcards.channel]['cut']} " if 'cut' in config['channels'][wildcards.channel] and config['channels'][wildcards.channel]['cut'] not in ['', 'sum'] else '')
            + (f"--data_process {config['channels'][wildcards.channel]['data_process']} " if 'data_process' in config['channels'][wildcards.channel] else '')
            + f"--multijet_process {config['channels'][wildcards.channel].get('multijet_process', config['make_combine_inputs']['multijet_process'])} "
            f"--tt_processes {' '.join(config['channels'][wildcards.channel].get('tt_processes', config['make_combine_inputs']['tt_processes']))}"
        ),
        container_wrapper = config['stats_container_wrapper']
    log: f"{out}logs/make_combine_inputs_unblinded_mixeddata_rebin{{rebin}}_{{channel}}.log"

localrules: all_bkg_syst_F_4, make_mixeddata_ave_json, make_combine_inputs_mixeddata
