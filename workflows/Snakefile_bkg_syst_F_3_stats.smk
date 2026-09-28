# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_F_3_stats.smk
#
# Stage F_3: Blinded Combine Statistical Interpretation
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Builds CMS Combine datacards for the ttH(bb) analysis incorporating the full
# background systematic covariance matrices derived in Stage F_2. Executes the
# statistical inference pipeline under blinded Signal Region (SR) conditions,
# computing expected asymptotic limits, signal significance, post-fit distributions,
# multidimensional likelihood profiles, and nuisance parameter impact rankings.
#
# STATISTICAL METHODOLOGY:
#   - Test Statistic: Profile Likelihood Ratio
#       q_\mu = -2 \ln \frac{\mathcal{L}(\mu, \hat{\hat{\boldsymbol{\theta}}}_\mu)}{\mathcal{L}(\hat{\mu}, \hat{\boldsymbol{\theta}})}
#   - Systematic Nuisances:
#       * Background shape uncertainties parameterized by the Stage F_2 closure
#         eigenvectors (injected via `--bkgsyst` pickle file).
#       * Standard experimental uncertainties (lumi, JEC, JER, b-tagging SFs).
#       * Theory and normalization uncertainties on ttbar and minor backgrounds.
#   - Blinded Evaluation: Signal region data observations remain blinded (`--blind`),
#     evaluating the Asimov expected sensitivity.
#
# WORKFLOW EXECUTION PIPELINE / RULES BREAKDOWN:
#   1. Histogram JSON Export (`convert_hist_to_json`):
#      - Converts nominal stitched Coffea histograms into structured JSON format
#        via `src/tools/convert_hist_to_json.py`.
#   2. Datacard & Shape ROOT Generation (`make_combine_inputs`):
#      - Executes `coffea4bees/stats_analysis/make_combine_inputs.py`.
#      - Reads nominal JSON, applies binning/rebinning, injects background closure
#        eigenvectors from Stage F_2 (`hists_closure_*.pkl`), and writes
#        datacards (`datacard__{channel}.txt`) and shape ROOT files.
#   3. Statistical Inference (Modular `combine.smk` & `stat_analysis.smk`):
#      - Text2Workspace: Converts datacards into Combine RooFit binary workspaces.
#      - AsymptoticLimits: Evaluates expected 95% CL upper limits on signal strength r.
#      - Significance: Computes expected signal discovery significance.
#      - FitDiagnostics: Evaluates pre-fit and post-fit nuisance distributions and covariance.
#      - MultiDimFit: Performs likelihood scans over r and key nuisance parameters.
#      - CombineHarvester Impacts: Computes pull and impact ranking plots for all nuisances.
#   4. Remote Archival & Notifications (`final_output`):
#      - Copies summary plots and datacards to EOS / CERNBox and dispatches email alert.
#
# TARGET RULES:
#   - `all_bkg_syst_F_3`: Master target evaluating limits, postfit, significance,
#     likelihood scans, and impacts across all configured analysis channels.
#   - `final_output`: Default pipeline rule copying results to CERNBox/EOS.
#
# INPUTS:
#   - Nominal Coffea/JSON: inputs/histAll_ttHbb_stitched.coffea (or nominal path)
#   - Stage F_2 Systematic Pickle: output/.../bkg_syst_F_2_run_two_stage_closure/closure_fits/.../hists_closure_*.pkl
#   - Channel Metadata: coffea4bees/stats_analysis/metadata/{channel}.yml
#
# OUTPUTS:
#   - Combine Datacards: datacards/stat_analysis/{channel}/datacards/datacard__{channel}.txt
#   - Asymptotic Limits: datacards/stat_analysis/{channel}/limits/datacard_limits__{signal}.json
#   - Post-fit Distributions: datacards/stat_analysis/{channel}/postfit/datacard_postfit__{signal}.pdf
#   - Likelihood Scans: datacards/stat_analysis/{channel}/likelihood_scan/datacard_likelihood_scan__{signal}.pdf
#   - Nuisance Impacts: datacards/stat_analysis/{channel}/impacts/datacard_impacts__{signal}.pdf
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc (using `./run_container combine` / Combine CMSSW container)
# ==============================================================================

import os
import sys
from datetime import datetime

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_mixeddata_closure/")
config.setdefault('combine_flags', "--blind")

config.setdefault('phase_f_output_path', "output/ttHbb/")
config.setdefault('phase_f_label', "ttHbb")

phase_f_out = config.get('phase_f_output_path', "output/ttHbb/")
if not phase_f_out.endswith("/"):
    phase_f_out += "/"
phase_f_lbl = config.get('phase_f_label', "ttHbb")

config.setdefault('convert_hist_to_json', {})
config['convert_hist_to_json'].setdefault('syst_flag', "")
config['convert_hist_to_json'].setdefault(
    'histos',
    [ch_config['variable'] for ch_config in config.get('channels', {}).values()]
)

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

out = config['output_path']
if not out.endswith("/"):
    out += "/"
default_nominal_coffea = "inputs/histAll_ttHbb_stitched.coffea" if os.path.exists("inputs/histAll_ttHbb_stitched.coffea") else f"{phase_f_out}histAll_{phase_f_lbl}.coffea"
default_nominal_json = "inputs/histAll_ttHbb_stitched.json" if os.path.exists("inputs/histAll_ttHbb_stitched.json") else f"{phase_f_out}histAll_{phase_f_lbl}.json"

nominal_coffea = config.get('nominal_coffea', default_nominal_coffea)
nominal_json = config.get('nominal_json', default_nominal_json)

# Decoupled config definitions and path resolution
def get_bkgsyst_for_channel(channel):
    ch_config = config.get('channels', {}).get(channel, {})
    if 'bkgsyst' in ch_config and ch_config['bkgsyst']:
        return ch_config['bkgsyst']
    global_bkgsyst = config.get('make_combine_inputs', {}).get('bkgsyst') or config.get('bkgsyst')
    if global_bkgsyst:
        return global_bkgsyst.format(
            output_path=out,
            channel=channel,
            closure_subdir=ch_config.get('closure_subdir', channel)
        )
    closure_subdir = ch_config.get('closure_subdir', config.get('channel', 'ttHbb'))
    mix_name = config.get('mix_name', 'ttHbb_mixeddata')
    var = ch_config.get('closure_var', config.get('variable', 'SvB_MA_ps_ttHbb'))
    rebin_val = config.get('rebin', '1')
    rebin_str = f"rebin{rebin_val}"
    return f"{out}bkg_syst_F_2_run_two_stage_closure/closure_fits/{closure_subdir}/{var}/hists_closure_{mix_name}_{var}_{rebin_str}.pkl"

for ch_name, ch_config in config.get('channels', {}).items():
    ch_config.setdefault('bkgsyst', get_bkgsyst_for_channel(ch_name))

# Constrain channel wildcard
wildcard_constraints:
    channel = "|".join(config['channels'].keys()) if config['channels'] else "[a-zA-Z0-9_]+"

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
    # 1. Check channel-specific setting
    ch_config = config.get('channels', {}).get(channel, {})
    if 'region' in ch_config:
        return ch_config['region']
    
    # 2. Check if region is specified inside combine_flags
    import shlex
    flags = config.get('combine_flags', '')
    tokens = shlex.split(flags)
    for idx, t in enumerate(tokens[:-1]):
        if t == '--region':
            return tokens[idx+1]
            
    # 3. Fallback to default SR
    return 'SR'

module stat_analysis:
    snakefile: "rules/stat_analysis.smk"
    config: config

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: config

# Resolve absolute CERNBox destination path
cern_user = config.get("cern_user", os.environ.get("USER", "algomez"))
cern_path = config.get("cern_path", "www/ttHbb/Plots/")
if not cern_path.startswith("/"):
    first_letter = cern_user[0]
    cern_path = f"/eos/user/{first_letter}/{cern_user}/{cern_path}"

# The first rule defined in the master file remains the default target
rule final_output:
    input:
        lambda wildcards: rules.all_bkg_syst_F_3.input
    params:
        output_dir = f"{datetime.now().strftime('%Y%m%d')}_{config['label']}/",
        cern_path = cern_path,
        email = lambda wildcards: config.get('email', "")
    shell:
        """
        echo "Copying results to eos"
        if [ -f "proxy/x509_proxy" ]; then
            export X509_USER_PROXY="$(pwd)/proxy/x509_proxy"
        fi
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d {params.cern_path}{params.output_dir} -t || echo "Warning: copy to EOS failed. Skipping remote upload."
        if [ -n "{params.email}" ]; then
            echo "Workflow for {config[label]} completed successfully on $(date)." | mail -s "Snakemake Success: {config[label]}" "{params.email}" || echo "Warning: failed to send success notification email."
        fi
        """

rule all_bkg_syst_F_3:
    input:
        [
            f"{out_f3}stat_analysis/{channel}/limits/datacard_limits__{ch_config['signallabel']}.json"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{out_f3}stat_analysis/{channel}/postfit/datacard_postfit__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{out_f3}stat_analysis/{channel}/significance/datacard_significance__{ch_config['signallabel']}.log"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{out_f3}stat_analysis/{channel}/significance/datacard_significance__{ch_config['signallabel']}.json"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{out_f3}stat_analysis/{channel}/likelihood_scan/datacard_likelihood_scan__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{out_f3}stat_analysis/{channel}/impacts/datacard_impacts__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ]

use rule convert_hist_to_json from stat_analysis with:
    input:
        coffea_file = nominal_coffea,
        script = "src/tools/convert_hist_to_json.py"
    output: nominal_json
    params:
        syst_flag=lambda wildcards, input: (
            f"{config['convert_hist_to_json']['syst_flag']} "
            f"--histos {' '.join(config['convert_hist_to_json']['histos'])}"
            if config['convert_hist_to_json']['histos'] else config['convert_hist_to_json']['syst_flag']
        ),
        container_wrapper = "./run_container"
    container: None
    log: f"{out}logs/convert_hist_to_json_{phase_f_lbl}.log"

use rule make_combine_inputs from stat_analysis with:
    input:
        injson = nominal_json,
        injsonsyst = list([]),
        bkgsyst = lambda wildcards: get_bkgsyst_for_channel(wildcards.channel),
        script = "coffea4bees/stats_analysis/make_combine_inputs.py",
        metadata_file = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0])
    output: f"{out_f3}stat_analysis/{{channel}}/datacards/datacard__{{channel}}.txt"
    params:
        variable = lambda wildcards: config['channels'][wildcards.channel]['variable'],
        syst_file = lambda wildcards, input: f"--syst_file {config['make_combine_inputs']['syst_file']}" if config['make_combine_inputs']['syst_file'] else "",
        rebin = lambda wildcards, input: config['make_combine_inputs']['rebin'],
        metadata = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0]),
        output_dir = lambda wildcards: f"{out_f3}stat_analysis/{wildcards.channel}/datacards/",
        variable_binning = lambda wildcards, input: config['make_combine_inputs']['variable_binning'],
        stat_only = lambda wildcards, input: get_stat_only_flag(wildcards.channel),
        signal = lambda wildcards: wildcards.channel,
        tag_flags = lambda wildcards: (
            f"{config['channels'][wildcards.channel].get('combine_flags', config['combine_flags'])} "
            f"--region {get_region_for_channel(wildcards.channel)} "
            + (f"--cut {config['channels'][wildcards.channel]['cut']} " if 'cut' in config['channels'][wildcards.channel] and config['channels'][wildcards.channel]['cut'] not in ['', 'sum'] else '')
            + (f"--data_process {config['channels'][wildcards.channel]['data_process']} " if 'data_process' in config['channels'][wildcards.channel] else '')
            + f"--multijet_process {config['channels'][wildcards.channel].get('multijet_process', config['make_combine_inputs']['multijet_process'])} "
            f"--tt_processes {' '.join(config['channels'][wildcards.channel].get('tt_processes', config['make_combine_inputs']['tt_processes']))}"
        ),
        container_wrapper = config['stats_container_wrapper']
    log: f"{out}logs/make_combine_inputs_{{channel}}.log"

use rule * from combine as *

localrules: final_output, all_bkg_syst_F_3, convert_hist_to_json, make_combine_inputs
