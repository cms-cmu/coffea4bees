import os

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
config.setdefault('combine_flags', "--blind")
config.setdefault('closure_base', "coffea4bees/stats_analysis/files/HIG-24-010")

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
config.setdefault('likelihood_scan_points', 20)
config.setdefault('likelihood_scan_split_size', 10)
config.setdefault('likelihood_scan_r_min', -10)
config.setdefault('likelihood_scan_r_max', 10)

config.setdefault('year_eras', {
    'UL16_preVFP':  ['C', 'D', 'E', 'F'],
    'UL16_postVFP': ['F', 'G', 'H'],
    'UL17':         ['C', 'D', 'E', 'F'],
    'UL18':         ['A', 'B', 'C', 'D'],
})
config.setdefault('channels', {})
config.setdefault('combine_outdir', "datacards/HH4b_fine")

### Containers
config.setdefault('container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('combine_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.0-harvester_v3.1.0")
config.setdefault('container_wrapper', "./run_container combine")
config.setdefault('stats_container_wrapper', config.get('container_wrapper', "./run_container combine"))

# Decoupled config definitions and path resolution
def get_bkgsyst_for_channel(channel):
    ch_config = config['channels'].get(channel, {})
    if 'bkgsyst' in ch_config and ch_config['bkgsyst']:
        return ch_config['bkgsyst']
    global_bkgsyst = config.get('make_combine_inputs', {}).get('bkgsyst') or config.get('bkgsyst')
    if global_bkgsyst:
        return global_bkgsyst.format(
            output_path=config.get('output_path', 'output/'),
            channel=channel,
            closure_subdir=ch_config.get('closure_subdir', channel)
        )
    closure_subdir = ch_config.get('closure_subdir', channel)
    closure_base = config.get('closure_base', "coffea4bees/stats_analysis/files/HIG-24-010")
    return f"{closure_base}/{closure_subdir}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{closure_subdir}_rebin1.pkl"

for channel, ch_config in config.get('channels', {}).items():
    ch_config.setdefault('bkgsyst', get_bkgsyst_for_channel(channel))

# Constrain channel wildcard
wildcard_constraints:
    channel = "|".join(config['channels'].keys()) if config['channels'] else "[a-zA-Z0-9_]+"

def get_stat_only_flag():
    val = config.get('make_combine_inputs', {}).get('stat_only', '--stat_only')
    if isinstance(val, bool):
        return '--stat_only' if val else ''
    if val in ['--stat_only', '']:
        return val
    if str(val).lower() in ['true', '1']:
        return '--stat_only'
    if str(val).lower() in ['false', '0', 'none', '']:
        return ''
    return str(val)

def get_region_for_channel(channel):
    # 1. Check channel-specific setting
    ch_config = config['channels'].get(channel, {})
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

include: "helpers/common.smk"
module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: config

rule all_stats:
    input:
        [
            f"{config['output_path']}stat_analysis/{channel}/limits/datacard_limits__{ch_config['signallabel']}.json"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{config['output_path']}stat_analysis/{channel}/postfit/datacard_postfit__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{config['output_path']}stat_analysis/{channel}/significance/datacard_significance__{ch_config['signallabel']}.log"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{config['output_path']}stat_analysis/{channel}/likelihood_scan/datacard_likelihood_scan__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ]

use rule convert_hist_to_json from stat_analysis with:
    input:
        coffea_file = f"{config['output_path']}histAll_{config['label']}.coffea",
        script = "src/tools/convert_hist_to_json.py"
    output: f"{config['output_path']}histAll_{config['label']}.json"
    params:
        syst_flag=lambda wildcards, input: (
            f"{config['convert_hist_to_json']['syst_flag']} "
            f"--histos {' '.join(config['convert_hist_to_json']['histos'])}"
            if config['convert_hist_to_json']['histos'] else config['convert_hist_to_json']['syst_flag']
        ),
        container_wrapper = "./run_container"
    container: None
    log: f"{config['output_path']}logs/convert_hist_to_json_{config['label']}.log"

use rule make_combine_inputs from stat_analysis with:
    input:
        injson = f"{config['output_path']}histAll_{config['label']}.json",
        injsonsyst = list([]),
        bkgsyst = lambda wildcards: config['channels'][wildcards.channel]['bkgsyst'],
        script = "coffea4bees/stats_analysis/make_combine_inputs.py",
        metadata_file = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0])
    output: f"{config['output_path']}stat_analysis/{{channel}}/datacards/datacard__{{channel}}.txt"
    params:
        variable = lambda wildcards: config['channels'][wildcards.channel]['variable'],
        syst_file = lambda wildcards, input: f"--syst_file {config['make_combine_inputs']['syst_file']}" if config['make_combine_inputs']['syst_file'] else "",
        rebin = lambda wildcards, input: config['make_combine_inputs']['rebin'],
        metadata = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0]),
        output_dir = lambda wildcards: f"{config['output_path']}stat_analysis/{wildcards.channel}/datacards/",
        variable_binning = lambda wildcards, input: config['make_combine_inputs']['variable_binning'],
        stat_only = lambda wildcards, input: get_stat_only_flag(),
        signal = lambda wildcards: wildcards.channel,
        tag_flags = lambda wildcards: (
            f"{config['combine_flags']} "
            f"--region {get_region_for_channel(wildcards.channel)} "
            + (f"--cut {config['channels'][wildcards.channel]['cut']} " if 'cut' in config['channels'][wildcards.channel] and config['channels'][wildcards.channel]['cut'] not in ['', 'sum'] else '')
            + f"--multijet_process {config['make_combine_inputs']['multijet_process']} "
            f"--tt_processes {' '.join(config['make_combine_inputs']['tt_processes'])}"
        ),
        container_wrapper = config['stats_container_wrapper']
    log: f"{config['output_path']}logs/make_combine_inputs_{{channel}}.log"

# Import all rules from combine module
use rule * from combine as *

localrules: convert_hist_to_json, make_combine_inputs_channel
