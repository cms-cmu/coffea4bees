import os

# Fallback defaults for backwards compatibility or running direct
config.setdefault('label', "nominal_wNewSvB")
config.setdefault('output_path', "output/nominal_wNewSvB/")
base_output = config.get('output_path', "output/nominal_wNewSvB/")
if not base_output.endswith('/'):
    base_output += '/'
if base_output.rstrip('/').endswith('PhaseF_2_stats') or base_output.rstrip('/').endswith('stat_analysis'):
    F2_OUTPUT_PATH = base_output
else:
    F2_OUTPUT_PATH = os.path.join(base_output, 'PhaseF_2_stats/')
if not F2_OUTPUT_PATH.endswith('/'):
    F2_OUTPUT_PATH += '/'
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
config.setdefault('analysis_container_wrapper', config.get('analysis_wrapper', "./run_container"))

# Decoupled config definitions and path resolution
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

def get_bkgsyst_for_channel(channel):
    if get_stat_only_flag(channel):
        return []
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
    candidate = f"{closure_base}/{closure_subdir}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{closure_subdir}_rebin1.pkl"
    return candidate if os.path.exists(candidate) else []

for channel, ch_config in config.get('channels', {}).items():
    bkg_file = get_bkgsyst_for_channel(channel)
    if bkg_file:
        ch_config.setdefault('bkgsyst', bkg_file)

# Constrain channel wildcard
wildcard_constraints:
    channel = "|".join(config['channels'].keys()) if config['channels'] else "[a-zA-Z0-9_]+"

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

def get_metadata_for_channel(channel):
    ch_cfg = config.get('channels', {}).get(channel, {})
    if 'metadata' in ch_cfg and ch_cfg['metadata']:
        return ch_cfg['metadata']
    sig = ch_cfg.get('signal', ch_cfg.get('signallabel', 'ttHbb'))
    tmpl = config.get('make_combine_inputs', {}).get('metadata_template', "coffea4bees/stats_analysis/metadata/{channel}.yml")
    cand = tmpl.format(channel=sig)
    if os.path.exists(cand):
        return cand
    cand2 = tmpl.format(channel=channel.split('_')[0])
    if os.path.exists(cand2):
        return cand2
    if os.path.exists("coffea4bees/stats_analysis/metadata/ttHbb.yml"):
        return "coffea4bees/stats_analysis/metadata/ttHbb.yml"
    return cand

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
            f"{F2_OUTPUT_PATH}{channel}/limits/datacard_limits__{ch_config['signallabel']}.json"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/postfit/datacard_postfit__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/significance/datacard_significance__{ch_config['signallabel']}.log"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/significance/datacard_significance__{ch_config['signallabel']}.json"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/likelihood_scan/datacard_likelihood_scan__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/impacts/datacard_impacts__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel') and not get_stat_only_flag(channel)
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/impacts/datacard_impacts_split__{ch_config['signallabel']}"
            for channel, ch_config in config['channels'].items() if ch_config.get('signallabel') and not get_stat_only_flag(channel)
        ] + [
            f"{F2_OUTPUT_PATH}summary.html",
        ]
    run:
        import os
        legacy_stats = os.path.join(base_output, 'stat_analysis')
        if os.path.abspath(legacy_stats) != os.path.abspath(F2_OUTPUT_PATH.rstrip('/')):
            if not os.path.exists(legacy_stats) and not os.path.islink(legacy_stats):
                try:
                    os.symlink(os.path.basename(F2_OUTPUT_PATH.rstrip('/')), legacy_stats)
                except OSError:
                    pass
        legacy_json = os.path.join(base_output, f"histAll_{config['label']}.json")
        f2_json = os.path.join(F2_OUTPUT_PATH, f"histAll_{config['label']}.json")
        if os.path.abspath(legacy_json) != os.path.abspath(f2_json):
            if not os.path.exists(legacy_json) and not os.path.islink(legacy_json):
                try:
                    os.symlink(os.path.relpath(f2_json, base_output), legacy_json)
                except OSError:
                    pass

def stat_summary_inputs(wildcards):
    """Everything the summary page reads, for every channel with a signal label."""
    base = F2_OUTPUT_PATH.rstrip('/')
    files = []
    for channel, ch_config in config['channels'].items():
        sig = ch_config.get('signallabel')
        if not sig:
            continue
        files += [
            f"{base}/{channel}/datacards/datacard__{channel}.txt",
            f"{base}/{channel}/limits/datacard_limits__{sig}.json",
            f"{base}/{channel}/significance/datacard_significance__{sig}.json",
            f"{base}/{channel}/likelihood_scan/datacard_likelihood_scan__{sig}.pdf",
            f"{base}/{channel}/postfit/datacard_postfit__{sig}.pdf",
        ]
        if not get_stat_only_flag(channel):
            files += [
                f"{base}/{channel}/impacts/datacard_impacts__{sig}.pdf",
                f"{base}/{channel}/impacts/datacard_impacts_split__{sig}",
            ]
    return files

rule stat_summary:
    # One-page HTML overview of the Combine results (src/stat_analysis/stat_summary.py in barista):
    # per channel the expected significance, the expected limit with its bands, the datacard yields
    # per year, the likelihood-scan status, the postfit/scan figures and links to the raw outputs.
    # Published by roast like the plot galleries (*.html).
    input: stat_summary_inputs
    output:
        html = f"{F2_OUTPUT_PATH}summary.html",
        md = f"{F2_OUTPUT_PATH}summary.md"
    log: f"{F2_OUTPUT_PATH}logs/stat_summary.log"
    params:
        stat_dir = F2_OUTPUT_PATH.rstrip('/'),
        channels = " ".join(f"--channel {ch}={cfg['signallabel']}" for ch, cfg in config['channels'].items() if cfg.get('signallabel')),
        variables = " ".join(f"--variable {ch}={cfg['variable']}" for ch, cfg in config['channels'].items() if cfg.get('signallabel') and cfg.get('variable')),
        blind = lambda wildcards: "--blind" if "--blind" in config.get('combine_flags', '') else "",
        label = config['label'],
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python")
    container: None
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        if [ -f src/stat_analysis/stat_summary.py ]; then
            {params.run_container_wrapper} {params.python_bin} src/stat_analysis/stat_summary.py {params.stat_dir} \
                -o {output.html} --md {output.md} {params.channels} {params.variables} {params.blind} \
                --label {params.label} 2>&1 | tee {log}
        else
            echo "src/stat_analysis/stat_summary.py not found in this barista checkout; skipping summary" 2>&1 | tee {log}
            echo "<p>stat analysis summary not available (barista checkout predates src/stat_analysis/stat_summary.py)</p>" > {output.html}
            cp {log} {output.md}
        fi
        """

if (base_output.rstrip('/').endswith(('V4', 'V4_analysis'))
        or 'TTbar4b_from_MvD' in config.get('make_combine_inputs', {}).get('tt_processes', [])):
    nominal_coffea_file = f"{base_output}histAll_{config['label']}.coffea"
elif os.path.exists(f"{base_output}histAll_{config['label']}.coffea") and not os.path.exists(f"{base_output}PhaseF_1_analysis/histAll_{config['label']}.coffea"):
    nominal_coffea_file = f"{base_output}histAll_{config['label']}.coffea"
else:
    nominal_coffea_file = f"{base_output}PhaseF_1_analysis/histAll_{config['label']}.coffea"

use rule convert_hist_to_json from stat_analysis with:
    input:
        coffea_file = nominal_coffea_file,
        script = "src/tools/convert_hist_to_json.py"
    output: f"{F2_OUTPUT_PATH}histAll_{config['label']}.json"
    params:
        syst_flag=lambda wildcards, input: (
            f"{config['convert_hist_to_json']['syst_flag']} "
            f"--histos {' '.join(config['convert_hist_to_json']['histos'])}"
            if config['convert_hist_to_json']['histos'] else config['convert_hist_to_json']['syst_flag']
        ),
        container_wrapper = "./run_container"
    container: None
    log: f"{F2_OUTPUT_PATH}logs/convert_hist_to_json_{config['label']}.log"

use rule make_combine_inputs from stat_analysis with:
    input:
        injson = f"{F2_OUTPUT_PATH}histAll_{config['label']}.json",
        injsonsyst = lambda wildcards: [config['make_combine_inputs']['syst_file']] if config.get('make_combine_inputs', {}).get('syst_file') else [],
        bkgsyst = lambda wildcards: get_bkgsyst_for_channel(wildcards.channel),
        script = "coffea4bees/stats_analysis/make_combine_inputs.py",
        metadata_file = lambda wildcards: get_metadata_for_channel(wildcards.channel)
    output: f"{F2_OUTPUT_PATH}{{channel}}/datacards/datacard__{{channel}}.txt"
    params:
        variable = lambda wildcards: config['channels'][wildcards.channel]['variable'],
        syst_file = lambda wildcards, input: f"--syst_file {input.injsonsyst[0]}" if (hasattr(input, 'injsonsyst') and input.injsonsyst) else (f"--syst_file {config['make_combine_inputs']['syst_file']}" if config.get('make_combine_inputs', {}).get('syst_file') else ""),
        bkg_syst_file = lambda wildcards, input: f'--bkg_syst_file "{input.bkgsyst}"' if getattr(input, 'bkgsyst', None) else "",
        rebin = lambda wildcards, input: config['channels'][wildcards.channel].get('rebin', config['make_combine_inputs'].get('rebin', 1)),
        metadata = lambda wildcards: get_metadata_for_channel(wildcards.channel),
        output_dir = lambda wildcards: f"{F2_OUTPUT_PATH}{wildcards.channel}/datacards/",
        variable_binning = lambda wildcards, input: config['make_combine_inputs']['variable_binning'],
        stat_only = lambda wildcards, input: get_stat_only_flag(wildcards.channel),
        signal = lambda wildcards: wildcards.channel,
        tag_flags = lambda wildcards: (
            f"{config['channels'][wildcards.channel].get('combine_flags', config['combine_flags'])} "
            f"--region {get_region_for_channel(wildcards.channel)} "
            + (f"--cut {config['channels'][wildcards.channel]['cut']} " if 'cut' in config['channels'][wildcards.channel] and config['channels'][wildcards.channel]['cut'] not in ['', 'sum'] else '')
            + (f"--data_process {config['channels'][wildcards.channel]['data_process']} " if 'data_process' in config['channels'][wildcards.channel] else '')
            + f"--multijet_process {config['channels'][wildcards.channel].get('multijet_process', config['make_combine_inputs']['multijet_process'])} "
            f"--tt_processes {' '.join(config['channels'][wildcards.channel].get('tt_processes', config['make_combine_inputs']['tt_processes']))} "
            + ("--unify_background " if (config.get('unify_background', False) or config.get('make_combine_inputs', {}).get('unify_background', False) or config['channels'][wildcards.channel].get('unify_background', False)) else "")
        ),
        container_wrapper = config['stats_container_wrapper']
    log: f"{F2_OUTPUT_PATH}logs/make_combine_inputs_{{channel}}.log"

# Import all rules from combine module
use rule * from combine as *

localrules: convert_hist_to_json, make_combine_inputs, stat_summary, all_stats
