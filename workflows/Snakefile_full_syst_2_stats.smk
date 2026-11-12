# ==============================================================================
# coffea4bees/workflows/Snakefile_full_syst_2_stats.smk
#
# Stage F.2: Full Systematics Combine Inputs, Plots, Gallery & Statistical Inference
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Transforms histogram outputs from Stage F.1 and upstream background closure models
# into CMS Combine datacards and ROOT shape files, visualizes shape variations,
# compiles an interactive HTML gallery, and executes the complete statistical inference
# pipeline (limits, significance, likelihood scans, post-fit distributions, and impacts).
#
# KEY PIPELINE STAGES & RULES:
#   1. Histogram JSON Serialization:
#      - `convert_nominal_to_json_full_syst`: Extracts nominal histograms from upstream
#        stat-only/nominal analysis and serializes them to `histAll_nominal.json`.
#      - `convert_signal_syst_to_json_full_syst`: Serializes all signal systematic variations
#        from Stage F.1 into `histAll_signals_syst.json`.
#
#   2. Combine Datacard & ROOT Shape Construction (`make_combine_inputs_full_syst`):
#      - Integrates nominal background shapes, signal systematic variations, and background
#        systematics closure transfer functions (`bkg_syst_file`).
#      - Automatically harmonizes the histogram binning (`rebin`) to match the background
#        closure determination (e.g. `rebin: 24` for ttHbb SvB_MA.ps_ttHbb).
#      - Outputs CMS Combine text datacards (`datacard__<channel>.txt`) and binary shape
#        ROOT files (`shapes.root`).
#
#   3. Systematic Variations Visualization & Gallery:
#      - `plot_systematics_full_syst`: Invokes `coffea4bees/plots/make_syst_plots.py` to
#        generate overlaid shape comparison plots (Nominal vs. Up vs. Down) and ratio
#        panels for every systematic nuisance parameter.
#      - `gallery_systematics_full_syst`: Invokes `src/plotting/make_gallery.py` to compile
#        an interactive HTML gallery (`index.html`) displaying all systematic variation plots.
#
#   4. Statistical Inference (CMS Combine Suite):
#      - `workspace`: Compiles text datacard into Combine physics workspace.
#      - `limits`: Computes asymptotic CLs upper limits on signal strength modifier (r).
#      - `significance`: Computes expected / blinded signal significance.
#      - `likelihood_scan` & chunks: Evaluates profile likelihood scan across range of r.
#      - `postfit`: Computes background-only and signal-plus-background post-fit shapes.
#      - `impacts`: Full nuisance parameter fit suite (`impacts_initial_fit`, `impacts_do_fits`,
#        `impacts_collect`) producing impact summary PDF.
#      - `split_impacts`: Computes grouped and decoupled impact uncertainties.
#
#   5. Consolidated Statistical Summary (`stat_summary_full_syst`):
#      - Ingests all statistical results, limit JSONs, significance logs, scans, and galleries,
#        producing `summary.html` and `summary.md`.
#
# EXECUTION TARGETS:
#   - `all_full_syst_F_2`: Master target building datacards, variation plots, galleries,
#     complete Combine fits, and statistical summary report.
# ==============================================================================
import os
import re

config.setdefault('label', "ttHbb_full_syst")
config.setdefault('output_path', f"output/{config['label']}/")
base_output = config.get('output_path', f"output/{config['label']}/")
if not base_output.endswith('/'):
    base_output += '/'
if base_output.rstrip('/').endswith('PhaseF_2_stats') or base_output.rstrip('/').endswith('stat_analysis'):
    F2_OUTPUT_PATH = base_output
else:
    F2_OUTPUT_PATH = os.path.join(base_output, 'PhaseF_2_stats/')
if not F2_OUTPUT_PATH.endswith('/'):
    F2_OUTPUT_PATH += '/'

F1_OUTPUT_PATH = os.path.join(base_output, 'PhaseF_1_analysis/')
if not F1_OUTPUT_PATH.endswith('/'):
    F1_OUTPUT_PATH += '/'

config.setdefault('combine_flags', "--blind")
config.setdefault('closure_base', "coffea4bees/stats_analysis/files/HIG-24-010")
config.setdefault('channels', {})
config.setdefault('combine_outdir', "datacards/ttHbb")

config.setdefault('convert_hist_to_json', {})
config['convert_hist_to_json'].setdefault('syst_flag', "")
config['convert_hist_to_json'].setdefault(
    'histos',
    [ch_config['variable'] for ch_config in config.get('channels', {}).values() if ch_config.get('variable')]
)

config.setdefault('make_combine_inputs', {})
config['make_combine_inputs'].setdefault('stat_only', "")
config['make_combine_inputs'].setdefault('bkgsyst', "")
config['make_combine_inputs'].setdefault('syst_file', "")
config['make_combine_inputs'].setdefault('metadata_template', config.get('metadata_template', "coffea4bees/stats_analysis/metadata/{channel}.yml"))
config['make_combine_inputs'].setdefault('multijet_process', "data")
config['make_combine_inputs'].setdefault('tt_processes', ["TTTo", "TTbar4b_from_d3"])
config['make_combine_inputs'].setdefault('variable_binning', "")

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

config.setdefault('container_wrapper', "./run_container combine")
config.setdefault('stats_container_wrapper', config.get('container_wrapper', "./run_container combine"))
config.setdefault('analysis_container_wrapper', config.get('analysis_wrapper', "./run_container"))
config.setdefault('python_bin', os.getenv("CONTAINER_PYTHON", "python3"))

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
    val = config.get('make_combine_inputs', {}).get('stat_only', '')
    if isinstance(val, bool):
        return '--stat_only' if val else ''
    if str(val).lower() in ['true', '1', '--stat_only', '--stat-only']:
        return '--stat_only'
    return ''

def normalize_to_posix(path: str) -> str:
    if not path:
        return ""
    if path.startswith("root://cmseos.fnal.gov/"):
        rel = path.replace("root://cmseos.fnal.gov/", "").lstrip("/")
        return f"/eos/uscms/{rel}"
    return path

def get_bkgsyst_for_channel(channel):
    ch_config = config.get('channels', {}).get(channel, {})
    if 'bkgsyst' in ch_config and ch_config['bkgsyst']:
        return normalize_to_posix(ch_config['bkgsyst'])
    global_bkgsyst = config.get('make_combine_inputs', {}).get('bkgsyst') or config.get('bkgsyst')
    if global_bkgsyst:
        return normalize_to_posix(global_bkgsyst.format(
            output_path=config.get('output_path', 'output/'),
            channel=channel,
            closure_subdir=ch_config.get('closure_subdir', channel)
        ))
    if config.get('inputs', {}).get('bkg_syst_file'):
        return normalize_to_posix(config['inputs']['bkg_syst_file'])
    upstream = config.get('inputs', {}).get('upstream_roasts', {})
    if isinstance(upstream, dict) and 'bkg_syst' in upstream:
        bkg_id = upstream['bkg_syst']
        import glob
        for base in [f"roasts/{bkg_id}", f"/uscms_data/d3/algomez/HH4b/prod/{bkg_id}/barista"]:
            cands = glob.glob(f"{base}/output/**/hists_closure_*.pkl", recursive=True)
            if cands:
                return cands[0]
    closure_subdir = ch_config.get('closure_subdir', channel)
    closure_base = config.get('closure_base', "coffea4bees/stats_analysis/files/HIG-24-010")
    candidate = f"{closure_base}/{closure_subdir}/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_{closure_subdir}_rebin1.pkl"
    return candidate if os.path.exists(candidate) else ""

def resolve_rebin_for_channel(channel):
    ch_cfg = config.get('channels', {}).get(channel, {})
    if 'rebin' in ch_cfg and ch_cfg['rebin']:
        return int(ch_cfg['rebin'])
    if config.get('make_combine_inputs', {}).get('rebin'):
        return int(config['make_combine_inputs']['rebin'])
    if config.get('rebin'):
        return int(config['rebin'])
    bkg_file = get_bkgsyst_for_channel(channel)
    if bkg_file and isinstance(bkg_file, str):
        m = re.search(r'rebin(\d+)', bkg_file)
        if m:
            return int(m.group(1))
    return 1

for channel, ch_config in config.get('channels', {}).items():
    bkg_file = get_bkgsyst_for_channel(channel)
    if bkg_file:
        ch_config.setdefault('bkgsyst', bkg_file)

wildcard_constraints:
    channel = "|".join(config['channels'].keys()) if config['channels'] else "[a-zA-Z0-9_]+"

def get_region_for_channel(channel):
    ch_config = config.get('channels', {}).get(channel, {})
    if 'region' in ch_config:
        return ch_config['region']
    import shlex
    flags = config.get('combine_flags', '')
    tokens = shlex.split(flags)
    for idx, t in enumerate(tokens[:-1]):
        if t == '--region':
            return tokens[idx+1]
    return 'SR'

def get_nominal_coffea_file():
    cand = ""
    if config.get('inputs', {}).get('nominal_hists'):
        cand = config['inputs']['nominal_hists']
    elif config.get('inputs', {}).get('stat_only_coffea'):
        cand = config['inputs']['stat_only_coffea']
    elif config.get('nominal_coffea'):
        cand = config['nominal_coffea']
    else:
        upstream = config.get('inputs', {}).get('upstream_roasts', {})
        if isinstance(upstream, dict):
            nom_id = upstream.get('stat_only') or upstream.get('nominal')
            if nom_id:
                for base in [f"roasts/{nom_id}", f"/uscms_data/d3/algomez/HH4b/prod/{nom_id}/barista"]:
                    c1 = f"{base}/output/histAll_{nom_id}.coffea"
                    if os.path.exists(c1):
                        return c1
                    c2 = f"{base}/output/PhaseF_1_analysis/histAll_{nom_id}.coffea"
                    if os.path.exists(c2):
                        return c2
    posix_cand = normalize_to_posix(cand)
    if os.path.exists(posix_cand):
        return posix_cand
    if os.path.exists(cand):
        return cand
    return posix_cand if posix_cand else f"{base_output}histAll_nominal.coffea"

include: "helpers/common.smk"

module combine:
    snakefile: os.path.join(os.getcwd(), "src/stat_analysis/combine.smk")
    config: config

rule convert_nominal_to_json_full_syst:
    input:
        coffea_file = lambda wildcards: get_nominal_coffea_file(),
        script = "src/tools/convert_hist_to_json.py"
    output: f"{F2_OUTPUT_PATH}histAll_nominal.json"
    params:
        histos = lambda wildcards: " ".join([cfg['variable'] for cfg in config.get('channels', {}).values() if cfg.get('variable')]),
        run_container_wrapper = "./run_container",
        python_bin = lambda wildcards: config.get("python_bin", "python3")
    log: f"{F2_OUTPUT_PATH}logs/convert_nominal_to_json_{config['label']}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} {input.script} \
            -i {input.coffea_file} -o {output} \
            --histos {params.histos} 2>&1 | tee {log}
        """

rule convert_signal_syst_to_json_full_syst:
    input:
        coffea_file = f"{F1_OUTPUT_PATH}histAll_{config['label']}_signals_syst.coffea",
        script = "src/tools/convert_hist_to_json.py"
    output: f"{F2_OUTPUT_PATH}histAll_signals_syst.json"
    params:
        histos = lambda wildcards: " ".join([cfg['variable'] for cfg in config.get('channels', {}).values() if cfg.get('variable')]),
        run_container_wrapper = "./run_container",
        python_bin = lambda wildcards: config.get("python_bin", "python3")
    log: f"{F2_OUTPUT_PATH}logs/convert_signal_syst_to_json_{config['label']}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output}) $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} {input.script} \
            -i {input.coffea_file} -o {output} --syst_file \
            --histos {params.histos} 2>&1 | tee {log}
        """

rule make_combine_inputs_full_syst:
    input:
        injson = f"{F2_OUTPUT_PATH}histAll_nominal.json",
        injsonsyst = f"{F2_OUTPUT_PATH}histAll_signals_syst.json",
        bkgsyst = lambda wildcards: [get_bkgsyst_for_channel(wildcards.channel)] if get_bkgsyst_for_channel(wildcards.channel) else [],
        script = "coffea4bees/stats_analysis/make_combine_inputs.py",
        metadata_file = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0])
    output:
        datacard = f"{F2_OUTPUT_PATH}{{channel}}/datacards/datacard__{{channel}}.txt",
        shapes = f"{F2_OUTPUT_PATH}{{channel}}/datacards/shapes.root"
    params:
        variable = lambda wildcards: config['channels'][wildcards.channel]['variable'],
        rebin = lambda wildcards, input: resolve_rebin_for_channel(wildcards.channel),
        metadata = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0]),
        output_dir = lambda wildcards: f"{F2_OUTPUT_PATH}{wildcards.channel}/datacards/",
        variable_binning = lambda wildcards, input: config['make_combine_inputs']['variable_binning'],
        stat_only = lambda wildcards, input: get_stat_only_flag(wildcards.channel),
        signal = lambda wildcards: wildcards.channel,
        bkg_flag = lambda wildcards, input: f'--bkg_syst_file "{input.bkgsyst[0]}"' if (hasattr(input, 'bkgsyst') and input.bkgsyst) else "",
        tag_flags = lambda wildcards: (
            f"{config['channels'][wildcards.channel].get('combine_flags', config.get('combine_flags', ''))} "
            f"--region {get_region_for_channel(wildcards.channel)} "
            + (f"--cut {config['channels'][wildcards.channel]['cut']} " if 'cut' in config['channels'][wildcards.channel] and config['channels'][wildcards.channel]['cut'] not in ['', 'sum'] else '')
            + (f"--data_process {config['channels'][wildcards.channel]['data_process']} " if 'data_process' in config['channels'][wildcards.channel] else '')
            + f"--multijet_process {config['channels'][wildcards.channel].get('multijet_process', config.get('make_combine_inputs', {}).get('multijet_process', 'data'))} "
            f"--tt_processes {' '.join(config['channels'][wildcards.channel].get('tt_processes', config.get('make_combine_inputs', {}).get('tt_processes', ['TTTo', 'TTbar4b_from_d3'])))} "
            + ("--unify_background " if (config.get('unify_background', False) or config.get('make_combine_inputs', {}).get('unify_background', False) or config['channels'][wildcards.channel].get('unify_background', False)) else "")
        ),
        container_wrapper = config['stats_container_wrapper']
    log: f"{F2_OUTPUT_PATH}logs/make_combine_inputs_{{channel}}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p {params.output_dir} $(dirname {log})
        {params.container_wrapper} python3 {input.script} \
            -f {input.injson} \
            -s {input.injsonsyst} \
            {params.bkg_flag} \
            -r {params.rebin} \
            -o {params.output_dir} \
            -v {params.variable} \
            -m {params.metadata} \
            {params.tag_flags} 2>&1 | tee {log}
        """

rule plot_systematics_full_syst:
    input:
        datacard = f"{F2_OUTPUT_PATH}{{channel}}/datacards/datacard__{{channel}}.txt",
        shapes = f"{F2_OUTPUT_PATH}{{channel}}/datacards/shapes.root",
        metadata = lambda wildcards: config['make_combine_inputs']['metadata_template'].format(channel=wildcards.channel.split('_')[0])
    output:
        done = f"{F2_OUTPUT_PATH}{{channel}}/plots_systematics/plots_done.txt"
    params:
        out_dir = f"{F2_OUTPUT_PATH}{{channel}}/plots_systematics/",
        variable = lambda wildcards: config['channels'][wildcards.channel]['variable'].replace(".", "_"),
        signal = lambda wildcards: config['channels'][wildcards.channel].get('signallabel', wildcards.channel),
        run_container_wrapper = config.get('analysis_container_wrapper', './run_container'),
        python_bin = lambda wildcards: config.get("python_bin", "python3")
    log: f"{F2_OUTPUT_PATH}logs/plot_systematics_{{channel}}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p {params.out_dir} $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} coffea4bees/plots/make_syst_plots.py \
            -i {input.shapes} \
            -d {input.datacard} \
            -o {params.out_dir} \
            -v {params.variable} \
            -s {params.signal} \
            -m {input.metadata} 2>&1 | tee {log}
        touch {output.done}
        """

rule gallery_systematics_full_syst:
    input:
        done = f"{F2_OUTPUT_PATH}{{channel}}/plots_systematics/plots_done.txt"
    output:
        html = f"{F2_OUTPUT_PATH}{{channel}}/plots_systematics/index.html"
    params:
        plot_dir = f"{F2_OUTPUT_PATH}{{channel}}/plots_systematics/",
        title = lambda wildcards: f"Systematic Variations — {wildcards.channel}",
        run_container_wrapper = config.get('analysis_container_wrapper', './run_container'),
        python_bin = lambda wildcards: config.get("python_bin", "python3")
    log: f"{F2_OUTPUT_PATH}logs/gallery_systematics_{{channel}}.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} src/plotting/make_gallery.py \
            {params.plot_dir} \
            --title "{params.title}" \
            --output {output.html} 2>&1 | tee {log}
        """

def stat_summary_inputs_full_syst(wildcards):
    base = F2_OUTPUT_PATH.rstrip('/')
    files = []
    for channel, ch_config in config.get('channels', {}).items():
        sig = ch_config.get('signallabel')
        if not sig:
            continue
        files += [
            f"{base}/{channel}/datacards/datacard__{channel}.txt",
            f"{base}/{channel}/limits/datacard_limits__{sig}.json",
            f"{base}/{channel}/significance/datacard_significance__{sig}.json",
            f"{base}/{channel}/likelihood_scan/datacard_likelihood_scan__{sig}.pdf",
            f"{base}/{channel}/postfit/datacard_postfit__{sig}.pdf",
            f"{base}/{channel}/plots_systematics/index.html",
        ]
        if not get_stat_only_flag(channel):
            files += [
                f"{base}/{channel}/impacts/datacard_impacts__{sig}.pdf",
                f"{base}/{channel}/impacts/datacard_impacts_split__{sig}",
            ]
    return files

rule stat_summary_full_syst:
    input: stat_summary_inputs_full_syst
    output:
        html = f"{F2_OUTPUT_PATH}summary.html",
        md = f"{F2_OUTPUT_PATH}summary.md"
    log: f"{F2_OUTPUT_PATH}logs/stat_summary.log"
    params:
        stat_dir = F2_OUTPUT_PATH.rstrip('/'),
        channels = " ".join(f"--channel {ch}={cfg['signallabel']}" for ch, cfg in config.get('channels', {}).items() if cfg.get('signallabel')),
        variables = " ".join(f"--variable {ch}={cfg['variable']}" for ch, cfg in config.get('channels', {}).items() if cfg.get('signallabel') and cfg.get('variable')),
        blind = lambda wildcards: "--blind" if "--blind" in config.get('combine_flags', '') else "",
        label = config['label'],
        run_container_wrapper = config['analysis_container_wrapper'],
        python_bin = lambda wildcards: config.get("python_bin", "python3")
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {log})
        {params.run_container_wrapper} {params.python_bin} src/stat_analysis/stat_summary.py {params.stat_dir} \
            -o {output.html} --md {output.md} {params.channels} {params.variables} {params.blind} \
            --label {params.label} 2>&1 | tee {log}
        """

rule all_full_syst_F_2:
    input:
        [
            f"{F2_OUTPUT_PATH}{channel}/limits/datacard_limits__{ch_config['signallabel']}.json"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/postfit/datacard_postfit__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/significance/datacard_significance__{ch_config['signallabel']}.log"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/significance/datacard_significance__{ch_config['signallabel']}.json"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/likelihood_scan/datacard_likelihood_scan__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/impacts/datacard_impacts__{ch_config['signallabel']}.pdf"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel') and not get_stat_only_flag(channel)
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/impacts/datacard_impacts_split__{ch_config['signallabel']}"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel') and not get_stat_only_flag(channel)
        ] + [
            f"{F2_OUTPUT_PATH}{channel}/plots_systematics/index.html"
            for channel, ch_config in config.get('channels', {}).items() if ch_config.get('signallabel')
        ] + [
            f"{F2_OUTPUT_PATH}summary.html",
        ]

# Import Combine rules from combine.smk
use rule * from combine as *

localrules: convert_nominal_to_json_full_syst, convert_signal_syst_to_json_full_syst, make_combine_inputs_full_syst, plot_systematics_full_syst, gallery_systematics_full_syst, stat_summary_full_syst, all_full_syst_F_2
