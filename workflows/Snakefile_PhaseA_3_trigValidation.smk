# coffea4bees/workflows/Snakefile_PhaseA_3_trigValidation.smk
# A.3: validate the Phase A.2 trigger weights on signal MC. Included by Snakefile_TrigWeights.smk
# after Snakefile_PhaseA_2_trigWeights.smk, whose module `analysis`, years, config['dataset'] and
# merged friend index it uses.
#
# processor_HH4b runs three times over the same signal samples, differing only in the trigger:
#   noTrig  config_overrides cut_on_HLT_decision: false, apply_trigWeight: false
#   HLT     the MC HLT decision (OR of triggers_HH4b.yml), apply_trigWeight: false
#   HLT_SF  HLT decision x the trigger SF (emulated Data/MC efficiency, event_weights.py)
#           read from A.2's friend index = the nominal Run 3 treatment
# Each output's process axis is renamed to <dataset>__<variant> (scripts/trig_validation.py rename)
# so the variants can be merged into one file and overlaid by makePlots: per dataset, one gallery
# per era plus the Run 3 sum, fourTag, ratio to noTrig. The report gives, per dataset, the yields and
# HLT/noTrig, HLT_SF/noTrig and the mean SF (HLT_SF/HLT) per era and cut.
#
# trig_validation: {enabled: false} turns it off; datasets / years: what to validate (default: every
# A.2 dataset and year -- name a subset when A.2 makes friends for overlapping samples, e.g. the
# inclusive and stitched ttbar); runner: the processor_HH4b runner block (keep it identical to
# trigger_weights.runner -- under --shared-dask the first job's spec wins); regions: plotted
# regions (default SR, SB).

import re

TV = config.get('trig_validation') or {}
TV_ENABLED = bool(TV.get('enabled', True))
TV_OUT = f"{config['output_path']}trig_validation/"
TV_VARIANTS = ["noTrig", "HLT", "HLT_SF"]
TV_FRIENDS = f"{config['output_path']}trigger_weights/trigger_weights_friends.json"
TV_HISTALL = f"{TV_OUT}histAll_trigValidation.coffea"
TV_DATASETS = list(TV.get('datasets') or config['dataset'])
TV_YEARS = list(TV.get('years') or years)
for _d in TV_DATASETS:
    if _d not in config['dataset']:
        raise ValueError(f"trig_validation.datasets: {_d} is not an A.2 dataset (no trigger-weight friend)")
for _y in TV_YEARS:
    if _y not in years:
        raise ValueError(f"trig_validation.years: {_y} is not an A.2 year")
TV_PLOT_YEARS = TV_YEARS + ["Run3"]
TV_SCRIPT = "coffea4bees/workflows/scripts/trig_validation.py"
TV_WRAPPER = config['analysis_container_wrapper']
TV_PYTHON = config['python_bin']


def _tv_write_yaml(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def _tv_config(variant):
    """processor_HH4b config for one variant: signal only, no FvT/JCM/SvB, tight fourTag."""
    runner = copy.deepcopy(TV.get('runner') or config.get('runner') or {})
    runner.setdefault('write_coffea_output', True)
    if config.get('test', False):
        runner.update({'condor': False, 'shared_dask': False, 'run_performance': False})
    c = {
        'apply_FvT': False,
        'apply_JCM': False,
        'run_SvB': False,
        'SvB': None,
        'SvB_MA': None,
        'apply_btagSF': True,
        'fourTag_use_tight': True,      # nominal Run 3 fourTag
        'blind': False,
        'apply_trigWeight': variant == "HLT_SF",
        'require_trigWeight': variant == "HLT_SF",
        'friends_include': ["trigWeight"] if variant == "HLT_SF" else [],
    }
    if variant == "HLT_SF":
        # the friend index A.2 just made (parsed driver-side, so a local path is fine)
        c['friends'] = {'trigWeight': f"{TV_FRIENDS}@@trigWeight"}
    if variant == "noTrig":
        c['config_overrides'] = {'cut_on_HLT_decision': False}
    return {
        'processor': "coffea4bees/analysis/processors/processor_HH4b.py",
        'dataset_location': config.get('dataset_location', "coffea4bees/metadata/datasets/"),
        'friend_file': "coffea4bees/metadata/friends/friends_HH4b.yml",
        'weights_file': "coffea4bees/metadata/weights/weights_HH4b.yml",
        'runner': runner,
        'config': c,
    }


rule TV_config:
    input: get_trigger_weights_config_inputs
    output: f"{TV_OUT}configs/trig_validation_{{variant}}.yml"
    wildcard_constraints:
        variant = "|".join(TV_VARIANTS)
    run:
        _tv_write_yaml(output[0], _tv_config(wildcards.variant))

use rule analysis_processor from analysis as TV_hists with:
    input:
        runner_script = "runner.py",
        config_file = f"{TV_OUT}configs/trig_validation_{{variant}}.yml",
        friends = lambda wildcards: [TV_FRIENDS] if wildcards.variant == "HLT_SF" else []
    output: f"{TV_OUT}singlefiles/hist__{{variant}}__{{year}}.coffea"
    log: f"{TV_OUT}logs/hists__{{variant}}__{{year}}.log"
    wildcard_constraints:
        variant = "|".join(TV_VARIANTS),
        year = "|".join(TV_YEARS)
    params:
        datasets = " ".join(TV_DATASETS),
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = lambda wildcards: " ".join(filter(None, [
            "-t" if config.get("test", False) else "",
            config.get("additional_parameters", "")
        ])),
        run_container_wrapper = TV_WRAPPER,
        python_bin = TV_PYTHON

rule TV_rename:
    """Process axis <dataset> -> <dataset>__<variant> (and the cutflow keys <dataset>_<year> likewise)."""
    input:
        coffea = f"{TV_OUT}singlefiles/hist__{{variant}}__{{year}}.coffea",
        script = TV_SCRIPT
    output: f"{TV_OUT}singlefiles/renamed__{{variant}}__{{year}}.coffea"
    log: f"{TV_OUT}logs/rename__{{variant}}__{{year}}.log"
    wildcard_constraints:
        variant = "|".join(TV_VARIANTS),
        year = "|".join(TV_YEARS)
    params:
        datasets = " ".join(TV_DATASETS)
    shell:
        """
        set -eo pipefail
        {TV_WRAPPER} {TV_PYTHON} {input.script} rename -i {input.coffea} -o {output} \
            --processes {params.datasets} --suffix __{wildcards.variant} 2>&1 | tee {log}
        """

use rule merging_coffea_files from analysis as TV_merge with:
    input:
        files = expand(f"{TV_OUT}singlefiles/renamed__{{variant}}__{{year}}.coffea", variant=TV_VARIANTS, year=TV_YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: TV_HISTALL
    log: f"{TV_OUT}logs/merge.log"
    params:
        run_performance = False,
        run_container_wrapper = TV_WRAPPER,
        python_bin = TV_PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

_TV_STYLE = {"noTrig": ("No trigger", "#7f7f7f", "dashed"),
             "HLT": ("MC HLT decision", "#1f77b4", "solid"),
             "HLT_SF": ("MC HLT x trigger SF", "#e42536", "solid")}

rule TV_plot_config:
    """One dataset's three variants overlaid, fourTag, ratio panel = variant / noTrig."""
    output: f"{TV_OUT}plots_{{tv_dataset}}.yml"
    wildcard_constraints:
        tv_dataset = "|".join(re.escape(d) for d in TV_DATASETS)
    run:
        ds = wildcards.tv_dataset
        hists = {f"{ds}__{v}": {'process': f"{ds}__{v}", 'tag': 'fourTag', 'label': lab,
                                'edgecolor': col, 'fillcolor': col, 'histtype': 'step',
                                'linestyle': ls, 'scalefactor': 1}
                 for v, (lab, col, ls) in _TV_STYLE.items()}
        ratios = {f"{v}_to_noTrig": {'numerator': {'type': 'hists', 'key': f"{ds}__{v}"},
                                     'denominator': {'type': 'hists', 'key': f"{ds}__noTrig"},
                                     'uncertianty': 'nominal', 'color': _TV_STYLE[v][1], 'marker': "o"}
                  for v in ("HLT", "HLT_SF")}
        _tv_write_yaml(output[0], {'hists': hists, 'ratios': ratios, 'doRatio': 1,
                                   'regions': list(TV.get('regions', ["SR", "SB"])),
                                   'summary': ['m4j', 'v4j.mass', 'nSelJets', 'canJet0.pt', 'canJet3.pt',
                                               'quadJet_selected.lead.mass', 'quadJet_selected.subl.mass']})

use rule make_plots from analysis as TV_plots with:
    input:
        coffea_file = TV_HISTALL,
        metadata_file = f"{TV_OUT}plots_{{tv_dataset}}.yml",
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{TV_OUT}plots_{{tv_dataset}}/{{plot_year}}/plots_done.txt"
    # Plotting runs on the login node (localrule). makePlots forks `-p` workers per gallery, and
    # snakemake ran up to --jobs of these at once: 8 x 8 processes, each holding the merged
    # histograms, and two LPC nodes (cmslpc320, cmslpc322; 2026-10-07) went unresponsive during
    # this step. threads + -p {threads} caps it at cores/4 galleries x 4 workers.
    threads: 4
    wildcard_constraints:
        tv_dataset = "|".join(re.escape(d) for d in TV_DATASETS),
        plot_year = "|".join(TV_PLOT_YEARS)
    params:
        output_dir = lambda wildcards: f"{TV_OUT}plots_{wildcards.tv_dataset}/{wildcards.plot_year}/",
        metadata = lambda wildcards: f"{TV_OUT}plots_{wildcards.tv_dataset}.yml",
        extra_arguments = lambda wildcards, threads: f"-s xW -f png --year {wildcards.plot_year} -p {threads}",
        run_container_wrapper = TV_WRAPPER,
        python_bin = TV_PYTHON
    log: f"{TV_OUT}logs/plots_{{tv_dataset}}_{{plot_year}}.log"

rule TV_report:
    input:
        hists = TV_HISTALL,
        script = TV_SCRIPT
    output:
        txt = f"{TV_OUT}trig_validation.txt",
        yml = f"{TV_OUT}trig_validation.yml"
    log: f"{TV_OUT}logs/report.log"
    shell:
        """
        set -eo pipefail
        {TV_WRAPPER} {TV_PYTHON} {input.script} report -i {input.hists} \
            --datasets {TV_DATASETS} --variants {TV_VARIANTS} --years {TV_YEARS} -o {TV_OUT}trig_validation 2>&1 | tee {log}
        """

rule all_trig_validation:
    input:
        ([f"{TV_OUT}trig_validation.txt"] + [f"{TV_OUT}plots_{d}/{y}/plots_done.txt"
                                             for d in TV_DATASETS for y in TV_PLOT_YEARS])
        if TV_ENABLED else []

localrules: TV_config, TV_rename, TV_merge, TV_plot_config, TV_plots, TV_report, all_trig_validation
