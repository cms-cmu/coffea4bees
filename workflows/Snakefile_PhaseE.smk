# coffea4bees/workflows/Snakefile_PhaseE.smk
# Phase E: Master Coordinator for Mixed / Synthetic Data Analysis, Plotting, and Closure

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config[k] = v

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb/closure_studies/")
config.setdefault('mix_name', "3bDvTMix4bDvT")
config.setdefault('classifier', "SvB_MA")
config.setdefault('variable', "SvB_MA_ps")
config.setdefault('channel', "ttHbb")
config.setdefault('rebin', "1")

# Top master target rule
rule all_PhaseE:
    input:
        f"{config['output_path']}histAll_{config['label']}.coffea",
        f"{config['output_path']}cutflow_validation_{config['label']}.txt",
        f"{config['output_path']}plots_comparison/plots_done.txt",
        f"{config['output_path']}plots_analysis/plots_done.txt",
        f"{config['output_path']}closure_fits/{config['mix_name']}/{config['classifier']}/rebin{config['rebin']}/SR/{config['channel']}/hists_closure_{config['mix_name']}_{config['variable']}_rebin{config['rebin']}.pkl"

# Sub-workflows
include: "Snakefile_PhaseE_1_analysis.smk"
include: "Snakefile_PhaseE_2_1_plots_comparison.smk"
include: "Snakefile_PhaseE_2_2_plots_analysis.smk"
include: "Snakefile_PhaseE_2_closure.smk"

localrules: all_PhaseE
