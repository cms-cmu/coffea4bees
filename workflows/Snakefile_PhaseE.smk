# coffea4bees/workflows/Snakefile_PhaseE.smk
# Phase E: Master Coordinator for Mixed Data Production, FvT Training, Analysis, and Closure

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_mixeddata")
config.setdefault('output_path', "output/ttHbb_mixeddata_closure/")
config.setdefault('mix_name', "3bDvTMix4bDvT")
config.setdefault('classifier', "SvB_MA")
config.setdefault('variable', "SvB_MA_ps")
config.setdefault('channel', "ttHbb")
config.setdefault('rebin', "1")

out = config['output_path']
if not out.endswith("/"):
    out += "/"
mix_name = config['mix_name']
classifier = config['classifier']
rebin_str = f"rebin{config['rebin']}"
channel = config['channel']
var = config['variable']

# Top master target rule
rule all_PhaseE:
    input:
        f"{out}histAll_{config['label']}.coffea",
        f"{out}plots_comparison/plots_done.txt",
        f"{out}plots_analysis/plots_done.txt",
        f"{out}closure_fits/{mix_name}/{classifier}/{rebin_str}/SR/{channel}/hists_closure_{mix_name}_{var}_{rebin_str}.pkl"

# Sub-workflows
include: "Snakefile_PhaseE_1_make_mixeddata.smk"
include: "Snakefile_PhaseE_2_make_subsamples.smk"
include: "Snakefile_PhaseE_3_make_ttbar_psdata.smk"
include: "Snakefile_PhaseE_4_FvT_training.smk"
include: "Snakefile_PhaseE_5_analysis.smk"
include: "Snakefile_PhaseE_6_closure.smk"

localrules: all_PhaseE
