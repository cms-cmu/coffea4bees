# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst.smk
#
# Master Coordinator for the Complete Background Systematics Workflow
# (Mixed Data Creation, Subsamples, JCM, FvT Neural Networks, Closure & Stats)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Coordinates the end-to-end background systematics pipeline for the ttH(bb)
# analysis across all stages:
#
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE A: Mixed-Data Ensemble Generation (CPU on cmslpc)                │
#   │ - A_1: Make mixed data events (3b + 1b permutation)                   │
#   │ - A_2: Make ttbar pseudo-data for ttbar subtraction                    │
#   │ - A_3: Partition into 16 statistically independent subsamples (v0..v15)│
#   │ - A_4: Process unweighted 4-tag subsamples                             │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE B: Jet Combinatoric Model Calibration (CPU on cmslpc)            │
#   │ - B_1: Fit 16 dedicated JCM transfer functions (v0..v15) in SB region  │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE C: FvT Neural Network Classifier Pipeline (Falcon / Bridges-2)   │
#   │ - C_1: Stage training inputs per subsample (cmslpc)                    │
#   │ - C_2: Train 16 independent FvT models on GPUs (falcon / bridges2)     │
#   │ - C_3: High-throughput evaluation & ROOT friend tree generation (GPUs) │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE F: Background Analysis, Two-Stage Closure & Combine Stats        │
#   │ - F_1: 3-tag Data background processing & closure verification         │
#   │ - F_2: Two-stage closure fit (Variance, Bias, Spurious signal)         │
#   │ - F_3: Blinded Combine statistical interpretation (limits, scans)      │
#   │ - F_4: Unblinded pseudo-data interpretation on average mixed data      │
#   └────────────────────────────────────────────────────────────────────────┘
#
# EXECUTION TARGETS:
#   - `all_bkg_syst_A`: Run all of Stage A (A_1 through A_4)
#   - `all_bkg_syst_B_1`: Run Stage B_1 JCM calibration
#   - `all_bkg_syst_C_1`, `all_bkg_syst_C_2`, `all_bkg_syst_C_3`: Stage C
#   - `all_bkg_syst_F_1`, `all_bkg_syst_F_2`, `all_bkg_syst_F_3`, `all_bkg_syst_F_4`: Stage F
#   - `all_bkg_syst_F`: Run all Stage F sub-workflows
#   - `all_bkg_syst`: Run the entire end-to-end background systematics pipeline
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/common.smk"

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_bkg_syst")
config.setdefault('output_path', "output/ttHbb_bkg_syst/")
config.setdefault('mix_name', "ttHbb_bkg_syst")
config.setdefault('classifier', "SvB_MA")
config.setdefault('variable', "SvB_MA_ps_ttHbb")
config.setdefault('channel', "ttHbb")
config.setdefault('rebin', "1")

out = config['output_path']
if not out.endswith("/"):
    out += "/"

out_a1 = f"{out}bkg_syst_A_1_make_mixeddata/"
out_a2 = f"{out}bkg_syst_A_2_make_ttbar_psdata/"
out_a3 = f"{out}bkg_syst_A_3_make_subsamples/"
out_a4 = f"{out}bkg_syst_A_4_process_subsamples/"
out_b1 = f"{out}bkg_syst_B_1_computeJCM/"
out_c1 = f"{out}bkg_syst_C_1_inputs/"
out_c  = f"{out}bkg_syst_C_FvT/"
out_f1 = f"{out}bkg_syst_F_1_analysis/"
out_f2 = f"{out}bkg_syst_F_2_run_two_stage_closure/"
out_f3 = f"{out}bkg_syst_F_3_stats/"
out_f4 = f"{out}bkg_syst_F_4_stats_mixeddata/"

mix_name = config['mix_name']
classifier = config['classifier']
rebin_str = f"rebin{config['rebin']}"
channel = config['channel']
var = config['variable']

# Sub-workflows
include: "Snakefile_bkg_syst_A_1_make_mixeddata.smk"
include: "Snakefile_bkg_syst_A_2_make_ttbar_psdata.smk"
include: "Snakefile_bkg_syst_A_3_make_subsamples.smk"
include: "Snakefile_bkg_syst_A_4_process_subsamples.smk"
include: "Snakefile_bkg_syst_B_1_computeJCM.smk"
include: "Snakefile_bkg_syst_C.smk"
include: "Snakefile_bkg_syst_F.smk"

# Phase A aggregate target (A_1 through A_4)
rule all_bkg_syst_A:
    input:
        rules.all_bkg_syst_A_1.input,
        rules.all_bkg_syst_A_2.input,
        rules.all_bkg_syst_A_3.input,
        rules.all_bkg_syst_A_4.input

# Pre-FvT master target rule (A_1, A_2, A_3, A_4, and B_1)
rule all_pre_fvt:
    input:
        rules.all_bkg_syst_A_1.input,
        rules.all_bkg_syst_A_2.input,
        rules.all_bkg_syst_A_3.input,
        rules.all_bkg_syst_A_4.input,
        rules.all_bkg_syst_B_1.input

# Top master target rule
rule all_bkg_syst:
    input:
        rules.all_bkg_syst_A.input,
        rules.all_bkg_syst_B_1.input,
        rules.all_bkg_syst_C.input,
        rules.all_bkg_syst_F.input

localrules: all_bkg_syst, all_bkg_syst_A, all_pre_fvt, test_v0_jcm_closure, all_test_subsample_closure, all_bkg_syst_C_1
