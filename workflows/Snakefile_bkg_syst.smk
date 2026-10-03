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
#   │ STAGE A: Closure subsamples from a mixeddata roast (CPU on cmslpc)     │
#   │   (inputs.mixeddata_all / ttbar_psdata: a MakeMixedData roast handoff) │
#   │ - A_1: Mixed-data JCM fit in the analysis selection (inputs.jcm_hists) │
#   │ - A_2: Split into N independent subsamples (v0..vN-1) + ttbar psdata   │
#   │ - A_3: Process unweighted 4-tag subsamples                             │
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
#   - `all_bkg_syst_A`: Run all of Stage A (A_1 through A_3)
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

phase_e_cfg = resolve_config_section(config, primary_key='phase_e', fallback_keys=['phase_e_fvt', 'phaseE', 'closure'])
for k, v in phase_e_cfg.items():
    config.setdefault(k, v)

config.setdefault('label', "ttHbb_bkg_syst")
config.setdefault('output_path', "output/ttHbb_bkg_syst/")
config.setdefault('mix_name', "ttHbb_bkg_syst")
config.setdefault('classifier', "SvB_MA")
config.setdefault('variable', "SvB_MA_ps_ttHbb")
config.setdefault('channel', "ttHbb")
config.setdefault('rebin', "1")

mix_name = config['mix_name']
classifier = config['classifier']
rebin_str = f"rebin{config['rebin']}"
channel = config['channel']
var = config['variable']

# Sub-workflows
include: "Snakefile_bkg_syst_A_1_mixed_jcm.smk"
include: "Snakefile_bkg_syst_A_2_make_subsamples.smk"
include: "Snakefile_bkg_syst_A_3_process_subsamples.smk"
include: "Snakefile_bkg_syst_B_1_computeJCM.smk"
include: "Snakefile_bkg_syst_C.smk"
include: "Snakefile_bkg_syst_F.smk"

# Phase A aggregate target (A_1 through A_3)
rule all_bkg_syst_A:
    input:
        rules.all_bkg_syst_A_1.input,
        rules.all_bkg_syst_A_2.input,
        rules.all_bkg_syst_A_3.input

# Pre-FvT master target rule (A_1 through A_3, B_1, and the handoff C reads on the GPU host)
rule all_pre_fvt:
    input:
        rules.all_bkg_syst_A.input,
        rules.all_bkg_syst_B_1.input,
        rules.bkg_syst_AB_handoff.output

# Top master target rule (default_target: the C files mark their own all_* rules as default too)
rule all_bkg_syst:
    default_target: True
    input:
        rules.all_bkg_syst_A.input,
        rules.all_bkg_syst_B_1.input,
        rules.all_bkg_syst_C.input,
        rules.all_bkg_syst_F.input

localrules: all_bkg_syst, all_bkg_syst_A, all_pre_fvt, test_v0_jcm_closure, all_test_subsample_closure, all_bkg_syst_C_1
