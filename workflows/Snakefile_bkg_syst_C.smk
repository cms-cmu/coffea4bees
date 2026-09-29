# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C.smk
#
# Stage C: Master Coordinator for the FvT Neural Network Classifier Pipeline
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Master coordinator for Stage C of the ttH(bb) background systematics measurement.
# Stage C trains, evaluates, and validates 16 independent FvT (Four-versus-Three tag)
# neural network models (mix_0..mix_15) across the mixed data subsamples.
#
# Stage C is decomposed into 3 decoupled workflows to respect cluster boundaries:
#
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE C_1: Feature Inspection, Validation Plots & Analysis             │
#   │ File: Snakefile_bkg_syst_C_1_inputs.smk                                │
#   │ Rules: analyze, plot_inputs_raw, plot_inputs_dataprep, plot_weights    │
#   │ Outputs: Loss/ROC curves, feature histograms, event weight plots       │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       │
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE C_2: FvT Neural Network Training & Yields (GPU on falcon/bridges)│
#   │ File: Snakefile_bkg_syst_C_2_train.smk                                 │
#   │ Rules: train, yields                                                   │
#   │ Features: Dedicated JCM per subsample, mixeddata_4b (subsample+PSttbar)│
#   │ Outputs: models/mix_{m}/train.done, yields.html, yields.json           │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       │
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE C_3: High-Throughput Model Evaluation (GPU on falcon / bridges2) │
#   │ File: Snakefile_bkg_syst_C_3_evaluate.smk                              │
#   │ Rules: evaluate, extract_friend_manifest                               │
#   │ Outputs: ROOT Friend Trees on EOS, friends_FvT_ttHbb_bkg_syst_v{m}.json│
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       │ (Friend JSONs synced to cmslpc)
#                                       ▼
#         Downstream: Stage F_1 (analysis & closure check on cmslpc)
#
# MULTI-CLUSTER EXECUTION INSTRUCTIONS:
#
#   Step 1 (Train models & compute yields on falcon GPUs):
#     ssh falcon_tgomezes "bash -l -c 'cd <repo> && ./run_container snakemake \
#       -s coffea4bees/workflows/Snakefile_bkg_syst_C_2_train.smk --cores 4 all_bkg_syst_C_2'"
#
#   Step 2 (Evaluate friend trees on falcon GPUs):
#     ssh falcon_tgomezes "bash -l -c 'cd <repo> && ./run_container snakemake \
#       -s coffea4bees/workflows/Snakefile_bkg_syst_C_3_evaluate.smk --cores 4 all_bkg_syst_C_3'"
#
#   Step 3 (Optional: Inspect feature plots & ROC curves):
#     ssh cmslpc-el9.fnal.gov "bash -l -c 'cd <repo> && ./run_container snakemake \
#       -s coffea4bees/workflows/Snakefile_bkg_syst_C_1_inputs.smk --cores 4 all_bkg_syst_C_1'"
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# Include the modular sub-workflows
include: "Snakefile_bkg_syst_C_1_inputs.smk"
include: "Snakefile_bkg_syst_C_2_train.smk"
include: "Snakefile_bkg_syst_C_3_evaluate.smk"

localrules: all_bkg_syst_C, all_bkg_syst_C_all, all_bkg_syst_C_plots, all_bkg_syst_C_inputs, all_bkg_syst_C_train, all_bkg_syst_C_eval

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_bkg_syst_C:
    default_target: True
    input:
        rules.all_bkg_syst_C_2.input,
        rules.all_bkg_syst_C_3.input

# ── Pipeline Aliases ──────────────────────────────────────────────────────────
rule all_bkg_syst_C_all:
    input:
        rules.all_bkg_syst_C_1.input,
        rules.all_bkg_syst_C_2.input,
        rules.all_bkg_syst_C_3.input

rule all_bkg_syst_C_plots:
    input:
        rules.all_bkg_syst_C_1.input

rule all_bkg_syst_C_inputs:
    input:
        rules.all_bkg_syst_C_1.input

rule all_bkg_syst_C_train:
    input:
        rules.all_bkg_syst_C_2.input

rule all_bkg_syst_C_eval:
    input:
        rules.all_bkg_syst_C_3.input
