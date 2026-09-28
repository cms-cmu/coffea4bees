# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C.smk
#
# Stage C: Master Coordinator for the FvT Neural Network Classifier Pipeline
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Master coordinator for Stage C of the ttH(bb) background systematics measurement.
# Stage C trains and evaluates 16 independent FvT (Four-versus-Three tag)
# neural network models (mix_0..mix_15) across the mixed data subsamples.
#
# Stage C is strictly decomposed into 3 decoupled workflows to respect cluster boundaries:
#
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE C_1: HCR Training Inputs Generation (CPU on cmslpc / HTCondor)   │
#   │ File: Snakefile_bkg_syst_C_1_inputs.smk                                │
#   │ Rules: stage_nominal_inputs, stage_mixeddata_subsample_inputs          │
#   │ Outputs: classifier_inputs_ttHbb.json, inputs_mixeddata_v{m}.json      │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       │ (Inputs synced to falcon via Mutagen)
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE C_2: FvT Neural Network Training (GPU on falcon / bridges2)       │
#   │ File: Snakefile_bkg_syst_C_2_train.smk                                 │
#   │ Rules: create_fvt_train_config, train_fvt_subsample                    │
#   │ Features: Dedicated JCM per subsample, mixeddata_4b (subsample+PSttbar)│
#   │ Outputs: models/mix_{m}/model.pt, train.done                           │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       │
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE C_3: High-Throughput Model Evaluation (GPU on falcon / bridges2) │
#   │ File: Snakefile_bkg_syst_C_3_evaluate.smk                              │
#   │ Rules: evaluate_fvt_subsample, extract_fvt_friend_manifest             │
#   │ Outputs: ROOT Friend Trees on EOS, friends_FvT_ttHbb_bkg_syst_v{m}.json│
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       │ (Friend JSONs synced to cmslpc)
#                                       ▼
#         Downstream: Stage F_1 (analysis & closure check on cmslpc)
#
# MULTI-CLUSTER EXECUTION INSTRUCTIONS:
#
#   Step 1 (Generate/stage inputs on cmslpc):
#     ssh cmslpc-el9.fnal.gov "bash -l -c 'cd <repo> && ./run_container snakemake \
#       -s coffea4bees/workflows/Snakefile_bkg_syst_C_1_inputs.smk --cores 4 all_bkg_syst_C_1'"
#
#   Step 2 (Train models on falcon GPUs):
#     ssh falcon_tgomezes "bash -l -c 'cd <repo> && ./run_container snakemake \
#       -s coffea4bees/workflows/Snakefile_bkg_syst_C_2_train.smk --cores 4 all_bkg_syst_C_2'"
#
#   Step 3 (Evaluate friend trees on falcon GPUs):
#     ssh falcon_tgomezes "bash -l -c 'cd <repo> && ./run_container snakemake \
#       -s coffea4bees/workflows/Snakefile_bkg_syst_C_3_evaluate.smk --cores 4 all_bkg_syst_C_3'"
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

n_models = int(config.get('n_models', config.get('n_subsamples', 16)))
MIX_INDICES = list(range(n_models))

mix_name = config.get('mix_name', "ttHbb_bkg_syst")
out_c1 = f"{out}bkg_syst_C_1_inputs/"
out_c  = f"{out}bkg_syst_C_FvT/"

# Include the modular sub-workflows
include: "Snakefile_bkg_syst_C_1_inputs.smk"
include: "Snakefile_bkg_syst_C_2_train.smk"
include: "Snakefile_bkg_syst_C_3_evaluate.smk"

localrules: all_bkg_syst_C, all_bkg_syst_C_inputs, all_bkg_syst_C_train, all_bkg_syst_C_eval

# ── Master Pipeline Target ────────────────────────────────────────────────────
rule all_bkg_syst_C:
    input:
        rules.all_bkg_syst_C_1.input,
        rules.all_bkg_syst_C_2.input,
        rules.all_bkg_syst_C_3.input

# ── Sub-Stage Aliases ─────────────────────────────────────────────────────────
rule all_bkg_syst_C_inputs:
    input:
        rules.all_bkg_syst_C_1.input

rule all_bkg_syst_C_train:
    input:
        rules.all_bkg_syst_C_2.input

rule all_bkg_syst_C_eval:
    input:
        rules.all_bkg_syst_C_3.input

