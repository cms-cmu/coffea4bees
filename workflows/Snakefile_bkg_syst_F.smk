# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_F.smk
#
# Stage F: Master Coordinator for Background Systematics Analysis, Closure & Stats
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Master coordinator for Stage F of the ttH(bb) background systematics pipeline.
# Stage F consumes the 16 trained FvT friend tree models (mix_0..mix_15) from
# Stage C, performs full analysis histogramming on 3-tag collision data to
# establish the background model, verifies two-stage closure against 4-tag
# mixed data, extracts systematic uncertainty covariance matrices, and performs
# statistical interpretations (datacard generation, limits, significance).
#
# SUB-WORKFLOW ARCHITECTURE:
#
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE F_1: Data Background Analysis & Closure Plots (CPU on cmslpc)    │
#   │ File: Snakefile_bkg_syst_F_1_analysis.smk                              │
#   │ Target: all_bkg_syst_F_1                                               │
#   │ Outputs: closure_v{m}/histAll_data_v{m}.coffea, closure plots, gallery │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE F_2: Two-Stage Closure Fit & Systematics Extraction (cmslpc)     │
#   │ File: Snakefile_bkg_syst_F_2_run_two_stage_closure.smk                 │
#   │ Target: all_bkg_syst_F_2                                               │
#   │ Outputs: hists_closure_{mix_name}_{var}_rebin{r}.pkl                   │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE F_3: Blinded Combine Statistical Interpretation (cmslpc)         │
#   │ File: Snakefile_bkg_syst_F_3_stats.smk                                 │
#   │ Target: all_bkg_syst_F_3                                               │
#   │ Outputs: datacards, limits, significance, likelihood scans             │
#   └───────────────────────────────────┬────────────────────────────────────┘
#                                       ▼
#   ┌────────────────────────────────────────────────────────────────────────┐
#   │ STAGE F_4: Unblinded Combine Stats on Average Mixed Data (cmslpc)      │
#   │ File: Snakefile_bkg_syst_F_4_stats_mixeddata.smk                       │
#   │ Target: all_bkg_syst_F_4                                               │
#   │ Outputs: datacards and limit scans for average mixed data pseudo-data  │
#   └────────────────────────────────────────────────────────────────────────┘
#
# USAGE INSTRUCTIONS (on cmslpc):
#   Run only Stage F_1 (Analysis & Closure Plots):
#     ./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_F.smk \
#       --configfile coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml --cores 4 all_bkg_syst_F_1
#
#   Run only Stage F_2 (Two-Stage Closure Fit):
#     ./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_F.smk \
#       --configfile coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml --cores 4 all_bkg_syst_F_2
#
#   Run only Stage F_3 (Blinded Combine Statistical Interpretation):
#     ./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_F.smk \
#       --configfile coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml --cores 4 all_bkg_syst_F_3
#
#   Run only Stage F_4 (Unblinded Mixed Data Statistical Interpretation):
#     ./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_F.smk \
#       --configfile coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml --cores 4 all_bkg_syst_F_4
#
#   Run all Stage F:
#     ./run_container snakemake -s coffea4bees/workflows/Snakefile_bkg_syst_F.smk \
#       --configfile coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml --cores 4 all_bkg_syst_F
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# Include modular Stage F sub-workflows
include: "Snakefile_bkg_syst_F_1_analysis.smk"
include: "Snakefile_bkg_syst_F_2_run_two_stage_closure.smk"
include: "Snakefile_bkg_syst_F_3_stats.smk"
include: "Snakefile_bkg_syst_F_4_stats_mixeddata.smk"

localrules: all_bkg_syst_F, all_bkg_syst_F_1_alias, all_bkg_syst_F_2_alias, all_bkg_syst_F_3_alias, all_bkg_syst_F_4_alias

# ── Master Stage F Target ─────────────────────────────────────────────────────
rule all_bkg_syst_F:
    input:
        rules.all_bkg_syst_F_1.input,
        rules.all_bkg_syst_F_2.input,
        rules.all_bkg_syst_F_3.input,
        rules.all_bkg_syst_F_4.input

# ── Stage Aliases ─────────────────────────────────────────────────────────────
rule all_bkg_syst_F_1_alias:
    input:
        rules.all_bkg_syst_F_1.input

rule all_bkg_syst_F_2_alias:
    input:
        rules.all_bkg_syst_F_2.input

rule all_bkg_syst_F_3_alias:
    input:
        rules.all_bkg_syst_F_3.input

rule all_bkg_syst_F_4_alias:
    input:
        rules.all_bkg_syst_F_4.input
