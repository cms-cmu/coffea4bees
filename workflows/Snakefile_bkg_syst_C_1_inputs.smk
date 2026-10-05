# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_1_inputs.smk
#
# Stage C_1: Classifier Input Feature Inspection, Validation Plots & Analysis
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Generates validation distributions and performance diagnostics for the HCR
# neural network classifier inputs, feature representations, training loss / ROC
# curves, and output event weights across all 16 background systematic subsamples
# (mix_0..mix_15).
#
# WORKFLOW EXECUTION PIPELINE:
#   1. Configuration Staging:
#      - Stages concrete per-subsample workflow configs (train.yml, evaluate.yml,
#        common.yml) via helpers.stage_configs.stage_phaseC_configs.
#   2. Raw Input Feature Plotting (`rule plot_inputs_raw`):
#      - Evaluates raw kinematic and tagging distributions from friend trees
#        and input picoAODs split by physics processes (d4, d3, t4, t3).
#   3. Embedded DataPrep Plotting (`rule plot_inputs_dataprep`):
#      - Plots normalized embedded feature representations produced by the
#        HCR network's inputEmbed.dataPrep() layer.
#   4. Network Performance & Loss Analysis (`rule analyze`):
#      - Evaluates training loss curves, ROC distributions, and background
#        reweighting fidelity from model checkpoints (result.json).
#   5. Classifier Weight Distributions (`rule plot_weights`):
#      - Compares data/MC weight distributions and closure metrics across
#        control and signal regions.
#
# INPUTS:
#   - Trained Model Checkpoints: {out_c}models/mix_{m}/train.done (From C_2)
#   - Classifier Input Manifests: coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json
#   - Per-subsample Mixed Inputs: {out_a3}histAll_ttHbb_mixeddata_v{m}.json
#
# OUTPUTS:
#   - Analysis ROC/Loss plots: {plot_base}/{DATE}_{label}_mix_{m}/analyze/
#   - Raw Feature Plots:       {plot_base}/{DATE}_{label}_mix_{m}/inputs_raw/
#   - DataPrep Plots:          {plot_base}/{DATE}_{label}_mix_{m}/inputs_dataprep/
#   - Weight Distribution Plots: {plot_base}/{DATE}_{label}_mix_{m}/weights/
#   - Completion Tokens:       {out_c}models/mix_{m}/analyze.done, etc.
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc / falcon / bridges2
#   - Container: /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:classifier_latest
# ==============================================================================

import os

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

n_models = int(config.get('n_models', config.get('n_subsamples', 16)))
subsample_indices = config.get('subsample_indices', list(range(n_models)))
if isinstance(subsample_indices, str):
    subsample_indices = [int(x) for x in subsample_indices.split()]
MIX_INDICES = [int(x) for x in subsample_indices]

out_c = f"{out}bkg_syst_C_FvT/"
os.makedirs(out_c, exist_ok=True)

from helpers.stage_configs import stage_phaseC_configs
cfg_files = stage_phaseC_configs(config, out_c)

if not globals().get("_CLASSIFIER_WORKFLOW_INCLUDED", False):
    _CLASSIFIER_WORKFLOW_INCLUDED = True
    include: "../../src/classifier/workflow/Snakefile"

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_bkg_syst_C_1:
    default_target: True
    input:
        expand(f"{out_c}models/mix_{{m}}/analyze.done", m=MIX_INDICES),
        expand(f"{out_c}models/mix_{{m}}/plot_inputs_raw.done", m=MIX_INDICES),
        expand(f"{out_c}models/mix_{{m}}/plot_inputs_dataprep.done", m=MIX_INDICES),
        expand(f"{out_c}models/mix_{{m}}/plot_weights.done", m=MIX_INDICES),

rule all_bkg_syst_C_1_analyze:
    input:
        expand(f"{out_c}models/mix_{{m}}/analyze.done", m=MIX_INDICES),

rule all_bkg_syst_C_1_plots:
    input:
        expand(f"{out_c}models/mix_{{m}}/plot_inputs_raw.done", m=MIX_INDICES),
        expand(f"{out_c}models/mix_{{m}}/plot_inputs_dataprep.done", m=MIX_INDICES),
        expand(f"{out_c}models/mix_{{m}}/plot_weights.done", m=MIX_INDICES),
