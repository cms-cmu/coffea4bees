# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_2_train.smk
#
# Stage C_2: FvT Neural Network Training across 16 Subsamples (GPU on falcon / bridges2)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Trains 16 independent FvT (Four-versus-Three tag) PyTorch neural networks
# (mix_0..mix_15) for the ttH(bb) background systematics evaluation and computes
# the per-class loaded yields tables.
#
# Each model is trained on a dedicated pseudo-experiment realization to evaluate
# background shape and normalization uncertainties. To ensure strict statistical
# independence and physical accuracy:
#   1. Dedicated JCM: Each training mix_v{m} reweights 3-tag collision data
#      using its OWN dedicated JCM transfer function derived in Stage B_1:
#      output/ttHbb_bkg_syst/bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{m}.yml
#   2. Full 4b Target (`mixeddata_4b:{m}`): Each 4b target contains both the
#      hemisphere-mixed multijet subsample AND the sliced pseudotagged ttbar MC
#      (subsample + PSttbar from Stage A_2 / A_3).
#   3. Full Detector Sample: Contains Data 3b, ttbar MC 4b, and ttbar MC 3b
#      (`--data-source detector mixed --no-detector-4b`), allowing the classifier
#      to separate multijet QCD from ttbar and derive both the multijet reweighting
#      and ttbar fraction simultaneously.
#   4. Yield Reports: Produces HTML and JSON yield summaries (rule yields) for
#      each model to verify data/model class counts and normalization ratios.
#
# NEURAL NETWORK ARCHITECTURE & TRAINING DETAILS:
#   - Framework: PyTorch with HCR (Hierarchical Classifier for Resonances).
#   - Inputs: Jet kinematics (pT, eta, phi, mass, deepJet/PNet b-tag scores),
#     candidate quadjet pairings, and event-level variables.
#   - Training Schedule: EarlyStopStep with validation benchmarking.
#   - Precision: Mixed precision (bf16 or fp16).
#   - Hardware Allocation: 1 GPU per training instance (SLURM partition: GPU-shared / work).
#
# INPUTS:
#   - Dedicated JCMs: {out_b1}jetCombinatoricModel_SB_mix_v{m}.yml (From B_1)
#   - Detector Inputs: coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json
#   - Mixed Inputs: {out_a4}histAll_ttHbb_mixeddata_v{m}.json (From A_4)
#   - Staged Wfs: {out_c}models/mix_{m}/wfs/train.yml & common.yml
#
# OUTPUTS:
#   - Trained Model Checkpoints: {out_c}models/mix_{m}/model.pt
#   - Completion Tokens: {out_c}models/mix_{m}/train.done
#   - Yield Summaries:   {out_c}models/mix_{m}/yields.html, yields.json
#
# EXECUTION ENVIRONMENT:
#   - Cluster: falcon (GPU partition work/light) or bridges2 (GPU-shared)
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
rule all_bkg_syst_C_2:
    default_target: True
    input:
        expand(f"{out_c}models/mix_{{m}}/train.done", m=MIX_INDICES),
        expand(f"{out_c}models/mix_{{m}}/yields.html", m=MIX_INDICES),
