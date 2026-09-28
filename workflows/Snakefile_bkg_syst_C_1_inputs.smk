# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_1_inputs.smk
#
# Stage C_1: HCR Classifier Training Input Generation (CPU on cmslpc)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Generates the HCR (Hierarchical Classifier for Resonances) classifier training
# inputs (ROOT trees and dataset JSON manifests) for the ttH(bb) background
# systematics pipeline across Run 2 eras (UL16_preVFP, UL16_postVFP, UL17, UL18).
#
# For each of the 16 independent pseudo-experiments (v0..v15), the FvT neural
# network trains a reweighting function that maps 3-tag collision data to 4-tag
# background. To achieve this, the training sample requires:
#   1. Baseline Collision Data (3b): Detector 3b events serving as source domain.
#   2. Baseline ttbar MC (3b & 4b): Simulated ttbar events in both 3b and 4b
#      regions, allowing the classifier to learn the ttbar component and separate
#      it from multijet QCD.
#   3. Mixed Data 4b Subsamples (v0..v15): Sliced 4b events where each subsample
#      v{m} is constructed from hemisphere-mixed multijet data PLUS sliced
#      pseudotagged ttbar MC (subsample + PSttbar from mixeddata_4b.yml).
#
# WORKFLOW EXECUTION PIPELINE:
#   1. Baseline Detector Inputs:
#      - Evaluates processor_ttHbb.py over Data (3b) and stitched ttbar MC
#        (TTTo2L2Nu, TTToSemiLeptonic, TTToHadronic) with dump_classifier_inputs.
#      - Writes detector HCR inputs: classifier_inputs_ttHbb.json
#   2. Per-Subsample Mixed-Data Inputs:
#      - Evaluates processor_ttHbb.py over each slice `mixeddata_4b:{m}`
#        (which unifies mixeddata subsample m + PSttbar) with unit weights.
#      - Writes per-subsample HCR manifests:
#        classifier_inputs_mixeddata_ttHbb_v{m}.json
#
# INPUTS:
#   - Detector Datasets: coffea4bees/metadata/datasets/data.yml, TT_stitched.yml
#   - Multi-Sample Mixed Dataset: output/ttHbb_bkg_syst/coffea4bees/metadata/datasets/mixeddata_4b.yml
#   - Friends Manifest: coffea4bees/metadata/friends/friends_ttHbb.yml
#
# OUTPUTS:
#   - Detector Input JSON: {out_c1}inputs/classifier_inputs_ttHbb.json
#   - Per-Subsample Input JSONs: {out_c1}inputs/classifier_inputs_mixeddata_ttHbb_v{m}.json
#
# EXECUTION ENVIRONMENT:
#   - Cluster: cmslpc (CPU only, using `./run_container`)
#   - Batch Scheduler: HTCondor / Dask (--cores 4)
# ==============================================================================

import os
import copy
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# ── Configuration Resolution ──────────────────────────────────────────────────
c1_cfg = config.get('phase_c_inputs', config.get('classifier_inputs', {}))
n_models = int(config.get('n_models', config.get('n_subsamples', 16)))
MIX_INDICES = list(range(n_models))

out_c1 = f"{out}bkg_syst_C_1_inputs/"
os.makedirs(out_c1, exist_ok=True)
inputs_dir = f"{out_c1}inputs/"
os.makedirs(inputs_dir, exist_ok=True)

container_wrapper = config.get('analysis_container_wrapper', config.get('container_wrapper', './run_container'))
python_bin = config.get('python_bin', 'python')
condor_flags = "" if config.get("test", False) else "--shared-dask --condor --worker-memory 4GB"

multisample_ds_yaml = config.get('multisample_dataset_yaml', f"{out}coffea4bees/metadata/datasets/mixeddata_4b.yml")
nominal_ci_json = config.get('nominal_classifier_inputs', "coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json")

localrules: all_bkg_syst_C_1, all_inputs_nominal, all_inputs_mixeddata, stage_nominal_inputs, stage_mixeddata_subsample_inputs

# ── Master Target ─────────────────────────────────────────────────────────────
rule all_bkg_syst_C_1:
    input:
        f"{inputs_dir}classifier_inputs_ttHbb.json",
        expand(f"{inputs_dir}classifier_inputs_mixeddata_ttHbb_v{{m}}.json", m=MIX_INDICES)

rule all_inputs_nominal:
    input:
        f"{inputs_dir}classifier_inputs_ttHbb.json"

rule all_inputs_mixeddata:
    input:
        expand(f"{inputs_dir}classifier_inputs_mixeddata_ttHbb_v{{m}}.json", m=MIX_INDICES)

# ── Stage 1: Stage or Verify Baseline Detector Classifier Inputs ──────────────
rule stage_nominal_inputs:
    input:
        nominal_ci_json
    output:
        f"{inputs_dir}classifier_inputs_ttHbb.json"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output})
        cp {input} {output}
        """

# ── Stage 2: Stage or Generate Classifier Inputs for Mixed Subsample v{m} ─────
# Subsamples contain hemisphere-mixed multijet data + sliced PSttbar MC.
source_mixed_ci_template = config.get(
    'source_mixed_ci_template',
    "coffea4bees/metadata/datasets/classifier_inputs_mixeddata/classifier_inputs_mixeddata_ttHbb_v{m}.json"
)

rule stage_mixeddata_subsample_inputs:
    input:
        lambda w: source_mixed_ci_template.format(m=w.m)
    output:
        f"{inputs_dir}classifier_inputs_mixeddata_ttHbb_v{{m}}.json"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output})
        cp {input} {output}
        """

