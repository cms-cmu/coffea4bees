# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_C_3_evaluate.smk
#
# Stage C_3: High-Throughput FvT Model Evaluation & Friend Tree Generation (GPU)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Evaluates the 16 trained FvT models (mix_0..mix_15) over 3-tag collision data
# to produce per-event reweighting friend trees for the ttH(bb) background
# systematics pipeline.
#
# FRIEND TREE SCHEMA & BRANCHES:
# The evaluation computes neural network classifier predictions for each event:
#   - `FvT`: Primary multijet transfer weight (ratio of 4b to 3b probability).
#   - `d3_to_t4`: Probability transfer score mapping 3b multijet to 4b ttbar.
#   - `d3_to_t3`: Probability transfer score mapping 3b multijet to 3b ttbar.
#
# EVALUATION PIPELINE:
#   1. Configuration Staging:
#      - Stages concrete per-subsample workflow configs (train.yml, evaluate.yml,
#        common.yml) via helpers.stage_configs.stage_phaseC_configs.
#   2. Batched Inference (`src.classifier.task.main`):
#      - Evaluates the trained checkpoint over 3b Data across Run 2 eras.
#      - Streams resulting branches directly into ROOT friend trees on EOS:
#        {eos_base}/friend/FvT/{mix_name}_v{m}/
#   3. Friend Manifest Extraction (`extract_friend_manifest.py`):
#      - Parses the evaluation `result.json` and produces a clean JSON manifest:
#        {out_c}friends/friends_FvT_{mix_name}_v{m}.json
#      - This JSON is subsequently used by coffea processors in Stage F to
#        attach the friend tree during histogram filling.
#
# INPUTS:
#   - Trained Model Checkpoint: {out_c}models/mix_{m}/train.done (From C_2)
#   - Evaluation Workflow: {out_c}models/mix_{m}/wfs/evaluate.yml & common.yml
#
# OUTPUTS:
#   - Friend Tree ROOT Files on EOS: {eos_base}/friend/FvT/{mix_name}_v{m}/*.root
#   - Friend Manifest JSON: {out_c}friends/friends_FvT_{mix_name}_v{m}.json
#   - Completion Tokens: {out_c}models/mix_{m}/evaluate.done
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
MIX_INDICES = list(range(n_models))

out_c = f"{out}bkg_syst_C_FvT/"
os.makedirs(out_c, exist_ok=True)

from helpers.stage_configs import stage_phaseC_configs
cfg_files = stage_phaseC_configs(config, out_c)

phase_e_fvt = config.get("phase_e_fvt", {})
eos_base = config.get("eos_base", phase_e_fvt.get("eos_base", "root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/Run2_v2"))
mix_name = config.get("mix_name", phase_e_fvt.get("mix_name", "ttHbb_bkg_syst"))
container_wrapper = config.get('container_wrapper', './run_container')
python_bin = config.get('python_bin', 'python3')

if not globals().get("_CLASSIFIER_WORKFLOW_INCLUDED", False):
    _CLASSIFIER_WORKFLOW_INCLUDED = True
    include: "../../src/classifier/workflow/Snakefile"

# ── Stage FvT Friend Tree JSON Manifest for Downstream Analysis ───────────────
if not globals().get("_EXTRACT_FRIEND_MANIFEST_INCLUDED", False):
    _EXTRACT_FRIEND_MANIFEST_INCLUDED = True

    rule extract_friend_manifest:
        input:
            f"{out_c}models/mix_{{m}}/evaluate.done"
        output:
            json = f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json"
        container: ANALYSIS_CONTAINER
        params:
            container_wrapper = container_wrapper,
            python_bin = python_bin,
            eos_base = eos_base,
            fvt_friend_result = f"{eos_base}/friend/FvT/{mix_name}_v{{m}}/result.json",
        shell:
            """
            mkdir -p $(dirname {output.json})
            if [ -z "$X509_USER_PROXY" ] && [ -f ./proxy/x509_proxy ]; then
                export X509_USER_PROXY="$PWD/proxy/x509_proxy"
            fi
            if command -v xrdcp &>/dev/null; then
                TMP_RES=$(mktemp -p . --suffix=.json)
                if [[ "{params.eos_base}" == root://* ]]; then
                    xrdcp -f '{params.fvt_friend_result}' "$TMP_RES"
                else
                    cp -f '{params.fvt_friend_result}' "$TMP_RES"
                fi
                {params.python_bin} coffea4bees/stats_analysis/extract_friend_manifest.py "$TMP_RES" "{output.json}"
                rm -f "$TMP_RES"
            else
                {params.python_bin} coffea4bees/stats_analysis/extract_friend_manifest.py '{params.fvt_friend_result}' '{output.json}'
            fi
            """

rule bkg_syst_C_handoff:
    input:
        expand(f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json", m=MIX_INDICES)
    output:
        done = f"{out_c}handoff/bkg_syst_C_handoff.done"
    log:
        f"{out_c}logs/bkg_syst_C_handoff.log"
    params:
        eos = HANDOFF_EOS
    shell:
        """
        mkdir -p $(dirname {output.done}) $(dirname {log})
        echo "=== Background Systematics Stage C Handoff $(date) ===" > {log}
        if [ -n "{params.eos}" ]; then
            echo "Publishing friend manifests to EOS handoff: {params.eos}/friends" >> {log}
            for f in {input}; do
                dst="{params.eos}/friends/$(basename $f)"
                if command -v xrdcp &>/dev/null; then
                    xrdcp -f "$f" "$dst" 2>&1 | tee -a {log}
                else
                    cp -f "$f" "$dst" 2>&1 | tee -a {log}
                fi
            done
        else
            echo "No handoff.eos_base configured; local friend manifests only" >> {log}
        fi
        touch {output.done}
        """

localrules: bkg_syst_C_handoff

rule all_bkg_syst_C_3:
    default_target: True
    input:
        expand(f"{out_c}models/mix_{{m}}/evaluate.done", m=MIX_INDICES),
        expand(f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json", m=MIX_INDICES),
        rules.bkg_syst_C_handoff.output
