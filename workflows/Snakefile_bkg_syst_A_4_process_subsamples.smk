# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_4_process_subsamples.smk
#
# Stage A_4: Unified Subsample Processing (Single Pass per Subsample)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Evaluates each mixed-data subsample (mixeddata_4b:v0 .. v14) in a unified single
# pass using processor_{channel}.py. Simultaneously produces:
#   1. Pre-JCM Histograms (histAll_{channel}_mixeddata_v{v}.coffea) containing
#      selJets_noJCM.n / tagJets_noJCM.n distributions required by Stage B_1 (computeJCM).
#   2. Classifier Input trees on EOS & per-subsample JSON manifests, merged into
#      classifier_inputs_mixeddata_{channel}.json required by Stage C_1 (FvT).
#   3. SvB Friend Trees on EOS & merged friend manifest (friends_{channel}_mixeddata_4b.json)
#      required by Stage F_1 (analysis).
#
# INPUTS:
#   - Master multi-sample registry: coffea4bees/metadata/datasets/mixeddata_4b.yml (From A_3)
#   - Staged runtime config: {out_a4}configs/process_subsamples_mixeddata_{channel}.yml
#
# OUTPUTS:
#   - Pre-JCM histograms: {out_a4}histAll_{channel}_mixeddata_v{v}.coffea
#   - Per-subsample friend JSONs: {out_a4}histAll_{channel}_mixeddata_v{v}.json
#   - Merged classifier inputs: {out_a4}classifier_inputs/classifier_inputs_mixeddata_{channel}.json
#   - Merged friend manifest: {mixeddata_friend_json}
#
# USAGE:
#   snakemake -s Snakefile_bkg_syst_A_4_process_subsamples.smk \
#             --configfile config/analysis_ttHbb_bkg_syst.yml -np all_bkg_syst_A_4
# ==============================================================================

import os

_SNAKEFILE_PROCESS_SUBSAMPLES_INCLUDED = True
_SNAKEFILE_BKG_SYST_A_4_INCLUDED = True

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

# ── Stage A_4 Runtime Config Staging (Generated into {out_a4}configs/) ────────
from helpers.stage_configs import stage_phaseA_4_configs
cfg_files = stage_phaseA_4_configs(config, out_a4)

ci_json = config['classifier_inputs_json']
friend_json = config['mixeddata_friend_json']

localrules: all_bkg_syst_A_4, all_bkg_syst_A_4_alias, all_subsample_coffea, all_classifier_inputs_mixeddata, all_friends_mixeddata, merge_classifier_inputs_mixeddata_json, merge_mixeddata_friends_json

# ── Master Target Rules ───────────────────────────────────────────────────────
rule all_bkg_syst_A_4:
    input:
        expand(f"{out_a4}histAll_{channel}_mixeddata_v{{v}}.coffea", v=SUBSAMPLES),
        ci_json,
        friend_json,

rule all_bkg_syst_A_4_alias:
    input:
        rules.all_bkg_syst_A_4.input

rule all_subsample_coffea:
    input:
        expand(f"{out_a4}histAll_{channel}_mixeddata_v{{v}}.coffea", v=SUBSAMPLES)

rule all_classifier_inputs_mixeddata:
    input:
        ci_json

rule all_friends_mixeddata:
    input:
        friend_json

# ── Unified Single-Pass Subsample Processing ───────────────────────────────────
rule process_subsample_single_pass:
    input:
        ds_file = get_multisample_dataset_file,
        cfg = cfg_files['process_subsamples'],
    output:
        coffea = f"{out_a4}histAll_{channel}_mixeddata_v{{v}}.coffea",
        json_meta = f"{out_a4}histAll_{channel}_mixeddata_v{{v}}.json",
    log:
        f"{out_a4}logs/process_subsample_v{{v}}.log"
    params:
        processor = config.get('analysis_processor') or (config.get('analysis_config', {}) or {}).get('processor') or f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = lambda wildcards: f"{config['multisample_dataset_name']}:{wildcards.v}",
        output_path = out_a4,
        years = " ".join(YEARS),
        container_wrapper = container_wrapper,
        condor_flags = condor_flags,
        python_bin = python_bin,
        weights_file = config.get('weights_file', f"coffea4bees/metadata/weights/weights_{channel}.yml"),
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.coffea}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} runner.py {input.cfg} \
            --processor {params.processor} \
            --metadata {input.ds_file} \
            --datasets {params.dataset} \
            --years {params.years} \
            --weights {params.weights_file} \
            --output-path {params.output_path} \
            --output $(basename {output.coffea}) \
            {params.condor_flags} 2>&1 | tee {log}
        """

# ── Merge Classifier Inputs JSON Manifest ─────────────────────────────────────
rule merge_classifier_inputs_mixeddata_json:
    input:
        jsons = expand(f"{out_a4}histAll_{channel}_mixeddata_v{{v}}.json", v=SUBSAMPLES),
    output:
        target_json = ci_json,
        done = f"{out_a4}classifier_inputs/merge_all_subsamples.done",
    params:
        nominal_ci = config.get(
            "nominal_classifier_inputs",
            "coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json"
            if channel == "ttHbb" else
            "coffea4bees/metadata/datasets/classifier_inputs.json"
        ),
        python_bin = python_bin,
    log:
        f"{out_a4}logs/merge_classifier_inputs_mixeddata_json.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.target_json}) $(dirname {output.done}) $(dirname {log})
        nom_arg=""
        if [ -n "{params.nominal_ci}" ] && [ -f "{params.nominal_ci}" ]; then
            nom_arg="--nominal-ci {params.nominal_ci}"
        fi
        {params.python_bin} coffea4bees/hemisphere_mixing/merge_classifier_inputs.py \
            --inputs {input.jsons} \
            --output-json {output.target_json} \
            --output-done {output.done} \
            $nom_arg 2>&1 | tee {log}
        """

# ── Merge Friend Trees Metadata JSON ──────────────────────────────────────────
rule merge_mixeddata_friends_json:
    input:
        jsons = expand(f"{out_a4}histAll_{channel}_mixeddata_v{{v}}.json", v=SUBSAMPLES),
    output:
        target_json = friend_json,
        done = f"{out_a4}friends/merge_all_friends.done",
    params:
        container_wrapper = container_wrapper,
        python_bin = python_bin,
    log:
        f"{out_a4}logs/merge_mixeddata_friends_json.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.target_json}) $(dirname {output.done}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} coffea4bees/hemisphere_mixing/merge_subsample_friends.py \
            --inputs {input.jsons} \
            --output-json {output.target_json} \
            --output-done {output.done} 2>&1 | tee {log}
        """

