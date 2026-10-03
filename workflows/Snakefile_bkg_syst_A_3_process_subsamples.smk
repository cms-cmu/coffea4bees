# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_3_process_subsamples.smk
#
# Stage A_3: Unified Subsample Processing (Single Pass per Subsample)
# ==============================================================================
#
# OVERVIEW & OBJECTIVE:
# Evaluates each mixed-data subsample (mixeddata_4b:v0 .. v15) in a unified single
# pass using processor_{channel}.py. Simultaneously produces:
#   1. Pre-JCM Histograms (histAll_{channel}_mixeddata_v{v}.coffea) containing
#      selJets_noJCM.n / tagJets_noJCM.n distributions required by Stage B_1 (computeJCM).
#   2. Classifier Input trees on EOS & per-subsample JSON manifests, merged into
#      classifier_inputs_mixeddata_{channel}.json required by Stage C_1 (FvT).
#   3. SvB Friend Trees on EOS & merged friend manifest (friends_{channel}_mixeddata_4b.json)
#      required by Stage F_1 (analysis).
#
# INPUTS:
#   - Multi-sample dataset: {out_a2}mixeddata_4b.yml (from Stage A_2)
#   - Staged runtime configs: {out_a3}configs/process_subsamples_mixeddata_{channel}_v{v}.yml
#
# OUTPUTS:
#   - Pre-JCM histograms: {out_a3}histAll_{channel}_mixeddata_v{v}.coffea
#   - Per-subsample friend JSONs: {out_a3}histAll_{channel}_mixeddata_v{v}.json
#   - Merged classifier inputs: {out_a3}classifier_inputs/classifier_inputs_mixeddata_{channel}.json
#   - Merged friend manifest: {mixeddata_friend_json}
#
# USAGE:
#   snakemake -s Snakefile_bkg_syst_A_3_process_subsamples.smk \
#             --configfile config/analysis_ttHbb_bkg_syst.yml -np all_bkg_syst_A_3
# ==============================================================================

import os

_SNAKEFILE_PROCESS_SUBSAMPLES_INCLUDED = True
_SNAKEFILE_BKG_SYST_A_3_INCLUDED = True

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"
if "A2_STUDY" not in globals():            # standalone: A_2 provides the dataset
    include: "Snakefile_bkg_syst_A_2_make_subsamples.smk"

# ── Stage A_3 Runtime Config Staging (Generated into {out_a3}configs/) ────────
from helpers.stage_configs import stage_phaseA_3_configs
# own name: cfg_files is reassigned by later includes (F_1), and input functions run after parsing
A3_CFG_FILES = stage_phaseA_3_configs(config, out_a3, SUBSAMPLES)['process_subsamples']

ci_json = config['classifier_inputs_json']
friend_json = config['mixeddata_friend_json']

localrules: all_bkg_syst_A_3, A3_validation, all_subsample_coffea, all_classifier_inputs_mixeddata, all_friends_mixeddata, merge_classifier_inputs_mixeddata_json, merge_mixeddata_friends_json

# ── Master Target Rules ───────────────────────────────────────────────────────
rule all_bkg_syst_A_3:
    input:
        expand(f"{out_a3}histAll_{channel}_mixeddata_v{{v}}.coffea", v=SUBSAMPLES),
        ci_json,
        friend_json,
        f"{out_a3}validation/index.html",

rule all_subsample_coffea:
    input:
        expand(f"{out_a3}histAll_{channel}_mixeddata_v{{v}}.coffea", v=SUBSAMPLES)

rule all_classifier_inputs_mixeddata:
    input:
        ci_json

rule all_friends_mixeddata:
    input:
        friend_json

# ── Unified Single-Pass Subsample Processing ───────────────────────────────────
rule process_subsample_single_pass:
    input:
        ds_file = MULTISAMPLE_DATASET,
        cfg = lambda w: A3_CFG_FILES[w.v],
    output:
        coffea = f"{out_a3}histAll_{channel}_mixeddata_v{{v}}.coffea",
        json_meta = f"{out_a3}histAll_{channel}_mixeddata_v{{v}}.json",
    log:
        f"{out_a3}logs/process_subsample_v{{v}}.log"
    params:
        processor = config.get('analysis_processor') or (config.get('analysis_config', {}) or {}).get('processor') or f"coffea4bees/analysis/processors/processor_{channel}.py",
        dataset = lambda wildcards: f"{config['multisample_dataset_name']}:{wildcards.v}",
        output_path = out_a3,
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
        test -s {output.json_meta}      # runner's friend manifest (HCR_input + SvB_MA): C and F read it
        """

# ── Merge Classifier Inputs JSON Manifest ─────────────────────────────────────
rule merge_classifier_inputs_mixeddata_json:
    input:
        jsons = expand(f"{out_a3}histAll_{channel}_mixeddata_v{{v}}.json", v=SUBSAMPLES),
    output:
        target_json = ci_json,
        done = f"{out_a3}classifier_inputs/merge_all_subsamples.done",
    params:
        nominal_ci = config.get(
            "nominal_classifier_inputs",
            "coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json"
            if channel == "ttHbb" else
            "coffea4bees/metadata/datasets/classifier_inputs.json"
        ),
        python_bin = python_bin,
    log:
        f"{out_a3}logs/merge_classifier_inputs_mixeddata_json.log"
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
        jsons = expand(f"{out_a3}histAll_{channel}_mixeddata_v{{v}}.json", v=SUBSAMPLES),
    output:
        target_json = friend_json,
        done = f"{out_a3}friends/merge_all_friends.done",
    params:
        container_wrapper = container_wrapper,
        python_bin = python_bin,
    log:
        f"{out_a3}logs/merge_mixeddata_friends_json.log"
    shell:
        """
        set -eo pipefail
        mkdir -p $(dirname {output.target_json}) $(dirname {output.done}) $(dirname {log})
        {params.container_wrapper} {params.python_bin} coffea4bees/hemisphere_mixing/merge_subsample_friends.py \
            --inputs {input.jsons} \
            --output-json {output.target_json} \
            --output-done {output.done} 2>&1 | tee {log}
        """

# ── Validation: do the closure samples look like the 4b data / the nominal model (SvB first)? ──
# points 4b data, stack nominal model (3b x JCM x FvT + its ttbar) -- both from inputs.nominal_hists --
# and the N mixed subsamples (mean, min-max) + one of them, per region (SR, SB) and selection
# (inclusive + validation.cuts).
VAL = config.get('validation') or {}
rule A3_validation:
    input:
        nominal = config['nominal_coffea'],
        mixed = expand(f"{out_a3}histAll_{channel}_mixeddata_v{{v}}.coffea", v=SUBSAMPLES),
    output:
        index = f"{out_a3}validation/index.html",
        yields = f"{out_a3}validation/yields.yml",
    log:
        f"{out_a3}logs/validation.log"
    params:
        hists = " ".join(VAL.get('hists', [f"SvB_MA.ps_{channel}", "SvB_MA.ps", "SvB_MA.tt_vs_mj"])),
        # the nominal model's ttbar: ttHbb's Phase F has TTbar4b_from_d3 (filled as threeTag), no MC
        ttbar = " ".join(VAL.get('model_ttbar', ["TTbar4b_from_d3"])),
        ttbar_tag = VAL.get('model_ttbar_tag', "threeTag"),
        cuts = " ".join(VAL.get('cuts', [])),
        compare = VAL.get('subsample', 0),
        sample_prefix = SUB_PREFIX,
        outdir = f"{out_a3}validation/",
        container_wrapper = container_wrapper,
        python_bin = python_bin,
    shell:
        """
        set -eo pipefail
        mkdir -p {params.outdir} $(dirname {log})
        export MPLCONFIGDIR=/tmp/matplotlib
        {params.container_wrapper} {params.python_bin} coffea4bees/workflows/scripts/bkg_syst_mixed_validation.py \
            --nominal {input.nominal} --mixed {input.mixed} --sample-prefix {params.sample_prefix} \
            --ttbar {params.ttbar} --ttbar-tag {params.ttbar_tag} --cuts {params.cuts} \
            --hists {params.hists} --compare {params.compare} \
            -o {params.outdir} 2>&1 | tee {log}
        """
