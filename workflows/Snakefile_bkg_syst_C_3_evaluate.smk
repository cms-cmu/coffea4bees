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
    configfile: "coffea4bees/workflows/config/ttHbb_bkg_syst.yml"

include: "helpers/bkg_syst_common.smk"

n_models = int(config.get('n_models', config.get('n_subsamples', 16)))
subsample_indices = config.get('subsample_indices', list(range(n_models)))
if isinstance(subsample_indices, str):
    subsample_indices = [int(x) for x in subsample_indices.split()]
MIX_INDICES = [int(x) for x in subsample_indices]

out_c = f"{out}bkg_syst_C_FvT/"
os.makedirs(out_c, exist_ok=True)

if not globals().get("_BKG_SYST_C_MODEL_CONFIG_INCLUDED", False):
    _BKG_SYST_C_MODEL_CONFIG_INCLUDED = True
    include: "helpers/bkg_syst_c_model_config.smk"

eos_base = config['eos_base']                     # set by bkg_syst_common (publish_base)
mix_name = config.get("mix_name", "ttHbb_bkg_syst")
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
            if [ -z "${{X509_USER_PROXY:-}}" ] && [ -f ./proxy/x509_proxy ]; then
                export X509_USER_PROXY="$PWD/proxy/x509_proxy"
            fi
            if [ -d /workspace/shared/certificates ] && [ ! -d /etc/grid-security/certificates ]; then
                mkdir -p /etc/grid-security && ln -sfn /workspace/shared/certificates /etc/grid-security/certificates
            fi
            export X509_CERT_DIR=/etc/grid-security/certificates
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
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p $(dirname {output.done}) $(dirname {log})
        echo "=== Background Systematics Stage C Handoff $(date) ===" > {log}
        if [ -n "{params.eos}" ]; then
            echo "Publishing friend manifests to EOS handoff: {params.eos}/friends" >> {log}
            for f in {input}; do
                dst="{params.eos}/friends/$(basename $f)"
                if command -v xrdcp &>/dev/null; then
                    xrdcp -f "$f" "$dst" 2>&1 | tee -a {log}
                elif [[ "$dst" == root://* ]]; then
                    # no xrdcp (the Nautilus orchestrator image): fsspec-xrootd; `cp` cannot write root://
                    python3 -c 'import os, sys, fsspec
fs, path = fsspec.core.url_to_fs(sys.argv[2])
fs.makedirs(os.path.dirname(path), exist_ok=True)
with fs.open(path, "wb") as out, open(sys.argv[1], "rb") as src:
    out.write(src.read())' "$f" "$dst" 2>&1 | tee -a {log}
                else
                    cp -f "$f" "$dst" 2>&1 | tee -a {log}
                fi
            done
        else
            echo "No handoff.eos_base configured; local friend manifests only" >> {log}
        fi
        touch {output.done}
        """

# ── FvT on the SYNTHETIC four-tag sample (C_4 closure subsample[s]) ───────────
# Each bkg_syst FvT is trained (--no-detector-4b) against its subsample's synthetic four-tag events, so
# its closure is checked against those events (Stage C_4), which needs the FvT evaluated on them too.
# The detector evaluation above is unchanged (F_1 uses it). For the closure subsample(s) only
# (config `synthetic_eval_subsamples`, default: the C_4 closure subsample):
#   wfs/evaluate.yml + wfs/train.yml -> wfs/evaluate_synthetic.yml (data source = the subsample's synthetic
#   events, with their classifier-input friends; friend output <FvT>_synthetic) -> evaluate_synthetic ->
#   friends_synthetic/ manifest -> published to <handoff>/friends_synthetic/ with the detector manifest,
#   so C_4 can run on another host as soon as the closure subsample is done.
SYNTH_INDICES = [int(x) for x in (config.get('synthetic_eval_subsamples')
                                  or [config.get('closure_subsample', MIX_INDICES[0] if MIX_INDICES else 0)])]


def _synthetic_eval_wf(eval_wf: dict, train_wf: dict) -> dict:
    """evaluate.yml of one subsample -> its evaluation on the synthetic four-tag sample.

    The dataset options of the FvT evaluation module become those of the training's synthetic part:
    --data-source mixed with the training's --data-mixed-name / --data-mixed-samples, and the
    training's --friends "" (detector + subsample classifier inputs). The friend output gets a
    `_synthetic` suffix so the detector friend (F_1's input) is untouched."""
    import copy

    def opts(wf, section):
        for mod in wf.get(section) or []:
            if isinstance(mod, dict) and 'option' in mod:
                return mod['option']
        raise ValueError(f"no {section} module with options")

    def flag_of(o):
        return o.split()[0] if isinstance(o, str) and o.startswith("--") else None

    out = copy.deepcopy(eval_wf)
    t_opts = opts(train_wf, 'dataset')
    keep = {flag_of(o): o for o in t_opts if flag_of(o) in ("--data-mixed-name", "--data-mixed-samples", "--friends")}
    missing = {"--data-mixed-name", "--data-mixed-samples"} - set(keep)
    if missing:
        raise ValueError(f"train.yml has no {sorted(missing)}: not a synthetic-trained FvT")
    new = []
    for o in opts(out, 'dataset'):
        f = flag_of(o)
        if f == "--data-source":
            new.append("--data-source mixed")
        elif f in ("--friends", "--data-mixed-name", "--data-mixed-samples"):
            continue
        else:
            new.append(o)
    new += [keep["--data-mixed-name"], keep["--data-mixed-samples"]]
    if "--friends" in keep:
        new.append(keep["--friends"])
    for mod in out['dataset']:
        if isinstance(mod, dict) and 'option' in mod:
            mod['option'] = new
            break
    for mod in out.get('setting') or []:
        for o in (mod.get('option') or []) if isinstance(mod, dict) else []:
            if isinstance(o, dict) and isinstance(o.get('output'), str):
                o['output'] = o['output'].rstrip('/') + "_synthetic"
    return out


rule stage_bkg_syst_eval_synthetic_wf:
    input:
        eval_wf = f"{out_c}models/mix_{{m}}/wfs/evaluate.yml",
        train_wf = f"{out_c}models/mix_{{m}}/wfs/train.yml",
    output:
        f"{out_c}models/mix_{{m}}/wfs/evaluate_synthetic.yml"
    wildcard_constraints:
        m = r"\d+"
    run:
        import yaml
        with open(input.eval_wf) as f:
            ev = yaml.safe_load(f)
        with open(input.train_wf) as f:
            tr = yaml.safe_load(f)
        with open(output[0], 'w') as f:
            yaml.dump(_synthetic_eval_wf(ev, tr), f, default_flow_style=False, sort_keys=False)


rule evaluate_synthetic:
    retries: 0 if config.get("test", False) else config.get("retries", 3)
    input:
        train_flag = f"{out_c}models/mix_{{m}}/train.done",
        eval_wf    = f"{out_c}models/mix_{{m}}/wfs/evaluate_synthetic.yml",
        common     = f"{out_c}models/mix_{{m}}/common.yml",
    output:
        flag = f"{out_c}models/mix_{{m}}/evaluate_synthetic.done",
    log:
        f"{out_c}models/mix_{{m}}/evaluate_synthetic.log",
    wildcard_constraints:
        m = r"\d+"
    container: CLASSIFIER
    # one subsample's synthetic events only: a small GPU job (gpus: Nautilus routing; gres: SLURM)
    resources:
        runtime         = config.get("eval_runtime", 480),
        mem_mb          = config.get("synthetic_eval_mem", 16000),
        gpus            = 1,
        gres            = config.get("eval_gres", default_gpu_gres),
        qos             = config.get("eval_qos", default_gpu_qos),
        slurm_partition = config.get("eval_partition", default_gpu_partition),
        slurm_account   = config.get("slurm_account", default_gpu_account),
    threads: config.get("eval_threads", 8)
    params:
        init                    = INIT,
        classifier_config_paths = CLASSIFIER_CONFIG_PATHS,
        template_str            = "",
    shell:
        EVAL_SHELL


rule extract_friend_manifest_synthetic:
    input:
        f"{out_c}models/mix_{{m}}/evaluate_synthetic.done"
    output:
        json = f"{out_c}friends_synthetic/friends_FvT_{mix_name}_v{{m}}.json"
    wildcard_constraints:
        m = r"\d+"
    container: ANALYSIS_CONTAINER
    params:
        python_bin = python_bin,
        eos_base = eos_base,
        fvt_friend_result = f"{eos_base}/friend/FvT/{mix_name}_v{{m}}_synthetic/result.json",
    shell:
        """
        mkdir -p $(dirname {output.json})
        if [ -z "${{X509_USER_PROXY:-}}" ] && [ -f ./proxy/x509_proxy ]; then
            export X509_USER_PROXY="$PWD/proxy/x509_proxy"
        fi
        if [ -d /workspace/shared/certificates ] && [ ! -d /etc/grid-security/certificates ]; then
            mkdir -p /etc/grid-security && ln -sfn /workspace/shared/certificates /etc/grid-security/certificates
        fi
        export X509_CERT_DIR=/etc/grid-security/certificates
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


rule bkg_syst_C_closure_handoff:
    # the closure subsample(s)' detector + synthetic friend manifests, published as soon as they exist
    # (bkg_syst_C_handoff waits for all subsamples), so C_4 can run on another host during C
    input:
        detector = expand(f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json", m=SYNTH_INDICES),
        synthetic = expand(f"{out_c}friends_synthetic/friends_FvT_{mix_name}_v{{m}}.json", m=SYNTH_INDICES),
    output:
        done = f"{out_c}handoff/bkg_syst_C_closure_handoff.done"
    log:
        f"{out_c}logs/bkg_syst_C_closure_handoff.log"
    params:
        eos = HANDOFF_EOS
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p $(dirname {output.done}) $(dirname {log})
        echo "=== Background Systematics Stage C closure-subsample handoff $(date) ===" > {log}
        put() {{
            if command -v xrdcp &>/dev/null; then
                xrdcp -f "$1" "$2" 2>&1 | tee -a {log}
            elif [[ "$2" == root://* ]]; then
                python3 -c 'import os, sys, fsspec
fs, path = fsspec.core.url_to_fs(sys.argv[2])
fs.makedirs(os.path.dirname(path), exist_ok=True)
with fs.open(path, "wb") as out, open(sys.argv[1], "rb") as src:
    out.write(src.read())' "$1" "$2" 2>&1 | tee -a {log}
            else
                mkdir -p "$(dirname "$2")" && cp -f "$1" "$2" 2>&1 | tee -a {log}
            fi
        }}
        if [ -n "{params.eos}" ]; then
            for f in {input.detector}; do put "$f" "{params.eos}/friends/$(basename $f)"; done
            for f in {input.synthetic}; do put "$f" "{params.eos}/friends_synthetic/$(basename $f)"; done
        else
            echo "No handoff.eos_base configured; local friend manifests only" >> {log}
        fi
        touch {output.done}
        """

localrules: bkg_syst_C_handoff, extract_friend_manifest, stage_bkg_syst_eval_synthetic_wf, extract_friend_manifest_synthetic, bkg_syst_C_closure_handoff

rule all_bkg_syst_C_3:
    default_target: True
    input:
        expand(f"{out_c}models/mix_{{m}}/evaluate.done", m=MIX_INDICES),
        expand(f"{out_c}friends/friends_FvT_{mix_name}_v{{m}}.json", m=MIX_INDICES),
        rules.bkg_syst_C_handoff.output,
        rules.bkg_syst_C_closure_handoff.output
