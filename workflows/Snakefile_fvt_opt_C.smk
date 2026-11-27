# ==============================================================================
# coffea4bees/workflows/Snakefile_fvt_opt_C.smk
#
# Phase OPT_C: FvT Optimization Sweep Training & Evaluation (GPU on Nautilus / Falcon)
# Runs all (variant x mix) models in parallel with NO pilot gate.
# ==============================================================================

import glob
import os
import shutil
import subprocess
import yaml

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/ttHbb_fvt_opt_cat_year.yml"

include: "helpers/fvt_opt_common.smk"

eos_base = config["eos_base"]
mix_name = config.get("mix_name", f"{channel}_fvt_opt")
container_wrapper = config.get("container_wrapper", "./run_container")
python_bin = config.get("python_bin", "python3")

phase_c_cfg = config.get("phase_e_fvt", config.get("phaseC", config.get("phase_c", {}))) or {}
nominal_ci = phase_c_cfg.get(
    "nominal_classifier_inputs",
    config.get("nominal_classifier_inputs", "coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json"),
)
config.setdefault("metadata", f"{nominal_ci}@@HCR_input")

train_templates = config.setdefault("train_templates", {})
eval_templates = config.setdefault("eval_templates", {})
models_dict = config.setdefault("models", {})
friends_dict = config.setdefault("friends_dict", {})

for run in OPT_RUNS:
    for m in OPT_MIXES:
        mdir = f"{out_opt_c}models/{run}/mix_{m}"
        train_templates[mdir] = f"model: {eos_base}/classifier/{mix_name}_{run}_v{m}"
        eval_templates[mdir] = f"model: {eos_base}/classifier/{mix_name}_{run}_v{m}, FvT: {eos_base}/friend/FvT/{mix_name}_{run}_v{m}"
        models_dict[mdir] = f"{eos_base}/classifier/{mix_name}_{run}_v{m}"
        friends_dict[mdir] = f"{eos_base}/friend/FvT/{mix_name}_{run}_v{m}"

_handoff_in = config.get("input_handoff_eos") or globals().get("INPUT_HANDOFF_EOS") or HANDOFF_EOS
CLASSIFIER_METADATA_DIR = config["classifier_metadata"].rstrip("/")
_FETCH_METADATA = bool(_handoff_in) and not os.path.isdir(CLASSIFIER_METADATA_DIR)
_METADATA_FETCHED = f"{out_opt_c}handoff/classifier_metadata.fetched"

if _FETCH_METADATA:
    rule fetch_fvt_opt_classifier_metadata:
        output:
            touch(_METADATA_FETCHED)
        params:
            src = f"{_handoff_in}/classifier_metadata",
            dst = CLASSIFIER_METADATA_DIR,
        run:
            import fsspec
            fs, root = fsspec.core.url_to_fs(params.src)
            names = [os.path.basename(f.rstrip("/")) for f in fs.ls(root, detail=False)]
            files = [n for n in names if n.endswith((".yml", ".yaml"))]
            if not files:
                raise ValueError(f"no dataset YAMLs in {params.src}: rerun bkg_syst_AB_handoff")
            os.makedirs(params.dst, exist_ok=True)
            for n in files:
                fs.get(f"{root.rstrip('/')}/{n}", os.path.join(params.dst, n))
            print(f"fetched {len(files)} dataset YAMLs {params.src} -> {params.dst}")

    localrules: fetch_fvt_opt_classifier_metadata

if not globals().get("_CLASSIFIER_WORKFLOW_INCLUDED", False):
    _CLASSIFIER_WORKFLOW_INCLUDED = True
    include: "../../src/classifier/workflow/Snakefile"

wildcard_constraints:
    run = "|".join(OPT_RUNS),
    m = r"\d+",


def _format_structure(obj, mapping):
    if isinstance(obj, str):
        res = obj
        for k, v in mapping.items():
            res = res.replace(f"{{{k}}}", str(v))
        return res
    elif isinstance(obj, dict):
        return {k: _format_structure(v, mapping) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_format_structure(elem, mapping) for elem in obj]
    return obj


rule stage_fvt_opt_model_config:
    input:
        [_METADATA_FETCHED] if _FETCH_METADATA else []
    output:
        train = f"{out_opt_c}models/{{run}}/mix_{{m}}/wfs/train.yml",
        eval = f"{out_opt_c}models/{{run}}/mix_{{m}}/wfs/evaluate.yml",
        common = f"{out_opt_c}models/{{run}}/mix_{{m}}/common.yml",
    run:
        run_id = wildcards.run
        m = wildcards.m
        roast_tag = config.get("roast_id", config.get("label", f"{channel}_fvt_opt"))

        if per_year_jcm:
            has_local_jcm = bool(glob.glob(f"{out_b1}jetCombinatoricModel_SB_mix_v{m}_*.yml"))
        else:
            has_local_jcm = os.path.exists(f"{out_b1}jetCombinatoricModel_SB_mix_v{m}.yml")

        if _handoff_in and not has_local_jcm:
            default_mixed_ci = f"{_handoff_in}/classifier_inputs/histAll_{channel}_mixeddata_v{{m}}.json"
            if per_year_jcm:
                default_jcm = f"{_handoff_in}/JCM/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml"
            else:
                default_jcm = f"{_handoff_in}/JCM/jetCombinatoricModel_SB_mix_v{{m}}.yml"
        else:
            default_jcm = (
                f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml"
                if per_year_jcm
                else f"{out_b1}jetCombinatoricModel_SB_mix_v{{m}}.yml"
            )
            default_mixed_ci = f"{out_a3}histAll_{channel}_mixeddata_v{{m}}.json"

        mixed_ci_template = phase_c_cfg.get("mixed_classifier_inputs_template") or config.get("mixed_classifier_inputs_template") or default_mixed_ci
        for pk, pv in _placeholders.items():
            mixed_ci_template = mixed_ci_template.replace("{" + pk + "}", str(pv))
        jcm_template = phase_c_cfg.get("jcm_template") or config.get("jcm_template") or default_jcm
        for pk, pv in _placeholders.items():
            jcm_template = jcm_template.replace("{" + pk + "}", str(pv))
        jcm_dir = os.path.dirname(jcm_template) if jcm_template else out_b1

        sub_name = globals().get("SUB_NAME", config.get("multisample_dataset_name", "mixeddata_4b"))
        meta_dir = config.get("classifier_metadata")
        if not meta_dir or not os.path.isdir(meta_dir):
            meta_dir = f"{out_opt_c}classifier_metadata/"
            if not os.path.isdir(meta_dir) or not os.listdir(meta_dir):
                os.makedirs(meta_dir, exist_ok=True)
                repo = config.get("dataset_location", "coffea4bees/metadata/datasets/")
                for path in sorted(glob.glob(os.path.join(repo, "*.yml"))):
                    with open(path) as f:
                        keys = set(yaml.safe_load(f) or {})
                    if sub_name in keys:
                        continue
                    shutil.copy(path, meta_dir)
                target_ds = os.path.join(meta_dir, f"{sub_name}.yml")
                if not os.path.exists(target_ds):
                    local_ds = f"{out_a2}{sub_name}.yml"
                    if os.path.exists(local_ds):
                        shutil.copy(local_ds, target_ds)
                    elif _handoff_in:
                        xrdcp_bin = shutil.which("xrdcp") or (
                            "/cvmfs/cms.cern.ch/common/xrdcp" if os.path.exists("/cvmfs/cms.cern.ch/common/xrdcp") else None
                        )
                        if xrdcp_bin:
                            subprocess.run([xrdcp_bin, "-f", "-s", f"{_handoff_in}/datasets/{sub_name}.yml", target_ds], check=False)
                        else:
                            import fsspec
                            with fsspec.open(f"{_handoff_in}/datasets/{sub_name}.yml", "rb") as fin, open(target_ds, "wb") as fout:
                                fout.write(fin.read())

        if os.path.isabs(meta_dir):
            own_dir = f"{out_opt_c}classifier_metadata_local/"
            os.makedirs(own_dir, exist_ok=True)
            for path in sorted(glob.glob(os.path.join(meta_dir, "*.yml"))):
                shutil.copy(path, own_dir)
            meta_dir = own_dir

        data_svb_friend = config.get("data_svb_ma_friend") or config.get("data_svb_friend") or ""
        mapping = {
            "m": str(m),
            "mix": str(m),
            "metadata": meta_dir,
            "mixed_name": sub_name,
            "model": f"{eos_base}/classifier/{mix_name}_{run_id}_v{m}",
            "FvT": f"{eos_base}/friend/FvT/{mix_name}_{run_id}_v{m}",
            "nominal_ci": nominal_ci,
            "mixed_ci": mixed_ci_template.format(m=m),
            "data_svb": data_svb_friend,
        }
        for y in YEARS:
            if "{year}" in jcm_template:
                mapping[f"jcm_{y}"] = jcm_template.format(m=m, year=y)
            elif per_year_jcm:
                mapping[f"jcm_{y}"] = f"{jcm_dir}/jetCombinatoricModel_SB_mix_v{m}_{y}.yml"
            else:
                mapping[f"jcm_{y}"] = f"{jcm_dir}/jetCombinatoricModel_SB_mix_v{m}.yml"

        raw_train_wf = config.get("fvt_train_workflow", config.get("train_workflow", {}))
        formatted_train_wf = _format_structure(raw_train_wf, mapping)
        formatted_train_wf = apply_variant_to_model_wf(formatted_train_wf, OPT_RUN_SPECS[run_id])

        active_years = set(str(y) for y in YEARS)
        for ds in (formatted_train_wf.get("dataset") or []):
            if isinstance(ds, dict) and isinstance(ds.get("option"), list):
                filtered_opts = []
                for o in ds["option"]:
                    if isinstance(o, str) and o.startswith("--JCM-weight year:"):
                        tokens = o.split()
                        yr = tokens[1].split(":", 1)[1] if len(tokens) > 1 and ":" in tokens[1] else ""
                        if yr and yr not in active_years:
                            continue
                    filtered_opts.append(o)
                ds["option"] = filtered_opts

        raw_eval_wf = config.get("fvt_eval_workflow", config.get("eval_workflow", {}))
        common_cfg = config.get("classifier_setting", [
            {"module": "ml.Training", "option": [{"precision": config.get("precision", "")}]}
        ])
        common_data = (
            {"setting": common_cfg}
            if isinstance(common_cfg, list)
            else (common_cfg if isinstance(common_cfg, dict) and "setting" in common_cfg else {"setting": []})
        )

        os.makedirs(os.path.dirname(output.train), exist_ok=True)
        os.makedirs(os.path.dirname(output.eval), exist_ok=True)
        os.makedirs(os.path.dirname(output.common), exist_ok=True)
        with open(output.train, "w") as f:
            yaml.dump(formatted_train_wf, f, default_flow_style=False, sort_keys=False)
        with open(output.eval, "w") as f:
            yaml.dump(_format_structure(raw_eval_wf, mapping), f, default_flow_style=False, sort_keys=False)
        with open(output.common, "w") as f:
            yaml.dump(common_data, f, default_flow_style=False, sort_keys=False)


rule extract_fvt_opt_friend_manifest:
    input:
        f"{out_opt_c}models/{{run}}/mix_{{m}}/evaluate.done",
    output:
        json = f"{out_opt_c}friends/{{run}}/friends_FvT_{mix_name}_{{run}}_v{{m}}.json",
    container: ANALYSIS_CONTAINER
    params:
        python_bin = python_bin,
        eos_base = eos_base,
        fvt_friend_result = f"{eos_base}/friend/FvT/{mix_name}_{{run}}_v{{m}}/result.json",
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


rule fvt_opt_C_handoff:
    input:
        expand(f"{out_opt_c}friends/{{run}}/friends_FvT_{mix_name}_{{run}}_v{{m}}.json", run=OPT_RUNS, m=OPT_MIXES),
    output:
        done = f"{out_opt_c}handoff/fvt_opt_C_handoff.done",
    log:
        f"{out_opt_c}logs/fvt_opt_C_handoff.log",
    params:
        eos = HANDOFF_EOS,
        out_c_friends = f"{out_opt_c}friends",
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p $(dirname {output.done}) $(dirname {log})
        echo "=== fvt_opt OPT_C Handoff $(date) ===" > {log}
        if [ -n "{params.eos}" ]; then
            for f in {input}; do
                rel="${{f#{params.out_c_friends}/}}"
                dst="{params.eos}/friends/$rel"
                if command -v xrdcp &>/dev/null; then
                    xrdcp -f "$f" "$dst" 2>&1 | tee -a {log}
                elif [[ "$dst" == root://* ]]; then
                    python3 -c 'import os, sys, fsspec
fs, path = fsspec.core.url_to_fs(sys.argv[2])
fs.makedirs(os.path.dirname(path), exist_ok=True)
with fs.open(path, "wb") as out, open(sys.argv[1], "rb") as src:
    out.write(src.read())' "$f" "$dst" 2>&1 | tee -a {log}
                else
                    mkdir -p "$(dirname "$dst")"
                    cp -f "$f" "$dst" 2>&1 | tee -a {log}
                fi
            done
        fi
        touch {output.done}
        """


rule fvt_opt_variant_training_summary:
    input:
        expand(f"{out_opt_c}models/{{{{run}}}}/mix_{{m}}/analyze.done", m=OPT_MIXES),
        expand(f"{out_opt_c}models/{{{{run}}}}/mix_{{m}}/gallery.done", m=OPT_MIXES),
        expand(f"{out_opt_c}models/{{{{run}}}}/mix_{{m}}/yields.html", m=OPT_MIXES),
    output:
        html = f"{out_opt_c}training_summary_{{run}}.html",
    params:
        models_dir = f"{out_opt_c}models/{{run}}",
        title = lambda wildcards: f"fvt_opt ({wildcards.run}) Training Summary — {config.get('roast_id', config.get('label', 'fvt_opt'))}",
    shell:
        """
        python3 src/classifier/training_summary.py \
            --models-dir {params.models_dir} \
            --output {output.html} \
            --title "{params.title}"
        """


localrules: all_fvt_opt_C, stage_fvt_opt_model_config, fvt_opt_C_handoff, fvt_opt_variant_training_summary

rule all_fvt_opt_C:
    default_target: True
    input:
        expand(f"{out_opt_c}models/{{run}}/mix_{{m}}/train.done", run=OPT_RUNS, m=OPT_MIXES),
        expand(f"{out_opt_c}models/{{run}}/mix_{{m}}/yields.html", run=OPT_RUNS, m=OPT_MIXES),
        expand(f"{out_opt_c}models/{{run}}/mix_{{m}}/analyze.done", run=OPT_RUNS, m=OPT_MIXES),
        expand(f"{out_opt_c}models/{{run}}/mix_{{m}}/gallery.done", run=OPT_RUNS, m=OPT_MIXES),
        expand(f"{out_opt_c}models/{{run}}/mix_{{m}}/evaluate.done", run=OPT_RUNS, m=OPT_MIXES),
        f"{out_opt_c}handoff/fvt_opt_C_handoff.done",
        expand(f"{out_opt_c}training_summary_{{run}}.html", run=OPT_RUNS),
