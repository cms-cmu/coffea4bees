# ==============================================================================
# coffea4bees/workflows/helpers/bkg_syst_c_model_config.smk
# Explicit Snakemake DAG rule for Stage C per-subsample classifier configs
# ==============================================================================

import os
import yaml
import glob
import shutil
import subprocess

eos_base = config['eos_base']
mix_name = config.get("mix_name", "ttHbb_bkg_syst")
phase_c_cfg = config.get('phase_e_fvt', config.get('phaseC', config.get('phase_c', {}))) or {}
nominal_ci = phase_c_cfg.get(
    'nominal_classifier_inputs',
    config.get('nominal_classifier_inputs', 'coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json')
)
config.setdefault('metadata', f"{nominal_ci}@@HCR_input")
train_templates = config.setdefault("train_templates", {})
eval_templates = config.setdefault("eval_templates", {})
models_dict = config.setdefault("models", {})
friends_dict = config.setdefault("friends_dict", {})
for m in SUBSAMPLES:
    mdir = f"{out_c}models/mix_{m}"
    train_templates[mdir] = f"model: {eos_base}/classifier/{mix_name}_v{m}"
    eval_templates[mdir] = f"model: {eos_base}/classifier/{mix_name}_v{m}, FvT: {eos_base}/friend/FvT/{mix_name}_v{m}"
    models_dict[mdir] = f"{eos_base}/classifier/{mix_name}_v{m}"
    friends_dict[mdir] = f"{eos_base}/friend/FvT/{mix_name}_v{m}"

# The classifier's --metadata is A_2's classifier_metadata/ directory, which exists only on the A/B host.
# On a GPU host (roast BKG_C on falcon / Nautilus) fetch the copy bkg_syst_AB_handoff put on EOS; the
# YAMLs list files by root:// URL, so they work from anywhere. Without it the trainer silently finds
# no dataset metadata and fails with "Dataset loaded 0 events".
CLASSIFIER_METADATA_DIR = config['classifier_metadata'].rstrip('/')
_FETCH_METADATA = bool(HANDOFF_EOS) and not os.path.isdir(CLASSIFIER_METADATA_DIR)
_METADATA_FETCHED = f"{out_c}handoff/classifier_metadata.fetched"

if _FETCH_METADATA:
    rule fetch_bkg_syst_classifier_metadata:
        output:
            touch(_METADATA_FETCHED)
        params:
            src = f"{HANDOFF_EOS}/classifier_metadata",
            dst = CLASSIFIER_METADATA_DIR,
        run:
            import fsspec
            fs, root = fsspec.core.url_to_fs(params.src)
            # xrootd ls returns bare names: rebuild the full path
            names = [os.path.basename(f.rstrip("/")) for f in fs.ls(root, detail=False)]
            files = [n for n in names if n.endswith((".yml", ".yaml"))]
            if not files:
                raise ValueError(f"no dataset YAMLs in {params.src}: rerun bkg_syst_AB_handoff")
            os.makedirs(params.dst, exist_ok=True)
            for n in files:
                fs.get(f"{root.rstrip('/')}/{n}", os.path.join(params.dst, n))
            print(f"fetched {len(files)} dataset YAMLs {params.src} -> {params.dst}")

    localrules: fetch_bkg_syst_classifier_metadata

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

rule stage_bkg_syst_c_model_config:
    input:
        [_METADATA_FETCHED] if _FETCH_METADATA else [],
    output:
        train = f"{out_c}models/mix_{{m}}/wfs/train.yml",
        eval = f"{out_c}models/mix_{{m}}/wfs/evaluate.yml",
        common = f"{out_c}models/mix_{{m}}/common.yml"
    run:
        m = wildcards.m
        default_subsample_dir = "bkg_syst_A_3_process_subsamples"
        roast_tag = config.get('roast_id', config.get('label', f"{channel}_bkg_syst"))

        # Determine defaults based on EOS handoff or local disk copies
        if HANDOFF_EOS:
            default_mixed_ci = f"{HANDOFF_EOS}/classifier_inputs/histAll_{channel}_mixeddata_v{{m}}.json"
            if per_year_jcm:
                default_jcm = f"{HANDOFF_EOS}/JCM/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml"
            else:
                default_jcm = f"{HANDOFF_EOS}/JCM/jetCombinatoricModel_SB_mix_v{{m}}.yml"
        else:
            local_jcm_dir = f"coffea4bees/metadata/weights/JCM/{roast_tag}"
            local_ci_file = f"coffea4bees/metadata/datasets/{roast_tag}/histAll_{channel}_mixeddata_v{{m}}.json"
            if os.path.exists(local_jcm_dir):
                default_jcm = f"{local_jcm_dir}/jetCombinatoricModel_SB_mix_v{{m}}_{{year}}.yml" if per_year_jcm else f"{local_jcm_dir}/jetCombinatoricModel_SB_mix_v{{m}}.yml"
            else:
                default_jcm = f"{out}bkg_syst_B_1_computeJCM/jetCombinatoricModel_SB_mix_v{{m}}.yml"

            if os.path.exists(os.path.dirname(local_ci_file)):
                default_mixed_ci = local_ci_file
            else:
                default_mixed_ci = f"{out}{default_subsample_dir}/histAll_{channel}_mixeddata_v{{m}}.json"

        mixed_ci_template = phase_c_cfg.get('mixed_classifier_inputs_template') or config.get('mixed_classifier_inputs_template') or default_mixed_ci
        jcm_template = phase_c_cfg.get('jcm_template') or config.get('jcm_template') or default_jcm
        jcm_dir = os.path.dirname(jcm_template) if jcm_template else os.path.join(out, "bkg_syst_B_1_computeJCM")

        sub_name = globals().get('SUB_NAME', config.get('multisample_dataset_name', 'mixeddata_4b'))

        meta_dir = config.get('classifier_metadata')
        if not meta_dir or not os.path.isdir(meta_dir):
            meta_dir = f"{out_c}classifier_metadata/"
            if not os.path.isdir(meta_dir) or not os.listdir(meta_dir):
                os.makedirs(meta_dir, exist_ok=True)
                repo = config.get('dataset_location', "coffea4bees/metadata/datasets/")
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
                    elif HANDOFF_EOS:
                        xrdcp_bin = shutil.which("xrdcp")
                        if not xrdcp_bin and os.path.exists("/cvmfs/cms.cern.ch/common/xrdcp"):
                            xrdcp_bin = "/cvmfs/cms.cern.ch/common/xrdcp"
                        if xrdcp_bin:
                            subprocess.run([xrdcp_bin, "-f", "-s", f"{HANDOFF_EOS}/datasets/{sub_name}.yml", target_ds], check=False)
                        else:
                            try:
                                import fsspec
                                with fsspec.open(f"{HANDOFF_EOS}/datasets/{sub_name}.yml", "rb") as fin:
                                    with open(target_ds, "wb") as fout:
                                        fout.write(fin.read())
                            except Exception as e:
                                pass
                if not os.path.exists(target_ds) or os.path.getsize(target_ds) == 0:
                    raise FileNotFoundError(
                        f"Stage C model config: required multisample dataset metadata '{target_ds}' is missing! "
                        f"Checked local '{out_a2}{sub_name}.yml' and EOS '{HANDOFF_EOS}/datasets/{sub_name}.yml'."
                    )

        mapping = {
            "m": str(m),
            "mix": str(m),
            "metadata": meta_dir,
            "mixed_name": sub_name,
            "model": f"{eos_base}/classifier/{mix_name}_v{m}",
            "FvT": f"{eos_base}/friend/FvT/{mix_name}_v{m}",
            "nominal_ci": nominal_ci,
            "mixed_ci": mixed_ci_template.format(m=m),
        }
        for y in YEARS:
            if "{year}" in jcm_template:
                mapping[f"jcm_{y}"] = jcm_template.format(m=m, year=y)
            elif per_year_jcm:
                mapping[f"jcm_{y}"] = f"{jcm_dir}/jetCombinatoricModel_SB_mix_v{m}_{y}.yml"
            else:
                mapping[f"jcm_{y}"] = f"{jcm_dir}/jetCombinatoricModel_SB_mix_v{m}.yml"

        raw_train_wf = config.get("fvt_train_workflow", config.get("train_workflow", {}))
        raw_eval_wf = config.get("fvt_eval_workflow", config.get("eval_workflow", {}))

        common_cfg = config.get("classifier_setting", [
            {
                "module": "ml.Training",
                "option": [{"precision": config.get("precision", "")}]
            }
        ])
        if isinstance(common_cfg, list):
            common_data = {"setting": common_cfg}
        elif isinstance(common_cfg, dict):
            common_data = common_cfg if "setting" in common_cfg else {"setting": common_cfg}
        else:
            common_data = {"setting": []}

        os.makedirs(os.path.dirname(output.train), exist_ok=True)
        os.makedirs(os.path.dirname(output.eval), exist_ok=True)
        os.makedirs(os.path.dirname(output.common), exist_ok=True)

        with open(output.train, 'w') as f:
            yaml.dump(_format_structure(raw_train_wf, mapping), f, default_flow_style=False, sort_keys=False)
        with open(output.eval, 'w') as f:
            yaml.dump(_format_structure(raw_eval_wf, mapping), f, default_flow_style=False, sort_keys=False)
        with open(output.common, 'w') as f:
            yaml.dump(common_data, f, default_flow_style=False, sort_keys=False)

localrules: stage_bkg_syst_c_model_config
