# coffea4bees/workflows/Snakefile_FitQualityReplay.smk
# Fit-quality replay of already-trained FvT / MvD classifiers (config: fit_quality_replay_run3.yml).
#
# For each `fit_quality_replay:` entry: regenerate the upstream roast's training workflow
# (train.yml, common.yml) from its captured config with the code Phase C / MvD V.2 use, then run
# src.classifier.fit_quality_replay on a GPU (NRP Job via the nautilus profile): load the saved
# fold models, zero training epochs, and the Baseline benchmark records the fit quality.
# Upstream roasts are only read. Everything this writes is under config["output_path"].
#
# Targets: all_fit_quality (default: every entry + index.html), fitq_validate (the first entry only,
# to check the replay reproduces the original training before spending GPUs on the rest), or
# <output_path>/<name>/fit_quality.json for one entry.

import filecmp
import json
import os
import shutil
import tempfile

import yaml

include: "helpers/common.smk"

OUT = config.get("output_path", "output/fit_quality_replay/").rstrip("/")
ENTRIES = config.get("fit_quality_replay") or {}
if not ENTRIES:
    raise ValueError("config `fit_quality_replay:` is empty")
ALIASES = upstream_aliases(config)
MODELS = (config.get("inputs") or {}).get("models") or {}
TOLERANCE = float(config.get("fit_quality_tolerance", 2e-3))
FALLBACK_KEYS = {"fvt": ["fvt_classifier", "classifier"]}


def _sync_tree(src, dst):
    """Make dst match src, replacing only files whose content changed (atomically). Every NRP job
    re-parses this Snakefile and so regenerates the workflow; an in-place rewrite could truncate a
    train.yml another job is reading."""
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in files:
            a, b = os.path.join(root, fn), os.path.join(dst, rel, fn)
            if not (os.path.exists(b) and filecmp.cmp(a, b, shallow=False)):
                tmp = f"{b}.tmp.{os.getpid()}"
                shutil.copyfile(a, tmp)
                os.replace(tmp, b)


def regenerate_workflow(name, entry):
    """Write <OUT>/<name>/workflow/{wfs/train.yml, common.yml} from the upstream's captured config.

    Mirrors Snakefile_PhaseC.smk (`fvt:`) and helpers/classifier_step.smk (`mvd:`): the block is
    resolved over its fallbacks, {roast_id}/{eos_prod}/{roast:...} are filled in as the upstream
    saw them, and the templates get the block's workflow_overrides / inserts / modules.
    """
    rid = ALIASES[entry["upstream"]]
    captured = os.path.join("roasts", rid, "config.yml")
    with open(captured) as fh:
        cap = yaml.safe_load(fh) or {}
    area = _upstream_base(rid, config["eos_prod"])            # <its eos_prod>/<rid>
    ph = {**_user_profile(), "roast_id": rid, "eos_prod": area.rsplit("/", 1)[0]}
    cap = substitute_placeholders(cap, ph)
    cap = substitute_upstreams(cap, ph["eos_prod"], {}, upstream_aliases(cap))
    key = entry["block"]
    blk = resolve_config_section(cap, primary_key=key, fallback_keys=FALLBACK_KEYS.get(key, []),
                                 inherit_keys=[])
    if not blk.get("label") or not blk.get("wfs_base"):
        raise ValueError(f"{captured}: `{key}:` has no label / wfs_base")
    merged = {**cap, **blk}
    label = blk["label"]
    eos_base = blk.get("eos_base") or area
    final = os.path.join(OUT, name, "workflow")
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    d = tempfile.mkdtemp(prefix=".workflow.", dir=os.path.join(OUT, name))
    common = None
    if "classifier_setting" in merged:
        common = os.path.join(d, "common.yml")
        with open(common, "w") as f:
            yaml.dump({"setting": merged["classifier_setting"]}, f, default_flow_style=False)
    wfs = os.path.join(d, "wfs")
    if "train_workflow" in merged:
        os.makedirs(wfs)
        with open(os.path.join(wfs, "train.yml"), "w") as f:
            yaml.dump(merged["train_workflow"], f, default_flow_style=False)
    elif merged.get("workflow_overrides") or merged.get("workflow_inserts") or merged.get("workflow_modules"):
        write_workflow_overrides(merged["wfs_base"], merged.get("workflow_overrides"), wfs, log=None,
                                 inserts=merged.get("workflow_inserts"), modules=merged.get("workflow_modules"))
    else:
        os.makedirs(wfs)
        shutil.copy(os.path.join(merged["wfs_base"], "train.yml"), os.path.join(wfs, "train.yml"))
    _sync_tree(d, final)
    shutil.rmtree(d)
    model = f"{eos_base}/classifier/{label}"
    declared = MODELS.get(name)
    if declared and declared.rstrip("/") != model.rstrip("/"):
        raise ValueError(f"{name}: inputs.models says {declared}, but {captured} trained into {model}")
    out_dir = (blk.get("output_dir") or f"output/{label}/").rstrip("/")
    return {
        "upstream": rid,
        "label": label,
        "train_wf": os.path.join(final, "wfs", "train.yml"),
        "common": os.path.join(final, "common.yml") if common else None,
        "model": model,
        "reference": f"{model}/result.json",
        "archived_wf": f"{eos_base}/{out_dir}/wfs/train.yml",
        "floor": entry.get("floor"),
        "mem_mb": int(entry.get("mem_mb", 64000)),
    }


PLAN = {name: regenerate_workflow(name, e) for name, e in ENTRIES.items()}

localrules: all_fit_quality, fitq_validate, fitq_summary
for _n, _p in PLAN.items():
    print(f"fit-quality replay {_n}: {_p['upstream']} {_p['label']} <- {_p['model']}")


rule all_fit_quality:
    default_target: True
    input:
        f"{OUT}/index.html",


rule fitq_validate:
    input:
        f"{OUT}/{next(iter(PLAN))}/fit_quality.json",


rule fitq_replay:
    output:
        f"{OUT}/{{name}}/fit_quality.json",
    log:
        f"{OUT}/{{name}}/fitq_replay.log",
    wildcard_constraints:
        name="|".join(PLAN),
    threads: 4
    resources:
        gpus=1,
        cpu=4,
        mem_mb=lambda wc: PLAN[wc.name]["mem_mb"],
        runtime=360,
    params:
        replay=lambda wc: f"{OUT}/{wc.name}/replay",
        report=lambda wc: f"{OUT}/{wc.name}/report",
        title=lambda wc: f"{ENTRIES[wc.name].get('method', wc.name)} {ENTRIES[wc.name].get('era', '')}".strip(),
        args=lambda wc: " ".join(
            [
                f"--train-wf {PLAN[wc.name]['train_wf']}",
                f"--common {PLAN[wc.name]['common']}" if PLAN[wc.name]["common"] else "",
                f"--pretrained {PLAN[wc.name]['model']}",
                f"--reference {PLAN[wc.name]['reference']}",
                f"--archived-wf {PLAN[wc.name]['archived_wf']}",
                f"--floor {PLAN[wc.name]['floor']}" if PLAN[wc.name]["floor"] is not None else "",
                f"--tolerance {TOLERANCE}",
            ]
        ),
    shell:
        # replay/ is this rule's own scratch: cleared so a retry never resumes a stale checkpoint
        """
        rm -rf {params.replay}
        CLASSIFIER_CONFIG_PATHS=coffea4bees python -m src.classifier.fit_quality_replay \
            {params.args} --output {params.replay} --run > {log} 2>&1
        rm -rf {params.report}
        python -m src.classifier.fit_quality_report entry --result {params.replay}/result.json --stage baseline \
            --consistency {params.replay}/consistency.json --wf-check {params.replay}/train_wf_check.json \
            --name {wildcards.name} --title "{params.title}" --out {params.report} >> {log} 2>&1
        cp {params.replay}/fit_quality.json {output}
        """


rule fitq_summary:
    input:
        expand(f"{OUT}/{{name}}/fit_quality.json", name=list(PLAN)),
    output:
        html=f"{OUT}/index.html",
    params:
        entries=" ".join(f"{OUT}/{n}/report" for n in PLAN),
        title=config.get("fit_quality_title", "Fit-quality benchmarks"),
    shell:
        """
        python -m src.classifier.fit_quality_report summary --entries {params.entries} \
            --title "{params.title}" --out {output.html}
        """
