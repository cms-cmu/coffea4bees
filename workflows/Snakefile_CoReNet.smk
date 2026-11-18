# Optional CoReNet (FvT + SvB) analysis workflow. Instructions: coffea4bees/workflows/docs/CoReNet_boosted_workflow.md

import copy
import os
import re

import yaml

if not workflow.configfiles:
    raise ValueError(
        "Snakefile_CoReNet.smk needs a config, e.g. "
        "--configfile coffea4bees/workflows/config/corenet_boosted.yml"
    )

include: "helpers/common.smk"

_inputs = config.get("inputs") or {}
_corenet = config.get("corenet") or {}

DATASET_METADATA = _inputs.get("dataset_metadata", "PLACEHOLDER_datasets/")
CLASSIFIER_INPUTS = _inputs.get("classifier_inputs", "PLACEHOLDER_inputs/classifier_inputs_boosted.json@@HCR_input")
JCM_WEIGHTS = _inputs.get("jcm_weights", "") or ""
OUTPUT_BASE = str(_inputs.get("output_base", "PLACEHOLDER_output/" + str(config["roast_id"]))).rstrip("/")
NETWORK = _corenet.get("network", "coffea4bees/classifier/config/corenet/HH4b_boosted.yml")
OUTPUT_PATH = str(config.setdefault("output_path", f"output/{config.get('label', 'corenet')}/")).rstrip("/") + "/"
config.setdefault("label", "corenet")
config["output_path"] = OUTPUT_PATH

_ALL_STEPS = ("fvt", "svb", "analysis", "stats")
STEPS = config.get("steps", list(_ALL_STEPS))
if isinstance(STEPS, str):
    STEPS = STEPS.replace(",", " ").split()
_unknown = [s for s in STEPS if s not in _ALL_STEPS]
if _unknown:
    raise ValueError(f"steps: unknown entries {_unknown} (known: {list(_ALL_STEPS)})")

# The generic classifier Snakefile reads these top-level keys in single-model mode; here every
# model is addressed by its directory instead, so they must not be set.
for _k in ("wfs_base", "common", "model", "friend", "train_template", "eval_template", "eos_base"):
    if _k in config:
        raise ValueError(f"top-level `{_k}:` is not used by Snakefile_CoReNet.smk; "
                         f"use the fvt: / svb: / inputs: blocks")
config.setdefault("classifier_config_paths", "coffea4bees")


def _classifier(key, default_label):
    blk = config.get(key) or {}
    if not isinstance(blk, dict):
        raise ValueError(f"`{key}:` must be a mapping")
    label = blk.get("label", default_label)
    return {
        "key": key,
        "label": label,
        "wfs_base": blk.get("wfs_base", f"coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/{'FvT' if key == 'fvt' else 'SvB'}").rstrip("/"),
        "dir": f"{OUTPUT_PATH}classifiers/{label}",
        "model": f"{OUTPUT_BASE}/classifier/{label}",
        "friend": f"{OUTPUT_BASE}/friend/{label}",
        "model_options": blk.get("model_options") or {},
    }


FVT = _classifier("fvt", "FvT_boosted_corenet")
SVB = _classifier("svb", "SvB_boosted_corenet")
if "fvt" not in STEPS:
    FVT["friend"] = str(_inputs.get("fvt_friend") or "").rstrip("/")
    if not FVT["friend"] and ("svb" in STEPS or "analysis" in STEPS):
        raise ValueError("`fvt` is not in steps: set inputs.fvt_friend to an existing FvT friend directory")
if "svb" not in STEPS:
    SVB["friend"] = str(_inputs.get("svb_friend") or "").rstrip("/")
    if not SVB["friend"] and "analysis" in STEPS:
        raise ValueError("`svb` is not in steps: set inputs.svb_friend to an existing SvB friend directory")

TEMPLATE_KEYS = {
    "fvt_model": FVT["model"],
    "fvt_friend": FVT["friend"] or "unused",
    "svb_model": SVB["model"],
    "svb_friend": SVB["friend"] or "unused",
    "metadata": DATASET_METADATA,
    "inputs": CLASSIFIER_INPUTS,
    "jcm": JCM_WEIGHTS or "unused",
    "network": NETWORK,
}
for _k, _v in TEMPLATE_KEYS.items():
    if not _v or re.search(r"[\s,{}\"']", str(_v)) or ": " in str(_v):
        raise ValueError(f"{_k} = {_v!r}: paths must be non-empty and free of spaces, commas, quotes and braces")
TEMPLATE_STR = ", ".join(f"{k}: {v}" for k, v in TEMPLATE_KEYS.items())

for _c in (FVT, SVB):
    config.setdefault("train_templates", {})[_c["dir"]] = TEMPLATE_STR
    config.setdefault("eval_templates", {})[_c["dir"]] = TEMPLATE_STR
    config.setdefault("models", {})[_c["dir"]] = _c["model"]
    config.setdefault("friends_dict", {})[_c["dir"]] = _c["friend"]


def _option_groups(options):
    """Split a module's option list into [flag entry, continuation entries...] groups."""
    groups = []
    for opt in options:
        if isinstance(opt, str) and opt.startswith("--"):
            groups.append([opt])
        elif groups:
            groups[-1].append(opt)
        else:
            groups.append([opt])
    return groups


def _render_option(flag, value):
    """The option entries of `flag` for a model_options value (None / False: no entries)."""
    if value is None or value is False:
        return None
    if value is True:
        return [flag]
    head, rest = [flag], []
    for item in (value if isinstance(value, list) else [value]):
        if isinstance(item, (dict, list)):
            rest.append(item)
        elif rest:
            rest.append(str(item))
        else:
            head.append(str(item))
    return [" ".join(head)] + rest


def _set_options(mod, overrides):
    groups = _option_groups(mod.get("option") or [])
    for flag, value in overrides.items():
        new = _render_option(flag, value)
        hits = [i for i, g in enumerate(groups) if isinstance(g[0], str) and g[0].split()[0] == flag]
        for i in reversed(hits[1:]):
            del groups[i]
        if hits:
            if new is None:
                del groups[hits[0]]
            else:
                groups[hits[0]] = new
        elif new is not None:
            groups.append(new)
    mod["option"] = [opt for g in groups for opt in g]


def _write_template(wf, path):
    text = yaml.safe_dump(wf, sort_keys=False, default_flow_style=False, width=float("inf"))
    # the staged file is a template again: only the known keys may appear in braces
    stray = sorted(set(re.findall(r"\{[^{}]*\}", text)) - {"{" + k + "}" for k in TEMPLATE_KEYS})
    if "{" in re.sub(r"\{(%s)\}" % "|".join(TEMPLATE_KEYS), "", text) or stray:
        raise ValueError(f"{path}: braces other than the template keys {sorted(TEMPLATE_KEYS)} "
                         f"(an empty mapping or list in model_options / corenet?): {stray}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def stage_classifier(clf, out_train, out_eval, out_common):
    """Copy the templates of `clf` into its run directory with the config's changes applied:
    corenet.inputs / corenet.architecture, <clf>.model_options, no JCM, classifier_setting."""
    with open(f"{clf['wfs_base']}/train.yml") as f:
        train = yaml.safe_load(f)
    with open(f"{clf['wfs_base']}/evaluate.yml") as f:
        evaluate = yaml.safe_load(f)
    overrides = {}
    if _corenet.get("inputs"):
        overrides["--inputs"] = [dict(_corenet["inputs"])]
    if _corenet.get("architecture"):
        overrides["--architecture"] = [dict(_corenet["architecture"])]
    overrides.update(clf["model_options"])
    models = train.get("model") or []
    if len(models) != 1:
        raise ValueError(f"{clf['wfs_base']}/train.yml: expected exactly one model module")
    _set_options(models[0], overrides)
    if not JCM_WEIGHTS:
        for mod in train.get("dataset") or []:
            _set_options(mod, {"--JCM-weight": None})
    _write_template(train, out_train)
    _write_template(evaluate, out_eval)
    if config.get("classifier_setting"):
        common = {"setting": config["classifier_setting"]}
    else:
        with open(f"{clf['wfs_base']}/../common.yml") as f:
            common = yaml.safe_load(f)
    os.makedirs(os.path.dirname(out_common), exist_ok=True)
    with open(out_common, "w") as f:
        yaml.safe_dump(common, f, sort_keys=False, default_flow_style=False)


def _staged(clf):
    """What the staged files depend on: a change here (and only here) restages and retrains."""
    return {
        "templates": TEMPLATE_STR,
        "corenet": {k: _corenet.get(k) for k in ("inputs", "architecture")},
        "model_options": clf["model_options"],
        "classifier_setting": config.get("classifier_setting"),
        "jcm": bool(JCM_WEIGHTS),
    }


_INPUTS_CHECKED = f"{OUTPUT_PATH}classifiers/inputs_checked.txt"

rule corenet_check_inputs:
    output: _INPUTS_CHECKED
    params:
        values = {k: str(v) for k, v in TEMPLATE_KEYS.items()},
    run:
        left = {k: v for k, v in params.values.items() if "PLACEHOLDER" in v}
        if left:
            raise ValueError("placeholder inputs left in the config (inputs: / corenet:): "
                             + ", ".join(f"{k} = {v}" for k, v in left.items()))
        with open(output[0], "w") as f:
            for k, v in params.values.items():
                f.write(f"{k}: {v}\n")


if "fvt" in STEPS:
    rule corenet_stage_fvt:
        input:
            checked = _INPUTS_CHECKED,
            train = f"{FVT['wfs_base']}/train.yml",
            evaluate = f"{FVT['wfs_base']}/evaluate.yml",
        output:
            train = f"{FVT['dir']}/wfs/train.yml",
            evaluate = f"{FVT['dir']}/wfs/evaluate.yml",
            common = f"{FVT['dir']}/common.yml",
        params:
            staged = _staged(FVT),
        run:
            stage_classifier(FVT, output.train, output.evaluate, output.common)

    FVT_TARGETS = [f"{FVT['dir']}/{t}" for t in
                   ("train.done", "yields.html", "analyze.done", "gallery.done", "evaluate.done")]
    FVT_FRIEND_READY = [f"{FVT['dir']}/evaluate.done"]

    rule corenet_fvt:
        input: FVT_TARGETS

    localrules: corenet_stage_fvt, corenet_fvt
else:
    FVT_TARGETS, FVT_FRIEND_READY = [], []

if "svb" in STEPS:
    rule corenet_stage_svb:
        # staged once the FvT friend tree exists: this is what makes the SvB training wait for it
        input:
            checked = _INPUTS_CHECKED,
            train = f"{SVB['wfs_base']}/train.yml",
            evaluate = f"{SVB['wfs_base']}/evaluate.yml",
            fvt_friend = FVT_FRIEND_READY,
        output:
            train = f"{SVB['dir']}/wfs/train.yml",
            evaluate = f"{SVB['dir']}/wfs/evaluate.yml",
            common = f"{SVB['dir']}/common.yml",
        params:
            staged = _staged(SVB),
        run:
            stage_classifier(SVB, output.train, output.evaluate, output.common)

    SVB_TARGETS = [f"{SVB['dir']}/{t}" for t in
                   ("train.done", "yields.html", "analyze.done", "gallery.done", "evaluate.done")]
    SVB_FRIEND_READY = [f"{SVB['dir']}/evaluate.done"]

    rule corenet_svb:
        input: SVB_TARGETS

    localrules: corenet_stage_svb, corenet_svb
else:
    SVB_TARGETS, SVB_FRIEND_READY = [], []

if "fvt" in STEPS or "svb" in STEPS:
    include: "../../src/classifier/workflow/Snakefile"

localrules: corenet_check_inputs, corenet_all


ANALYSIS_TARGETS = []
if "analysis" in STEPS:
    _acfg = config.get("analysis_config")
    if isinstance(_acfg, dict):
        _names = config.get("analysis_friend_names") or {}
        _pc = _acfg.setdefault("config", {})
        _friends = _pc.setdefault("friends", {})
        _friends.setdefault(_names.get("fvt", "FvT"), f"{FVT['friend']}/result.json@@analysis.0.merged")
        _friends.setdefault(_names.get("svb", "SvB_MA"), f"{SVB['friend']}/result.json@@analysis.0.merged")
        if JCM_WEIGHTS:
            _pc.setdefault("JCM_file", JCM_WEIGHTS.split("@@")[0])

    include: "Snakefile_PhaseF_1_analysis.smk"

    # Same rule as F.1's create_analysis_config, with the two friend trees as extra inputs, so
    # that no histogramming job starts before the classifiers have been evaluated.
    use rule create_analysis_config as corenet_analysis_config with:
        input:
            get_analysis_config_inputs,
            FVT_FRIEND_READY + SVB_FRIEND_READY,

    ruleorder: corenet_analysis_config > create_analysis_config
    localrules: corenet_analysis_config

    ANALYSIS_TARGETS = [
        f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea",
        f"{F1_OUTPUT_PATH}plots_{config['label']}/plots_done.txt",
        f"{F1_OUTPUT_PATH}cutflow_validation_{config['label']}.txt",
        f"{F1_OUTPUT_PATH}cutflow_{config['label']}.yml",
    ]

    rule corenet_analysis:
        input: ANALYSIS_TARGETS

    localrules: corenet_analysis


STATS_TARGETS = []
if "stats" in STEPS:
    include: "Snakefile_PhaseF_2_stats.smk"

    STATS_TARGETS = list(rules.all_stats.input)

    rule corenet_stats:
        input: STATS_TARGETS

    localrules: corenet_stats


rule corenet_all:
    default_target: True
    input:
        FVT_TARGETS,
        SVB_TARGETS,
        ANALYSIS_TARGETS,
        STATS_TARGETS,
