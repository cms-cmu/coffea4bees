# Optional CoReNet classifier workflow for boosted studies.
# Instructions: coffea4bees/workflows/docs/CoReNet_boosted_workflow.md

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

OUTPUT_BASE = str(_inputs.get("output_base") or "PLACEHOLDER_output").rstrip("/")
NETWORK = _corenet.get("network") or ""  # optional: replaces the decay-chain file written in train.yml
OUTPUT_PATH = str(config.setdefault("output_path", f"output/{config.get('label', 'corenet')}/")).rstrip("/") + "/"
config.setdefault("label", "corenet")
config["output_path"] = OUTPUT_PATH

_ALL_STEPS = ("classifier", "analysis", "stats")
STEPS = config.get("steps", ["classifier"])
if isinstance(STEPS, str):
    STEPS = STEPS.replace(",", " ").split()
_unknown = [s for s in STEPS if s not in _ALL_STEPS]
if _unknown:
    raise ValueError(f"steps: unknown entries {_unknown} (known: {list(_ALL_STEPS)})")

# The generic classifier Snakefile reads these top-level keys in single-model mode; here the
# model is addressed by its directory instead, so they must not be set.
for _k in ("wfs_base", "common", "model", "friend", "train_template", "eval_template", "eos_base"):
    if _k in config:
        raise ValueError(f"top-level `{_k}:` is not used by Snakefile_CoReNet.smk; "
                         f"use the classifier: / inputs: blocks")
config.setdefault("classifier_config_paths", "coffea4bees")


def _check_path(name, value):
    if not value or re.search(r"[\s,{}\"']", str(value)) or ": " in str(value):
        raise ValueError(f"{name} = {value!r}: paths must be non-empty and free of spaces, commas, quotes and braces")


# inputs.classes: {class name: [ROOT files]}
CLASSES = {}
for _name, _files in (_inputs.get("classes") or {}).items():
    _files = [_files] if isinstance(_files, str) else list(_files or [])
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(_name)):
        raise ValueError(f"inputs.classes: class name {_name!r} must be letters, digits and underscores")
    for _f in _files:
        _check_path(f"inputs.classes.{_name}", _f)
    CLASSES[str(_name)] = [str(f) for f in _files]

_blk = config.get("classifier") or {}
if not isinstance(_blk, dict):
    raise ValueError("`classifier:` must be a mapping")
_label = _blk.get("label", "boosted_corenet")
CLF = {
    "label": _label,
    "wfs_base": _blk.get("wfs_base", "coffea4bees/classifier/config/workflows/HH4b_boosted_corenet").rstrip("/"),
    "dir": f"{OUTPUT_PATH}classifiers/{_label}",
    "model": f"{OUTPUT_BASE}/classifier/{_label}",
    "friend": f"{OUTPUT_BASE}/friend/{_label}",
    "model_options": _blk.get("model_options") or {},
}
if "classifier" in STEPS:
    if len(CLASSES) < 2 or not all(CLASSES.values()):
        raise ValueError("inputs.classes: give at least two classes, each with at least one ROOT file")
else:
    CLF["friend"] = str(_inputs.get("friend") or "").rstrip("/")
    if not CLF["friend"] and "analysis" in STEPS:
        raise ValueError("`classifier` is not in steps: set inputs.friend to an existing friend directory")

TEMPLATE_KEYS = {"model": CLF["model"], "friend": CLF["friend"] or "unused"}
for _k, _v in TEMPLATE_KEYS.items():
    _check_path(_k, _v)
if NETWORK:
    _check_path("corenet.network", NETWORK)
TEMPLATE_STR = ", ".join(f"{k}: {v}" for k, v in TEMPLATE_KEYS.items())

config.setdefault("train_templates", {})[CLF["dir"]] = TEMPLATE_STR
config.setdefault("eval_templates", {})[CLF["dir"]] = TEMPLATE_STR
config.setdefault("models", {})[CLF["dir"]] = CLF["model"]
config.setdefault("friends_dict", {})[CLF["dir"]] = CLF["friend"]


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


def _set_files(mod):
    """Replace the --files entries of a dataset module by one per class of inputs.classes."""
    groups = [g for g in _option_groups(mod.get("option") or [])
              if not (isinstance(g[0], str) and g[0].split()[0] == "--files")]
    groups += [[f"--files label:{name} " + " ".join(files)] for name, files in CLASSES.items()]
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


def stage_classifier(out_train, out_eval, out_common):
    """Copy the job templates into the run directory with the config applied: inputs.classes,
    corenet.network / corenet.inputs / corenet.architecture, classifier.model_options,
    classifier_setting."""
    with open(f"{CLF['wfs_base']}/train.yml") as f:
        train = yaml.safe_load(f)
    with open(f"{CLF['wfs_base']}/evaluate.yml") as f:
        evaluate = yaml.safe_load(f)
    for wf in (train, evaluate):
        datasets = wf.get("dataset") or []
        if len(datasets) != 1:
            raise ValueError(f"{CLF['wfs_base']}: expected exactly one dataset module per job file")
        _set_files(datasets[0])
    overrides = {}
    if NETWORK:
        overrides["--network"] = "file:##" + NETWORK
    if _corenet.get("inputs"):
        overrides["--inputs"] = [dict(_corenet["inputs"])]
    if _corenet.get("architecture"):
        overrides["--architecture"] = [dict(_corenet["architecture"])]
    overrides.update(CLF["model_options"])
    models = train.get("model") or []
    if len(models) != 1:
        raise ValueError(f"{CLF['wfs_base']}/train.yml: expected exactly one model module")
    _set_options(models[0], overrides)
    _write_template(train, out_train)
    _write_template(evaluate, out_eval)
    if config.get("classifier_setting"):
        common = {"setting": config["classifier_setting"]}
    else:
        with open(f"{CLF['wfs_base']}/common.yml") as f:
            common = yaml.safe_load(f)
    os.makedirs(os.path.dirname(out_common), exist_ok=True)
    with open(out_common, "w") as f:
        yaml.safe_dump(common, f, sort_keys=False, default_flow_style=False)


_INPUTS_CHECKED = f"{OUTPUT_PATH}classifiers/inputs_checked.txt"

rule corenet_check_inputs:
    output: _INPUTS_CHECKED
    params:
        values = dict(TEMPLATE_KEYS) | {f"classes.{k}": " ".join(v) for k, v in CLASSES.items()},
    run:
        left = {k: v for k, v in params.values.items() if "PLACEHOLDER" in v}
        if left:
            raise ValueError("placeholder inputs left in the config (inputs:): "
                             + ", ".join(f"{k} = {v}" for k, v in left.items()))
        with open(output[0], "w") as f:
            for k, v in params.values.items():
                f.write(f"{k}: {v}\n")


CLF_TARGETS, CLF_FRIEND_READY = [], []
if "classifier" in STEPS:
    rule corenet_stage:
        input:
            checked = _INPUTS_CHECKED,
            train = f"{CLF['wfs_base']}/train.yml",
            evaluate = f"{CLF['wfs_base']}/evaluate.yml",
        output:
            train = f"{CLF['dir']}/wfs/train.yml",
            evaluate = f"{CLF['dir']}/wfs/evaluate.yml",
            common = f"{CLF['dir']}/common.yml",
        params:
            # what the staged files depend on: a change here (and only here) restages and retrains
            staged = {
                "templates": TEMPLATE_STR,
                "classes": CLASSES,
                "corenet": {k: _corenet.get(k) for k in ("network", "inputs", "architecture")},
                "model_options": CLF["model_options"],
                "classifier_setting": config.get("classifier_setting"),
            },
        run:
            stage_classifier(output.train, output.evaluate, output.common)

    CLF_TARGETS = [f"{CLF['dir']}/{t}" for t in ("train.done", "analyze.done", "evaluate.done")]
    CLF_FRIEND_READY = [f"{CLF['dir']}/evaluate.done"]

    rule corenet_classifier:
        input: CLF_TARGETS

    include: "../../src/classifier/workflow/Snakefile"

    localrules: corenet_stage, corenet_classifier

localrules: corenet_check_inputs, corenet_all


# Empty slots: histogramming and fit. Off unless listed in `steps:`; nothing is provided for them.
ANALYSIS_TARGETS = []
if "analysis" in STEPS:
    _acfg = config.get("analysis_config")
    if not isinstance(_acfg, dict) or not _acfg.get("processor"):
        raise ValueError("`analysis` is in steps: set analysis_config (processor, dataset_location, ...)")
    _friends = _acfg.setdefault("config", {}).setdefault("friends", {})
    _friends.setdefault(config.get("analysis_friend_name", CLF["label"]),
                        f"{CLF['friend']}/result.json@@analysis.0.merged")

    include: "Snakefile_PhaseF_1_analysis.smk"

    # F.1's create_analysis_config with the friend tree as an extra input, so that no
    # histogramming job starts before the classifier has been evaluated.
    use rule create_analysis_config as corenet_analysis_config with:
        input:
            get_analysis_config_inputs,
            CLF_FRIEND_READY,

    ruleorder: corenet_analysis_config > create_analysis_config
    localrules: corenet_analysis_config

    ANALYSIS_TARGETS = [
        f"{F1_OUTPUT_PATH}histAll_{config['label']}.coffea",
        f"{F1_OUTPUT_PATH}plots_{config['label']}/plots_done.txt",
    ]

    rule corenet_analysis:
        input: ANALYSIS_TARGETS

    localrules: corenet_analysis


STATS_TARGETS = []
if "stats" in STEPS:
    if not config.get("channels"):
        raise ValueError("`stats` is in steps: set channels (and the other fit keys)")

    include: "Snakefile_PhaseF_2_stats.smk"

    STATS_TARGETS = list(rules.all_stats.input)

    rule corenet_stats:
        input: STATS_TARGETS

    localrules: corenet_stats


rule corenet_all:
    default_target: True
    input:
        CLF_TARGETS,
        ANALYSIS_TARGETS,
        STATS_TARGETS,
