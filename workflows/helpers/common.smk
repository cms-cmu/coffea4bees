import sys
import os
import copy
import yaml

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from src.stat_analysis.helpers import make_poi_maps, get_default_othersignals, get_grid_split_points, get_likelihood_scan_chunks

def substitute_placeholders(obj, mapping):
    """Replace {name} placeholders in every string of a nested config structure."""
    if isinstance(obj, dict):
        return {k: substitute_placeholders(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_placeholders(v, mapping) for v in obj]
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("{" + k + "}", str(v))
    return obj


# {roast_id} names a production run so run-scoped paths (e.g. EOS outputs) are unique.
# `roast` sets it with --config roast_id=<id>; outside roast it falls back to the config
# label, so a plain `snakemake --configfile ...` run still gets a sensible directory.
_roast_id = config.get('roast_id') or config.get('label') or 'nominal'
config['roast_id'] = _roast_id
for _k, _v in substitute_placeholders(dict(config), {'roast_id': _roast_id}).items():
    config[_k] = _v


def check_handoff_refs(config_dict):
    """Fail if what Phases C/D are told to read is not what Phase B publishes.

    Phase B's `phaseB_handoff` rule writes the JCM fit and the classifier-input manifest to
    `handoff.eos_base`, and the `fvt`/`svb` blocks name those same URLs for the classifier to
    read. The two are stated separately -- and include the JCM `tag` in the filename -- so they
    can drift apart, and the failure is silent in the worst way: the classifier trains happily
    against another production's weights and only the loss curve looks slightly off.

    Pointing at an earlier roast's handoff is a legitimate thing to do (rerunning C or D with
    new code against fixed inputs), so that is allowed explicitly with
    `handoff.allow_external_refs: true` rather than by weakening this into a warning.
    """
    handoff = config_dict.get('handoff') or {}
    if not isinstance(handoff, dict):
        return
    base = str(handoff.get('eos_base') or "").rstrip('/')
    if not base or handoff.get('allow_external_refs'):
        return

    problems = []
    for section in ('fvt', 'svb'):
        blk = config_dict.get(section) or {}
        if not isinstance(blk, dict):
            continue
        refs = {'metadata': blk.get('metadata')}
        overrides = blk.get('workflow_overrides') or {}
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if k == '--JCM-weight' or k.startswith('--friends'):
                    refs[k] = v
        for key, val in refs.items():
            if not isinstance(val, str):
                continue
            path = val.split('@@')[0].replace('""', '').strip()
            # Only remote refs are checked: a local path is read from the checkout, which is
            # the other, deliberate half of the handoff (C.4 and F.1 read the local JCM copy).
            if path.startswith('root://') and not path.startswith(base + '/'):
                problems.append(f"  {section}.{key}\n      reads      {path}\n      not under  {base}/")
    if problems:
        raise ValueError(
            "handoff mismatch: Phase C/D are pointed at remote files Phase B does not publish.\n"
            + "\n".join(problems)
            + "\n  Fix the paths, or set `handoff.allow_external_refs: true` if reading another"
              " roast's handoff is intended."
        )


check_handoff_refs(config)


def write_workflow_overrides(wfs_base, overrides, out_dir, log=None):
    """Copy the classifier workflow templates in `wfs_base` (train.yml, evaluate.yml, ...) to
    `out_dir`, replacing command-line options by flag.

    `overrides` maps a flag (optionally followed by its first argument(s), to disambiguate
    repeated flags) to the replacement for everything after the key, e.g.
        {"--JCM-weight": '"" path/to/JCM.yml@@JCM_weights',
         "--friends \"\"": 'path/to/classifier_inputs.json@@HCR_input'}
    The second key matches `--friends "" <anything>` but not `--friends label:data <FvT>`.
    Every string entry of every module's `option` list that starts with a key (followed by a
    space or the end of the entry) becomes "<key> <replacement>"; the longest matching key wins;
    everything else is copied verbatim, so the checked-in templates stay the single source of
    truth for the model/training settings and a production only states what differs (typically
    the inputs). Returns `out_dir`.
    """
    import glob
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    flags = dict(overrides or {})
    keys = sorted(flags, key=len, reverse=True)
    used = {k: 0 for k in flags}

    def match(opt):
        for k in keys:
            if opt == k or opt.startswith(k + " "):
                return k
        return None
    for src in sorted(glob.glob(os.path.join(wfs_base, "*"))):
        dst = os.path.join(out_dir, os.path.basename(src))
        if os.path.isdir(src):
            continue
        if not src.endswith((".yml", ".yaml")):
            shutil.copy2(src, dst)
            continue
        with open(src) as f:
            wf = yaml.safe_load(f) or {}
        for section in wf.values():
            if not isinstance(section, list):
                continue
            for module in section:
                opts = module.get("option") if isinstance(module, dict) else None
                if not isinstance(opts, list):
                    continue
                for i, opt in enumerate(opts):
                    k = match(opt) if isinstance(opt, str) else None
                    if k is not None:
                        opts[i] = f"{k} {flags[k]}".rstrip()
                        used[k] += 1
        with open(dst, "w") as f:
            yaml.dump(wf, f, default_flow_style=False, sort_keys=False)
    unused = [k for k, n in used.items() if n == 0]
    if unused and log is not None:
        log(f"workflow_overrides: flags not found in any template under {wfs_base}: {unused}")
    return out_dir


def resolve_config_section(config_dict, primary_key=None, fallback_keys=None, inherit_keys=None):
    """
    Extracts and parses a sub-configuration block from the global Snakemake config dict.
    If the value is a string path to a YAML file, loads it.
    Inherits specified keys from fallback sections or the global config dict if they are missing.
    """
    if fallback_keys is None:
        fallback_keys = []
    if inherit_keys is None:
        inherit_keys = ['processor', 'dataset_location', 'friend_file', 'weights_file', 'runner', 'config']

    base = {}
    for fk in fallback_keys:
        if fk in config_dict:
            val = config_dict[fk]
            if isinstance(val, str) and os.path.exists(val):
                with open(val, 'r') as f:
                    val = yaml.safe_load(f) or {}
            if isinstance(val, dict):
                base = copy.deepcopy(val)
                break

    res = copy.deepcopy(base)
    if primary_key and primary_key in config_dict:
        raw = config_dict[primary_key]
        if isinstance(raw, str) and os.path.exists(raw):
            with open(raw, 'r') as f:
                raw = yaml.safe_load(f) or {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if isinstance(v, dict) and isinstance(res.get(k), dict):
                    res[k].update(copy.deepcopy(v))
                else:
                    res[k] = copy.deepcopy(v)
        elif raw is not None:
            res = copy.deepcopy(raw)

    for k in inherit_keys:
        if k not in res and k in config_dict:
            res[k] = copy.deepcopy(config_dict[k])
    return res