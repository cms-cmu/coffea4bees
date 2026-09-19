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