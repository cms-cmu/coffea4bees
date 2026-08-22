import sys
import os
import copy
import yaml

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from src.stat_analysis.helpers import make_poi_maps, get_default_othersignals, get_grid_split_points, get_likelihood_scan_chunks

def resolve_config_section(config_dict, primary_key=None, fallback_keys=None, inherit_keys=None):
    """
    Extracts and parses a sub-configuration block from the global Snakemake config dict.
    If the value is a string path to a YAML file, loads it.
    Inherits specified keys from the global config dict if they are missing in the sub-block.
    """
    if fallback_keys is None:
        fallback_keys = []
    if inherit_keys is None:
        inherit_keys = ['processor', 'dataset_location', 'friend_file', 'weights_file', 'runner', 'config']

    raw = None
    if primary_key and primary_key in config_dict:
        raw = config_dict[primary_key]
    else:
        for fk in fallback_keys:
            if fk in config_dict:
                raw = config_dict[fk]
                break

    if raw is None:
        raw = {}

    if isinstance(raw, str) and os.path.exists(raw):
        with open(raw, 'r') as f:
            raw = yaml.safe_load(f) or {}
    elif not isinstance(raw, dict):
        raw = {}

    res = copy.deepcopy(raw)
    for k in inherit_keys:
        if k not in res and k in config_dict:
            res[k] = copy.deepcopy(config_dict[k])
    return res