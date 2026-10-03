#!/usr/bin/env python3
"""Stages analysis configuration for testing a subsample with calibrated JCM weights."""

import argparse
import os
import sys
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Stage subsample analysis config with calibrated JCM weights.")
    parser.add_argument("--base-config", required=True, help="Path to base processor config YAML")
    parser.add_argument("--jcm-files", nargs="+", required=True, help="One or more calibrated JCM YAML files")
    parser.add_argument("--output", required=True, help="Path to write staged analysis config YAML")
    parser.add_argument("--years", nargs="*", default=None, help="Optional list of years for per-year mapping")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.base_config):
        print(f"Error: base config file not found: {args.base_config}", file=sys.stderr)
        sys.exit(1)

    with open(args.base_config, "r") as f:
        config = yaml.safe_load(f) or {}

    config.setdefault("config", {})
    config["config"]["apply_JCM"] = True

    if len(args.jcm_files) == 1:
        config["config"]["JCM_file"] = args.jcm_files[0]
    else:
        # Per-year mapping
        jcm_map = {}
        years = args.years if args.years else ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18", "2022_preEE", "2022_postEE", "2023_preBPix", "2023_postBPix"]
        for jcm_file in args.jcm_files:
            matched_year = None
            for y in years:
                if f"_{y}." in jcm_file or jcm_file.endswith(f"_{y}.yml") or jcm_file.endswith(f"_{y}.yaml"):
                    matched_year = y
                    break
            if matched_year:
                jcm_map[matched_year] = jcm_file
            else:
                base = os.path.splitext(os.path.basename(jcm_file))[0]
                token = base.split("_")[-1]
                jcm_map[token] = jcm_file
        config["config"]["JCM_file"] = jcm_map

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully staged analysis config with JCM: {args.output}")


if __name__ == "__main__":
    main()
