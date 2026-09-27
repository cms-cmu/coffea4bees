#!/usr/bin/env python3
"""Build multisample registry YAML from per-subsample registries.

Combines per-subsample picoAOD registries and ttbar PSData manifests into a
unified multisample dataset YAML (e.g. mixeddata_4b.yml) using vXXX templating.

Usage:
    python coffea4bees/hemisphere_mixing/build_multisample_registry.py \\
        --registry output/path/per_subsample/picoaod_datasets_mix_v0.yml \\
        --output coffea4bees/metadata/datasets/mixeddata_4b.yml \\
        --dataset-name mixeddata_4b \\
        --n-samples 16 \\
        --years 2016 2017 2018 \\
        --psdata-manifest coffea4bees/metadata/datasets/ttbar_PSData.yml
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multisample registry YAML from per-subsample registries."
    )
    parser.add_argument(
        "--registries",
        nargs="+",
        required=True,
        help="Input per-subsample registry YAML file(s). First entry is used as template.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset YAML file path.",
    )
    parser.add_argument(
        "--dataset-name",
        default="mixeddata_4b",
        help="Top-level dataset key in output YAML (default: mixeddata_4b).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Total number of subsamples (default: 16).",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2016", "2017", "2018"],
        help="List of data years / eras to configure.",
    )
    parser.add_argument(
        "--psdata-manifest",
        default=None,
        help="Optional path to ttbar PSData manifest file.",
    )
    parser.add_argument(
        "--seed-manifest",
        default="coffea4bees/metadata/datasets/mixeddata_4b.yml",
        help="Optional seed manifest to preserve static entries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    existing_psdata: dict[str, list[str]] = {}
    target_path = (
        args.output
        if os.path.exists(args.output)
        else args.seed_manifest
    )
    if os.path.exists(target_path):
        with open(target_path, "r") as f_old:
            try:
                old_data = yaml.safe_load(f_old) or {}
                for y, y_info in old_data.get(args.dataset_name, {}).items():
                    if isinstance(y_info, dict) and "picoAOD" in y_info:
                        existing_psdata[y] = [
                            f
                            for f in y_info["picoAOD"].get("files_template", [])
                            if "PSData" in f or "vXXX" not in f
                        ]
            except Exception as e:
                print(f"Warning reading seed manifest {target_path}: {e}")

    # Ensure ttbar PSData files are loaded for each year
    psdata_sources = [
        args.psdata_manifest,
        "coffea4bees/metadata/datasets/ttbar_PSData.yml",
        "coffea4bees/metadata/datasets/ttbar_PSData_stitched.yml",
    ]
    for ps_src in psdata_sources:
        if ps_src and os.path.exists(ps_src):
            try:
                with open(ps_src, "r") as f_ps:
                    ps_yaml = yaml.safe_load(f_ps) or {}
                    for top_k, top_v in ps_yaml.items():
                        if isinstance(top_v, dict):
                            for y, y_info in top_v.items():
                                if isinstance(y_info, dict) and "picoAOD" in y_info:
                                    files = y_info["picoAOD"].get(
                                        "files", []
                                    ) or y_info["picoAOD"].get(
                                        "files_template", []
                                    )
                                    if files:
                                        for f in files:
                                            if f not in existing_psdata.setdefault(y, []):
                                                existing_psdata[y].append(f)
                if any(existing_psdata.values()):
                    print(
                        f"Loaded ttbar PSData files from {ps_src}: {list(existing_psdata.keys())}"
                    )
                    break
            except Exception as e:
                print(f"Warning loading psdata from {ps_src}: {e}")

    multisample_data: dict = {
        args.dataset_name: {
            "nSamples": int(args.n_samples),
            "xs": {"Run2": 1, "Run3": 1},
        }
    }

    era_files: dict[str, list[str]] = {}
    curr_era = None
    in_files = False
    # Only the first per-subsample registry is parsed: every subsample produces the same set
    # of era/file layouts, and the filenames are templated to vXXX below.
    template_reg = args.registries[0]
    with open(template_reg, "r") as f:
        for line in f:
            if not line.startswith(" ") and ":" in line:
                curr_era = line.split(":")[0].strip()
                era_files[curr_era] = []
                in_files = False
            elif curr_era and line.strip().startswith("files:"):
                in_files = True
            elif in_files:
                stripped = line.strip()
                if stripped.startswith("- "):
                    era_files[curr_era].append(stripped[2:].strip())
                elif stripped and not stripped.startswith("#"):
                    in_files = False

    for year in args.years:
        multisample_data[args.dataset_name][year] = {"picoAOD": {"files_template": []}}
        for era_key, files in era_files.items():
            if f"_{year}" in era_key:
                for fp in files:
                    t = re.sub(r"/subsamples/v\d+/", "/subsamples/vXXX/", fp)
                    # Collapse the per-subsample picoAOD name to the vXXX template
                    t = re.sub(r"_v\d+(\.chunk\d+)?\.root", r"_vXXX.root", t)
                    if (
                        t
                        not in multisample_data[args.dataset_name][year][
                            "picoAOD"
                        ]["files_template"]
                    ):
                        multisample_data[args.dataset_name][year]["picoAOD"][
                            "files_template"
                        ].append(t)
        # Add back any static PSData files
        for static_file in existing_psdata.get(year, []):
            if (
                static_file
                not in multisample_data[args.dataset_name][year]["picoAOD"][
                    "files_template"
                ]
            ):
                multisample_data[args.dataset_name][year]["picoAOD"][
                    "files_template"
                ].append(static_file)

    print(
        f"Building multisample registry with {len(args.years)} years into {args.output}"
    )
    with open(args.output, "w") as f:
        yaml.dump(multisample_data, f, default_flow_style=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
