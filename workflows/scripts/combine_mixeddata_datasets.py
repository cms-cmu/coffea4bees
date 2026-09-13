"""Combine per-sample mixed-data registries into a master dataset YAML.

Constructs a multi-sample dataset YAML file (like mixeddata_4b.yml) with
picoAOD file lists for each year and sample (v0..v{N-1}), or template files_template entries.

Usage:
    python combine_mixeddata_datasets.py \
        --registries output/mixeddata/registries/picoaod_datasets_mixeddata__v*.yml \
        --output coffea4bees/metadata/datasets/mixeddata_4b.yml \
        --dataset-name mixeddata_4b
"""

import argparse
import glob
import re
import sys
import yaml


class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores python-object/tuple tags."""


_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", lambda loader, suffix, node: None
)
_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/new:", lambda loader, suffix, node: None
)
_TolerantLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object:", lambda loader, suffix, node: None
)
_TolerantLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: None
)


def parse_args():
    parser = argparse.ArgumentParser(description="Combine mixeddata registries into dataset YAML")
    parser.add_argument("-r", "--registries", nargs="+", required=True, help="Input registry YAML files")
    parser.add_argument("-o", "--output", required=True, help="Output master dataset YAML file")
    parser.add_argument("-n", "--dataset-name", default="mixeddata_4b", help="Top-level dataset key")
    return parser.parse_args()


def extract_sample_id(filename):
    match = re.search(r"__(v\d+)\.ya?ml", filename)
    if match:
        return match.group(1)
    match2 = re.search(r"_(v\d+)", filename)
    if match2:
        return match2.group(1)
    return "v0"


def main():
    args = parse_args()

    sample_registries = {}
    for reg_path in args.registries:
        sample_id = extract_sample_id(reg_path)
        with open(reg_path, "r") as f:
            data = yaml.load(f, Loader=_TolerantLoader) or {}
        sample_registries[sample_id] = data

    out_dataset = {args.dataset_name: {"nSamples": len(sample_registries)}}

    # Group files by year
    years_seen = set()
    for s_id, reg_data in sample_registries.items():
        for dname, dinfo in reg_data.items():
            year = dinfo.get("year", None)
            if not year:
                # Try to infer year from dataset name (e.g. data_UL18A -> UL18)
                for y_cand in ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18", "2016", "2017", "2018", "2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix"]:
                    if y_cand in dname:
                        year = y_cand
                        break
            if year:
                years_seen.add(year)

    for year in sorted(years_seen):
        out_dataset[args.dataset_name][year] = {"picoAOD": {"files": []}}
        all_files = []
        for s_id in sorted(sample_registries.keys(), key=lambda x: int(x.replace("v", "")) if x.replace("v", "").isdigit() else 0):
            reg_data = sample_registries[s_id]
            for dname, dinfo in reg_data.items():
                if year in dname or dinfo.get("year") == year:
                    files = dinfo.get("files", [])
                    if isinstance(files, list):
                        all_files.extend(files)
        out_dataset[args.dataset_name][year]["picoAOD"]["files"] = sorted(list(set(all_files)))
        # Also create files_template based on v0 (or first sample)
        first_s_id = sorted(sample_registries.keys())[0]
        v_num = first_s_id.lstrip("v")
        v0_files = []
        for dname, dinfo in sample_registries[first_s_id].items():
            if year in dname or dinfo.get("year") == year:
                v0_files.extend(dinfo.get("files", []))
        if v0_files:
            templates = [re.sub(rf"_v{v_num}([/._])", r"_vXXX\1", f) for f in v0_files]
            out_dataset[args.dataset_name][year]["picoAOD"]["files_template"] = sorted(list(set(templates)))

    # Also generate individual sample dataset entries (e.g. mixeddata_4b_v0, mixeddata_4b_v1, ...)
    for s_id in sorted(sample_registries.keys(), key=lambda x: int(x.replace("v", "")) if x.replace("v", "").isdigit() else 0):
        s_dataset_name = f"{args.dataset_name}_{s_id}"
        out_dataset[s_dataset_name] = {}
        reg_data = sample_registries[s_id]
        for year in sorted(years_seen):
            s_files = []
            for dname, dinfo in reg_data.items():
                if year in dname or dinfo.get("year") == year:
                    files = dinfo.get("files", [])
                    if isinstance(files, list):
                        s_files.extend(files)
            if s_files:
                out_dataset[s_dataset_name][year] = {"picoAOD": {"files": sorted(list(set(s_files)))}}

    with open(args.output, "w") as f:
        yaml.safe_dump(out_dataset, f, default_flow_style=False, sort_keys=False)

    print(f"Successfully generated {args.output} with {len(sample_registries)} samples across {len(years_seen)} years.")


if __name__ == "__main__":
    main()
