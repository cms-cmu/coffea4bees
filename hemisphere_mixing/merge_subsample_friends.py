#!/usr/bin/env python3
"""Merge per-subsample friend tree JSON manifests into a unified master manifest.

Deduplicates shared target chunks (e.g. ttbar_PSData_stitched processed across multiple
subsamples) so that Friend.from_json / Friend.__iadd__ does not encounter chunk collisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))
from src.data_formats.root import Friend
from src.utils.json import DefaultEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-subsample friend tree JSON manifests with target deduplication."
    )
    parser.add_argument(
        "--inputs",
        "-i",
        nargs="+",
        required=True,
        help="Input per-subsample friend JSON files.",
    )
    parser.add_argument(
        "--output-json",
        "-o",
        required=True,
        help="Output merged friend JSON file path.",
    )
    parser.add_argument(
        "--output-done",
        default=None,
        help="Optional output touch/done marker file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    target_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(target_dir, exist_ok=True)

    merged: dict[str, dict] = {}
    seen_targets: dict[str, set] = {}

    for jf in args.inputs:
        if not os.path.exists(jf):
            print(f"Warning: input JSON {jf} does not exist, skipping.")
            continue

        with open(jf, "r") as f:
            meta = json.load(f)

        for k, v in meta.items():
            if not isinstance(v, dict) or not {"name", "branches", "data"}.issubset(v.keys()):
                continue

            if k not in merged:
                merged[k] = {
                    "name": v["name"],
                    "branches": list(v.get("branches", [])),
                    "data": [],
                }
                seen_targets[k] = set()

            for entry in v.get("data", []):
                # entry is [target_chunk_dict, [friend_items...]]
                target_dict = entry[0]
                target_key = (
                    target_dict.get("path"),
                    target_dict.get("name"),
                    target_dict.get("uuid"),
                )
                if target_key in seen_targets[k]:
                    continue
                seen_targets[k].add(target_key)
                merged[k]["data"].append(entry)

    merged_friends = {}
    for k, v in merged.items():
        fr = Friend.from_json(v)
        merged_friends[k] = fr
        print(f"Validated Friend tree '{k}': {len(fr._data)} target chunks.")

    tmp = f"{args.output_json}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(merged_friends, f, cls=DefaultEncoder, indent=2)
        os.replace(tmp, args.output_json)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e

    print(f"Successfully wrote merged friend manifest to {args.output_json}")

    if args.output_done:
        done_dir = os.path.dirname(os.path.abspath(args.output_done))
        os.makedirs(done_dir, exist_ok=True)
        with open(args.output_done, "w") as f:
            f.write("done\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
