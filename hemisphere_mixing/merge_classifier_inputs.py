#!/usr/bin/env python3
"""Merge per-subsample classifier inputs JSON manifests.

Merges per-subsample HCR classifier input JSON manifests into a single master
JSON file, deduplicating events and generating per-subsample JSON snapshots.

Usage:
    python coffea4bees/hemisphere_mixing/merge_classifier_inputs.py \\
        --inputs output/.../histAll_ttHbb_mixeddata_v*.json \\
        --output-json output/.../classifier_inputs_mixeddata_ttHbb.json \\
        --output-done output/.../merge_all_subsamples.done \\
        --nominal-ci coffea4bees/metadata/datasets/classifier_inputs_ttHbb.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-subsample classifier inputs JSON files into master JSON."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input per-subsample JSON files.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output merged classifier inputs JSON file path.",
    )
    parser.add_argument(
        "--output-done",
        required=True,
        help="Output touch/done marker file path.",
    )
    parser.add_argument(
        "--nominal-ci",
        default=None,
        help="Optional path to nominal reference classifier inputs JSON for branch alignment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    target = args.output_json
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)

    if os.path.exists(target):
        try:
            with open(target, "r") as f:
                curr = json.load(f)
        except Exception:
            curr = {"HCR_input": {"name": "HCR_input", "branches": [], "data": []}}
    else:
        curr = {"HCR_input": {"name": "HCR_input", "branches": [], "data": []}}

    ref_branches = None
    if args.nominal_ci and os.path.exists(args.nominal_ci):
        try:
            with open(args.nominal_ci, "r") as f_ref:
                ref_d = json.load(f_ref)
            if "HCR_input" in ref_d and "branches" in ref_d["HCR_input"]:
                ref_branches = set(ref_d["HCR_input"]["branches"])
                print(f"Loaded {len(ref_branches)} reference branches from {args.nominal_ci}")
        except Exception as e:
            print(f"Warning loading nominal classifier reference {args.nominal_ci}: {e}")
            ref_branches = None

    all_branches = set(curr.get("HCR_input", {}).get("branches", []))

    # Keep non-mixeddata entries. Filter out existing mixeddata entries to avoid accumulation on re-run.
    friend_any_re = re.compile(r"/mix_v\d+_")
    source_any_re = re.compile(r"_v\d+(\.chunk\d+)?\.root$|/v\d+/")

    def is_any_subsample(entry: list) -> bool:
        try:
            friend_path = entry[1][0]["chunk"]["path"]
        except (IndexError, KeyError, TypeError):
            friend_path = ""
        if friend_any_re.search(friend_path):
            return True
        return bool(source_any_re.search(entry[0]["path"]))

    all_entries = [
        e for e in curr.get("HCR_input", {}).get("data", []) if not is_any_subsample(e)
    ]

    for v_idx, jf in enumerate(args.inputs):
        if not os.path.exists(jf):
            print(f"Warning: input JSON {jf} does not exist, skipping.")
            continue
        with open(jf, "r") as f:
            d = json.load(f)
        if "HCR_input" in d:
            if ref_branches is not None:
                d["HCR_input"]["branches"] = sorted(
                    list(set(d["HCR_input"].get("branches", [])).intersection(ref_branches))
                )
            all_branches.update(d["HCR_input"].get("branches", []))
            for entry in d["HCR_input"].get("data", []):
                all_entries.append(entry)

        per_sub_target = target.replace(".json", f"_v{v_idx}.json")
        os.makedirs(os.path.dirname(os.path.abspath(per_sub_target)), exist_ok=True)
        with open(per_sub_target, "w") as f_sub:
            json.dump(d, f_sub, indent=2)

    # Guard against the same (source file, friend chunk) pair being listed twice
    seen = set()
    deduped = []
    for entry in all_entries:
        try:
            key = (
                entry[0]["path"],
                entry[1][0]["chunk"]["path"],
                entry[1][0].get("start"),
                entry[1][0].get("stop"),
            )
        except (IndexError, KeyError, TypeError):
            key = None
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        deduped.append(entry)
    all_entries = deduped

    if ref_branches is not None:
        all_branches = all_branches.intersection(ref_branches)
    curr["HCR_input"]["branches"] = sorted(list(all_branches))
    curr["HCR_input"]["data"] = all_entries

    print(
        f"Writing merged classifier inputs JSON with {len(all_entries)} entries "
        f"and {len(curr['HCR_input']['branches'])} branches to {target}"
    )
    with open(target, "w") as f:
        json.dump(curr, f, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_done)), exist_ok=True)
    with open(args.output_done, "w") as f:
        f.write("done\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
