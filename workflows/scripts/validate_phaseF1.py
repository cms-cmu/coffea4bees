#!/usr/bin/env python3
"""Validate Phase F.1 (main HH4b analysis) outputs: merged .coffea and cutflow YAML."""

import argparse
import math
import os
from pathlib import Path

import yaml


def _is_finite(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def validate_coffea(coffea_path, lines):
    from coffea.util import load

    if not os.path.exists(coffea_path):
        raise AssertionError(f"Missing merged coffea output: {coffea_path}")
    out = load(coffea_path)
    if not isinstance(out, dict) or not out:
        raise AssertionError(f"Empty or invalid coffea output: {coffea_path}")
    hists = out.get("hists", {})
    if not hists:
        raise AssertionError(f"No histograms found in {coffea_path}")
    lines.append(f"=== Phase F.1 Coffea Output ({len(hists)} histograms) ===")
    lines.append(f"  File: {coffea_path} (size={os.path.getsize(coffea_path)} bytes)  PASS")


def validate_cutflow(cutflow_path, lines):
    if not os.path.exists(cutflow_path):
        raise AssertionError(f"Missing cutflow YAML: {cutflow_path}")
    with open(cutflow_path) as f:
        cf = yaml.safe_load(f) or {}

    c3 = cf.get("counts3", {})
    c4 = cf.get("counts4", {})
    c3_unit = cf.get("counts3_unit", {})
    if not c3 or not c4:
        raise AssertionError(f"Missing counts3/counts4 in {cutflow_path}")

    lines.append("")
    lines.append("=== Phase F.1 Cutflow (passPreSel) ===")
    lines.append(f"{'Dataset':35s} {'threeTag':>12s} {'fourTag':>12s} Status")
    lines.append("-" * 68)

    for ds in sorted(set(c3) | set(c4)):
        v3 = c3.get(ds, {}).get("passPreSel")
        v4 = c4.get(ds, {}).get("passPreSel")
        ok = _is_finite(v3) and _is_finite(v4) and v3 > 0 and v4 > 0
        lines.append(f"{ds:35s} {v3 if v3 is not None else 0:12.2f} {v4 if v4 is not None else 0:12.2f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError(f"Invalid passPreSel cutflow for {ds}: threeTag={v3}, fourTag={v4}")

    if "data_UL18A" in c3 and "data_UL18A" in c3_unit:
        w3 = c3["data_UL18A"].get("passPreSel", 0)
        u3 = c3_unit["data_UL18A"].get("passPreSel", 0)
        if not (w3 > 0 and u3 > 0 and w3 != u3):
            raise AssertionError(
                f"Expected JCM*FvT weighted data_UL18A threeTag != unweighted ({w3} vs {u3})"
            )
        lines.append(f"  data_UL18A JCM*FvT reweight check: weighted={w3:.2f} vs raw={u3:.2f}  PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coffea", required=True, help="Path to merged histAll .coffea")
    parser.add_argument("--cutflow", required=True, help="Path to cutflow .yml")
    parser.add_argument("--output", default=None, help="Optional output validation summary text file")
    args = parser.parse_args()

    lines = []
    validate_coffea(args.coffea, lines)
    validate_cutflow(args.cutflow, lines)
    lines.append("")
    lines.append("Phase F.1 validation PASSED.")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
