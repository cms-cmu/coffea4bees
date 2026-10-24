#!/usr/bin/env python3
"""
validate_f1_normalization.py

Validation tool for Stage F_1 background model normalizations.
Inspects histAll_mixeddata_bkgs.coffea to ensure that the multijet background
derived from 3-tag collision data (with JCM and FvT weights) has physical yields
and does not explode before downstream closure fits and statistical steps run.

Fails with exit code 1 if yields or reweighting ratios fall outside physical bounds.
"""

import argparse
import os
import sys
import coffea.util
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Stage F_1 data background normalization in histAll_mixeddata_bkgs.coffea"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to histAll_mixeddata_bkgs.coffea",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to write validation status token on success",
    )
    parser.add_argument(
        "--min-sb-yield",
        type=float,
        default=20000.0,
        help="Minimum acceptable SB yield for multijet data background (default: 20000)",
    )
    parser.add_argument(
        "--max-sb-yield",
        type=float,
        default=200000.0,
        help="Maximum acceptable SB yield for multijet data background (default: 200000)",
    )
    parser.add_argument(
        "--max-sr-yield",
        type=float,
        default=800000.0,
        help="Maximum acceptable SR yield for multijet data background (default: 800000)",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.40,
        help="Minimum acceptable ratio of SvB_MA / SvB_MA_noFvT in SB (default: 0.40)",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=2.50,
        help="Maximum acceptable ratio of SvB_MA / SvB_MA_noFvT in SB (default: 2.50)",
    )
    return parser.parse_args()


def get_yield(hist_obj, process="data", region="SB"):
    """Extract scalar sum of yields for a given process and region from a hist."""
    try:
        sub_h = hist_obj[{"process": process, "region": region}]
        return float(sub_h.values().sum())
    except Exception as e:
        return None


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.input}...")
    data = coffea.util.load(args.input)
    hists = data.get("hists", data)

    # Locate nominal SvB_MA and noFvT histograms
    h_nom = hists.get("SvB_MA.ps") or hists.get("SvB_MA") or hists.get("SvB.ps")
    h_nofvt = hists.get("SvB_MA_noFvT.ps") or hists.get("SvB_MA_noFvT") or hists.get("SvB_noFvT.ps")

    if h_nom is None:
        print("ERROR: Could not find SvB_MA.ps or SvB_MA in histogram file!", file=sys.stderr)
        sys.exit(1)

    failures = []
    summary_lines = []

    summary_lines.append("=" * 80)
    summary_lines.append("Stage F_1 Multijet Background Normalization Validation")
    summary_lines.append("=" * 80)

    # 1. Check nominal data background in Sideband (SB)
    sb_nom = get_yield(h_nom, process="data", region="SB")
    sb_nofvt = get_yield(h_nofvt, process="data", region="SB") if h_nofvt is not None else None

    if sb_nom is None:
        failures.append("Could not extract process='data' region='SB' yield from SvB_MA.")
        summary_lines.append("  [FAIL] process='data' region='SB' missing from SvB_MA")
    else:
        sb_status = "PASS"
        if sb_nom < args.min_sb_yield or sb_nom > args.max_sb_yield:
            sb_status = "FAIL"
            failures.append(
                f"Data SB yield {sb_nom:,.2f} is outside acceptable bounds "
                f"[{args.min_sb_yield:,.0f}, {args.max_sb_yield:,.0f}]."
            )
        summary_lines.append(
            f"  [{sb_status}] SB Yield (data, SvB_MA):         {sb_nom:>14,.2f}  "
            f"(bounds: [{args.min_sb_yield:,.0f}, {args.max_sb_yield:,.0f}])"
        )

        if sb_nofvt is not None and sb_nofvt > 0:
            ratio_sb = sb_nom / sb_nofvt
            r_status = "PASS"
            if ratio_sb < args.min_ratio or ratio_sb > args.max_ratio:
                r_status = "FAIL"
                failures.append(
                    f"Data SB FvT/noFvT ratio {ratio_sb:.3f} is outside bounds "
                    f"[{args.min_ratio:.2f}, {args.max_ratio:.2f}]."
                )
            summary_lines.append(
                f"  [{r_status}] SB Yield (data, SvB_MA_noFvT):   {sb_nofvt:>14,.2f}"
            )
            summary_lines.append(
                f"  [{r_status}] SB Ratio (SvB_MA / noFvT):      {ratio_sb:>14.3f}  "
                f"(bounds: [{args.min_ratio:.2f}, {args.max_ratio:.2f}])"
            )

    # 2. Check nominal data background in Signal Region (SR)
    sr_nom = get_yield(h_nom, process="data", region="SR")
    if sr_nom is not None:
        sr_status = "PASS"
        if sr_nom > args.max_sr_yield:
            sr_status = "FAIL"
            failures.append(
                f"Data SR yield {sr_nom:,.2f} exceeds maximum acceptable bound "
                f"{args.max_sr_yield:,.0f}."
            )
        summary_lines.append(
            f"  [{sr_status}] SR Yield (data, SvB_MA):         {sr_nom:>14,.2f}  "
            f"(max allowed: {args.max_sr_yield:,.0f})"
        )

    # 3. Check Subsample Models (v0 .. v15)
    subsample_keys = [k for k in hists.keys() if k.startswith("SvB_MA_v") and k.endswith(".ps")]
    if subsample_keys:
        summary_lines.append("-" * 80)
        summary_lines.append(f"Subsample Validation ({len(subsample_keys)} models detected):")
        for sk in sorted(subsample_keys):
            v_name = sk.replace("SvB_MA_", "").replace(".ps", "")
            h_sub = hists[sk]
            sub_sb = get_yield(h_sub, process="data", region="SB")
            if sub_sb is not None:
                sub_status = "PASS"
                if sub_sb < args.min_sb_yield or sub_sb > args.max_sb_yield:
                    sub_status = "FAIL"
                    failures.append(f"Subsample {v_name} SB yield {sub_sb:,.2f} outside bounds.")
                summary_lines.append(f"    [{sub_status}] Subsample {v_name:<4s} SB Yield: {sub_sb:>14,.2f}")

    summary_lines.append("=" * 80)

    # Print summary
    for line in summary_lines:
        print(line)

    if failures:
        print("\n*** FATAL: Stage F_1 Normalization Validation FAILED ***", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nAborting workflow: background yields deviate significantly from physical targets.\n"
            "Verify that 'pairing: can_ttH' is enabled for ttHbb and that FvT extrapolation is physical.",
            file=sys.stderr
        )
        sys.exit(1)

    print("\nStage F_1 Normalization Validation PASSED.")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f_out:
            f_out.write("\n".join(summary_lines) + "\nValidation status: PASSED\n")
        print(f"Status token written to: {args.output}")


if __name__ == "__main__":
    main()
