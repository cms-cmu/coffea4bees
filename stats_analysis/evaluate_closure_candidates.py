#!/usr/bin/env python3
"""Evaluates two-stage closure candidates across rebin values and outputs closure_summary.json."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate two-stage closure candidates.")
    parser.add_argument("--status-files", nargs="+", required=True, help="List of closure_status.json files")
    parser.add_argument("--channel", default="ttHbb", help="Analysis channel name")
    parser.add_argument("--output", "-o", required=True, help="Output closure_summary.json path")
    return parser.parse_args()


def parse_spurious_signal_from_dir(status_path: Path) -> tuple[bool | None, float | None]:
    """Fallback: checks directory for closure log if spurious signal was not recorded in json."""
    cand_dir = status_path.parent
    for log_path in cand_dir.glob("*.log"):
        try:
            with open(log_path, "r") as f:
                content = f.read()
            if "SS f-test" in content:
                for line in content.splitlines():
                    if "SS f-test =" in line:
                        m = re.search(r"SS f-test\s*=\s*(\d+)%", line)
                        fprob = float(m.group(1)) / 100.0 if m else None
                        if "Do not need to include spurious signal systematic" in line:
                            return True, fprob
                        elif "STRONG EVIDENCE" in line:
                            return False, fprob
        except Exception:
            pass
    return None, None


def main():
    args = parse_args()

    passing_rebins = []
    summary_records = []
    log_lines = []

    log_lines.append("=" * 78)
    log_lines.append(f"TWO-STAGE CLOSURE EVALUATION SUMMARY ({args.channel}):")
    log_lines.append(f"{'Rebin':<8} {'Bins':<8} {'Variance':<16} {'Bias':<16} {'Spurious Sig':<16} {'Overall':<10}")
    log_lines.append("-" * 78)

    # Sort status files by rebin if possible
    def get_rebin_sort_key(p_str: str) -> int:
        m = re.search(r"rebin(\d+)", p_str)
        return int(m.group(1)) if m else 9999

    sorted_files = sorted(args.status_files, key=get_rebin_sort_key)

    for sf_str in sorted_files:
        sf = Path(sf_str)
        try:
            with open(sf, "r") as jf:
                data = json.load(jf)
        except Exception as e:
            data = {"passed": False, "rebin": None, "error": str(e)}

        rebin_val = data.get("rebin")
        n_bins = data.get("n_bins")
        if n_bins is None and rebin_val:
            try:
                n_bins = 240 // int(rebin_val)
            except Exception:
                n_bins = "?"

        # Variance
        v_passed = data.get("variance_passed", False)
        v_basis = data.get("multijet_basis")
        var_basis_str = f"(basis {v_basis})" if v_basis is not None else ""
        var_p = f"{'PASS' if v_passed else 'FAIL'} {var_basis_str}".strip()

        # Bias
        b_passed = data.get("bias_passed", False)
        b_basis = data.get("selected_basis")
        bias_basis_str = f"(basis {b_basis})" if b_passed and b_basis is not None else ("(no basis)" if not b_passed else "")
        bias_p = f"{'PASS' if b_passed else 'FAIL'} {bias_basis_str}".strip()

        # Spurious Signal
        ss_val = data.get("spurious_signal_passed")
        ss_basis_val = data.get("spurious_signal_basis")
        ss_fprob_val = data.get("spurious_signal_fprob")
        if ss_val is None and sf.exists():
            parsed_pass, parsed_fprob = parse_spurious_signal_from_dir(sf)
            if parsed_pass is not None:
                ss_val = parsed_pass
                ss_fprob_val = parsed_fprob
                ss_basis_val = b_basis
                data["spurious_signal_passed"] = ss_val
                data["spurious_signal_basis"] = ss_basis_val
                data["spurious_signal_fprob"] = ss_fprob_val

        ss_basis_str = f"(basis {ss_basis_val if ss_basis_val is not None else b_basis})" if ss_val is not None else ""
        if ss_val is True:
            ss_p = f"PASS {ss_basis_str}".strip()
        elif ss_val is False:
            ss_p = f"FAIL {ss_basis_str}".strip()
        else:
            ss_p = "-"

        overall = "PASS" if data.get("passed") else "FAIL"
        log_lines.append(f"{str(rebin_val):<8} {str(n_bins):<8} {var_p:<16} {bias_p:<16} {ss_p:<16} {overall:<10}")

        summary_records.append(data)
        if data.get("passed"):
            passing_rebins.append(int(rebin_val))

    log_lines.append("=" * 78)
    log_lines.append(f"Passing rebins eligible for Combine: {passing_rebins}")
    summary_text = "\n".join(log_lines)
    print("\n" + summary_text + "\n")

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as out_f:
        json.dump({
            "passing_rebins": passing_rebins,
            "all_evaluations": summary_records
        }, out_f, indent=2)

    print(f"Successfully generated closure summary JSON: {out_file}")


if __name__ == "__main__":
    main()
