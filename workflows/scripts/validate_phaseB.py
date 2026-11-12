"""Validate Phase B.1 (JCM fit & NoJCM cutflow) and Phase B.2 (wJCM cutflow & classifier_inputs friend trees)."""
import argparse
import json
import math
import sys
import yaml


def validate_jcm(jcm_path: str, cutflow_path: str) -> int:
    """Validate fitted JCM YAML parameters and NoJCM cutflow counts."""
    with open(jcm_path) as f:
        jcm = yaml.safe_load(f) or {}
    with open(cutflow_path) as f:
        cf = yaml.safe_load(f) or {}

    failures = []
    print("=== Phase B.1 JCM Parameters ===")
    print(f"{'Parameter':<35} {'Value':>18} {'Status':<6}")
    print("-" * 62)

    # Check scalar numeric parameters in JCM YAML
    checked_params = 0
    for k, v in jcm.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            val = float(v)
            ok = math.isfinite(val)
            status = "PASS" if ok else "FAIL"
            if not ok:
                failures.append(f"JCM param {k} is not finite: {v}")
            print(f"{k:<35} {val:>18.6g} {status:<6}")
            checked_params += 1
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                    val = float(sv)
                    ok = math.isfinite(val)
                    status = "PASS" if ok else "FAIL"
                    if not ok:
                        failures.append(f"JCM param {k}.{sk} is not finite: {sv}")
                    print(f"{k + '.' + sk:<35} {val:>18.6g} {status:<6}")
                    checked_params += 1

    if checked_params == 0:
        failures.append(f"No numeric parameters found in {jcm_path}")

    # Check NoJCM cutflow has non-zero data 3b/4b
    data_keys = [k for k in cf if k.startswith("data")]
    if not data_keys:
        failures.append(f"No data entries found in {cutflow_path}")
    else:
        print("\n=== Phase B.1 NoJCM Cutflow (passPreSel) ===")
        for dk in sorted(data_keys):
            three_b = cf[dk].get("threeTag", {}).get("passPreSel", 0)
            four_b = cf[dk].get("fourTag", {}).get("passPreSel", 0)
            print(f"  {dk:<25} threeTag={three_b:.2f}  fourTag={four_b:.2f}")
            if three_b <= 0 or four_b <= 0:
                failures.append(f"{dk} has non-positive passPreSel counts (3b={three_b}, 4b={four_b})")

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("\nPhase B.1 JCM validation PASSED.")
    return 0


def validate_classifier_inputs(
    friends_path: str,
    cutflow_nojcm_path: str,
    cutflow_wjcm_path: str,
) -> int:
    """Validate classifier_inputs_friends.json ROOT trees and wJCM vs NoJCM cutflow."""
    import numpy as np
    import uproot

    with open(friends_path) as f:
        friends = json.load(f)
    with open(cutflow_nojcm_path) as f:
        cf_no = yaml.safe_load(f) or {}
    with open(cutflow_wjcm_path) as f:
        cf_w = yaml.safe_load(f) or {}

    failures = []

    # 1. Validate wJCM vs NoJCM cutflows
    print("=== Phase B.2 Cutflow Comparison (NoJCM vs wJCM at passPreSel) ===")
    print(f"{'Dataset':<28} {'Tag':<10} {'NoJCM':>12} {'wJCM':>12} {'Status':<6}")
    print("-" * 72)
    for dk in sorted(k for k in cf_w if k.startswith("data")):
        for tag, expect_same in [("threeTag", False), ("fourTag", True)]:
            v_no = float(cf_no.get(dk, {}).get(tag, {}).get("passPreSel", 0.0))
            v_w = float(cf_w.get(dk, {}).get(tag, {}).get("passPreSel", 0.0))
            if expect_same:
                ok = abs(v_no - v_w) <= 1e-4 * max(1.0, abs(v_no))
                if not ok:
                    failures.append(f"{dk} {tag}: fourTag changed under JCM ({v_no} -> {v_w})")
            else:
                ok = (v_w > 0) and (abs(v_no - v_w) > 1e-3)
                if not ok:
                    failures.append(f"{dk} {tag}: threeTag did not change under JCM ({v_no} -> {v_w})")
            print(f"{dk:<28} {tag:<10} {v_no:>12.2f} {v_w:>12.2f} {'PASS' if ok else 'FAIL':<6}")

    # 2. Validate classifier_inputs friend trees
    ci_block = friends.get("HCR_input") or (next(iter(friends.values())) if friends else None)
    if not ci_block:
        failures.append(f"Empty friend metadata in {friends_path}")
    else:
        branches = ci_block.get("branches", [])
        chunks = [entry[1][0]["chunk"] for entry in ci_block.get("data", [])]
        if not chunks:
            failures.append(f"No chunks found in {friends_path}")
        print(f"\n=== Phase B.2 Classifier Input Friend Trees ({len(chunks)} files, {len(branches)} branches) ===")
        print(f"{'File':<55} {'Entries':>10} {'NaNs':>6} {'Infs':>6} {'Status':<6}")
        print("-" * 88)
        total_entries = 0
        for ch in chunks:
            fpath = ch["path"]
            tname = ch.get("name", "Events")
            with uproot.open(fpath) as rf:
                tree = rf[tname]
                n = tree.num_entries
                total_entries += n
                n_nan = 0
                n_inf = 0
                for b in tree.keys():
                    arr = tree[b].array(library="np")
                    if np.issubdtype(arr.dtype, np.floating):
                        n_nan += int(np.isnan(arr).sum())
                        n_inf += int(np.isinf(arr).sum())
                ok = (n > 0) and (n_nan == 0) and (n_inf == 0)
                if not ok:
                    failures.append(f"{fpath}: entries={n}, nan={n_nan}, inf={n_inf}")
                short_p = fpath[-54:] if len(fpath) > 54 else fpath
                print(f"{short_p:<55} {n:>10d} {n_nan:>6d} {n_inf:>6d} {'PASS' if ok else 'FAIL':<6}")

        if total_entries == 0:
            failures.append("Total classifier input entries across all files is 0")

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("\nPhase B.2 wJCM & classifier_inputs validation PASSED.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_jcm = sub.add_parser("jcm", help="Validate Phase B.1 JCM fit and NoJCM cutflow")
    p_jcm.add_argument("--jcm", required=True)
    p_jcm.add_argument("--cutflow", required=True)

    p_ci = sub.add_parser("classifier_inputs", help="Validate Phase B.2 wJCM cutflow and classifier_inputs")
    p_ci.add_argument("--friends", required=True)
    p_ci.add_argument("--cutflow-nojcm", required=True)
    p_ci.add_argument("--cutflow-wjcm", required=True)

    args = parser.parse_args()
    if args.cmd == "jcm":
        return validate_jcm(args.jcm, args.cutflow)
    return validate_classifier_inputs(args.friends, args.cutflow_nojcm, args.cutflow_wjcm)


if __name__ == "__main__":
    sys.exit(main())
