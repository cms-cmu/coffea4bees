"""Validate Phase A skimmer counts/weights and trigger-weight friend trees against known references."""
import argparse
import json
import sys
import yaml


def _rel_diff(actual: float, expected: float) -> float:
    if expected == 0:
        return abs(actual - expected)
    return abs(actual - expected) / abs(expected)


def validate_skimmer(new_datasets_path: str, known_path: str, tol: float) -> int:
    """Compare new_datasets.yml skimmer counts and sumw/sumw2 against known reference."""
    with open(new_datasets_path) as f:
        new_data = yaml.safe_load(f) or {}
    with open(known_path) as f:
        known = yaml.safe_load(f) or {}

    expected_skims = known.get("skimmer", {})
    actual_ds = new_data.get("datasets", {})
    failures = []

    print(f"{'Dataset':<25} {'Year':<8} {'Metric':<14} {'Expected':>14} {'Actual':>14} {'RelDiff':>10} {'Status':<6}")
    print("-" * 97)

    for ds, years_map in expected_skims.items():
        for year, exp in years_map.items():
            pico = actual_ds.get(ds, {}).get(year, {}).get("picoAOD", {})
            if not pico:
                failures.append(f"Missing picoAOD entry for {ds} / {year}")
                print(f"{ds:<25} {year:<8} {'picoAOD':<14} {'present':>14} {'MISSING':>14} {'-':>10} FAIL")
                continue

            for key in ("total_events", "saved_events", "sumw", "sumw2"):
                if key not in exp:
                    continue
                e_val = float(exp[key])
                a_val = float(pico.get(key, float("nan")))
                diff = _rel_diff(a_val, e_val)
                ok = diff <= tol
                status = "PASS" if ok else "FAIL"
                if not ok:
                    failures.append(f"{ds}/{year} {key}: expected {e_val}, got {a_val} (rel_diff={diff:.6g} > {tol})")
                print(f"{ds:<25} {year:<8} {key:<14} {e_val:>14.4f} {a_val:>14.4f} {diff:>10.2e} {status:<6}")

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print("\nPhase A.1 skimmer count/weight validation PASSED.")
    return 0


def validate_trigweights(friends_path: str, known_path: str, tol: float) -> int:
    """Validate trigger_weights_friends.json entry count, branches, and weight distributions."""
    import numpy as np
    import uproot

    with open(friends_path) as f:
        friends = json.load(f)
    with open(known_path) as f:
        known = yaml.safe_load(f) or {}

    tw_ref = known.get("trigger_weights", {})
    expected_entries = tw_ref.get("num_entries")
    expected_branches = tw_ref.get("branches", ["Data", "MC"])
    branch_means_ref = tw_ref.get("branch_means", {})

    tw_block = friends.get("trigWeight")
    if not tw_block:
        print("FAIL: Missing 'trigWeight' key in trigger_weights_friends.json")
        return 1

    branches = tw_block.get("branches", [])
    if set(expected_branches) - set(branches):
        print(f"FAIL: Expected branches {expected_branches}, got {branches}")
        return 1

    chunks = [entry[1][0]["chunk"] for entry in tw_block.get("data", [])]
    if not chunks:
        print("FAIL: No chunks found in trigger_weights_friends.json")
        return 1

    total_entries = 0
    failures = []
    print(f"{'File':<45} {'Branch':<10} {'Entries':>9} {'NaN':>5} {'Inf':>5} {'Min':>9} {'Max':>9} {'Mean':>9} {'Status':<6}")
    print("-" * 115)

    for ch in chunks:
        fpath = ch["path"]
        tname = ch.get("name", "Events")
        with uproot.open(fpath) as rf:
            tree = rf[tname]
            n = tree.num_entries
            total_entries += n
            for b in expected_branches:
                arr = tree[b].array(library="np")
                n_nan = int(np.isnan(arr).sum())
                n_inf = int(np.isinf(arr).sum())
                vmin = float(np.min(arr)) if n > 0 else float("nan")
                vmax = float(np.max(arr)) if n > 0 else float("nan")
                vmean = float(np.mean(arr)) if n > 0 else float("nan")
                ok = (n > 0) and (n_nan == 0) and (n_inf == 0) and (0.0 < vmean < 2.0)
                if b in branch_means_ref:
                    ok = ok and (_rel_diff(vmean, float(branch_means_ref[b])) <= tol)
                status = "PASS" if ok else "FAIL"
                if not ok:
                    failures.append(f"{fpath}:{b} invalid stats (n={n}, nan={n_nan}, inf={n_inf}, mean={vmean})")
                short_path = fpath[-44:] if len(fpath) > 44 else fpath
                print(f"{short_path:<45} {b:<10} {n:>9d} {n_nan:>5d} {n_inf:>5d} {vmin:>9.4f} {vmax:>9.4f} {vmean:>9.4f} {status:<6}")

    if expected_entries is not None and total_entries != int(expected_entries):
        failures.append(f"Total entries mismatch: expected {expected_entries}, got {total_entries}")

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return 1
    print(f"\nPhase A.2 trigger-weights validation PASSED (total_entries={total_entries}).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_skim = sub.add_parser("skimmer", help="Validate Phase A.1 skimmer new_datasets.yml")
    p_skim.add_argument("--new-datasets", required=True)
    p_skim.add_argument("--known", required=True)
    p_skim.add_argument("--tol", type=float, default=1e-3)

    p_tw = sub.add_parser("trigweights", help="Validate Phase A.2 trigger_weights_friends.json")
    p_tw.add_argument("--friends", required=True)
    p_tw.add_argument("--known", required=True)
    p_tw.add_argument("--tol", type=float, default=1e-3)

    args = parser.parse_args()
    if args.cmd == "skimmer":
        return validate_skimmer(args.new_datasets, args.known, args.tol)
    return validate_trigweights(args.friends, args.known, args.tol)


if __name__ == "__main__":
    sys.exit(main())
