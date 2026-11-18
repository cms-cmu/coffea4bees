"""Validate Phase B outputs: fitted JCM YAML(s), NoJCM vs wJCM cutflows, and classifier_inputs friend trees."""
import argparse
import json
import math
import sys
import yaml


CUTFLOW_MAPS = ("counts4", "counts3", "counts4_unit", "counts3_unit")
EXPECTED_CUTS = ("passJetMult", "passPreSel", "passDiJetMass", "SR_woTrig", "SR", "SB_woTrig", "SB")
JCM_REGION_CUTS = ("passPreSel", "passDiJetMass", "SR", "SB")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))


def _check_cutflow_completeness(cf: dict, name: str, failures: list[str]) -> None:
    """Verify all 4 cutflow maps and all 7 expected cuts exist for every non-null dataset."""
    for map_name in CUTFLOW_MAPS:
        section = cf.get(map_name)
        if not isinstance(section, dict) or not section:
            failures.append(f"{name}: missing or empty section '{map_name}'")
            continue
        for ds, cuts in section.items():
            if cuts is None:
                continue
            if not isinstance(cuts, dict):
                failures.append(f"{name}.{map_name}.{ds}: expected dict of cuts, got {type(cuts).__name__}")
                continue
            missing = [c for c in EXPECTED_CUTS if c not in cuts]
            if missing:
                failures.append(f"{name}.{map_name}.{ds}: missing cuts {missing}")


def _check_jcm_files(jcm_paths: list[str], cf_no: dict, failures: list[str], is_lowpt: bool = False) -> None:
    """Validate fitted JCM YAML parameters and NoJCM cutflow counts."""
    print("=== Phase B.1 JCM Parameters ===")
    print(f"{'File':<35} {'Parameter':<30} {'Value':>15} {'Status':<6}")
    print("-" * 90)

    for jcm_path in jcm_paths:
        with open(jcm_path) as f:
            jcm = yaml.safe_load(f) or {}

        short_name = jcm_path.split("/")[-1]
        checked_params = 0
        for k, v in jcm.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                val = float(v)
                ok = math.isfinite(val)
                if not ok:
                    failures.append(f"{short_name}: param {k} is not finite: {v}")
                print(f"{short_name:<35} {k:<30} {val:>15.6g} {'PASS' if ok else 'FAIL':<6}")
                checked_params += 1
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (int, float)) and not isinstance(sv, bool):
                        val = float(sv)
                        ok = math.isfinite(val)
                        if not ok:
                            failures.append(f"{short_name}: param {k}.{sk} is not finite: {sv}")
                        print(f"{short_name:<35} {k + '.' + sk:<30} {val:>15.6g} {'PASS' if ok else 'FAIL':<6}")
                        checked_params += 1

        if checked_params == 0:
            failures.append(f"No numeric parameters found in {jcm_path}")

    _check_cutflow_completeness(cf_no, "NoJCM", failures)

    counts3 = cf_no.get("counts3", {})
    counts4 = cf_no.get("counts4", {})
    counts3_unit = cf_no.get("counts3_unit", {})
    counts4_unit = cf_no.get("counts4_unit", {})
    data_keys = sorted(k for k in (set(counts3) | set(counts4)) if k.startswith("data"))
    if not data_keys:
        failures.append("No data entries found in NoJCM cutflow")
    else:
        print("\n=== Phase B.1 NoJCM Cutflow (passPreSel & unit weights) ===")
        for dk in data_keys:
            three_b = float((counts3.get(dk) or {}).get("passPreSel", 0))
            four_b = float((counts4.get(dk) or {}).get("passPreSel", 0))
            if is_lowpt:
                four_jetmult = float((counts4.get(dk) or {}).get("passJetMult", 0))
                print(f"  {dk:<25} threeTag={three_b:.2f}  fourTag(passPreSel)={four_b:.2f}  passJetMult={four_jetmult:.2f}")
                if three_b <= 0 or four_jetmult <= 0:
                    failures.append(f"{dk} has non-positive NoJCM counts (3b_passPreSel={three_b}, passJetMult={four_jetmult})")
            else:
                print(f"  {dk:<25} threeTag={three_b:.2f}  fourTag={four_b:.2f}")
                if three_b <= 0 or four_b <= 0:
                    failures.append(f"{dk} has non-positive NoJCM passPreSel counts (3b={three_b}, 4b={four_b})")
            for cut in EXPECTED_CUTS:
                v4 = float((counts4.get(dk) or {}).get(cut, 0.0))
                u4 = float((counts4_unit.get(dk) or {}).get(cut, 0.0))
                v3 = float((counts3.get(dk) or {}).get(cut, 0.0))
                u3 = float((counts3_unit.get(dk) or {}).get(cut, 0.0))
                if not _close(v4, u4):
                    failures.append(f"NoJCM {dk}.{cut}: counts4 ({v4}) != counts4_unit ({u4})")
                if not _close(v3, u3):
                    failures.append(f"NoJCM {dk}.{cut}: counts3 ({v3}) != counts3_unit ({u3})")


def _check_wjcm_and_friends(
    friends_path: str,
    friends: dict,
    cf_no: dict,
    cf_w: dict,
    failures: list[str],
    max_files: int = 0,
) -> None:
    """Validate wJCM vs NoJCM cutflows across all 4 maps and classifier_inputs friend trees."""
    import numpy as np
    import uproot

    _check_cutflow_completeness(cf_w, "wJCM", failures)

    print("\n=== Phase B.2 Cutflow Comparison (NoJCM vs wJCM) ===")
    print(f"{'Dataset':<28} {'Cut':<14} {'NoJCM_3b':>10} {'wJCM_3b':>10} {'Status':<6}")
    print("-" * 74)
    w_counts3 = cf_w.get("counts3", {})
    w_counts4 = cf_w.get("counts4", {})
    no_counts3 = cf_no.get("counts3", {})
    no_counts4 = cf_no.get("counts4", {})
    shared_keys = sorted(
        k for k in (set(no_counts3) & set(w_counts3))
        if isinstance(no_counts3.get(k), dict) and isinstance(w_counts3.get(k), dict)
    )
    data_keys = [k for k in shared_keys if k.startswith("data")]
    if not data_keys:
        failures.append("No data entries found in wJCM cutflow")

    for ds in shared_keys:
        for invariant_map in ("counts4", "counts4_unit", "counts3_unit"):
            m_no = (cf_no.get(invariant_map) or {}).get(ds) or {}
            m_w = (cf_w.get(invariant_map) or {}).get(ds) or {}
            for cut in EXPECTED_CUTS:
                v_no = float(m_no.get(cut, 0.0))
                v_w = float(m_w.get(cut, 0.0))
                if not _close(v_no, v_w):
                    failures.append(f"{ds} {invariant_map}.{cut}: changed under JCM ({v_no} -> {v_w})")

    for dk in data_keys:
        for cut in EXPECTED_CUTS:
            v4 = float(((w_counts4.get(dk)) or {}).get(cut, 0.0))
            u4 = float((((cf_w.get("counts4_unit")) or {}).get(dk) or {}).get(cut, 0.0))
            if not _close(v4, u4):
                failures.append(f"wJCM {dk}.{cut}: counts4 ({v4}) != counts4_unit ({u4})")
        for cut in JCM_REGION_CUTS:
            v_no = float((no_counts3.get(dk) or {}).get(cut, 0.0))
            v_w = float((w_counts3.get(dk) or {}).get(cut, 0.0))
            u3_w = float((((cf_w.get("counts3_unit")) or {}).get(dk) or {}).get(cut, 0.0))
            ok = (v_w > 0) and (abs(v_no - v_w) > 1e-3) and (abs(v_w - u3_w) > 1e-3)
            if not ok:
                failures.append(f"{dk} counts3.{cut}: did not change under JCM (NoJCM={v_no}, wJCM={v_w}, raw={u3_w})")
            print(f"{dk:<28} {cut:<14} {v_no:>10.2f} {v_w:>10.2f} {'PASS' if ok else 'FAIL':<6}")

    ci_block = friends.get("HCR_input") or friends.get("HCR_input_lowpt") or (next(iter(friends.values())) if friends else None)
    if not ci_block:
        failures.append(f"Empty friend metadata in {friends_path}")
        return

    branches = ci_block.get("branches", [])
    chunks = [entry[1][0]["chunk"] for entry in ci_block.get("data", [])]
    if not chunks:
        failures.append(f"No chunks found in {friends_path}")
        return

    to_check = chunks[:max_files] if max_files > 0 else chunks
    print(f"\n=== Phase B.2 Classifier Input Friend Trees ({len(to_check)}/{len(chunks)} files, {len(branches)} branches) ===")
    print(f"{'File':<55} {'Entries':>10} {'NaNs':>6} {'Infs':>6} {'Status':<6}")
    print("-" * 88)
    total_entries = 0
    for ch in to_check:
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
        failures.append("Total classifier input entries across checked files is 0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jcm", nargs="+", required=True, help="Fitted JCM YAML file(s)")
    parser.add_argument("--friends", required=True, help="classifier_inputs_friends.json")
    parser.add_argument("--cutflow-nojcm", required=True, help="cutflow_NoJCM.yml")
    parser.add_argument("--cutflow-wjcm", required=True, help="cutflow_wJCM.yml")
    parser.add_argument("--max-files", type=int, default=0, help="Max friend files to open (0 = all)")
    parser.add_argument("-o", "--output", help="Optional path to write validation report")

    args = parser.parse_args()
    with open(args.cutflow_nojcm) as f:
        cf_no = yaml.safe_load(f) or {}
    with open(args.cutflow_wjcm) as f:
        cf_w = yaml.safe_load(f) or {}
    with open(args.friends) as f:
        friends = json.load(f) or {}

    is_lowpt = "HCR_input_lowpt" in friends
    failures: list[str] = []
    _check_jcm_files(args.jcm, cf_no, failures, is_lowpt=is_lowpt)
    _check_wjcm_and_friends(args.friends, friends, cf_no, cf_w, failures, max_files=args.max_files)

    if failures:
        print("\nFAILURES:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("\nPhase B.3 validation PASSED.")
    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as out_f:
            out_f.write("Phase B.3 validation PASSED.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
