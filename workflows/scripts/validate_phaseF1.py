#!/usr/bin/env python3
"""Validate Phase F.1 (main HH4b analysis) outputs: merged .coffea and cutflow YAML."""

import argparse
import math
import os
from pathlib import Path

import yaml


STANDARD_CUTS = ("passJetMult", "passPreSel", "passDiJetMass", "SR", "SB")
TTBAR_D3_CUTS = ("passPreSel", "passDiJetMass", "SR", "SB")
DILEP_CUTS = ("passMuMu", "passElMu")
CUTFLOW_MAPS = ("counts4", "counts3", "counts4_unit", "counts3_unit")
COFFEA_CF_KEYS = {
    "counts4": "cutFlowFourTag",
    "counts3": "cutFlowThreeTag",
    "counts4_unit": "cutFlowFourTagUnitWeight",
    "counts3_unit": "cutFlowThreeTagUnitWeight",
}


def _is_finite(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def _close(a, b):
    return abs(a - b) <= 1e-4 * max(1.0, abs(a), abs(b))


def _exact_val(mname, ds, cut, yaml_val, coffea_out=None):
    """Fall back to unrounded .coffea cutflow when weighted YAML rounds tiny yields to 0.0."""
    if yaml_val == 0 and mname in ("counts3", "counts4") and coffea_out is not None:
        cf = coffea_out.get(COFFEA_CF_KEYS[mname], {})
        if ds in cf and cut in cf[ds]:
            return float(cf[ds][cut])
    return yaml_val


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
    return out


def validate_cutflow(cutflow_path, lines, coffea_out=None):
    if not os.path.exists(cutflow_path):
        raise AssertionError(f"Missing cutflow YAML: {cutflow_path}")
    with open(cutflow_path) as f:
        cf = yaml.safe_load(f) or {}

    c3 = cf.get("counts3", {})
    c4 = cf.get("counts4", {})
    c3_unit = cf.get("counts3_unit", {})
    c4_unit = cf.get("counts4_unit", {})
    for mname, m in zip(CUTFLOW_MAPS, (c4, c3, c4_unit, c3_unit)):
        if not isinstance(m, dict) or not m:
            raise AssertionError(f"Missing or empty '{mname}' in {cutflow_path}")

    lines.append("")
    lines.append("=== Phase F.1 Cutflow (passPreSel & all maps/cuts) ===")
    lines.append(f"{'Dataset':35s} {'threeTag':>12s} {'fourTag':>12s} Status")
    lines.append("-" * 68)

    data_keys = sorted(k for k in c3 if k.startswith("data_"))
    if not data_keys:
        raise AssertionError("Missing required 'data_*' dataset in counts3")
    sig_prefixes = ("GluGluToHHTo4B_", "GluGlutoHHto4B_", "ttHbb_", "ZZ4b", "ZH4b", "ggZH4b")
    sig_keys = sorted(k for k in c3 if k.startswith(sig_prefixes))
    if not sig_keys:
        raise AssertionError(f"Expected at least one signal dataset starting with {sig_prefixes} in counts3")

    non_ttbar_keys = sorted(k for k in (set(c3) | set(c4)) if not k.startswith("TTbar_from_d3"))
    is_lowpt = bool(non_ttbar_keys) and all((c4.get(ds) or {}).get("passPreSel", 0) == 0 for ds in non_ttbar_keys)

    # Verify required standard cuts exist in every map for every dataset
    for mname, m in zip(CUTFLOW_MAPS, (c4, c3, c4_unit, c3_unit)):
        for ds, cuts in m.items():
            if not isinstance(cuts, dict):
                raise AssertionError(f"{mname}.{ds}: expected cut dict, got {type(cuts).__name__}")
            req_cuts = TTBAR_D3_CUTS if ds.startswith("TTbar_from_d3") else STANDARD_CUTS
            missing = [c for c in req_cuts if c not in cuts]
            if missing:
                raise AssertionError(f"{mname}.{ds}: missing cuts {missing}")

    for ds in non_ttbar_keys:
        v3 = (c3.get(ds) or {}).get("passPreSel")
        v4 = (c4.get(ds) or {}).get("passPreSel")
        if is_lowpt:
            v4_jm = (c4.get(ds) or {}).get("passJetMult")
            ok = _is_finite(v3) and _is_finite(v4_jm) and v3 > 0 and v4_jm > 0
            lines.append(f"{ds:35s} {v3 if v3 is not None else 0:12.2f} {v4_jm if v4_jm is not None else 0:12.2f} {'PASS' if ok else 'FAIL'} (4b=passJetMult)")
            if not ok:
                raise AssertionError(f"Invalid lowpt cutflow for {ds}: threeTag(passPreSel)={v3}, fourTag(passJetMult)={v4_jm}")
        else:
            e3 = _exact_val("counts3", ds, "passPreSel", v3, coffea_out)
            e4 = _exact_val("counts4", ds, "passPreSel", v4, coffea_out)
            ok = (
                _is_finite(v3) and v3 >= 0 and _is_finite(e3) and e3 > 0
                and _is_finite(v4) and v4 >= 0 and _is_finite(e4) and e4 > 0
            )
            lines.append(f"{ds:35s} {v3 if v3 is not None else 0:12.2f} {v4 if v4 is not None else 0:12.2f} {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise AssertionError(f"Invalid passPreSel cutflow for {ds}: threeTag={e3}, fourTag={e4}")

        # Verify positive weighted & unweighted counts across cuts and MC weight application
        if ds.startswith(sig_prefixes) or ds.startswith("TTTo"):
            checks = [("counts3", c3), ("counts3_unit", c3_unit)]
            for cut in ("passJetMult", "passPreSel", "passDiJetMass", "SR"):
                cur_checks = list(checks)
                if not is_lowpt or cut == "passJetMult":
                    cur_checks.extend((("counts4", c4), ("counts4_unit", c4_unit)))
                for mname, m in cur_checks:
                    val = m[ds][cut]
                    exact_val = _exact_val(mname, ds, cut, val, coffea_out)
                    if not (_is_finite(val) and _is_finite(exact_val) and val >= 0 and exact_val > 0):
                        raise AssertionError(f"{mname}.{ds}.{cut}: expected > 0, got {exact_val}")
            w_cut = "passJetMult" if is_lowpt else "passPreSel"
            if _close(c4[ds][w_cut], c4_unit[ds][w_cut]):
                raise AssertionError(f"{ds}: expected weighted counts4.{w_cut} != raw counts4_unit.{w_cut}")

    tt_d3_keys = sorted(k for k in c4 if k.startswith("TTbar_from_d3_"))
    if not tt_d3_keys:
        raise AssertionError("Expected at least one 'TTbar_from_d3_*' entry in counts4 (FvT applied)")
    for tk in tt_d3_keys:
        tt_d3 = (c4.get(tk) or {}).get("passPreSel")
        ok_tt_d3 = _is_finite(tt_d3) and tt_d3 > 0
        lines.append(f"{tk:35s} {'N/A':>12s} {tt_d3 if tt_d3 is not None else 0:12.2f} {'PASS' if ok_tt_d3 else 'FAIL'}")
        for cut in TTBAR_D3_CUTS:
            v4 = c4[tk][cut]
            v3 = c3[tk][cut]
            u4 = c4_unit[tk][cut]
            u3 = c3_unit[tk][cut]
            if not (_is_finite(v4) and v4 > 0 and _is_finite(v3) and v3 > 0 and _is_finite(u3) and u3 > 0 and _close(u4, u3)):
                raise AssertionError(
                    f"{tk}.{cut}: invalid TTbar_from_d3 counts (c4={v4}, c3={v3}, u4={u4}, u3={u3})"
                )

    for dk in data_keys:
        for cut in STANDARD_CUTS:
            v4 = c4[dk][cut]
            u4 = c4_unit[dk][cut]
            u3 = c3_unit[dk][cut]
            if not (_is_finite(v4) and _is_finite(u4) and _close(v4, u4)):
                raise AssertionError(f"{dk}.{cut}: data counts4 ({v4}) != counts4_unit ({u4})")
            if not (_is_finite(u3) and u3 > 0):
                raise AssertionError(f"{dk}.{cut}: data counts3_unit must be > 0, got {u3}")
            if not is_lowpt and v4 <= 0:
                raise AssertionError(f"{dk}.{cut}: data counts4 must be > 0, got {v4}")
        if c4[dk]["passJetMult"] <= 0:
            raise AssertionError(f"{dk}.passJetMult: data counts4 must be > 0")
        if not _close(c3[dk]["passJetMult"], c3_unit[dk]["passJetMult"]):
            raise AssertionError(f"{dk}.passJetMult: expected counts3 == counts3_unit")
        for cut in ("passPreSel", "passDiJetMass", "SR", "SB"):
            w3 = c3[dk][cut]
            u3 = c3_unit[dk][cut]
            if not (_is_finite(w3) and w3 > 0 and abs(w3 - u3) > 1e-3):
                raise AssertionError(
                    f"Expected JCM*FvT weighted {dk}.{cut} threeTag != unweighted ({w3} vs {u3})"
                )
        lines.append(f"  {dk} JCM*FvT reweight check (passPreSel..SB): weighted={c3[dk]['passPreSel']:.2f} vs raw={c3_unit[dk]['passPreSel']:.2f}  PASS")

    # Optional dilepton ttbar crosscheck cuts (enabled for processor_HH4b.py)
    has_dilep = any("passMuMu" in (cuts or {}) or "passElMu" in (cuts or {}) for cuts in c4.values())
    if has_dilep:
        for mname, m in zip(CUTFLOW_MAPS, (c4, c3, c4_unit, c3_unit)):
            for ds, cuts in m.items():
                if ds.startswith("TTbar_from_d3"):
                    continue
                for cut in DILEP_CUTS:
                    if cut not in cuts:
                        raise AssertionError(f"{mname}.{ds}: missing dilepton crosscheck cut '{cut}'")
        probe_keys = sorted(k for k in c4 if k.startswith("TTTo2L2Nu")) or data_keys
        for cut in DILEP_CUTS:
            pos = any(
                _is_finite(m[ds][cut]) and m[ds][cut] > 0
                for ds in probe_keys
                for m in (c4, c4_unit)
            )
            if not pos:
                raise AssertionError(f"Expected positive '{cut}' counts in {probe_keys}")
        lines.append(f"  Dilepton ttbar crosscheck ({','.join(DILEP_CUTS)} in {probe_keys[0]}):  PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coffea", required=True, help="Path to merged histAll .coffea")
    parser.add_argument("--cutflow", required=True, help="Path to cutflow .yml")
    parser.add_argument("--output", default=None, help="Optional output validation summary text file")
    args = parser.parse_args()

    lines = []
    coffea_out = validate_coffea(args.coffea, lines)
    validate_cutflow(args.cutflow, lines, coffea_out=coffea_out)
    lines.append("")
    lines.append("Phase F.1 validation PASSED.")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
