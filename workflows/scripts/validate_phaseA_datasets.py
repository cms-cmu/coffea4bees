#!/usr/bin/env python3
"""Validate Phase A dataset-production workflow outputs (MakeMixedData and DeClustered)."""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import uproot
import yaml


def _year_entry(mapping: dict, year: str) -> Any:
    """Look up `year` in a YAML mapping allowing either string or integer keys."""
    if not isinstance(mapping, dict):
        return None
    if str(year) in mapping:
        return mapping[str(year)]
    if str(year).isdigit() and int(year) in mapping:
        return mapping[int(year)]
    return None


def _is_finite_pos(val: Any) -> bool:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0.0


def _is_finite(val: Any) -> bool:
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _load_yaml(path: Path, failures: list[str], label: str) -> Any:
    if not path.is_file():
        failures.append(f"{label}: missing YAML file {path}")
        return None
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        failures.append(f"{label}: empty YAML file {path}")
    return data


def _check_nonempty_files(paths: list[Path], failures: list[str], label: str) -> None:
    for p in paths:
        if not p.is_file() or p.stat().st_size == 0:
            failures.append(f"{label}: missing or empty file {p}")


def _check_exists(paths: list[Path], failures: list[str], label: str) -> None:
    for p in paths:
        if not p.exists():
            failures.append(f"{label}: missing path {p}")


def _check_root_trees(
    file_paths: list[str],
    failures: list[str],
    label: str,
    required_branches: list[str] | None = None,
    require_each_nonempty: bool = False,
) -> int:
    """Open ROOT files via uproot, verify Events trees (and optional branches), return total entries."""
    if not file_paths:
        failures.append(f"{label}: empty ROOT file list")
        return 0
    total = 0
    for fp in file_paths:
        if not str(fp).startswith("root://") and not Path(fp).is_file():
            failures.append(f"{label}: missing ROOT file {fp}")
            continue
        try:
            with uproot.open(fp) as rf:
                if "Events" not in rf:
                    failures.append(f"{label}: missing 'Events' tree in {fp}")
                    continue
                tree = rf["Events"]
                n = int(tree.num_entries)
                total += n
                if require_each_nonempty and n <= 0:
                    failures.append(f"{label}: zero Events entries in {fp}")
                if required_branches:
                    keys = set(tree.keys())
                    missing = [b for b in required_branches if b not in keys]
                    if missing:
                        failures.append(f"{label}: missing branches {missing} in {fp}")
        except Exception as exc:
            failures.append(f"{label}: failed reading {fp}: {exc}")
    if total <= 0:
        failures.append(f"{label}: total Events entries is {total} (expected > 0)")
    return total


def _check_cutflow_entry(
    counts_block: dict,
    key: str,
    cuts: tuple[str, ...],
    failures: list[str],
    label: str,
) -> None:
    entry = counts_block.get(key) if isinstance(counts_block, dict) else None
    if not isinstance(entry, dict):
        failures.append(f"{label}: missing cutflow key {key!r}")
        return
    for c in cuts:
        if not _is_finite_pos(entry.get(c)):
            failures.append(f"{label}: {key!r} cut {c!r}={entry.get(c)!r} (expected finite > 0)")


def validate_mixeddata(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    y, e = args.year, args.era
    failures: list[str] = []

    # Mix.1: hemisphere library + stats
    _check_exists([out / "M1" / "published.done"], failures, "Mix.1")
    hemi_lib = _load_yaml(out / "M1" / "hemisphere_library.yml", failures, "Mix.1")
    m1_entries = 0
    if isinstance(hemi_lib, dict):
        lib_files = _year_entry(hemi_lib, y) or []
        m1_entries = _check_root_trees(list(lib_files), failures, "Mix.1 hemi_library", require_each_nonempty=True)

    stats = _load_yaml(out / "M1" / "stats" / f"hemi_statistics_{y}.yml", failures, "Mix.1 stats")
    n_sectors = 0
    if isinstance(stats, dict):
        svars = stats.get("hemi_summary_vars")
        if not isinstance(svars, dict) or not svars:
            failures.append("Mix.1 stats: missing or empty hemi_summary_vars")
        else:
            n_sectors = len(svars)
            for sector, vals in svars.items():
                if not isinstance(vals, dict) or not _is_finite_pos(vals.get("count")):
                    failures.append(f"Mix.1 stats: invalid count in sector {sector}")
                else:
                    for vname, vstats in vals.items():
                        if isinstance(vstats, dict) and (
                            not _is_finite(vstats.get("mean")) or not _is_finite(vstats.get("RMS"))
                        ):
                            failures.append(f"Mix.1 stats: non-finite mean/RMS in {sector}.{vname}")

    # Mix.2: mixeddata_all handoff
    _check_exists([out / "M2" / "published.done"], failures, "Mix.2")
    m2_yml = _load_yaml(out / "handoff" / "mixeddata_all.yml", failures, "Mix.2 handoff")
    m2_entries = 0
    if isinstance(m2_yml, dict):
        y_blk = _year_entry(m2_yml.get("mixeddata_all") or {}, y) or {}
        m2_files = (((y_blk.get("picoAOD") or {}).get(e)) or {}).get("files") or []
        m2_entries = _check_root_trees(list(m2_files), failures, "Mix.2 mixeddata_all", require_each_nonempty=True)

    # Mix.3: mixed JCM + cutflow
    _check_exists([out / "M3" / "published.done"], failures, "Mix.3")
    jcm = _load_yaml(
        out / "M3" / "JCM_mixeddata" / "jetCombinatoricModel_SB_mixeddata.yml",
        failures,
        "Mix.3 JCM",
    )
    if isinstance(jcm, dict):
        ptp = jcm.get("pseudoTagProb")
        ttf = jcm.get("threeTightTagFraction")
        if not (_is_finite(ptp) and 0.0 < float(ptp) < 1.0):
            failures.append(f"Mix.3 JCM: invalid pseudoTagProb={ptp!r}")
        if not _is_finite_pos(ttf):
            failures.append(f"Mix.3 JCM: invalid threeTightTagFraction={ttf!r}")

    m3_cf = _load_yaml(out / "M3" / "cutflow_mixedJCM.yml", failures, "Mix.3 cutflow")
    if isinstance(m3_cf, dict):
        _check_cutflow_entry(m3_cf.get("counts4") or {}, f"mixeddata_all_{y}{e}", ("SR", "SB"), failures, "Mix.3 counts4")

    # Mix.5: ttbar_PSData handoff
    _check_exists([out / "M5" / "published.done"], failures, "Mix.5")
    ps_yml = _load_yaml(out / "handoff" / "ttbar_PSData.yml", failures, "Mix.5 handoff")
    m5_entries = 0
    if isinstance(ps_yml, dict):
        y_blk = _year_entry(ps_yml.get("ttbar_PSData") or {}, y) or {}
        ps_files = (y_blk.get("picoAOD") or {}).get("files") or []
        m5_entries = _check_root_trees(list(ps_files), failures, "Mix.5 ttbar_PSData", require_each_nonempty=False)

    # Mix.6: validation cutflow + plots
    m6_cf = _load_yaml(out / "M6" / "cutflow_validation.yml", failures, "Mix.6 cutflow")
    if isinstance(m6_cf, dict):
        c4 = m6_cf.get("counts4") or {}
        _check_cutflow_entry(c4, f"mixeddata_all_{y}{e}", ("passPreSel", "SR", "SB"), failures, "Mix.6 counts4")
        _check_cutflow_entry(c4, f"ttbar_PSData_{y}", ("passPreSel",), failures, "Mix.6 counts4")
    _check_nonempty_files(
        [out / "M6" / "cutflow_validation.html", out / "M6" / "cutflow_validation.txt"],
        failures,
        "Mix.6 closure page",
    )
    _check_exists([out / "M6" / "plots" / "plots_done.txt"], failures, "Mix.6 plots")

    # Mix.7: signal check
    m7_sig = _load_yaml(out / "M7" / "signal_check.yml", failures, "Mix.7 signal_check")
    if isinstance(m7_sig, dict):
        sig_entry = m7_sig.get(args.signal_dataset)
        if not isinstance(sig_entry, dict):
            failures.append(f"Mix.7 signal_check: missing signal dataset {args.signal_dataset!r}")
        else:
            for k in ("4b", "3b x JCM", "mixed 3b x JCM", "mixed 4b"):
                if not isinstance(sig_entry.get(k), dict):
                    failures.append(f"Mix.7 signal_check: missing {k!r} block")
            for k in ("4b", "mixed 3b x JCM", "mixed 4b"):
                sr_val = (sig_entry.get(k) or {}).get("SR")
                if not _is_finite_pos(sr_val):
                    failures.append(f"Mix.7 signal_check: {k}.SR={sr_val!r} (expected > 0)")
            b4 = sig_entry.get("4b") or {}
            m4 = sig_entry.get("mixed 4b") or {}
            lead_4b = (b4.get("quadJet_selected.lead.mass") or {}).get("frac_100_150")
            lead_m4 = (m4.get("quadJet_selected.lead.mass") or {}).get("frac_100_150")
            if not (_is_finite(lead_4b) and _is_finite(lead_m4) and float(lead_m4) < float(lead_4b)):
                failures.append(f"Mix.7 signal_check: expected mixed 4b lead.mass frac_100_150 ({lead_m4}) < 4b ({lead_4b})")
            srf_4b = b4.get("SR_fraction")
            srf_m4 = m4.get("SR_fraction")
            if not (_is_finite(srf_4b) and _is_finite(srf_m4) and float(srf_m4) < float(srf_4b)):
                failures.append(f"Mix.7 signal_check: expected mixed 4b SR_fraction ({srf_m4}) < 4b ({srf_4b})")

    _check_exists(
        [out / "M7" / "plots_signal" / "plots_done.txt", out / "M7" / "plots_mixeddata" / "plots_done.txt"],
        failures,
        "Mix.7 plots",
    )

    if failures:
        print("MakeMixedData validation FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print(
        f"MakeMixedData validation PASSED: "
        f"Mix.1 hemi={m1_entries} ({n_sectors} sectors), "
        f"Mix.2 mixed={m2_entries}, Mix.5 ttbar_PSData={m5_entries}"
    )
    return 0


def validate_declustered(args: argparse.Namespace) -> int:
    out = Path(args.output_dir)
    y, e, n_seeds = args.year, args.era, int(args.n_seeds)
    failures: list[str] = []

    # Decl.1: splitting library + summary
    _check_exists([out / "D1" / "library" / "published.done"], failures, "Decl.1")
    lib_reg = _load_yaml(out / "D1" / "library" / "splitting_library.yml", failures, "Decl.1 registry")
    d1_entries = 0
    if isinstance(lib_reg, dict):
        lib_files = _year_entry(lib_reg, y) or []
        d1_entries = _check_root_trees(list(lib_files), failures, "Decl.1 library", require_each_nonempty=True)

    d1_sum = _load_yaml(
        out / "D1" / "library" / "summary" / f"splitting_library_summary_{y}.yml",
        failures,
        "Decl.1 summary",
    )
    if isinstance(d1_sum, dict):
        totals = d1_sum.get("totals") or {}
        for k in ("rows", "clean_rows", "exact_types"):
            if not _is_finite_pos(totals.get(k)):
                failures.append(f"Decl.1 summary: totals.{k}={totals.get(k)!r} (expected > 0)")

    # Decl.3: declustered handoffs + per-seed ROOT checks
    _check_exists([out / "D3" / "published.done"], failures, "Decl.3")
    handoff_dir = out / "test" / "handoff"
    mj_yml = _load_yaml(handoff_dir / f"{args.multijet_name}.yml", failures, "Decl.3 multijet handoff")
    ds_yml = _load_yaml(handoff_dir / f"{args.dataset_name}.yml", failures, "Decl.3 combined handoff")
    all_yml = _load_yaml(handoff_dir / f"{args.all_name}.yml", failures, "Decl.3 all handoff")

    mj_templates: list[str] = []
    if isinstance(mj_yml, dict):
        mj = mj_yml.get(args.multijet_name) or {}
        if int(mj.get("nSamples", -1)) != n_seeds:
            failures.append(f"Decl.3 multijet: nSamples={mj.get('nSamples')!r} != {n_seeds}")
        mj_templates = list(((_year_entry(mj, y) or {}).get("picoAOD") or {}).get("files_template") or [])
        if not mj_templates:
            failures.append(f"Decl.3 multijet: empty files_template for {y}")

    d3_entries = 0
    for s in range(n_seeds):
        seed_files = [t.replace("XXX", str(s)) for t in mj_templates]
        d3_entries += _check_root_trees(
            seed_files,
            failures,
            f"Decl.3 seed{s}",
            required_branches=["Jet_lib_index"],
            require_each_nonempty=True,
        )

    if isinstance(ds_yml, dict):
        ds = ds_yml.get(args.dataset_name) or {}
        if int(ds.get("nSamples", -1)) != n_seeds:
            failures.append(f"Decl.3 combined: nSamples={ds.get('nSamples')!r} != {n_seeds}")
        ds_templates = list(((_year_entry(ds, y) or {}).get("picoAOD") or {}).get("files_template") or [])
        if len(ds_templates) <= len(mj_templates):
            failures.append(
                f"Decl.3 combined: files_template len ({len(ds_templates)}) must exceed multijet ({len(mj_templates)})"
            )

    if isinstance(all_yml, dict):
        all_entry = all_yml.get(args.all_name) or {}
        all_files = ((((_year_entry(all_entry, y) or {}).get("picoAOD") or {}).get(e)) or {}).get("files") or []
        if len(all_files) < n_seeds:
            failures.append(f"Decl.3 all: found {len(all_files)} files (expected >= {n_seeds})")
        _check_exists([Path(p) for p in all_files if not str(p).startswith("root://")], failures, "Decl.3 all files")

    # Decl.4: histAll + cutflows
    _check_nonempty_files([out / "D4" / "histAll_declustered.coffea"], failures, "Decl.4 hists")
    d4_cf = _load_yaml(out / "D4" / "cutflow_declustered.yml", failures, "Decl.4 cutflow")
    if isinstance(d4_cf, dict):
        for block_name in ("counts4", "counts4_unit"):
            blk = d4_cf.get(block_name) or {}
            for s in range(n_seeds):
                _check_cutflow_entry(blk, f"syn_v{s}_{y}", ("passPreSel", "SR", "SB"), failures, f"Decl.4 {block_name}")
            _check_cutflow_entry(blk, f"ttbar_PSData_{y}", ("passPreSel",), failures, f"Decl.4 {block_name}")

    # Decl.5: plots, closure table, seed study, SvB bootstrap
    _check_exists(
        [
            out / "D5" / "plots" / "plots_done.txt",
            out / "D5" / "seed_study" / "index.html",
            out / "D5" / "svb_bootstrap" / "index.html",
        ],
        failures,
        "Decl.5 markers",
    )
    _check_nonempty_files(
        [out / "D5" / "cutflow_monitoring.html", out / "D5" / "cutflow_monitoring.txt"],
        failures,
        "Decl.5 closure page",
    )

    seed_sum = _load_yaml(out / "D5" / "seed_study" / "summary.yml", failures, "Decl.5 seed_study")
    if isinstance(seed_sum, dict):
        if int(seed_sum.get("n_seeds", -1)) != n_seeds:
            failures.append(f"Decl.5 seed_study: n_seeds={seed_sum.get('n_seeds')!r} != {n_seeds}")
        if not _is_finite_pos(seed_sum.get("mean_common_events")):
            failures.append(f"Decl.5 seed_study: invalid mean_common_events={seed_sum.get('mean_common_events')!r}")
        overlap = seed_sum.get("mean_draw_overlap")
        if not (_is_finite(overlap) and 0.0 <= float(overlap) < 1.0):
            failures.append(f"Decl.5 seed_study: invalid mean_draw_overlap={overlap!r}")
        ident = seed_sum.get("mean_identical_events")
        if not (_is_finite(ident) and float(ident) < 0.5):
            failures.append(f"Decl.5 seed_study: mean_identical_events={ident!r} (expected < 0.5)")

    svb_sum = _load_yaml(out / "D5" / "svb_bootstrap" / "summary.yml", failures, "Decl.5 svb_bootstrap")
    if isinstance(svb_sum, dict):
        if int(svb_sum.get("n_seeds", -1)) != n_seeds:
            failures.append(f"Decl.5 svb_bootstrap: n_seeds={svb_sum.get('n_seeds')!r} != {n_seeds}")
        for k in ("sr_entries_all_seeds", "events_per_seed", "bins_used", "n_eff_pooled"):
            if not _is_finite_pos(svb_sum.get(k)):
                failures.append(f"Decl.5 svb_bootstrap: {k}={svb_sum.get(k)!r} (expected finite > 0)")

    if failures:
        print("DeClustered validation FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print(
        f"DeClustered validation PASSED: "
        f"Decl.1 library={d1_entries} rows ({d1_sum['totals']['exact_types']} exact types), "
        f"Decl.3 declustered={d3_entries} events across {n_seeds} seeds, "
        f"Decl.5 n_eff_pooled={svb_sum['n_eff_pooled']} (bins_used={svb_sum['bins_used']})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mix = sub.add_parser("mixeddata", help="Validate MakeMixedData workflow outputs")
    p_mix.add_argument("--output-dir", required=True)
    p_mix.add_argument("--year", default="UL18")
    p_mix.add_argument("--era", default="A")
    p_mix.add_argument("--signal-dataset", default="GluGluToHHTo4B_cHHH1")

    p_dec = sub.add_parser("declustered", help="Validate DeClustered workflow outputs")
    p_dec.add_argument("--output-dir", required=True)
    p_dec.add_argument("--year", default="2024")
    p_dec.add_argument("--era", default="C")
    p_dec.add_argument("--n-seeds", type=int, default=2)
    p_dec.add_argument("--multijet-name", default="synthetic_data_lib16_2024_multijet")
    p_dec.add_argument("--dataset-name", default="synthetic_data_lib16_2024_4b")
    p_dec.add_argument("--all-name", default="mixeddata_all_declib16_2024")

    args = parser.parse_args()
    if args.cmd == "mixeddata":
        return validate_mixeddata(args)
    return validate_declustered(args)


if __name__ == "__main__":
    sys.exit(main())
