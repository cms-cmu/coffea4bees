"""
Stage 3 of the tt+bb stitching: turn the stage-2 skimmer output into a dataset
YAML entry, and verify the genWeight closure on the produced files.

For each channel the two stage-2 outputs (inclusive-minus-ttB and TTbb-ttB-only)
are merged into a *single* dataset entry whose ``picoAOD.files`` is the union of
both file lists. The entry inherits the inclusive sample's cross section and its
generator-level ``sumw``, so the analysis normalisation

    w = genWeight * lumi * xs / genEventSumw

is numerically identical to running on the unmodified inclusive sample, while
the tt+B events and their shapes come from the dedicated TTbb sample.

Which closure is asserted depends on the region stage 1 measured k over, recorded
as ``stitch[channel][era]["selection"]`` in the factors JSON:

``picoaod``
    The identity that defines a picoAOD-level stitch:

        sumw(stitched picoAOD) == sumw(inclusive picoAOD)

    checked from the ``stitch_sumw_*`` counters the processor emitted.

``analysis``
    k was measured over the analysis preselection, so the picoAOD-level identity
    above is **deliberately not** satisfied and asserting it would reject a
    correct sample. What holds instead is

        k * sumw_ttB(TTbb, analysis sel) == sumw_ttB(inclusive, analysis sel)

    i.e. the tt+B yield is preserved where the analysis operates. That identity is
    exact by construction in stage 1, so what stage 3 can usefully verify here is
    that stage 2 actually applied the intended factors: the TTbb side scaled by
    exactly k and the inclusive side left unscaled. The picoAOD-level ratio is
    still reported, as information rather than as a gate.

Run from the barista root::

    ./run_container python coffea4bees/skimmer/tools/make_stitched_dataset.py \\
        -i coffea4bees/skimmer/metadata/picoaod_datasets_ttbb_stitched.yml \\
        -f coffea4bees/skimmer/metadata/ttbb_stitch_factors.json \\
        -m coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT.yml \\
        -o coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT_stitched.yml
"""

import argparse
import json
import os
import sys

import yaml

# Relative agreement required between the stitched and inclusive genWeight sums.
# The scaled genWeight is stored as float32, so a small rounding drift is
# expected; 1e-6 is far below it and far above float32 noise.
CLOSURE_TOL = 1e-6


def missing_on_disk(files):
    """Return the subset of `files` that does not exist on the filesystem.

    Stage 2's merge has twice reported success while silently failing to write
    some of its output: the first Run 3 production left picoAOD.chunk143.root
    absent from TTToHadronic_2024 (332 of 333), and a full regeneration ended
    with 6 of 333 missing -- no KilledWorker, no MemoryError, every intermediate
    cleaned, and "JOB EXECUTION COMPLETED SUCCESSFULLY" in the log. Because the
    merge deletes the previous output before writing the new one, a failed write
    leaves a hole rather than a stale file.

    Nothing else catches this. The closure gate compares numbers carried in the
    stage-2 metadata, which describe what the merge *intended* to write, so it
    passes cleanly while the dataset on disk is short. A stitched entry inherits
    the inclusive generator-level sumw, so every absent chunk biases that era's
    yield low in direct proportion -- the 6 missing files were ~1.8% of the era.
    """
    return [f for f in files
            if not os.path.exists(f.replace("root://cmseos.fnal.gov/", "/eos/uscms"))]


def build(factors, skim_out, datasets, suffix, tol=CLOSURE_TOL):
    out = {}
    report = []
    failures = []

    for channel, eras in sorted(factors["stitch"].items()):
        for era, spec in sorted(eras.items()):
            incl_name, ttbb_name = spec["inclusive"], spec["ttbb"]
            incl_key, ttbb_key = f"{incl_name}_{era}", f"{ttbb_name}_{era}"

            missing = [k for k in (incl_key, ttbb_key) if k not in skim_out]
            if missing:
                print(f"[skip] {channel} {era}: not in skimmer output: {missing}")
                continue

            a, b = skim_out[incl_key], skim_out[ttbb_key]
            for key, blob in ((incl_key, a), (ttbb_key, b)):
                for field in (
                    "stitch_sumw_in",
                    "stitch_sumw_selected_scaled",
                    "stitch_n_out",
                    "files",
                ):
                    if field not in blob:
                        raise SystemExit(f"{key}: missing {field!r} in skimmer output")

            # --- the closure test -------------------------------------------------
            sumw_before = a["stitch_sumw_in"]
            sumw_after = a["stitch_sumw_selected_scaled"] + b["stitch_sumw_selected_scaled"]
            rel = (sumw_after - sumw_before) / sumw_before

            if spec.get("selection") == "analysis":
                # picoAOD-level preservation is not the property being claimed;
                # verify instead that stage 2 applied the intended scale factors.
                for key, blob in ((incl_key, a), (ttbb_key, b)):
                    if "stitch_sumw_selected_raw" not in blob:
                        raise SystemExit(
                            f"{key}: missing 'stitch_sumw_selected_raw', needed to "
                            f"verify the applied scale for an analysis-level k"
                        )
                applied_ttbb = (
                    b["stitch_sumw_selected_scaled"] / b["stitch_sumw_selected_raw"]
                    if b["stitch_sumw_selected_raw"] else float("nan")
                )
                applied_incl = (
                    a["stitch_sumw_selected_scaled"] / a["stitch_sumw_selected_raw"]
                    if a["stitch_sumw_selected_raw"] else float("nan")
                )
                d_ttbb = abs(applied_ttbb / spec["scale"] - 1.0)
                d_incl = abs(applied_incl - 1.0)
                ok = d_ttbb <= tol and d_incl <= tol
                if not ok:
                    failures.append((channel, era, applied_ttbb, spec["scale"],
                                     max(d_ttbb, d_incl)))
                # report the scale check; keep the picoAOD ratio for the entry
                report.append((channel, era, spec["scale"], applied_ttbb,
                               max(d_ttbb, d_incl), ok))
            else:
                ok = abs(rel) <= tol
                if not ok:
                    failures.append((channel, era, sumw_before, sumw_after, rel))
                report.append((channel, era, sumw_before, sumw_after, rel, ok))

            # --- the merged dataset entry ----------------------------------------
            src = datasets[incl_name]
            src_pico = src[era]["picoAOD"]
            name = f"{incl_name}{suffix}"

            files = sorted(a["files"]) + sorted(b["files"])
            if len(set(files)) != len(files):
                raise SystemExit(f"{name} {era}: duplicate files in merged list")

            entry = out.setdefault(name, {})
            entry["xs"] = src.get("xs")
            entry[era] = {
                "picoAOD": {
                    "files": files,
                    # Generator-level normalisation inherited from the inclusive
                    # sample: this is what runner.py reads as genEventSumw.
                    "sumw": src_pico["sumw"],
                    "sumw2": src_pico.get("sumw2"),
                    "total_events": src_pico.get("total_events"),
                    "count": src_pico.get("count"),
                    # Events actually stored in the stitched files.
                    "saved_events": int(a["stitch_n_out"] + b["stitch_n_out"]),
                    "stitch": {
                        "inclusive": incl_name,
                        "ttbb": ttbb_name,
                        "from_inclusive": "ttC + ttLF (genTtbarId notB)",
                        "from_ttbb": "ttB",
                        "genWeight_scale_on_ttbb": spec["scale"],
                        "sumw_picoaod_inclusive_before": sumw_before,
                        "sumw_picoaod_stitched_after": sumw_after,
                        "closure_rel": rel,
                        # region k was measured over; with "analysis" the
                        # closure_rel above is expected to be non-zero by design
                        "k_selection": spec.get("selection", "picoaod"),
                        "closure_analysis_rel": spec.get("closure_analysis_rel"),
                        "analysis_eff_inclusive": spec.get("analysis_eff_inclusive"),
                        "analysis_eff_ttbb": spec.get("analysis_eff_ttbb"),
                        "n_from_inclusive": int(a["stitch_n_out"]),
                        "n_from_ttbb": int(b["stitch_n_out"]),
                    },
                }
            }

    return out, report, failures


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", required=True, help="stage-2 picoaod_datasets YAML")
    ap.add_argument(
        "-f", "--factors",
        default="coffea4bees/skimmer/metadata/ttbb_stitch_factors.json",
        help="stage-1 scale factor JSON",
    )
    ap.add_argument(
        "-m", "--metadata",
        default="coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT.yml",
        help="dataset YAML holding the original inclusive entries",
    )
    ap.add_argument("-o", "--output", required=True, help="output dataset YAML")
    ap.add_argument(
        "--suffix", default="_stitched", help="appended to the inclusive dataset name"
    )
    ap.add_argument("--tolerance", type=float, default=CLOSURE_TOL)
    ap.add_argument(
        "--allow-closure-failure", action="store_true",
        help="write the YAML even if the genWeight closure fails (debugging only)",
    )
    ap.add_argument(
        "--allow-missing-files", action="store_true",
        help="write the YAML even if some stitched picoAODs are absent from disk "
             "(debugging only; the entry will under-count that era)",
    )
    args = ap.parse_args(argv)

    factors = json.load(open(args.factors))
    skim_out = yaml.safe_load(open(args.input))
    datasets = yaml.safe_load(open(args.metadata))

    out, report, failures = build(
        factors, skim_out, datasets, args.suffix, args.tolerance
    )

    analysis_k = any(
        e.get("selection") == "analysis"
        for eras in factors.get("stitch", {}).values() for e in eras.values()
    )
    if analysis_k:
        print("k measured over the ANALYSIS preselection: the gate below checks that")
        print("stage 2 applied it (TTbb scaled by k, inclusive unscaled), not the")
        print("picoAOD-level sumw identity, which is non-zero by design here.\n")
        print(f"{'channel/era':30s} {'k expected':>20s} {'k applied':>20s} {'dev':>12s}  ok")
    else:
        print(f"{'channel/era':30s} {'sumw before':>20s} {'sumw after':>20s} {'rel':>12s}  ok")
    print("-" * 92)
    for channel, era, before, after, rel, ok in report:
        print(
            f"{channel + '/' + era:30s} {before:20.6e} {after:20.6e} "
            f"{rel:+12.3e}  {'OK' if ok else 'FAIL'}"
        )

    if failures and not args.allow_closure_failure:
        what = "applied-scale check" if analysis_k else "genWeight closure"
        raise SystemExit(
            f"\n{what} failed for {len(failures)} (channel, era) "
            f"combination(s); refusing to write {args.output}"
        )

    # On-disk gate. The closure test above only reads stage-2's own bookkeeping;
    # it cannot see a merge that silently dropped an output file. Check the
    # filesystem before writing a dataset entry that claims those files.
    absent = {}
    for name, entry in out.items():
        for era, blob in entry.items():
            if era == "xs":
                continue
            miss = missing_on_disk(blob["picoAOD"]["files"])
            if miss:
                absent[f"{name}/{era}"] = miss
    if absent:
        total = sum(len(v) for v in absent.values())
        print(f"\n{total} stitched picoAOD(s) listed in the metadata are NOT on disk:")
        for key, miss in sorted(absent.items()):
            n = len(out[key.split("/")[0]][key.split("/", 1)[1]]["picoAOD"]["files"])
            print(f"  {key}: {len(miss)} of {n} missing")
            for m in miss[:5]:
                print(f"      {m.split('/')[-1]}")
            if len(miss) > 5:
                print(f"      ... and {len(miss) - 5} more")
        if not args.allow_missing_files:
            raise SystemExit(
                f"\nrefusing to write {args.output}: re-run stage 2 for the affected "
                f"dataset/era (the campaign pin makes it resume) and verify again"
            )
    else:
        n = sum(len(b["picoAOD"]["files"])
                for e in out.values() for k, b in e.items() if k != "xs")
        print(f"\non-disk check: all {n} stitched picoAOD files present")

    if not out:
        raise SystemExit("nothing to write - no (channel, era) pair was complete")

    with open(args.output, "w") as fh:
        yaml.dump(out, fh, default_flow_style=False, sort_keys=True)
    print(f"\nwrote {args.output}")
    for name, entry in sorted(out.items()):
        eras = sorted(k for k in entry if k != "xs")
        nfiles = sum(len(entry[e]["picoAOD"]["files"]) for e in eras)
        print(f"  {name}: xs={entry['xs']} eras={eras} files={nfiles}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
