"""
Stage 1 of the tt+bb stitching: measure the per-category genWeight sums and
derive the scale factor applied to the tt+B events taken from the TTbb sample.

For every (channel, era) this reads only ``genWeight`` and ``genTtbarId`` from
the inclusive ttbar and the dedicated TTbb picoAODs, and sums genWeight split
into the ttB / ttC / ttLF categories of
:mod:`coffea4bees.analysis.helpers.ttbar_categories`.

The scale factor is

    k = sumw_ttB(inclusive) / sumw_ttB(TTbb)

so that the tt+B piece taken from TTbb carries exactly the genWeight sum of the
tt+B piece removed from the inclusive sample.

``--selection`` chooses the region over which those sums are taken:

``picoaod`` (default)
    Sum over every event in the picoAOD, i.e. at the skim boundary. This
    preserves the inclusive tt+B yield *within the skim acceptance*, and is
    correct only if both sides were skimmed with the same selection.

``analysis``
    Apply the analysis preselection
    (``lumimask & passNoiseFilter & passHLT & passJetMult & passPreSel``) to
    both sides first, using the real analysis helpers, and sum over the
    survivors. Use this when the two sides were skimmed differently: the
    inclusive and TTbb picoAODs for Run 3 were produced by different code
    generations, so their skim acceptances differ (measured ratios 0.52 for
    2022/2023 and 0.81 for 2024). A picoAOD-level k then mis-normalises tt+B by
    1/f_ttB once the analysis cut is applied (~1.7x and ~1.15x respectively),
    and the picoAOD-level closure test cannot see it because k is constructed to
    satisfy exactly that identity.

    Measuring k over a region contained in *both* skims makes the common
    inefficiency cancel, so the tt+B yield matches the inclusive prediction
    where the analysis actually operates. The analysis preselection qualifies:
    it is tighter than the skim cut on both sides (four medium tags above
    30 GeV implies four above 15 GeV).

Run from the barista root, inside the container::

    ./run_container python coffea4bees/skimmer/tools/ttbb_stitch_factors.py \\
        -m coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT.yml \\
        -o coffea4bees/skimmer/metadata/ttbb_stitch_factors.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml

# Invoked as a plain script (``python coffea4bees/skimmer/tools/...``), so the
# barista root is not on sys.path; add it so the shared helper is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from coffea4bees.analysis.helpers.ttbar_categories import (  # noqa: E402
    is_ttB,
    is_ttC,
    is_ttLF,
)

# channel -> (inclusive dataset, dedicated TTbb dataset)
CHANNELS = {
    "dilepton": ("TTTo2L2Nu", "TTbb_2L2Nu"),
    "semileptonic": ("TTToSemiLeptonic", "TTbb_SemiLeptonic"),
    "hadronic": ("TTToHadronic", "TTbb_Hadronic"),
}
ERAS = ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]

CATEGORY_FN = {"ttB": is_ttB, "ttC": is_ttC, "ttLF": is_ttLF}


def scan_file(path, retries=3):
    """Per-category genWeight sums for a single picoAOD file."""
    last = None
    for attempt in range(retries):
        try:
            with uproot.open(path) as fh:
                arrays = fh["Events"].arrays(["genWeight", "genTtbarId"], library="np")
            break
        except Exception as exc:  # noqa: BLE001 - retry transient xrootd failures
            last = exc
            if attempt == retries - 1:
                raise RuntimeError(f"failed to read {path}: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    else:  # pragma: no cover
        raise RuntimeError(f"failed to read {path}: {last}")

    gw = arrays["genWeight"].astype(np.float64)
    gid = arrays["genTtbarId"].astype(np.int64)

    out = {
        "n_tot": int(gw.size),
        "sumw_tot": float(gw.sum()),
        "sumw2_tot": float((gw**2).sum()),
        "n_negative_genTtbarId": int((gid < 0).sum()),
    }

    masks = {}
    for cat, fn in CATEGORY_FN.items():
        mask = fn(gid)
        masks[cat] = mask
        out[f"n_{cat}"] = int(mask.sum())
        out[f"sumw_{cat}"] = float(gw[mask].sum())
        out[f"sumw2_{cat}"] = float((gw[mask] ** 2).sum())

    # The three categories must partition the sample exactly.
    overlap = (masks["ttB"] & masks["ttC"]) | (masks["ttB"] & masks["ttLF"]) | (
        masks["ttC"] & masks["ttLF"]
    )
    if overlap.any() or int(sum(m.sum() for m in masks.values())) != gw.size:
        raise RuntimeError(f"ttB/ttC/ttLF do not partition {path}")

    out["mod100_counts"] = {
        str(k): int(v) for k, v in Counter((np.abs(gid) % 100).tolist()).items()
    }
    return out


# Cuts making up the analysis preselection, in cutflow order. This is the common
# entry point to every analysis region, so normalising here leaves any deeper
# differences as genuine 4FS-vs-5FS shape effects rather than normalisation ones.
ANALYSIS_CUTS = ("lumimask", "passNoiseFilter", "passHLT", "passJetMult", "passPreSel")


def load_analysis_inputs(corrections_path, triggers_path, object_selection_cfg):
    """Load the shared inputs the analysis selection needs (once, not per file)."""
    from coffea4bees.analysis.helpers.object_selection import (
        load_object_selection_config,
    )

    corrections = yaml.safe_load(open(corrections_path))
    triggers = yaml.safe_load(open(triggers_path))["triggers"]
    sel_cfg = load_object_selection_config(object_selection_cfg)
    return corrections, triggers, sel_cfg


def scan_file_analysis(path, dataset, era, process, corrections, triggers, sel_cfg,
                       retries=3, chunk_size=100_000):
    """Per-category genWeight sums *after the analysis preselection*.

    Calls the same helpers the analysis processor calls -- ``apply_event_selection``
    then ``apply_4b_selection`` -- so the region is the analysis region by
    construction and not a reimplementation of it. Applied identically to the
    inclusive and the TTbb side, which is what makes their differing skim
    acceptances cancel in the ratio.

    Returns the same keys as :func:`scan_file` (computed over the survivors) plus
    ``n_read``/``sumw_read`` for the pre-selection totals, which is what the
    metadata cross-check in :func:`main` compares against ``saved_events``.
    """
    from coffea.nanoevents import NanoAODSchema

    from src.compat import nano_from_root
    from src.physics.event_selection import apply_event_selection
    from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
    from coffea4bees.analysis.helpers.processor_config import processor_config

    NanoAODSchema.warn_missing_crossrefs = False

    meta = {
        "year": era,
        "dataset": dataset,
        "processName": process,
        "trigger": triggers[era],
    }

    def _read(entry_start=None, entry_stop=None):
        """Open one entry range. ``mode="virtual"`` (coffea >= 2025.12) materialises
        only the branches the selection touches, which keeps a 400-600 branch
        picoAOD from being pulled into memory wholesale -- reading eagerly gets the
        process OOM-killed."""
        kw = dict(schemaclass=NanoAODSchema, metadata=meta)
        if entry_start is not None:
            kw.update(entry_start=entry_start, entry_stop=entry_stop)
        for extra in ({"mode": "virtual"}, {"delayed": False}):
            try:
                return nano_from_root({path: "Events"}, **kw, **extra).events()
            except TypeError:
                continue
        raise RuntimeError("no supported NanoEventsFactory read mode")

    # entry count, with retries for transient xrootd failures
    last = None
    for attempt in range(retries):
        try:
            with uproot.open(path) as fh:
                n_entries = fh["Events"].num_entries
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == retries - 1:
                raise RuntimeError(f"failed to read {path}: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    else:  # pragma: no cover
        raise RuntimeError(f"failed to read {path}: {last}")

    gw_parts, gid_parts, gw_all_parts = [], [], []
    for start in range(0, max(n_entries, 1), chunk_size):
        stop = min(start + chunk_size, n_entries)
        if start >= stop:
            break
        last = None
        for attempt in range(retries):
            try:
                events = _read(start, stop)
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt == retries - 1:
                    raise RuntimeError(f"failed to read {path}[{start}:{stop}]: {exc}") from exc
                time.sleep(2 * (attempt + 1))
        else:  # pragma: no cover
            raise RuntimeError(f"failed to read {path}: {last}")

        if "genTtbarId" not in events.fields:
            raise RuntimeError(f"{path} has no genTtbarId branch")

        config = processor_config(process, dataset, events)
        events = apply_event_selection(
            events, corrections[era], cut_on_lumimask=config["cut_on_lumimask"]
        )
        events = apply_4b_selection(
            events, corrections[era], config=config, dataset=dataset, sel_cfg=sel_cfg
        )

        keep = np.ones(len(events), dtype=bool)
        for cut in ANALYSIS_CUTS:
            if cut not in events.fields:
                raise RuntimeError(
                    f"{path}: analysis cut {cut!r} not produced by the selection"
                )
            keep &= np.asarray(ak.to_numpy(events[cut]), dtype=bool)

        g = np.asarray(ak.to_numpy(events.genWeight)).astype(np.float64)
        i = np.asarray(ak.to_numpy(events.genTtbarId)).astype(np.int64)
        gw_all_parts.append(g)
        gw_parts.append(g[keep])
        gid_parts.append(i[keep])
        del events, keep, g, i

    gw_all = np.concatenate(gw_all_parts) if gw_all_parts else np.zeros(0)
    gw = np.concatenate(gw_parts) if gw_parts else np.zeros(0)
    gid = np.concatenate(gid_parts) if gid_parts else np.zeros(0, dtype=np.int64)

    out = {
        # pre-selection totals: used for the saved_events cross-check
        "n_read": int(gw_all.size),
        "sumw_read": float(gw_all.sum()),
        # post-selection totals: these define k
        "n_tot": int(gw.size),
        "sumw_tot": float(gw.sum()),
        "sumw2_tot": float((gw**2).sum()),
        "n_negative_genTtbarId": int((gid < 0).sum()),
    }

    masks = {}
    for cat, fn in CATEGORY_FN.items():
        mask = fn(gid)
        masks[cat] = mask
        out[f"n_{cat}"] = int(mask.sum())
        out[f"sumw_{cat}"] = float(gw[mask].sum())
        out[f"sumw2_{cat}"] = float((gw[mask] ** 2).sum())

    overlap = (masks["ttB"] & masks["ttC"]) | (masks["ttB"] & masks["ttLF"]) | (
        masks["ttC"] & masks["ttLF"]
    )
    if overlap.any() or int(sum(m.sum() for m in masks.values())) != gw.size:
        raise RuntimeError(f"ttB/ttC/ttLF do not partition {path} (after selection)")

    out["mod100_counts"] = {
        str(k): int(v) for k, v in Counter((np.abs(gid) % 100).tolist()).items()
    }
    return out


def merge(acc, res):
    for key, val in res.items():
        if key == "mod100_counts":
            dest = acc.setdefault("mod100_counts", {})
            for k, v in val.items():
                dest[k] = dest.get(k, 0) + v
        else:
            acc[key] = acc.get(key, 0) + val
    acc["n_files"] = acc.get("n_files", 0) + 1
    return acc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "-m", "--metadata",
        default="coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT.yml",
        help="dataset YAML holding the inclusive and TTbb entries",
    )
    ap.add_argument(
        "-o", "--output",
        default="coffea4bees/skimmer/metadata/ttbb_stitch_factors.json",
        help="output JSON with the per (channel, era) scale factors",
    )
    ap.add_argument("-j", "--workers", type=int, default=16, help="concurrent file reads")
    ap.add_argument("--eras", nargs="+", default=ERAS)
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS))
    ap.add_argument(
        "--selection", choices=("picoaod", "analysis"), default="picoaod",
        help="region over which the genWeight sums are taken (see module docstring). "
             "'analysis' is required when the two sides were skimmed differently.",
    )
    ap.add_argument(
        "--corrections", default="src/physics/corrections.yml",
        help="corrections metadata (analysis selection only)",
    )
    ap.add_argument(
        "--triggers", default="coffea4bees/metadata/triggers_HH4b.yml",
        help="trigger lists, for passHLT (analysis selection only)",
    )
    ap.add_argument(
        "--object-selection-cfg",
        default="coffea4bees/analysis/metadata/object_selection_thresholds.yml",
        help="object selection thresholds (analysis selection only)",
    )
    ap.add_argument(
        "--chunk-size", type=int, default=100_000,
        help="events per read in analysis mode; bounds peak memory",
    )
    args = ap.parse_args(argv)

    analysis_mode = args.selection == "analysis"
    if analysis_mode:
        corrections, triggers, sel_cfg = load_analysis_inputs(
            args.corrections, args.triggers, args.object_selection_cfg
        )

    datasets = yaml.safe_load(open(args.metadata))

    jobs = []
    for chan in args.channels:
        for name in CHANNELS[chan]:
            for era in args.eras:
                pico = datasets.get(name, {}).get(era, {}).get("picoAOD")
                if not pico or not pico.get("files"):
                    print(f"[warn] no picoAOD for {name} {era}", flush=True)
                    continue
                jobs += [(name, era, f) for f in pico["files"]]

    print(f"scanning {len(jobs)} files with {args.workers} workers "
          f"[selection={args.selection}]", flush=True)
    samples, bad, done = {}, [], 0

    def submit(pool, dataset, era, path):
        if analysis_mode:
            # dataset key carries the era so processor_config can detect Run 3
            return pool.submit(scan_file_analysis, path, f"{dataset}_{era}", era,
                               dataset, corrections, triggers, sel_cfg,
                               chunk_size=args.chunk_size)
        return pool.submit(scan_file, path)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {submit(pool, d, e, p): (d, e, p) for d, e, p in jobs}
        for fut in as_completed(futures):
            dataset, era, path = futures[fut]
            done += 1
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[BAD] {exc}", flush=True)
                bad.append(path)
                continue
            merge(samples.setdefault(dataset, {}).setdefault(era, {}), res)
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}", flush=True)

    if bad:
        raise SystemExit(f"aborting: {len(bad)} unreadable files:\n  " + "\n  ".join(bad))

    # Cross-check the scanned file count and attach the dataset-level metadata.
    for dataset, eras in samples.items():
        for era, acc in eras.items():
            pico = datasets[dataset][era]["picoAOD"]
            if acc["n_files"] != len(pico["files"]):
                raise SystemExit(
                    f"{dataset} {era}: scanned {acc['n_files']} files but metadata lists "
                    f"{len(pico['files'])}"
                )
            # In analysis mode n_tot counts the survivors, so the metadata
            # cross-check has to use the pre-selection count instead.
            n_read = acc["n_read"] if analysis_mode else acc["n_tot"]
            if n_read != pico["saved_events"]:
                raise SystemExit(
                    f"{dataset} {era}: picoAOD holds {n_read} events but metadata "
                    f"records saved_events={pico['saved_events']}"
                )
            acc["metadata_sumw"] = pico.get("sumw")
            acc["metadata_total_events"] = pico.get("total_events")
            acc["metadata_saved_events"] = pico.get("saved_events")
            acc["xs"] = datasets[dataset].get("xs")

    stitch = {}
    for chan in args.channels:
        incl_name, ttbb_name = CHANNELS[chan]
        for era in args.eras:
            a = samples.get(incl_name, {}).get(era)
            b = samples.get(ttbb_name, {}).get(era)
            if not a or not b:
                continue
            if b["sumw_ttB"] <= 0:
                raise SystemExit(f"{ttbb_name} {era}: non-positive tt+B sumw")
            k = a["sumw_ttB"] / b["sumw_ttB"]
            stitch.setdefault(chan, {})[era] = {
                "inclusive": incl_name,
                "ttbb": ttbb_name,
                "scale": k,
                # picoAOD-level sums that define the closure test
                "sumw_inclusive_total": a["sumw_tot"],
                "sumw_inclusive_ttB": a["sumw_ttB"],
                "sumw_inclusive_notB": a["sumw_ttC"] + a["sumw_ttLF"],
                "sumw_ttbb_ttB": b["sumw_ttB"],
                "sumw_stitched_expected": (a["sumw_ttC"] + a["sumw_ttLF"])
                + k * b["sumw_ttB"],
                # generator-level normalisation inherited by the stitched entry
                "xs": a["xs"],
                "metadata_sumw_inclusive": a["metadata_sumw"],
                "n_inclusive_notB": a["n_ttC"] + a["n_ttLF"],
                "n_ttbb_ttB": b["n_ttB"],
                # the region k was measured over -- consumers must not assume picoAOD
                "selection": args.selection,
            }
            if analysis_mode:
                # With an analysis-level k the picoAOD-level identity
                #   sumw_notB(incl) + k*sumw_ttB(TTbb) == sumw(incl)
                # is deliberately NOT satisfied; the identity that holds instead is
                #   k * sumw_ttB(TTbb, A) == sumw_ttB(incl, A)
                # i.e. the tt+B yield is preserved where the analysis operates.
                stitch[chan][era].update({
                    "sumw_inclusive_ttB_presel": a["sumw_read"],
                    "sumw_ttbb_ttB_presel": b["sumw_read"],
                    "analysis_eff_inclusive": a["sumw_tot"] / a["sumw_read"],
                    "analysis_eff_ttbb": b["sumw_tot"] / b["sumw_read"],
                    "n_inclusive_read": a["n_read"],
                    "n_ttbb_read": b["n_read"],
                    # exact by construction; recorded so stage 3/4 can verify it
                    "closure_analysis_rel": (
                        k * b["sumw_ttB"] - a["sumw_ttB"]
                    ) / a["sumw_ttB"],
                })

    payload = {
        "meta": {
            "metadata_file": args.metadata,
            "selection": args.selection,
            "level": (
                "analysis-preselection genWeight sums ("
                + " & ".join(ANALYSIS_CUTS) + ")"
                if args.selection == "analysis"
                else "picoAOD (post-skim) genWeight sums"
            ),
            "categorization": "coffea4bees.analysis.helpers.ttbar_categories",
            "channels": {c: list(CHANNELS[c]) for c in args.channels},
            "eras": args.eras,
        },
        "samples": samples,
        "stitch": stitch,
    }
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(f"\nwrote {args.output}")

    for chan, eras in stitch.items():
        print(f"\n=== {chan} ===")
        for era, s in eras.items():
            closure = s["sumw_stitched_expected"] / s["sumw_inclusive_total"] - 1
            line = (
                f"  {era:14s} k={s['scale']:9.4f}"
                f"  fracB_incl={s['sumw_inclusive_ttB']/s['sumw_inclusive_total']:.4f}"
                f"  fracB_ttbb={s['sumw_ttbb_ttB']/(samples[s['ttbb']][era]['sumw_tot']):.4f}"
            )
            if s.get("selection") == "analysis":
                line += (
                    f"  eff_incl={s['analysis_eff_inclusive']:.4f}"
                    f"  eff_ttbb={s['analysis_eff_ttbb']:.4f}"
                    f"  closure_A={s['closure_analysis_rel']:+.3e}"
                )
            else:
                line += f"  closure={closure:+.3e}"
            print(line)

    all_mod = Counter()
    n_neg = 0
    for eras in samples.values():
        for acc in eras.values():
            all_mod.update({int(k): v for k, v in acc["mod100_counts"].items()})
            n_neg += acc["n_negative_genTtbarId"]
    print(f"\nabs(genTtbarId) % 100 values seen: {sorted(all_mod)}")
    print(f"negative genTtbarId entries: {n_neg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
