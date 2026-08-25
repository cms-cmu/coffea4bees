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
tt+B piece removed from the inclusive sample. Both sums are taken over picoAOD
(i.e. post-skim) events, which is the level at which the stitched dataset is
built and at which the closure test is measurable.

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
    args = ap.parse_args(argv)

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

    print(f"scanning {len(jobs)} files with {args.workers} workers", flush=True)
    samples, bad, done = {}, [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_file, p): (d, e, p) for d, e, p in jobs}
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
            if acc["n_tot"] != pico["saved_events"]:
                raise SystemExit(
                    f"{dataset} {era}: picoAOD holds {acc['n_tot']} events but metadata "
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
            }

    payload = {
        "meta": {
            "metadata_file": args.metadata,
            "level": "picoAOD (post-skim) genWeight sums",
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
            print(
                f"  {era:14s} k={s['scale']:9.4f}"
                f"  fracB_incl={s['sumw_inclusive_ttB']/s['sumw_inclusive_total']:.4f}"
                f"  fracB_ttbb={s['sumw_ttbb_ttB']/(samples[s['ttbb']][era]['sumw_tot']):.4f}"
                f"  closure={closure:+.3e}"
            )

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
