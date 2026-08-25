"""
Verify the produced stitched picoAODs by re-reading them.

This is the independent, end-to-end form of the closure test: rather than
trusting the counters the processor emitted, it opens every file of the stitched
dataset and checks

1. **purity** - each output file is either entirely tt+B (it came from the
   dedicated TTbb sample) or entirely non-tt+B (it came from the inclusive
   sample). A file mixing both means the genTtbarId selection leaked.
2. **event counts** - the tt+B and non-tt+B event counts match what stage 3
   recorded in the dataset entry.
3. **the genWeight closure** - the total sum of genWeight over the stitched
   dataset equals the sum over the original inclusive picoAOD, which is the
   property that makes the stitched dataset inherit the inclusive normalisation:

       sumw(stitched) == sumw(inclusive)   [both at picoAOD level]

Run from the barista root, after stage 3::

    ./run_container python coffea4bees/skimmer/tools/ttbb_stitch_verify.py \\
        -s coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT_stitched.yml
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import uproot
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from coffea4bees.analysis.helpers.ttbar_categories import is_ttB  # noqa: E402

# float32 storage of the rescaled genWeight limits how exactly the sums can agree
TOLERANCE = 1e-6


def scan_file(path, retries=3):
    for attempt in range(retries):
        try:
            with uproot.open(path) as fh:
                a = fh["Events"].arrays(["genWeight", "genTtbarId"], library="np")
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise RuntimeError(f"failed to read {path}: {exc}") from exc
            time.sleep(2 * (attempt + 1))

    gw = a["genWeight"].astype(np.float64)
    b = is_ttB(a["genTtbarId"])
    return {
        "path": path,
        "n": int(gw.size),
        "n_ttB": int(b.sum()),
        "sumw": float(gw.sum()),
        "sumw_ttB": float(gw[b].sum()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-s", "--stitched", required=True, help="stage-3 stitched dataset YAML")
    ap.add_argument(
        "-f", "--factors",
        default="coffea4bees/skimmer/metadata/ttbb_stitch_factors.json",
        help="stage-1 factors JSON, holding the original inclusive picoAOD sums",
    )
    ap.add_argument("-j", "--workers", type=int, default=16)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args(argv)

    stitched = yaml.safe_load(open(args.stitched))
    factors = json.load(open(args.factors))

    # inclusive dataset name -> {era: measured picoAOD sums}
    reference = {}
    for _channel, eras in factors["stitch"].items():
        for era, spec in eras.items():
            reference.setdefault(spec["inclusive"], {})[era] = spec

    jobs = []
    for name, entry in stitched.items():
        for era in (e for e in entry if e != "xs"):
            for path in entry[era]["picoAOD"]["files"]:
                jobs.append((name, era, path))

    print(f"reading {len(jobs)} stitched files with {args.workers} workers")
    per_era = {}
    impure = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_file, p): (n, e, p) for n, e, p in jobs}
        done = 0
        for fut in as_completed(futures):
            name, era, path = futures[fut]
            done += 1
            res = fut.result()
            acc = per_era.setdefault((name, era), {
                "n": 0, "sumw": 0.0, "n_ttB": 0, "sumw_ttB": 0.0,
                "n_notB": 0, "sumw_notB": 0.0, "files": 0,
            })
            acc["files"] += 1
            acc["n"] += res["n"]
            acc["sumw"] += res["sumw"]
            acc["n_ttB"] += res["n_ttB"]
            acc["sumw_ttB"] += res["sumw_ttB"]
            acc["n_notB"] += res["n"] - res["n_ttB"]
            acc["sumw_notB"] += res["sumw"] - res["sumw_ttB"]
            # a file must be pure: all tt+B, or none
            if res["n"] and res["n_ttB"] not in (0, res["n"]):
                impure.append(
                    f"{name} {era}: {res['path']} mixes "
                    f"{res['n_ttB']} ttB with {res['n'] - res['n_ttB']} non-ttB"
                )
            if done % 100 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}")

    failures = list(impure)
    print()
    header = (
        f"{'dataset/era':40s} {'sumw stitched':>18s} {'sumw inclusive':>18s} "
        f"{'rel':>11s}  ok"
    )
    print(header)
    print("-" * len(header))

    for (name, era), acc in sorted(per_era.items()):
        # Take the originating dataset from the provenance stage 3 recorded, so
        # this does not depend on the --suffix used to name the entry.
        pico = stitched[name][era]["picoAOD"]
        incl_name = pico.get("stitch", {}).get("inclusive")
        spec = reference.get(incl_name, {}).get(era) if incl_name else None
        if spec is None:
            failures.append(
                f"{name} {era}: no stage-1 reference for inclusive={incl_name!r}"
            )
            continue

        want = spec["sumw_inclusive_total"]
        rel = (acc["sumw"] - want) / want
        ok = abs(rel) <= args.tolerance
        if not ok:
            failures.append(f"{name} {era}: sumw closure off by {rel:+.3e}")
        print(f"{name + '/' + era:40s} {acc['sumw']:18.8e} {want:18.8e} "
              f"{rel:+11.3e}  {'OK' if ok else 'FAIL'}")

        # the tt+B piece must carry exactly the removed inclusive tt+B weight
        rel_b = (acc["sumw_ttB"] - spec["sumw_inclusive_ttB"]) / spec["sumw_inclusive_ttB"]
        if abs(rel_b) > args.tolerance:
            failures.append(f"{name} {era}: tt+B sumw off by {rel_b:+.3e}")

        # counts must match what stage 3 recorded
        rec = pico.get("stitch", {})
        if rec:
            if acc["n_ttB"] != rec.get("n_from_ttbb"):
                failures.append(
                    f"{name} {era}: {acc['n_ttB']} ttB events in files but "
                    f"{rec.get('n_from_ttbb')} recorded"
                )
            if acc["n_notB"] != rec.get("n_from_inclusive"):
                failures.append(
                    f"{name} {era}: {acc['n_notB']} non-ttB events in files but "
                    f"{rec.get('n_from_inclusive')} recorded"
                )
        if acc["files"] != len(pico["files"]):
            failures.append(f"{name} {era}: read {acc['files']} of {len(pico['files'])} files")

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
        return 1
    print(f"all {len(per_era)} (dataset, era) combinations verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
