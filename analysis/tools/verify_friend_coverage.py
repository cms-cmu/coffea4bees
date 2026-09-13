"""Check that a friend index actually covers every picoAOD a dataset points at.

A friend tree is bound to the exact picoAOD file it was built against (file UUID
plus entry range), so an index can look healthy while being useless: it may cover
a *superseded copy* of the skim, or be missing whole eras, and nothing downstream
complains -- the events simply arrive with the friend branches unfilled.

That is not hypothetical. The Run 3 index that shipped before this
(metadata/friends/trigger_weights_TTbar_friends.json) covers only the three
inclusive TT samples for 2022/2023, and is keyed to smurthy/XX4b/2025_Run3_skims
while TT.yml points at a re-chunked copy under algomez/XX4b/Run3_nanov12 -- so
none of its 1570 entries match anything in use.

This tool answers three questions per (dataset, era):

  1. is every picoAOD file listed in the metadata present in the index?
  2. for each of those files, do the friend chunks tile [0, n) with no gap or
     overlap? (a partial tile means some events silently lose the friend)
  3. does the index contain targets the metadata does NOT list? (stale entries
     left over from an earlier production)

Entry counts come from the index itself, so no ROOT file is opened. Pass
``--check-files`` to additionally stat every friend ROOT file.

Usage
-----
    ./run_container python coffea4bees/analysis/tools/verify_friend_coverage.py \
        -f output/trigweights_run3/trigger_weights/trigger_weights_friends.json \
        -m coffea4bees/metadata/datasets/ \
        -d TTTo2L2Nu TTToSemiLeptonic TTToHadronic \
           TTTo2L2Nu_stitched TTToSemiLeptonic_stitched TTToHadronic_stitched ttHbb \
        -y 2022_preEE 2022_EE 2023_preBPix 2023_BPix 2024
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import yaml


def load_metadata(path):
    """Merge every yml under `path` (or just the one file), like runner.py does."""
    files = (
        sorted(glob.glob(os.path.join(path, "*.yml")))
        if os.path.isdir(path)
        else [path]
    )
    merged = {}
    for f in files:
        try:
            d = yaml.safe_load(open(f)) or {}
        except Exception as e:  # a malformed neighbour must not hide real results
            print(f"[warn] could not parse {f}: {e}")
            continue
        for k, v in d.items():
            if isinstance(v, dict):
                merged.setdefault(k, v)
    return merged


def expected_files(meta, datasets, years):
    """{(dataset, era): [picoAOD paths]} from the dataset metadata."""
    out = {}
    for ds in datasets:
        entry = meta.get(ds)
        if entry is None:
            print(f"[warn] dataset {ds} not found in metadata")
            continue
        for era in years:
            blob = entry.get(era, entry.get(str(era)))
            if not isinstance(blob, dict):
                print(f"[warn] {ds} has no era {era}")
                continue
            pico = blob.get("picoAOD") or {}
            files = pico.get("files")
            if not files:
                print(f"[warn] {ds} {era} has no picoAOD files")
                continue
            out[(ds, era)] = list(files)
    return out


def index_coverage(index_path, friend_name):
    """{target path: (uuid, [(start, stop), ...], [friend paths])} from the index."""
    blob = json.load(open(index_path))
    friend = blob.get(friend_name)
    if friend is None:
        sys.exit(
            f"friend '{friend_name}' not in {index_path} "
            f"(has: {sorted(blob.keys())})"
        )
    cov = {}
    for target, chunks in friend["data"]:
        spans, fpaths = [], []
        for c in chunks:
            spans.append((c["start"], c["stop"]))
            fpaths.append(c["chunk"]["path"])
        cov[target["path"]] = (target["uuid"], sorted(spans), fpaths)
    return cov, friend.get("branches", [])


def tiling_error(spans):
    """Return a message if `spans` do not tile [0, max_stop) exactly."""
    if not spans:
        return "no friend chunks"
    if spans[0][0] != 0:
        return f"starts at {spans[0][0]}, not 0"
    prev = 0
    for start, stop in spans:
        if start > prev:
            return f"gap [{prev}, {start})"
        if start < prev:
            return f"overlap at {start} (previous ended {prev})"
        prev = stop
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-f", "--index", required=True, help="friend index JSON")
    ap.add_argument(
        "-m", "--metadata", default="coffea4bees/metadata/datasets/",
        help="dataset yml or directory of ymls",
    )
    ap.add_argument("-d", "--datasets", nargs="+", required=True)
    ap.add_argument("-y", "--years", nargs="+", required=True)
    ap.add_argument("--friend-name", default="trigWeight")
    ap.add_argument(
        "--check-files", action="store_true",
        help="also stat every friend ROOT file (slower)",
    )
    args = ap.parse_args(argv)

    meta = load_metadata(args.metadata)
    want = expected_files(meta, args.datasets, args.years)
    cov, branches = index_coverage(args.index, args.friend_name)

    print(f"index    : {args.index}")
    print(f"friend   : {args.friend_name}  branches={branches}")
    print(f"targets  : {len(cov)} in index")
    print()

    hdr = f"{'dataset':<28}{'era':<15}{'files':>7}{'covered':>9}{'events':>14}  status"
    print(hdr)
    print("-" * len(hdr))

    matched = set()
    bad = defaultdict(list)
    total_missing = 0

    for (ds, era) in sorted(want):
        files = want[(ds, era)]
        n_cov = 0
        n_events = 0
        problems = []
        for f in files:
            rec = cov.get(f)
            if rec is None:
                problems.append(f"MISSING {os.path.basename(f)}")
                continue
            matched.add(f)
            n_cov += 1
            _uuid, spans, fpaths = rec
            err = tiling_error(spans)
            if err:
                problems.append(f"{os.path.basename(f)}: {err}")
            else:
                n_events += spans[-1][1]
            if args.check_files:
                for fp in fpaths:
                    local = fp.replace("root://cmseos.fnal.gov/", "/eos/uscms")
                    if not os.path.exists(local):
                        problems.append(f"friend file absent: {fp}")

        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        total_missing += len(files) - n_cov
        if problems:
            bad[(ds, era)] = problems
        print(
            f"{ds:<28}{str(era):<15}{len(files):>7}{n_cov:>9}{n_events:>14,}  {status}"
        )

    extra = sorted(set(cov) - matched)
    print()
    if bad:
        print("PROBLEMS")
        for (ds, era), problems in sorted(bad.items()):
            print(f"  {ds} {era}:")
            for p in problems[:10]:
                print(f"      {p}")
            if len(problems) > 10:
                print(f"      ... and {len(problems) - 10} more")
        print()
    if extra:
        # Not fatal on its own: one index may legitimately serve several
        # productions. It is only a red flag when it is the *whole* index.
        print(f"[note] {len(extra)} target(s) in the index are not listed by the "
              f"requested datasets/eras, e.g.:")
        for p in extra[:5]:
            print(f"      {p}")
        print()

    ok = not bad and total_missing == 0
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
