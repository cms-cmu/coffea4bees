"""Refuse an incomplete skim registry (runner.py -s picoaod_datasets*.yml).

runner.py exits 0 when a skim loses chunks: the skimmer swallows per-chunk exceptions
(skipbadfiles), integrity_check records them under each dataset's `missing`, and
process_skimming_output then skips the merge -- the registry is still written, with no usable
files. Downstream, that silently drops events (or a whole era) from the published dataset.

This checks one registry: every expected (year, era) has files, every file is a path (an
unmerged registry holds Chunk objects, which the tolerant loader turns into None), and no
dataset records missing files/chunks. On failure the registry is renamed to
<registry>.incomplete, so the next snakemake run (roast resume) re-runs the job that made it,
and the script exits 1. On success it writes OK_FILE.

Usage: check_skim_registry.py REGISTRY.yml OK_FILE --expect YEAR:ERA[,ERA...] ... [--test]
  --test   (roast submit -t) the slice is partial by design: report problems, never fail
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_mixeddata_registries import _TolerantLoader  # noqa: E402

sys.path.insert(0, os.getcwd())
from src.tools.make_dataset_yml import parse_dataset_key  # noqa: E402


def problems(registry, expect):
    out = []
    if not registry:
        return ["registry is empty"]
    have = set()
    for key, entry in registry.items():
        entry = entry or {}
        missing = {k: v for k, v in (entry.get("missing") or {}).items() if v}
        for kind, items in missing.items():
            out.append(f"{key}: {kind} ({len(items)}), e.g. {items[0]}")
        files = entry.get("files") or []
        if any(not isinstance(f, str) for f in files):
            out.append(f"{key}: files are not paths (the merge was skipped)")
        elif files:
            have.add(parse_dataset_key(key))
    for year, eras in expect.items():
        for era in eras:
            if (year, era) not in have:
                out.append(f"no files for {year}{era}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registry")
    ap.add_argument("ok_file")
    ap.add_argument("--expect", nargs="*", default=[], help="YEAR:ERA[,ERA...] that must have files")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()

    expect = {}
    for item in a.expect:
        year, _, eras = item.partition(":")
        expect[year] = [e for e in eras.split(",") if e] or [""]

    with open(a.registry) as f:
        registry = yaml.load(f, Loader=_TolerantLoader) or {}
    found = problems(registry, expect)
    for p in found:
        print(f"{'WARNING' if a.test else 'ERROR'}: {a.registry}: {p}")
    if found and not a.test:
        bad = f"{a.registry}.incomplete"
        os.replace(a.registry, bad)
        raise SystemExit(f"{a.registry}: incomplete skim ({len(found)} problems) -- moved to {bad}; "
                         f"re-running the workflow re-runs this job")
    with open(a.ok_file, "w") as f:
        f.write("ok\n")
    print(f"{a.registry}: complete ({len(registry)} datasets)")


if __name__ == "__main__":
    main()
