"""Regroup the per-dataset hemisphere file lists from a cluster .coffea into a
single ``{year: [files]}`` YAML entry.

Each cluster job processes one year, so every dataset key in the .coffea
(data_2022_EEE, data_2022_EEF, ...) belongs to that year; we concatenate all
their hemisphere-file lists under the single year key. The dump_friend_trees
entries are the only top-level values carrying both ``files`` and ``source``.

With --max-hemis N the library is prescaled to about N hemispheres: every dataset (era)
keeps the same fraction N / (total saved_hemis) of its files, evenly spaced over its
sorted file list, so the eras keep their proportions. A mixer worker loads its year's
whole library (~3 GB per million hemispheres), which is what the cap bounds.

Usage: regroup_hemi_library.py YEAR INFILE.coffea OUTFILE.yml [--max-hemis N]
"""
import argparse
import os
import sys

# Run from the repo root (snakemake/run_container preserve cwd) so that
# unpickling the .coffea — which holds src.storage.eos.EOS objects — can import
# `src`. Without this, sys.path[0] is this script's dir and the load() fails
# with "No module named 'src'".
sys.path.insert(0, os.getcwd())

import yaml
from coffea.util import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year")
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--max-hemis", type=int, default=0,
                    help="prescale the library to about this many hemispheres (0: keep all)")
    args = ap.parse_args()

    data = load(args.infile)
    datasets = {key: value for key, value in data.items()
                if isinstance(value, dict) and "files" in value and "source" in value}
    if not any(value["files"] for value in datasets.values()):
        raise SystemExit(f"No hemisphere files found in {args.infile} for {args.year}")

    total = sum(value.get("saved_hemis", 0) for value in datasets.values())
    fraction = 1.0
    if args.max_hemis and total > args.max_hemis:
        fraction = args.max_hemis / total

    files, kept_hemis = [], 0.0
    for key, value in sorted(datasets.items()):
        ds_files = sorted(str(f) for f in value["files"])
        n_keep = len(ds_files) if fraction == 1.0 else max(1, round(fraction * len(ds_files)))
        keep = [ds_files[i * len(ds_files) // n_keep] for i in range(n_keep)]
        files.extend(keep)
        kept_hemis += value.get("saved_hemis", 0) * n_keep / max(1, len(ds_files))
        print(f"{key}: {n_keep}/{len(ds_files)} files, ~{value.get('saved_hemis', 0) * n_keep / max(1, len(ds_files)):,.0f} hemispheres")

    with open(args.outfile, "w") as fh:
        yaml.dump({args.year: files}, fh, default_flow_style=False)
    print(f"{args.year}: {len(files)} hemisphere files, ~{kept_hemis:,.0f} of {total:,} hemispheres "
          f"(prescale fraction {fraction:.4f}) -> {args.outfile}")


if __name__ == "__main__":
    main()
