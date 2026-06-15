"""Regroup the per-dataset hemisphere file lists from a cluster .coffea into a
single ``{year: [files]}`` YAML entry.

Each cluster job processes one year, so every dataset key in the .coffea
(data_2022_EEE, data_2022_EEF, ...) belongs to that year; we concatenate all
their hemisphere-file lists under the single year key. The dump_friend_trees
entries are the only top-level values carrying both ``files`` and ``source``.

Usage: regroup_hemi_library.py YEAR INFILE.coffea OUTFILE.yml
"""
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
    year, infile, outfile = sys.argv[1], sys.argv[2], sys.argv[3]
    data = load(infile)
    files = []
    for _key, value in data.items():
        if isinstance(value, dict) and "files" in value and "source" in value:
            files.extend(str(f) for f in value["files"])
    if not files:
        raise SystemExit(f"No hemisphere files found in {infile} for {year}")
    with open(outfile, "w") as fh:
        yaml.dump({year: files}, fh, default_flow_style=False)
    print(f"{year}: {len(files)} hemisphere files -> {outfile}")


if __name__ == "__main__":
    main()
