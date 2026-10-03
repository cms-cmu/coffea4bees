"""Merge one year's splitting-library chunk files into a single ROOT file.

D.1 writes one small file per cluster chunk (~1300 per year). Every DeClusterer worker loads
the whole year's library, and reading it from that many files costs ~60 ms of xrootd latency
per file (48 branches each), ~80 s per load. One zstd file, which SplittingLibrary.from_files
xrdcp's locally (~0.5 s) before reading, loads in ~10 s. Same Events schema
(the jagged jet_flavor keeps its njet_flavor counter).

Usage: consolidate_splitting_library.py YEAR IN_REGISTRY.yml OUT_FILE.root
       (IN_REGISTRY {year: [files]}, as D1_library_regroup writes it)
"""
import sys
import time

import uproot
import yaml


def main():
    year, registry, out_file = sys.argv[1:4]
    with open(registry) as f:
        files = yaml.safe_load(f)[year]
    files = [files] if isinstance(files, str) else files
    t0 = time.time()
    n_rows = 0
    with uproot.recreate(out_file, compression=uproot.ZSTD(3)) as fout:
        for i, batch in enumerate(uproot.iterate({f: "Events" for f in files}, library="ak", step_size=500_000)):
            batch = {k: batch[k] for k in batch.fields if k != "njet_flavor"}   # rewritten as jet_flavor's counter
            if i == 0:
                fout["Events"] = batch
            else:
                fout["Events"].extend(batch)
            n_rows += len(batch["pt"])
    print(f"{year}: {len(files)} files, {n_rows} splittings -> {out_file} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
