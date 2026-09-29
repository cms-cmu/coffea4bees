"""Summary statistics of a splitting library (processor_cluster_4b.py splitting_library_base_path),
per year: splittings per exact jet_flavor type, and which lookup group each type resolves to.

    python coffea4bees/jet_clustering/summarize_splitting_library.py <registry.yml> <year> \
        -o <out_dir> [--min-entries 10]

<registry.yml> is the {year: [files]} registry (local or root://). Writes
<out_dir>/splitting_library_summary_<year>.yml and .txt:
  totals   rows, clean-tree rows (the ones the declustering looks up), exact types, and the types /
           clean rows whose lookup resolves at each level (exact, child_content, parent_content,
           coarse) given min_entries
  types    per exact type (clean-tree rows, most frequent first): all rows, clean rows, the level
           and key its lookups resolve to and that group's size, parent pT quantiles (10/50/90%),
           median |eta|
"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())

import awkward as ak
import fsspec
import numpy as np
import uproot
import yaml

from coffea4bees.jet_clustering.splitting_library import SplittingLibrary, decode_flavor


def load_rows(registry, year):
    with fsspec.open(registry, "r") as f:
        files = yaml.safe_load(f)[year]
    files = [files] if isinstance(files, str) else files
    return ak.concatenate(list(uproot.iterate({f: "Events" for f in files}, library="ak", step_size=500_000))), len(files)


def summarize(rows, min_entries):
    all_flavor = decode_flavor(rows.jet_flavor)
    all_types, all_counts = np.unique(all_flavor, return_counts=True)
    n_all = dict(zip(all_types, all_counts.tolist()))

    lib = SplittingLibrary(rows, min_entries=min_entries)          # clean-tree rows, as the declusterer
    types, counts = np.unique(lib.flavor, return_counts=True)
    order = np.argsort(-counts, kind="stable")

    per_level = {name: {"types": 0, "clean_rows": 0} for name in SplittingLibrary.LEVELS}
    table = []
    for i in order:
        f, n = str(types[i]), int(counts[i])
        level, key = lib.resolve_keys(f)[0]
        members = lib._groups[level][key]
        sel = lib._groups[0][f]
        pt = lib.data["pt"][sel]
        name = SplittingLibrary.LEVELS[level]
        per_level[name]["types"] += 1
        per_level[name]["clean_rows"] += n
        table.append({
            "type": f,
            "rows": int(n_all.get(f, 0)),
            "clean_rows": n,
            "lookup_level": name,
            "lookup_key": str(key),
            "group_size": int(len(members)),
            "pt_q10_q50_q90": [round(float(q), 1) for q in np.quantile(pt, [0.1, 0.5, 0.9])],
            "abs_eta_median": round(float(np.median(np.abs(lib.data["eta"][sel]))), 3),
        })

    totals = {
        "rows": int(len(rows)),
        "clean_rows": int(len(lib.flavor)),
        "exact_types": int(len(types)),
        "min_entries": int(min_entries),
        "by_lookup_level": per_level,
    }
    return totals, table


def write_text(path, year, n_files, totals, table):
    with open(path, "w") as out:
        out.write(f"# splitting library summary: {year} ({n_files} files)\n\n")
        out.write(f"rows {totals['rows']:,}   clean-tree rows {totals['clean_rows']:,}   "
                  f"exact types {totals['exact_types']}   min_entries {totals['min_entries']}\n\n")
        out.write(f"{'lookup level':16s} {'types':>6s} {'clean rows':>12s}\n")
        for name, v in totals["by_lookup_level"].items():
            out.write(f"{name:16s} {v['types']:6d} {v['clean_rows']:12,d}\n")
        out.write(f"\n{'type':28s} {'rows':>10s} {'clean':>10s}  {'level':15s} {'group':>9s}  "
                  f"{'pT q10/q50/q90':>20s} {'|eta| med':>9s}  lookup key\n")
        for r in table:
            q = "/".join(f"{x:.0f}" for x in r["pt_q10_q50_q90"])
            out.write(f"{r['type']:28s} {r['rows']:10,d} {r['clean_rows']:10,d}  {r['lookup_level']:15s} "
                      f"{r['group_size']:9,d}  {q:>20s} {r['abs_eta_median']:9.3f}  {r['lookup_key']}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registry")
    ap.add_argument("year")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--min-entries", type=int, default=10)
    args = ap.parse_args()

    rows, n_files = load_rows(args.registry, args.year)
    totals, table = summarize(rows, args.min_entries)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"splitting_library_summary_{args.year}")
    with open(f"{stem}.yml", "w") as f:
        yaml.dump({"year": args.year, "registry": args.registry, "files": n_files,
                   "totals": totals, "types": table}, f, default_flow_style=None, sort_keys=False)
    write_text(f"{stem}.txt", args.year, n_files, totals, table)
    print(f"{args.year}: {totals['clean_rows']:,} clean rows, {totals['exact_types']} types -> {stem}.{{yml,txt}}")


if __name__ == "__main__":
    main()
