"""MakeMixedData M.7 signal check (Snakefile_MakeMixedData_7_signal.smk).

  dataset  build the dataset YAML: one entry per --prefix (synthetic_mc_3b_, synthetic_mc_4b_), all
           on the same mixed picoAODs from the merged M7 registry (keys <signal>_<year>...), with the
           original sample's normalisation (the mixing keeps every mixed event's generator weight, so
           the original sumw / xs give comparable yields; with --subsample N, the mixer's event % N
           thinning, sumw and sumw2 are divided by N). processor_HH4b keeps each origin's events.
  report   four samples, all years: the 4b signal and the 3b signal x the M.2 JCM, and the
           mixed signal from 3b input events (x the same JCM) and from 4b input events, the mixed
           ones in fourTag (the mixed data's 4b). Per sample: SR / SB yields and SR fraction,
           Higgs-candidate masses (100 < m < 150 fraction, mean, RMS) and the SR SvB ps_hh (sentinel
           fraction, mean, fractions and yields above 0.5 / 0.8 / 0.95), and mixed / original per
           origin -> signal_check.{txt,yml}. The plots are M7_plots_*' (makePlots + gallery).

A mixing that decorrelates the hemispheres turns the signal background-like: the mixed SR fraction
and 100 < m < 150 fraction fall towards the data values and the SvB piles up at low ps_hh. What
survives at high SvB is signal the mixed background model would carry into MvD and the SvB fit.

Run from the barista root (the container's cwd).
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import yaml

NORM_KEYS = ("sumw", "sumw2", "count", "total_events", "saved_events")
SVB_CUTS = (0.5, 0.8, 0.95)


def load_metadata(paths):
    meta = {}
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "*.yml"))) if os.path.isdir(p) else [p]
        for f in files:
            with open(f) as fh:
                d = yaml.safe_load(fh) or {}
            if isinstance(d, dict):
                meta.update({k: v for k, v in d.items() if isinstance(v, dict)})
    return meta


def registry_files(registry, sig, year):
    """Files of <sig> in <year>: the skimmer keys its registry <dataset>_<year> (MC has no era)."""
    keys = [k for k in registry if k == f"{sig}_{year}"] or \
           [k for k in registry if k.startswith(f"{sig}_{year}")]
    return [str(f) for k in keys for f in ((registry[k] or {}).get("files") or [])]


def cmd_dataset(args):
    with open(args.registry) as f:
        registry = yaml.safe_load(f) or {}
    meta = load_metadata(args.metadata)
    out = {}
    for sig in args.signals:
        if sig not in meta:
            raise SystemExit(f"{sig} not in the dataset metadata {args.metadata}")
        entry = {"xs": meta[sig]["xs"]} if "xs" in meta[sig] else {}
        for year in args.years:
            files = registry_files(registry, sig, year)
            orig = ((meta[sig].get(year) or {}).get("picoAOD")) or {}
            if not files:
                if orig:
                    raise SystemExit(f"no mixed files for {sig} {year} (registry keys: {sorted(registry)[:6]})")
                continue
            norm = {k: orig[k] for k in NORM_KEYS if k in orig}
            # event % N thinning in the mixer: 1/N of the generator weight is kept
            for k in ("sumw", "sumw2"):
                if k in norm:
                    norm[k] = norm[k] / args.subsample
            entry[year] = {"picoAOD": {"files": files, **norm}}
            print(f"{sig} {year}: {len(files)} mixed files, sumw {norm.get('sumw')} (original / {args.subsample})")
        if not any(y in entry for y in args.years):
            raise SystemExit(f"no mixed files for {sig} in any of {args.years}")
        for prefix in args.prefix:
            out[f"{prefix}{sig}"] = entry
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)


def _project(h, process, tags, regions, flow=False):
    vals = 0
    for t in tags:
        for r in regions:
            sub = h[{"process": process, "tag": t, "region": r}]
            vals = vals + sub[{"year": sum}].values(flow=flow)
    return np.asarray(vals, dtype=float)


def _find_process(h, name):
    procs = list(h.axes["process"])
    if name in procs:
        return name
    cands = [p for p in procs if p.startswith(name) or name.startswith(p)]
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(f"process {name!r} not in {procs}")


def _summarise(hists, masses, svb, process, tags):
    h0 = hists[masses[0] if masses else svb]
    sr = _project(h0, process, tags, ["SR"]).sum()
    sb = _project(h0, process, tags, ["SB"]).sum()
    r = {"SR": float(sr), "SB": float(sb), "SR_fraction": float(sr / (sr + sb)) if sr + sb > 0 else float("nan")}
    for v in masses:
        h = hists[v]
        y = _project(h, process, tags, ["SR", "SB"])
        c = h.axes[-1].centers
        tot = y.sum()
        if tot > 0:
            mean = float(np.sum(y * c) / tot)
            r[v] = {"mean": mean, "rms": float(np.sqrt(np.sum(y * (c - mean) ** 2) / tot)),
                    "frac_100_150": float(y[(c > 100) & (c < 150)].sum() / tot)}
    if svb:
        # ps_hh underflow = the sentinel (-2: p_ggF <= 0.01, or HH not the largest signal class):
        # part of the SR yield, so it counts in the fraction denominators; the mean is over [0, 1]
        yf = _project(hists[svb], process, tags, ["SR"], flow=True)
        y = yf[1:-1]
        c = hists[svb].axes[-1].centers
        tot = yf.sum()
        if tot > 0:
            r["SvB_SR"] = {"mean": float(np.sum(y * c) / y.sum()) if y.sum() > 0 else float("nan"),
                           "yield": float(tot), "sentinel_frac": float(yf[0] / tot)}
            for cut in SVB_CUTS:
                key = f"{cut:g}".replace(".", "p")
                r["SvB_SR"][f"frac_gt_{key}"] = float(y[c > cut].sum() / tot)
                r["SvB_SR"][f"yield_gt_{key}"] = float(y[c > cut].sum())
    return r


def cmd_report(args):
    from coffea.util import load

    hists = load(args.hists)["hists"]
    os.makedirs(args.output_dir, exist_ok=True)
    masses = [v for v in ("quadJet_selected.lead.mass", "quadJet_selected.subl.mass") if v in hists]
    svb = next((v for v in ("SvB_MA.ps_hh_fine", "SvB_MA.ps_hh") if v in hists), None)
    if svb is None:
        print("WARNING: no SvB_MA.ps_hh histogram -- was the SvB evaluated? (inputs.SvB_model)")
    h0 = hists[svb or masses[0]]
    summary = {}
    lines = ["# M.7 signal check: original vs mixed signal MC (all years)", ""]
    cols = ("4b", "3b x JCM", "mixed 3b x JCM", "mixed 4b")
    for triple in args.samples:
        orig_name, mix3_name, mix4_name = triple.split(":")
        po, p3, p4 = (_find_process(h0, n) for n in (orig_name, mix3_name, mix4_name))
        samples = {cols[0]: (po, ["fourTag"]), cols[1]: (po, ["threeTag"]),
                   cols[2]: (p3, ["fourTag"]), cols[3]: (p4, ["fourTag"])}
        res = {label: _summarise(hists, masses, svb, p, tags) for label, (p, tags) in samples.items()}
        summary[orig_name] = res

        def row(name, fmt, get):
            vals = []
            for c in cols:
                try:
                    vals.append(fmt.format(get(res[c])))
                except (KeyError, TypeError):
                    vals.append("-")
            return f"{name:36s} " + " ".join(f"{v:>15s}" for v in vals)

        lines += [f"## {orig_name}  (processes {po}, {p3}, {p4})", "",
                  f"{'':36s} " + " ".join(f"{c:>15s}" for c in cols),
                  row("SR + SB yield (weighted)", "{:.2f}", lambda r: r["SR"] + r["SB"]),
                  row("SR fraction SR/(SR+SB)", "{:.3f}", lambda r: r["SR_fraction"])]
        for v in masses:
            short = v.replace("quadJet_selected.", "")
            lines.append(row(f"{short}  100<m<150 frac", "{:.3f}", lambda r, v=v: r[v]["frac_100_150"]))
            lines.append(row(f"{short}  mean", "{:.1f}", lambda r, v=v: r[v]["mean"]))
            lines.append(row(f"{short}  rms", "{:.1f}", lambda r, v=v: r[v]["rms"]))
        if svb:
            lines.append(row("SR SvB ps_hh sentinel (<0) frac", "{:.3f}", lambda r: r["SvB_SR"]["sentinel_frac"]))
            lines.append(row("SR SvB ps_hh mean (in [0,1])", "{:.3f}", lambda r: r["SvB_SR"]["mean"]))
            for cut in SVB_CUTS:
                key = f"{cut:g}".replace(".", "p")
                lines.append(row(f"SR SvB ps_hh > {cut:g} fraction", "{:.3f}", lambda r, k=key: r["SvB_SR"][f"frac_gt_{k}"]))
                lines.append(row(f"SR SvB ps_hh > {cut:g} yield", "{:.3f}", lambda r, k=key: r["SvB_SR"][f"yield_gt_{k}"]))
            # mixed / original per origin: what of each signal population survives the mixing
            for mixed, orig, key in ((cols[2], cols[1], "mixed3b_over_3b"), (cols[3], cols[0], "mixed4b_over_4b")):
                try:
                    y_m = res[mixed]["SvB_SR"]["yield_gt_0p8"]
                    y_o = res[orig]["SvB_SR"]["yield_gt_0p8"]
                    res[f"SvB_gt_0p8_{key}"] = float(y_m / y_o) if y_o else float("nan")
                    lines.append(f"{f'SR SvB > 0.8 yield, {mixed} / {orig}':52s} {res[f'SvB_gt_0p8_{key}']:8.3f}")
                except KeyError:
                    pass
        lines.append("")

    lines += ["Mixing works if the mixed signal looks background-like: its SR fraction and 100<m<150",
              "fraction fall towards the data values (4b data SR/(SR+SB) ~ 0.34), and its SvB piles up at",
              "low ps_hh. Whatever stays at high SvB is signal that the mixed background model",
              "(mixeddata_all -> MvD, SvB background) would absorb: mixed 3b x JCM is the signal in the",
              "3b data carried into the mixed data. Yields use the original sample's sumw (/ subsample);",
              "the JCM is M.2's (the one the 3b events see before mixing), on the input event's untagged",
              "loose jets."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.output_dir, "signal_check.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(args.output_dir, "signal_check.yml"), "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--registry", required=True)
    d.add_argument("--metadata", nargs="+", required=True, help="dataset metadata dir(s) / file(s)")
    d.add_argument("--signals", nargs="+", required=True)
    d.add_argument("--years", nargs="+", required=True)
    d.add_argument("--prefix", nargs="+", default=["synthetic_mc_"],
                   help="one dataset per prefix, all on the same files")
    d.add_argument("--subsample", type=int, default=1, help="the mixer's event %% N thinning: sumw / N")
    d.add_argument("-o", "--output", required=True)
    r = sub.add_parser("report")
    r.add_argument("--hists", required=True)
    r.add_argument("--samples", nargs="+", required=True,
                   help="original:mixed-3b:mixed-4b process names, one per signal")
    r.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()
    {"dataset": cmd_dataset, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
