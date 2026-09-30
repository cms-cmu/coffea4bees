"""MakeMixedData M.7 signal check (Snakefile_MakeMixedData_7_signal.smk).

  dataset  build the synthetic_mc_<signal> dataset YAML: the mixed picoAODs from the merged M7
           registry (keys <signal>_<year>...) + the original sample's normalisation (the mixing keeps
           every mixed event's generator weight, so the original sumw / xs give comparable yields)
  report   original vs mixed signal, all years. The original is shown twice: fourTag (the signal
           region the analysis uses) and threeTag + fourTag (every preselected event -- what was
           mixed). The mixed signal is shown in fourTag (the mixed data's 4b). Per sample:
           SR / SB yields and SR fraction, Higgs-candidate masses (100 < m < 150 fraction, mean,
           RMS) and the SR SvB ps_hh (mean, fractions and yields above 0.5 / 0.8 / 0.95).
           Overlays, unit area and absolute -> signal_check.{txt,yml}, plots/*.png, index.html

A mixing that decorrelates the hemispheres turns the signal background-like: the mixed SR fraction
and 100 < m < 150 fraction fall towards the data values and the SvB piles up at low ps_hh. What
survives at high SvB is signal the mixed background model would carry into MvD and the SvB fit.

Run from the barista root (the container's cwd).
"""
import argparse
import glob
import html
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
            entry[year] = {"picoAOD": {"files": files, **{k: orig[k] for k in NORM_KEYS if k in orig}}}
            print(f"{sig} {year}: {len(files)} mixed files, sumw {orig.get('sumw')}")
        if not any(y in entry for y in args.years):
            raise SystemExit(f"no mixed files for {sig} in any of {args.years}")
        out[f"{args.prefix}{sig}"] = entry
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hists = load(args.hists)["hists"]
    plots = os.path.join(args.output_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    masses = [v for v in ("quadJet_selected.lead.mass", "quadJet_selected.subl.mass") if v in hists]
    svb = next((v for v in ("SvB_MA.ps_hh_fine", "SvB_MA.ps_hh") if v in hists), None)
    if svb is None:
        print("WARNING: no SvB_MA.ps_hh histogram -- was the SvB evaluated? (inputs.SvB_model)")
    variables = ([svb] if svb else []) + masses + [v for v in ("m4j", "quadJet_selected.xHH") if v in hists]
    summary = {}
    lines = ["# M.7 signal check: original vs mixed signal MC (all years)", ""]
    cols = ("original 4b", "original 3b+4b", "mixed 4b")
    page = []
    for pair in args.pairs:
        orig_name, mix_name = pair.split(":")
        h0 = hists[variables[0]]
        po, pm = _find_process(h0, orig_name), _find_process(h0, mix_name)
        samples = {cols[0]: (po, ["fourTag"]), cols[1]: (po, ["threeTag", "fourTag"]), cols[2]: (pm, ["fourTag"])}
        res = {label: _summarise(hists, masses, svb, p, tags) for label, (p, tags) in samples.items()}
        mixed_all = _summarise(hists, masses, svb, pm, ["threeTag", "fourTag"])
        res["mixed_fourTag_fraction"] = float((res[cols[2]]["SR"] + res[cols[2]]["SB"]) /
                                              max(mixed_all["SR"] + mixed_all["SB"], 1e-12))
        summary[orig_name] = res

        def row(name, fmt, get):
            vals = []
            for c in cols:
                try:
                    vals.append(fmt.format(get(res[c])))
                except (KeyError, TypeError):
                    vals.append("-")
            return f"{name:36s} " + " ".join(f"{v:>15s}" for v in vals)

        lines += [f"## {orig_name}  (processes {po} vs {pm})", "",
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
            try:
                y_mix = res[cols[2]]["SvB_SR"]["yield_gt_0p8"]
                y_all = res[cols[1]]["SvB_SR"]["yield_gt_0p8"]
                y_4b = res[cols[0]]["SvB_SR"]["yield_gt_0p8"]
                res["SvB_gt_0p8_mixed_over_original_3b4b"] = float(y_mix / y_all) if y_all else float("nan")
                res["SvB_gt_0p8_mixed_over_original_4b"] = float(y_mix / y_4b) if y_4b else float("nan")
                lines.append(f"{'SR SvB > 0.8 yield, mixed / original':36s} "
                             f"{res['SvB_gt_0p8_mixed_over_original_4b']:15.3f} {res['SvB_gt_0p8_mixed_over_original_3b4b']:15.3f}")
            except KeyError:
                pass
        lines += [f"{'mixed signal in fourTag':36s} {res['mixed_fourTag_fraction']:15.3f}", ""]

        page.append(f"<h2>{html.escape(orig_name)}</h2>")
        for v in variables:
            h = hists[v]
            edges = h.axes[-1].edges
            regions = ["SR"] if v == svb else ["SR", "SB"]
            for norm in ("unit", "abs"):
                fig, ax = plt.subplots(figsize=(6, 4))
                for (label, (p, tags)), style in zip(samples.items(), ("-", ":", "--")):
                    y = _project(h, p, tags, regions)
                    if y.sum() > 0:
                        ax.stairs(y / y.sum() if norm == "unit" else y, edges, label=label, linestyle=style)
                ax.set_xlabel(v)
                ax.set_ylabel(("unit area" if norm == "unit" else "events") + f" ({'+'.join(regions)})")
                ax.legend(fontsize=8)
                if v == svb:
                    ax.set_yscale("log")
                ax.set_title(orig_name, fontsize=8)
                fig.tight_layout()
                name = f"{orig_name}__{v.replace('.', '_')}__{norm}.png"
                fig.savefig(os.path.join(plots, name), dpi=110)
                plt.close(fig)
                page.append(f'<a href="plots/{name}"><img src="plots/{name}" width="420"></a>')
    lines += ["Mixing works if the mixed signal looks background-like: its SR fraction and 100<m<150",
              "fraction fall towards the data values (4b data SR/(SR+SB) ~ 0.34), and its SvB piles up at",
              "low ps_hh. Whatever stays at high SvB is signal that the mixed background model",
              "(mixeddata_all -> MvD, SvB background) would absorb. Yields use the original sample's",
              "sumw; the 3b events carry no JCM weight (a shape test)."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.output_dir, "signal_check.txt"), "w") as f:
        f.write(text)
    with open(os.path.join(args.output_dir, "signal_check.yml"), "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(args.output_dir, "index.html"), "w") as f:
        f.write("<html><head><title>M.7 signal check</title></head><body>\n"
                f"<h1>M.7 signal check: original vs mixed signal</h1>\n<pre>{html.escape(text)}</pre>\n"
                + "\n".join(page) + "\n</body></html>\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--registry", required=True)
    d.add_argument("--metadata", nargs="+", required=True, help="dataset metadata dir(s) / file(s)")
    d.add_argument("--signals", nargs="+", required=True)
    d.add_argument("--years", nargs="+", required=True)
    d.add_argument("--prefix", default="synthetic_mc_")
    d.add_argument("-o", "--output", required=True)
    r = sub.add_parser("report")
    r.add_argument("--hists", required=True)
    r.add_argument("--pairs", nargs="+", required=True, help="original:mixed process names")
    r.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()
    {"dataset": cmd_dataset, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
