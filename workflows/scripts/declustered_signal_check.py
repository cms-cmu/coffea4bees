"""DeClustered D.6 signal scrambling check (Snakefile_DeClustered_6_signal.smk).

  dataset  build the synthetic_mc_<signal> dataset YAML: the declustered picoAODs from the merged
           D6 registry (keys <signal>_<year>) + the original sample's normalisation (every event
           and its generator weight survive the declustering, so sumw / count / xs carry over)
  report   original vs declustered signal, fourTag, all years: SR fraction SR/(SR+SB), fraction of
           Higgs candidates with 100 < m < 150 GeV, mean/RMS of the candidate masses, and (when SvB was
           run) the SR SvB ps_hh: mean and fractions above 0.8 / 0.95; overlays of the candidate
           masses, m4j and SvB (unit area) -> signal_check.{txt,yml}, plots/*.png

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
            files = (registry.get(f"{sig}_{year}") or {}).get("files") or []
            orig = ((meta[sig].get(year) or {}).get("picoAOD")) or {}
            if not files:
                if orig:
                    raise SystemExit(f"no declustered files for {sig} {year} (registry keys: {sorted(registry)[:6]})")
                continue
            entry[year] = {"picoAOD": {"files": [str(f) for f in files],
                                       **{k: orig[k] for k in NORM_KEYS if k in orig}}}
            print(f"{sig} {year}: {len(files)} declustered files, sumw {orig.get('sumw')}")
        out[f"{args.prefix}{sig}"] = entry
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)


def _project(h, process, regions):
    vals = 0
    for r in regions:
        sub = h[{"process": process, "tag": "fourTag", "region": r}]
        vals = vals + sub[{"year": sum}].values(flow=False)
    return np.asarray(vals, dtype=float)


def _find_process(h, name):
    procs = list(h.axes["process"])
    if name in procs:
        return name
    cands = [p for p in procs if p.startswith(name) or name.startswith(p)]
    if len(cands) == 1:
        return cands[0]
    raise SystemExit(f"process {name!r} not in {procs}")


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
    variables = masses + [v for v in ("m4j", "quadJet_selected.xHH") if v in hists] + ([svb] if svb else [])
    summary, lines = {}, ["# D.6 signal scrambling check: original vs declustered signal MC (fourTag, all years)", ""]
    for pair in args.pairs:
        orig_name, syn_name = pair.split(":")
        h0 = hists[variables[0]]
        po, ps = _find_process(h0, orig_name), _find_process(h0, syn_name)
        res = {}
        for label, p in (("original", po), ("declustered", ps)):
            sr = _project(h0, p, ["SR"]).sum()
            sb = _project(h0, p, ["SB"]).sum()
            r = {"SR": float(sr), "SB": float(sb), "SR_fraction": float(sr / (sr + sb)) if sr + sb > 0 else float("nan")}
            for v in masses:
                h = hists[v]
                y = _project(h, p, ["SR", "SB"])
                c = h.axes[-1].centers
                tot = y.sum()
                if tot > 0:
                    mean = float(np.sum(y * c) / tot)
                    r[v] = {"mean": mean, "rms": float(np.sqrt(np.sum(y * (c - mean) ** 2) / tot)),
                            "frac_100_150": float(y[(c > 100) & (c < 150)].sum() / tot)}
            if svb:
                y = _project(hists[svb], p, ["SR"])
                c = hists[svb].axes[-1].centers
                tot = y.sum()
                if tot > 0:
                    r["SvB_SR"] = {"mean": float(np.sum(y * c) / tot),
                                   "frac_gt_0p8": float(y[c > 0.8].sum() / tot),
                                   "frac_gt_0p95": float(y[c > 0.95].sum() / tot),
                                   "yield_gt_0p8": float(y[c > 0.8].sum())}
            res[label] = r
        o, s = res["original"], res["declustered"]
        ratio = s["SR_fraction"] / o["SR_fraction"] if o["SR_fraction"] > 0 else float("nan")
        res["SR_fraction_ratio"] = float(ratio)
        summary[orig_name] = res
        lines += [f"## {orig_name}  (processes {po} vs {ps})", "",
                  f"{'':34s} {'original':>12s} {'declustered':>12s}",
                  f"{'fourTag SR + SB events (weighted)':34s} {o['SR'] + o['SB']:12.2f} {s['SR'] + s['SB']:12.2f}",
                  f"{'SR fraction SR/(SR+SB)':34s} {o['SR_fraction']:12.3f} {s['SR_fraction']:12.3f}   (declustered/original {ratio:.3f})"]
        for v in masses:
            if v in o and v in s:
                lines.append(f"{v + '  100<m<150 frac':34s} {o[v]['frac_100_150']:12.3f} {s[v]['frac_100_150']:12.3f}")
                lines.append(f"{v + '  mean / rms':34s} {o[v]['mean']:6.1f}/{o[v]['rms']:5.1f} {s[v]['mean']:6.1f}/{s[v]['rms']:5.1f}")
        if "SvB_SR" in o and "SvB_SR" in s:
            so, ss = o["SvB_SR"], s["SvB_SR"]
            lines.append(f"{'SR SvB ps_hh mean':34s} {so['mean']:12.3f} {ss['mean']:12.3f}")
            lines.append(f"{'SR SvB ps_hh > 0.8  fraction':34s} {so['frac_gt_0p8']:12.3f} {ss['frac_gt_0p8']:12.3f}")
            lines.append(f"{'SR SvB ps_hh > 0.95 fraction':34s} {so['frac_gt_0p95']:12.3f} {ss['frac_gt_0p95']:12.3f}")
            lines.append(f"{'SR SvB ps_hh > 0.8  yield':34s} {so['yield_gt_0p8']:12.3f} {ss['yield_gt_0p8']:12.3f}"
                         f"   (declustered/original {ss['yield_gt_0p8'] / so['yield_gt_0p8'] if so['yield_gt_0p8'] else float('nan'):.3f})")
        lines.append("")
        for v in variables:
            h = hists[v]
            edges = h.axes[-1].edges
            fig, ax = plt.subplots(figsize=(6, 4))
            regions = ["SR"] if v == svb else ["SR", "SB"]
            for label, p, style in (("original", po, "-"), ("declustered", ps, "--")):
                y = _project(h, p, regions)
                if y.sum() > 0:
                    ax.stairs(y / y.sum(), edges, label=f"{label} ({p})", linestyle=style)
            ax.set_xlabel(v); ax.set_ylabel(f"unit area (fourTag, {'+'.join(regions)})"); ax.legend(fontsize=7)
            if v == svb:
                ax.set_yscale("log")
            ax.set_title(orig_name, fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(plots, f"{orig_name}__{v.replace('.', '_')}.png"), dpi=110)
            plt.close(fig)
    lines += ["A scrambled resonance: the declustered SR fraction and 100<m<150 fraction fall towards the",
              "background-like values (4b data SR/(SR+SB) ~ 0.34), and its SvB piles up at low ps_hh; unchanged",
              "values mean the peak survived (signal in the data would leak into the synthetic background)."]
    with open(os.path.join(args.output_dir, "signal_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.output_dir, "signal_check.yml"), "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print("\n".join(lines))


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
    r.add_argument("--pairs", nargs="+", required=True, help="original:declustered process names")
    r.add_argument("-o", "--output-dir", required=True)
    args = ap.parse_args()
    {"dataset": cmd_dataset, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
