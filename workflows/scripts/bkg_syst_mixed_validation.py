#!/usr/bin/env python3
"""Validation plots of the bkg_syst closure samples (Snakefile_bkg_syst_A_3, rule A3_validation).

Do the mixed-data closure samples (mixed multijet + ttbar pseudodata, unit weight) look like the 4b
data and like the nominal background model -- in particular in SvB?

    points  4b data                                   (nominal roast)
    stack   nominal background model: 3b data x JCM x FvT ("data", threeTag) + its ttbar
            (--ttbar processes in --ttbar-tag; ttHbb: TTbar4b_from_d3 in threeTag -- the nominal
            Phase F file has no ttbar MC, and its *_from_d3 fourTag entries are copies of the 4b data)
    band    the N mixed subsamples: mean, min-max     (A_3 histograms, fourTag)
    line    one subsample (--compare)

per histogram (--hists), region (SR, SB) and selection (inclusive + each --cuts Boolean axis =
True, e.g. pass_nSelJets_gt6), with ratios to the model, plus yields.yml and index.html. The nominal histograms and the A_3 ones are filled with different hist_cuts (Boolean
axes): every axis other than the variable is selected (process / tag / region) or summed.

    bkg_syst_mixed_validation.py --nominal histAll_ttHbb.coffea \
        --mixed histAll_ttHbb_mixeddata_v0.coffea ... --sample-prefix mix \
        --ttbar TTbar4b_from_d3 --ttbar-tag threeTag --cuts pass_nSelJets_gt6 \
        --hists SvB_MA.ps_ttHbb SvB_MA.ps --compare 0 -o validation/
"""
import argparse
import os
import re
import sys

import numpy as np
import yaml

# the pickled histograms reference barista's src.* classes: run from the barista root
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

CATEGORY_AXES = ("process", "year", "tag", "region")


def load_coffea(path):
    try:
        from coffea.util import load
    except ImportError:            # same format: lz4-framed cloudpickle
        import cloudpickle
        import lz4.frame

        def load(p):
            with lz4.frame.open(p) as f:
                return cloudpickle.load(f)
    return load(path)


def project(h, process, tag, region, cut=None):
    """1D (values, variances, edges) of `h` for one or several processes, one tag and one region
    (and Boolean axis `cut` = True), summed over every other axis; None when a category is absent
    (e.g. no such process)."""
    names = [ax.name for ax in h.axes]
    procs = [process] if isinstance(process, str) else list(process)
    total = None
    for proc in procs:
        sel = {}
        for name in names:
            ax = h.axes[name]
            if name == "process":
                if proc not in ax:
                    break
                sel[name] = proc
            elif name == "tag":
                if tag not in ax:
                    break
                sel[name] = tag
            elif name == "region":
                if region not in ax:
                    break
                sel[name] = region
            elif cut is not None and name == cut:
                sel[name] = True
            elif name in CATEGORY_AXES or type(ax).__name__ in ("Boolean", "StrCategory", "IntCategory"):
                sel[name] = sum
        else:
            if cut is not None and cut not in names:
                raise ValueError(f"{h.name}: no Boolean axis {cut!r} (axes: {names})")
            p = h[sel]
            if len(p.axes) != 1:
                raise ValueError(f"{h.name}: {len(p.axes)} axes left after selection ({[a.name for a in p.axes]})")
            v = np.asarray(p.values(flow=False), dtype=float)
            w = p.variances(flow=False)
            w = np.asarray(w if w is not None else v, dtype=float)
            if total is None:
                total = [v, w, np.asarray(p.axes[0].edges)]
            else:
                total[0] = total[0] + v
                total[1] = total[1] + w
    return None if total is None else tuple(total)


def plot_one(path, edges, title, data, model_mj, model_tt, mixed_all, mixed_k, k, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    model = None
    if model_mj is not None or model_tt is not None:
        mj = model_mj[0] if model_mj is not None else np.zeros(len(centers))
        tt = model_tt[0] if model_tt is not None else np.zeros(len(centers))
        ax.hist([centers, centers], bins=edges, weights=[tt, mj], stacked=True, histtype="stepfilled",
                color=["#85D1FB", "#FFDF7F"], edgecolor="k", label=[r"$t\bar{t}$ (nominal model)", "Multijet (3b x JCM x FvT)"])
        model = mj + tt
    if mixed_all:
        stackv = np.array([m[0] for m in mixed_all])
        mean, lo, hi = stackv.mean(0), stackv.min(0), stackv.max(0)
        ax.fill_between(edges, np.append(lo, lo[-1]), np.append(hi, hi[-1]), step="post", color="#e42536",
                        alpha=0.25, label=f"Mixed + $t\\bar{{t}}$ PS: {len(mixed_all)} subsamples (min-max)")
        ax.stairs(mean, edges, color="#e42536", lw=1.5, label="Mixed + $t\\bar{t}$ PS: mean")
    if mixed_k is not None:
        ax.stairs(mixed_k[0], edges, color="#7a21dd", lw=1.2, ls="--", label=f"Mixed + $t\\bar{{t}}$ PS: v{k}")
    if data is not None and data[0].sum() > 0:
        ax.errorbar(centers, data[0], yerr=np.sqrt(data[1]), fmt="o", color="k", ms=4, label="4b data")
    ax.set_ylabel("Events")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

    def ratio(num, den):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)
    if model is not None:
        if mixed_all:
            rax.fill_between(edges, np.append(ratio(lo, model), ratio(lo, model)[-1]),
                             np.append(ratio(hi, model), ratio(hi, model)[-1]), step="post",
                             color="#e42536", alpha=0.25)
            rax.stairs(ratio(mean, model), edges, color="#e42536", lw=1.5)
        if data is not None and data[0].sum() > 0:
            rax.errorbar(centers, ratio(data[0], model), yerr=ratio(np.sqrt(data[1]), model), fmt="o", color="k", ms=3)
        rax.axhline(1, color="gray", lw=0.8)
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("/ model")
    rax.set_xlabel(xlabel)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nominal", required=True, help="nominal roast's Phase F histograms (coffea)")
    ap.add_argument("--mixed", nargs="+", required=True, help="A_3 histograms, one per subsample, in order")
    ap.add_argument("--sample-prefix", default="mix", help="process of subsample k: <prefix>_v<k>")
    ap.add_argument("--ttbar", nargs="+", required=True, help="ttbar processes of the nominal model")
    ap.add_argument("--ttbar-tag", default="threeTag",
                    help="tag the --ttbar processes are filled in (threeTag for *_from_d3, fourTag for MC)")
    ap.add_argument("--data-process", default="data")
    ap.add_argument("--cuts", nargs="*", default=[], help="Boolean axes: extra plots with axis = True")
    ap.add_argument("--hists", nargs="+", required=True)
    ap.add_argument("--regions", nargs="+", default=["SR", "SB"])
    ap.add_argument("--compare", type=int, default=0, help="the subsample drawn on its own")
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    nominal = load_coffea(args.nominal)["hists"]
    mixed = []
    for path in args.mixed:
        m = re.search(r"_v(\d+)\.coffea$", path)
        if m is None:
            raise ValueError(f"{path}: cannot tell the subsample index (expected ..._v<k>.coffea)")
        mixed.append((int(m.group(1)), load_coffea(path)["hists"]))

    yields, pngs, missing = {}, [], []
    for key in args.hists:
        if key not in nominal or any(key not in h for _, h in mixed):
            missing.append(key)
            print(f"WARNING: histogram {key} not in every input; skipped")
            continue
        for cut in [None, *args.cuts]:
            sel_name = cut or "inclusive"
            for region in args.regions:
                data = project(nominal[key], args.data_process, "fourTag", region, cut)
                mj = project(nominal[key], args.data_process, "threeTag", region, cut)
                tt = project(nominal[key], args.ttbar, args.ttbar_tag, region, cut)
                mixed_all, mixed_k = [], None
                for k, h in mixed:
                    p = project(h[key], f"{args.sample_prefix}_v{k}", "fourTag", region, cut)
                    if p is None:
                        raise ValueError(f"{key}: no process {args.sample_prefix}_v{k} in subsample v{k}'s histograms")
                    mixed_all.append(p)
                    if k == args.compare:
                        mixed_k = p
                edges = next(x[2] for x in (data, mj, tt, *mixed_all) if x is not None)
                png = f"{key.replace('.', '_')}_{sel_name}_{region}.png"
                plot_one(os.path.join(args.outdir, png), edges, f"{key}  ({region}, {sel_name})", data, mj, tt,
                         mixed_all, mixed_k, args.compare, nominal[key].axes[-1].label or key)
                pngs.append(png)
                sums = np.array([m[0].sum() for m in mixed_all])
                yields.setdefault(key, {}).setdefault(sel_name, {})[region] = {
                    "data_4b": float(data[0].sum()) if data is not None else None,
                    "model_multijet": float(mj[0].sum()) if mj is not None else None,
                    "model_ttbar": float(tt[0].sum()) if tt is not None else None,
                    "mixed_mean": float(sums.mean()), "mixed_std": float(sums.std(ddof=1)) if len(sums) > 1 else 0.0,
                    "mixed_per_subsample": {f"v{k}": float(x) for (k, _), x in zip(mixed, sums)},
                }

    with open(os.path.join(args.outdir, "yields.yml"), "w") as f:
        yaml.dump(yields, f, default_flow_style=False, sort_keys=False)
    with open(os.path.join(args.outdir, "index.html"), "w") as f:
        f.write("<html><body><h2>bkg_syst A_3 validation: 4b data / nominal model / mixed subsamples</h2>\n")
        if missing:
            f.write(f"<p>missing histograms: {', '.join(missing)}</p>\n")
        f.write("<p><a href='yields.yml'>yields.yml</a></p>\n")
        for png in pngs:
            f.write(f"<div><img src='{png}' width='600'></div>\n")
        f.write("</body></html>\n")
    print(f"wrote {len(pngs)} plots to {args.outdir}")
    if not pngs:
        raise SystemExit("no plot made: none of --hists is in every input")


if __name__ == "__main__":
    main()
