"""SvB_MA vs SvB_FeynNet comparison plots.

Reads histograms produced by the Snakefile_Run3.smk quadjet_run2 mode with
`enable_SvB_FeynNet_comparison=true`. Produces:

  1. 2D correlation (ps_hh vs p_ggHH_vs_bkg) for signal and background
  2. Per-channel ROC curves (HH / ZH / ZZ)
  3. Sig-eff at fixed bkg-rejection table
  4. Conditional 1D distributions after a tight cut on the other classifier
  5. Disagreement-corner input-feature distributions (m4j, jet mults, dijet
     masses, xW, xbW)

Usage (inside the analysis container):

    ./run_container python coffea4bees/plots/SvB_FeynNet_comparison.py \\
        --sig output/Run3_quadjet_run2/histAll_Run3_quadjet_run2.coffea \\
        --bkg output/Run3_quadjet_run2/histAll_FvT_quadjet_run2.coffea \\
        --outdir output/Run3_quadjet_run2/plots_SvB_vs_FeynNet/
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
from coffea.util import load
import hist  # noqa: F401  (imports the .loc / .sum operators on Hist)

hep.style.use("CMS")


# ─────────────────────────────────────────────────────────────────────────────
# Loading and slicing
# ─────────────────────────────────────────────────────────────────────────────

def load_hists(path: str) -> dict:
    """Load a .coffea file and return the top-level 'hists' dict."""
    out = load(path)
    return out["hists"] if isinstance(out, dict) and "hists" in out else out


def list_signal_processes(hists: dict, hist_name: str = "SvB_MA.ps_hh", pattern: str = "GluGluto") -> list[str]:
    """Find signal-process category values present in a representative hist."""
    if hist_name not in hists:
        return []
    procs = list(hists[hist_name].axes["process"])
    return [p for p in procs if pattern in p]


def slice_hist(h, *, processes, year=sum, tag="fourTag", region="SR"):
    """Slice a categorical hist down to a single process+year+tag+region selection.

    `processes` is a list of process names; they are summed.
    `year=sum` sums all years (default). Pass a string to pick one.
    Returns a Hist with only the kinematic / classifier axes remaining.
    """
    procs = processes if isinstance(processes, (list, tuple)) else [processes]
    parts = [h[{"process": p, "year": year, "tag": tag, "region": region}] for p in procs]
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def values_with_overflow(h):
    """Return h.values() including underflow/overflow for cumulative sums."""
    return h.values(flow=True)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: 2D correlation
# ─────────────────────────────────────────────────────────────────────────────

def plot_2d_correlation(sig_hist, bkg_hist, *, channel: str, outdir: Path,
                        sig_label: str = "HH4b", bkg_label: str = "Total bkg (3b×JCM×FvT)"):
    """Side-by-side 2D ps_hh × p_ggHH_vs_bkg for signal and background."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, h, title in [(axes[0], sig_hist, sig_label), (axes[1], bkg_hist, bkg_label)]:
        v = h.values()
        x_edges = h.axes[0].edges
        y_edges = h.axes[1].edges
        positive = v[v > 0]
        if positive.size > 1 and positive.max() / max(positive.min(), 1e-30) > 10:
            # Enough dynamic range for a log scale
            v_disp = np.where(v > 0, v, np.nan)
            norm = LogNorm(vmin=positive.min(), vmax=positive.max())
            mesh = ax.pcolormesh(x_edges, y_edges, v_disp.T, norm=norm, cmap="viridis")
        else:
            # Sparse / narrow-range data — use linear scale, no NaN masking
            mesh = ax.pcolormesh(x_edges, y_edges, v.T, cmap="viridis")
        fig.colorbar(mesh, ax=ax, label="weighted events")
        # Prefer the histogram-stored label, but normalise FeynNet so it
        # always reads `SvB_FeynNet.p_*_vs_bkg` instead of the shorter
        # "FeynNet P(...)" label baked in at fill time.
        def _norm(lbl, name):
            if name and name.startswith("SvB_FeynNet."):
                return name
            return lbl or name
        ax.set_xlabel(_norm(h.axes[0].label, h.axes[0].name))
        ax.set_ylabel(_norm(h.axes[1].label, h.axes[1].name))
        ax.set_title(f"{title}  ({channel}, fourTag SR)")
        ax.plot([0, 1], [0, 1], "w--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / f"correlation_2d_{channel}.pdf")
    fig.savefig(outdir / f"correlation_2d_{channel}.png", dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 + 3: ROC curves and sig-eff at fixed bkg-rej
# ─────────────────────────────────────────────────────────────────────────────

def roc_from_1d(sig_h, bkg_h):
    """Compute ROC (FPR, TPR, AUC) from 1D classifier-score histograms."""
    sig = sig_h.values()
    bkg = bkg_h.values()
    # Cumulate from high score to low: TPR/FPR at threshold = cut on left edge of bin
    sig_cum = np.cumsum(sig[::-1])[::-1]
    bkg_cum = np.cumsum(bkg[::-1])[::-1]
    if sig_cum[0] <= 0 or bkg_cum[0] <= 0:
        return None
    tpr = sig_cum / sig_cum[0]
    fpr = bkg_cum / bkg_cum[0]
    # np.trapz sign depends on x-array direction; fpr is decreasing here
    # so take abs() to give the conventional positive AUC.
    auc = float(np.abs(np.trapz(tpr, fpr)))
    return fpr, tpr, auc


def sig_eff_at_bkg_rej(fpr, tpr, target_rej: float) -> float:
    """Linear-interpolate signal eff at FPR = 1 / target_rej."""
    target_fpr = 1.0 / target_rej
    # fpr is decreasing from 1 to 0 as we walk the array; sort to interp
    order = np.argsort(fpr)
    return float(np.interp(target_fpr, fpr[order], tpr[order]))


def plot_roc_per_channel(sig_hists: dict, bkg_hists: dict, outdir: Path):
    """Three-panel ROC: HH (ps_hh vs p_ggHH_vs_bkg), ZH, ZZ."""
    channels = [
        ("HH", "SvB_MA.ps_hh",  "SvB_FeynNet.p_ggHH_vs_bkg"),
        ("ZH", "SvB_MA.ps_zh",  "SvB_FeynNet.p_ZH_vs_bkg"),
        ("ZZ", "SvB_MA.ps_zz",  "SvB_FeynNet.p_ZZ_vs_bkg"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    table_rows = []
    for ax, (chan, ma_key, fn_key) in zip(axes, channels):
        aucs = {}
        for label, color, key in [("SvB_MA", "C0", ma_key),
                                  ("SvB_FeynNet", "C1", fn_key)]:
            if key not in sig_hists or key not in bkg_hists:
                ax.text(0.5, 0.5, f"missing {key}", ha="center", transform=ax.transAxes)
                continue
            roc = roc_from_1d(sig_hists[key], bkg_hists[key])
            if roc is None:
                continue
            fpr, tpr, auc = roc
            # Swap: x = signal eff (linear), y = bkg eff (log)
            ax.plot(tpr, fpr, color=color, lw=2, label=label)
            aucs[label] = auc
            for rej in (10, 100, 1000, 10000):
                eff = sig_eff_at_bkg_rej(fpr, tpr, rej)
                table_rows.append((chan, label, rej, eff))
        # Inset AUC values (per-channel, so can't live in a shared legend)
        if aucs:
            txt = "\n".join(f"AUC {k} = {v:.3f}" for k, v in aucs.items())
            ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                    fontsize="small", va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray", alpha=0.85))
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1)
        ax.set_xlabel(f"Signal efficiency — {chan} channel")
        ax.set_ylabel("Background efficiency")
        ax.grid(True, which="both", alpha=0.3)
    # Single shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(labels), fontsize="small", frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / "roc_per_channel.pdf", bbox_inches="tight")
    fig.savefig(outdir / "roc_per_channel.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Sig-eff table
    with open(outdir / "sig_eff_table.txt", "w") as fh:
        fh.write(f"{'Channel':<6} {'Classifier':<15} {'BkgRej':>8} {'SigEff':>8}\n")
        for chan, label, rej, eff in table_rows:
            fh.write(f"{chan:<6} {label:<15} {rej:>8} {eff:>8.4f}\n")
    print(f"[sig-eff] table → {outdir / 'sig_eff_table.txt'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Conditional 1D distributions
# ─────────────────────────────────────────────────────────────────────────────

def conditional_marginal(h2d, *, cut_axis: int, cut_value: float):
    """Return the 1D marginal on the *other* axis after a cut on `cut_axis`.

    Works with raw `.values()` arrays — does NOT use Hist.project() after
    slicing because boost-histogram's project() re-aggregates from the
    original storage, ignoring the slice view, which silently produced
    identical results for all cut values.

    Returns (edges_other, values_other).
    """
    edges_cut = h2d.axes[cut_axis].edges
    other_axis = 1 - cut_axis
    edges_other = h2d.axes[other_axis].edges
    idx = int(np.searchsorted(edges_cut, cut_value, side="left"))
    full_vals = h2d.values()
    if cut_axis == 0:
        sliced = full_vals[idx:, :]
        marginal = sliced.sum(axis=0)
    else:
        sliced = full_vals[:, idx:]
        marginal = sliced.sum(axis=1)
    return edges_other, marginal


def plot_conditional_distributions(sig_2d, bkg_2d, *, outdir: Path,
                                   thresholds=(0.50, 0.90, 0.95)):
    """SvB_MA.ps_hh distribution after tight cuts on FeynNet, and vice versa."""
    panels = [
        # (panel_title, cut_axis, plot_axis, legend_template, xlabel)
        ("SvB_MA.ps_hh | SvB_FeynNet.p_ggHH_vs_bkg>X",
         1, 0, "SvB_FeynNet.p_ggHH_vs_bkg > X", "SvB_MA.ps_hh"),
        ("SvB_FeynNet.p_ggHH_vs_bkg | SvB_MA.ps_hh>X",
         0, 1, "SvB_MA.ps_hh > X",              "SvB_FeynNet.p_ggHH_vs_bkg"),
    ]
    for title, cut_axis, plot_axis, leg_template, xlabel in panels:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, h2d, source in [(axes[0], sig_2d, "Signal HH4b"), (axes[1], bkg_2d, "Total bkg (3b×FvT)")]:
            edges, base_vals = conditional_marginal(h2d, cut_axis=cut_axis, cut_value=0.0)
            widths = np.diff(edges)
            if base_vals.sum() > 0:
                ax.step(edges[:-1], base_vals / base_vals.sum() / widths, where="post",
                        color="black", lw=2, label="no cut")
            for t, color in zip(thresholds, ("C0", "C1", "C2")):
                _, vals = conditional_marginal(h2d, cut_axis=cut_axis, cut_value=t)
                if vals.sum() <= 0:
                    continue
                ax.step(edges[:-1], vals / vals.sum() / widths, where="post",
                        color=color, lw=2,
                        label=leg_template.replace("X", f"{t:.2f}"))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("normalized")
            ax.set_title(source)
            if base_vals.sum() > 0:
                ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
        # Single shared legend above both panels.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(labels), fontsize="small", frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safename = title.replace(">", "gt").replace("|", "_given_").replace(" ", "").replace("/", "_")
        fig.savefig(outdir / f"conditional_{safename}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"conditional_{safename}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Disagreement-corner feature distributions
# ─────────────────────────────────────────────────────────────────────────────

CORNERS = {
    "agree_high": dict(ma=(0.95, 1.0), fn=(0.95, 1.0)),
    # "only" corners use a looser pass-threshold (>0.9) on their own
    # classifier but a strict fail (<0.5) on the other. Still disjoint
    # from agree_high because the fail-side axis is non-overlapping.
    "MA_only":    dict(ma=(0.90, 1.0), fn=(0.0,  0.50)),
    "FN_only":    dict(ma=(0.0,  0.50), fn=(0.90, 1.0)),
}
CORNER_COLORS = {"agree_high": "black", "MA_only": "C0", "FN_only": "red"}


def corner_marginal_2d(h2d, *, corner: dict, project_axis: int):
    """Slice a 2D (MA × FN) hist by `corner` and return the marginal on
    `project_axis` (0 = MA axis, 1 = FN axis).

    Returns (edges, values) — both restricted to the corner's range on the
    projected axis (so the resulting 1D distribution lives only inside the
    corner cut, not the full [0, 1] range).

    Uses .values()-based slicing because Hist.project() ignores slice views.
    """
    ma_lo, ma_hi = corner["ma"]
    fn_lo, fn_hi = corner["fn"]
    ma_axis = h2d.axes[0]
    fn_axis = h2d.axes[1]
    ma_lo_idx = int(np.searchsorted(ma_axis.edges, ma_lo, side="left"))
    ma_hi_idx = int(np.searchsorted(ma_axis.edges, ma_hi, side="right")) - 1
    fn_lo_idx = int(np.searchsorted(fn_axis.edges, fn_lo, side="left"))
    fn_hi_idx = int(np.searchsorted(fn_axis.edges, fn_hi, side="right")) - 1
    full = h2d.values()
    sliced = full[ma_lo_idx:ma_hi_idx, fn_lo_idx:fn_hi_idx]
    if project_axis == 0:
        edges = ma_axis.edges[ma_lo_idx : ma_hi_idx + 1]
        return edges, sliced.sum(axis=1)
    else:
        edges = fn_axis.edges[fn_lo_idx : fn_hi_idx + 1]
        return edges, sliced.sum(axis=0)


def plot_score_corners(sig_2d_dict, bkg_2d_dict, *, outdir: Path):
    """Corner-sliced 1D distributions of the classifier scores themselves.

    Three plots:
      1. SvB_MA.ps_hh in (ps_hh, p_ggHH)-defined corners
      2. SvB_MA.ps    in (ps,    p_ggHH)-defined corners
      3. SvB_FeynNet.p_ggHH_vs_bkg in (ps_hh, p_ggHH)-defined corners
    """
    plots = [
        # (panel_title, hist_key, project_axis, xlabel)
        ("ps_hh_corner",
         "SvB_vs_FeynNet.ps_hh_vs_p_ggHH", 0,
         "SvB_MA P(HH | HH largest)"),
        ("ps_corner",
         "SvB_vs_FeynNet.ps_vs_p_ggHH",    0,
         "SvB_MA P(any signal)"),
        ("p_ggHH_corner",
         "SvB_vs_FeynNet.ps_hh_vs_p_ggHH", 1,
         "SvB_FeynNet.p_ggHH_vs_bkg"),
    ]
    for title, hist_key, project_axis, xlabel in plots:
        if hist_key not in sig_2d_dict or hist_key not in bkg_2d_dict:
            print(f"[score_corners] skip {title} — missing {hist_key}")
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, h2d, source in [(axes[0], sig_2d_dict[hist_key], "Signal HH4b"),
                                (axes[1], bkg_2d_dict[hist_key], "Total bkg (3b×JCM×FvT)")]:
            for cname, corner in CORNERS.items():
                edges, vals = corner_marginal_2d(h2d, corner=corner, project_axis=project_axis)
                n = vals.sum()
                if n <= 0 or len(edges) < 2:
                    continue
                widths = np.diff(edges)
                ax.step(edges[:-1], vals / n / widths, where="post",
                        color=CORNER_COLORS[cname], lw=2, label=cname)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("normalized")
            ax.set_title(source)
            ax.grid(True, which="both", alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(labels), fontsize="small", frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(outdir / f"score_corner_{title}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"score_corner_{title}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def slice_corner(h3d, *, corner: dict):
    """h3d axes are (feature, MA.ps_hh, FeynNet.p_ggHH_vs_bkg).
    Slice the two classifier axes by `corner`, then sum over them via numpy.

    Returns (edges, values) for the 1D feature marginal — cannot use
    Hist.project() after slicing (project re-aggregates from the original
    storage and ignores the slice).
    """
    ma_lo, ma_hi = corner["ma"]
    fn_lo, fn_hi = corner["fn"]
    ma_axis = h3d.axes[1]
    fn_axis = h3d.axes[2]
    ma_lo_idx = int(np.searchsorted(ma_axis.edges, ma_lo, side="left"))
    ma_hi_idx = int(np.searchsorted(ma_axis.edges, ma_hi, side="right")) - 1
    fn_lo_idx = int(np.searchsorted(fn_axis.edges, fn_lo, side="left"))
    fn_hi_idx = int(np.searchsorted(fn_axis.edges, fn_hi, side="right")) - 1
    full_vals = h3d.values()  # shape (feature, MA, FN)
    sliced = full_vals[:, ma_lo_idx:ma_hi_idx, fn_lo_idx:fn_hi_idx]
    feat_marginal = sliced.sum(axis=(1, 2))
    return h3d.axes[0].edges, feat_marginal


def plot_corner_features(sig_hists: dict, bkg_hists: dict, *, outdir: Path):
    feature_keys = [k for k in sig_hists if k.endswith("_vs_2cls")]
    if not feature_keys:
        print("[corners] no *_vs_2cls hists found, skipping")
        return
    for key in feature_keys:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, hsrc, source in [(axes[0], sig_hists, "Signal HH4b"),
                                 (axes[1], bkg_hists, "Total bkg (3b×FvT)")]:
            if key not in hsrc:
                ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
                continue
            h3d = hsrc[key]
            for cname, c in CORNERS.items():
                edges, v = slice_corner(h3d, corner=c)
                norm = v.sum()
                if norm <= 0:
                    continue
                widths = np.diff(edges)
                ax.step(edges[:-1], v / norm / widths, where="post",
                        color=CORNER_COLORS[cname], lw=2,
                        label=cname)
            ax.set_xlabel(h3d.axes[0].label or h3d.axes[0].name)
            ax.set_ylabel("normalized")
            ax.set_title(source)
            ax.grid(True, which="both", alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
                   ncol=len(labels), fontsize="small", frameon=False)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safekey = key.replace(".", "__")
        fig.savefig(outdir / f"corners_{safekey}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"corners_{safekey}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sig", required=True, help="Coffea file with HH4b signal histograms (e.g. histAll_Run3_quadjet_run2.coffea)")
    p.add_argument("--bkg", required=True, help="Coffea file with FvT-weighted data (e.g. histAll_FvT_quadjet_run2.coffea)")
    p.add_argument("--outdir", required=True, help="Output directory for plots")
    p.add_argument("--sig-pattern", default="GluGluto", help="Substring matching signal process names")
    p.add_argument("--bkg-process", default="data", help="Background process name(s) — comma-separated to sum (default: data)")
    p.add_argument("--bkg-tag", default="threeTag", help="Background tag slice (default: threeTag — FvT-projected)")
    p.add_argument("--year", default=None, help="Single year (default: sum all)")
    p.add_argument("--cut-thresholds", default="0.50,0.90,0.95",
                   help="Comma-separated thresholds for conditional-distribution plots")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    year_sel = args.year if args.year else sum
    thresholds = tuple(float(x) for x in args.cut_thresholds.split(","))

    print(f"[load] sig: {args.sig}")
    sig_all = load_hists(args.sig)
    print(f"[load] bkg: {args.bkg}")
    bkg_all = load_hists(args.bkg)

    sig_procs = list_signal_processes(sig_all, pattern=args.sig_pattern)
    if not sig_procs:
        sys.exit(f"[error] no signal processes match pattern {args.sig_pattern!r} in {args.sig}")
    bkg_procs = [p.strip() for p in args.bkg_process.split(",")]
    print(f"[sig] processes: {sig_procs}")
    print(f"[bkg] processes={bkg_procs}, tag={args.bkg_tag!r}")

    # Slice every needed hist down to the (process, year, tag, region) selection
    def sig_slice(name, *, tag="fourTag"):
        return slice_hist(sig_all[name], processes=sig_procs, year=year_sel, tag=tag, region="SR") if name in sig_all else None

    def bkg_slice(name):
        return slice_hist(bkg_all[name], processes=bkg_procs, year=year_sel, tag=args.bkg_tag, region="SR") if name in bkg_all else None

    # 1. Per-channel 2D correlation
    for chan, key in [("HH", "SvB_vs_FeynNet.ps_hh_vs_p_ggHH"),
                      ("ZH", "SvB_vs_FeynNet.ps_zh_vs_p_ZH"),
                      ("ZZ", "SvB_vs_FeynNet.ps_zz_vs_p_ZZ")]:
        s = sig_slice(key)
        b = bkg_slice(key)
        if s is None or b is None:
            print(f"[2d {chan}] skip — missing {key}")
            continue
        plot_2d_correlation(s, b, channel=chan, outdir=outdir)
        print(f"[2d {chan}] → correlation_2d_{chan}.pdf")

    # 2 + 3. Per-channel ROC + sig-eff table (uses existing 1D hists)
    sig_1d = {k: sig_slice(k) for k in
              ["SvB_MA.ps_hh", "SvB_MA.ps_zh", "SvB_MA.ps_zz",
               "SvB_FeynNet.p_ggHH_vs_bkg", "SvB_FeynNet.p_ZH_vs_bkg", "SvB_FeynNet.p_ZZ_vs_bkg"]
              if sig_slice(k) is not None}
    bkg_1d = {k: bkg_slice(k) for k in
              ["SvB_MA.ps_hh", "SvB_MA.ps_zh", "SvB_MA.ps_zz",
               "SvB_FeynNet.p_ggHH_vs_bkg", "SvB_FeynNet.p_ZH_vs_bkg", "SvB_FeynNet.p_ZZ_vs_bkg"]
              if bkg_slice(k) is not None}
    plot_roc_per_channel(sig_1d, bkg_1d, outdir=outdir)
    print(f"[roc] → roc_per_channel.pdf")

    # 4. Conditional distributions (uses 2D ps_hh × p_ggHH_vs_bkg)
    sig_2d = sig_slice("SvB_vs_FeynNet.ps_hh_vs_p_ggHH")
    bkg_2d = bkg_slice("SvB_vs_FeynNet.ps_hh_vs_p_ggHH")
    if sig_2d is not None and bkg_2d is not None:
        plot_conditional_distributions(sig_2d, bkg_2d, outdir=outdir, thresholds=thresholds)
        print(f"[conditional] → conditional_*.pdf")

    # 5. Disagreement-corner feature distributions (uses 3D feature × MA × FN hists)
    sig_3d = {k: sig_slice(k) for k in sig_all if k.startswith("SvB_vs_FeynNet.") and k.endswith("_vs_2cls")}
    bkg_3d = {k: bkg_slice(k) for k in bkg_all if k.startswith("SvB_vs_FeynNet.") and k.endswith("_vs_2cls")}
    plot_corner_features(sig_3d, bkg_3d, outdir=outdir)
    print(f"[corners] → corners_*.pdf")

    # 6. Corner-sliced 1D distributions of the classifier scores themselves
    score_2d_keys = ["SvB_vs_FeynNet.ps_hh_vs_p_ggHH", "SvB_vs_FeynNet.ps_vs_p_ggHH"]
    sig_score = {k: sig_slice(k) for k in score_2d_keys if k in sig_all}
    bkg_score = {k: bkg_slice(k) for k in score_2d_keys if k in bkg_all}
    plot_score_corners(sig_score, bkg_score, outdir=outdir)
    print(f"[score_corners] → score_corner_*.pdf")

    print(f"\nAll plots written to {outdir}/")


if __name__ == "__main__":
    main()
