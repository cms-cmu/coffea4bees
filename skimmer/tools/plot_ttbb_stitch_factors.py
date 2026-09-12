"""Plot the tt+bb stitching scale factor k per channel and era.

k is the factor applied to the ``genWeight`` of the tt+B events taken from the
dedicated 4FS TTbb samples so that they carry the tt+B weight removed from the
inclusive 5FS sample:

    k = sumw_ttB(inclusive) / sumw_ttB(TTbb)

The values are read back from the stitched dataset metadata, where stage 3 stored
them per (channel, era) under ``picoAOD.stitch.genWeight_scale_on_ttbb``.

A NOTE ON WHAT k MEANS PER ERA
------------------------------
``picoAOD.stitch.k_selection`` records the region the ratio was taken over:

``analysis``
    Both sides restricted to the analysis preselection
    (lumimask & passNoiseFilter & passHLT & passJetMult & passPreSel) before
    summing. This is the one that preserves the tt+B cross section where the
    analysis operates, and is what the Run 3 entries use.

``picoaod`` (or absent, for entries written before the option existed)
    Summed over every event in the picoAOD, i.e. at the skim boundary. Correct
    only when both sides were skimmed identically. The Run 2 entries predate the
    analysis-level option and carry this.

The two are not interchangeable, so the plots label which one each panel shows
rather than presenting them as the same quantity.
"""

import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

CHANNELS = [
    ("TTTo2L2Nu_stitched", "TTTo2L2Nu"),
    ("TTToSemiLeptonic_stitched", "TTToSemiLeptonic"),
    ("TTToHadronic_stitched", "TTToHadronic"),
]

# Era order per run, so the offsets come out in a sensible left-to-right order.
RUN2_ERAS = ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]
RUN3_ERAS = ["2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix", "2024"]

MARKERS = ["o", "s", "^", "v", "D", "P", "X"]


def read_factors(path):
    """{run: {era: {channel_key: (k, k_selection)}}} from the stitched dataset yml."""
    datasets = yaml.safe_load(open(path))
    out = defaultdict(lambda: defaultdict(dict))
    for key, _ in CHANNELS:
        entry = datasets.get(key)
        if entry is None:
            print(f"[warn] {key} not in {path}")
            continue
        for era, blob in entry.items():
            if era == "xs" or not isinstance(blob, dict):
                continue
            stitch = (blob.get("picoAOD") or {}).get("stitch")
            if not stitch or "genWeight_scale_on_ttbb" not in stitch:
                print(f"[warn] no stitch factor for {key} {era}")
                continue
            run = "Run2" if str(era).startswith("UL") else "Run3"
            out[run][str(era)][key] = (
                float(stitch["genWeight_scale_on_ttbb"]),
                stitch.get("k_selection", "picoaod"),
            )
    return out


def plot_run(run, per_era, outdir, fmt="png"):
    """One panel: x = channel, one point per era, offset so points do not overlap."""
    order = RUN2_ERAS if run == "Run2" else RUN3_ERAS
    eras = [e for e in order if e in per_era] + [e for e in per_era if e not in order]
    if not eras:
        print(f"[warn] no {run} eras found, skipping")
        return None

    selections = {sel for era in eras for _, sel in per_era[era].values()}
    sel_label = "/".join(sorted(selections))

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 7))

    x = np.arange(len(CHANNELS))
    # Spread the eras symmetrically around each channel tick. Half-width 0.26
    # keeps neighbouring channels clearly separated (ticks are 1.0 apart).
    span = 0.26
    offsets = (
        np.linspace(-span, span, len(eras)) if len(eras) > 1 else np.array([0.0])
    )
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(eras)))

    for i, era in enumerate(eras):
        vals, xs = [], []
        for j, (key, _) in enumerate(CHANNELS):
            if key in per_era[era]:
                vals.append(per_era[era][key][0])
                xs.append(x[j] + offsets[i])
        if not vals:
            continue
        ax.plot(
            xs, vals,
            marker=MARKERS[i % len(MARKERS)], markersize=11, linestyle="none",
            color=colors[i], markeredgecolor="black", markeredgewidth=0.8,
            label=era.replace("_", " "),
        )

    ax.set_xticks(x)
    # Channel names are long; shrink so neighbouring labels do not run together.
    ax.set_xticklabels([label for _, label in CHANNELS], fontsize=17)
    ax.set_xlim(-0.5, len(CHANNELS) - 0.5)
    ax.set_ylabel(r"$k = \mathrm{sumw}_{t\bar{t}+B}^{\,\mathrm{incl}}\,/\,"
                  r"\mathrm{sumw}_{t\bar{t}+B}^{\,\mathrm{TTbb}}$")
    ax.set_xlabel("")
    # Faint separators between channels, so the era clusters read as groups.
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="grey", alpha=0.3, lw=1, ls=":")
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend(title=f"{run} eras", ncol=2, fontsize=15, title_fontsize=16,
              loc="best", framealpha=0.9)

    # Pad both ends: headroom for the CMS label and legend, and a margin at the
    # bottom so the lowest point is not clipped by the axis.
    vals_all = [v for era in eras for v, _ in per_era[era].values()]
    lo, hi = min(vals_all), max(vals_all)
    rng = (hi - lo) or max(abs(hi), 1.0)
    ax.set_ylim(lo - 0.12 * rng, hi + 0.45 * rng)
    hep.cms.label(ax=ax, label="Simulation Preliminary", data=False, com=13 if run == "Run2" else 13.6)

    # Say which region k was measured over -- picoAOD-level and analysis-level k
    # are different quantities and must not be read as the same number.
    ax.text(0.03, 0.88, f"$k$ measured in: {sel_label}", transform=ax.transAxes,
            fontsize=14, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.85))

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"ttbb_stitch_kfactor_{run}.{fmt}")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input",
                    default="coffea4bees/metadata/datasets/TT_stitched.yml",
                    help="stitched dataset YAML to read the k factors from")
    ap.add_argument("-o", "--outdir", default="output/ttbb_stitch_plots",
                    help="directory for the plots")
    ap.add_argument("--format", default="png", help="image format")
    args = ap.parse_args(argv)

    factors = read_factors(args.input)
    if not factors:
        raise SystemExit(f"no stitch factors found in {args.input}")

    print(f"\n{'channel':<20}{'era':<15}{'k':>10}  k measured in")
    print("-" * 62)
    for run in ("Run2", "Run3"):
        order = RUN2_ERAS if run == "Run2" else RUN3_ERAS
        for era in [e for e in order if e in factors.get(run, {})]:
            for key, label in CHANNELS:
                if key in factors[run][era]:
                    k, sel = factors[run][era][key]
                    print(f"{label:<20}{era:<15}{k:>10.4f}  {sel}")

    made = [p for run in ("Run2", "Run3")
            if (p := plot_run(run, factors.get(run, {}), args.outdir, args.format))]

    mixed = {sel for run in factors.values() for era in run.values()
             for _, sel in era.values()}
    if len(mixed) > 1:
        print("\n[note] the two panels do NOT show the same quantity:")
        print("       " + ", ".join(sorted(mixed)) + " appear in this file.")
        print("       analysis-level k is a ratio over the analysis preselection;")
        print("       picoAOD-level k is a ratio over the whole picoAOD. They are")
        print("       not comparable era-to-era, which is why each panel is labelled.")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
