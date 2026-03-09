"""
Plot histograms from the dropout study processor.

Usage (inside the container):
  python coffea4bees/plots/makePlotsDropout.py output/test.coffea -o output/dropout_plots
  python coffea4bees/plots/makePlotsDropout.py output/test.coffea -o output/dropout_plots --debug
  python coffea4bees/plots/makePlotsDropout.py output/test.coffea -o output/dropout_plots --only genb_pt gen_HT
"""
import os
import argparse
import warnings
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from coffea.util import load

plt.style.use([hep.style.CMS, {'font.size': 22}])

warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')

CATEGORIES = ["all", "failTrigger", "failPresel", "failSR", "passSR"]
CAT_COLORS = {
    "all":         "k",
    "failTrigger": "#e42536",
    "failPresel":  "#f89c20",
    "failSR":      "#7a21dd",
    "passSR":      "#5790fc",
}
CAT_LABELS = {
    "all":         "All events",
    "failTrigger": "Fail trigger",
    "failPresel":  "Fail presel",
    "failSR":      "Fail SR (3-tag)",
    "passSR":      "Pass SR (4-tag)",
}


def slice_hist(h, process, year, category):
    """Slice a hist.Hist on its category axes, returning a 1D histogram."""
    opts = {"process": process, "year": year, "category": category}
    axis_names = {a.name for a in h.axes}
    opts = {k: v for k, v in opts.items() if k in axis_names}
    return h[opts]


def plot_1d(h1d, ax, color="k", label=""):
    """Plot a 1D hist.Hist on a matplotlib axis."""
    values = h1d.values()
    edges = h1d.axes[0].edges
    ax.stairs(values, edges, color=color, label=label, linewidth=1.5)
    return values


def get_process_and_year(data):
    """Extract the process name and year from loaded coffea data."""
    hists = data["hists"]
    var0 = list(hists.keys())[0]
    h = hists[var0]
    process = list(h.axes["process"])
    year = list(h.axes["year"])
    return process[0], year[0]


def get_xlabel(h, var):
    """Get the x-axis label from a histogram."""
    for a in h.axes:
        if a.name == var:
            return a.label
    return var


def plot_overlay_categories(hists_dict, var, process, year, output_dir, debug=False):
    """Overlay all failure categories for one variable."""
    if var not in hists_dict:
        return

    h = hists_dict[var]
    fig, ax = plt.subplots()

    has_data = False
    for cat in CATEGORIES:
        try:
            h1d = slice_hist(h, process, year, cat)
        except Exception as e:
            if debug:
                print(f"  skip {var}/{cat}: {e}")
            continue

        vals = h1d.values()
        if np.sum(vals) == 0:
            continue

        has_data = True
        plot_1d(h1d, ax, color=CAT_COLORS[cat], label=CAT_LABELS[cat])

    if not has_data:
        plt.close()
        return

    ax.set_xlabel(get_xlabel(h, var))
    ax.set_ylabel("Events")
    ax.legend()
    hep.cms.label("Internal", data=True, year=year, loc=0, ax=ax)
    ax.set_ylim(bottom=0)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{var}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{var}.png"), bbox_inches="tight", dpi=150)
    if debug:
        print(f"  saved {output_dir}/{var}.pdf")
    plt.close()


def plot_normalized_categories(hists_dict, var, process, year, output_dir, debug=False):
    """Overlay failure categories normalized to unit area (shape comparison)."""
    if var not in hists_dict:
        return

    h = hists_dict[var]
    fig, ax = plt.subplots()

    has_data = False
    for cat in CATEGORIES:
        if cat == "all":
            continue
        try:
            h1d = slice_hist(h, process, year, cat)
        except Exception:
            continue

        vals = h1d.values()
        total = np.sum(vals)
        if total == 0:
            continue

        has_data = True
        edges = h1d.axes[0].edges
        ax.stairs(vals / total, edges, color=CAT_COLORS[cat],
                  label=CAT_LABELS[cat], linewidth=1.5)

    if not has_data:
        plt.close()
        return

    ax.set_xlabel(get_xlabel(h, var))
    ax.set_ylabel("Fraction of events")
    ax.legend()
    hep.cms.label("Internal", data=True, year=year, loc=0, ax=ax)

    subdir = os.path.join(output_dir, "normalized")
    os.makedirs(subdir, exist_ok=True)
    fig.savefig(os.path.join(subdir, f"{var}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(subdir, f"{var}.png"), bbox_inches="tight", dpi=150)
    if debug:
        print(f"  saved {subdir}/{var}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot dropout study histograms")
    parser.add_argument("inputFile", help="Input .coffea file")
    parser.add_argument("-o", "--output", default="output/dropout_plots",
                        help="Output directory")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only plot these variables")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip these variables")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.inputFile} ...")
    data = load(args.inputFile)
    hists_dict = data["hists"]
    process, year = get_process_and_year(data)
    print(f"Process: {process},  Year: {year}")

    var_list = list(hists_dict.keys())
    if args.only:
        var_list = [v for v in args.only if v in hists_dict]
    if args.skip:
        var_list = [v for v in var_list if not any(s in v for s in args.skip)]

    print(f"Variables to plot ({len(var_list)}): {var_list}")

    # 1. Category overlay (absolute counts)
    for var in var_list:
        plot_overlay_categories(hists_dict, var, process, year,
                                args.output, debug=args.debug)

    # 2. Normalized shape comparison
    for var in var_list:
        plot_normalized_categories(hists_dict, var, process, year,
                                   args.output, debug=args.debug)

    # Print cutflow if available
    if "cutflow" in data:
        print("\n--- Cutflow ---")
        for ds, cuts in data["cutflow"].items():
            print(f"\n{ds}:")
            for cut_name, val in cuts.items():
                print(f"  {cut_name:20s}: {val}")

    print(f"\nPlots saved to {args.output}/")


if __name__ == "__main__":
    main()
