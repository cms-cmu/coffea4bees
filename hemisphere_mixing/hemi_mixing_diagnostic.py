#!/usr/bin/env python3
"""Post-process the hemisphere-mixing 2D diagnostic histograms.

Reads a coffea hist output produced with `compute_hemi_mixing_diagnostics:
true` on the HH4b processor and, per (sample, year, tag, region, ...) bin,
extracts:

    n               total weight
    <x+>, <x->      marginal means
    var_+, var_-    marginal variances
    cov(x+, x-)     inter-hemi covariance
    corr(x+, x-)    inter-hemi correlation coefficient

for x in (eta, pz, mass, pt) of the tagged-jet 4-vector sum per hemisphere.

The 2D histograms encode the full joint and the marginals; everything above
is computed from bin contents alone (binning bias is O(Δx²/12), negligible
at the binnings we book).

Optional outputs (--plot):
    - 2D heatmaps per sample (side-by-side)
    - Truth/Mixed ratio map
    - 1D q-distribution: density of (x+_i - <x+>)(x-_i - <x->) per event,
      derived offline from the 2D joint after subtracting the sample means.

Usage:

    python hemi_mixing_diagnostic.py output/hist_HH4b.coffea
    python hemi_mixing_diagnostic.py output/hist_HH4b.coffea \\
        --plot --out plots/hemi_diag --processes data,mixeddata_v0
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

try:
    from coffea.util import load
except ImportError:
    print("ERROR: coffea.util.load required (run inside the analysis container)",
          file=sys.stderr)
    raise


def _var_list(collection: str, inclusive: bool = False):
    """Return list of (var_key, hist_name, var_label) for a jet collection.
    collection in {'can', 'all', 'other'} (or 'tag' for backwards compat).
    If inclusive=True, points at the *_2d_inclusive histograms (no region axis).
    """
    suffix = '_inclusive' if inclusive else ''
    return [
        ('eta',  f'hemi_{collection}_eta_2d{suffix}',  r'$\eta$'),
        ('pz',   f'hemi_{collection}_pz_2d{suffix}',   r'$p_z$ [GeV]'),
        ('mass', f'hemi_{collection}_mass_2d{suffix}', r'$m$ [GeV]'),
        ('pt',   f'hemi_{collection}_pt_2d{suffix}',   r'$p_T$ [GeV]'),
    ]


# legacy module-level list (kept so other importers don't break); points at 'can'
VAR_LIST = _var_list('can')


# --------------------------- moment extraction ---------------------------

def moments_from_2d(values, centers_x, centers_y):
    """Return scalar diagnostics from a 2D histogram (values shape (Nx, Ny))."""
    N = float(values.sum())
    out = {'n': N, 'mean_x': np.nan, 'mean_y': np.nan,
           'var_x': np.nan, 'var_y': np.nan,
           'cov': np.nan, 'corr': np.nan}
    if N <= 0:
        return out

    X = centers_x[:, None]
    Y = centers_y[None, :]

    mean_x = (values * X).sum() / N
    mean_y = (values * Y).sum() / N
    var_x  = (values * (X - mean_x) ** 2).sum() / N
    var_y  = (values * (Y - mean_y) ** 2).sum() / N
    cov    = (values * (X - mean_x) * (Y - mean_y)).sum() / N
    corr   = cov / np.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else np.nan
    out.update(mean_x=mean_x, mean_y=mean_y, var_x=var_x, var_y=var_y,
               cov=cov, corr=corr)
    return out


def q_distribution(values, centers_x, centers_y, mean_x, mean_y, nbins=60):
    """Centered-product distribution: re-bin (x_i - μx)(y_j - μy) weighted by H_ij.

    Equivalent to the toy's q histogram. Returns (centers, density)."""
    q_grid = (centers_x[:, None] - mean_x) * (centers_y[None, :] - mean_y)
    q_flat = q_grid.ravel()
    w_flat = values.ravel()
    if w_flat.sum() == 0:
        return np.zeros(nbins), np.zeros(nbins)

    abs_max = max(abs(q_flat[w_flat > 0].min()), abs(q_flat[w_flat > 0].max()))
    edges   = np.linspace(-abs_max, abs_max, nbins + 1)
    h, _    = np.histogram(q_flat, bins=edges, weights=w_flat)
    centers = 0.5 * (edges[:-1] + edges[1:])
    h       = h / (w_flat.sum() * (edges[1] - edges[0]))   # density
    return centers, h


# --------------------------- hist iteration ------------------------------

def categorical_axes(h):
    """Return list of (axis_name, axis_object) for non-numeric axes.

    The 2D pair of fill axes is at the end; everything else (process, year,
    tag, region, histCuts) is categorical here.
    """
    cat = []
    for ax in h.axes[:-2]:
        cat.append((ax.name, ax))
    return cat


def iter_category_combinations(h, restrict=None):
    """Yield (cat_dict, sub_hist) for every combination of categorical axes.

    restrict: dict of {axis_name: [allowed_values]} to filter combinations.
    """
    restrict = restrict or {}
    cat_axes = categorical_axes(h)
    names    = [n for n, _ in cat_axes]
    options  = [list(ax) for _, ax in cat_axes]
    for combo in itertools.product(*options):
        cat = dict(zip(names, combo))
        keep = True
        for k, allowed in restrict.items():
            if k in cat and allowed is not None and cat[k] not in allowed:
                keep = False
                break
        if not keep:
            continue
        sub = h[cat]
        yield cat, sub


# --------------------------- printing ------------------------------------

def print_table(rows, var_label):
    if not rows:
        print(f"  (no non-empty bins for {var_label})")
        return
    header_keys = sorted({k for r, _ in rows for k in r})
    header = "  " + "  ".join(f"{k:>12s}" for k in header_keys)
    header += f"   {'n':>10s} {'<x+>':>9s} {'<x->':>9s} {'σ+':>8s} {'σ-':>8s} {'cov':>10s} {'corr':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cat, m in rows:
        cat_str = "  ".join(f"{str(cat.get(k, '')):>12s}" for k in header_keys)
        s_x = np.sqrt(m['var_x']) if m['var_x'] == m['var_x'] else float('nan')
        s_y = np.sqrt(m['var_y']) if m['var_y'] == m['var_y'] else float('nan')
        print(f"  {cat_str}   "
              f"{m['n']:10.1f} {m['mean_x']:+9.3f} {m['mean_y']:+9.3f} "
              f"{s_x:8.3f} {s_y:8.3f} {m['cov']:+10.4f} {m['corr']:+8.3f}")


# --------------------------- plotting ------------------------------------

def make_plots(rows_by_var, out_prefix, var_key, var_label,
               truth_key, mixed_key, target_key=None,
               truth_label='Truth', mixed_label='Mixed', target_label='Target'):
    """For one variable, save heatmaps for truth/target/mixed + ratio map +
    q-distribution overlay. Sample keys are (process, tag) tuples; pass
    None for a key to skip that sample.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # find the three samples in rows by (process, tag) tuple
    def find(key):
        if key is None:
            return None
        proc, tag = key
        for cat, m, data in rows_by_var:
            if (cat.get('process') == proc
                    and (tag is None or cat.get('tag') == tag)):
                return cat, m, data
        return None

    truth  = find(truth_key)
    mixed  = find(mixed_key)
    target = find(target_key) if target_key else None
    if truth is None or mixed is None:
        print(f"  [plot] skipping {var_key}: missing truth or mixed sample")
        return

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    truth_proc_name  = truth_label
    target_proc_name = target_label if target_key is not None else None
    mixed_proc_name  = mixed_label
    samples = [(truth, truth_proc_name),
               (target, target_proc_name),
               (mixed, mixed_proc_name)]

    Hs   = []
    cxs  = None
    cys  = None
    for sample, _ in samples:
        if sample is None:
            Hs.append(None)
            continue
        _, _, (values, cx, cy) = sample
        if cxs is None:
            cxs, cys = cx, cy
        Hs.append(values / max(values.sum(), 1e-9))
    vmax = max((h.max() for h in Hs if h is not None), default=1.0)

    # Row 0: heatmaps for 4b, 3b, Mixed
    for ax, (sample, proc_name), H in zip(axes[0], samples, Hs):
        if sample is None or H is None:
            ax.set_title(f"{proc_name or '(absent)'}: no data")
            ax.axis('off')
            continue
        _, m, _ = sample
        edges_x = np.r_[cxs - (cxs[1] - cxs[0]) / 2, cxs[-1] + (cxs[1] - cxs[0]) / 2]
        edges_y = np.r_[cys - (cys[1] - cys[0]) / 2, cys[-1] + (cys[1] - cys[0]) / 2]
        im = ax.pcolormesh(edges_x, edges_y, H.T, cmap='viridis',
                           vmin=0, vmax=vmax)
        ax.set_xlabel(f'{var_label} (+ hemi)')
        ax.set_ylabel(f'{var_label} (− hemi)')
        ax.set_title(f"{proc_name}  corr={m['corr']:+.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # Row 1: three ratio maps -- 4b/Mixed, 4b/3b, 3b/Mixed
    H_t, H_tgt, H_m = Hs[0], Hs[1], Hs[2]
    edges_x = np.r_[cxs - (cxs[1] - cxs[0]) / 2, cxs[-1] + (cxs[1] - cxs[0]) / 2]
    edges_y = np.r_[cys - (cys[1] - cys[0]) / 2, cys[-1] + (cys[1] - cys[0]) / 2]

    def _ratio_panel(ax, H_num, H_den, num_name, den_name):
        if H_num is None or H_den is None:
            ax.set_title(f"{num_name} / {den_name}: missing")
            ax.axis('off')
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(H_den > 0, H_num / H_den, np.nan)
        im = ax.pcolormesh(edges_x, edges_y, ratio.T,
                           cmap='RdBu_r', vmin=0.5, vmax=1.5)
        ax.set_xlabel(f'{var_label} (+ hemi)')
        ax.set_ylabel(f'{var_label} (− hemi)')
        ax.set_title(f'{num_name} / {den_name} ratio')
        fig.colorbar(im, ax=ax, fraction=0.046)

    _ratio_panel(axes[1, 0], H_t,   H_m,   truth_proc_name,  mixed_proc_name)
    _ratio_panel(axes[1, 1], H_t,   H_tgt, truth_proc_name,  target_proc_name)
    _ratio_panel(axes[1, 2], H_tgt, H_m,   target_proc_name, mixed_proc_name)

    fig.suptitle(f'Hemi-mixing 2D diagnostic ({var_label})', y=1.02)
    fig.tight_layout()
    out_2d = f'{out_prefix}_{var_key}_2d.png'
    fig.savefig(out_2d, dpi=120, bbox_inches='tight')
    print(f"  wrote {out_2d}")
    plt.close(fig)

    # q-distribution overlay (centered product per event)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for (sample, proc_name), color in zip(
            samples, ['C0', 'C1', 'C2']):
        if sample is None:
            continue
        _, m, (values, cx, cy) = sample
        qc, qh = q_distribution(values, cx, cy, m['mean_x'], m['mean_y'])
        ax.plot(qc, qh, drawstyle='steps-mid', lw=2, color=color,
                label=f"{proc_name}: cov={m['cov']:+.3f}")
    ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'$(x_+ - \\bar x_+)(x_- - \\bar x_-)$  [{var_label}]')
    ax.set_ylabel('density')
    ax.set_title(f'Per-event covariance contribution ({var_label})')
    ax.legend(fontsize=9)
    out_q = f'{out_prefix}_{var_key}_q.png'
    fig.savefig(out_q, dpi=120, bbox_inches='tight')
    print(f"  wrote {out_q}")
    plt.close(fig)


def make_marginal_plots(rows_by_var, out_prefix, var_key, var_label,
                        truth_key, mixed_key, target_key=None,
                        truth_label='Truth', mixed_label='Mixed',
                        target_label='Target'):
    """1D marginal overlays for the + and - hemi distributions with ratio
    panels. Each marginal is x+ (or x-) for the three samples, normalised.
    Sample keys are (process, tag) tuples."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def find(key):
        if key is None:
            return None
        proc, tag = key
        for cat, m, data in rows_by_var:
            if (cat.get('process') == proc
                    and (tag is None or cat.get('tag') == tag)):
                return cat, m, data
        return None

    truth  = find(truth_key)
    mixed  = find(mixed_key)
    target = find(target_key) if target_key else None
    if truth is None or mixed is None:
        return

    truth_proc  = truth_label
    target_proc = target_label if target is not None else None
    mixed_proc  = mixed_label

    samples = [(truth,  truth_proc,  'C0', '-'),
               (target, target_proc, 'C1', '--') if target else None,
               (mixed,  mixed_proc,  'C2', '-')]
    samples = [s for s in samples if s is not None]

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.25)

    # Left col: + hemi marginal; right col: - hemi marginal
    for col, side_idx in enumerate([0, 1]):  # 0=+, 1=-
        ax_main  = fig.add_subplot(gs[0, col])
        ax_ratio = fig.add_subplot(gs[1, col], sharex=ax_main)

        marginals = {}
        for sample, name, color, ls in samples:
            _, m, (values, cx, cy) = sample
            centers = cx if side_idx == 0 else cy
            marg = values.sum(axis=1 - side_idx)
            norm = marg.sum() if marg.sum() > 0 else 1.0
            density = marg / norm / (centers[1] - centers[0])
            marginals[name] = density
            ax_main.plot(centers, density, drawstyle='steps-mid', lw=2,
                         color=color, linestyle=ls,
                         label=f"{name}  μ={m['mean_x' if side_idx == 0 else 'mean_y']:+.3f}  "
                               f"σ={np.sqrt(m['var_x' if side_idx == 0 else 'var_y']):.3f}")
        ax_main.set_ylabel('density')
        ax_main.set_title(f'{var_label} ({"+" if side_idx == 0 else "−"} hemi)')
        ax_main.legend(fontsize=8, loc='upper right')
        ax_main.tick_params(labelbottom=False)

        # Ratios truth/mixed (and truth/target if available)
        centers = cx if side_idx == 0 else cy
        h_truth = marginals.get(truth_proc)
        h_mixed = marginals.get(mixed_proc)
        h_target = marginals.get(target_proc) if target_proc else None
        with np.errstate(divide='ignore', invalid='ignore'):
            if h_truth is not None and h_mixed is not None:
                r_mixed = np.where(h_mixed > 0, h_truth / h_mixed, np.nan)
                ax_ratio.step(centers, r_mixed, where='mid', color='C2',
                              label=f'{truth_label} / {mixed_label}')
            if h_target is not None and h_truth is not None:
                r_target = np.where(h_target > 0, h_truth / h_target, np.nan)
                ax_ratio.step(centers, r_target, where='mid', color='C1',
                              linestyle='--', label=f'{truth_label} / {target_label}')
        ax_ratio.axhline(1, color='gray', lw=0.6, linestyle=':')
        ax_ratio.set_ylim(0.7, 1.3)
        ax_ratio.set_xlabel(f'{var_label} ({"+" if side_idx == 0 else "−"} hemi)')
        ax_ratio.set_ylabel('ratio')
        ax_ratio.legend(fontsize=7, loc='upper right')

    fig.suptitle(f'Per-hemi marginals ({var_label})', y=1.00)
    fig.tight_layout()
    out_1d = f'{out_prefix}_{var_key}_1d.png'
    fig.savefig(out_1d, dpi=120, bbox_inches='tight')
    print(f"  wrote {out_1d}")
    plt.close(fig)


# --------------------------- driver --------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='coffea hist output file')
    ap.add_argument('--variables', default='eta,pz,mass,pt',
                    help='comma-separated subset: eta,pz,mass,pt')
    ap.add_argument('--processes', default=None,
                    help='comma-separated subset of process axis values (default: all)')
    ap.add_argument('--years', default=None,
                    help='comma-separated subset of year axis values (default: all)')
    ap.add_argument('--regions', default=None,
                    help='comma-separated subset of region axis values (e.g. SR,SB)')
    ap.add_argument('--tags', default=None,
                    help='comma-separated subset of tag axis values (e.g. fourTag,threeTag)')
    ap.add_argument('--plot', action='store_true',
                    help='also write 2D heatmaps + ratio + q-distribution PNGs')
    ap.add_argument('--truth-process', default='data',
                    help='process to treat as Truth in plots (default: data)')
    ap.add_argument('--truth-tag', default='fourTag',
                    help='tag value for Truth rows (default: fourTag)')
    ap.add_argument('--mixed-process', default='mixeddata_all',
                    help='process to treat as Mixed in plots')
    ap.add_argument('--mixed-tag', default='fourTag',
                    help='tag value for Mixed rows (default: fourTag)')
    ap.add_argument('--target-process', default=None,
                    help='process to treat as Target (3b) in plots (optional)')
    ap.add_argument('--target-tag', default=None,
                    help='tag value for Target rows (default: same as truth-tag)')
    ap.add_argument('--truth-label',  default='Truth',
                    help='label for the Truth sample in plots (default: Truth)')
    ap.add_argument('--target-label', default='Target',
                    help='label for the Target sample in plots (default: Target)')
    ap.add_argument('--mixed-label',  default='Mixed',
                    help='label for the Mixed sample in plots (default: Mixed)')
    ap.add_argument('--collection', default='can',
                    choices=['can', 'all', 'other', 'tag'],
                    help='jet collection: can (4 HH cand jets, default; the '
                         'HH observable); all (event.Jet, what the matching '
                         'pins -- gives closure check); other (notCanJet, '
                         'the slack carrier); tag (legacy tagJet)')
    ap.add_argument('--inclusive', action='store_true',
                    help='use the *_2d_inclusive histograms (no region axis -- '
                         'all events that passed the selection contribute, not '
                         'just SR or SB)')
    ap.add_argument('--out', default='hemi_diag',
                    help='output prefix for plots (default: hemi_diag)')
    ap.add_argument('--mixed-input', default=None,
                    help='optional second hist file; if given, the mixed-process '
                         'rows come from this file (the primary input still '
                         'supplies truth/target). Useful for comparing un-weighted '
                         'mixed data against the weighted analysis output.')
    ap.add_argument('--sum-axes', default=None,
                    help='comma-separated axis names to sum over before '
                         'iterating (e.g. "region" to combine SR+SB). '
                         'Use with --regions/--tags etc. to restrict first '
                         'then sum what survives.')
    args = ap.parse_args()

    def _flatten(out_obj):
        if isinstance(out_obj, dict) and 'hists' in out_obj:
            return out_obj['hists']
        for v in out_obj.values():
            if isinstance(v, dict) and 'hists' in v:
                return v['hists']
        return out_obj

    hist_dict       = _flatten(load(args.input))
    hist_dict_mixed = _flatten(load(args.mixed_input)) if args.mixed_input else None

    sum_axes = [a.strip() for a in args.sum_axes.split(',')] if args.sum_axes else []

    def _maybe_sum_axes(h, restrict_dict):
        """Restrict then project-sum axes listed in sum_axes."""
        if not sum_axes:
            return h, restrict_dict
        h_out = h
        for ax_name in sum_axes:
            allowed = restrict_dict.get(ax_name)
            if allowed:
                pieces = [h_out[{ax_name: v}] for v in allowed if v in list(h_out.axes[ax_name])]
                if not pieces:
                    return h_out, restrict_dict
                h_out = pieces[0]
                for p in pieces[1:]:
                    h_out = h_out + p
            else:
                h_out = h_out[{ax_name: sum}]
        new_restrict = {k: v for k, v in restrict_dict.items() if k not in sum_axes}
        return h_out, new_restrict

    var_filter = {v.strip() for v in args.variables.split(',')}
    restrict = {
        'process': args.processes.split(',') if args.processes else None,
        'year':    args.years.split(',')     if args.years     else None,
        'region':  args.regions.split(',')   if args.regions   else None,
        'tag':     args.tags.split(',')      if args.tags      else None,
    }

    print(f"Input: {args.input}")
    print(f"Collection: {args.collection}"
          + ("  (inclusive: no region axis)" if args.inclusive else ""))
    print()

    var_list = _var_list(args.collection, inclusive=args.inclusive)
    for key, hist_name, label in var_list:
        if key not in var_filter:
            continue
        if hist_name not in hist_dict:
            print(f"== {key}: '{hist_name}' not in input -- skipping")
            continue

        h_full = hist_dict[hist_name]
        cx = h_full.axes[-2].centers
        cy = h_full.axes[-1].centers

        print(f"== {key}  ({label})  axes={[ax.name for ax in h_full.axes]}")
        if sum_axes:
            print(f"  (summing over axes: {sum_axes})")
        rows = []
        rows_with_data = []

        def _collect(_h, _restrict):
            out_rows = []
            for cat, sub in iter_category_combinations(_h, restrict=_restrict):
                values = sub.values()
                if values.sum() == 0:
                    continue
                m = moments_from_2d(values, cx, cy)
                out_rows.append((cat, m, (values, cx, cy)))
            return out_rows

        if hist_dict_mixed is not None and hist_name in hist_dict_mixed:
            # Primary file: everything except mixed-process. Secondary: mixed only.
            primary_restrict = dict(restrict,
                                    process=[p for p in (restrict['process'] or [])
                                             if p != args.mixed_process]
                                    if restrict['process'] else None)
            h_p, r_p = _maybe_sum_axes(h_full, primary_restrict)
            rows_with_data += _collect(h_p, r_p)
            h_s, r_s = _maybe_sum_axes(
                hist_dict_mixed[hist_name],
                dict(restrict, process=[args.mixed_process]))
            rows_with_data += _collect(h_s, r_s)
            print(f"  (mixed rows pulled from {args.mixed_input})")
        else:
            h_used, r_used = _maybe_sum_axes(h_full, restrict)
            rows_with_data = _collect(h_used, r_used)

        rows = [(cat, m) for cat, m, _ in rows_with_data]
        print_table(rows, label)

        if args.plot:
            # Group by everything except process and tag — those are the
            # axes that identify "samples" (4b/3b/Mixed are distinguished
            # by their process+tag pairs).
            def group_key(cat):
                return tuple(sorted((k, v) for k, v in cat.items()
                                     if k not in ('process', 'tag')))
            groups = {}
            for cat, m, data in rows_with_data:
                groups.setdefault(group_key(cat), []).append((cat, m, data))

            truth_key  = (args.truth_process,  args.truth_tag)
            mixed_key  = (args.mixed_process,  args.mixed_tag)
            target_key = ((args.target_process,
                           args.target_tag or args.truth_tag)
                          if args.target_process else None)

            for gkey, items in groups.items():
                if len(items) < 2:
                    continue
                tag_str = '_'.join(f"{k}-{v}" for k, v in gkey)
                prefix = f"{args.out}_{tag_str or 'inclusive'}"
                make_plots(items, prefix, key, label,
                           truth_key, mixed_key, target_key,
                           truth_label=args.truth_label,
                           mixed_label=args.mixed_label,
                           target_label=args.target_label)
                make_marginal_plots(items, prefix, key, label,
                                    truth_key, mixed_key, target_key,
                                    truth_label=args.truth_label,
                                    mixed_label=args.mixed_label,
                                    target_label=args.target_label)
        print()


if __name__ == '__main__':
    main()
