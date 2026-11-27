#!/usr/bin/env python3
"""Score FvT optimization sweep variants using paired, signal-projected SR residuals.

Computes per-(run, mix) weighted least-squares spurious signal (mu_spur),
orthogonalized Fourier bias projection norm (orders 1..3), and chi2/ndf in SR
for SvB_MA.ps_ttHbb, takes paired differences on identical synthetic mixes against
the reference run, measures training seed noise from ref_s1 - ref_s0, and emits
a pre-registered PASS/KILL verdict + JSON summary + HTML dashboard.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import hist
import numpy as np
from coffea.util import load


def _load_coffea(path_or_url: str) -> dict:
    if "://" not in path_or_url:
        return load(path_or_url)
    with tempfile.NamedTemporaryFile(suffix=".coffea", delete=False) as tf:
        tmp = tf.name
    try:
        xrdcp_bin = shutil.which("xrdcp") or (
            "/cvmfs/cms.cern.ch/common/xrdcp" if os.path.exists("/cvmfs/cms.cern.ch/common/xrdcp") else None
        )
        if xrdcp_bin:
            subprocess.run([xrdcp_bin, "-f", "-s", path_or_url, tmp], check=True)
        else:
            import fsspec
            with fsspec.open(path_or_url, "rb") as fin, open(tmp, "wb") as fout:
                fout.write(fin.read())
        return load(tmp)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _extract_1d(h, process: str, tag: str, region: str = "SR", year: str | None = None):
    sel = {"process": process, "tag": tag, "region": region}
    if year is not None:
        sel["year"] = year
    elif "year" in h.axes.name:
        sel["year"] = sum
    for ax in h.axes:
        if ax.name.startswith(("pass", "fail")) and ax.name not in sel:
            sel[ax.name] = sum
    h_sub = h[sel]
    vals = np.asarray(h_sub.values(), dtype=np.float64).reshape(-1)
    vars_ = np.asarray(h_sub.variances(), dtype=np.float64).reshape(-1) if h_sub.variances() is not None else np.maximum(vals, 0.0)
    return vals, vars_


def _rebin(vals: np.ndarray, vars_: np.ndarray, rebin: int) -> tuple[np.ndarray, np.ndarray]:
    if rebin <= 1:
        return vals, vars_
    n = len(vals)
    k = n // rebin
    if k * rebin != n:
        raise ValueError(f"Cannot rebin {n} bins by {rebin}")
    return vals.reshape(k, rebin).sum(axis=1), vars_.reshape(k, rebin).sum(axis=1)


def _fourier_basis(n_bins: int, max_order: int = 3) -> np.ndarray:
    """Columns are [1, sin(k*pi*x), cos(k*pi*x)] evaluated at bin midpoints on [0, 1]."""
    x = np.linspace(0.0, 1.0, n_bins + 1)
    xm = 0.5 * (x[:-1] + x[1:])
    cols = [np.ones_like(xm)]
    for k in range(1, max_order + 1):
        cols.append(np.sin(k * np.pi * xm))
        cols.append(np.cos(k * np.pi * xm))
    return np.column_stack(cols)


def compute_mix_metrics(
    target: np.ndarray,
    target_var: np.ndarray,
    model: np.ndarray,
    model_var: np.ndarray,
    signal: np.ndarray,
    max_fourier_order: int = 3,
) -> dict:
    diff = target - model
    tot_var = np.maximum(target_var + model_var, 1e-12)
    inv_var = 1.0 / tot_var
    sigma = np.sqrt(tot_var)
    pulls = diff / sigma

    # WLS spurious signal mu_spur: minimizes (diff - mu * s)^T W (diff - mu * s)
    s_w_s = float(np.sum(inv_var * signal * signal))
    if s_w_s > 0:
        mu_spur = float(np.sum(inv_var * signal * diff) / s_w_s)
        mu_spur_err = float(1.0 / math.sqrt(s_w_s))
    else:
        mu_spur = 0.0
        mu_spur_err = 0.0

    # Orthogonalized Fourier bias projection norm in the W-metric (unit-variance pull space)
    B = _fourier_basis(len(diff), max_order=max_fourier_order)
    Bw = B / sigma[:, None]
    Q, _ = np.linalg.qr(Bw)
    coeff = Q.T @ pulls
    bias_norm = float(np.linalg.norm(coeff))

    chi2 = float(np.sum(pulls ** 2))
    ndf = max(1, len(diff))
    return {
        "mu_spur": mu_spur,
        "abs_mu_spur": abs(mu_spur),
        "mu_spur_err": mu_spur_err,
        "bias_norm": bias_norm,
        "chi2_ndf": chi2 / ndf,
        "pull_rms": float(np.sqrt(np.mean(pulls ** 2))),
        "pulls": pulls.tolist(),
        "target_sum": float(np.sum(target)),
        "model_sum": float(np.sum(model)),
    }


def _find_signal_hist(nom_hists: dict, hist_key: str):
    if hist_key in nom_hists:
        return nom_hists[hist_key]
    raise KeyError(f"Signal histogram {hist_key!r} not found in nominal_hists")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="Ordered list of run names")
    ap.add_argument("--f1-files", nargs="+", required=True, help="histAll_mixeddata_bkgs.coffea per run (same order)")
    ap.add_argument("--mixes", nargs="+", required=True, help="Mix indices (e.g. 0 1 2 3)")
    ap.add_argument("--a3-dir", required=True, help="Directory with histAll_<channel>_mixeddata_v<m>.coffea")
    ap.add_argument("--channel", default="ttHbb")
    ap.add_argument("--nominal-hists", required=True, help="Path or root:// URL to nominal coffea file with ttHbb signal")
    ap.add_argument("--signal-process", default="ttHbb")
    ap.add_argument("--hist-key", default="SvB_MA.ps_ttHbb")
    ap.add_argument("--rebin", type=int, default=16)
    ap.add_argument("--extra-rebins", nargs="*", type=int, default=[12, 15, 24])
    ap.add_argument("--reference", default="ref")
    ap.add_argument("--threshold-sigma", type=float, default=2.0)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-html", required=True)
    args = ap.parse_args()

    if len(args.runs) != len(args.f1_files):
        raise ValueError("--runs and --f1-files must have the same length")

    clf_name, var_stem = args.hist_key.split(".", 1)
    rebins = [args.rebin] + [r for r in args.extra_rebins if r != args.rebin]

    # Load signal template
    nom_data = _load_coffea(args.nominal_hists)
    nom_hists = nom_data.get("hists", nom_data)
    h_sig = _find_signal_hist(nom_hists, args.hist_key)
    sig_proc = args.signal_process
    if sig_proc not in h_sig.axes["process"]:
        cand = [p for p in h_sig.axes["process"] if "ttH" in p or "TTH" in p]
        if not cand:
            raise KeyError(f"Signal process {sig_proc!r} not in {list(h_sig.axes['process'])}")
        sig_proc = cand[0]
    sig_raw, _ = _extract_1d(h_sig, sig_proc, "fourTag", region="SR")

    # Load synthetic 4b targets for each mix
    targets_raw = {}
    years = list(h_sig.axes["year"]) if "year" in h_sig.axes.name else []
    for m in args.mixes:
        a3_file = os.path.join(args.a3_dir, f"histAll_{args.channel}_mixeddata_v{m}.coffea")
        a3_hists = load(a3_file)["hists"]
        mh = a3_hists[args.hist_key]
        mproc = f"mix_v{m}" if f"mix_v{m}" in mh.axes["process"] else next(p for p in mh.axes["process"] if "mix" in p)
        t_val, t_var = _extract_1d(mh, mproc, "fourTag", region="SR")
        t_by_yr = {yr: _extract_1d(mh, mproc, "fourTag", region="SR", year=yr) for yr in mh.axes["year"]}
        targets_raw[m] = {"sum": (t_val, t_var), "years": t_by_yr}

    # Load models for each run & mix
    models_raw = {r: {} for r in args.runs}
    for run, f1_path in zip(args.runs, args.f1_files):
        f1_hists = load(f1_path)["hists"]
        for m in args.mixes:
            key = f"{clf_name}_v{m}.{var_stem}"
            h_mod = f1_hists[key]
            d3_val, d3_var = _extract_1d(h_mod, "data", "threeTag", region="SR")
            tt_val, tt_var = _extract_1d(h_mod, "TTbar4b_from_d3", "threeTag", region="SR")
            m_by_yr = {}
            for yr in h_mod.axes["year"]:
                yd3, yd3v = _extract_1d(h_mod, "data", "threeTag", region="SR", year=yr)
                ytt, yttv = _extract_1d(h_mod, "TTbar4b_from_d3", "threeTag", region="SR", year=yr)
                m_by_yr[yr] = (yd3 + ytt, yd3v + yttv)
            models_raw[run][m] = {"sum": (d3_val + tt_val, d3_var + tt_var), "years": m_by_yr}

    # Evaluate across rebins
    results_by_rebin = {}
    for rb in rebins:
        sig_rb, _ = _rebin(sig_raw, np.zeros_like(sig_raw), rb)
        per_run = {}
        for run in args.runs:
            mix_res = {}
            pulls_mat = []
            for m in args.mixes:
                t_val, t_var = _rebin(*targets_raw[m]["sum"], rb)
                m_val, m_var = _rebin(*models_raw[run][m]["sum"], rb)
                met = compute_mix_metrics(t_val, t_var, m_val, m_var, sig_rb)
                mix_res[m] = met
                pulls_mat.append(met["pulls"])
            pulls_arr = np.asarray(pulls_mat, dtype=np.float64)
            cross_mix_spread = float(np.mean(np.std(pulls_arr, axis=0, ddof=1))) if len(args.mixes) > 1 else 0.0
            per_run[run] = {
                "mixes": mix_res,
                "mean_abs_mu_spur": float(np.mean([mix_res[m]["abs_mu_spur"] for m in args.mixes])),
                "mean_mu_spur": float(np.mean([mix_res[m]["mu_spur"] for m in args.mixes])),
                "mean_bias_norm": float(np.mean([mix_res[m]["bias_norm"] for m in args.mixes])),
                "mean_chi2_ndf": float(np.mean([mix_res[m]["chi2_ndf"] for m in args.mixes])),
                "cross_mix_spread": cross_mix_spread,
            }

        # Identify reference runs (ref_s0, ref_s1 if seeded)
        ref_runs = [r for r in args.runs if r == args.reference or r.startswith(f"{args.reference}_s")]
        if not ref_runs:
            ref_runs = [args.runs[0]]
        ref0 = ref_runs[0]

        if len(ref_runs) >= 2:
            ref1 = ref_runs[1]
            d_seed_mu = [per_run[ref1]["mixes"][m]["abs_mu_spur"] - per_run[ref0]["mixes"][m]["abs_mu_spur"] for m in args.mixes]
            d_seed_bias = [per_run[ref1]["mixes"][m]["bias_norm"] - per_run[ref0]["mixes"][m]["bias_norm"] for m in args.mixes]
            sigma_train_mu = max(1e-6, float(np.std(d_seed_mu, ddof=1)) if len(d_seed_mu) > 1 else abs(d_seed_mu[0]))
            sigma_train_bias = max(1e-6, float(np.std(d_seed_bias, ddof=1)) if len(d_seed_bias) > 1 else abs(d_seed_bias[0]))
        else:
            sigma_train_mu = 0.0
            sigma_train_bias = 0.0

        k_mixes = max(1, len(args.mixes))
        thresh_mu = args.threshold_sigma * sigma_train_mu / math.sqrt(k_mixes)
        ref_spread = float(np.mean([per_run[r]["cross_mix_spread"] for r in ref_runs]))

        paired = {}
        for run in args.runs:
            # Paired comparison against mean of ref seeds per mix (reduces ref seed noise by sqrt(2))
            d_mu = [
                float(np.mean([per_run[r]["mixes"][m]["abs_mu_spur"] for r in ref_runs])) - per_run[run]["mixes"][m]["abs_mu_spur"]
                for m in args.mixes
            ]
            d_bias = [
                float(np.mean([per_run[r]["mixes"][m]["bias_norm"] for r in ref_runs])) - per_run[run]["mixes"][m]["bias_norm"]
                for m in args.mixes
            ]
            mean_d_mu = float(np.mean(d_mu))
            mean_d_bias = float(np.mean(d_bias))
            spread_ratio = (per_run[run]["cross_mix_spread"] / ref_spread) if ref_spread > 0 else 1.0

            if run in ref_runs:
                verdict = "REFERENCE"
            else:
                beats_mu = mean_d_mu > thresh_mu
                bias_ok = mean_d_bias >= -sigma_train_bias
                spread_ok = spread_ratio <= 1.05
                verdict = "PASS" if (beats_mu and bias_ok and spread_ok) else "KILL"

            paired[run] = {
                "delta_abs_mu_spur_per_mix": {m: v for m, v in zip(args.mixes, d_mu)},
                "mean_delta_abs_mu_spur": mean_d_mu,
                "delta_bias_norm_per_mix": {m: v for m, v in zip(args.mixes, d_bias)},
                "mean_delta_bias_norm": mean_d_bias,
                "spread_ratio_vs_ref": spread_ratio,
                "verdict": verdict,
            }

        results_by_rebin[str(rb)] = {
            "rebin": rb,
            "n_bins": len(sig_rb),
            "ref_runs": ref_runs,
            "sigma_train_mu": sigma_train_mu,
            "sigma_train_bias": sigma_train_bias,
            "threshold_mu": thresh_mu,
            "runs": per_run,
            "paired": paired,
        }

    # Per-year breakdown at primary rebin
    per_year = {}
    for yr in years:
        sig_y, _ = _rebin(*_extract_1d(h_sig, sig_proc, "fourTag", region="SR", year=yr), args.rebin)
        per_year[yr] = {}
        for run in args.runs:
            ms = {}
            for m in args.mixes:
                tv, tv2 = _rebin(*targets_raw[m]["years"][yr], args.rebin)
                mv, mv2 = _rebin(*models_raw[run][m]["years"][yr], args.rebin)
                ms[m] = compute_mix_metrics(tv, tv2, mv, mv2, sig_y)
            per_year[yr][run] = {
                "mean_abs_mu_spur": float(np.mean([ms[m]["abs_mu_spur"] for m in args.mixes])),
                "mean_bias_norm": float(np.mean([ms[m]["bias_norm"] for m in args.mixes])),
                "mean_chi2_ndf": float(np.mean([ms[m]["chi2_ndf"] for m in args.mixes])),
            }

    summary = {
        "primary_rebin": args.rebin,
        "mixes": args.mixes,
        "runs": args.runs,
        "reference": args.reference,
        "threshold_sigma": args.threshold_sigma,
        "hist_key": args.hist_key,
        "by_rebin": results_by_rebin,
        "per_year_primary": per_year,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # Render HTML dashboard
    prim = results_by_rebin[str(args.rebin)]
    rows = []
    for run in args.runs:
        r = prim["runs"][run]
        p = prim["paired"][run]
        badge_cls = {"PASS": "pass", "KILL": "kill", "REFERENCE": "ref"}[p["verdict"]]
        rows.append(
            f"<tr>"
            f"<td><b>{html.escape(run)}</b></td>"
            f"<td><span class='badge {badge_cls}'>{p['verdict']}</span></td>"
            f"<td>{r['mean_abs_mu_spur']:.4f}</td>"
            f"<td>{r['mean_mu_spur']:+.4f}</td>"
            f"<td>{p['mean_delta_abs_mu_spur']:+.4f}</td>"
            f"<td>{r['mean_bias_norm']:.3f}</td>"
            f"<td>{p['mean_delta_bias_norm']:+.3f}</td>"
            f"<td>{r['mean_chi2_ndf']:.3f}</td>"
            f"<td>{r['cross_mix_spread']:.3f} ({p['spread_ratio_vs_ref']:.2f}x)</td>"
            f"</tr>"
        )

    rebin_sections = []
    for rb in rebins:
        sec = results_by_rebin[str(rb)]
        sub_rows = []
        for run in args.runs:
            r = sec["runs"][run]
            p = sec["paired"][run]
            sub_rows.append(
                f"<tr><td>{html.escape(run)}</td><td>{p['verdict']}</td>"
                f"<td>{r['mean_abs_mu_spur']:.4f}</td><td>{p['mean_delta_abs_mu_spur']:+.4f}</td>"
                f"<td>{r['mean_bias_norm']:.3f}</td><td>{r['mean_chi2_ndf']:.3f}</td></tr>"
            )
        rebin_sections.append(
            f"<h3>Rebin {rb} ({sec['n_bins']} bins) — noise &sigma;_train(&mu;)={sec['sigma_train_mu']:.4f}, threshold={sec['threshold_mu']:.4f}</h3>"
            f"<table><tr><th>Run</th><th>Verdict</th><th>Mean |&mu;_spur|</th><th>&Delta;|&mu;_spur| (+better)</th><th>Mean Bias Norm</th><th>Mean &chi;&sup2;/ndf</th></tr>"
            + "".join(sub_rows) + "</table>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>fvt_opt Dashboard ({html.escape(args.hist_key)})</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
h1, h2, h3 {{ color: #0f172a; }}
.card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 8px; }}
th, td {{ border: 1px solid #e2e8f0; padding: 7px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f1f5f9; }}
.badge {{ padding: 2px 8px; border-radius: 999px; font-weight: 600; font-size: 12px; }}
.pass {{ background: #dcfce7; color: #166534; }}
.kill {{ background: #fee2e2; color: #991b1b; }}
.ref {{ background: #e0e7ff; color: #1e40af; }}
</style></head><body>
<h1>FvT Optimization Sweep Dashboard</h1>
<div class="card">
  <b>Observable:</b> {html.escape(args.hist_key)} | <b>Primary Rebin:</b> {args.rebin} ({prim['n_bins']} bins) |
  <b>Mixes:</b> {', '.join(args.mixes)} | <b>Noise Floor &sigma;_train(&mu;):</b> {prim['sigma_train_mu']:.4f} |
  <b>Threshold ({args.threshold_sigma}&sigma;/&radic;K):</b> {prim['threshold_mu']:.4f} |
  <b>Bias Noise &sigma;_train:</b> {prim['sigma_train_bias']:.3f}
</div>
<div class="card">
  <h2>Primary Summary (Rebin {args.rebin})</h2>
  <table>
    <tr><th>Run</th><th>Verdict</th><th>Mean |&mu;_spur|</th><th>Mean &mu;_spur</th><th>Paired &Delta;|&mu;_spur| (+better)</th>
    <th>Mean Bias Norm</th><th>Paired &Delta;Bias (+better)</th><th>Mean &chi;&sup2;/ndf</th><th>Cross-Mix Spread</th></tr>
    {''.join(rows)}
  </table>
</div>
<div class="card">
  <h2>All Rebins</h2>
  {''.join(rebin_sections)}
</div>
</body></html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(args.out_html)), exist_ok=True)
    with open(args.out_html, "w") as f:
        f.write(html_doc)
    print(f"Wrote {args.out_json} and {args.out_html}")


if __name__ == "__main__":
    main()
