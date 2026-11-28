#!/usr/bin/env python3
"""Score FvT optimization sweep variants using shape-only (c0-marginalized) SR closure.

For each (run, mix), separates overall SR normalization (M/T) from pure SR shape
closure on SvB_MA.ps_ttHbb:
  - Rescales M -> M_s = M * (sum T / sum M) and jointly fits [M_s, signal] by WLS
    to marginalize overall normalization c0, giving mu_shape (per-mix) and
    shape_mu_ens (ensemble-averaged Stage-1 proxy).
  - Computes shape chi2/ndf (ndf = n_bins - 1), pure-shape Fourier bias norm
    (orders 1..3 skipping constant c0), and cross-mix shape pull spread.
  - Aggregates runs sharing a variant base name across seeds (e.g. ref_s0/s1/s2)
    and compares variants using seed-averaged means and combined seed SE.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PARENTS = Path(__file__).resolve().parents
_REPO_ROOT = str(_PARENTS[2]) if len(_PARENTS) > 2 else os.getcwd()
for _p in (".", _REPO_ROOT, os.path.join(_REPO_ROOT, "coffea4bees")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    vars_ = (
        np.asarray(h_sub.variances(), dtype=np.float64).reshape(-1)
        if h_sub.variances() is not None
        else np.maximum(vals, 0.0)
    )
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
    sb_target_sum: float = 0.0,
    sb_model_sum: float = 0.0,
    max_fourier_order: int = 3,
) -> dict:
    # Raw (unnormalized) metrics (informational only)
    diff_raw = target - model
    tot_var_raw = np.maximum(target_var + model_var, 1e-12)
    inv_var_raw = 1.0 / tot_var_raw
    s_w_s_raw = float(np.sum(inv_var_raw * signal * signal))
    mu_raw = float(np.sum(inv_var_raw * signal * diff_raw) / s_w_s_raw) if s_w_s_raw > 0 else 0.0

    # Shape-only (scale M -> M_s to match target sum, plus joint [M_s, signal] WLS fit to marginalize c0)
    t_sum = float(np.sum(target))
    m_sum = float(np.sum(model))
    k = (t_sum / m_sum) if m_sum > 0 else 1.0
    model_s = model * k
    model_var_s = model_var * (k ** 2)

    diff_shp = target - model_s
    tot_var_shp = np.maximum(target_var + model_var_s, 1e-12)
    inv_var_shp = 1.0 / tot_var_shp
    sigma_shp = np.sqrt(tot_var_shp)
    pulls_shp = diff_shp / sigma_shp

    X = np.column_stack([model_s, signal])
    XtW = X.T * inv_var_shp[None, :]
    xtwx = XtW @ X
    try:
        beta = np.linalg.solve(xtwx, XtW @ diff_shp)
        cov = np.linalg.inv(xtwx)
        mu_shape = float(beta[1])
        mu_shape_err = float(math.sqrt(max(0.0, cov[1, 1])))
    except np.linalg.LinAlgError:
        mu_shape = 0.0
        mu_shape_err = 0.0

    # Pure-shape Fourier bias norm (orders 1..max_fourier_order, excluding constant c0 mode)
    B = _fourier_basis(len(diff_shp), max_order=max_fourier_order)
    Bw = B / sigma_shp[:, None]
    Q, _ = np.linalg.qr(Bw)
    coeff = Q.T @ pulls_shp
    bias_shape_norm = float(np.linalg.norm(coeff[1:]))

    chi2_shp = float(np.sum(pulls_shp ** 2))
    ndf_shp = max(1, len(diff_shp) - 1)
    return {
        "target_sum": t_sum,
        "model_sum": m_sum,
        "sr_mt": (m_sum / t_sum) if t_sum > 0 else 1.0,
        "sb_mt": (sb_model_sum / sb_target_sum) if sb_target_sum > 0 else 1.0,
        "mu_raw": mu_raw,
        "abs_mu_raw": abs(mu_raw),
        "mu_shape": mu_shape,
        "abs_mu_shape": abs(mu_shape),
        "mu_shape_err": mu_shape_err,
        "bias_shape_norm": bias_shape_norm,
        "chi2_ndf_shape": chi2_shp / ndf_shp,
        "pull_rms_shape": float(np.sqrt(np.mean(pulls_shp ** 2))),
        "pulls_shape": pulls_shp.tolist(),
    }


def _variant_of(run_name: str) -> str:
    return re.sub(r"_s\d+$", "", run_name)


def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = float(np.mean(vals))
    s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return m, s


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
    ap.add_argument("--threshold-sigma", type=float, default=1.0)
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
    if args.hist_key not in nom_hists:
        raise KeyError(f"Signal histogram {args.hist_key!r} not found in nominal_hists")
    h_sig = nom_hists[args.hist_key]
    sig_proc = args.signal_process
    if sig_proc not in h_sig.axes["process"]:
        cand = [p for p in h_sig.axes["process"] if "ttH" in p or "TTH" in p]
        if not cand:
            raise KeyError(f"Signal process {sig_proc!r} not in {list(h_sig.axes['process'])}")
        sig_proc = cand[0]
    sig_raw, _ = _extract_1d(h_sig, sig_proc, "fourTag", region="SR")
    years = list(h_sig.axes["year"]) if "year" in h_sig.axes.name else []

    # Load synthetic 4b targets (SR + SB)
    targets_raw = {}
    for m in args.mixes:
        a3_file = os.path.join(args.a3_dir, f"histAll_{args.channel}_mixeddata_v{m}.coffea")
        mh = load(a3_file)["hists"][args.hist_key]
        mproc = f"mix_v{m}" if f"mix_v{m}" in mh.axes["process"] else next(p for p in mh.axes["process"] if "mix" in p)
        t_val, t_var = _extract_1d(mh, mproc, "fourTag", region="SR")
        t_sb_val, _ = _extract_1d(mh, mproc, "fourTag", region="SB")
        t_by_yr = {yr: _extract_1d(mh, mproc, "fourTag", region="SR", year=yr) for yr in mh.axes["year"]}
        targets_raw[m] = {"sum": (t_val, t_var), "sb_sum": float(np.sum(t_sb_val)), "years": t_by_yr}

    # Load models (SR + SB)
    models_raw = {r: {} for r in args.runs}
    for run, f1_path in zip(args.runs, args.f1_files):
        f1_hists = load(f1_path)["hists"]
        for m in args.mixes:
            h_mod = f1_hists[f"{clf_name}_v{m}.{var_stem}"]
            d3_val, d3_var = _extract_1d(h_mod, "data", "threeTag", region="SR")
            tt_val, tt_var = _extract_1d(h_mod, "TTbar4b_from_d3", "threeTag", region="SR")
            d3_sb, _ = _extract_1d(h_mod, "data", "threeTag", region="SB")
            tt_sb, _ = _extract_1d(h_mod, "TTbar4b_from_d3", "threeTag", region="SB")
            m_by_yr = {}
            for yr in h_mod.axes["year"]:
                yd3, yd3v = _extract_1d(h_mod, "data", "threeTag", region="SR", year=yr)
                ytt, yttv = _extract_1d(h_mod, "TTbar4b_from_d3", "threeTag", region="SR", year=yr)
                m_by_yr[yr] = (yd3 + ytt, yd3v + yttv)
            models_raw[run][m] = {
                "sum": (d3_val + tt_val, d3_var + tt_var),
                "sb_sum": float(np.sum(d3_sb) + np.sum(tt_sb)),
                "years": m_by_yr,
            }

    # Evaluate across rebins
    results_by_rebin = {}
    for rb in rebins:
        sig_rb, _ = _rebin(sig_raw, np.zeros_like(sig_raw), rb)
        per_run = {}
        for run in args.runs:
            mix_res = {}
            pulls_mat = []
            diff_ens = np.zeros_like(sig_rb)
            var_ens = np.zeros_like(sig_rb)
            t_avg = np.zeros_like(sig_rb)
            n_m = len(args.mixes)
            for m in args.mixes:
                t_val, t_var = _rebin(*targets_raw[m]["sum"], rb)
                m_val, m_var = _rebin(*models_raw[run][m]["sum"], rb)
                met = compute_mix_metrics(
                    t_val, t_var, m_val, m_var, sig_rb,
                    sb_target_sum=targets_raw[m]["sb_sum"],
                    sb_model_sum=models_raw[run][m]["sb_sum"],
                )
                mix_res[m] = met
                pulls_mat.append(met["pulls_shape"])

                k = (float(np.sum(t_val)) / float(np.sum(m_val))) if np.sum(m_val) > 0 else 1.0
                mv_s, mv2_s = m_val * k, m_var * (k ** 2)
                diff_ens += (t_val - mv_s) / n_m
                var_ens += (t_var + mv2_s) / (n_m ** 2)
                t_avg += t_val / n_m

            we = 1.0 / np.maximum(var_ens, 1e-12)
            X_e = np.column_stack([t_avg, sig_rb])
            XtW_e = X_e.T * we[None, :]
            try:
                shape_mu_ens = float(np.linalg.solve(XtW_e @ X_e, XtW_e @ diff_ens)[1])
            except np.linalg.LinAlgError:
                shape_mu_ens = 0.0

            pulls_arr = np.asarray(pulls_mat, dtype=np.float64)
            shape_spread = float(np.mean(np.std(pulls_arr, axis=0, ddof=1))) if n_m > 1 else 0.0
            sr_mts = [mix_res[m]["sr_mt"] for m in args.mixes]
            sb_mts = [mix_res[m]["sb_mt"] for m in args.mixes]
            per_run[run] = {
                "variant": _variant_of(run),
                "mixes": mix_res,
                "sr_mt_mean": float(np.mean(sr_mts)),
                "sr_mt_std": float(np.std(sr_mts, ddof=1)) if n_m > 1 else 0.0,
                "sb_mt_mean": float(np.mean(sb_mts)),
                "sb_mt_std": float(np.std(sb_mts, ddof=1)) if n_m > 1 else 0.0,
                "mean_abs_mu_shape": float(np.mean([mix_res[m]["abs_mu_shape"] for m in args.mixes])),
                "shape_mu_ens": shape_mu_ens,
                "mean_abs_mu_raw": float(np.mean([mix_res[m]["abs_mu_raw"] for m in args.mixes])),
                "mean_bias_shape_norm": float(np.mean([mix_res[m]["bias_shape_norm"] for m in args.mixes])),
                "mean_chi2_ndf_shape": float(np.mean([mix_res[m]["chi2_ndf_shape"] for m in args.mixes])),
                "shape_spread": shape_spread,
            }

        # Aggregate across seeds per variant
        variant_order = []
        by_var = {}
        for run in args.runs:
            v = _variant_of(run)
            if v not in by_var:
                variant_order.append(v)
                by_var[v] = []
            by_var[v].append(run)

        variants_summary = {}
        for v in variant_order:
            rlist = by_var[v]
            mu_m, mu_s = _mean_std([per_run[r]["mean_abs_mu_shape"] for r in rlist])
            ens_m, ens_s = _mean_std([abs(per_run[r]["shape_mu_ens"]) for r in rlist])
            c2_m, c2_s = _mean_std([per_run[r]["mean_chi2_ndf_shape"] for r in rlist])
            bs_m, bs_s = _mean_std([per_run[r]["mean_bias_shape_norm"] for r in rlist])
            sp_m, sp_s = _mean_std([per_run[r]["shape_spread"] for r in rlist])
            srm_m, srm_s = _mean_std([per_run[r]["sr_mt_mean"] for r in rlist])
            srstd_m, _ = _mean_std([per_run[r]["sr_mt_std"] for r in rlist])
            variants_summary[v] = {
                "runs": rlist,
                "n_seeds": len(rlist),
                "sr_mt_mean": srm_m,
                "sr_mt_seed_std": srm_s,
                "sr_mt_mix_std_mean": srstd_m,
                "shape_abs_mu_mean": mu_m,
                "shape_abs_mu_seed_std": mu_s,
                "shape_abs_mu_ens_mean": ens_m,
                "shape_abs_mu_ens_seed_std": ens_s,
                "shape_chi2_mean": c2_m,
                "shape_chi2_seed_std": c2_s,
                "shape_bias_mean": bs_m,
                "shape_bias_seed_std": bs_s,
                "shape_spread_mean": sp_m,
                "shape_spread_seed_std": sp_s,
            }

        ref_v = args.reference if args.reference in variants_summary else variant_order[0]
        rv = variants_summary[ref_v]
        for v in variant_order:
            vs = variants_summary[v]
            if v == ref_v:
                vs["verdict"] = "REFERENCE"
                vs["delta_shape_mu_vs_ref"] = 0.0
                vs["se_shape_mu"] = rv["shape_abs_mu_seed_std"] / math.sqrt(max(1, rv["n_seeds"]))
                continue
            d_mu = rv["shape_abs_mu_mean"] - vs["shape_abs_mu_mean"]
            d_chi2 = rv["shape_chi2_mean"] - vs["shape_chi2_mean"]
            se_mu = math.sqrt(
                (rv["shape_abs_mu_seed_std"] ** 2) / max(1, rv["n_seeds"])
                + (vs["shape_abs_mu_seed_std"] ** 2) / max(1, vs["n_seeds"])
            )
            se_chi2 = math.sqrt(
                (rv["shape_chi2_seed_std"] ** 2) / max(1, rv["n_seeds"])
                + (vs["shape_chi2_seed_std"] ** 2) / max(1, vs["n_seeds"])
            )
            thr = args.threshold_sigma * max(se_mu, 1e-6)
            if d_mu > thr and d_chi2 >= -max(se_chi2, 1e-6):
                verdict = "BETTER"
            elif d_mu < -thr or d_chi2 < -2.0 * max(se_chi2, 1e-6):
                verdict = "WORSE"
            else:
                verdict = "INCONCLUSIVE"
            vs["verdict"] = verdict
            vs["delta_shape_mu_vs_ref"] = d_mu
            vs["delta_shape_chi2_vs_ref"] = d_chi2
            vs["se_shape_mu"] = se_mu

        results_by_rebin[str(rb)] = {
            "rebin": rb,
            "n_bins": len(sig_rb),
            "reference_variant": ref_v,
            "variants": variants_summary,
            "runs": per_run,
        }

    # Per-year breakdown at primary rebin
    per_year = {}
    for yr in years:
        sig_y, _ = _rebin(*_extract_1d(h_sig, sig_proc, "fourTag", region="SR", year=yr), args.rebin)
        per_year[yr] = {}
        for run in args.runs:
            ms = []
            for m in args.mixes:
                tv, tv2 = _rebin(*targets_raw[m]["years"][yr], args.rebin)
                mv, mv2 = _rebin(*models_raw[run][m]["years"][yr], args.rebin)
                ms.append(compute_mix_metrics(tv, tv2, mv, mv2, sig_y))
            per_year[yr][run] = {
                "mean_abs_mu_shape": float(np.mean([x["abs_mu_shape"] for x in ms])),
                "mean_chi2_ndf_shape": float(np.mean([x["chi2_ndf_shape"] for x in ms])),
                "sr_mt_mean": float(np.mean([x["sr_mt"] for x in ms])),
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
    var_rows = []
    for v, vs in prim["variants"].items():
        badge_cls = {"BETTER": "pass", "WORSE": "kill", "INCONCLUSIVE": "warn", "REFERENCE": "ref"}[vs["verdict"]]
        var_rows.append(
            f"<tr>"
            f"<td><b>{html.escape(v)}</b> ({vs['n_seeds']} seeds)</td>"
            f"<td><span class='badge {badge_cls}'>{vs['verdict']}</span></td>"
            f"<td>{vs['sr_mt_mean']:.4f} ± {vs['sr_mt_mix_std_mean']:.4f}</td>"
            f"<td>{vs['shape_abs_mu_mean']:.3f} ± {vs['shape_abs_mu_seed_std']:.3f}</td>"
            f"<td>{vs['shape_abs_mu_ens_mean']:.3f} ± {vs['shape_abs_mu_ens_seed_std']:.3f}</td>"
            f"<td>{vs['shape_chi2_mean']:.2f} ± {vs['shape_chi2_seed_std']:.2f}</td>"
            f"<td>{vs['shape_bias_mean']:.2f} ± {vs['shape_bias_seed_std']:.2f}</td>"
            f"<td>{vs['shape_spread_mean']:.3f} ± {vs['shape_spread_seed_std']:.3f}</td>"
            f"</tr>"
        )

    run_rows = []
    for run in args.runs:
        r = prim["runs"][run]
        run_rows.append(
            f"<tr>"
            f"<td><b>{html.escape(run)}</b></td>"
            f"<td>{r['sb_mt_mean']:.4f} ± {r['sb_mt_std']:.4f}</td>"
            f"<td>{r['sr_mt_mean']:.4f} ± {r['sr_mt_std']:.4f}</td>"
            f"<td>{r['mean_abs_mu_shape']:.3f}</td>"
            f"<td>{r['shape_mu_ens']:+.3f}</td>"
            f"<td>{r['mean_chi2_ndf_shape']:.2f}</td>"
            f"<td>{r['mean_bias_shape_norm']:.2f}</td>"
            f"<td>{r['shape_spread']:.3f}</td>"
            f"<td>{r['mean_abs_mu_raw']:.3f}</td>"
            f"</tr>"
        )

    rebin_sections = []
    for rb in rebins:
        sec = results_by_rebin[str(rb)]
        sub_rows = []
        for v, vs in sec["variants"].items():
            sub_rows.append(
                f"<tr><td><b>{html.escape(v)}</b> ({vs['n_seeds']})</td><td>{vs['verdict']}</td>"
                f"<td>{vs['shape_abs_mu_mean']:.3f} ± {vs['shape_abs_mu_seed_std']:.3f}</td>"
                f"<td>{vs['shape_chi2_mean']:.2f} ± {vs['shape_chi2_seed_std']:.2f}</td>"
                f"<td>{vs['shape_spread_mean']:.3f}</td></tr>"
            )
        rebin_sections.append(
            f"<h3>Rebin {rb} ({sec['n_bins']} bins)</h3>"
            f"<table><tr><th>Variant</th><th>Verdict</th><th>Shape |&mu;| (mean ± seed std)</th>"
            f"<th>Shape &chi;&sup2;/ndf</th><th>Cross-Mix Spread</th></tr>"
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
.warn {{ background: #fef3c7; color: #92400e; }}
.kill {{ background: #fee2e2; color: #991b1b; }}
.ref {{ background: #e0e7ff; color: #1e40af; }}
</style></head><body>
<h1>FvT Optimization Sweep Dashboard (Shape-Only, c0-Marginalized)</h1>
<div class="card">
  <b>Observable:</b> {html.escape(args.hist_key)} | <b>Primary Rebin:</b> {args.rebin} ({prim['n_bins']} bins) |
  <b>Mixes:</b> {', '.join(args.mixes)} | <b>Total Runs:</b> {len(args.runs)}
</div>
<div class="card">
  <h2>1. Multi-Seed Variant Summary (Rebin {args.rebin})</h2>
  <table>
    <tr><th>Variant</th><th>Verdict</th><th>SR M/T (mean ± mix std)</th><th>Shape |&mu;| (mean ± seed std)</th>
    <th>Ensemble Shape |&mu;|</th><th>Shape &chi;&sup2;/ndf</th><th>Shape Bias Norm</th><th>Cross-Mix Spread</th></tr>
    {''.join(var_rows)}
  </table>
</div>
<div class="card">
  <h2>2. Individual Run Breakdown (Rebin {args.rebin})</h2>
  <table>
    <tr><th>Run</th><th>SB M/T</th><th>SR M/T</th><th>Shape |&mu;|</th><th>Ens Shape &mu;</th>
    <th>Shape &chi;&sup2;/ndf</th><th>Shape Bias</th><th>Shape Spread</th><th>Raw |&mu;| (info)</th></tr>
    {''.join(run_rows)}
  </table>
</div>
<div class="card">
  <h2>3. All Rebins (Variant Seed Means)</h2>
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
