"""DeClustered-data validation report (Snakefile_DeClustered_5_monitoring.smk, step D.5).

    declustered_validation_report.py cutflow CUTFLOW.yml OUTDIR --prefix syn --seed K --n-seeds N
                                             [--psdata ttbar_PSData | --ttbar-in-sample]
    declustered_validation_report.py pdfs    OUTDIR ERA_DIR [ERA_DIR ...]

cutflow: four-tag cutflow page: declustered data (seed K, and the mean over all N seeds) next to the
         4b data it was made from, plus ttbar MC. With --ttbar-in-sample (no ttbar subtraction in the
         declustering) the model is the declustered sample alone; otherwise declustered + ttbar MC,
         and --psdata adds the ttbar pseudodata next to the ttbar MC.
pdfs:    index page of D.2's sampling tests (make_jet_splitting_PDFs.py test_sampling_* plots,
         splitting multiplicities), one section per era.
"""
import argparse
import glob
import html
import os

import numpy as np
import yaml

YEARS = ["2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix",
         "UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]

STYLE = ("<style>body{font-family:sans-serif} table{border-collapse:collapse;margin-bottom:1.5em}"
         "td,th{border:1px solid #ccc;padding:2px 8px;text-align:right} th{background:#eee}"
         "td:first-child{text-align:left} li{margin:2px 0}</style>")


def _year(key):
    for y in sorted(YEARS, key=len, reverse=True):
        if f"_{y}" in key:
            return y
    return None


def _fmt(x):
    return f"{x:,.0f}"


def _ratio(n, d):
    if d <= 0:
        return "-"
    err = np.sqrt(n) / d if n > 0 else 0.0
    return f"{n / d:.3f} ± {err:.3f}"


def cutflow(path, outdir, prefix, k, n_seeds, ttbar_in_sample, psdata=None):
    os.makedirs(outdir, exist_ok=True)
    with open(path) as f:
        counts = yaml.safe_load(f)["counts4"]

    def group(key):
        if psdata and key.startswith(f"{psdata}_"):
            return "psdata"
        if key.startswith("data_"):
            return "data"
        if key.startswith("TTTo"):
            return "ttbar_mc"
        for s in range(n_seeds):
            if key.startswith(f"{prefix}_v{s}_"):
                return ("seed", s)
        return None

    cuts = []
    for v in counts.values():
        for c in (v or {}):
            if c not in cuts:
                cuts.append(c)
    groups, unknown = {}, set()          # (year|"all", group) -> {cut: count}
    for key, v in counts.items():
        g = group(key)
        if g is None:
            unknown.add(key)
            continue
        for y in ("all", _year(key) or "?"):
            d = groups.setdefault((y, g), {})
            for c, n in (v or {}).items():
                d[c] = d.get(c, 0.0) + float(n or 0)
    missing = [s for s in range(n_seeds) if ("all", ("seed", s)) not in groups]
    years = ["all"] + [y for y in YEARS if any(g[0] == y for g in groups)]
    model = "declustered" if ttbar_in_sample else "declustered + ttbar MC"
    head = ["cut", "data 4b", f"seed {k}", "ttbar MC", f"model (seed {k})", "data / model",
            f"seed mean (N={n_seeds})", "data / mean model", "seed rms / mean"]
    if psdata:
        head += ["ttbar pseudodata", "PSdata / ttbar MC"]
    txt, blocks = [], []
    for y in years:
        rows = []
        for c in cuts:
            da = groups.get((y, "data"), {}).get(c, 0.0)
            tt = groups.get((y, "ttbar_mc"), {}).get(c, 0.0)
            seeds = np.array([groups.get((y, ("seed", s)), {}).get(c, 0.0) for s in range(n_seeds)])
            sk = groups.get((y, ("seed", k)), {}).get(c, 0.0)
            add = 0.0 if ttbar_in_sample else tt
            mean = float(seeds.mean()) if len(seeds) else 0.0
            rms = f"{seeds.std() / mean:.4f}" if len(seeds) > 1 and mean > 0 else "-"
            row = [c, _fmt(da), _fmt(sk), _fmt(tt), _fmt(sk + add), _ratio(da, sk + add),
                   _fmt(mean), _ratio(da, mean + add), rms]
            if psdata:
                ps = groups.get((y, "psdata"), {}).get(c, 0.0)
                row += [_fmt(ps), _ratio(ps, tt)]
            rows.append(row)
        widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
        title = "all years" if y == "all" else y
        txt.append(title)
        txt.append("  ".join(h.ljust(w) for h, w in zip(head, widths)))
        txt += ["  ".join(str(x).ljust(w) for x, w in zip(r, widths)) for r in rows]
        txt.append("")
        th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
        tr = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in rows)
        blocks.append(f"<h2>{title}</h2><table><tr>{th}</tr>{tr}</table>")
    note = (f"<p>Four-tag counts (weighted). Model = {model}"
            + (" (the declustering did not subtract ttbar, so the declustered sample already models it; "
               "ttbar MC is shown for reference only)" if ttbar_in_sample else "")
            + (f". The ttbar pseudodata ({psdata}), folded into the consumer dataset, should follow the ttbar MC"
               if psdata else "")
            + f". Declustered samples: <code>{html.escape(prefix)}_v&lt;seed&gt;</code>; data / model "
              f"should be consistent with 1 at every cut.</p>")
    if missing:
        note += f"<p><b>No cutflow entries for seeds {missing}.</b></p>"
    if unknown:
        note += f"<p>Not grouped: {html.escape(', '.join(sorted(unknown)[:10]))}</p>"
    with open(os.path.join(outdir, "cutflow_monitoring.html"), "w") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>declustered-data validation cutflow</title>{STYLE}"
                f"</head><body><h1>DeClustered-data validation cutflow</h1>{note}{''.join(blocks)}</body></html>")
    with open(os.path.join(outdir, "cutflow_monitoring.txt"), "w") as f:
        f.write("\n".join(txt))
    print("\n".join(txt[:len(cuts) + 3]))
    if missing:
        raise SystemExit(f"no cutflow entries for seeds {missing} in {path}")


def pdfs(outdir, era_dirs):
    os.makedirs(outdir, exist_ok=True)
    sections = []
    for d in era_dirs:
        era = os.path.basename(os.path.normpath(d))
        files = sorted(glob.glob(os.path.join(d, "*")))
        items = "".join(f"<li><a href='{html.escape(os.path.relpath(p, outdir))}'>{html.escape(os.path.basename(p))}</a></li>"
                        for p in files)
        sections.append(f"<h2>{html.escape(era)}</h2><ul>{items}</ul>")
    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>declustering PDFs</title>{STYLE}</head><body>"
                f"<h1>Declustering PDFs and sampling tests (D.2)</h1>"
                f"<p>Per era: the PDF YAML the DeClusterer samples from, the sampling test plots "
                f"(test_sampling_pt_*: sampled vs input splitting variables) and the splitting "
                f"multiplicities.</p>{''.join(sections)}</body></html>")
    print(f"wrote {os.path.join(outdir, 'index.html')} ({len(era_dirs)} eras)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cutflow")
    c.add_argument("input"); c.add_argument("outdir")
    c.add_argument("--prefix", required=True, help="sample prefix, syn_noTT or syn")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--n-seeds", type=int, default=1)
    c.add_argument("--ttbar-in-sample", action="store_true")
    c.add_argument("--psdata", default=None, help="ttbar pseudodata process name (e.g. ttbar_PSData)")
    p = sub.add_parser("pdfs")
    p.add_argument("outdir"); p.add_argument("era_dirs", nargs="+")
    a = ap.parse_args()
    if a.mode == "cutflow":
        cutflow(a.input, a.outdir, a.prefix, a.seed, a.n_seeds, a.ttbar_in_sample, a.psdata)
    else:
        pdfs(a.outdir, a.era_dirs)


if __name__ == "__main__":
    main()
