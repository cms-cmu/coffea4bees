"""DeClustered-data validation pages (Snakefile_DeClustered_5_monitoring.smk, step D.5).

    declustered_validation_report.py pdfs    OUTDIR ERA_DIR [ERA_DIR ...]

(The D.5 cutflow page is the shared src/tools/cutflow_closure.py, --multijet-process.)

pdfs:    index page of D.2's sampling tests (make_jet_splitting_PDFs.py test_sampling_* plots,
         splitting multiplicities), one section per era.
"""
import argparse
import glob
import html
import os

STYLE = ("<style>body{font-family:sans-serif} table{border-collapse:collapse;margin-bottom:1.5em}"
         "td,th{border:1px solid #ccc;padding:2px 8px;text-align:right} th{background:#eee}"
         "td:first-child{text-align:left} li{margin:2px 0}</style>")


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
    p = sub.add_parser("pdfs")
    p.add_argument("outdir"); p.add_argument("era_dirs", nargs="+")
    a = ap.parse_args()
    pdfs(a.outdir, a.era_dirs)


if __name__ == "__main__":
    main()
