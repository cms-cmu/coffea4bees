"""Helpers for the trigger-weight validation (Snakefile_PhaseA_3_trigValidation.smk).

rename  rewrite a processor_HH4b output so the signal shows up as one variant: every histogram's
        process-axis categories in --processes become --new-name, and the cutflow dataset keys
        <process>_<year> become <new-name>_<year>. Three renamed outputs (noTrig, HLT, HLT_SF) can
        then be merged into one file and overlaid by makePlots.
report  four-tag yields per variant, era and cut from the cutflow, with HLT/noTrig (the MC
        trigger efficiency), HLT_SF/noTrig and HLT_SF/HLT (the mean trigger SF).
"""
import argparse
import logging
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)

import hist
import yaml
from coffea.util import load, save

CUTS = ["passJetMult", "passPreSel", "passDiJetMass", "SR", "SB"]


def _rename_hist(h, mapping):
    names = [ax.name for ax in h.axes]
    if "process" not in names:
        return h
    pax = h.axes["process"]
    if not isinstance(pax, hist.axis.StrCategory):
        return h
    cats = [mapping.get(c, c) for c in pax]
    if len(set(cats)) != len(cats):
        raise ValueError(f"renaming {list(pax)} -> {cats} merges categories")
    new_ax = hist.axis.StrCategory(cats, name=pax.name, label=pax.label,
                                   growth=pax.traits.growth, overflow=pax.traits.overflow)
    out = hist.Hist(*[new_ax if ax.name == "process" else ax for ax in h.axes],
                    storage=h.storage_type())
    out.view(flow=True)[...] = h.view(flow=True)
    return out


def _walk(obj, mapping):
    if isinstance(obj, hist.Hist):
        return _rename_hist(obj, mapping)
    if isinstance(obj, dict):
        return {k: _walk(v, mapping) for k, v in obj.items()}
    return obj


def _rename_key(key, processes, new_name):
    for p in processes:
        if key == p or key.startswith(p + "_"):
            return new_name + key[len(p):]
    return key


def cmd_rename(args):
    out = load(args.input)
    mapping = {p: args.new_name for p in args.processes}
    n_hists = 0
    for key, val in list(out.items()):
        if key.startswith("cutFlow") or key == "cutflow_hists":
            out[key] = {_rename_key(k, args.processes, args.new_name): v for k, v in val.items()}
        elif key.startswith("hists") or isinstance(val, (dict, hist.Hist)):
            out[key] = _walk(val, mapping)
            n_hists += sum(1 for v in (val.values() if isinstance(val, dict) else [val])
                           if isinstance(v, hist.Hist))
    save(out, args.output)
    logging.info(f"{args.input} -> {args.output}: {args.processes} -> {args.new_name} ({n_hists} top-level hists)")


def _yields(out, variant, year):
    flows = out.get("cutFlowFourTag", {})
    key = f"HH4b_{variant}_{year}"
    if key not in flows:
        raise KeyError(f"no cutFlowFourTag entry {key}; have {sorted(flows)}")
    return {c: float(flows[key].get(c, float("nan"))) for c in CUTS}


def cmd_report(args):
    out = load(args.input)
    rows = {}
    for year in args.years:
        rows[year] = {v: _yields(out, v, year) for v in args.variants}
    rows["Run3"] = {v: {c: sum(rows[y][v][c] for y in args.years) for c in CUTS} for v in args.variants}

    def ratio(a, b):
        return a / b if b else float("nan")

    result, lines, flags = {}, [], []
    hdr = f"{'era':<13}{'cut':<15}" + "".join(f"{v:>12}" for v in args.variants) + \
          f"{'HLT/noTrig':>12}{'SF/noTrig':>12}{'mean SF':>10}"
    lines += ["Four-tag weighted yields (cutFlowFourTag) and ratios", hdr, "-" * len(hdr)]
    for era, by_var in rows.items():
        result[era] = {}
        for c in CUTS:
            y = {v: by_var[v][c] for v in args.variants}
            eff = ratio(y.get("HLT", 0), y.get("noTrig", 0))
            sf_eff = ratio(y.get("HLT_SF", 0), y.get("noTrig", 0))
            sf = ratio(y.get("HLT_SF", 0), y.get("HLT", 0))
            result[era][c] = {**y, "HLT_over_noTrig": eff, "HLT_SF_over_noTrig": sf_eff, "mean_SF": sf}
            lines.append(f"{era:<13}{c:<15}" + "".join(f"{y[v]:>12.4g}" for v in args.variants) +
                         f"{eff:>12.4f}{sf_eff:>12.4f}{sf:>10.4f}")
            if c == "passPreSel" and era != "Run3" and eff > 0.999:
                flags.append(f"{era}: HLT/noTrig = {eff:.4f} at passPreSel -- the picoAODs look HLT-filtered "
                             f"or carry no HLT branches (passHLT defaults to True), so noTrig is not a no-trigger sample")
        lines.append("")
    if flags:
        lines += ["WARNINGS"] + [f"  {f}" for f in flags]
    text = "\n".join(lines) + "\n"
    print(text)
    Path(f"{args.output}.txt").write_text(text)
    with open(f"{args.output}.yml", "w") as f:
        yaml.dump({"yields": result, "warnings": flags}, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rename")
    r.add_argument("-i", "--input", required=True)
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--processes", nargs="+", required=True)
    r.add_argument("--new-name", required=True)
    p = sub.add_parser("report")
    p.add_argument("-i", "--input", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--years", nargs="+", required=True)
    p.add_argument("-o", "--output", required=True, help="output path stem (.txt and .yml are written)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    {"rename": cmd_rename, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
