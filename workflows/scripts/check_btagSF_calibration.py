"""Checks of the b-tagging SF calibration workflow (Snakefile_btagSF_calibration.smk).

Run on the output of the workflow for a config whose processor section sets
`btagSF_calibration_file` (the CI config, workflows/config/btagSF_calibration_ci.yml), so that
the processor has also applied the installed maps event by event.

  A. workflow outputs: one correctionlib file per year with every sample, ttbar category and
     b-tag SF variation; finite, positive maps; exact closure of the maps on the events they
     were derived from; nothing installed in test mode.
  B. application: the calibration applied in the processor (add_btagweights) is the installed
     map. Njet_HT_btagSF, reweighted bin by bin with the installed file, must reproduce
     Njet_HT_btagSF_calib for every b-tag SF variation. No statistics involved.
  C. closure of the installed maps: after b-tag SF x calibration the Njet and HT distributions
     agree with the ones without the SF. Njet_HT_btagSF_calibDiff holds the difference and its
     statistical uncertainty (the two yields are the same events, fully correlated); a bin
     fails if the difference is both significant (--max-pull) and larger than --rel-floor.

    python coffea4bees/workflows/scripts/check_btagSF_calibration.py \
        -c coffea4bees/workflows/config/btagSF_calibration_ci.yml
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import yaml
from coffea.util import load

from src.physics.btag_calibration import (
    CENTRAL,
    CORRECTION_NAME,
    btag_sf_calibration_samples,
    build_btag_sf_calibration,
    load_btag_sf_calibration,
)
from src.tools.compute_btagSF_calibration import NO_SF, SF, SF_CALIB, CalibrationInput, select

SF_CALIB_DIFF = "btagSF_calibDiff"


class Report:
    """Collects the failures of the checks."""

    def __init__(self):
        self.failures = []

    def check(self, ok: bool, message: str) -> bool:
        if not ok:
            self.failures.append(message)
            logging.error(f"FAIL: {message}")
        return bool(ok)


def map_edges(text: str) -> list[list[float]]:
    """(Njet, HT) bin edges of a calibration file."""
    for corr in json.loads(text)["corrections"]:
        if corr["name"] == CORRECTION_NAME:
            node = corr["data"]
            while node.get("nodetype") == "category":
                node = node["content"][0]["value"]
            return [list(map(float, e)) for e in node["edges"]]
    raise KeyError(f"Correction '{CORRECTION_NAME}' not found")


def evaluate_on_grid(correction, key: str, variation: str, njet_axis, ht_axis) -> np.ndarray:
    """Map of `key` on the (Njet, HT) bins of the histograms."""
    njet, ht = np.meshgrid(njet_axis.centers, ht_axis.centers, indexing="ij")
    return np.asarray(correction.evaluate(key, variation, njet.ravel(), ht.ravel())).reshape(njet.shape)


def components(process: str, datasets: list, categories: list, sample_map: dict) -> list[tuple[str, str]]:
    """[(histogram process, map key)] whose reweighted sum gives the calibrated `process`.

    `process` is a dataset or one of its ttbar categories `<dataset>_<category>`. A dataset
    mapped to {category: key} uses one map per category, so its inclusive histogram is the sum
    of its categories; anything else uses the single key of its dataset.
    """
    dataset, category = process, None
    if process not in datasets:
        for d in datasets:
            for c in categories:
                if process == f"{d}_{c}":
                    dataset, category = d, c
    key = sample_map.get(dataset, dataset)
    if not isinstance(key, dict):
        return [(process, key)]
    if category is not None:
        return [(process, key[category])]
    return [(f"{dataset}_{c}", k) for c, k in key.items()]


def check_outputs(report, inp, cfg, maps_dir, expected, install_dir, tol, closure_tol):
    """A. Files written by the workflow."""
    summary = {}
    for year in cfg["years"]:
        path = os.path.join(maps_dir, f"btagSF_calibrationSF_{year}.json")
        if not report.check(os.path.exists(path), f"[A] {path} not written"):
            continue
        text = load_btag_sf_calibration(path)
        correction = build_btag_sf_calibration(text)
        samples = btag_sf_calibration_samples(text)
        missing = sorted(set(expected) - set(samples))
        report.check(not missing, f"[A] {year}: samples {missing} missing in {path}")
        n_bad = 0
        for sample in samples:
            for variation in inp.variations:
                try:
                    values = evaluate_on_grid(correction, sample, variation, inp.njet_axis, inp.ht_axis)
                except Exception as e:  # missing variation key
                    report.check(False, f"[A] {year} {sample}: variation '{variation}' cannot be evaluated ({e})")
                    continue
                n_bad += int((~np.isfinite(values) | (values <= 0)).sum())
        report.check(n_bad == 0, f"[A] {year}: {n_bad} non-finite or non-positive map values in {path}")
        summary[year] = {"file": path, "samples": len(samples), "variations": len(inp.variations)}

    yields_path = os.path.join(maps_dir, "btagSF_calibration_yields.yml")
    if report.check(os.path.exists(yields_path), f"[A] {yields_path} not written"):
        with open(yields_path) as f:
            yields = yaml.safe_load(f) or {}
        for year in cfg["years"]:
            for process in expected:
                entry = yields.get(year, {}).get(process)
                if not report.check(entry is not None, f"[A] {year} {process}: missing in {yields_path}"):
                    continue
                closure = entry["btagSF_calib_over_noBtagSF"]
                # not exact: a bin with a negative yield (negative generator weights) is given 1
                report.check(abs(closure - 1) < closure_tol,
                             f"[A] {year} {process}: derived maps do not close on their own events (SF x calib / no SF = {closure:.8f})")
                report.check(abs(entry["btagSF_over_noBtagSF"] - 1) > tol,
                             f"[A] {year} {process}: the b-tag SF does not change the yield, was it applied?")
    report.check(not os.path.exists(install_dir), f"[A] {install_dir} exists: test mode must not install the maps")
    return summary


def check_application(report, inp, cfg, expected, tol):
    """B. Processor-applied calibration == installed map applied bin by bin."""
    files = cfg["config"]["btagSF_calibration_file"]
    sample_map = cfg["config"].get("btagSF_calibration_sample_map") or {}
    categories = cfg["config"].get("ttbar_categories") or []
    h_sf, h_calib = inp.h_sf, inp.hists[f"{inp.sf_hist}_{SF_CALIB}"]
    hist_edges = [list(map(float, inp.njet_axis.edges)), list(map(float, inp.ht_axis.edges))]
    summary = {}
    for year in cfg["years"]:
        path = files[year] if isinstance(files, dict) else files.format(year=year)
        text = load_btag_sf_calibration(path)
        if not report.check(map_edges(text) == hist_edges,
                            f"[B] {year}: the binning of {path} differs from the one of the histograms"):
            continue
        correction = build_btag_sf_calibration(text)
        worst = 0.0
        for process in expected:
            for variation in inp.variations:
                applied, _ = select(h_calib, process, year, variation)
                if not report.check(applied is not None, f"[B] {year} {process} {variation}: no {inp.sf_hist}_{SF_CALIB}"):
                    continue
                offline = np.zeros_like(applied)
                for hist_process, key in components(process, cfg["datasets"], categories, sample_map):
                    sf, _ = select(h_sf, hist_process, year, variation)
                    if sf is None:  # category without events
                        continue
                    offline += sf * evaluate_on_grid(correction, key, variation, inp.njet_axis, inp.ht_axis)
                diff = float(np.abs(applied - offline).max() / (np.abs(offline).max() or 1.0))
                worst = max(worst, diff)
                report.check(diff < tol, f"[B] {year} {process} {variation}: processor vs installed map, max rel. diff {diff:.2e}")
        summary[year] = {"file": path, "max_rel_diff": worst}
        logging.info(f"[B] {year}: processor vs {path}, max rel. diff {worst:.2e} ({len(inp.variations)} variations)")
    return summary


def check_closure(report, inp, cfg, expected, max_pull, rel_floor, min_neff):
    """C. Njet and HT distributions with SF x installed calibration vs without the SF."""
    h_diff = inp.hists[f"{inp.sf_hist}_{SF_CALIB_DIFF}"]
    summary = {}
    logging.info(f"[C] {'year':<9}{'process':<24}{'SF/noSF-1':>10}{'calib/noSF-1':>14}{'+-':>9}{'pull':>7}{'max|pull| Njet':>16}{'HT':>7}{'bins':>6}")
    for year in cfg["years"]:
        for process in expected:
            no, no_var = select(inp.h_no, process, year)
            sf, _ = select(inp.h_sf, process, year)
            diff, diff_var = select(h_diff, process, year)
            if not report.check(no is not None and diff is not None and no.sum() > 0, f"[C] {year} {process}: no events"):
                continue
            entry = {"noBtagSF": float(no.sum()), "btagSF_over_noBtagSF_minus1": float(sf.sum() / no.sum() - 1)}

            def pulls(d, v, n, nv):
                """(pull, relative difference, tested) per bin; bins with too few events are not tested."""
                neff = np.divide(n**2, nv, out=np.zeros_like(n), where=nv > 0)
                tested = (neff >= min_neff) & (v > 0)
                pull = np.divide(d, np.sqrt(v), out=np.zeros_like(d), where=tested)
                rel = np.divide(d, n, out=np.zeros_like(d), where=tested)
                return pull, rel, tested

            pull, rel, tested = pulls(*(np.array([x.sum()]) for x in (diff, diff_var, no, no_var)))
            entry["inclusive"] = {"rel_diff": float(rel[0]), "rel_unc": float(np.sqrt(diff_var.sum()) / no.sum()), "pull": float(pull[0])}
            report.check(not (abs(pull[0]) > max_pull and abs(rel[0]) > rel_floor),
                         f"[C] {year} {process}: inclusive non-closure {rel[0]:+.4f} (pull {pull[0]:+.1f})")
            n_tested = 0
            for name, axis, edges in (("Njet", 1, inp.njet_axis.edges), ("HT", 0, inp.ht_axis.edges)):
                pull, rel, tested = pulls(*(x.sum(axis=axis) for x in (diff, diff_var, no, no_var)))
                n_tested += int(tested.sum())
                bad = tested & (np.abs(pull) > max_pull) & (np.abs(rel) > rel_floor)
                for i in np.flatnonzero(bad):
                    report.check(False, f"[C] {year} {process}: {name} bin [{edges[i]:g}, {edges[i + 1]:g}) "
                                        f"non-closure {rel[i]:+.4f} (pull {pull[i]:+.1f})")
                entry[name] = {
                    "pull": [round(float(p), 3) if t else None for p, t in zip(pull, tested)],
                    "rel_diff": [round(float(r), 5) if t else None for r, t in zip(rel, tested)],
                    "max_abs_pull": float(np.abs(pull[tested]).max()) if tested.any() else None,
                }
            report.check(n_tested > 0, f"[C] {year} {process}: no Njet or HT bin with >= {min_neff:g} effective events")
            inc = entry["inclusive"]
            logging.info(f"[C] {year:<9}{process:<24}{entry['btagSF_over_noBtagSF_minus1']:>+10.4f}{inc['rel_diff']:>+14.4f}"
                         f"{inc['rel_unc']:>9.4f}{inc['pull']:>+7.1f}{entry['Njet']['max_abs_pull'] or 0:>16.1f}"
                         f"{entry['HT']['max_abs_pull'] or 0:>7.1f}{n_tested:>6d}")
            summary.setdefault(year, {})[process] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", required=True, help="Workflow config (btagsf_calibration section)")
    parser.add_argument("-i", "--input", help="Merged coffea file [<output_path>/histAll_btagSF_calibration.coffea]")
    parser.add_argument("-m", "--maps", help="Folder with the maps derived by the workflow [<output_path>/calibration_<tag>/]")
    parser.add_argument("-o", "--output", help="Summary yml [<maps>/btagSF_calibration_check.yml]")
    parser.add_argument("--max-pull", type=float, default=4.0, help="[C] largest |non-closure| / stat. uncertainty")
    parser.add_argument("--rel-floor", type=float, default=0.005, help="[C] relative non-closure below which a bin never fails")
    parser.add_argument("--min-neff", type=float, default=20.0, help="[C] bins with fewer effective MC events are not tested")
    parser.add_argument("--exact-tol", type=float, default=1e-5, help="[A, B] tolerance of the exact comparisons")
    parser.add_argument("--self-closure-tol", type=float, default=1e-3,
                        help="[A] tolerance of the closure of the derived maps on their own events")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg["btagsf_calibration"]
    cfg["years"] = [str(y) for y in cfg["years"]]
    output_path = full_cfg.get("output_path", "output/btagSF_calibration/").rstrip("/") + "/"
    tag = str(cfg.get("tag", "v1"))
    hists_path = args.input or f"{output_path}histAll_btagSF_calibration.coffea"
    maps_dir = args.maps or f"{output_path}calibration_{tag}/"
    install_dir = cfg.get("install_dir", f"coffea4bees/metadata/weights/btagSF_calibration/{tag}/")
    if not cfg.get("config", {}).get("btagSF_calibration_file"):
        parser.error(f"{args.config} does not set btagsf_calibration.config.btagSF_calibration_file")

    inp = CalibrationInput(load(hists_path)["hists"], cfg.get("sf_hist", "Njet_HT"))
    categories = cfg["config"].get("ttbar_categories") or []
    expected = []
    for dataset in cfg["datasets"]:
        expected.append(dataset)
        if dataset.startswith("TT"):
            expected += [f"{dataset}_{c}" for c in categories]

    report = Report()
    for name in (f"{inp.sf_hist}_{SF_CALIB}", f"{inp.sf_hist}_{SF_CALIB_DIFF}"):
        report.check(name in inp.hists, f"{name} not in {hists_path}: the processor did not apply the calibration")
    for year in cfg["years"]:
        missing = [p for p in expected if select(inp.h_no, p, year)[0] is None]
        report.check(not missing, f"{year}: no {inp.sf_hist}_{NO_SF} / {SF} histograms for {missing}")

    summary = {"outputs": check_outputs(report, inp, cfg, maps_dir, expected, install_dir, args.exact_tol, args.self_closure_tol)}
    if f"{inp.sf_hist}_{SF_CALIB}" in inp.hists:
        summary["application"] = check_application(report, inp, cfg, expected, args.exact_tol)
    if f"{inp.sf_hist}_{SF_CALIB_DIFF}" in inp.hists:
        summary["closure"] = check_closure(report, inp, cfg, expected, args.max_pull, args.rel_floor, args.min_neff)
    summary["thresholds"] = {k: getattr(args, k) for k in ("max_pull", "rel_floor", "min_neff", "exact_tol", "self_closure_tol")}
    summary["failures"] = report.failures

    out = args.output or os.path.join(maps_dir, "btagSF_calibration_check.yml")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    logging.info(f"Wrote {out}")
    if report.failures:
        logging.error(f"{len(report.failures)} check(s) failed")
        sys.exit(1)
    logging.info("All b-tag SF calibration checks passed")


if __name__ == "__main__":
    main()
