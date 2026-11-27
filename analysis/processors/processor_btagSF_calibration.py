"""Histograms for the b-tagging SF calibration.

Runs on MC NanoAOD (the picoAODs already contain a b-tag requirement) and fills, for events with
>= 4 selected jets and NO b-tag requirement, the same distributions with and without the b-tag
shape SF:

    Njet_HT_noBtagSF, Njet_HT_btagSF (one entry per b-tag SF variation), Njet_HT_btagSFcross
    <var>_noBtagSF, <var>_btagSF      diagnostic variables, binned also in (Njet, HT)

src/tools/compute_btagSF_calibration.py turns them into the (Njet, HT) calibration maps and into
the closure plots. If `btagSF_calibration_file` is given, the calibration is also applied event
by event with the analysis code (add_btagweights) and `*_btagSF_calib` histograms are filled, to
validate the application, together with `Njet_HT_btagSF_calibDiff` (weight w * (SF x calib - 1)):
the non-closure and its statistical uncertainty (workflows/scripts/check_btagSF_calibration.py).

No trigger selection or trigger weight is applied: the 4b triggers contain online b-tagging.
"""

import logging
import warnings

import awkward as ak
import numpy as np
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
from coffea.nanoevents import NanoAODSchema

from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.event_weights import add_btagweights
from coffea4bees.analysis.helpers.object_selection import apply_jet_calibration, load_object_selection_config
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.ttbar_categories import ttbar_category_mask
from src.hist_tools import Collection, Fill
from src.physics.btag_calibration import CENTRAL, BtagSFCalibration
from src.physics.event_selection import apply_event_selection
from src.physics.event_weights import add_weights

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

# Years without a b-tag shape SF (see HH4bBaseProcessor.apply_btag_sf)
YEARS_WITHOUT_SHAPE_SF = ("2024", "2025")

N_LEADING_JETS = 4

# Events with |w - mean| / std above this, within a chunk, are removed (as in the analysis)
GENWEIGHT_MAX_ZSCORE = 30


def variation_key(modifier: str) -> str:
    """coffea weight modifier (CMS_btag_hfUp) -> b-tag SF variation key (up_hf)."""
    name = modifier.replace("CMS_btag_", "", 1)
    for suffix, direction in (("Up", "up"), ("Down", "down")):
        if name.endswith(suffix):
            return f"{direction}_{name[: -len(suffix)]}"
    raise ValueError(f"Unexpected b-tag SF weight variation '{modifier}'")


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        corrections_metadata: dict = None,
        object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
        njet_bins: list = (4, 5, 6, 7, 8, 9, 11, 20),
        ht_bins: list = (0, 500, 650, 800, 1000, 1200, 1400, 1600, 1800, 2000, 5000),
        btagSF_calibration_file: str | dict | None = None,
        btagSF_calibration_sample_map: dict | None = None,
        ttbar_categories: list = ("ttB", "ttC", "ttLF"),
        **kwargs,
    ):
        self.corrections_metadata = corrections_metadata
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None
        self.njet_bins = [float(x) for x in njet_bins]
        self.ht_bins = [float(x) for x in ht_bins]
        # ttbar samples are also filled as "<process>_<category>" (categories of
        # ttbar_categories.py, from genTtbarId): the effect of the b-tag SF depends on the flavour
        # of the additional jets, so ttB, ttC and ttLF are calibrated as separate samples, each
        # with its own map. The stitched ttbar samples take the ttB events from the TTbb samples
        # and ttC, ttLF from the inclusive ones.
        self.ttbar_categories = list(ttbar_categories or [])
        self.btagSF_calibration = (
            BtagSFCalibration(btagSF_calibration_file, btagSF_calibration_sample_map)
            if btagSF_calibration_file else None
        )

    def process(self, event):
        dataset = event.metadata["dataset"]
        year = event.metadata["year"]
        processName = event.metadata["processName"]
        year_meta = self.corrections_metadata[year]
        config = processor_config(processName, dataset, event)
        nEvent = len(event)

        if not config["isMC"]:
            raise ValueError(f"The b-tag SF calibration is derived on MC only (got dataset {dataset})")
        if year in YEARS_WITHOUT_SHAPE_SF:
            logging.warning(f"No b-tag shape SF for {year}: nothing to calibrate for {dataset}")
            return {"nEvent": {dataset: nEvent}}

        #
        # Same objects as in the analysis
        #
        event = apply_event_selection(event, year_meta, cut_on_lumimask=config["cut_on_lumimask"])
        event["Jet"] = apply_jet_calibration(
            event, corrections_metadata=year_meta, isMC=True, run_systematics=False, dataset=dataset
        )
        event = apply_4b_selection(event, year_meta, config=config, dataset=dataset, sel_cfg=self.sel_cfg)

        #
        # Weights: generator, pileup, prefiring, then the b-tag SF with all its variations
        #
        weights, list_weight_names = add_weights(
            event,
            do_MC_weights=True,
            dataset=dataset,
            year_label=year_meta["year_label"],
            corrections_metadata=year_meta,
            run_systematics=False,
        )
        weight_noBtagSF = weights.weight()

        btag_args = dict(
            shift_name=None,
            run_systematics=True,
            corrections_metadata=year_meta,
            isRun3=config["isRun3"],
        )
        weights, _ = add_btagweights(event, weights, list_weight_names=list(list_weight_names), **btag_args)
        btag_modifiers = [v for v in weights.variations if v.startswith("CMS_btag_")]
        variations = {CENTRAL: None} | {variation_key(m): m for m in btag_modifiers}

        # Calibration applied with the analysis code, for the validation of the application
        weights_calib = None
        if self.btagSF_calibration is not None:
            correction, sample = self.btagSF_calibration.get(year, processName)
            if isinstance(sample, dict):
                sample = {k: ttbar_category_mask(event.genTtbarId, cat) for cat, k in sample.items()}
            weights_calib = Weights(len(event), storeIndividual=True)
            weights_calib.add("noBtagSF", weight_noBtagSF)
            weights_calib, _ = add_btagweights(
                event, weights_calib, list_weight_names=[], btag_calibration=correction,
                btag_calibration_sample=sample, **btag_args,
            )

        #
        # Selection: >= 4 selected jets, no b-tag requirement
        #
        selections = PackedSelection()
        selections.add("passNoiseFilter", event.passNoiseFilter)
        selections.add("passJetMult", event.passJetMult)
        # Same generator-weight outlier veto as the analysis (HH4bBaseProcessor.add_genweight_check):
        # a few events with huge weights (e.g. in the Run 2 ggF HH4b samples) would otherwise
        # dominate the maps, while the analysis never sees them.
        w_std = np.std(weight_noBtagSF)
        pass_clean = (
            np.abs(weight_noBtagSF - np.mean(weight_noBtagSF)) / w_std < GENWEIGHT_MAX_ZSCORE
            if w_std > 0 else np.full(len(event), True)
        )
        if not np.all(pass_clean):
            logging.warning(f"{dataset}: {int(np.sum(~pass_clean))} event(s) removed by the generator-weight outlier veto")
        selections.add("passCleanGenWeight", pass_clean)
        sel = selections.all("passNoiseFilter", "passJetMult", "passCleanGenWeight")
        selev = event[sel]

        jets = selev.selJet_no_bRegCorr  # the jets entering the b-tag SF
        selev["calib_nJet"] = ak.num(jets)
        selev["calib_HT"] = ak.sum(jets.pt, axis=1)
        # true light jets (hadronFlavour 0: u, d, s, gluon or unmatched), the jets that get the
        # light-flavour b-tag SF
        selev["nJet_light_true"] = ak.sum(jets.hadronFlavour == 0, axis=1)
        for i in range(N_LEADING_JETS):
            selev[f"jet{i}_pt"] = jets.pt[:, i]
            selev[f"jet{i}_eta"] = jets.eta[:, i]
            selev[f"jet{i}_btagScore"] = jets.btagScore[:, i]

        sf_central = weights.partial_weight(include=["CMS_btag"])[sel]
        selev["weight_noBtagSF"] = weight_noBtagSF[sel]
        # variances of this histogram give sum(w^2 SF): correlation term of the ratio uncertainty
        selev["weight_btagSFcross"] = weight_noBtagSF[sel] * np.sqrt(np.clip(sf_central, 0, None))

        #
        # Histograms
        #
        # (process label, events): the full sample, plus its ttbar categories
        subsets = {processName: slice(None)}
        if processName.startswith("TT") and "genTtbarId" in selev.fields:
            for category in self.ttbar_categories:
                subsets[f"{processName}_{category}"] = np.asarray(ak.to_numpy(ttbar_category_mask(selev.genTtbarId, category)))

        hist = Collection(process=list(subsets), year=[year], variation=list(variations))
        base = Fill(year=year)

        njet_axis = (self.njet_bins, ("calib_nJet", "Jet multiplicity"))
        ht_axis = (self.ht_bins, ("calib_HT", "H_{T} [GeV]"))
        diagnostics = {
            "nJet_selected": (12, 3.5, 15.5, ("nJet_selected", "Number of selected jets")),
            "hT_selected": (50, 0, 2500, ("hT_selected", "H_{T} (selected jets) [GeV]")),
            "nJet_tagged": (9, -0.5, 8.5, ("nJet_tagged", "Number of b-tagged jets")),
            "nJet_tagged_loose": (9, -0.5, 8.5, ("nJet_tagged_loose", "Number of loose b-tagged jets")),
            "nJet_light_true": (13, -0.5, 12.5, ("nJet_light_true", "Number of true light jets (hadronFlavour 0)")),
        }
        for i in range(N_LEADING_JETS):
            diagnostics[f"jet{i}_pt"] = (50, 0, 500, (f"jet{i}_pt", f"Jet {i} p_{{T}} [GeV]"))
            diagnostics[f"jet{i}_eta"] = (25, -2.5, 2.5, (f"jet{i}_eta", f"Jet {i} #eta"))
            diagnostics[f"jet{i}_btagScore"] = (25, 0, 1, (f"jet{i}_btagScore", f"Jet {i} b-tag score"))

        def book(suffix, with_diagnostics=True):
            """Fill for the map (and diagnostic) histograms using the weight field `weight_<suffix>`."""
            fill = base + hist.add(f"Njet_HT_{suffix}", njet_axis, ht_axis, weight=f"weight_{suffix}")
            if with_diagnostics:
                for var, axis in diagnostics.items():
                    fill += hist.add(f"{var}_{suffix}", axis, njet_axis, ht_axis, weight=f"weight_{suffix}")
            return fill

        fill_no = book("noBtagSF")
        fill_cross = book("btagSFcross", with_diagnostics=False)
        fill_sf = book("btagSF")  # all variations
        # the diagnostics are filled for the central SF only
        fill_sf_map = Fill({"Njet_HT_btagSF": fill_sf._fills["Njet_HT_btagSF"]}, **fill_sf._kwargs)

        def fill_all(fill, variation):
            for process, subset in subsets.items():
                fill(selev[subset], hist, process=process, variation=variation)

        fill_all(fill_no, CENTRAL)
        fill_all(fill_cross, CENTRAL)
        for key, modifier in variations.items():
            selev["weight_btagSF"] = weights.weight(modifier=modifier)[sel]
            fill_all(fill_sf if key == CENTRAL else fill_sf_map, key)

        if weights_calib is not None:
            fill_calib = book("btagSF_calib")
            fill_calib_map = Fill({"Njet_HT_btagSF_calib": fill_calib._fills["Njet_HT_btagSF_calib"]}, **fill_calib._kwargs)
            for key, modifier in variations.items():
                selev["weight_btagSF_calib"] = weights_calib.weight(modifier=modifier)[sel]
                fill_all(fill_calib if key == CENTRAL else fill_calib_map, key)
            # Non-closure w * (SF x calibration - 1): its variances are sum(w^2 (SF x calib - 1)^2),
            # the statistical uncertainty of the difference to the no-SF yield, with the full
            # correlation between the two (same events, different weights).
            selev["weight_btagSF_calibDiff"] = weights_calib.weight()[sel] - selev.weight_noBtagSF
            fill_all(book("btagSF_calibDiff", with_diagnostics=False), CENTRAL)

        output = hist.to_dict(nonempty=True)
        output["nEvent"] = {dataset: nEvent}
        output["nGenWeightOutliers"] = {dataset: int(np.sum(~pass_clean))}
        output["sumw"] = {
            dataset: {
                "noBtagSF": float(ak.sum(selev.weight_noBtagSF)),
                "btagSF": float(np.sum(weights.weight()[sel])),
            }
        }
        return output

    def postprocess(self, accumulator):
        return accumulator
