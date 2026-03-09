"""
Processor for HH→4b signal dropout study on Run 3 raw NanoAOD.

Compares generator-level truth quantities against reconstructed-level cuts
to determine why signal events fail Run 3 selections.

Failure categories (sequential):
  1. Failed Trigger (but passed lumimask + noise filter)
  2. Failed Preselection (passed trigger, failed jet mult / tagging)
  3. Failed Signal Region (passed presel, failed SR criteria)
  4. Passed SR (signal acceptance)

Studies implemented:
  - Acceptance:       gen-b matching efficiency vs pT, eta; dR(gen-b, reco jet)
  - Collimation:      overlap-region bb pairs (0.4 < dR < 0.8) reconstruction fate
  - Low-pT intruders: pT-rank of matched signal jets; btag score comparison
  - Trigger/kine:     gen-level HT, 4th-leading gen-b pT
  - Boosted matching:  dR(GenHiggs, FatJet) < 0.8 geometric matching
  - FSR recovery:     all studies split with/without FSR recovery
"""

from __future__ import annotations

import logging
import warnings

import awkward as ak
import numpy as np

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from src.hist_tools import Collection, Fill
from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.truth_tools import find_genpart
from coffea4bees.analysis.helpers.processor_config import processor_config

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


def _safe_flat(arr):
    """Convert a potentially option-typed awkward array to a plain numpy array.

    Handles the tricky case where awkward v1.x `ak.fill_none` / `ak.where`
    leave residual option-type wrappers that cause `ak.to_numpy` to choke.
    Falls back to `ak.to_list()` → numpy conversion when direct conversion fails.
    """
    flat = ak.flatten(arr)
    try:
        return np.asarray(ak.to_numpy(flat))
    except ValueError:
        return np.array([v for v in ak.to_list(flat) if v is not None])


# ---------------------------------------------------------------------------
# Gen-truth matching helpers
# ---------------------------------------------------------------------------

def build_gen_truth(event):
    """Build all generator-level truth objects on `event`.

    Adds to event:
        genb_all         : all b-quarks from H/Z decays (all copies)
        genb              : status==23 (hard-process) b-quarks from H decays
        genH              : the two gen Higgs bosons (pdgId 25, isLastCopy)
        genb_higgs_idx    : per-b index (0 or 1) indicating which Higgs parent
        genb_pair_dR      : dR between the two b-quarks from each Higgs (len=2 per evt)
    """
    # --- b-quarks from Higgs (or Z for VH backgrounds) ---
    event["genb_all"] = find_genpart(event.GenPart, [5], [25])

    # Hard-process b-quarks (status 23 = outgoing from hard interaction)
    if "status" in event.genb_all.fields:
        event["genb"] = event.genb_all[event.genb_all.status == 23]
    else:
        # Fallback: use isLastCopy (after FSR) if status is missing
        logging.warning("GenPart.status missing — using isLastCopy for gen b-quarks")
        event["genb"] = event.genb_all[event.genb_all.hasFlags(["isLastCopy"])]

    # --- Gen Higgs bosons (pdgId=25, isLastCopy) ---
    higgs_mask = (np.abs(event.GenPart.pdgId) == 25) & event.GenPart.hasFlags(["isLastCopy"])
    event["genH"] = event.GenPart[higgs_mask]

    # --- Assign each b-quark to its parent Higgs (index 0 or 1) ---
    # In HH→4b, each Higgs decays to 2 b-quarks, so nearest-dR reliably
    # assigns parentage.
    genH_pad = ak.pad_none(event.genH, 2)
    dR_to_H0 = event.genb.delta_r(genH_pad[:, 0])
    dR_to_H1 = event.genb.delta_r(genH_pad[:, 1])
    event["genb_higgs_idx"] = ak.where(dR_to_H0 <= dR_to_H1, 0, 1)

    # --- dR(b,b) for each Higgs ---
    # Group b-quarks by Higgs parent and compute intra-pair dR
    b_from_H0 = event.genb[event.genb_higgs_idx == 0]
    b_from_H1 = event.genb[event.genb_higgs_idx == 1]
    # Pad to exactly 2 per Higgs (should already be 2 in signal)
    b_H0_pad = ak.pad_none(b_from_H0, 2)
    b_H1_pad = ak.pad_none(b_from_H1, 2)
    dR_H0 = ak.fill_none(b_H0_pad[:, 0].delta_r(b_H0_pad[:, 1]), -1.0)
    dR_H1 = ak.fill_none(b_H1_pad[:, 0].delta_r(b_H1_pad[:, 1]), -1.0)
    # Stack as (events, 2) — one dR per Higgs
    event["genb_pair_dR"] = ak.concatenate(
        [dR_H0[:, np.newaxis], dR_H1[:, np.newaxis]], axis=1
    )

    return event


def match_genb_to_reco_ak4(event, dR_threshold=0.4):
    """Match gen-b quarks to reco AK4 jets and compute per-b matching info.

    Uses coffea's `.nearest()` for dR matching.

    Adds to event:
        genb_matched_jet  : nearest reco Jet for each gen-b (None if no match)
        genb_is_matched   : bool per gen-b, True if a reco jet is within dR_threshold
        genb_match_dR     : dR to nearest reco jet (filled with 999 if unmatched)
        genb_match_ptrank : pT-rank (0-based) of the matched reco jet among all
                            selected jets (filled with -1 if unmatched)
    """
    # Use all jets (not just selected) for matching — we want to see if a
    # gen-b has *any* reco counterpart, even outside the selection pT/eta cuts
    reco_jets = event.Jet

    nearest_jet = event.genb.nearest(reco_jets, threshold=dR_threshold)
    is_matched = ~ak.is_none(nearest_jet, axis=1)

    # dR to the nearest matched jet
    # fill_none with self (gen-b) so delta_r gives 0 for unmatched entries,
    # then overwrite unmatched with sentinel
    match_dR = event.genb.delta_r(ak.fill_none(nearest_jet, event.genb))
    match_dR = ak.where(is_matched, match_dR, 999.0)

    # pT-rank of matched jet:  argsort jets by pT descending, find position
    jet_pt_sorted_idx = ak.argsort(reco_jets.pt, axis=1, ascending=False)
    # Build a rank array: rank[i] = position in pT-sorted list for jet index i
    # This is the inverse permutation of argsort
    jet_rank = ak.argsort(jet_pt_sorted_idx, axis=1)

    # For each matched jet, look up its index in the original Jet collection
    # nearest() returns the actual jet object — we need its index
    # Use a workaround: match by comparing pt + eta + phi (exact float match)
    # A cleaner approach: use the jetIdx from nearest
    # Coffea's nearest returns objects with the same fields, but no explicit index.
    # We compute rank by finding the nearest jet's pT rank via broadcast.
    matched_pt = ak.fill_none(nearest_jet.pt, -1.0)
    # Compare each matched_pt against the reco jets to find the index
    # For efficiency, use argmin of |pt - matched_pt| (exact match expected)
    # This is event × genb × jet, so we broadcast carefully
    reco_pt = reco_jets.pt  # events × jets

    # Build cross-product: for each (event, genb), find which jet index has this pt
    # Use nested broadcasting
    matched_pt_bc = matched_pt[:, :, np.newaxis]        # events × genb × 1
    reco_pt_bc = reco_pt[:, np.newaxis, :]              # events × 1 × jets
    pt_diff = np.abs(matched_pt_bc - reco_pt_bc)
    matched_jet_local_idx = ak.argmin(pt_diff, axis=2)  # events × genb

    # Look up the pT-rank for each matched jet
    # jet_rank is events × jets; matched_jet_local_idx is events × genb
    # We can't directly fancy-index jagged×jagged, so flatten and index
    match_ptrank = ak.where(
        is_matched,
        jet_rank[matched_jet_local_idx],
        -1,
    )

    event["genb_matched_jet"] = nearest_jet
    event["genb_is_matched"] = is_matched
    event["genb_match_dR"] = match_dR
    event["genb_match_ptrank"] = match_ptrank

    return event


def match_genH_to_fatjet(event, dR_threshold=0.8):
    """Geometric matching between gen Higgs bosons and reco FatJets.

    No b-tagging requirement on the FatJets — purely geometric dR < 0.8.

    Adds to event:
        genH_nearest_fatjet    : nearest FatJet for each gen Higgs (None if unmatched)
        genH_fatjet_dR         : dR to nearest FatJet (999 if unmatched)
        genH_fatjet_is_matched : bool per gen Higgs
        nHiggs_fatjet_matched  : int per event — how many gen Higgs matched
    """
    if "FatJet" not in event.fields or ak.all(ak.num(event.FatJet) == 0):
        # No FatJets in sample — fill with defaults
        event["genH_fatjet_dR"] = ak.full_like(event.genH.pt, 999.0)
        event["genH_fatjet_is_matched"] = ak.full_like(event.genH.pt, False, dtype=bool)
        event["nHiggs_fatjet_matched"] = np.zeros(len(event), dtype=int)
        return event

    fatjets = event.FatJet
    nearest_fj = event.genH.nearest(fatjets, threshold=dR_threshold)
    is_matched = ~ak.is_none(nearest_fj, axis=1)

    dR_val = event.genH.delta_r(ak.fill_none(nearest_fj, event.genH))
    dR_val = ak.where(is_matched, dR_val, 999.0)

    event["genH_nearest_fatjet"] = nearest_fj
    event["genH_fatjet_dR"] = dR_val
    event["genH_fatjet_is_matched"] = is_matched
    event["nHiggs_fatjet_matched"] = ak.sum(is_matched, axis=1)

    return event


def classify_overlap_pairs(event, dR_lo=0.4, dR_hi=0.8, ak4_threshold=0.4):
    """For b-quark pairs in the overlap region, classify reco fate.

    Considers pairs of gen b-quarks from the same Higgs with
    dR_lo < dR(b,b) < dR_hi.

    Reconstruction fate per pair:
        0 = lost     — fewer than 2 AK4 jets matched AND no AK8 match
        1 = resolved — both b-quarks matched to distinct AK4 jets
        2 = merged   — parent Higgs matched to an AK8 FatJet (dR < 0.8)

    Adds to event:
        overlap_mask       : bool (events, 2) — which Higgs pair is in overlap
        overlap_reco_fate  : int  (events, 2) — fate code (valid where overlap_mask)
    """
    pair_dR = event.genb_pair_dR  # (events, 2)
    overlap_mask = (pair_dR > dR_lo) & (pair_dR < dR_hi)

    # Per-Higgs: are both daughter b-quarks matched to AK4?
    b_matched = event.genb_is_matched  # (events, n_b)
    higgs_idx = event.genb_higgs_idx   # (events, n_b) — 0 or 1

    n_matched_H0 = ak.sum(b_matched & (higgs_idx == 0), axis=1)
    n_matched_H1 = ak.sum(b_matched & (higgs_idx == 1), axis=1)
    n_matched_per_H = ak.concatenate(
        [n_matched_H0[:, np.newaxis], n_matched_H1[:, np.newaxis]], axis=1
    )  # (events, 2)

    # Per-Higgs: is the gen Higgs matched to a FatJet?
    H_fj_matched = event.genH_fatjet_is_matched  # (events, nH)
    H_fj_pad = ak.pad_none(H_fj_matched, 2)
    H_fj_pad = ak.fill_none(H_fj_pad, False)

    # Classify — resolved takes priority over merged
    resolved = (n_matched_per_H == 2)
    merged = H_fj_pad & ~resolved  # only "merged" if NOT already resolved
    fate = ak.where(resolved, 1, ak.where(merged, 2, 0))

    event["overlap_mask"] = overlap_mask
    event["overlap_reco_fate"] = fate

    return event


# ---------------------------------------------------------------------------
# FSR helpers (inlined from processor_FSR_recovery to avoid transitive deps)
# ---------------------------------------------------------------------------

def select_fsr_b_quarks(selev):
    """Select hard-process b-quarks from H decays and compute per-b FSR metrics.

    Hard-process b-quarks are the isFirstCopy b-quarks: the 4 direct H->bb
    daughters before any FSR showering begins.  FSR is identified by the
    presence of gluon (pdgId=21) children of these b-quarks.

    Adds to selev:
        b_hard_has_fsr   (jagged bool,  events x b-quarks): any gluon child
        b_hard_fsr_frac  (jagged float, events x b-quarks): sum(pT_gluon) / pT_b
        evt_has_fsr      (flat bool,    events): any b-quark in event has FSR
        evt_max_fsr_frac (flat float,   events): largest FSR fraction in event
    """
    b_hard = selev.bfrom_Z_or_H[selev.bfrom_Z_or_H.hasFlags(['isFirstCopy'])]

    dR = b_hard.delta_r(b_hard.children)
    gluon_mask = (np.abs(b_hard.children.pdgId) == 21) & (dR < 0.8)

    b_hard_has_fsr = ak.any(gluon_mask, axis=2)

    pt_b        = b_hard.pt
    pt_children = b_hard.children.pt
    pt_fsr      = ak.sum(ak.where(gluon_mask, pt_children, 0.0), axis=2)
    b_hard_fsr_frac = ak.where(pt_b > 0, pt_fsr / pt_b, ak.zeros_like(pt_b))

    evt_has_fsr      = ak.any(b_hard_has_fsr, axis=1)
    evt_max_fsr_frac = ak.fill_none(ak.max(b_hard_fsr_frac, axis=1), 0.0)

    selev['b_hard_has_fsr']   = b_hard_has_fsr
    selev['b_hard_fsr_frac']  = b_hard_fsr_frac
    selev['evt_has_fsr']      = evt_has_fsr
    selev['evt_max_fsr_frac'] = evt_max_fsr_frac

    return selev


def compute_fsr_energy_loss(selev, fsr_threshold=0.20):
    """Compute FSR pT loss by matching isFirstCopy to isLastCopy b-quarks.

    FSR pT fraction = (pT_first - pT_last) / pT_first

    Adds to selev:
        b_chain_fsr_frac       (jagged float, events x 4): per-b FSR pT fraction
        b_chain_has_fsr        (jagged bool,  events x 4): fsr_frac > fsr_threshold
        evt_chain_max_fsr_frac (flat float,   events):     max across b-quarks
    """
    b_all = selev.bfrom_Z_or_H

    b_first = b_all[b_all.hasFlags(['isFirstCopy'])]
    b_last  = b_all[b_all.hasFlags(['isLastCopy'])]

    b_last_b    = b_last[b_last.pdgId > 0]
    b_last_bbar = b_last[b_last.pdgId < 0]

    matched_last_b    = b_first.nearest(b_last_b)
    matched_last_bbar = b_first.nearest(b_last_bbar)

    pt_first     = b_first.pt
    pt_last_b    = ak.fill_none(matched_last_b.pt,    0.0)
    pt_last_bbar = ak.fill_none(matched_last_bbar.pt, 0.0)

    pt_last = ak.where(b_first.pdgId > 0, pt_last_b, pt_last_bbar)

    b_chain_fsr_frac       = ak.where(pt_first > 0, (pt_first - pt_last) / pt_first, 0.0)
    b_chain_has_fsr        = b_chain_fsr_frac > fsr_threshold
    evt_chain_max_fsr_frac = ak.fill_none(ak.max(b_chain_fsr_frac, axis=1), 0.0)

    selev['b_chain_fsr_frac']       = b_chain_fsr_frac
    selev['b_chain_has_fsr']        = b_chain_has_fsr
    selev['evt_chain_max_fsr_frac'] = evt_chain_max_fsr_frac

    return selev


# ---------------------------------------------------------------------------
# Histogram helpers
# ---------------------------------------------------------------------------

def _build_dropout_histograms(processName: str, year: str):
    """Declare all histograms for the dropout study.

    Returns (fill, hist_collection) where `fill` accumulates Fill objects
    and `hist_collection` is the Collection with category axes.

    Category axes:
        process  – signal process name
        year     – data-taking year
        category – failure category:
                   "all", "failTrigger", "failPresel", "failSR", "passSR"
    """
    hist = Collection(
        process=[processName],
        year=[year],
        category=["all", "failTrigger", "failPresel", "failSR", "passSR"],
    )
    fill = Fill(process=processName, year=year, weight="weight")

    # ==================================================================
    # RECO-LEVEL HISTOGRAMS
    # ==================================================================

    # -- Reco jet multiplicity and kinematics --
    fill += hist.add(
        "nJet_selected",
        (0, 15, ("nJet_selected", "N selected jets")),
    )
    fill += hist.add(
        "nJet_tagged",
        (0, 10, ("nJet_tagged", "N b-tagged jets (medium WP)")),
    )
    fill += hist.add(
        "nJet_tagged_loose",
        (0, 10, ("nJet_tagged_loose", "N b-tagged jets (loose WP)")),
    )
    # Leading 4 selected jet pT
    for i in range(4):
        fill += hist.add(
            f"selJet{i}_pt",
            (60, 0, 600, (f"selJet{i}_pt", f"Selected jet {i} $p_T$ [GeV]")),
        )
    # All selected jet pT and eta (flattened)
    fill += hist.add(
        "selJet_pt",
        (60, 0, 600, ("selJet_pt", r"Selected jet $p_T$ [GeV]")),
    )
    fill += hist.add(
        "selJet_eta",
        (50, -3, 3, ("selJet_eta", r"Selected jet $\eta$")),
    )
    # All selected jet btag score
    fill += hist.add(
        "selJet_btag",
        (50, 0, 1, ("selJet_btag", "Selected jet b-tag score (PNet)")),
    )

    # -- Reco HT --
    fill += hist.add(
        "reco_hT",
        (60, 0, 1500, ("reco_hT", r"Reco $H_T$ [GeV]")),
    )

    # ==================================================================
    # GEN-LEVEL MATCHING HISTOGRAMS
    # ==================================================================

    # -- Acceptance: gen-b matching --
    fill += hist.add(
        "genb_pt",
        (60, 0, 600, ("genb_pt", r"Gen $b$ $p_T$ [GeV]")),
    )
    fill += hist.add(
        "genb_eta",
        (50, -5, 5, ("genb_eta", r"Gen $b$ $\eta$")),
    )
    fill += hist.add(
        "genb_matched_pt",
        (60, 0, 600, ("genb_matched_pt", r"Matched gen $b$ $p_T$ [GeV]")),
    )
    fill += hist.add(
        "genb_matched_eta",
        (50, -5, 5, ("genb_matched_eta", r"Matched gen $b$ $\eta$")),
    )
    fill += hist.add(
        "genb_reco_dR",
        (50, 0, 0.5, ("genb_reco_dR", r"$\Delta R$(gen $b$, reco jet)")),
    )
    fill += hist.add(
        "n_genb_matched",
        (0, 5, ("n_genb_matched", "N gen b-quarks matched to AK4 jet")),
    )

    # -- Collimation & overlap region --
    fill += hist.add(
        "genHiggs_bb_dR",
        (60, 0, 3.0, ("genHiggs_bb_dR", r"$\Delta R(b,b)$ same Higgs")),
    )
    fill += hist.add(
        "genHiggs_pt",
        (60, 0, 1200, ("genHiggs_pt", r"Gen Higgs $p_T$ [GeV]")),
    )
    fill += hist.add(
        "overlap_reco_fate",
        ([0, 1, 2, 3], ("overlap_reco_fate", "Reco fate (0=lost, 1=resolved, 2=merged)")),
    )
    fill += hist.add(
        "overlap_bb_dR",
        (40, 0.4, 0.8, ("overlap_bb_dR", r"$\Delta R(b,b)$ overlap region")),
    )

    # -- Low-pT intruder jets --
    fill += hist.add(
        "matched_jet_ptrank",
        (0, 10, ("matched_jet_ptrank", r"$p_T$-rank of matched reco jet")),
    )
    fill += hist.add(
        "intruder_jet_pt",
        (60, 0, 300, ("intruder_jet_pt", r"Matched reco jet $p_T$ [GeV] (rank $\geq$ 4)")),
    )
    fill += hist.add(
        "leading4_btag",
        (50, 0, 1, ("leading4_btag", "b-tag score (4 leading jets)")),
    )
    fill += hist.add(
        "matched4_btag",
        (50, 0, 1, ("matched4_btag", "b-tag score (4 matched jets)")),
    )

    # -- Gen-level kinematics --
    fill += hist.add(
        "gen_HT",
        (60, 0, 1200, ("gen_HT", r"Gen $H_T$ (4b) [GeV]")),
    )
    fill += hist.add(
        "gen_b4_pt",
        (50, 0, 250, ("gen_b4_pt", r"4th gen $b$ $p_T$ [GeV]")),
    )

    # -- Boosted / AK8 matching --
    fill += hist.add(
        "genHiggs_fatjet_dR",
        (50, 0, 2.0, ("genHiggs_fatjet_dR", r"$\Delta R$(Gen Higgs, FatJet)")),
    )
    fill += hist.add(
        "nHiggs_fatjet_matched",
        (0, 3, ("nHiggs_fatjet_matched", "N Gen Higgs matched to FatJet")),
    )

    # -- FSR diagnostics (single set, no axis splitting) --
    fill += hist.add(
        "fsr_max_frac",
        (50, 0, 1, ("fsr_max_frac", "Max FSR pT fraction")),
    )

    return fill, hist


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class analysis(processor.ProcessorABC):
    """Dropout study processor for HH→4b Run 3 on raw NanoAOD.

    Designed to run on signal MC (GluGluToHHTo4B_cHHH1) without friend trees
    or classifiers.  Applies JEC/JER from JSON-POG, then evaluates trigger,
    preselection, and SR cuts sequentially to categorize every event.
    """

    def __init__(
        self,
        *,
        corrections_metadata: dict = None,
        **kwargs,
    ):
        self.corrections_metadata = corrections_metadata or {}
        # Store any extra config keys from the YAML
        self.extra_config = kwargs

    # -------------------------------------------------------------------------
    # process()
    # -------------------------------------------------------------------------
    def process(self, event):
        # -- Chunk metadata --------------------------------------------------
        fname       = event.metadata["filename"]
        dataset     = event.metadata["dataset"]
        year        = event.metadata["year"]
        processName = event.metadata["processName"]
        year_label  = self.corrections_metadata[year]["year_label"]
        nEvent      = len(event)
        logging.info(f"[dropout] {dataset} {year}  nEvent={nEvent}")

        # -- Determine process flags -----------------------------------------
        config = processor_config(processName, dataset, event)
        config["isRun3"] = True   # force Run 3 mode
        isMC = config["isMC"]

        # -- Event-level selection (lumimask, HLT, noise filter) -------------
        event = apply_event_selection(
            event,
            self.corrections_metadata[year],
            cut_on_lumimask=not isMC,   # MC always passes lumimask
        )

        # -- Object + preselection -------------------------------------------
        # JEC/JER is applied internally by jet_selection() inside
        # apply_4b_selection using the legacy tar-file pathway for Run 3.
        event = apply_4b_selection(
            event,
            self.corrections_metadata[year],
            config=config,
            dataset=dataset,
        )

        # =====================================================================
        # GEN TRUTH — always available for signal MC
        # =====================================================================
        # 1. Build gen-level objects (b-quarks, Higgs, pair dR)
        event = build_gen_truth(event)

        # 2. FSR characterization (adds b_chain_fsr_frac, evt_chain_max_fsr_frac, etc.)
        event["bfrom_Z_or_H"] = event.genb_all  # alias expected by FSR helpers
        event = select_fsr_b_quarks(event)
        event = compute_fsr_energy_loss(event)

        # 3. Match gen b-quarks → reco AK4 jets (dR < 0.4)
        event = match_genb_to_reco_ak4(event, dR_threshold=0.4)

        # 4. Match gen Higgs → reco FatJet (dR < 0.8, no b-tag requirement)
        event = match_genH_to_fatjet(event, dR_threshold=0.8)

        # 5. Classify overlap-region bb pairs
        event = classify_overlap_pairs(event)

        # 6. Gen-level kinematics
        genb_pt_sorted = event.genb[ak.argsort(event.genb.pt, ascending=False)]
        genb_pt_sorted_pad = ak.pad_none(genb_pt_sorted, 4)
        event["gen_HT"] = ak.sum(event.genb.pt, axis=1)
        event["gen_b4_pt"] = ak.fill_none(genb_pt_sorted_pad[:, 3].pt, 0.0)

        # =====================================================================
        # SEQUENTIAL FAILURE CATEGORIZATION
        # =====================================================================
        # Signal MC: always passes lumimask and noise filter.
        # Sequential cuts:
        #   passHLT  → passPreSel (≥4 jets with ≥3/4 tagged)  → fourTag
        # "failSR" here means passed presel but is NOT fourTag (i.e. threeTag).
        passTrig   = event.passHLT
        passPreSel = event.passPreSel  # set by apply_4b_selection
        passSR     = event.fourTag     # strictest tagging requirement

        cat_all         = np.full(nEvent, True)
        cat_failTrigger = ~passTrig
        cat_failPresel  = passTrig & ~passPreSel
        cat_failSR      = passTrig & passPreSel & ~passSR
        cat_passSR      = passTrig & passPreSel & passSR

        categories = {
            "all":         cat_all,
            "failTrigger": cat_failTrigger,
            "failPresel":  cat_failPresel,
            "failSR":      cat_failSR,
            "passSR":      cat_passSR,
        }

        # =====================================================================
        # WEIGHTS
        # =====================================================================
        event["weight"] = np.ones(nEvent)

        # =====================================================================
        # PREPARE FLAT ARRAYS FOR HISTOGRAM FILLING
        # =====================================================================
        # We need event-level fields so the Fill API can pick them up.
        # For per-object histograms (genb_pt, etc.) we will fill in a
        # flattened loop per category.

        # b-tag score field name (PNet for Run 3)
        # btag_field = "btagPNetB"
        btag_field = "btagScore"

        # =====================================================================
        # HISTOGRAM FILLING
        # =====================================================================
        fill, hist_collection = _build_dropout_histograms(processName, year)

        for cat_name, cat_mask in categories.items():
            ev = event[cat_mask]
            nev = len(ev)
            if nev == 0:
                continue

            fill_kw = dict(category=cat_name)
            w_ev = np.ones(nev)

            # =============================================================
            # RECO-LEVEL HISTOGRAMS
            # =============================================================

            # Jet multiplicities
            hist_collection._hists["nJet_selected"].fill(
                nJet_selected=np.asarray(ev.nJet_selected),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._hists["nJet_tagged"].fill(
                nJet_tagged=np.asarray(ev.nJet_tagged),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._hists["nJet_tagged_loose"].fill(
                nJet_tagged_loose=np.asarray(ev.nJet_tagged_loose),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._filled.update(
                {"nJet_selected", "nJet_tagged", "nJet_tagged_loose"}
            )

            # Leading selected jet pT (pad to 4, fill per-jet)
            sel_jets = ev.selJet
            sel_jets_sorted = sel_jets[ak.argsort(sel_jets.pt, ascending=False)]
            sel_jets_pad = ak.pad_none(sel_jets_sorted, 4)
            for i in range(4):
                jet_i_pt = ak.fill_none(sel_jets_pad[:, i].pt, -1.0)
                jet_i_pt_arr = np.asarray(jet_i_pt)
                valid = jet_i_pt_arr > 0
                if np.any(valid):
                    hist_collection._hists[f"selJet{i}_pt"].fill(
                        **{f"selJet{i}_pt": jet_i_pt_arr[valid]},
                        weight=np.ones(np.sum(valid)),
                        process=processName, year=year, **fill_kw,
                    )
                    hist_collection._filled.add(f"selJet{i}_pt")

            # All selected jet pT, eta, btag (flattened)
            flat_sel_pt = _safe_flat(sel_jets.pt)
            flat_sel_eta = _safe_flat(sel_jets.eta)
            flat_sel_btag = _safe_flat(getattr(sel_jets, btag_field))
            n_sel_flat = len(flat_sel_pt)
            if n_sel_flat > 0:
                w_sel = np.ones(n_sel_flat)
                hist_collection._hists["selJet_pt"].fill(
                    selJet_pt=flat_sel_pt,
                    weight=w_sel,
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._hists["selJet_eta"].fill(
                    selJet_eta=flat_sel_eta,
                    weight=w_sel,
                    process=processName, year=year, **fill_kw,
                )
                # btag: filter out invalid values
                valid_btag = flat_sel_btag[(flat_sel_btag >= 0) & (flat_sel_btag <= 1)]
                if len(valid_btag) > 0:
                    hist_collection._hists["selJet_btag"].fill(
                        selJet_btag=valid_btag,
                        weight=np.ones(len(valid_btag)),
                        process=processName, year=year, **fill_kw,
                    )
                    hist_collection._filled.add("selJet_btag")
                hist_collection._filled.update({"selJet_pt", "selJet_eta"})

            # Reco HT
            hist_collection._hists["reco_hT"].fill(
                reco_hT=np.asarray(ev.hT),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._filled.add("reco_hT")

            # =============================================================
            # GEN-LEVEL MATCHING HISTOGRAMS
            # =============================================================
            genb = ev.genb
            genb_matched = ev.genb_is_matched
            genb_dR = ev.genb_match_dR
            genb_ptrank = ev.genb_match_ptrank

            # Gen-b pT/eta (all and matched)
            flat_genb_pt = ak.flatten(genb.pt)
            flat_genb_eta = ak.flatten(genb.eta)
            n_flat = len(flat_genb_pt)
            if n_flat > 0:
                w_flat = np.ones(n_flat)
                hist_collection._hists["genb_pt"].fill(
                    genb_pt=flat_genb_pt,
                    weight=w_flat,
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._hists["genb_eta"].fill(
                    genb_eta=flat_genb_eta,
                    weight=w_flat,
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.update({"genb_pt", "genb_eta"})

            # Number of matched gen-b per event
            n_matched_per_evt = np.asarray(ak.sum(genb_matched, axis=1))
            hist_collection._hists["n_genb_matched"].fill(
                n_genb_matched=n_matched_per_evt,
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._filled.add("n_genb_matched")

            # Matched gen-b
            matched_genb = genb[genb_matched]
            flat_mpt = _safe_flat(matched_genb.pt)
            flat_meta = _safe_flat(matched_genb.eta)
            flat_mdR = _safe_flat(genb_dR[genb_matched])
            n_matched = len(flat_mpt)
            if n_matched > 0:
                w_matched = np.ones(n_matched)
                hist_collection._hists["genb_matched_pt"].fill(
                    genb_matched_pt=flat_mpt,
                    weight=w_matched,
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._hists["genb_matched_eta"].fill(
                    genb_matched_eta=flat_meta,
                    weight=w_matched,
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._hists["genb_reco_dR"].fill(
                    genb_reco_dR=flat_mdR,
                    weight=w_matched,
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.update(
                    {"genb_matched_pt", "genb_matched_eta", "genb_reco_dR"}
                )

            # Collimation & overlap region
            flat_bb_dR = _safe_flat(ev.genb_pair_dR)
            flat_bb_dR = flat_bb_dR[flat_bb_dR >= 0]
            if len(flat_bb_dR) > 0:
                hist_collection._hists["genHiggs_bb_dR"].fill(
                    genHiggs_bb_dR=flat_bb_dR,
                    weight=np.ones(len(flat_bb_dR)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.add("genHiggs_bb_dR")

            flat_H_pt = ak.flatten(ev.genH.pt)
            if len(flat_H_pt) > 0:
                hist_collection._hists["genHiggs_pt"].fill(
                    genHiggs_pt=flat_H_pt,
                    weight=np.ones(len(flat_H_pt)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.add("genHiggs_pt")

            # Overlap-region reconstruction fate
            overlap = ev.overlap_mask
            fate = ev.overlap_reco_fate
            bb_dR_arr = ev.genb_pair_dR
            flat_fate = _safe_flat(fate[overlap])
            flat_olap_dR = _safe_flat(bb_dR_arr[overlap])
            if len(flat_fate) > 0:
                hist_collection._hists["overlap_reco_fate"].fill(
                    overlap_reco_fate=flat_fate,
                    weight=np.ones(len(flat_fate)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._hists["overlap_bb_dR"].fill(
                    overlap_bb_dR=flat_olap_dR,
                    weight=np.ones(len(flat_olap_dR)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.update(
                    {"overlap_reco_fate", "overlap_bb_dR"}
                )

            # Low-pT intruder jets
            flat_ptrank = _safe_flat(genb_ptrank[genb_matched])
            flat_ptrank = flat_ptrank[flat_ptrank >= 0]
            if len(flat_ptrank) > 0:
                hist_collection._hists["matched_jet_ptrank"].fill(
                    matched_jet_ptrank=flat_ptrank,
                    weight=np.ones(len(flat_ptrank)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.add("matched_jet_ptrank")

            # pT of matched reco jets outside the leading 4 (rank >= 4)
            matched_jet_pt = _safe_flat(
                ak.fill_none(ev.genb_matched_jet.pt, -1.0)[genb_matched]
            )
            flat_ptrank_for_pt = _safe_flat(genb_ptrank[genb_matched])
            intruder_mask = (flat_ptrank_for_pt >= 4) & (matched_jet_pt > 0)
            intruder_pt = matched_jet_pt[intruder_mask]
            if len(intruder_pt) > 0:
                hist_collection._hists["intruder_jet_pt"].fill(
                    intruder_jet_pt=intruder_pt,
                    weight=np.ones(len(intruder_pt)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.add("intruder_jet_pt")

            # b-tag: 4 leading pT jets vs 4 matched jets
            all_jets_sorted = ev.Jet[ak.argsort(ev.Jet.pt, ascending=False)]
            lead4 = ak.pad_none(all_jets_sorted, 4)[:, :4]
            flat_lead4_btag = _safe_flat(
                ak.fill_none(getattr(lead4, btag_field), -1.0)
            )
            flat_lead4_btag = flat_lead4_btag[flat_lead4_btag >= 0]
            if len(flat_lead4_btag) > 0:
                hist_collection._hists["leading4_btag"].fill(
                    leading4_btag=flat_lead4_btag,
                    weight=np.ones(len(flat_lead4_btag)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.add("leading4_btag")

            matched_jet_col = ev.genb_matched_jet
            matched_btag_raw = getattr(matched_jet_col, btag_field)
            flat_matched_btag = np.array([
                v for sub in ak.to_list(matched_btag_raw)
                for v in sub if v is not None and v >= 0
            ])
            if len(flat_matched_btag) > 0:
                hist_collection._hists["matched4_btag"].fill(
                    matched4_btag=flat_matched_btag,
                    weight=np.ones(len(flat_matched_btag)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.add("matched4_btag")

            # Gen-level kinematics
            hist_collection._hists["gen_HT"].fill(
                gen_HT=np.asarray(ev.gen_HT),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._hists["gen_b4_pt"].fill(
                gen_b4_pt=np.asarray(ev.gen_b4_pt),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._filled.update({"gen_HT", "gen_b4_pt"})

            # Boosted / AK8 matching
            flat_H_fj_dR = _safe_flat(ev.genH_fatjet_dR)
            flat_H_fj_dR_valid = flat_H_fj_dR[flat_H_fj_dR < 900]
            if len(flat_H_fj_dR_valid) > 0:
                hist_collection._hists["genHiggs_fatjet_dR"].fill(
                    genHiggs_fatjet_dR=flat_H_fj_dR_valid,
                    weight=np.ones(len(flat_H_fj_dR_valid)),
                    process=processName, year=year, **fill_kw,
                )
                hist_collection._filled.add("genHiggs_fatjet_dR")

            hist_collection._hists["nHiggs_fatjet_matched"].fill(
                nHiggs_fatjet_matched=np.asarray(ev.nHiggs_fatjet_matched),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._filled.add("nHiggs_fatjet_matched")

            # FSR diagnostics
            hist_collection._hists["fsr_max_frac"].fill(
                fsr_max_frac=np.asarray(ev.evt_chain_max_fsr_frac),
                weight=w_ev,
                process=processName, year=year, **fill_kw,
            )
            hist_collection._filled.add("fsr_max_frac")

        # =====================================================================
        # CUTFLOW SUMMARY
        # =====================================================================
        processOutput = {}
        processOutput["cutflow"] = {
            dataset: {
                "nEvent":       int(nEvent),
                "passHLT":      int(np.sum(passTrig)),
                "passPreSel":   int(np.sum(passPreSel)),
                "fourTag":      int(np.sum(passSR)),
                "failTrigger":  int(np.sum(cat_failTrigger)),
                "failPresel":   int(np.sum(cat_failPresel)),
                "failSR":       int(np.sum(cat_failSR)),
                "passSR":       int(np.sum(cat_passSR)),
            }
        }

        return hist_collection.to_dict(nonempty=True) | processOutput

    # -------------------------------------------------------------------------
    def postprocess(self, accumulator):
        return accumulator
