# coffea4bees/analysis/processors/processor_overlap_study.py
import awkward as ak
import numpy as np
import warnings

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection

from coffea4bees.analysis.helpers.event_selection import (
    apply_4b_selection,
    apply_boosted_4b_selection,
    apply_semiresolved_4b_selection,
    apply_4b_lowpt_selection,
)
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from coffea4bees.analysis.helpers.truth_tools import find_genpart
from src.physics.event_selection import apply_event_selection
from src.physics.common import mask_event_decision

import logging


# ---------------------------------------------------------------------------
# Gen-truth matching helpers (inlined from processor_dropout_Run3)
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
    event["genb_all"] = find_genpart(event.GenPart, [5], [25])

    if "status" in event.genb_all.fields:
        event["genb"] = event.genb_all[event.genb_all.status == 23]
    else:
        logging.warning("GenPart.status missing — using isLastCopy for gen b-quarks")
        event["genb"] = event.genb_all[event.genb_all.hasFlags(["isLastCopy"])]

    higgs_mask = (np.abs(event.GenPart.pdgId) == 25) & event.GenPart.hasFlags(["isLastCopy"])
    event["genH"] = event.GenPart[higgs_mask]

    genH_pad = ak.pad_none(event.genH, 2)
    dR_to_H0 = event.genb.delta_r(genH_pad[:, 0])
    dR_to_H1 = event.genb.delta_r(genH_pad[:, 1])
    event["genb_higgs_idx"] = ak.where(dR_to_H0 <= dR_to_H1, 0, 1)

    b_from_H0 = event.genb[event.genb_higgs_idx == 0]
    b_from_H1 = event.genb[event.genb_higgs_idx == 1]
    b_H0_pad = ak.pad_none(b_from_H0, 2)
    b_H1_pad = ak.pad_none(b_from_H1, 2)
    dR_H0 = ak.fill_none(b_H0_pad[:, 0].delta_r(b_H0_pad[:, 1]), -1.0)
    dR_H1 = ak.fill_none(b_H1_pad[:, 0].delta_r(b_H1_pad[:, 1]), -1.0)
    event["genb_pair_dR"] = ak.concatenate(
        [dR_H0[:, np.newaxis], dR_H1[:, np.newaxis]], axis=1
    )

    return event


def match_genb_to_reco_ak4(event, dR_threshold=0.4):
    """Match gen-b quarks to reco AK4 jets and compute per-b matching info.

    Adds to event:
        genb_matched_jet  : nearest reco Jet for each gen-b (None if no match)
        genb_is_matched   : bool per gen-b, True if a reco jet is within dR_threshold
        genb_match_dR     : dR to nearest reco jet (filled with 999 if unmatched)
        genb_match_ptrank : pT-rank (0-based) of the matched reco jet among all
                            selected jets (filled with -1 if unmatched)
    """
    reco_jets = event.Jet

    nearest_jet = event.genb.nearest(reco_jets, threshold=dR_threshold)
    is_matched = ~ak.is_none(nearest_jet, axis=1)

    match_dR = event.genb.delta_r(ak.fill_none(nearest_jet, event.genb))
    match_dR = ak.where(is_matched, match_dR, 999.0)

    jet_pt_sorted_idx = ak.argsort(reco_jets.pt, axis=1, ascending=False)
    jet_rank = ak.argsort(jet_pt_sorted_idx, axis=1)

    matched_pt = ak.fill_none(nearest_jet.pt, -1.0)
    reco_pt = reco_jets.pt

    matched_pt_bc = matched_pt[:, :, np.newaxis]
    reco_pt_bc = reco_pt[:, np.newaxis, :]
    pt_diff = np.abs(matched_pt_bc - reco_pt_bc)
    matched_jet_local_idx = ak.argmin(pt_diff, axis=2)

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

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            corrections_metadata: dict = None,
            object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml",
            resolved_triggers_cfg: str = "coffea4bees/metadata/triggers_HH4b.yml",
            boosted_triggers_cfg: str = "coffea4bees/metadata/boosted_triggers_HH4b.yml",
            **kwargs
    ):
        self.corrections_metadata = corrections_metadata
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None

        import yaml
        with open(resolved_triggers_cfg, "r") as f:
            self.resolved_triggers = yaml.safe_load(f)["triggers"]
        with open(boosted_triggers_cfg, "r") as f:
            self.boosted_triggers = yaml.safe_load(f)["triggers"]

    def process(self, event):
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        isMC    = True if event.run[0] == 1 else False
        nEvent  = len(event)

        event = apply_event_selection(event, self.corrections_metadata[year], cut_on_lumimask=False)

        event = apply_4b_lowpt_selection(
            event,
            self.corrections_metadata[year],
            sel_cfg=self.sel_cfg,
            isMC=isMC,
        )
        event = apply_boosted_4b_selection(event)
        event = apply_semiresolved_4b_selection(event)

        # Compute selection-specific HLT trigger masks
        if 'HLT' in event.fields:
            resolved_trig_list = self.resolved_triggers.get(year, [])
            boosted_trig_list = self.boosted_triggers.get(year, [])
            semiresolved_trig_list = list(set(resolved_trig_list + boosted_trig_list))

            passHLT_resolved = mask_event_decision(
                event, decision="OR", branch="HLT", list_to_mask=resolved_trig_list
            )
            passHLT_boosted = mask_event_decision(
                event, decision="OR", branch="HLT", list_to_mask=boosted_trig_list
            )
            passHLT_semiresolved = mask_event_decision(
                event, decision="OR", branch="HLT", list_to_mask=semiresolved_trig_list
            )
        else:
            passHLT_resolved = np.full(nEvent, True)
            passHLT_boosted = np.full(nEvent, True)
            passHLT_semiresolved = np.full(nEvent, True)

        # Gen-matching (MC only)
        if isMC and 'GenPart' in event.fields:
            event = build_gen_truth(event)
            event = match_genb_to_reco_ak4(event, dR_threshold=0.4)
            has_genmatching = True
        else:
            has_genmatching = False

        passResolved     = event.fourTag & passHLT_resolved
        passBoosted      = event.passBoostedSel & passHLT_boosted
        passSemiResolved = event.passSemiResolvedSel & passHLT_semiresolved
        passLowPt        = event.lowpt_fourTag & passHLT_resolved

        selections = PackedSelection()
        selections.add("lumimask",         event.lumimask)
        selections.add("passNoiseFilter",  event.passNoiseFilter)
        selections.add("passResolved",     passResolved)
        selections.add("passBoostedSel",   passBoosted)
        selections.add("passSemiResolved", passSemiResolved)
        selections.add("passLowPt",        passLowPt)

        base      = ["lumimask", "passNoiseFilter"]
        base_mask = selections.require(**{k: True for k in base})

        def count(**kwargs):
            return int(ak.sum(selections.require(**{k: True for k in base}, **kwargs)))

        # Events that pass the base cuts but fail both resolved AND lowPt.
        # NOTE: some of these events may still pass boosted or semiresolved —
        # they are NOT lost to the analysis, but we include them here to study
        # whether resolved/lowPt could be improved to capture them too.
        none_mask = base_mask & ~passResolved & ~passLowPt

        def cn(cut):
            """Count 'none' events also satisfying 'cut'."""
            return int(ak.sum(none_mask & cut))

        # ── Per-jet component masks (re-using flags already set by apply_4b_lowpt_selection) ──
        j = event.Jet
        btagWP        = self.corrections_metadata[year]['btagWP']
        mask_pt40     = j.pt >= 40
        mask_eta24    = np.abs(j.eta) <= 2.4
        mask_jetId    = j.jetId >= 2
        mask_noPU     = ~j.pileup
        mask_lepCl    = j.lepton_cleaned
        mask_btag_M   = j.btagScore >= btagWP['M']
        mask_btag_L   = j.btagScore >= btagWP['L']

        # Sequential resolved cut chain: how many jets survive each successive gate?
        n_pt40           = ak.sum(mask_pt40, axis=1)
        n_pt40_eta       = ak.sum(mask_pt40 & mask_eta24, axis=1)
        n_pt40_eta_id    = ak.sum(mask_pt40 & mask_eta24 & mask_jetId, axis=1)
        n_pt40_eta_id_pu = ak.sum(mask_pt40 & mask_eta24 & mask_jetId & mask_noPU, axis=1)
        n_selected       = event.nJet_selected   # = pt40 & eta24 & jetId & ~pileup & lepCl
        n_tagged         = event.nJet_tagged      # = selected & btag>=M

        # Hypothetical jet counts with each individual cut loosened (all other cuts kept)
        n_pt35_rest    = ak.sum((j.pt >= 35)           & mask_eta24 & mask_jetId & mask_noPU & mask_lepCl, axis=1)
        n_eta25_rest   = ak.sum(mask_pt40 & (np.abs(j.eta) <= 2.5) & mask_jetId & mask_noPU & mask_lepCl, axis=1)
        n_noPU_rest    = ak.sum(mask_pt40 & mask_eta24 & mask_jetId              & mask_lepCl, axis=1)
        n_noLep_rest   = ak.sum(mask_pt40 & mask_eta24 & mask_jetId & mask_noPU,               axis=1)
        n_tagL_rest    = ak.sum(j.selected & mask_btag_L, axis=1)  # loose btag on already-selected jets

        # Low-pT path
        n_lowpt_sel    = event.nJet_selected_lowpt
        n_lowpt_tagged = event.nJet_tagged_lowpt

        # Hypothetical: lower the low-pT pt floor from 15 to 10 GeV
        mask_pt10_lowpt = (j.pt >= 10) & (j.pt < 40)
        n_lowpt10_tagged = ak.sum(
            mask_pt10_lowpt & mask_eta24 & mask_jetId & mask_noPU & mask_lepCl & mask_btag_M, axis=1
        )

        cutflow = {
            # Total "none" population (pass base, fail resolved AND lowPt)
            # Includes events that pass boosted/semiresolved — see sub-breakdown below.
            'none_total': int(ak.sum(none_mask)),

            # ── Overlap with boosted/semiresolved ──────────────────────────────────
            # Of the events failing resolved+lowPt, how many are already captured
            # by boosted or semiresolved? These are NOT truly lost — but they are
            # candidates for improving resolved/lowPt coverage.
            'none_also_boosted':       cn(passBoosted),
            'none_also_semiresolved':  cn(passSemiResolved),
            'none_also_boosted_or_sr': cn(passBoosted | passSemiResolved),
            'none_truly_lost':         cn(~passBoosted & ~passSemiResolved),

            # ── Resolved cut chain: total jets in "none" events surviving each gate ──
            # A big drop between two consecutive entries identifies the dominant cut.
            'none_jets_after_pt40':           int(ak.sum(none_mask * n_pt40)),
            'none_jets_after_pt40_eta24':     int(ak.sum(none_mask * n_pt40_eta)),
            'none_jets_after_pt40_eta_id':    int(ak.sum(none_mask * n_pt40_eta_id)),
            'none_jets_after_pt40_eta_id_pu': int(ak.sum(none_mask * n_pt40_eta_id_pu)),
            'none_jets_after_fullsel':        int(ak.sum(none_mask * n_selected)),
            'none_jets_tagged_medium':        int(ak.sum(none_mask * n_tagged)),

            # ── How many "none" events reach >=4 jets at each gate? ──
            'none_events_ge4_after_pt40':           cn(n_pt40 >= 4),
            'none_events_ge4_after_pt40_eta24':     cn(n_pt40_eta >= 4),
            'none_events_ge4_after_pt40_eta_id':    cn(n_pt40_eta_id >= 4),
            'none_events_ge4_after_pt40_eta_id_pu': cn(n_pt40_eta_id_pu >= 4),
            'none_events_ge4_fullsel':              cn(n_selected >= 4),
            'none_events_ge4_tagged_medium':        cn(n_tagged >= 4),  # sanity: should be 0

            # ── B-tag breakdown for "none" events with >=4 selected jets ──
            'none_4jets_0btag': cn((n_selected >= 4) & (n_tagged == 0)),
            'none_4jets_1btag': cn((n_selected >= 4) & (n_tagged == 1)),
            'none_4jets_2btag': cn((n_selected >= 4) & (n_tagged == 2)),
            'none_4jets_3btag': cn((n_selected >= 4) & (n_tagged == 3)),

            # ── LowPt sub-investigation ──
            # "none" events with 3 medium b-tags + >=4 jets are the lowPt target population.
            # If they are still "none", the low-pT b-tagged jet is missing.
            'none_3btag_4jets_total':              cn((n_selected >= 4) & (n_tagged == 3)),
            'none_3btag_4jets_has_lowpt_tag':      cn((n_selected >= 4) & (n_tagged == 3) & (n_lowpt_tagged > 0)),  # sanity: 0
            'none_3btag_4jets_no_lowpt_tag':       cn((n_selected >= 4) & (n_tagged == 3) & (n_lowpt_tagged == 0)),
            'none_3btag_4jets_has_lowpt_untagged': cn((n_selected >= 4) & (n_tagged == 3) & (n_lowpt_tagged == 0) & (n_lowpt_sel > 0)),

            # ── Gain counters: events recovered by loosening one cut at a time ──
            # How many currently-failing events would reach >=4 jets with a looser threshold?
            'gain_pt35_reaches_4jets':    cn((n_selected < 4) & (n_pt35_rest >= 4)),
            'gain_eta25_reaches_4jets':   cn((n_selected < 4) & (n_eta25_rest >= 4)),
            'gain_noPU_reaches_4jets':    cn((n_selected < 4) & (n_noPU_rest >= 4)),
            'gain_noLepCl_reaches_4jets': cn((n_selected < 4) & (n_noLep_rest >= 4)),

            # How many events with >=4 jets + >=3 medium btags would pass if we allow 3M+1L?
            'gain_3M1L_reaches_resolved': cn((n_selected >= 4) & (n_tagged >= 3) & (n_tagL_rest >= 4)),

            # How many "none" 3btag+4jet events gain a lowPt tag by lowering pt floor to 10 GeV?
            'gain_lowpt10_reaches_lowpt': cn((n_selected >= 4) & (n_tagged == 3) & (n_lowpt_tagged == 0) & (n_lowpt10_tagged > 0)),

            # ── Primary summary ──
            'none_fail_jetmult':           cn(n_selected < 4),
            'none_pass_jetmult_fail_btag': cn((n_selected >= 4) & (n_tagged < 4)),
        }

        # ── Gen-matching analysis of the "none" population ──────────────────────
        genmatch_cutflow = {}
        if has_genmatching:
            n_genb_matched          = ak.sum(event.genb_is_matched, axis=1)
            matched_jet_is_selected = ak.fill_none(event.genb_matched_jet.selected,      False)
            matched_jet_is_tagged   = ak.fill_none(event.genb_matched_jet.tagged,        False)
            matched_jet_is_lowpt    = ak.fill_none(event.genb_matched_jet.selected_lowpt, False)
            matched_jet_is_lowpt_tagged = ak.fill_none(event.genb_matched_jet.tagged_lowpt, False)

            n_genb_matched_selected     = ak.sum(event.genb_is_matched & matched_jet_is_selected,      axis=1)
            n_genb_matched_tagged       = ak.sum(event.genb_is_matched & matched_jet_is_tagged,        axis=1)
            n_genb_matched_lowpt_sel    = ak.sum(event.genb_is_matched & matched_jet_is_lowpt,         axis=1)
            n_genb_matched_lowpt_tagged = ak.sum(event.genb_is_matched & matched_jet_is_lowpt_tagged,  axis=1)

            # Global efficiency (all base events)
            genmatch_cutflow['all_base_n_genb_matched_any']      = int(ak.sum(base_mask * n_genb_matched))
            genmatch_cutflow['all_base_n_genb_matched_selected'] = int(ak.sum(base_mask * n_genb_matched_selected))
            genmatch_cutflow['all_base_n_genb_matched_tagged']   = int(ak.sum(base_mask * n_genb_matched_tagged))

            # "None" events: how many gen-b quarks land in each jet category?
            genmatch_cutflow['none_n_genb_matched_any']          = int(ak.sum(none_mask * n_genb_matched))
            genmatch_cutflow['none_n_genb_matched_selected']     = int(ak.sum(none_mask * n_genb_matched_selected))
            genmatch_cutflow['none_n_genb_matched_tagged']       = int(ak.sum(none_mask * n_genb_matched_tagged))
            genmatch_cutflow['none_n_genb_matched_lowpt_sel']    = int(ak.sum(none_mask * n_genb_matched_lowpt_sel))
            genmatch_cutflow['none_n_genb_matched_lowpt_tagged'] = int(ak.sum(none_mask * n_genb_matched_lowpt_tagged))

            # ── 3btag+4jets population (62k events, the lowPt target) ──
            # These events have 4 selected jets, 3 medium b-tags, and no low-pT b-tag.
            # We want to know: where is the 4th gen-b quark?
            mask_3btag_4jets = none_mask & (n_selected >= 4) & (n_tagged == 3)

            has_genb_in_lowpt_untagged = (
                event.genb_is_matched & matched_jet_is_lowpt & ~matched_jet_is_lowpt_tagged
            )
            n_genb_in_lowpt_untagged = ak.sum(has_genb_in_lowpt_untagged, axis=1)

            genmatch_cutflow['none_3btag_4jets_total']                 = int(ak.sum(mask_3btag_4jets))
            genmatch_cutflow['none_3btag_4jets_genb_in_lowpt_untagged'] = int(
                ak.sum(mask_3btag_4jets & (n_genb_in_lowpt_untagged > 0))
            )
            genmatch_cutflow['none_3btag_4jets_4th_genb_lost'] = int(
                ak.sum(mask_3btag_4jets & (n_genb_in_lowpt_untagged == 0))
            )

            # For events where 4th gen-b is "lost" (no reco match at all),
            # look at the gen-b quark kinematics directly.
            # We select the unmatched gen-b quarks in these events.
            mask_3btag_lost_bc = ak.broadcast_arrays(
                mask_3btag_4jets & (n_genb_in_lowpt_untagged == 0),
                event.genb_is_matched
            )[0]
            lost_genb = event.genb[mask_3btag_lost_bc & ~event.genb_is_matched]

            # Histograms stored as flat numpy arrays so they accumulate correctly
            # (dicts get deep-merged incorrectly across chunks; arrays get summed).
            pt_bins   = np.linspace(0, 100, 26)   # 25 bins, 0-100 GeV
            eta_bins  = np.linspace(0, 5,   26)   # 25 bins, 0-5
            btag_bins = np.linspace(0, 1,   26)   # 25 bins, 0-1

            lost_pt_flat  = ak.to_numpy(ak.flatten(lost_genb.pt))
            lost_eta_flat = ak.to_numpy(ak.flatten(np.abs(lost_genb.eta)))

            genmatch_cutflow['lost_genb_pt_hist'],  _ = np.histogram(lost_pt_flat,  bins=pt_bins)
            genmatch_cutflow['lost_genb_eta_hist'], _ = np.histogram(lost_eta_flat, bins=eta_bins)
            genmatch_cutflow['lost_genb_pt_bins']    = pt_bins.tolist()
            genmatch_cutflow['lost_genb_eta_bins']   = eta_bins.tolist()

            # For matched but untagged gen-b in low-pT jets: btag score distribution
            # (how close to the medium WP threshold are they?)
            lowpt_untagged_genb_bscore = ak.fill_none(
                event.genb_matched_jet.btagScore[
                    ak.broadcast_arrays(mask_3btag_4jets, event.genb_is_matched)[0] &
                    event.genb_is_matched & matched_jet_is_lowpt & ~matched_jet_is_lowpt_tagged
                ],
                -1.0
            )
            lowpt_btag_flat = ak.to_numpy(ak.flatten(lowpt_untagged_genb_bscore[lowpt_untagged_genb_bscore >= 0]))
            genmatch_cutflow['lost_genb_lowpt_btag_hist'], _ = np.histogram(lowpt_btag_flat, bins=btag_bins)
            genmatch_cutflow['btag_bins'] = btag_bins.tolist()

            # ── 2btag+4jets population (44k events, the 3M+1L target) ──
            mask_2btag_4jets = none_mask & (n_selected >= 4) & (n_tagged == 2)

            has_genb_in_selected_untagged = (
                event.genb_is_matched & matched_jet_is_selected & ~matched_jet_is_tagged
            )
            n_genb_in_selected_untagged = ak.sum(has_genb_in_selected_untagged, axis=1)

            genmatch_cutflow['none_2btag_4jets_total']                 = int(ak.sum(mask_2btag_4jets))
            genmatch_cutflow['none_2btag_4jets_has_genb_untagged_sel'] = int(
                ak.sum(mask_2btag_4jets & (n_genb_in_selected_untagged > 0))
            )
            genmatch_cutflow['none_2btag_4jets_n_genb_untagged_sel'] = int(
                ak.sum(mask_2btag_4jets * n_genb_in_selected_untagged)
            )

            # For the untagged-but-selected gen-b jets in 2btag events:
            # btag score distribution (how far below medium WP?)
            sel_untagged_genb_bscore = ak.fill_none(
                event.genb_matched_jet.btagScore[
                    ak.broadcast_arrays(mask_2btag_4jets, event.genb_is_matched)[0] &
                    event.genb_is_matched & matched_jet_is_selected & ~matched_jet_is_tagged
                ],
                -1.0
            )
            sel_btag_flat = ak.to_numpy(ak.flatten(sel_untagged_genb_bscore[sel_untagged_genb_bscore >= 0]))
            genmatch_cutflow['untagged_sel_genb_btag_hist'], _ = np.histogram(sel_btag_flat, bins=btag_bins)

        # ── Only Semi-Resolved diagnostics ──
        only_sr_mask = base_mask & passSemiResolved & ~passResolved & ~passLowPt & ~passBoosted
        total_only_sr = ak.sum(only_sr_mask)
        
        fail_jets = ak.sum(only_sr_mask & (n_selected < 4))
        fail_tags = ak.sum(only_sr_mask & (n_selected >= 4) & (n_tagged < 4))
        fail_trigger = ak.sum(only_sr_mask & (n_selected >= 4) & (n_tagged >= 4) & ~passHLT_resolved)
        fail_other = total_only_sr - (fail_jets + fail_tags + fail_trigger)

        # Low-pT failure modes
        fail_lowpt_jets = ak.sum(only_sr_mask & (n_selected < 4))
        fail_lowpt_tags_short = ak.sum(only_sr_mask & (n_selected >= 4) & (n_tagged < 3))
        fail_lowpt_tags_long = ak.sum(only_sr_mask & (n_selected >= 4) & (n_tagged > 3))
        fail_lowpt_no_lowpt_tag = ak.sum(
            only_sr_mask & (n_selected >= 4) & (n_tagged == 3) & (n_lowpt_tagged == 0)
        )
        fail_lowpt_trigger = ak.sum(
            only_sr_mask & (n_selected >= 4) & (n_tagged == 3) & (n_lowpt_tagged >= 1) & ~passHLT_resolved
        )
        fail_lowpt_other = total_only_sr - (
            fail_lowpt_jets + fail_lowpt_tags_short + fail_lowpt_tags_long +
            fail_lowpt_no_lowpt_tag + fail_lowpt_trigger
        )

        fj_mask = (
            (event.FatJet.pt > 250) &
            (np.abs(event.FatJet.eta) < 2.5) &
            (event.FatJet.msoftdrop > 50) &
            (event.FatJet.msoftdrop < 200) &
            (event.FatJet.particleNetMD_Xbb > 0.7)
        )
        candFatJets = event.FatJet[fj_mask]
        lead_fatjet = ak.pad_none(candFatJets, 1)[:, 0]
        
        sr_events_pt = lead_fatjet.pt[only_sr_mask]
        sr_events_mass = lead_fatjet.msoftdrop[only_sr_mask]
        sr_events_xbb = lead_fatjet.particleNetMD_Xbb[only_sr_mask]
        
        sr_pt_flat = ak.to_numpy(ak.fill_none(sr_events_pt, -1.0))
        sr_mass_flat = ak.to_numpy(ak.fill_none(sr_events_mass, -1.0))
        sr_xbb_flat = ak.to_numpy(ak.fill_none(sr_events_xbb, -1.0))
        
        sr_njets_flat = ak.to_numpy(n_selected[only_sr_mask])
        sr_ntags_flat = ak.to_numpy(n_tagged[only_sr_mask])
        sr_nlowpt_jets_flat = ak.to_numpy(n_lowpt_sel[only_sr_mask])
        sr_nlowpt_tags_flat = ak.to_numpy(n_lowpt_tagged[only_sr_mask])
        
        sr_pt_bins = np.linspace(200, 1000, 41)
        sr_mass_bins = np.linspace(0, 250, 26)
        sr_xbb_bins = np.linspace(0, 1, 26)
        sr_njets_bins = np.arange(0, 11)
        
        sr_njets_hist, _ = np.histogram(sr_njets_flat, bins=sr_njets_bins)
        sr_ntags_hist, _ = np.histogram(sr_ntags_flat, bins=sr_njets_bins)
        sr_nlowpt_jets_hist, _ = np.histogram(sr_nlowpt_jets_flat, bins=sr_njets_bins)
        sr_nlowpt_tags_hist, _ = np.histogram(sr_nlowpt_tags_flat, bins=sr_njets_bins)
        sr_pt_hist, _ = np.histogram(sr_pt_flat[sr_pt_flat >= 0], bins=sr_pt_bins)
        sr_mass_hist, _ = np.histogram(sr_mass_flat[sr_mass_flat >= 0], bins=sr_mass_bins)
        sr_xbb_hist, _ = np.histogram(sr_xbb_flat[sr_xbb_flat >= 0], bins=sr_xbb_bins)
        
        only_sr_diagnostics = {
            'total_only_sr': int(total_only_sr),
            'fail_resolved_jets': int(fail_jets),
            'fail_resolved_tags': int(fail_tags),
            'fail_resolved_trigger': int(fail_trigger),
            'fail_resolved_other': int(fail_other),
            'fail_lowpt_jets': int(fail_lowpt_jets),
            'fail_lowpt_tags_short': int(fail_lowpt_tags_short),
            'fail_lowpt_tags_long': int(fail_lowpt_tags_long),
            'fail_lowpt_no_lowpt_tag': int(fail_lowpt_no_lowpt_tag),
            'fail_lowpt_trigger': int(fail_lowpt_trigger),
            'fail_lowpt_other': int(fail_lowpt_other),
            'njets_hist': sr_njets_hist,
            'ntags_hist': sr_ntags_hist,
            'nlowpt_jets_hist': sr_nlowpt_jets_hist,
            'nlowpt_tags_hist': sr_nlowpt_tags_hist,
            'pt_hist': sr_pt_hist,
            'mass_hist': sr_mass_hist,
            'xbb_hist': sr_xbb_hist,
            'btag_wp_m': float(btagWP['M']),
        }

        if has_genmatching:
            matched_btags = ak.fill_none(event.genb_matched_jet.btagScore, -1.0)
            sr_matched_btags = matched_btags[only_sr_mask]
            sr_matched_btags_sorted = ak.sort(sr_matched_btags, axis=1, ascending=False)
            sr_matched_btags_pad = ak.pad_none(sr_matched_btags_sorted, 4, clip=True)
            
            btag_1 = ak.to_numpy(ak.fill_none(sr_matched_btags_pad[:, 0], -1.0))
            btag_2 = ak.to_numpy(ak.fill_none(sr_matched_btags_pad[:, 1], -1.0))
            btag_3 = ak.to_numpy(ak.fill_none(sr_matched_btags_pad[:, 2], -1.0))
            btag_4 = ak.to_numpy(ak.fill_none(sr_matched_btags_pad[:, 3], -1.0))
            
            sr_matched_btag_bins = np.linspace(0, 1, 51)
            hist_btag_1, _ = np.histogram(btag_1[btag_1 >= 0], bins=sr_matched_btag_bins)
            hist_btag_2, _ = np.histogram(btag_2[btag_2 >= 0], bins=sr_matched_btag_bins)
            hist_btag_3, _ = np.histogram(btag_3[btag_3 >= 0], bins=sr_matched_btag_bins)
            hist_btag_4, _ = np.histogram(btag_4[btag_4 >= 0], bins=sr_matched_btag_bins)

            # Delta R between jets matched to H0 and H1
            jets_h0 = event.genb_matched_jet[(event.genb_higgs_idx == 0) & event.genb_is_matched]
            jets_h1 = event.genb_matched_jet[(event.genb_higgs_idx == 1) & event.genb_is_matched]
            
            pairs = ak.cartesian([jets_h0, jets_h1], axis=1)
            dr_pairs = pairs["0"].delta_r(pairs["1"])
            min_dr = ak.min(dr_pairs, axis=1)
            
            sr_min_dr = ak.to_numpy(ak.fill_none(min_dr[only_sr_mask], -1.0))
            
            # Delta R between two jets of the same Higgs (H0 and H1)
            jets_h0_pad = ak.pad_none(jets_h0, 2, clip=True)
            dr_h0 = jets_h0_pad[:, 0].delta_r(jets_h0_pad[:, 1])
            sr_dr_h0 = ak.to_numpy(ak.fill_none(dr_h0[only_sr_mask], -1.0))
            
            jets_h1_pad = ak.pad_none(jets_h1, 2, clip=True)
            dr_h1 = jets_h1_pad[:, 0].delta_r(jets_h1_pad[:, 1])
            sr_dr_h1 = ak.to_numpy(ak.fill_none(dr_h1[only_sr_mask], -1.0))
            
            sr_dr_bins = np.linspace(0, 5, 51)
            hist_dr, _ = np.histogram(sr_min_dr[sr_min_dr >= 0], bins=sr_dr_bins)
            hist_dr_h0, _ = np.histogram(sr_dr_h0[sr_dr_h0 >= 0], bins=sr_dr_bins)
            hist_dr_h1, _ = np.histogram(sr_dr_h1[sr_dr_h1 >= 0], bins=sr_dr_bins)

            # Higgs pt (lead and sublead pt from matched reco jets)
            def sum_jets(jets):
                px = ak.sum(jets.pt * np.cos(jets.phi), axis=-1)
                py = ak.sum(jets.pt * np.sin(jets.phi), axis=-1)
                pz = ak.sum(jets.pt * np.sinh(jets.eta), axis=-1)
                e = ak.sum(np.sqrt(jets.pt**2 * np.cosh(jets.eta)**2 + jets.mass**2), axis=-1)
                pt = np.sqrt(px**2 + py**2)
                pt_safe = ak.where(pt > 0, pt, 1e-9)
                eta = np.arcsinh(pz / pt_safe)
                phi = np.arctan2(py, px)
                m2 = e**2 - px**2 - py**2 - pz**2
                mass = np.sqrt(ak.where(m2 > 0, m2, 0))
                return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass})

            h0_reco = sum_jets(jets_h0)
            h1_reco = sum_jets(jets_h1)
            h_pts = ak.concatenate([h0_reco.pt[:, np.newaxis], h1_reco.pt[:, np.newaxis]], axis=1)
            h_pts_sorted = ak.sort(h_pts, axis=1, ascending=False)
            lead_h_pt = h_pts_sorted[:, 0]
            sublead_h_pt = h_pts_sorted[:, 1]
            
            sr_lead_h_pt = ak.to_numpy(ak.fill_none(lead_h_pt[only_sr_mask], -1.0))
            sr_sublead_h_pt = ak.to_numpy(ak.fill_none(sublead_h_pt[only_sr_mask], -1.0))
            sr_h_pt_bins = np.linspace(0, 800, 41)
            hist_lead_pt, _ = np.histogram(sr_lead_h_pt[sr_lead_h_pt >= 0], bins=sr_h_pt_bins)
            hist_sublead_pt, _ = np.histogram(sr_sublead_h_pt[sr_sublead_h_pt >= 0], bins=sr_h_pt_bins)

            # Invariant mass of matched reco jets
            all_matched_jets = event.genb_matched_jet[event.genb_is_matched]
            total_reco_system = sum_jets(all_matched_jets)
            m4j = total_reco_system.mass
            
            sr_m4j = ak.to_numpy(ak.fill_none(m4j[only_sr_mask], -1.0))
            sr_m4j_bins = np.linspace(0, 1200, 61)
            hist_m4j, _ = np.histogram(sr_m4j[sr_m4j >= 0], bins=sr_m4j_bins)

            # Eta of 4th highest b-tag score jet (reco vs gen)
            matched_jets_pad = ak.pad_none(event.genb_matched_jet, 4, clip=True)
            btag_scores = ak.fill_none(matched_jets_pad.btagScore, -1.0)
            sort_idx = ak.argsort(btag_scores, axis=1, ascending=False)
            genb_pad = ak.pad_none(event.genb, 4, clip=True)
            
            matched_jets_sorted = matched_jets_pad[sort_idx]
            genb_sorted = genb_pad[sort_idx]
            
            reco_jet_4th = matched_jets_sorted[:, 3]
            genb_4th = genb_sorted[:, 3]
            
            sr_reco_eta = ak.to_numpy(ak.fill_none(reco_jet_4th.eta[only_sr_mask], -999.0))
            sr_gen_eta = ak.to_numpy(ak.fill_none(genb_4th.eta[only_sr_mask], -999.0))
            sr_eta_bins = np.linspace(-2.5, 2.5, 51)
            hist_reco_eta, _ = np.histogram(sr_reco_eta[sr_reco_eta > -900.0], bins=sr_eta_bins)
            hist_gen_eta, _ = np.histogram(sr_gen_eta[sr_gen_eta > -900.0], bins=sr_eta_bins)
            
            only_sr_diagnostics.update({
                'matched_btag_1_hist': hist_btag_1,
                'matched_btag_2_hist': hist_btag_2,
                'matched_btag_3_hist': hist_btag_3,
                'matched_btag_4_hist': hist_btag_4,
                'match_dr_h1h2_hist': hist_dr,
                'match_dr_h0_hist': hist_dr_h0,
                'match_dr_h1_hist': hist_dr_h1,
                'match_h_lead_pt_hist': hist_lead_pt,
                'match_h_sublead_pt_hist': hist_sublead_pt,
                'match_m4j_hist': hist_m4j,
                'match_4th_reco_eta_hist': hist_reco_eta,
                'match_4th_gen_eta_hist': hist_gen_eta,
            })

        output = {
            dataset: {
                'numEvents': nEvent,
                'passBase':   int(ak.sum(base_mask)),
                # exclusive categories
                'onlyResolved':     count(passResolved=True,  passBoostedSel=False, passSemiResolved=False, passLowPt=False),
                'onlyBoosted':      count(passResolved=False, passBoostedSel=True,  passSemiResolved=False, passLowPt=False),
                'onlySemiResolved': count(passResolved=False, passBoostedSel=False, passSemiResolved=True,  passLowPt=False),
                'onlyLowPt':        count(passResolved=False, passBoostedSel=False, passSemiResolved=False, passLowPt=True),
                # pairwise overlaps
                'resolved_and_boosted':      count(passResolved=True,  passBoostedSel=True,  passSemiResolved=False, passLowPt=False),
                'resolved_and_semiresolved': count(passResolved=True,  passBoostedSel=False, passSemiResolved=True,  passLowPt=False),
                'resolved_and_lowpt':        count(passResolved=True,  passBoostedSel=False, passSemiResolved=False, passLowPt=True),
                'boosted_and_semiresolved':  count(passResolved=False, passBoostedSel=True,  passSemiResolved=True,  passLowPt=False),
                'boosted_and_lowpt':         count(passResolved=False, passBoostedSel=True,  passSemiResolved=False, passLowPt=True),
                'semiresolved_and_lowpt':    count(passResolved=False, passBoostedSel=False, passSemiResolved=True,  passLowPt=True),
                # three-way overlaps
                'resolved_boosted_semiresolved': count(passResolved=True,  passBoostedSel=True,  passSemiResolved=True,  passLowPt=False),
                'resolved_boosted_lowpt':        count(passResolved=True,  passBoostedSel=True,  passSemiResolved=False, passLowPt=True),
                'resolved_semiresolved_lowpt':   count(passResolved=True,  passBoostedSel=False, passSemiResolved=True,  passLowPt=True),
                'boosted_semiresolved_lowpt':    count(passResolved=False, passBoostedSel=True,  passSemiResolved=True,  passLowPt=True),
                # all four
                'all_four': count(passResolved=True, passBoostedSel=True, passSemiResolved=True, passLowPt=True),
                'anySelection': int(ak.sum(
                    base_mask &
                    (selections.all("passResolved") | selections.all("passBoostedSel") |
                     selections.all("passSemiResolved") | selections.all("passLowPt"))
                )),
                # Events failing resolved+lowPt (may still pass boosted/semiresolved)
                'none_res_lowpt': int(ak.sum(none_mask)),
                # Events failing ALL four selections
                'none_all': count(passResolved=False, passBoostedSel=False, passSemiResolved=False, passLowPt=False),
                # diagnostic cutflow for the "none" population
                'cutflow': cutflow,
                'genmatch_cutflow': genmatch_cutflow,
                'only_sr_diagnostics': only_sr_diagnostics,
            }
        }

        return output

    def postprocess(self, accumulator):
        return accumulator
