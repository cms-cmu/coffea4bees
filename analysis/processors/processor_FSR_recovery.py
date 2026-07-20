import math
import numpy as np
from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor
import awkward as ak
from src.hist_tools import Collection, Fill
from coffea4bees.analysis.helpers.hist_templates import (
    QuadJetHistsSelected,
    QuadJetHistsMinDr,
    QuadJetHistsSRSingle,
    SvBHists,
)
import logging
from coffea4bees.analysis.helpers.truth_tools import find_genpart


FSR_FLAG_NAMES = [
    'isPrompt',
    'isHardProcess',
    'fromHardProcess',
    'isFirstCopy',
    'isLastCopy',
    'isLastCopyBeforeFSR',
    'fromHardProcessBeforeFSR',
]

# PDG particle ID → human-readable label
PDG_NAMES = {
     1: 'd',        -1: 'dbar',
     2: 'u',        -2: 'ubar',
     3: 's',        -3: 'sbar',
     4: 'c',        -4: 'cbar',
     5: 'b',        -5: 'bbar',
     6: 't',        -6: 'tbar',
    21: 'g',
    22: 'gamma',
    23: 'Z',
    24: 'W+',      -24: 'W-',
    25: 'H',
   111: 'pi0',
   211: 'pi+',    -211: 'pi-',
   221: 'eta',
   311: 'K0',     -311: 'K0bar',
   321: 'K+',     -321: 'K-',
   411: 'D+',     -411: 'D-',
   421: 'D0',     -421: 'D0bar',
   431: 'Ds+',    -431: 'Ds-',
   511: 'B0',     -511: 'B0bar',
   521: 'B+',     -521: 'B-',
   531: 'Bs0',    -531: 'Bs0bar',
   541: 'Bc+',    -541: 'Bc-',
   513: 'B*0',    -513: 'B*0bar',
   523: 'B*+',    -523: 'B*-',
   533: 'Bs*0',   -533: 'Bs*0bar',
   543: 'Bc*+',   -543: 'Bc*-',
}


def pdg_name(pdgId):
    """Return a human-readable label for a PDG particle ID ."""
    return PDG_NAMES.get(int(pdgId), f'#{int(pdgId):+d}')


def _energy(pt, eta, mass):
    """Compute energy from pT, eta, mass: E = sqrt((pT*cosh(eta))^2 + m^2)."""
    return math.sqrt((pt * math.cosh(eta))**2 + mass**2)


def print_event(selev, ievt=0):
    """Print gen-level b-quark four-vectors, children, and status flags for one event.

    Designed for FSR studies: b-quarks from H/Z whose children include gluons
    (pdgId=21) are candidates where FSR has been radiated.

    Args:
        selev: Selected event array with bfrom_Z_or_H already set.
        ievt:  Index within selev of the event to print (default 0).
    """
    b_all = selev.bfrom_Z_or_H
    nb = int(ak.num(b_all)[ievt])

    # Convert everything to plain Python lists at the event level so we can
    # iterate cleanly without worrying about awkward-array indexing quirks.
    pdgIds        = ak.to_list(b_all.pdgId[ievt])
    parent_pdgIds = ak.to_list(b_all.parent.pdgId[ievt])
    pts           = ak.to_list(b_all.pt[ievt])
    etas          = ak.to_list(b_all.eta[ievt])
    phis          = ak.to_list(b_all.phi[ievt])
    masses        = ak.to_list(b_all.mass[ievt])
    statuses      = ak.to_list(b_all.status[ievt])
    flags_evt     = {f: ak.to_list(b_all.hasFlags([f])[ievt]) for f in FSR_FLAG_NAMES}

    # Children (doubly nested: b-quarks × children)
    ch_pdgIds    = ak.to_list(b_all.children.pdgId[ievt])
    ch_pts       = ak.to_list(b_all.children.pt[ievt])
    ch_etas      = ak.to_list(b_all.children.eta[ievt])
    ch_phis      = ak.to_list(b_all.children.phi[ievt])
    ch_masses    = ak.to_list(b_all.children.mass[ievt])
    ch_statuses  = ak.to_list(b_all.children.status[ievt])
    ch_flags_evt = {f: ak.to_list(b_all.children.hasFlags([f])[ievt]) for f in FSR_FLAG_NAMES}

    print(f"\n{'='*72}")
    print(f"  Event {ievt}:  {nb} b-quarks from H/Z decays")
    print(f"{'='*72}")

    for i in range(nb):
        active_flags = [f for f in FSR_FLAG_NAMES if flags_evt[f][i]]
        E_parent = _energy(pts[i], etas[i], masses[i])
        print(f"\n  b[{i}]  {pdg_name(pdgIds[i]):<8s}  parent={pdg_name(parent_pdgIds[i]):<8s}  "
              f"status={statuses[i]}")
        print(f"         pt={pts[i]:8.2f}  eta={etas[i]:+7.3f}  "
              f"phi={phis[i]:+7.3f}  E={E_parent:8.2f} GeV")
        print(f"         flags: {', '.join(active_flags) if active_flags else 'none'}")

        nch = len(ch_pdgIds[i])
        print(f"         {nch} child(ren):")
        for j in range(nch):
            ch_active = [f for f in FSR_FLAG_NAMES if ch_flags_evt[f][i][j]]
            E_child = _energy(ch_pts[i][j], ch_etas[i][j], ch_masses[i][j])
            frac = E_child / E_parent if E_parent > 0 else 0.0
            print(f"           [{j}] {pdg_name(ch_pdgIds[i][j]):<8s}  status={ch_statuses[i][j]}  "
                  f"E_frac={frac:5.1%}")
            print(f"                pt={ch_pts[i][j]:8.2f}  eta={ch_etas[i][j]:+7.3f}  "
                  f"phi={ch_phis[i][j]:+7.3f}  E={E_child:8.2f} GeV")
            print(f"                flags: {', '.join(ch_active) if ch_active else 'none'}")


def filling_FSR_recovery_histograms(
        selev,
        JCM,
        processName: str = None,
        year: str = 'UL18',
        isMC: bool = False,
        histCuts: list = [],
        apply_FvT: bool = False,
        run_SvB: bool = False,
        top_reconstruction: bool = False,
        isDataForMixed: bool = False,
        tag_list: list = ["threeTag", "fourTag"],
        run_dilep_ttbar_crosscheck: bool = False,
        event_metadata: dict = {},
        weight_name = "weight"
        ):

    fill = Fill(process=processName, year=year, weight=weight_name)

    hist = Collection(
        process=[processName],
        year=[year],
        tag=["fourTag"],
        region=['SR', "SB"],
        **dict((s, ...) for s in histCuts)
    )

    # fill += hist.add("trigWeight", (40, 0, 2, ("trigWeight", 'Trigger weight')), weight='no_weight')

    fill += QuadJetHistsSelected(("quadJet_selected", "Selected Quad Jet"), "quadJet_selected")
    #fill += QuadJetHistsMinDr(("quadJet_min_dr", "Min dR Quad Jet"), "quadJet_min_dr")


    fill += hist.add("gen_b_fsr_frac_1", (100, 0, 2, ("b_hard_fsr_frac", 'FSR Frac')) )
    fill += hist.add("gen_b_fsr_frac_2", (100, 0, 2, ("b_chain_fsr_frac", 'FSR Frac')) )
    fill += hist.add("gen_b_fsr_frac_gluon", (50, 0, 1, ("b_hard_fsr_frac_gluon", 'Gluon FSR pT Frac')))

    # Truth info

    # fill histograms
    fill(selev, hist)
    return hist.to_dict(nonempty=True)



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

    Args:
        selev: Selected event array with bfrom_Z_or_H already set.

    Returns:
        selev with FSR fields added.
    """
    # Hard-process b-quarks: isFirstCopy = direct H daughters (4 per HH->4b event)
    b_hard = selev.bfrom_Z_or_H[selev.bfrom_Z_or_H.hasFlags(['isFirstCopy'])]

    # Mask over children: True where child is a gluon (FSR emission) within the jet cone.
    # The dR < 0.8 cut removes ghost children that are geometrically misattributed
    # from other decay vertices in the gen record.
    dR = b_hard.delta_r(b_hard.children)                              # events x b x children
    gluon_mask = (np.abs(b_hard.children.pdgId) == 21) & (dR < 0.8)  # events x b x children

    # Per b-quark: did any gluon child appear?
    b_hard_has_fsr = ak.any(gluon_mask, axis=2)        # events x b (bool)

    # Per b-quark: FSR pT fraction = sum(pT_gluon children) / pT_b
    pt_b        = b_hard.pt                                              # events x b
    pt_children = b_hard.children.pt                                     # events x b x ch
    pt_fsr      = ak.sum(ak.where(gluon_mask, pt_children, 0.0), axis=2) # events x b
    b_hard_fsr_frac = ak.where(pt_b > 0, pt_fsr / pt_b, ak.zeros_like(pt_b))

    # Per-event summary quantities
    evt_has_fsr      = ak.any(b_hard_has_fsr, axis=1)
    evt_max_fsr_frac = ak.fill_none(ak.max(b_hard_fsr_frac, axis=1), 0.0)

    selev['b_hard_has_fsr']   = b_hard_has_fsr
    selev['b_hard_fsr_frac']  = b_hard_fsr_frac
    selev['evt_has_fsr']      = evt_has_fsr
    selev['evt_max_fsr_frac'] = evt_max_fsr_frac

    return selev


def compute_fsr_energy_loss(selev, fsr_threshold=0.20):
    """Compute FSR pT loss by matching isFirstCopy to isLastCopy b-quarks.

    For each hard-process b-quark (isFirstCopy), the corresponding final-state
    b-quark (isLastCopy) is found by nearest-ΔR matching within same-charge
    pairs (b matched to b, bbar to bbar).

    FSR pT fraction = (pT_first - pT_last) / pT_first

    Particles that are both isFirstCopy and isLastCopy (no FSR) self-match at
    ΔR=0 and give fsr_frac=0.  Negative values are possible if the last copy
    is harder than the first (recoil effects) and are left unclipped.

    Adds to selev:
        b_chain_fsr_frac      (jagged float, events x 4): per-b FSR pT fraction
        b_chain_has_fsr       (jagged bool,  events x 4): fsr_frac > fsr_threshold
        evt_chain_max_fsr_frac (flat float,  events):     max across b-quarks

    Args:
        selev:         Selected event array with bfrom_Z_or_H already set.
        fsr_threshold: pT_frac threshold for the boolean flag (default 0.20).

    Returns:
        selev with chain-FSR fields added.
    """
    b_all = selev.bfrom_Z_or_H

    # b_first kept in original gen-record order so it aligns with b_hard_has_fsr
    b_first = b_all[b_all.hasFlags(['isFirstCopy'])]
    b_last  = b_all[b_all.hasFlags(['isLastCopy'])]

    # Build separate last-copy pools by charge for correct matching
    b_last_b    = b_last[b_last.pdgId > 0]
    b_last_bbar = b_last[b_last.pdgId < 0]

    # Match each b_first to its same-charge last copy by computing nearest()
    # against each charge pool, then selecting by pdgId sign.  This preserves
    # the original ordering of b_first (unlike split→match→concat).
    matched_last_b    = b_first.nearest(b_last_b)    # nearest b    (for every b_first)
    matched_last_bbar = b_first.nearest(b_last_bbar) # nearest bbar (for every b_first)

    pt_first    = b_first.pt
    pt_last_b    = ak.fill_none(matched_last_b.pt,    0.0)
    pt_last_bbar = ak.fill_none(matched_last_bbar.pt, 0.0)

    # Pick the match that corresponds to the same charge as b_first
    pt_last = ak.where(b_first.pdgId > 0, pt_last_b, pt_last_bbar)

    b_chain_fsr_frac       = ak.where(pt_first > 0, (pt_first - pt_last) / pt_first, 0.0)
    b_chain_has_fsr        = b_chain_fsr_frac > fsr_threshold
    evt_chain_max_fsr_frac = ak.fill_none(ak.max(b_chain_fsr_frac, axis=1), 0.0)

    selev['b_chain_fsr_frac']       = b_chain_fsr_frac
    selev['b_chain_has_fsr']        = b_chain_has_fsr
    selev['evt_chain_max_fsr_frac'] = evt_chain_max_fsr_frac

    return selev


def flag_fsr_gen_bjets(selev, fsr_threshold=0.10):
    """Flag gen b-jets with significant FSR energy loss.

    For each gen b-jet (GenJet with |partonFlavour|==5), find the nearest
    isFirstCopy b-quark via ΔR and compute:

        fsr_frac = (E_b_first - E_genJet) / E_b_first

    Positive fsr_frac: the gen jet is softer than the hard b-quark — energy was
    radiated outside the jet cone.
    Negative fsr_frac: the jet is harder — the FSR gluon was captured inside the
    jet cone.

    Args:
        selev:         Selected event array with bfrom_Z_or_H already set.
        fsr_threshold: E_frac threshold for the boolean flag (default 0.10).

    Adds to selev:
        sel_gen_bjets    (jagged GenJet, events x jets): b-flavored gen jets
        genjet_fsr_frac  (jagged float,  events x jets): per-jet FSR E_frac
        genjet_has_fsr   (jagged bool,   events x jets): fsr_frac > fsr_threshold
    """
    b_first = selev.bfrom_Z_or_H[selev.bfrom_Z_or_H.hasFlags(['isFirstCopy'])]

    # Gen b-jets: AK4 gen jets with b-quark parton flavour
    sel_gen_bjets = selev.GenJet[np.abs(selev.GenJet.partonFlavour) == 5]

    # For each gen b-jet, find its nearest isFirstCopy b-quark
    matched_b_first = sel_gen_bjets.nearest(b_first)

    # fill_none for the rare case of no b-quark match
    pt_b_first = ak.fill_none(matched_b_first.pt, 0.0)
    pt_genjet  = sel_gen_bjets.pt

    genjet_fsr_frac = ak.where(pt_b_first > 0, (pt_b_first - pt_genjet) / pt_b_first, 0.0)
    genjet_has_fsr  = genjet_fsr_frac > fsr_threshold

    selev['sel_gen_bjets']   = sel_gen_bjets
    selev['genjet_fsr_frac'] = genjet_fsr_frac
    selev['genjet_has_fsr']  = genjet_has_fsr

    return selev


class analysis(HH4bBaseProcessor):
    def __init__(
        self,
        **kwargs  # Accept additional arguments to pass to parent
    ):
        # Initialize parent without JCM (we'll handle it ourselves)
        super().__init__(**kwargs)



    def custom_processing(self, selev, config, selections, allcuts, nEventTot):
        logging.info(f"processor_FSR_recovery.custom_processing")

        #
        #  Make four tag cut
        #
        fourTag_sel = np.full(nEventTot, False)
        fourTag_sel[selections.all(*allcuts)] = selev.fourTag
        selections.add("fourTag", fourTag_sel)
        allcuts.append("fourTag")
        analysis_selections = selections.all(*allcuts)
        selev = selev[selev.fourTag]


        # Define low-pT jet selection criteria
        selev['Jet', 'selected_lowpt'] = (
            (selev.Jet.pt >= 15) &
            (np.abs(selev.Jet.eta) <= 3.0) &
            ~selev.Jet.pileup &
            (selev.Jet.jetId >= 2) &
            selev.Jet.lepton_cleaned &
            ~selev.Jet.tagged  # Exclude already selected jets
        )


        #
        #  Find b-quarks from H/Z decays and run truth-level FSR selection
        #
        selev['bfrom_Z_or_H'] = find_genpart(selev.GenPart, [5], [23, 25])
        selev = select_fsr_b_quarks(selev)
        selev = compute_fsr_energy_loss(selev)

        # For histogramming: mask non-FSR b-quarks to NaN so they flow into overflow
        selev['b_hard_fsr_frac_gluon'] = ak.where(selev.b_hard_has_fsr, selev.b_hard_fsr_frac, np.nan)

        # --- FSR gen-part summary for this chunk ---
        n_evt  = len(selev)
        n_b    = int(ak.sum(ak.num(selev.b_chain_fsr_frac)))  # total b-quarks (should be 4*n_evt)
        print(f"\n{'─'*60}")
        print(f"  FSR gen-part summary  ({n_evt} fourTag events, {n_b} isFirstCopy b-quarks)")
        print(f"{'─'*60}")

        m1        = ak.flatten(selev.b_hard_has_fsr)    # gluon-child flag (fixed)
        frac_chain = ak.flatten(selev.b_chain_fsr_frac) # continuous chain value

        def pct(n): return 100 * n / max(n_b, 1)

        print(f"\n  Threshold scan on chain pT_frac:")
        print(f"  {'Threshold':>10s}  {'Both':>10s}  {'Gluon only':>10s}  "
              f"{'Chain only':>10s}  {'Neither':>10s}")
        print(f"  {'─'*56}")
        for thr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
            m2 = frac_chain > thr
            nb  = int(ak.sum( m1 &  m2))
            ngo = int(ak.sum( m1 & ~m2))
            nco = int(ak.sum(~m1 &  m2))
            nn  = int(ak.sum(~m1 & ~m2))
            print(f"  {thr:>9.0%}  "
                  f"{nb:5d}({pct(nb):4.1f}%)  "
                  f"{ngo:5d}({pct(ngo):4.1f}%)  "
                  f"{nco:5d}({pct(nco):4.1f}%)  "
                  f"{nn:5d}({pct(nn):4.1f}%)")
        print(f"{'─'*60}")

        # --- Children of chain-only b-quarks (chain FSR, no gluon child seen) ---
        # These are the cases we want to understand: what causes the energy loss?
        chain_only_mask = selev.b_chain_has_fsr & ~selev.b_hard_has_fsr
        n_chain_only    = int(ak.sum(ak.flatten(chain_only_mask)))
        b_first = selev.bfrom_Z_or_H[selev.bfrom_Z_or_H.hasFlags(['isFirstCopy'])]
        b_chain_only = b_first[chain_only_mask]

        ch_pdgId = b_chain_only.children.pdgId                         # events x b x children
        ch_dR    = b_chain_only.delta_r(b_chain_only.children)         # events x b x children

        # Count children by |pdgId| (collapsed over events and b-quarks)
        from collections import Counter
        pdgid_flat = ak.to_list(ak.flatten(ch_pdgId, axis=None))
        counts = Counter(abs(x) for x in pdgid_flat)

        print(f"\n  Children of chain-only b-quarks ({n_chain_only} b-quarks, "
              f"{len(pdgid_flat)} children total):")
        print(f"  {'Particle':<12s}  {'|pdgId|':>8s}  {'Count':>7s}  {'Fraction':>8s}")
        print(f"  {'─'*42}")
        for pid, cnt in sorted(counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {pdg_name(pid):<12s}  {pid:8d}  {cnt:7d}  {100*cnt/max(len(pdgid_flat),1):7.1f}%")

        # How many of those children are gluons beyond the dR < 0.8 cone?
        n_gluon_out = int(ak.sum(ak.flatten((np.abs(ch_pdgId) == 21) & (ch_dR >= 0.8), axis=None)))
        n_gluon_in  = int(ak.sum(ak.flatten((np.abs(ch_pdgId) == 21) & (ch_dR <  0.8), axis=None)))
        print(f"\n  Gluon children in chain-only b-quarks:")
        print(f"    Within dR<0.8: {n_gluon_in}   Beyond dR>0.8: {n_gluon_out}")
        print(f"{'─'*60}\n")

        # Print a few events for manual inspection of the gen-level structure
        n_print = min(5, len(selev))
        for ievt in range(n_print):
            print_event(selev, ievt)

        return selev, selections.all(*allcuts)


    def histograms(self, event, selev, weights, analysis_selections, shift_name):
        """Fill histograms for analysis.

        Requires chunk-scoped variables: processName, year, config
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
            selev: Selected events array
            weights: Weights object
            analysis_selections: Boolean mask for analysis selection
            shift_name: Name of systematic shift (None for nominal)

        Returns:
            Dictionary with histogram outputs
        """
        print("\n processor_FSR_recovery.histogram \n")


        # Need to figure this out...

        return filling_FSR_recovery_histograms(selev,
                                               self.jcm_model,
                                               processName=self.processName,
                                               year=self.year,
                                               isMC=self.config["isMC"],
                                               histCuts=self.histCuts,
                                               apply_FvT=self.apply_FvT,
                                               run_SvB=self.run_SvB,
                                               run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                                               top_reconstruction=self.top_reconstruction,
                                               isDataForMixed=self.config['isDataForMixed'],
                                               event_metadata=event.metadata
                                               )
