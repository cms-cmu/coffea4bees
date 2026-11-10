import time
import awkward as ak
import numpy as np
import yaml
import warnings
from collections import OrderedDict

import fsspec
from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import vector as coffea_vector

from src.physics.event_selection import apply_event_selection
from src.physics.objects.jet_corrections import apply_jet_veto_maps
from src.hist_tools import Template, Collection, Fill
from src.hist_tools.object import LorentzVector, Jet, PFCand
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.analysis.helpers import synthetic_movement as sm

import logging

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

#
# Stored particleNetWithMass_HbbvsQCD cut per era, on the pT-leading and pT-subleading jet of the pair
# (65th percentile of each, i.e. 35 % kept, on the 2022-2023 data)
#
HBB_CUTS = {
    "2022_preEE":   (0.0148773193359375,  0.01141357421875),
    "2022_EE":      (0.019744873046875,   0.015960693359375),
    "2023_preBPix": (0.011566162109375,   0.01141357421875),
    "2023_BPix":    (0.00988006591796875, 0.01026153564453125),
}

BDT_MODEL = "root://cmseos.fnal.gov//store/user/leilei01/XX4b/bdt_models/25Feb5_v13_glopartv2_rawmass/trained_bdt.ubj"

#
# BDT: LPC-HH/HH4b 25Feb5_v13_glopartv2_rawmass, 20 inputs in the training order (v13_glopartv2.bdt_dataframe)
#
BDT_TXBB_EDGES = np.array([0.0, 0.8, 0.9, 0.94, 0.97, 0.99, 1.0])
BDT_PAD_VALUE = -99999.0

RANKS = ("lead", "subl")   # jet1 / jet2 = higher / lower stored T_Xbb
JET_ARRAYS = ["orig_idx", "pt", "eta", "phi", "msd", "hbb", "txbb", "mreg",
              "msd_syn", "m_subjetsum", "p_new_pt", "p_new_eta", "p_new_phi",
              "m_pfsum", "m_pfsum_moved", "m_pfsum_puppi", "m_pfsum_puppi_moved",
              "tau21_i0", "tau21_i1", "nbj"] + \
             [f"{sj}_{f}" for sj in ("i0", "i1", "i0_new", "i1_new") for f in ("pt", "eta", "phi", "mass")]
ARRAY_KEYS = ["evt_run", "evt_lumi", "evt_event", "npool", "bdt_ggf", "bdt_vbf"] + \
             [f"{k}_{rk}" for rk in RANKS for k in JET_ARRAYS]

_PDF_CACHE = {}
_BDT_CACHE = {}


def load_clustering_pdfs(path):
    if path not in _PDF_CACHE:
        with open(path) as f:
            _PDF_CACHE[path] = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    return _PDF_CACHE[path]


def ak8_tight_id(fatjets):
    if "jetId" in fatjets.fields:
        return fatjets.isTight
    abseta = np.abs(fatjets.eta)
    return (
        ((abseta <= 2.6) & (fatjets.neHEF < 0.99) & (fatjets.neEmEF < 0.9)
         & ((fatjets.chMultiplicity + fatjets.neMultiplicity) > 1)
         & (fatjets.chHEF > 0.01) & (fatjets.chMultiplicity > 0))
        | (((abseta > 2.6) & (abseta <= 2.7)) & (fatjets.neHEF < 0.90) & (fatjets.neEmEF < 0.99))
        | (((abseta > 2.7) & (abseta <= 3.0)) & (fatjets.neHEF < 0.99))
        | ((abseta > 3.0) & (fatjets.neMultiplicity >= 2) & (fatjets.neEmEF < 0.4))
    )


def stored_txbb_mreg(fatjets):
    """GloParT-v2 T_Xbb and regressed mass stored in PFNano, as LPC-HH/HH4b objects.py defines them
    (massVis is a ratio to the raw jet mass)."""
    fields = set(fatjets.fields)
    txbb_branch = next((n for n in ("globalParT2_XbbVsQCD", "globalParT_XbbVsQCD", "ParTTXbb") if n in fields), None)
    if txbb_branch is None:
        raise RuntimeError("no stored GloParT T_Xbb branch (FatJet_globalParT2_XbbVsQCD)")
    ratio_branch = next((n for n in ("globalParT2_massVis", "globalParT_massVis") if n in fields), None)
    if ratio_branch is not None:
        mreg = fatjets[ratio_branch] * (1.0 - fatjets.rawFactor) * fatjets.mass
    elif "ParTmassVis" in fields:
        mreg = fatjets.ParTmassVis
    else:
        raise RuntimeError("no stored GloParT regressed-mass branch (FatJet_globalParT2_massVis)")
    return ak.values_astype(fatjets[txbb_branch], np.float64), ak.values_astype(mreg, np.float64)


def jet_can_be_declustered(ev, jet):
    """jet has shape (events, 1): two distinct subjets in range, at least one PF candidate,
    PF indices in range and SV indices either -1 or in range."""
    nsub = ak.to_numpy(ak.num(ev.SubJet, axis=1))
    i1 = ak.to_numpy(jet.subJetIdx1[:, 0])
    i2 = ak.to_numpy(jet.subJetIdx2[:, 0])
    ok = (i1 >= 0) & (i2 >= 0) & (i1 < nsub) & (i2 < nsub) & (i1 != i2)
    slot = ak.firsts(jet.orig_idx)
    n_pf = ak.num(ev.PFCands, axis=1)
    pf = ev.FatJetPFCands[ev.FatJetPFCands.jetIdx == slot].pFCandsIdx
    ok &= ak.to_numpy(ak.num(pf, axis=1)) > 0
    ok &= ak.to_numpy(ak.all((pf >= 0) & (pf < n_pf), axis=1))
    n_sv = ak.num(ev.SV, axis=1)
    sv = ev.FatJetSVs[ev.FatJetSVs.jetIdx == slot].sVIdx
    ok &= ak.to_numpy(ak.all((sv == -1) | ((sv >= 0) & (sv < n_sv)), axis=1))
    return np.asarray(ok, dtype=bool)


def rank_by_stored_txbb(ev, pool):
    """Pool jets ordered by stored T_Xbb, highest first (ties keep the pool order).  A jet that cannot be
    declustered, or whose stored values are not finite, gets T_Xbb = -inf and sorts last."""
    npool = np.asarray(ak.to_numpy(ak.num(pool, axis=1)), dtype=np.int64)
    width = int(npool.max()) if len(npool) else 0
    txbb = np.full((len(ev), width), -np.inf, dtype=np.float64)
    mreg = np.full((len(ev), width), np.nan, dtype=np.float64)
    for k in range(width):
        idx = np.flatnonzero(npool > k)
        jet_k = pool[npool > k][:, k:k + 1]
        good = jet_can_be_declustered(ev[npool > k], jet_k)
        if not good.any():
            continue
        tx, m = stored_txbb_mreg(jet_k[good])
        tx = np.asarray(ak.to_numpy(ak.flatten(tx, axis=1)), dtype=np.float64)
        m = np.asarray(ak.to_numpy(ak.flatten(m, axis=1)), dtype=np.float64)
        finite = np.isfinite(tx) & np.isfinite(m)
        txbb[idx[good], k] = np.where(finite, tx, -np.inf)
        mreg[idx[good], k] = np.where(finite, m, np.nan)
    keep = np.arange(width)[None, :] < npool[:, None]
    txbb = ak.unflatten(txbb[keep], npool)
    pool = ak.with_field(pool, txbb, "txbb")
    pool = ak.with_field(pool, ak.unflatten(mreg[keep], npool), "mreg")
    return pool[ak.argsort(txbb, axis=1, ascending=False, stable=True)]


def pf_index_validity(ev):
    """PF and SV indices of the two selected jets in range, and no PF or SV shared by the two jets."""
    n_pf = ak.num(ev.PFCands, axis=1)
    n_sv = ak.num(ev.SV, axis=1)
    ok = np.ones(len(ev), dtype=bool)
    pf_sets, sv_sets = [], []
    for i in (0, 1):
        pf = ev.FatJetPFCands[ev.FatJetPFCands.jetIdx == ev.selFatJet.orig_idx[:, i]].pFCandsIdx
        ok &= ak.to_numpy(ak.all((pf >= 0) & (pf < n_pf), axis=1))
        pf_sets.append(pf)
        sv = ev.FatJetSVs[ev.FatJetSVs.jetIdx == ev.selFatJet.orig_idx[:, i]].sVIdx
        ok &= ak.to_numpy(ak.all((sv == -1) | ((sv >= 0) & (sv < n_sv)), axis=1))
        sv_sets.append(sv[sv >= 0])
    for sets in (pf_sets, sv_sets):
        both = ak.sort(ak.concatenate([sets[0], sets[1]], axis=1), axis=1)
        shared = ak.to_numpy(ak.fill_none(ak.any(both[:, 1:] == both[:, :-1], axis=1), False))
        ok &= ~(shared & ak.to_numpy(ak.num(both, axis=1) > 1))
    return ok


def get_met(events):
    if "MET" in events.fields:
        return events.MET
    if "PuppiMET" in events.fields:
        return events.PuppiMET
    raise RuntimeError("neither MET nor PuppiMET is present")


def bdt_features(selev, txbb_pair, mreg_pair):
    """The 20 BDT inputs: FatJet pT/eta/phi with the stored T_Xbb and regressed mass,
    H1 = the jet with the higher T_Xbb."""
    beh = coffea_vector.behavior
    tx_ld, tx_sl = (np.asarray(t, dtype=np.float64) for t in txbb_pair)
    m_ld, m_sl = (np.asarray(m, dtype=np.float64) for m in mreg_pair)
    lead_is_h1 = tx_ld >= tx_sl

    def pick(a_ld, a_sl, first):
        return np.where(lead_is_h1, a_ld, a_sl) if first else np.where(lead_is_h1, a_sl, a_ld)

    def vec(pt, eta, phi, mass):
        return ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": mass},
                      with_name="PtEtaPhiMLorentzVector", behavior=beh)

    def padvec(j):
        return vec(ak.fill_none(j.pt, BDT_PAD_VALUE), ak.fill_none(j.eta, BDT_PAD_VALUE),
                   ak.fill_none(j.phi, BDT_PAD_VALUE), ak.fill_none(j.mass, BDT_PAD_VALUE))

    def np64(x):
        return np.asarray(ak.to_numpy(x), dtype=np.float64)

    sfj = selev.selFatJet
    kin = {f + tag: np64(getattr(sfj[:, i], f)) for tag, i in (("ld", 0), ("sl", 1)) for f in ("pt", "eta", "phi")}
    h1 = vec(pick(kin["ptld"], kin["ptsl"], True), pick(kin["etald"], kin["etasl"], True),
             pick(kin["phild"], kin["phisl"], True), pick(m_ld, m_sl, True))
    h2 = vec(pick(kin["ptld"], kin["ptsl"], False), pick(kin["etald"], kin["etasl"], False),
             pick(kin["phild"], kin["phisl"], False), pick(m_ld, m_sl, False))
    hh = h1 + h2

    ak4 = selev.Jet[(selev.Jet.pt > 15) & (np.abs(selev.Jet.eta) < 4.7)
                    & selev.Jet.isTight & selev.Jet.isTightLeptonVeto]
    electrons = selev.Electron[selev.Electron.pt > 5]
    muons = selev.Muon[selev.Muon.pt > 7]
    lep_clean = (ak.all(ak4.metric_table(electrons) > 0.4, axis=2)
                 & ak.all(ak4.metric_table(muons) > 0.4, axis=2))
    vbf_sel = (ak4.pt >= 25) & (np.abs(ak4.eta) <= 4.7) & (ak4.delta_r(h1) > 1.2) & (ak4.delta_r(h2) > 1.2) & lep_clean
    vbf = ak.pad_none(ak4[vbf_sel], 2, axis=1, clip=True)
    vbf1, vbf2 = padvec(vbf[:, 0]), padvec(vbf[:, 1])
    away = ak4[(ak4.pt >= 25) & (np.abs(ak4.eta) <= 2.5) & (ak4.delta_r(h1) > 0.9) & (ak4.delta_r(h2) > 0.9) & lep_clean]

    def nearest(hvec):
        return padvec(ak.firsts(away[ak.argmin(away.delta_r(hvec), axis=1, keepdims=True)]))

    away1, away2 = nearest(h1), nearest(h2)

    f32max = float(np.finfo(np.float32).max)
    t32 = [np.nan_to_num(np64(sfj["tau3"][:, i] / sfj["tau2"][:, i]), nan=-1.0, posinf=f32max, neginf=-f32max)
           for i in (0, 1)]
    hh_mass, h1_pt, h2_pt = np64(hh.mass), np64(h1.pt), np64(h2.pt)
    cols = [
        np64(hh.pt),                                 # HHPt
        np64(hh.eta),                                # HHeta
        hh_mass,                                     # HHmass
        np64(get_met(selev).pt),                     # MET
        pick(t32[0], t32[1], True),                  # H1T32
        pick(t32[0], t32[1], False),                 # H2T32
        pick(m_ld, m_sl, True),                      # H1Mass
        h1_pt,                                       # H1Pt
        h2_pt,                                       # H2Pt
        np64(h1.eta),                                # H1eta
        np.clip(np.digitize(pick(tx_ld, tx_sl, True), BDT_TXBB_EDGES, right=False), 1, 6).astype(np.float64),  # H1Xbb
        h1_pt / hh_mass,                             # H1Pt_HHmass
        h2_pt / hh_mass,                             # H2Pt_HHmass
        h1_pt / h2_pt,                               # H1Pt/H2Pt
        np64((vbf1 + vbf2).mass),                    # VBFjjMass
        np.abs(np64(vbf1.eta) - np64(vbf2.eta)),     # VBFjjDeltaEta
        np64(h1.delta_r(away1)),                     # H1AK4JetAway1dR
        np64(h2.delta_r(away2)),                     # H2AK4JetAway2dR
        np64((h1 + away1).mass),                     # H1AK4JetAway1mass
        np64((h2 + away2).mass),                     # H2AK4JetAway2mass
    ]
    return np.column_stack([np.asarray(c, dtype=np.float32) for c in cols])


def bdt_scores(model_path, features):
    """BDT_ggF and BDT_VBF (classes ggF, VBF, QCD, ttbar).  The model is the .ubj conversion: the legacy
    trained_bdt.model cannot be read by xgboost >= 3.1.  Read through fsspec, so a root:// path works."""
    import xgboost as xgb
    bdt = _BDT_CACHE.get(model_path)
    if bdt is None:
        bdt = xgb.Booster()
        with fsspec.open(model_path, "rb") as f:
            bdt.load_model(bytearray(f.read()))
        _BDT_CACHE[model_path] = bdt
    preds = bdt.predict(xgb.DMatrix(np.asarray(features, dtype=np.float32)))
    bdt_ggf = preds[:, 0] / (preds[:, 0] + preds[:, 2] + preds[:, 3])
    bdt_vbf = preds[:, 1] / (preds[:, 1] + preds[:, 2] + preds[:, 3])
    return bdt_ggf, bdt_vbf


def pf_mass_sum(pf, use_puppi_weight=False):
    """Invariant mass of the summed PF-candidate four-vectors of each fat jet ([event][jet][candidate] in)."""
    pf_pt = ak.values_astype(pf.pt, np.float64)
    pf_eta = ak.values_astype(pf.eta, np.float64)
    pf_phi = ak.values_astype(pf.phi, np.float64)
    pf_mass = ak.values_astype(pf.mass, np.float64)
    w = ak.values_astype(pf.puppiWeight, np.float64) if use_puppi_weight else 1.0
    pf_px = w * pf_pt * np.cos(pf_phi)
    pf_py = w * pf_pt * np.sin(pf_phi)
    pf_pz = w * pf_pt * np.sinh(pf_eta)
    pf_E = w * np.sqrt(pf_pt ** 2 * np.cosh(pf_eta) ** 2 + pf_mass ** 2)
    return np.sqrt(np.maximum(ak.sum(pf_E, axis=2) ** 2 - ak.sum(pf_px, axis=2) ** 2
                              - ak.sum(pf_py, axis=2) ** 2 - ak.sum(pf_pz, axis=2) ** 2, 0.0))


class FatJetHists(Template):
    p = LorentzVector.plot_pair(("...", r"Fat Jet"), "p", skip=["n"],
                                bins={"mass": (100, 40, 300), "pt": (60, 250, 1000), "dr": (50, 0, 1.2), "dphi": (50, -1.5, 1.5)})
    p_new = Jet.plot(("...", r"declustered fatjet"), "p_new", skip=["deepjet_b", "deepjet_c", "id_jet", "id_pileup", "n"], bins={"pt": (60, 250, 1000), "mass": (100, 40, 300)})
    i0 = Jet.plot(("...", r"subjet 0"), "i0", skip=["deepjet_b", "deepjet_c", "id_jet", "id_pileup", "n"], bins={"mass": (100, 0, 200), "pt": (50, 100, 1000)})
    i1 = Jet.plot(("...", r"subjet 1"), "i1", skip=["deepjet_b", "deepjet_c", "id_jet", "id_pileup", "n"], bins={"mass": (50, 0, 100), "pt": (50, 100, 1000)})
    i0_new = Jet.plot(("...", r"declustered subjet 0"), "i0_new", skip=["deepjet_b", "deepjet_c", "id_jet", "id_pileup", "n"], bins={"mass": (100, 0, 200), "pt": (50, 100, 1000)})
    i1_new = Jet.plot(("...", r"declustered subjet 1"), "i1_new", skip=["deepjet_b", "deepjet_c", "id_jet", "id_pileup", "n"], bins={"mass": (50, 0, 100), "pt": (50, 100, 1000)})
    pf = PFCand.plot(("...", r"PFCands in selected fat jet"), "PFCands", skip=["pz", "energy"], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})
    pf_moved = PFCand.plot(("...", r"Moved PFCands in selected fat jet"), "PFCands_moved", skip=["pz", "energy"], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})


class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            corrections_metadata: dict = None,
            clustering_pdfs_file: str = "coffea4bees/jet_clustering/jet-splitting-PDFs-PFCands/clustering_pdfs_vs_pT_XXX.yml",
            declustering_rand_seed: int = 5,
            bdt_model_path: str = BDT_MODEL,
            fatjet_pt_cuts: list = (300, 250),
            msoftdrop_cut: float = 40,
            hbb_cuts: dict = None,
            **kwargs
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata
        self.clustering_pdfs_file = clustering_pdfs_file
        self.declustering_rand_seed = declustering_rand_seed
        self.bdt_model_path = bdt_model_path
        if not bdt_model_path:
            logging.warning("bdt_model_path is not set: bdt_ggf and bdt_vbf are written as -1")
        self.fatjet_pt_cuts = tuple(fatjet_pt_cuts)
        self.msoftdrop_cut = msoftdrop_cut
        self.hbb_cuts = HBB_CUTS if hbb_cuts is None else hbb_cuts

    def jet_veto_map(self, event, year):
        if "jet_veto_maps" not in self.corrections_metadata[year]:
            logging.warning("jet_veto_maps missing; skipping veto.")
            return np.full(len(event), True)
        ak4 = event.Jet[(event.Jet.pt > 15) & (np.abs(event.Jet.eta) < 4.7)
                        & event.Jet.isTight & event.Jet.isTightLeptonVeto]
        per_jet_ok = apply_jet_veto_maps(self.corrections_metadata[year]["jet_veto_maps"], ak4, event_veto=True)
        return ak.to_numpy(ak.all(per_jet_ok, axis=1))

    def lepton_veto(self, event):
        mu = event.Muon
        n_mu = ak.sum((mu.pt >= 10) & (np.abs(mu.eta) < 2.4) & mu.looseId & (mu["pfRelIso04_all"] < 0.25), axis=1)
        if "Electron" in event.fields:
            el = event.Electron
            n_el = ak.sum((el.pt >= 20) & (np.abs(el.eta) < 2.5) & (el.cutBased >= 2) & (el["miniPFRelIso_all"] < 0.4), axis=1)
        else:
            n_el = np.zeros(len(event), dtype=np.int64)
        return ak.to_numpy((n_mu == 0) & (n_el == 0))

    def process(self, event):

        ### Some useful variables
        tstart = time.time()
        fname   = event.metadata['filename']
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        processName = event.metadata['processName']
        isMC    = True if len(event) and event.run[0] == 1 else False
        nEvent = len(event)

        logging.info(fname)
        logging.info(f'Process {nEvent} Events')

        clustering_pdfs = load_clustering_pdfs(self.clustering_pdfs_file.replace("XXX", year))
        hbb_cut_lead, hbb_cut_subl = self.hbb_cuts[year]

        #
        # Event selection
        #
        event = apply_event_selection( event,
                                        self.corrections_metadata[year],
                                        cut_on_lumimask=(not isMC),
                                        )
        event["passJetVetoMap"] = self.jet_veto_map(event, year)
        event["passLeptonVeto"] = self.lepton_veto(event)

        event["FatJet", "orig_idx"] = ak.local_index(event.FatJet, axis=1)
        event["FatJet", "isPool"] = (event.FatJet.pt > 250) & (np.abs(event.FatJet.eta) < 2.5) & ak8_tight_id(event.FatJet)
        event["passNFatJets"] = ak.sum(event.FatJet.isPool, axis=1) >= 2
        event["weight"] = 1.0

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( "passJetVetoMap", event.passJetVetoMap )
        selections.add( "passLeptonVeto", event.passLeptonVeto )
        selections.add( "passNFatJets", ak.to_numpy(event.passNFatJets) )

        cutflow = cutflow_4b()
        cutflow.fill( "all", event, allTag=True )
        list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "passJetVetoMap", "passLeptonVeto", "passNFatJets" ]
        for i, cut in enumerate(list_of_cuts):
            cutflow.fill( cut, event[selections.all(*list_of_cuts[:i + 1])], allTag=True )

        clev = event[selections.all(*list_of_cuts)]
        if len(clev) == 0:
            return self.empty_output(cutflow, dataset)

        #
        # jet1 / jet2: the two pool jets with the highest stored T_Xbb, cuts on the original jets
        #
        clev["selFatJet"] = rank_by_stored_txbb(clev, clev.FatJet[clev.FatJet.isPool])[:, 0:2]
        jet1, jet2 = clev.selFatJet[:, 0], clev.selFatJet[:, 1]

        jet2_harder = jet2.pt > jet1.pt
        hbb_pt_lead = ak.values_astype(ak.where(jet2_harder, jet2.particleNetWithMass_HbbvsQCD, jet1.particleNetWithMass_HbbvsQCD), np.float64)
        hbb_pt_subl = ak.values_astype(ak.where(jet2_harder, jet1.particleNetWithMass_HbbvsQCD, jet2.particleNetWithMass_HbbvsQCD), np.float64)

        valid, _ = sm.subjet_validity_report(clev, jet1, jet2)
        valid = valid & pf_index_validity(clev)
        for jet in (jet1, jet2):
            valid = valid & np.isfinite(ak.to_numpy(jet.txbb)) & np.isfinite(ak.to_numpy(jet.mreg))

        jet_cuts = OrderedDict({
            "passFatJetPt"  : (jet1.pt > self.fatjet_pt_cuts[0]) & (jet2.pt > self.fatjet_pt_cuts[1]),
            "passMsoftdrop" : (jet1.msoftdrop > self.msoftdrop_cut) & (jet2.msoftdrop > self.msoftdrop_cut),
            "passHbb"       : (hbb_pt_lead > hbb_cut_lead) & (hbb_pt_subl > hbb_cut_subl),
            "passValidity"  : valid,
        })
        passed = np.ones(len(clev), dtype=bool)
        for cut, mask in jet_cuts.items():
            passed &= np.asarray(ak.to_numpy(mask), dtype=bool)
            cutflow.fill( cut, clev[passed], allTag=True )

        selev = clev[passed]
        if len(selev) == 0:
            return self.empty_output(cutflow, dataset)

        #
        # BDT on the original jets
        #
        txbb_pair = [np.asarray(ak.to_numpy(selev.selFatJet.txbb[:, i]), dtype=np.float64) for i in (0, 1)]
        mreg_pair = [np.asarray(ak.to_numpy(selev.selFatJet.mreg[:, i]), dtype=np.float64) for i in (0, 1)]
        if self.bdt_model_path:
            bdt_ggf, bdt_vbf = bdt_scores(self.bdt_model_path, bdt_features(selev, txbb_pair, mreg_pair))
        else:
            bdt_ggf = bdt_vbf = np.full(len(selev), -1.0)

        #
        # Declustering: new subjets from the splitting PDFs, PF candidates moved with their subjet
        #
        selev, fatjets, _ = sm.build_two_jet_containers(selev)
        selev, fatjets, ok, _ = sm.decluster_two_jets(selev, fatjets, clustering_pdfs, self.declustering_rand_seed, chunk)
        ok = np.asarray(ok, dtype=bool)
        bdt_ggf, bdt_vbf = bdt_ggf[ok], bdt_vbf[ok]
        fatjets = sm.move_pfcands(fatjets)

        pf, pf_moved = fatjets.PFCands, fatjets.PFCands_moved
        has_puppi = "puppiWeight" in pf.fields
        fatjets = ak.with_field(fatjets, fatjets.i0_new.mass - fatjets.i0.mass, "d_mass_i0")
        fatjets = ak.with_field(fatjets, fatjets.i1_new.mass - fatjets.i1.mass, "d_mass_i1")
        fatjets = ak.with_field(fatjets, fatjets.p_new.mass - fatjets.p.mass, "d_mass_fatjet")
        fatjets = ak.with_field(fatjets, pf_moved.pt - pf.pt, "pf_dpt")
        fatjets = ak.with_field(fatjets, pf_moved.eta - pf.eta, "pf_deta")
        fatjets = ak.with_field(fatjets, sm.wrap_phi(pf_moved.phi - pf.phi), "pf_dphi")
        fatjets = ak.with_field(fatjets, pf_mass_sum(pf), "m_pfsum")
        fatjets = ak.with_field(fatjets, pf_mass_sum(pf_moved), "m_pfsum_moved")
        fatjets = ak.with_field(fatjets, pf_mass_sum(pf, True) if has_puppi else fatjets.m_pfsum * np.nan, "m_pfsum_puppi")
        fatjets = ak.with_field(fatjets, pf_mass_sum(pf_moved, True) if has_puppi else fatjets.m_pfsum * np.nan, "m_pfsum_puppi_moved")
        fatjets["p", "lead"] = fatjets.i0
        fatjets["p", "subl"] = fatjets.i1
        fatjets["p", "st"] = fatjets.i0.pt + fatjets.i1.pt
        fatjets["p", "dr"] = fatjets.i0.delta_r(fatjets.i1)
        fatjets["p", "dphi"] = fatjets.i0.delta_phi(fatjets.i1)

        selev["fatjets"] = fatjets
        selev["bdt_ggf"] = bdt_ggf
        selev["bdt_vbf"] = bdt_vbf

        #
        # Hists
        #
        selev["fourTag"] = True
        selev['tag'] = ak.zip({
            "fourTag": selev.fourTag,
        })
        selev["SR"] = True
        selev["region"] = ak.zip({
            "SR": selev.SR,
        })

        fill = Fill(process=processName, year=year, weight="weight")
        histCuts = ["passNFatJets"]
        hist = Collection( process=[processName],
                           year=[year],
                           tag=["fourTag"],  # 3 / 4/ Other
                           region=['SR'],  # SR / SB / Other
                           **dict((s, ...) for s in histCuts)
                           )

        selev["selFatJet", "btagScore"] = selev.selFatJet.particleNetWithMass_HbbvsQCD
        fill += Jet.plot(("fatJets", "Selected Fat Jets"), "selFatJet", skip=["deepjet_c", "id_jet", "id_pileup"], bins={"pt": (50, 0, 1000)})
        fill += hist.add( "msoftdrop",       (100, 0, 300, ("selFatJet.msoftdrop",     'Soft Drop Mass')))
        fill += hist.add( "msd_syn",         (100, 0, 300, ("fatjets.p_new.mass",      "Synthetic soft drop mass m(subjet0' + subjet1')")))
        fill += hist.add( "txbb",            (100, 0, 1,   ("selFatJet.txbb",          r"Stored GloParT $T_{Xbb}$")))
        fill += hist.add( "mass_regression", (100, 0, 300, ("selFatJet.mreg",          'Stored GloParT regressed mass')))
        fill += hist.add( "bdt_ggf",         (100, 0, 1,   ("bdt_ggf",                 r"BDT$_{ggF}$")))
        fill += hist.add( "bdt_vbf",         (100, 0, 1,   ("bdt_vbf",                 r"BDT$_{VBF}$")))
        fill += hist.add( "d_mass_i0",       (100, -50, 50, ("fatjets.d_mass_i0",      "subjet0 mass new-old")))
        fill += hist.add( "d_mass_i1",       (100, -50, 50, ("fatjets.d_mass_i1",      "subjet1 mass new-old")))
        fill += hist.add( "d_mass_fatjet",   (100, -50, 50, ("fatjets.d_mass_fatjet",  "m(subjet0 + subjet1) new-old")))
        fill += hist.add( "pf_dpt",          (100, -5.0, 5.0, ("fatjets.pf_dpt",       r"PF $\Delta p_T$ (moved - original)")))
        fill += hist.add( "pf_deta",         (100, -0.8, 0.8, ("fatjets.pf_deta",      r"PF $\Delta\eta$ (moved - original)")))
        fill += hist.add( "pf_dphi",         (100, -0.8, 0.8, ("fatjets.pf_dphi",      r"PF $\Delta\phi$ (moved - original)")))
        fill += hist.add( "m_pfsum",         (100, 0, 300, ("fatjets.m_pfsum",         "mass(sum PF 4-vectors)")))
        fill += hist.add( "m_pfsum_moved",   (100, 0, 300, ("fatjets.m_pfsum_moved",   "mass(sum moved PF 4-vectors)")))
        fill += FatJetHists(("declustering", r""), "fatjets")

        fill(selev, hist)

        processOutput = {}
        cutflow.addOutput(processOutput, dataset)
        processOutput["arrays"] = {dataset: {k: processor.column_accumulator(v)
                                             for k, v in self.event_arrays(selev, fatjets).items()}}

        logging.info(f"{chunk}{len(selev)} events declustered in {time.time() - tstart:.1f} s")
        output = hist.output | processOutput

        return output

    def event_arrays(self, selev, fatjets):
        """One row per event: the original jets, their BDT and the synthetic jets, for cuts applied later."""
        def f32(x):
            return np.asarray(ak.to_numpy(x), dtype=np.float32)

        arrays = {
            "evt_run":   np.asarray(ak.to_numpy(selev.run), dtype=np.float64),
            "evt_lumi":  np.asarray(ak.to_numpy(selev.luminosityBlock), dtype=np.float64),
            "evt_event": np.asarray(ak.to_numpy(selev.event), dtype=np.float64),
            "npool":     f32(ak.sum(selev.FatJet.isPool, axis=1)),
            "bdt_ggf":   f32(selev.bdt_ggf),
            "bdt_vbf":   f32(selev.bdt_vbf),
        }
        for i, rk in enumerate(RANKS):
            fj = selev.selFatJet[:, i]
            p, p_new = fatjets.p[:, i], fatjets.p_new[:, i]
            i0, i1 = fatjets.i0[:, i], fatjets.i1[:, i]
            arrays.update({
                f"orig_idx_{rk}":            f32(fj.orig_idx),
                f"pt_{rk}":                  f32(fj.pt),
                f"eta_{rk}":                 f32(fj.eta),
                f"phi_{rk}":                 f32(fj.phi),
                f"msd_{rk}":                 f32(fj.msoftdrop),
                f"hbb_{rk}":                 f32(fj.particleNetWithMass_HbbvsQCD),
                f"txbb_{rk}":                f32(fj.txbb),
                f"mreg_{rk}":                f32(fj.mreg),
                f"msd_syn_{rk}":             f32(p_new.mass),
                f"m_subjetsum_{rk}":         f32(p.mass),
                f"p_new_pt_{rk}":            f32(p_new.pt),
                f"p_new_eta_{rk}":           f32(p_new.eta),
                f"p_new_phi_{rk}":           f32(p_new.phi),
                f"m_pfsum_{rk}":             f32(fatjets.m_pfsum[:, i]),
                f"m_pfsum_moved_{rk}":       f32(fatjets.m_pfsum_moved[:, i]),
                f"m_pfsum_puppi_{rk}":       f32(fatjets.m_pfsum_puppi[:, i]),
                f"m_pfsum_puppi_moved_{rk}": f32(fatjets.m_pfsum_puppi_moved[:, i]),
                f"tau21_i0_{rk}":            f32(i0.tau21),
                f"tau21_i1_{rk}":            f32(i1.tau21),
                f"nbj_{rk}":                 f32(i0.jet_flavor == "bj") + f32(i1.jet_flavor == "bj"),
            })
            for sj in ("i0", "i1", "i0_new", "i1_new"):
                for f in ("pt", "eta", "phi", "mass"):
                    arrays[f"{sj}_{f}_{rk}"] = f32(getattr(fatjets[sj][:, i], f))
        return arrays

    def empty_output(self, cutflow, dataset):
        processOutput = {}
        cutflow.addOutput(processOutput, dataset)
        processOutput["arrays"] = {dataset: {k: processor.column_accumulator(np.zeros(0, dtype=np.float64 if k.startswith("evt_") else np.float32))
                                             for k in ARRAY_KEYS}}
        return processOutput

    def postprocess(self, accumulator):
        return accumulator
