from coffea4bees.analysis.helpers.hist_templates import (
    FvTHists,
    QuadJetHistsSelected,
    QuadJetHistsMinDr,
    QuadJetHistsSRSingle,
    SvBHists,
    TopCandHists,
    WCandHists,
)
from src.hist_tools import H, Template
from src.hist_tools import Collection, Fill
from src.hist_tools.object import Elec, Jet, LorentzVector, Muon
import logging
from memory_profiler import profile
import awkward as ak
import numpy as np


### As defined by Marina in getL1Eff.py
bins = {}
bins["PFHT"] = np.array([i for i in range(200, 460, 10)]+[j for j in range(460, 600, 20)] + [k for k in range(600, 1320, 50)], dtype=float)
bins["ForthJetPt"] = np.array([30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100], dtype=float)
bins['eta'] = np.array([-2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])


""" Fill these for 
- each L1 seed (numerator) and for inclusive (denominator) for data and MC for calojets, alljets (where calojets have muon vetos)
- for HLT numerators and denominators for data and MC
"""

class htVsJet4Pt(Template):
    ht           = H((bins["PFHT"],   ("hT_trigger",   "H_T [GeV]")))
    jet4_pt      = H((bins["ForthJetPt"], ("jet4_pt", "jet4 pt [GeV]")))
    ht_vs_jet4_pt = H((bins["PFHT"], ("hT_trigger" , "H_T [GeV]")), 
                      ( bins["ForthJetPt"], ("jet4_pt", "jet4 pt [GeV]")))

# @profile
def filling_trigger_histograms(selev, 
                               processName: str = "",
                               year: str = 'UL18',
                               isMC: bool = False,
                               histCuts: list = [],
                               isDataForMixed: bool = False,
                               event_metadata: dict = {},
                               L1_seed_dict: dict = {},
                               ):

    fill = Fill(process=processName, year=year, weight="weight")
    
    hist = Collection( process=[processName],
                        year=[year],
                        **dict((s, ...) for s in histCuts)
                        )

    # fill += hist.add( "trigWeight", (40, 0, 2, ("trigWeight", 'Trigger weight')), weight='no_weight' )

    fill += hist.add( "nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")) )
    fill += hist.add( "nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")), )

    fill += hist.add( "hT", (50, 0, 1500, ("hT", "h_{T} [GeV]")) )
    fill += hist.add( "hT_selected", (50, 0, 1500, ("hT_selected", "h_{T} [GeV]")), )
    # fill += hist.add( "jet4_pt", (bins["ForthJetPt"], ("jet4_pt", "jet4_pt [GeV]")) )

    logging.info(f"jet4pt {selev.jet4_pt}")
    logging.info(f"jet4pt {selev.jet4_pt}")
    logging.info(f"jet4pt {selev.jet4_pt}")

    ##### denominators
    fill += htVsJet4Pt((f"hT_trigger_vs_jet4_pt",  f"hT_trigger_vs_jet4_pt"),  "hT_trigger_vs_jet4_pt")
    weight_type = 'test'
    fill += htVsJet4Pt((f"hT_trigger_vs_jet4_pt_{weight_type}",  f"hT_trigger_vs_jet4_pt {weight_type}"),  "hT_trigger_vs_jet4_pt", weight="weight_test")

    #
    # Jets
    #
    # skip_jet_list = ['energy', 'deepjet_c']
    # fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    # fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    # fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    # fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    
    #
    #  Leptons
    #
    # skip_muons = ["charge"] + Muon.skip_detailed_plots
    # if not isMC:
    #     skip_muons += ["genPartFlav"]
    # fill += Muon.plot( ("selMuons", "Selected Muons"), "selMuon", skip=skip_muons )

    # if "Elec" in selev.fields:
    #     skip_elecs = ["charge"] + Elec.skip_detailed_plots
    #     if not isMC:
    #         skip_elecs += ["genPartFlav"]
    #     fill += Elec.plot( ("selElecs", "Selected Elecs"), "selElec", skip=skip_elecs )



    ## ((name of histogram in coffea, label of histogram in coffea), variable in py file, weight)
    ## numerators
    L1_seed_weights = L1_seed_dict[year]
    for weight_type in L1_seed_weights:
        fill += htVsJet4Pt((f"hT_trigger_vs_jet4_pt_{weight_type}",  f"hT_trigger_vs_jet4_pt {weight_type}"),  "hT_trigger_vs_jet4_pt", weight=weight_type)

    fill(selev, hist)
    return hist.to_dict(nonempty=True)
    