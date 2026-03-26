from coffea4bees.analysis.helpers.hist_templates import (
    FvTHists,
    MvDHists,
    QuadJetHistsSelected,
    QuadJetHistsMinDr,
    QuadJetHistsSRSingle,
    SvBHists,
    TopCandHists,
    WCandHists,
    TrigEmHists,
)
from src.hist_tools import Collection, Fill
from src.hist_tools.object import Elec, Jet, LorentzVector, Muon
import logging
from memory_profiler import profile

YEAR_OVERRIDE_MAP = {
    "201": "Run2",
    "202": "Run3",
}

def _apply_year_override(year: str) -> str:
    for prefix, era in YEAR_OVERRIDE_MAP.items():
        if year.startswith(prefix):
            return era
    return year

# @profile
def filling_nominal_histograms(
    selev,
    JCM,
    processName: str = None,
    year: str = 'UL18',
    isMC: bool = False,
    histCuts: list = [],
    apply_FvT: bool = False,
    apply_MvD: bool = False,
    run_SvB: bool = False,
    top_reconstruction: bool = False,
    isDataForMixed: bool = False,
    tag_list: list = ["threeTag", "fourTag"],
    run_dilep_ttbar_crosscheck: bool = False,
    event_metadata: dict = {},
    weight_name = "weight",
    year_override: bool = False,
    weight_noMvD_override: str = None,
    weight_noFvT_override: str = None,
):
    if year_override:
        year = _apply_year_override(year)

    noMvD_weight = weight_noMvD_override or "weight_noMvD"
    noFvT_weight = weight_noFvT_override or "weight_noFvT"

    fill = Fill(process=processName, year=year, weight=weight_name)

    hist = Collection(
        process=[processName],
        year=[year],
        tag=tag_list,
        region=['SR', "SB"],
        **dict((s, ...) for s in histCuts)
    )

    fill += hist.add("trigWeight", (40, 0, 2, ("trigWeight", 'Trigger weight')), weight='no_weight')
    fill += hist.add("nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")))
    fill += hist.add("nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")))
    fill += hist.add("hT", (50, 0, 1500, ("hT", "h_{T} [GeV]")))
    fill += hist.add("hT_selected", (50, 0, 1500, ("hT_selected", "h_{T} [GeV]")))

    fill += TrigEmHists(("trigEm", "Trigger Emulation"), "trigEm")

    # Jets
    skip_jet_list = ['energy', 'deepjet_c']
    fill += Jet.plot(("selJets", "Selected Jets"), "selJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    fill += Jet.plot(("selJetsRun2", "Selected Jets (Run 2 criteria)"), "selJetRun2", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    fill += Jet.plot(("tagJets", "Tag Jets"), "tagJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    fill += Jet.plot(("tagJetsRun2", "Tag Jets (Run 2 criteria)"), "tagJetRun2", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
    fill += Jet.plot(("othJets", "Other Jets"), "notCanJet_coffea", skip=skip_jet_list, bins={"mass": (50, 0, 100)})

    # Make quad jet hists
    fill += LorentzVector.plot_pair(("v4j", R"$HH_{4b}$"), "v4j", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)})
    fill += QuadJetHistsSelected(("quadJet_selected", "Selected Quad Jet"), "quadJet_selected")
    fill += QuadJetHistsMinDr(("quadJet_min_dr", "Min dR Quad Jet"), "quadJet_min_dr")

    fill += hist.add("m4j", (120, 0, 1200, ("m4j", "m4j [GeV]")))
    fill += hist.add("m4j_hh", (120, 0, 1200, ("m4j_HHSR", "m4j HHSR [GeV]")))
    fill += hist.add("m4j_zh", (120, 0, 1200, ("m4j_ZHSR", "m4j ZHSR [GeV]")))
    fill += hist.add("m4j_zz", (120, 0, 1200, ("m4j_ZZSR", "m4j ZZSR [GeV]")))

    fill += hist.add("vbf_mjj", (100, 0, 2000, ("vbfJets_mjj", "VBF DiJet Mass [GeV]")))
    fill += hist.add("vbf_deta", (50, 0, 10, ("vbfJets_deta", "VBF DiJet #Delta#eta")))

    fill += QuadJetHistsSRSingle(("dijet_HHSR", "DiJet Mass HHSR"), "dijet_HHSR")
    fill += QuadJetHistsSRSingle(("dijet_ZHSR", "DiJet Mass ZHSR"), "dijet_ZHSR")
    fill += QuadJetHistsSRSingle(("dijet_ZZSR", "DiJet Mass ZZSR"), "dijet_ZZSR")

    skip_all_but_n = [
        "deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c",
    ]

    # Make classifier hists
    if apply_FvT:
        FvT_skip = []
        if "pt" not in selev.FvT.fields:
            FvT_skip = ["pt", "pm3", "pm4"]

        fill += FvTHists(("FvT", "FvT Classifier"), "FvT", skip=FvT_skip)
        #fill += FvTHists(("FvT_noFvT_noJCM", "FvT Classifier (NoJCM, NO FVT"), "FvT", weight="weight_noJCM_noFvT", skip=FvT_skip)
        fill += hist.add("quadJet_selected.FvT_score", (100, 0, 1, ("quadJet_selected.FvT_q_score", "Selected Quad Jet Diboson FvT q score")))
        fill += hist.add("quadJet_min_dr.FvT_score", (100, 0, 1, ("quadJet_min_dr.FvT_q_score", "Min dR Quad Jet Diboson FvT q score")))

        if JCM:
            fill += hist.add("FvT_noFvT", (100, 0, 5, ("FvT.FvT", "FvT reweight")), weight=noFvT_weight)


    if apply_MvD:
        fill += MvDHists(("MvD", "MvD Classifier"), "MvD")
        #fill += hist.add("quadJet_selected.MvD_score", (100, 0, 1, ("quadJet_selected.MvD_q_score", "Selected Quad Jet Diboson MvD q score")))
        #fill += hist.add("quadJet_min_dr.MvD_score", (100, 0, 1, ("quadJet_min_dr.MvD_q_score", "Min dR Quad Jet Diboson MvD q score")))

        if JCM:
            fill += hist.add("MvD_noMvD", (100, 0, 5, ("MvD.MvD", "MvD reweight")), weight=noMvD_weight)
        fill += Jet.plot(("selJets_noMvD", "Selected Jets"), "selJet", weight=noMvD_weight, skip=skip_all_but_n)


    fill += Jet.plot(("selJets_noJCM", "Selected Jets"), "selJet", weight="weight_noJCM_noFvT", skip=skip_all_but_n)
    fill += Jet.plot(("tagJets_noJCM", "Tag Jets"), "tagJet", weight="weight_noJCM_noFvT", skip=skip_all_but_n)
    fill += Jet.plot(("tagJets_loose_noJCM", "Loose Tag Jets"), "tagJet_loose", weight="weight_noJCM_noFvT", skip=skip_all_but_n)
    if JCM:
        fill += hist.add("nPSJets", (20, -0.5, 19.5, ("nJet_pseudotagged", "nPseudoTag Jets")))
        fill += hist.add("nPSplusTagJets", (20, -0.5, 19.5, ("nJet_ps_and_tag", "nPseudoTag + nTag Jets")))

    for iJ in range(4):
        fill += Jet.plot((f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], bins={"mass": (50, 0, 100)})

    # Leptons
    skip_muons = ["charge"] + Muon.skip_detailed_plots
    if not isMC:
        skip_muons += ["genPartFlav"]
    fill += Muon.plot(("selMuons", "Selected Muons"), "selMuon", skip=skip_muons)

    if "Elec" in selev.fields:
        skip_elecs = ["charge"] + Elec.skip_detailed_plots
        if not isMC:
            skip_elecs += ["genPartFlav"]
        fill += Elec.plot(("selElecs", "Selected Elecs"), "selElec", skip=skip_elecs)

    # Top Candidates
    if top_reconstruction in ["slow", "fast"]:
        fill += TopCandHists(("top_cand", "Top Candidate"), "top_cand")
        fill += hist.add("xW", (100, -12, 12, ("xW", "xW")))
        fill += hist.add("xbW", (100, -15, 15, ("xbW", "xbW")))

    if run_SvB:
        fill += SvBHists(("SvB", "SvB Classifier"), "SvB")
        fill += SvBHists(("SvB_MA", "SvB MA Classifier"), "SvB_MA")
        #fill += SvBHists(("SvB_noFvT", "SvB Classifier"), "SvB", weight="weight_noFvT")
        fill += SvBHists(("SvB_MA_noFvT", "SvB MA Classifier"), "SvB_MA", weight=noFvT_weight)
        #fill += SvBHists(("SvB_MA_noFvT_noJCM", "SvB MA Classifier"), "SvB_MA", weight="weight_noJCM_noFvT")
        fill += hist.add("quadJet_selected.SvB_q_score", (100, 0, 1, ("quadJet_selected.SvB_q_score", "Selected Quad Jet Diboson SvB q score")))
        fill += hist.add("quadJet_min_dr.SvB_MA_q_score", (100, 0, 1, ("quadJet_min_dr.SvB_MA_q_score", "Min dR Quad Jet Diboson SvB MA q score")))
        if isDataForMixed:
            for _FvT_name in event_metadata["FvT_names"]:
                fill += SvBHists((f"SvB_{_FvT_name}", "SvB Classifier"), "SvB", weight=f"weight_{_FvT_name}")
                fill += SvBHists((f"SvB_MA_{_FvT_name}", "SvB MA Classifier"), "SvB_MA", weight=f"weight_{_FvT_name}")
            for _FvT_name in event_metadata["FvT_names"]:
                fill += hist.add(f"m4j_{_FvT_name}", (120, 0, 1200, ("m4j", "m4j [GeV]")), weight=f"weight_{_FvT_name}")
                fill += hist.add(f"m4j_hh_{_FvT_name}", (120, 0, 1200, ("m4j_HHSR", "m4j HHSR [GeV]")), weight=f"weight_{_FvT_name}")
                fill += hist.add(f"m4j_zh_{_FvT_name}", (120, 0, 1200, ("m4j_ZHSR", "m4j ZHSR [GeV]")), weight=f"weight_{_FvT_name}")
                fill += hist.add(f"m4j_zz_{_FvT_name}", (120, 0, 1200, ("m4j_ZZSR", "m4j ZZSR [GeV]")), weight=f"weight_{_FvT_name}")

    # MC Truth
    if "truth_v4b" in selev.fields:
        fill += LorentzVector.plot_pair(("truth_v4b", R"$HH_{4b}$"), "truth_v4b", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)})

    # low pt
    if any('lowpt' in tag for tag in tag_list):
        fill += hist.add('lowpt_categories', (21, -0.5, 20.5, ('lowpt_categories', 'lowpt_categories')))
        fill += Jet.plot(("allselJets", "All Selected Jets"), "allSelJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
        fill += Jet.plot(("alltagJets", "All Tag Jets"), "allTagJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
        fill += Jet.plot(("selJets_lowpt", "Selected lowpt Jets"), "selJet_lowpt", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
        fill += Jet.plot(("tagJets_lowpt", "Selected lowpt tagged Jets"), "tagJet_lowpt", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
        fill += Jet.plot(("selJets_noJCM_lowpt", "Selected lowpt Jets"), "selJet_lowpt", weight="weight_noJCM_noFvT", skip=skip_all_but_n)
        fill += Jet.plot(("tagJets_noJCM_lowpt", "Tag lowpt Jets"), "tagJet_lowpt", weight="weight_noJCM_noFvT", skip=skip_all_but_n)
        fill += Jet.plot(("tagJets_loose_noJCM_lowpt", "Loose Tag lowpt Jets"), "tagJet_loose_lowpt", weight="weight_noJCM_noFvT", skip=skip_all_but_n)



    # fill histograms
    fill(selev, hist)

    if run_dilep_ttbar_crosscheck:
        fill_ttbar = Fill(process=processName, year=year, weight="weight_noJCM_noFvT")
        hist_ttbar = Collection(
            process=[processName],
            year=[year],
            tag=tag_list,
            **dict((s, ...) for s in ['passMuMu', 'passElMu'])
        )

        fill_ttbar += TrigEmHists(("trigEm_ll", "Trigger Emulation"), "trigEm")

        fill_ttbar += Jet.plot(("selJets_ll", "Selected Jets"), "selJet", skip=skip_jet_list, bins={"mass": (50, 0, 100)})
        fill_ttbar += Jet.plot(("tagJets_ll", "Tag Jets dilep ttbar"), "tagJet", skip=skip_jet_list)


        skip_muons = ["charge"] + Muon.skip_detailed_plots
        fill_ttbar += Muon.plot(("selMuons_ll", "Selected Muons"), "selMuon", skip=skip_muons)

        skip_elecs = ["charge"] + Elec.skip_detailed_plots
        fill_ttbar += Elec.plot(("selElecs_ll", "Selected Elecs"), "selElec", skip=skip_elecs)

        fill_ttbar += LorentzVector.plot(('MET_ll', R'MeT'), 'MET',  skip=['n','eta', 'mass', 'pz', 'energy'], bins={"pt": (60, 0, 300)})
        fill_ttbar += hist_ttbar.add('mll_ll', (100, 0, 300, ('mll', 'mll [GeV]')))

        fill_ttbar(selev, hist_ttbar)
        return hist.to_dict(nonempty=True) | {"hists_ttbar": hist_ttbar.to_dict(nonempty=True)["hists"], "categories_ttbar": hist_ttbar.to_dict(nonempty=True)["categories"]}
    else:
        return hist.to_dict(nonempty=True)


def filling_syst_histograms(selev, weights, analysis_selections,
                            shift_name: str = 'nominal',
                            processName: str = None,
                            year: str = 'UL18',
                            histCuts: list = [],
                            year_override: bool = False,
                            ):

    shift_name = "nominal" if not shift_name else shift_name
    if year_override:
        year = _apply_year_override(year)
    hist_SvB = Collection( process=[processName],
                            year=[year],
                            variation=[shift_name],
                            tag=["threeTag", "fourTag"],
                            region=['SR', "SB"],
                            **dict((s, ...) for s in histCuts),
                            )

    fill_SvB = Fill( process=processName, year=year)
    fill_SvB += SvBHists(("SvB",    "SvB Classifier"),    "SvB",    skip=["ps", "ptt"])
    fill_SvB += SvBHists(("SvB_MA", "SvB MA Classifier"), "SvB_MA", skip=["ps", "ptt"])

    fill_SvB(selev, hist_SvB, variation=shift_name, weight="weight")

    if "nominal" in shift_name:
        logging.info(f"Weight variations {weights.variations}")

        for ivar in list(weights.variations):
            selev[f"weight_{ivar}"] = weights.weight(modifier=ivar)[ analysis_selections ]
            logging.debug(f"{ivar} {selev['weight']}")
            fill_SvB(selev, hist_SvB, variation=ivar, weight=f"weight_{ivar}")

    return hist_SvB.to_dict(nonempty=True)
