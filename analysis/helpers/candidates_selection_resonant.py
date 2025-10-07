import numpy as np
import awkward as ak
import logging
from src.math_tools.random import Squares
from python.analysis.helpers.SvB_helpers import compute_SvB
from python.analysis.helpers.FvT_helpers import compute_FvT
from coffea.nanoevents.methods import vector
from coffea.analysis_tools import Weights

def create_cand_jet_dijet_quadjet(
    selev,
    apply_FvT: bool = False,
    classifier_FvT=None,
    run_SvB: bool = False,
    run_systematics: bool = False,
    classifier_SvB=None,
    classifier_SvB_MA=None,
    processOutput=None,
    isRun3=False,
    include_lowptjets=False,
    label3b: str = "threeTag",
    weights: Weights = None,
    list_weight_names: list[str] = None,
    analysis_selections: ak.Array = None,
):
    """
    Creates candidate jets, dijets, and quadjets for event selection.

    Parameters:
    -----------
    selev : ak.Array
        The selected events.
    apply_FvT : bool, optional
        Whether to apply FvT weights. Defaults to False.
    run_SvB : bool, optional
        Whether to run SvB classification. Defaults to False.
    run_systematics : bool, optional
        Whether to run systematics. Defaults to False.
    classifier_SvB : optional
        The SvB classifier. Defaults to None.
    classifier_SvB_MA : optional
        The SvB_MA classifier. Defaults to None.
    processOutput : optional
        Output dictionary for processing. Defaults to None.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.

    Returns:
    --------
    None
        Modifies the `selev` object in place.
    """
    #
    # To get test vectors
    #
    #jet_subset_dict = {key: getattr(selev.Jet,key)[0:10].tolist() for key in ["pt", "eta","phi", "mass","btagScore","bRegCorr","puId","jetId","selected", "selected_loose"]}
    #print(jet_subset_dict)

    #
    # Build and select boson candidate jets with bRegCorr applied
    #
    sorted_idx = ak.argsort( selev.Jet.btagScore * selev.Jet.selected, axis=1, ascending=False )
    if include_lowptjets:
        sorted_idx_lowpt = ak.argsort( selev.Jet.btagScore * selev.Jet.selected_lowpt, axis=1, ascending=False )
        canJet_idx = ak.concatenate([sorted_idx[:, 0:3], sorted_idx_lowpt[:, :1]], axis=1)
        logging.debug(f"lowpt selected {(selev.Jet.selected_lowpt)[:1]}")
        logging.debug(f"both lowpt {(selev.Jet.btagScore * selev.Jet.selected_lowpt)[:1]}")
        logging.debug(f"sorted_idx_lowpt {sorted_idx_lowpt[:1]}")

    else:
        canJet_idx = sorted_idx[:, 0:4]
    # Exclude canJet_idx from sorted_idx
    mask = ~ak.any(canJet_idx[:, :, np.newaxis] == sorted_idx[:, np.newaxis, :], axis=1)
    notCanJet_idx = sorted_idx[mask]
    
    logging.debug(f"canJet_idx {canJet_idx[:1]}")
    logging.debug(f"notCanJet_idx {notCanJet_idx[:1]}\n\n")
    

    # # apply bJES to canJets
    canJet = selev.Jet[canJet_idx] * selev.Jet[canJet_idx].bRegCorr
    canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
    canJet["btagScore"] = selev.Jet.btagScore[canJet_idx]
    canJet["puId"] = selev.Jet.puId[canJet_idx]
    canJet["jetId"] = selev.Jet.jetId[canJet_idx]

    # CutFlow Debugging
    #if "pt_jec" in selev.Jet.fields:
    #    canJet["PNetRegPtRawCorr"] = selev.Jet.PNetRegPtRawCorr[canJet_idx]
    #    canJet["PNetRegPtRawCorrNeutrino"] = selev.Jet.PNetRegPtRawCorrNeutrino[canJet_idx]
    #    canJet["pt_raw"] = selev.Jet.pt_raw[canJet_idx]

    if "hadronFlavour" in selev.Jet.fields:
        canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]

    #
    # pt sort canJets
    #
    canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
    selev["canJet"] = canJet
    for i in range(4):
        selev[f"canJet{i}"] = selev["canJet"][:, i]

    selev["v4j"] = canJet.sum(axis=1)
    notCanJet = selev.Jet[notCanJet_idx]
    notCanJet = notCanJet[notCanJet.selected_loose]
    notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

    notCanJet["isSelJet"] = 1 * ( (notCanJet.pt >= 40) & (np.abs(notCanJet.eta) < 2.4) )
    selev["notCanJet_coffea"] = notCanJet
    selev["nNotCanJet"] = ak.num(selev.notCanJet_coffea)

    # Build diJets, indexed by diJet[event,pairing,0/1]
    canJet = selev["canJet"]
    pairing = [([0, 2], [0, 1], [0, 1]), ([1, 3], [2, 3], [3, 2])]
    diJet = canJet[:, pairing[0]] + canJet[:, pairing[1]]
    diJet["lead"] = canJet[:, pairing[0]]
    diJet["subl"] = canJet[:, pairing[1]]
    diJet["st"] = diJet["lead"].pt + diJet["subl"].pt
    # diJet["mass"] = (diJet["lead"] + diJet["subl"]).mass
    diJet["dr"] = diJet["lead"].delta_r(diJet["subl"])
    diJet["dphi"] = diJet["lead"].delta_phi(diJet["subl"])

    # Sort diJets within views to be lead st, subl st
    if isRun3:
        diJet = diJet[ak.argsort(diJet.pt, axis=2, ascending=False)]
    else:
        diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
    diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]
    # Now indexed by diJet[event,pairing,lead/subl st]

    # Compute diJetMass cut with independent min/max for lead/subl
    minDiJetMass = np.array([[[52, 50]]])
    maxDiJetMass = np.array([[[180, 173]]])
    diJet["passDiJetMass"] = (minDiJetMass < diJet.mass) & ( diJet.mass < maxDiJetMass )

    # Compute MDRs
    min_m4j_scale = np.array([[360, 235]])
    min_dr_offset = np.array([[-0.5, 0.0]])
    max_m4j_scale = np.array([[650, 650]])
    max_dr_offset = np.array([[0.5, 0.7]])
    max_dr = np.array([[1.5, 1.5]])
    # m4j = np.repeat(np.reshape(np.array(selev["v4j"].mass), (-1, 1, 1)), 2, axis=2)
    m4j = selev["v4j"].mass[:, np.newaxis, np.newaxis]
    diJet["passMDR"] = (min_m4j_scale / m4j + min_dr_offset < diJet.dr) & ( diJet.dr < np.maximum(max_m4j_scale / m4j + max_dr_offset, max_dr) )

    #
    # Compute consistency of diJet masses with boson masses
    #
    mZ = 91.0
    mH = 125.0
    st_bias = np.array([[[1.02, 0.98]]])
    cZ = mZ * st_bias
    cH = mH * st_bias

    diJet["xZ"] = (diJet.mass - cZ) / (0.1 * diJet.mass)
    diJet["xH"] = (diJet.mass - cH) / (0.1 * diJet.mass)

    #
    # Build quadJets
    #
    rng_0 = Squares("quadJetSelection")
    rng_1 = rng_0.shift(1)
    rng_2 = rng_0.shift(2)
    counter = selev.event

    # print(f"{self.chunk} mass {diJet[:, :, 0].mass[0:5]}\n")
    # print(f"{self.chunk} mass view64 {np.asarray(diJet[:, :, 0].mass).view(np.uint64)[0:5]}\n")
    # print(f"{self.chunk} mass rounded view64 {np.round(np.asarray(diJet[:, :, 0].mass), 0).view(np.uint64)[0:5]}\n")
    # print(f"{self.chunk} mass rounded {np.round(np.asarray(diJet[:, :, 0].mass), 0)[0:5]}\n")

    quadJet = ak.zip( { "lead": diJet[:, :, 0],
                        "subl": diJet[:, :, 1],
                        "close": diJetDr[:, :, 0],
                        "other": diJetDr[:, :, 1],
                        "passDiJetMass": ak.all(diJet.passDiJetMass, axis=2),
                        "random": np.concatenate([rng_0.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
                                                  rng_1.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
                                                  rng_2.uniform(counter, low=0.1, high=0.9)[:, np.newaxis]], axis=1),

                       } )

    quadJet["dr"] = quadJet["lead"].delta_r(quadJet["subl"])
    quadJet["dphi"] = quadJet["lead"].delta_phi(quadJet["subl"])
    quadJet["deta"] = quadJet["lead"].eta - quadJet["subl"].eta
    quadJet["v4jmass"] = selev["v4j"].mass

    #
    # Compute Signal Regions
    #
    quadJet["xZZ"] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
    quadJet["xHH"] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
    quadJet["xZH"] = np.sqrt( np.minimum( quadJet.lead.xH**2 + quadJet.subl.xZ**2, quadJet.lead.xZ**2 + quadJet.subl.xH**2, ) )

    max_xZZ = 2.6
    max_xZH = 1.9
    max_xHH = 1.9
    quadJet["ZZSR"] = quadJet.xZZ < max_xZZ
    quadJet["ZHSR"] = quadJet.xZH < max_xZH
    quadJet["HHSR"] = ((quadJet.xHH < max_xHH) & selev.notInBoostedSel ) if 'notInBoostedSel' in selev.fields else (quadJet.xHH < max_xHH)  ## notInBoostedSel is true by default


    if isRun3:

        # Compute distances to diagonal
        #   https://gitlab.cern.ch/mkolosov/hh4b_run3/-/blob/run2/python/producers/hh4bTreeProducer.py#L3386
        diagonalXoYo = 1.04
        quadJet["dhh"] = (1.0/np.sqrt(1+pow(diagonalXoYo, 2)))*abs(quadJet["lead"].mass - ((diagonalXoYo)*quadJet["subl"].mass))

        # If the difference of the minimum and second minimum distance from the diagonal is less than 30, choose the pair that has the
        # maximum H1 pT in the 4-jet center-of-mass frame
        dhh_sorted = np.sort(quadJet["dhh"], axis=1)
        dhh_sorted_arg = np.argsort(quadJet["dhh"], axis=1)
        delta_dhh = abs(dhh_sorted[:,1] - dhh_sorted[:,0])


        # Get the two jets with the minimum and second minimum distance from the diagonal
        quadJet_min_dhh_mask = dhh_sorted[:,0] == quadJet.dhh
        quadJet_min_dhh = quadJet[quadJet_min_dhh_mask]

        quadJet_min2_dhh_mask = dhh_sorted[:,1] == quadJet.dhh
        quadJet_min2_dhh = quadJet[quadJet_min2_dhh_mask]

        # Create a boost vector from the 4-momentum sum
        boost_vec_v4j = ak.zip(
            {
                "x": selev.v4j.px / selev.v4j.energy,
                "y": selev.v4j.py / selev.v4j.energy,
                "z": selev.v4j.pz / selev.v4j.energy,
            },
            with_name="ThreeVector",
            behavior=vector.behavior,
        )

        # Boost the jets into the center of mass frame
        quadJet_min_dhh_lead_CM  = quadJet_min_dhh.lead[:,0].boost(-boost_vec_v4j)
        quadJet_min2_dhh_lead_CM = quadJet_min2_dhh.lead[:,0].boost(-boost_vec_v4j)

        use_dhh2_mask = (delta_dhh < 30) & (quadJet_min2_dhh_lead_CM.pt > quadJet_min_dhh_lead_CM.pt)

        quadJet["selected"] = ak.where(use_dhh2_mask, quadJet_min2_dhh_mask, quadJet_min_dhh_mask)


        #
        #   For CR selection
        #
        cLead = 125
        cSubl = 120
        SR_radius = 30
        CR_radius = 55

        quadJet["rhh"] = np.sqrt( (quadJet["lead"].mass - cLead)**2 + (quadJet["subl"].mass - cSubl)**2 )
        quadJet["SR"] = (quadJet.rhh < SR_radius)
        quadJet["SB"] =  (~quadJet.SR) & (quadJet.rhh < CR_radius)
        quadJet["passDiJetMass"] =  quadJet.SR | quadJet.SB

    else:

        #
        # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
        # How to eliminate events where all are not maximally satisfied?
        #
        quadJet["rank"] = ( 10 * quadJet.passDiJetMass + quadJet.lead.passMDR + quadJet.subl.passMDR + quadJet.random )
        quadJet["selected"] = quadJet.rank == np.max(quadJet.rank, axis=1)
        quadJet["SR"] = (quadJet.rank >= 12) & (quadJet.ZZSR | quadJet.ZHSR | quadJet.HHSR) #& passabovecurve
        quadJet["SB"] = quadJet.passDiJetMass & ~quadJet.SR & (quadJet.rank >= 12) #& passabovecurve

    if classifier_FvT is not None:
        logging.info("Computing FvT scores with classifier")

        compute_FvT(selev, selev[label3b], FvT=classifier_FvT)
        weight_FvT = np.ones(len(weights.weight()), dtype=float)
        weight_FvT[analysis_selections] *= ak.to_numpy(selev.FvT.FvT)
        weights.add("FvT", weight_FvT)
        list_weight_names.append("FvT")
        logging.debug( f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n" )
        apply_FvT = True

    if apply_FvT and ("FvT" in selev.fields):

        quadJet["FvT_q_score"] = np.concatenate( [
            selev.FvT.q_1234[:, np.newaxis],
            selev.FvT.q_1324[:, np.newaxis],
            selev.FvT.q_1423[:, np.newaxis],
        ], axis=1, )

    if run_SvB:

        if (classifier_SvB is not None) | (classifier_SvB_MA is not None):
            
            if run_systematics: tmp_mask = (selev.fourTag & quadJet[quadJet.selected][:, 0].SR)
            else: tmp_mask = np.full(len(selev), True)
            compute_SvB(selev,
                        tmp_mask,
                        SvB=classifier_SvB,
                        SvB_MA=classifier_SvB_MA,
                        doCheck=False)

        quadJet["SvB_q_score"] = np.concatenate( [
            selev.SvB.q_1234[:, np.newaxis],
            selev.SvB.q_1324[:, np.newaxis],
            selev.SvB.q_1423[:, np.newaxis],
            ], axis=1, )

        quadJet["SvB_MA_q_score"] = np.concatenate( [
            selev.SvB_MA.q_1234[:, np.newaxis],
            selev.SvB_MA.q_1324[:, np.newaxis],
            selev.SvB_MA.q_1423[:, np.newaxis],
            ], axis=1, )

    selev["diJet"] = diJet
    selev["quadJet"] = quadJet
    selev["quadJet_selected"] = quadJet[quadJet.selected][:, 0]
    selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)
    #
    #  Build the close dR and other quadjets
    #    (There is Probably a better way to do this ...
    #
    arg_min_close_dr = np.argmin(quadJet.close.dr, axis=1)
    arg_min_close_dr = arg_min_close_dr.to_numpy()
    selev["quadJet_min_dr"] = quadJet[ np.array(range(len(quadJet))), arg_min_close_dr ]


    selev["m4j"] = selev.v4j.mass
    selev["m4j_HHSR"] = ak.where(~selev.quadJet_selected.HHSR, -2, selev.m4j)
    selev["m4j_ZHSR"] = ak.where(~selev.quadJet_selected.ZHSR, -2, selev.m4j)
    selev["m4j_ZZSR"] = ak.where(~selev.quadJet_selected.ZZSR, -2, selev.m4j)

    passSvBUpCut = 0.8
    passSvB_hh = selev.SvB_MA.ps_hh > passSvBUpCut
    selev["m4j_HHSR_passHH"]  = ak.where((~passSvB_hh), -2, selev.m4j_HHSR)
    selev["m4j_ZHSR_passZH"]  = ak.where((selev.SvB_MA.ps_zh <= passSvBUpCut), -2, selev.m4j_ZHSR)
    selev["m4j_ZZSR_passZZ"]  = ak.where((selev.SvB_MA.ps_zz <= passSvBUpCut), -2, selev.m4j_ZZSR)
    
    # selev["m4j_HHSR_passSvB"] = ak.where((selev.SvB_MA.ps <= passSvBUpCut),    -2, selev.m4j_HHSR)
    # selev["m4j_ZHSR_passSvB"] = ak.where((selev.SvB_MA.ps <= passSvBUpCut),    -2, selev.m4j_ZHSR)
    # selev["m4j_ZZSR_passSvB"] = ak.where((selev.SvB_MA.ps <= passSvBUpCut),    -2, selev.m4j_ZZSR)

    # cut_SvB_hh = selev.SvB_MA.ps_hh > 0.8
    # m4j_pass_cut = ak.where((~cut_SvB_hh), -2, selev.m4j_HHSR)
    # svb_pass_cut = ak.where((~cut_SvB_hh), -2, selev.SvB_MA.ps_hh)
    # selev['v4j_mass_vs_SvB_p8_1']  = ak.zip( { "m4j": m4j_pass_cut, "SvB": svb_pass_cut } )

    # cut_SvB_hh = (selev.SvB_MA.ps_hh > 0.6) & (selev.SvB_MA.ps_hh <= 0.8)
    # m4j_pass_cut = ak.where((~cut_SvB_hh), -2, selev.m4j_HHSR)
    # svb_pass_cut = ak.where((~cut_SvB_hh), -2, selev.SvB_MA.ps_hh)
    # selev['v4j_mass_vs_SvB_p6_p8']  = ak.zip( { "m4j": m4j_pass_cut, "SvB": svb_pass_cut } )

    # cut_SvB_hh = (selev.SvB_MA.ps_hh > 0.4) & (selev.SvB_MA.ps_hh <= 0.8)
    # m4j_pass_cut = ak.where((~cut_SvB_hh), -2, selev.m4j_HHSR)
    # svb_pass_cut = ak.where((~cut_SvB_hh), -2, selev.SvB_MA.ps_hh)
    # selev['v4j_mass_vs_SvB_p4_p8']  = ak.zip( { "m4j": m4j_pass_cut, "SvB": svb_pass_cut } )

    # cut_SvB_hh = (selev.SvB_MA.ps_hh > 0.2) & (selev.SvB_MA.ps_hh <= 0.8)
    # m4j_pass_cut = ak.where((~cut_SvB_hh), -2, selev.m4j_HHSR)
    # svb_pass_cut = ak.where((~cut_SvB_hh), -2, selev.SvB_MA.ps_hh)
    # selev['v4j_mass_vs_SvB_p2_p8']  = ak.zip( { "m4j": m4j_pass_cut, "SvB": svb_pass_cut } )

    # cut_SvB_hh = (selev.SvB_MA.ps_hh > 0.2) & (selev.SvB_MA.ps_hh <= 1)
    # m4j_pass_cut = ak.where((~cut_SvB_hh), -2, selev.m4j_HHSR)
    # svb_pass_cut = ak.where((~cut_SvB_hh), -2, selev.SvB_MA.ps_hh)
    # selev['v4j_mass_vs_SvB_p2_1']  = ak.zip( { "m4j": m4j_pass_cut, "SvB": svb_pass_cut } )

    # cut_SvB_hh = (selev.SvB_MA.ps_hh <= 0.2)
    # m4j_pass_cut = ak.where((~cut_SvB_hh), -2, selev.m4j_HHSR)
    # svb_pass_cut = ak.where((~cut_SvB_hh), -2, selev.SvB_MA.ps_hh)
    # selev['v4j_mass_vs_SvB_0_p2']  = ak.zip( { "m4j": m4j_pass_cut, "SvB": svb_pass_cut } )

    # selev['v4j_mass_vs_SvB_0_1']  = ak.zip( { "m4j": selev.m4j_HHSR, "SvB": selev.SvB_MA.ps_hh } )

    # lsdr = (selev.quadJet_selected.lead).delta_r(selev.quadJet_selected.subl)
    # a,b,d,n,k = 59.5, 3.2, 3.6, 3.26, 3.1
    # selev["passLSdr"] = lsdr >= k - (((-d + selev.m4j/a)**n)  * np.exp(d - selev.m4j/a) / b)
    # selev["passLSdr"] = lsdr >= 2.6
    # selev["failLSdr"] = ~ selev.passLSdr
    # selev["passLSdrpassSvB"] = selev.passLSdr &  passSvB_hh
    # selev["passLSdrfailSvB"] = selev.passLSdr & ~passSvB_hh

    selev['leadStM_selected'] = selev.quadJet_selected.lead.mass
    selev['sublStM_selected'] = selev.quadJet_selected.subl.mass

    selev['dijet_HHSR'] = ak.zip( { "lead_m": ak.where(~selev.quadJet_selected.HHSR, -2, selev.leadStM_selected),
                                    "subl_m": ak.where(~selev.quadJet_selected.HHSR, -2, selev.sublStM_selected),
                                } )
    selev['dijet_ZHSR'] = ak.zip( { "lead_m": ak.where(~selev.quadJet_selected.ZHSR, -2, selev.leadStM_selected),
                                    "subl_m": ak.where(~selev.quadJet_selected.ZHSR, -2, selev.sublStM_selected),
                                    } )
    selev['dijet_ZZSR'] = ak.zip( { "lead_m": ak.where(~selev.quadJet_selected.ZZSR, -2, selev.leadStM_selected),
                                    "subl_m": ak.where(~selev.quadJet_selected.ZZSR, -2, selev.sublStM_selected),
                                    } )

    # selev['v4j_mass_vs_pt_l']  = ak.zip( { "m4j": selev.m4j,  "pt": selev.quadJet_selected.lead.pt } )
    # selev['v4j_mass_vs_pt_s']  = ak.zip( { "m4j": selev.m4j,  "pt": selev.quadJet_selected.subl.pt } )
    # selev['v4j_mass_vs_pt_ll'] = ak.zip( { "m4j": selev.m4j,  "pt": selev.quadJet_selected.lead.lead.pt } )
    # selev['v4j_mass_vs_pt_ls'] = ak.zip( { "m4j": selev.m4j,  "pt": selev.quadJet_selected.lead.subl.pt } )
    # selev['v4j_mass_vs_pt_sl'] = ak.zip( { "m4j": selev.m4j,  "pt": selev.quadJet_selected.subl.lead.pt } )
    # selev['v4j_mass_vs_pt_ss'] = ak.zip( { "m4j": selev.m4j,  "pt": selev.quadJet_selected.subl.subl.pt } )

    # selev['v4j_pt_vs_dr_l_s']   = ak.zip( { "pt": selev.v4j.pt, "dr": (selev.quadJet_selected.lead).delta_r(selev.quadJet_selected.subl) } )
    # selev[  'l_pt_vs_dr_l_s']   = ak.zip( { "pt": selev.quadJet_selected.lead.pt, "dr": (selev.quadJet_selected.lead).delta_r(selev.quadJet_selected.subl) } )
    # selev[  's_pt_vs_dr_l_s']   = ak.zip( { "pt": selev.quadJet_selected.subl.pt, "dr": (selev.quadJet_selected.lead).delta_r(selev.quadJet_selected.subl) } )

    # selev['svb_vs_dr_l_s']   = ak.zip( { "SvB": selev.SvB_MA.ps_hh,  "dr": (selev.quadJet_selected.lead).delta_r(selev.quadJet_selected.subl) } )
    # selev['v4j_mass_vs_dr_l_s']   = ak.zip( { "m4j": selev.m4j,  "dr": (selev.quadJet_selected.lead).delta_r(selev.quadJet_selected.subl) } )
    # selev['v4j_mass_vs_dr_ll_ls'] = ak.zip( { "m4j": selev.m4j,  "dr": (selev.quadJet_selected.lead.lead).delta_r(selev.quadJet_selected.lead.subl) } )
    # selev['v4j_mass_vs_dr_ll_sl'] = ak.zip( { "m4j": selev.m4j,  "dr": (selev.quadJet_selected.lead.lead).delta_r(selev.quadJet_selected.subl.lead) } )
    # selev['v4j_mass_vs_dr_ll_ss'] = ak.zip( { "m4j": selev.m4j,  "dr": (selev.quadJet_selected.lead.lead).delta_r(selev.quadJet_selected.subl.subl) } )
    # selev['v4j_mass_vs_dr_ls_sl'] = ak.zip( { "m4j": selev.m4j,  "dr": (selev.quadJet_selected.lead.subl).delta_r(selev.quadJet_selected.subl.lead) } )
    # selev['v4j_mass_vs_dr_ls_ss'] = ak.zip( { "m4j": selev.m4j,  "dr": (selev.quadJet_selected.lead.subl).delta_r(selev.quadJet_selected.subl.subl) } )
    # selev['v4j_mass_vs_dr_sl_ss'] = ak.zip( { "m4j": selev.m4j,  "dr": (selev.quadJet_selected.subl.lead).delta_r(selev.quadJet_selected.subl.subl) } )

    # selev['v4j_mass_vs_v4j_pt'] = ak.zip( { "m4j": selev.m4j,  "pt4j": selev.v4j.pt, } )
    # selev['v4j_mass_vs_hT'] = ak.zip( { "m4j": selev.m4j,  "hT": selev.hT, } )
    # selev['v4j_mass_vs_hT_trigger'] = ak.zip( { "m4j": selev.m4j,  "hT": selev.hT_trigger, } )
    # selev['v4j_mass_vs_hT_selected'] = ak.zip( { "m4j": selev.m4j,  "hT": selev.hT_selected, } )
    # selev['v4j_mass_vs_nJetSel'] = ak.zip({"m4j": selev.m4j, "nJetSel": selev.nJet_selected})
    # selev['v4j_pt_vs_nJetSel'] = ak.zip({"pt4j": selev.v4j.pt, "nJetSel": selev.nJet_selected})
    # selev['hT_sel_vs_nJetSel'] = ak.zip({"hT": selev.hT_selected, "nJetSel": selev.nJet_selected})
    # selev['hT_sel_vs_v4j_pt'] = ak.zip( {"hT": selev.hT_selected,  "pt4j": selev.v4j.pt, } )
    
    
    # HHSR_true = selev.quadJet_selected.HHSR
    # SvB_MA_HHSR_fields = {}
    # for key in selev.SvB_MA.fields:
       #  SvB_MA_HHSR_fields[key] = ak.where(~HHSR_true, -2, selev.SvB_MA[key])
    #selev["SvB_MA_HHSR"] = ak.zip(SvB_MA_HHSR_fields)
    #logging.info(selev.SvB_MA_HHSR)

    selev["region"] = ak.zip({
        "SR": selev["quadJet_selected"].SR,
        "SB": selev["quadJet_selected"].SB
        })

    #
    # Debugging the skimmer
    #
    ### selev_mask = selev.event == 434011
    ### out_data = {}
    ### out_data["debug_event"  ]            = selev.event[selev_mask]
    ### out_data["debug_qj_rank"  ]    = quadJet[selev_mask].rank.to_list()
    ### out_data["debug_qj_selected"  ]    = quadJet[selev_mask].selected.to_list()
    ### out_data["debug_qj_passDiJetMass"  ]    = quadJet[selev_mask].passDiJetMass.to_list()
    ### out_data["debug_qj_lead_passMDR"  ]    = quadJet[selev_mask].lead.passMDR.to_list()
    ### out_data["debug_qj_subl_passMDR"  ]    = quadJet[selev_mask].subl.passMDR.to_list()
    ### out_data["debug_qj_lead_mass"  ]    = quadJet[selev_mask].lead.mass.to_list()
    ### out_data["debug_qj_subl_mass"  ]    = quadJet[selev_mask].subl.mass.to_list()
    ### out_data["debug_qj_random"  ]    = quadJet[selev_mask].random.to_list()
    ### out_data["debug_qj_SR"  ]    = quadJet[selev_mask].SR.to_list()
    ### out_data["debug_qj_HHSR"  ]    = quadJet[selev_mask].HHSR.to_list()
    ### out_data["debug_qj_ZZSR"  ]    = quadJet[selev_mask].ZZSR.to_list()
    ### out_data["debug_qj_ZHSR"  ]    = quadJet[selev_mask].ZHSR.to_list()
    ### out_data["debug_qj_xZZ"  ]    = quadJet[selev_mask].xZZ.to_list()
    ### out_data["debug_qj_xZH"  ]    = quadJet[selev_mask].xZH.to_list()
    ### out_data["debug_qj_xHH"  ]    = quadJet[selev_mask].xHH.to_list()
    ### out_data["debug_qj_ZHSR"  ]    = quadJet[selev_mask].ZHSR.to_list()
    ### out_data["debug_qj_lead_xZ"  ]    = quadJet[selev_mask].lead.xZ.to_list()
    ### out_data["debug_qj_lead_xH"  ]    = quadJet[selev_mask].lead.xH.to_list()
    ### out_data["debug_qj_subl_xZ"  ]    = quadJet[selev_mask].subl.xZ.to_list()
    ### out_data["debug_qj_subl_xH"  ]    = quadJet[selev_mask].subl.xH.to_list()
    ### out_data["debug_qj_SB"  ]    = quadJet[selev_mask].SB.to_list()
    ### out_data["debug_counter"  ]    = counter[selev_mask].to_list()
    ### out_data["debug_SR"] = selev["quadJet_selected"][selev_mask].SR
    ### out_data["debug_SB"] = selev["quadJet_selected"][selev_mask].SB
    ### out_data["debug_threeTag"] = selev[selev_mask].threeTag
    ### out_data["debug_fourTag"] = selev[selev_mask].fourTag
    ### out_data["debug_qj_lead_pt"  ]         = quadJet[selev_mask].lead.pt.to_list()
    ### out_data["debug_qj_lead_lead_pt"  ]    = quadJet[selev_mask].lead.lead.pt.to_list()
    ### out_data["debug_qj_lead_lead_eta"  ]   = quadJet[selev_mask].lead.lead.eta.to_list()
    ### out_data["debug_qj_lead_lead_phi"  ]   = quadJet[selev_mask].lead.lead.phi.to_list()
    ### out_data["debug_qj_lead_lead_mass"  ]  = quadJet[selev_mask].lead.lead.mass.to_list()
    ### out_data["debug_qj_lead_subl_pt"  ]    = quadJet[selev_mask].lead.subl.pt.to_list()
    ### out_data["debug_qj_lead_subl_eta"  ]   = quadJet[selev_mask].lead.subl.eta.to_list()
    ### out_data["debug_qj_lead_subl_phi"  ]   = quadJet[selev_mask].lead.subl.phi.to_list()
    ### out_data["debug_qj_lead_subl_mass"  ]  = quadJet[selev_mask].lead.subl.mass.to_list()
    ###
    ### out_data["debug_qj_subl_pt"  ]         = quadJet[selev_mask].subl.pt.to_list()
    ### out_data["debug_qj_subl_lead_pt"  ]    = quadJet[selev_mask].subl.lead.pt.to_list()
    ### out_data["debug_qj_subl_lead_eta"  ]   = quadJet[selev_mask].subl.lead.eta.to_list()
    ### out_data["debug_qj_subl_lead_phi"  ]   = quadJet[selev_mask].subl.lead.phi.to_list()
    ### out_data["debug_qj_subl_lead_mass"  ]  = quadJet[selev_mask].subl.lead.mass.to_list()
    ###
    ### out_data["debug_qj_subl_subl_pt"  ]    = quadJet[selev_mask].subl.subl.pt.to_list()
    ### out_data["debug_qj_subl_subl_eta"  ]   = quadJet[selev_mask].subl.subl.eta.to_list()
    ### out_data["debug_qj_subl_subl_phi"  ]   = quadJet[selev_mask].subl.subl.phi.to_list()
    ### out_data["debug_qj_subl_subl_mass"  ]  = quadJet[selev_mask].subl.subl.mass.to_list()
    ###
    ###
    ### for out_k, out_v in out_data.items():
    ###     processOutput[out_k] = {}
    ###     processOutput[out_k][selev.metadata['dataset']] = list(out_v)

    if run_SvB:
        selev["passSvB"] = selev["SvB_MA"].ps > 0.80
        selev["failSvB"] = selev["SvB_MA"].ps < 0.05

    # After building canJet_idx and notCanJet_idx
    del sorted_idx
    if include_lowptjets:
        del sorted_idx_lowpt
    del canJet_idx, notCanJet_idx

    # After building canJet and notCanJet
    del canJet, notCanJet

    # After building diJet, diJetDr, pairing
    del diJet, diJetDr, pairing

    # After building quadJet and all quadJet selection logic
    del quadJet

    # After Run3 selection logic
    if isRun3:
        del quadJet_min_dhh_mask, quadJet_min_dhh, quadJet_min2_dhh_mask, quadJet_min2_dhh
        del dhh_sorted, dhh_sorted_arg, delta_dhh
        del boost_vec_v4j, quadJet_min_dhh_lead_CM, quadJet_min2_dhh_lead_CM, use_dhh2_mask

    # After region/CR selection
    del arg_min_close_dr

    # Final cleanup
    import gc
    gc.collect()

    return selev
