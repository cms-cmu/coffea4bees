import time
import awkward as ak
import numpy as np
import yaml
import warnings
from collections import OrderedDict

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection
import hist
from coffea4bees.analysis.helpers.cutflow import cutflow_4b

from src.physics.event_selection import apply_event_selection
from src.hist_tools import Collection, Fill
from coffea4bees.jet_clustering.clustering_hist_templates import ClusterHistsBoosted
from src.hist_tools.object import Jet

from coffea4bees.jet_clustering.declustering import compute_decluster_variables

import logging
import vector

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            corrections_metadata: dict = None,
            **kwargs
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata

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
        isMC    = True if event.run[0] == 1 else False
        nEvent = len(event)

        logging.info(fname)
        logging.info(f'Process {nEvent} Events')

        #
        # Event selection
        #
        event = apply_event_selection( event,
                                        self.corrections_metadata[year],
                                        cut_on_lumimask=(not isMC),
                                        )

        selFatJet = event.FatJet
        selFatJet = selFatJet[selFatJet.particleNetMD_Xbb > 0.8]
        selFatJet = selFatJet[selFatJet.subJetIdx1 >= 0]
        selFatJet = selFatJet[selFatJet.subJetIdx2 >= 0]



        # Hack for synthetic data
        selFatJet = selFatJet[selFatJet.subJetIdx1 < 4]
        selFatJet = selFatJet[selFatJet.subJetIdx2 < 4]

        selFatJet = selFatJet[(selFatJet.subjets [:, :, 0] + selFatJet.subjets [:, :, 1]).pt > 300]
        selFatJet = selFatJet[(selFatJet.subjets [:, :, 0] + selFatJet.subjets [:, :, 1]).mass > 50]


        #print(f" fields FatJets: {selFatJet.fields}")
        #print(f" fields subJet: {selFatJet.subjets.fields}")
        # print(f"  tau2 : {selFatJet.subjets.tau2}\n")
        # print(f"  tau1 : {selFatJet.subjets.tau2}\n")
        # print(f"  tau21 : {selFatJet.subjets.tau2 / selFatJet.subjets.tau1}\n")

        #print(f" fields nSubJets: {ak.num(selFatJet.subjets)},\n")

        #selFatJet = selFatJet[ak.num(selFatJet.subjets) > 1]
        event["selFatJet"] = selFatJet


        #  Check How often do we have >=2 Fat Jets?
        event["passNFatJets"]  = (ak.num(event.selFatJet) == 2)

        # Apply object selection (function does not remove events, adds content to objects)

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( "passNFatJets",  event.passNFatJets )
        ### add more selections, this can be useful

        event["weight"] = 1.0

        #
        # Do the cutflow
        #
        sel_dict = OrderedDict({
            'all'               : selections.require(lumimask=True),
            'passNoiseFilter'   : selections.require(lumimask=True, passNoiseFilter=True),
            'passHLT'           : selections.require(lumimask=True, passNoiseFilter=True, passHLT=True),
            'passNFatJets'      : selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passNFatJets=True),
        })
        #sel_dict['passJetMult'] = selections.all(*allcuts)

        self.cutFlow = cutFlow()
        for cut, sel in sel_dict.items():
            self.cutFlow.fill( cut, event[sel], allTag=True )



        list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "passNFatJets" ]
        analysis_selections = selections.all(*list_of_cuts)
        selev = event[analysis_selections]

        #
        # Event selection
        #
        #
        #print(f"Number of selected Fat Jets: {ak.num(selev.selFatJet)}")
        #print(f" Any passNFatJets: {ak.any(selev.passNFatJets)}")
        #print(f" Any passHLT: {ak.any(selev.passHLT)}")
        #print(f" FatJet pt: {selev.selFatJet.pt}")

        #print(f" nSubJets: {ak.num(selev.selFatJet.subjets, axis=2)}")
        #print(f" subjet pt: {selev.selFatJet.pt[0:10]}")

        #print(f" FatJet pt: {selev.selFatJet.pt}")
        #print(f" FatJet Subjet pt pt: {selev.selFatJet.subjets.pt}")
        #print(f" FatJet subjet pt A: {selev.selFatJet.subjets[:,:,0].pt}")
        #print(f" FatJet subjet pt B: {selev.selFatJet.subjets[:,:,1].pt}")
        #print(f" FatJet subjet len(: {len(selev.selFatJet.subjets[:,:,0].pt)}")

        #
        # btagDeepB
        #
        #print(f" FatJet subjet0 pt: {selev.selFatJet[:,0].subjets.pt}")
        #print(f" FatJet subjet0_0 pt: {selev.selFatJet[:,0].subjets[:,0].pt}")
        #print(f" FatJet subjet0_1 pt: {selev.selFatJet[:,0].subjets[:,1].pt}")
        #print(f" FatJet subjet1 pt: {selev.selFatJet[:,1].subjets.pt}")
        #print("SubJet 0",ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet.subjets[:,0]]))
        #print("SubJet 1",ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet.subjets[:,1]]))
        #str(round(v,3))
        #print("btag1",selev.selFatJet.subjets.btagDeepB,"\n")
        #print(selev.selFatJet.particleNet_HbbvsQCD)
        #print(ak.unflatten(particleNet_HbbvsQCD_flat_str, ak.num(selev.selFatJet.subjets)))
        #print(selev.selFatJet.btag_string)

        #
        #  Compute Soft drop
        #
        selev["selFatJet","subjetmass"] = (selev.selFatJet.subjets[:,:,0] + selev.selFatJet.subjets[:,:,1]).mass

        selev["selFatJet","subjetdr"]    = (selev.selFatJet.subjets[:,:,0].delta_r(selev.selFatJet.subjets[:,:,1]))
        selev["selFatJet","subjetpt0"]   = (selev.selFatJet.subjets[:,:,0].pt)
        selev["selFatJet","subjetpt1"]   = (selev.selFatJet.subjets[:,:,1].pt)


        # print( "softdrop mass",selev[0:5].selFatJet.msoftdrop.tolist(),"\n")
        # print( "subjet mass",selev[0:5].selFatJet.subjetmass.tolist(),"\n")
        #print( "subjet pt0",selev[0:5].selFatJet.subjets[:,:,0].pt.tolist(),"\n")
        #print( "subjet pt1",selev[0:5].selFatJet.subjets[:,:,1].pt.tolist(),"\n")

        #
        # Adding btag and jet flavor to fat jets
        #
        particleNet_HbbvsQCD_flat = ak.flatten(selev.selFatJet.particleNet_HbbvsQCD)
        particleNet_HbbvsQCD_flat_str = [ str(round(v,3)) for v in particleNet_HbbvsQCD_flat ]
        #selev["selFatJet", "btag_string"] = ak.unflatten(particleNet_HbbvsQCD_flat_str, ak.num(selev.selFatJet))


        indices = []
        indices_str = []
        for arr in selev.selFatJet.pt:
            indices_str.append( [f"({i},{i})" for i in range(len(arr))] )
            indices.append(list(range(len(arr))))

        selev["selFatJet", "btag_string"] = indices_str

        selev["selFatJet", "btagScore"] = selev.selFatJet.particleNetMD_Xbb


        fatjet_flavor_flat = np.array(['b'] * len(particleNet_HbbvsQCD_flat))
        selev["selFatJet", "jet_flavor"] = ak.unflatten(fatjet_flavor_flat, ak.num(selev.selFatJet))

        # ───────────── build per-sub-jet helper arrays ─────────────
        subj            = selev.selFatJet.subjets            # (evt,fj,sj)

        # 1) numeric scores  → fill None, round, stringify
        btag_num        = ak.fill_none(subj.pt, -2.0)
        flat_all        = ak.flatten(btag_num, axis=None)    # 1-D
        flat_all_str    = ak.Array(np.char.mod('%.3f',
                                               np.round(ak.to_numpy(flat_all), 3)))

        # 2) unflatten back to (evt,fj,sj)
        counts_sj       = ak.flatten(ak.num(subj, axis=2))
        lvl2            = ak.unflatten(flat_all_str, counts_sj)        # (evt*fj ,sj)
        counts_fj       = ak.num(selev.selFatJet)
        btag_string     = ak.unflatten(lvl2, counts_fj)                # (evt,fj,sj)

        # 3) dummy jet-flavor (same ragged shape, constant "b")
        jet_flavor      = ak.full_like(btag_string, "b")

        # ───────────── expose them in the event record ──────────────
        #    They sit alongside selFatJet so later code can do e.g.
        #    selev.subjet_btag_string[:, :, 1]   (same indices)
        selev["subjet_btag_string"] = btag_string
        selev["subjet_jet_flavor"]  = jet_flavor

        # Adding btag and jet flavor to subjets

        #
#        subjet_btagDeepB_flat = ak.flatten(selev.selFatJet.subjets.btagDeepB)
#        subjet_btagDeepB_flat_str = [ str(round(v,3)) for v in subjet_btagDeepB_flat ]
#        print(len(selFatJet.subjets))
#        print(len(subjet_btagDeepB_flat))
        #selFatJet["btag_string"] = ak.unflatten(particleNet_HbbvsQCD_flat_str, ak.num(selFatJet))

        #has_nan = np.any(np.isnan(selev.selFatJet.subjets[:, :, 0].pt.to_numpy()))
        #print("pt has_nan", has_nan, "\n")
        #print("is None loop",   np.any([ v == None for v in rotated_pt_A_pos_dphi.tolist()]), "\n")

#        print("fat jet pt0",    np.any([ v == None for v in selev.selFatJet.pt.to_numpy().tolist()]), "\n")
#        print("fat jet eta0",   np.any([ v == None for v in selev.selFatJet.eta.to_numpy().tolist()]), "\n")
#        print("fat jet phi0",   np.any([ v == None for v in selev.selFatJet.phi.to_numpy().tolist()]), "\n")
#        print("fat jet mass0",  np.any([ v == None for v in selev.selFatJet.mass.to_numpy().tolist()]), "\n")
#
#
#        print("pt0",    np.any([ v == None for v in selev.selFatJet.subjets[:, :, 0].pt.to_numpy().tolist()]), "\n")
#        print("eta0",   np.any([ v == None for v in selev.selFatJet.subjets[:, :, 0].eta.to_numpy().tolist()]), "\n")
#        print("phi0",   np.any([ v == None for v in selev.selFatJet.subjets[:, :, 0].phi.to_numpy().tolist()]), "\n")
#        print("mass0",  np.any([ v == None for v in selev.selFatJet.subjets[:, :, 0].mass.to_numpy().tolist()]), "\n")
#
#        print("pt0",    np.any([ v == None for v in selev.selFatJet.subjets[:, :, 1].pt.to_numpy().tolist()]), "\n")
#        print("eta0",   np.any([ v == None for v in selev.selFatJet.subjets[:, :, 1].eta.to_numpy().tolist()]), "\n")
#        print("phi0",   np.any([ v == None for v in selev.selFatJet.subjets[:, :, 1].phi.to_numpy().tolist()]), "\n")
#        print("mass0",  np.any([ v == None for v in selev.selFatJet.subjets[:, :, 1].mass.to_numpy().tolist()]), "\n")


        swap_flag_flat = (ak.flatten(selev.selFatJet.subjets [:, :, 0].pt) < ak.flatten(selev.selFatJet.subjets [:, :, 1].pt))

        subjet_lead_pt           = ak.flatten(selev.selFatJet.subjets [:, :, 0].pt).to_numpy()
        subjet_lead_eta          = ak.flatten(selev.selFatJet.subjets [:, :, 0].eta).to_numpy()
        subjet_lead_phi          = ak.flatten(selev.selFatJet.subjets [:, :, 0].phi).to_numpy()
        subjet_lead_mass         = ak.flatten(selev.selFatJet.subjets [:, :, 0].mass).to_numpy()
        subjet_lead_flavor       = ak.flatten(selev.subjet_jet_flavor [:, :, 0]).to_numpy()
        subjet_lead_btag_string  = ak.flatten(selev.subjet_btag_string[:, :, 0]).to_numpy()
        #subjet_lead_tau21        = ak.flatten(selev.selFatJet.subjets [:, :, 0].tau2 / (selev.selFatJet.subjets [:, :, 0].tau1 + 0.001)).to_numpy()

        subjet_lead_pt         [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 1].pt)   [swap_flag_flat]
        subjet_lead_eta        [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 1].eta)  [swap_flag_flat]
        subjet_lead_phi        [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 1].phi)  [swap_flag_flat]
        subjet_lead_mass       [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 1].mass) [swap_flag_flat]
        subjet_lead_flavor     [swap_flag_flat]  = ak.flatten(selev.subjet_jet_flavor [:, :, 1])      [swap_flag_flat]
        subjet_lead_btag_string[swap_flag_flat]  = ak.flatten(selev.subjet_btag_string[:, :, 1])      [swap_flag_flat]
        #subjet_lead_tau21      [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 1].tau2 / (selev.selFatJet.subjets [:, :, 1].tau1 + 0.001)) [swap_flag_flat]

        subjet_subl_pt           = ak.flatten(selev.selFatJet.subjets [:, :, 1].pt).to_numpy()
        subjet_subl_eta          = ak.flatten(selev.selFatJet.subjets [:, :, 1].eta).to_numpy()
        subjet_subl_phi          = ak.flatten(selev.selFatJet.subjets [:, :, 1].phi).to_numpy()
        subjet_subl_mass         = ak.flatten(selev.selFatJet.subjets [:, :, 1].mass).to_numpy()
        subjet_subl_flavor       = ak.flatten(selev.subjet_jet_flavor [:, :, 1]).to_numpy()
        subjet_subl_btag_string  = ak.flatten(selev.subjet_btag_string[:, :, 1]).to_numpy()
        #subjet_subl_tau21        = ak.flatten(selev.selFatJet.subjets [:, :, 1].tau2 / (selev.selFatJet.subjets [:, :, 1].tau1 + 0.001)).to_numpy()

        subjet_subl_pt         [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 0].pt)   [swap_flag_flat]
        subjet_subl_eta        [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 0].eta)  [swap_flag_flat]
        subjet_subl_phi        [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 0].phi)  [swap_flag_flat]
        subjet_subl_mass       [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 0].mass) [swap_flag_flat]
        subjet_subl_flavor     [swap_flag_flat]  = ak.flatten(selev.subjet_jet_flavor [:, :, 0])      [swap_flag_flat]
        subjet_subl_btag_string[swap_flag_flat]  = ak.flatten(selev.subjet_btag_string[:, :, 0])      [swap_flag_flat]
        #subjet_subl_tau21      [swap_flag_flat]  = ak.flatten(selev.selFatJet.subjets [:, :, 0].tau2 / (selev.selFatJet.subjets [:, :, 0].tau1 + 0.001)) [swap_flag_flat]


        # Create the PtEtaPhiMLorentzVectorArray
        fat_jet_splittings_events = ak.zip(
            {
                "pt":   (selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).pt  ,
                "eta":  (selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).eta ,
                "phi":  (selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).phi ,
                "mass": (selev.selFatJet.subjets [:, :, 0] + selev.selFatJet.subjets [:, :, 1]).mass,
                "jet_flavor": selev.selFatJet.jet_flavor,
                "btag_string": selev.selFatJet.btag_string,

                "part_A": ak.zip(
                    {
                        "pt":          ak.unflatten(subjet_lead_pt, ak.num(selev.selFatJet.subjets)),
                        "eta":         ak.unflatten(subjet_lead_eta, ak.num(selev.selFatJet.subjets)),
                        "phi":         ak.unflatten(subjet_lead_phi, ak.num(selev.selFatJet.subjets)),
                        "mass":        ak.unflatten(subjet_lead_mass, ak.num(selev.selFatJet.subjets)),
                        "jet_flavor":  ak.unflatten(subjet_lead_flavor, ak.num(selev.selFatJet.subjets)),
                        "btag_string": ak.unflatten(subjet_lead_btag_string, ak.num(selev.selFatJet.subjets)),
                        #"tau21":       ak.unflatten(subjet_lead_tau21, ak.num(selev.selFatJet.subjets)),
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.backends.awkward.behavior
                ),

                "part_B": ak.zip(
                    {
                        "pt":          ak.unflatten(subjet_subl_pt, ak.num(selev.selFatJet.subjets)),
                        "eta":         ak.unflatten(subjet_subl_eta, ak.num(selev.selFatJet.subjets)),
                        "phi":         ak.unflatten(subjet_subl_phi, ak.num(selev.selFatJet.subjets)),
                        "mass":        ak.unflatten(subjet_subl_mass, ak.num(selev.selFatJet.subjets)),
                        "jet_flavor":  ak.unflatten(subjet_subl_flavor, ak.num(selev.selFatJet.subjets)),
                        "btag_string": ak.unflatten(subjet_subl_btag_string, ak.num(selev.selFatJet.subjets)),
                        #"tau21":       ak.unflatten(subjet_subl_tau21, ak.num(selev.selFatJet.subjets)),
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.backends.awkward.behavior
                ),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.backends.awkward.behavior
        )

        # Look at this function
        compute_decluster_variables(fat_jet_splittings_events)

        fat_jet_splittings_events["splitting_name"] = "1b0j/1b0j"

        #
        # Sort clusterings by type
        #
        selev["splitting_1b0j/1b0j"]   = fat_jet_splittings_events

        fat_jet_splittings_events_low_mass = fat_jet_splittings_events[fat_jet_splittings_events.mass_AB < 75.0]
        selev["splitting_1b0j/1b0j_lowMass"]   = fat_jet_splittings_events_low_mass

        fat_jet_splittings_events_mid_mass = fat_jet_splittings_events[(fat_jet_splittings_events.mass_AB > 75.0) & (fat_jet_splittings_events.mass_AB < 200.0)]
        selev["splitting_1b0j/1b0j_midMass"]   = fat_jet_splittings_events_mid_mass

        fat_jet_splittings_events_high_mass = fat_jet_splittings_events[fat_jet_splittings_events.mass_AB > 200.0]
        selev["splitting_1b0j/1b0j_highMass"]   = fat_jet_splittings_events_high_mass


        dr_partA = selev["splitting_1b0j/1b0j"].delta_r(selev["splitting_1b0j/1b0j"].part_A)
        bad_match_A = ak.any(dr_partA > 1.0, axis=1)

        dr_partB = selev["splitting_1b0j/1b0j"].delta_r(selev["splitting_1b0j/1b0j"].part_B)
        bad_match_B = ak.any(dr_partB > 1.0, axis=1)

        # print(f"selev['splitting_1b0j/1b0j'].part_A.pt: {selev['splitting_1b0j/1b0j'].part_A.pt.tolist()[0:10]}")
        # print(f"selev['splitting_1b0j/1b0j'].part_A.tau21: {selev['splitting_1b0j/1b0j'].part_A.tau21.tolist()[0:10]}")

        bad_match_flag = bad_match_A | bad_match_B

        # print(f"RAW Bad Match A: {bad_match_A}")
        # print(f"RAW Bad Match B: {bad_match_B}")
        # print(f"RAW Bad Match OR: {bad_match_B | bad_match_A}")

        if ak.sum(bad_match_flag) > 0:

            print(f"Found {ak.sum(bad_match_flag)} bad matches in {len(selev['splitting_1b0j/1b0j'])} events")

            bad_splitting = selev["splitting_1b0j/1b0j"][bad_match_flag]
            badFatJet  = selev.selFatJet[bad_match_flag]
            dr_partA = dr_partA[bad_match_flag]
            dr_partB = dr_partB[bad_match_flag]

            print("zA",bad_splitting.zA.tolist()[0:10],"\n")
            #print("zA_num",selev["splitting_1b0j/1b0j"].zA_num.tolist(),"\n")
            print("fatJets\n\t  pt",badFatJet.pt,"\n\t subjetIdx1",badFatJet.subJetIdx1, "\n\t subjetIdx1",badFatJet.subJetIdx2,"\n")
            print("comb\n\t     pt:",bad_splitting.pt[0:10],        "\n\t eta:", bad_splitting.eta[0:10], "\n\t phi:", bad_splitting.phi[0:10],"\n")
            print("\tpart A\n\t pt:",bad_splitting.part_A.pt[0:10], "\n\t eta:", bad_splitting.part_A.eta[0:10], "\n\t phi:", bad_splitting.part_A.phi[0:10],"\n\t dr:", dr_partA.tolist()[0:10],"\n")
            print("\tpart B\n\t pt:",bad_splitting.part_B.pt[0:10], "\n\t eta:", bad_splitting.part_B.eta[0:10], "\n\t phi:", bad_splitting.part_B.phi[0:10],"\n\t dr:", dr_partB.tolist()[0:10],"\n")



        #        # ------------------------------------------------------------
#        # ❶  Grab the two part-level Lorentz‐vector branches
#        # ------------------------------------------------------------
#        vec_A = fat_jet_splittings_events.part_A      # (evt , fj , sj=0)
#        vec_B = fat_jet_splittings_events.part_B      # (evt , fj , sj=1)
#
#        # ------------------------------------------------------------
#        # ❷  Print their field lists – jet_flavor & btag_string
#        #     must be present here
#        # ------------------------------------------------------------
#        print("\n══ VECTOR BRANCH FIELD LISTS ══")
#        print("part_A fields :", vec_A.fields)
#        print("part_B fields :", vec_B.fields)
#
#        # ------------------------------------------------------------
#        # ❸  Peek at the first two fat-jets worth of strings
#        #     to eyeball that the values look right
#        # ------------------------------------------------------------
#        print("\n══ FIRST TWO FJ – STRING CHECK ══")
#        print("part_A.btag_string :", vec_A.btag_string[:2].to_list())
#        print("part_B.btag_string :", vec_B.btag_string[:2].to_list())
#        print("part_A.jet_flavor  :", vec_A.jet_flavor[:2].to_list())
#        print("part_B.jet_flavor  :", vec_B.jet_flavor[:2].to_list())

        #
        # Better Hists
        #

        # Hacking the tag variable
        selev["fourTag"] = True
        selev['tag'] = ak.zip({
            "fourTag": selev.fourTag,
        })


        # Hack the region varable
        selev["SR"] = True
        selev["region"] = ak.zip({
            "SR": selev.SR,
        })

        #selev.sel



        fill = Fill(process=processName, year=year, weight="weight")
        histCuts = ["passNFatJets"]

        hist = Collection( process=[processName],
                           year=[year],
                           tag=["fourTag"],  # 3 / 4/ Other
                           region=['SR'],  # SR / SB / Other
                           **dict((s, ...) for s in histCuts)
                           )


        #print(f" SubJets: {selev.selFatJet.subjets}")
        #print(f" SubJets fields: {selev.selFatJet.subjets.fields}")
        #selev["selFatJet_subjets"] = selev.selFatJet.subjets
        #print(f" SubJets pt: {selev.selFatJet_subjets.pt}")

        #
        # Jets
        #
        fill += Jet.plot(("fatJets", "Selected Fat Jets"),        "selFatJet",           skip=["deepjet_c"], bins={"pt": (50, 0, 1000)})

        #                 "Histogram Name", (nBins, min, max, (variable (selev.variable), title) )
        fill += hist.add( "msoftdrop",  (100, 40, 400, ("selFatJet.msoftdrop",   'Soft Drop Mass')))

        fill += hist.add( "msubjet",    (100, 40, 400, ("selFatJet.subjetmass",  'Sub Jet Mass')))
        fill += hist.add( "subjetdr",   (100, 0, 1.0, ("selFatJet.subjetdr",    'Sub Jet Delta R')))
        fill += hist.add( "subjetpt0",  (100, 0, 400, ("selFatJet.subjetpt0",   'Sub Jet0 Pt')))
        fill += hist.add( "subjetpt1",  (100, 0, 400, ("selFatJet.subjetpt1",   'Sub Jet1 Pt')))

        # print(f" SubJets pt {selev.selFatJet_subjets.pt[0:5]}\n")
        # fill += Jet.plot(("subJets", "Selected Fat Jet SubJet"),   "selFatJet_subjets",  skip=["deepjet_c","deepjet_b","id_pileup","id_jet","n"], bins={"pt": (50, 0, 1000)})

#        print("filling splitting_bb for", len(fat_jet_splittings_events), "events")



        fill += ClusterHistsBoosted( ("splitting_1b0j/1b0j", "1b0j/1b0j Splitting"), "splitting_1b0j/1b0j" )
        fill += ClusterHistsDetailedBoosted( ("detail_splitting_1b0j/1b0j", "1b0j/1b0j Splitting"), "splitting_1b0j/1b0j")


        fill += ClusterHistsBoosted( ("splitting_1b0j/1b0j_lowMass", "1b0j/1b0j Splitting (low Mass"), "splitting_1b0j/1b0j_lowMass" )
        fill += ClusterHistsDetailedBoosted( ("detail_splitting_1b0j/1b0j_lowMass", "1b0j/1b0j Splitting (low Mass)"), "splitting_1b0j/1b0j_lowMass")

        fill += ClusterHistsBoosted( ("splitting_1b0j/1b0j_midMass", "1b0j/1b0j Splitting (mid Mass"), "splitting_1b0j/1b0j_midMass" )
        fill += ClusterHistsDetailedBoosted( ("detail_splitting_1b0j/1b0j_midMass", "1b0j/1b0j Splitting (mid Mass)"), "splitting_1b0j/1b0j_midMass")

        fill += ClusterHistsBoosted( ("splitting_1b0j/1b0j_highMass", "1b0j/1b0j Splitting (high Mass"), "splitting_1b0j/1b0j_highMass" )
        fill += ClusterHistsDetailedBoosted( ("detail_splitting_1b0j/1b0j_highMass", "1b0j/1b0j Splitting (high Mass)"), "splitting_1b0j/1b0j_highMass")

        #
        # fill histograms
        #
        fill(selev, hist)

        processOutput = {}
        self.cutFlow.addOutput(processOutput, event.metadata["dataset"])

        output = hist.output | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
