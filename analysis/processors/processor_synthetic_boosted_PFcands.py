import time
import awkward as ak
import numpy as np
import yaml
import warnings

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection
import hist

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
                                        cut_on_lumimask=True
                                        )



        selFatJet = event.FatJet[event.FatJet.pt > 300]
        selFatJet = selFatJet[ak.num(selFatJet.subjets, axis=2) > 1]

        #print(f" fields FatJets: {selFatJet.fields}")
        #print(f" fields nSubJets: {selFatJet.subjets.fields}")
        #print(f" nSubJets: {ak.num(selFatJet.subjets, axis=1)}")

        #selFatJet = selFatJet[ak.num(selFatJet.subjets) > 1]
        event["selFatJet"] = selFatJet


        #  Cehck How often do we have >=2 Fat Jets?
        event["passNFatJets"]  = (ak.num(event.selFatJet) == 2)








        # Apply object selection (function does not remove events, adds content to objects)

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( "passNFatJets",  event.passNFatJets )
        ### add more selections, this can be useful


        #list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "passNFatJets" ]
        list_of_cuts = [ "passNFatJets" ]
        analysis_selections = selections.all(*list_of_cuts)
        selev = event[analysis_selections]

        #
        # Event selection
        #
        print(f"Number of FatJet PFCands: {ak.num(selev.FatJetPFCands)}")
        print(f"FatJet PFCands fields: {selev.FatJetPFCands.fields}")


        #print(f"Number of selected Fat Jets: {ak.num(selev.selFatJet)}")
        #print(f" Any passNFatJets: {ak.any(selev.passNFatJets)}")
        #print(f" Any passHLT: {ak.any(selev.passHLT)}")
        #print(f" FatJet pt: {selev.selFatJet.pt}")
        #
        #print(f" nSubJets: {ak.num(selev.selFatJet.subjets, axis=2)}")
        #print(f" subjet pt: {selev.selFatJet.pt[0:10]}")

        #print(f" FatJet pt: {selev.selFatJet.pt}")
        #print(f" FatJet Subjet pt pt: {selev.selFatJet.subjets.pt}")
        #print(f" FatJet subjet pt A: {selev.selFatJet.subjets[:,:,0].pt}")
        #print(f" FatJet subjet pt B: {selev.selFatJet.subjets[:,:,1].pt}")
        #print(f" FatJet subjet len(: {len(selev.selFatJet.subjets[:,:,0].pt)}")
        #print(f" SubJets fields: {selev.selFatJet.subjets.fields}")
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
        # Adding btag and jet flavor to fat jets
        #
        # particleNet_HbbvsQCD_flat = ak.flatten(selev.selFatJet.particleNet_HbbvsQCD)
        # particleNet_HbbvsQCD_flat_str = [ str(round(v,3)) for v in particleNet_HbbvsQCD_flat ]
        # selev["selFatJet", "btag_string"] = ak.unflatten(particleNet_HbbvsQCD_flat_str, ak.num(selev.selFatJet))

        # fatjet_flavor_flat = np.array(['b'] * len(particleNet_HbbvsQCD_flat))
        # selev["selFatJet", "jet_flavor"] = ak.unflatten(fatjet_flavor_flat, ak.num(selev.selFatJet))

        # ───────────── build per-sub-jet helper arrays ─────────────
        subj            = selev.selFatJet.subjets            # (evt,fj,sj)

        # 1) numeric scores  → fill None, round, stringify
        btag_num        = ak.fill_none(subj.btagDeepB, -2.0)
        flat_all        = ak.flatten(btag_num, axis=None)    # 1-D
        flat_all_str    = ak.Array(np.char.mod('%.3f',
                                               np.round(ak.to_numpy(flat_all), 3)))

        # 2) unflatten back to (evt,fj,sj)
        counts_sj       = ak.flatten(ak.num(subj, axis=2))
        lvl2            = ak.unflatten(flat_all_str, counts_sj)        # (evt*fj ,sj)
        counts_fj       = ak.num(selev.selFatJet)
        # btag_string     = ak.unflatten(lvl2, counts_fj)                # (evt,fj,sj)

        # 3) dummy jet-flavor (same ragged shape, constant "b")
        # jet_flavor      = ak.full_like(btag_string, "b")

        # ───────────── expose them in the event record ──────────────
        #    They sit alongside selFatJet so later code can do e.g.
        #    selev.subjet_btag_string[:, :, 1]   (same indices)
        # selev["subjet_btag_string"] = btag_string
        # selev["subjet_jet_flavor"]  = jet_flavor

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



        # Create the PtEtaPhiMLorentzVectorArray
        fat_jet_splittings_events = ak.zip(
            {
                "pt":   selev.selFatJet.pt,
                "eta":  selev.selFatJet.eta,
                "phi":  selev.selFatJet.phi,
                "mass": selev.selFatJet.mass,
                # "jet_flavor": selev.selFatJet.jet_flavor,
                # "btag_string": selev.selFatJet.btag_string,
                "part_A": ak.zip(
                    {
                        "pt":          selev.selFatJet.subjets[:, :, 0].pt,
                        "eta":         selev.selFatJet.subjets[:, :, 0].eta,
                        "phi":         selev.selFatJet.subjets[:, :, 0].phi,
                        "mass":        selev.selFatJet.subjets[:, :, 0].mass,
                        # "jet_flavor": selev.subjet_jet_flavor[:, :, 0], # "b"
                        # "btag_string": selev.subjet_btag_string[:, :, 0],
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.backends.awkward.behavior
                ),
                "part_B": ak.zip(
                    {
                        "pt":          selev.selFatJet.subjets[:, :, 1].pt,
                        "eta":         selev.selFatJet.subjets[:, :, 1].eta,
                        "phi":         selev.selFatJet.subjets[:, :, 1].phi,
                        "mass":        selev.selFatJet.subjets[:, :, 1].mass,
                        # "jet_flavor": selev.subjet_jet_flavor[:, :, 1],
                        # "btag_string": selev.subjet_btag_string[:, :, 1],
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.backends.awkward.behavior
                ),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.backends.awkward.behavior
        )

        # Look at this function
        fat_jet_splittings_events = compute_decluster_variables(fat_jet_splittings_events)
#        print("new fields:", fat_jet_splittings_events.fields)

        fat_jet_splittings_events["splitting_name"] = "bb"

        #
        # Sort clusterings by type
        #
        selev["splitting_bb"]   = fat_jet_splittings_events



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

        selev["weight"] = 1.0



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

        # print(f" SubJets pt {selev.selFatJet_subjets.pt[0:5]}\n")
        # fill += Jet.plot(("subJets", "Selected Fat Jet SubJet"),   "selFatJet_subjets",  skip=["deepjet_c","deepjet_b","id_pileup","id_jet","n"], bins={"pt": (50, 0, 1000)})

#        print("filling splitting_bb for", len(fat_jet_splittings_events), "events")


#        for _s_type in cleaned_splitting_name:
        fill += ClusterHistsBoosted( ("splitting_bb", "bb Splitting"), "splitting_bb" )

        #
        # fill histograms
        #
        fill(selev, hist)

        processOutput = {}

        output = hist.output | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
