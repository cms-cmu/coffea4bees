import os
import time
import gc
import argparse
import sys
from copy import deepcopy
import awkward as ak
import numpy as np
import uproot
import correctionlib
import yaml
import warnings

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector
from coffea import processor

from src.hist import Collection, Fill
from src.hist import H, Template
from src.hist.object import LorentzVector, Jet, Muon, Elec

from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from coffea4bees.analysis.helpers.topCandReconstruction import find_tops, dumpTopCandidateTestVectors, buildTop
from coffea4bees.analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHistsUnsup, WCandHists, TopCandHists

from functools import partial
from multiprocessing import Pool

from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from src.physics.common import apply_btag_sf, update_events
from src.physics.event_selection import apply_event_selection
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
import logging


#
# Setup
#
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")
ak.behavior.update(vector.behavior)

##### Build a new template
# class QuadJetHists(Template):
#     dr              = H((50,     0, 5,   ("dr",          'Diboson Candidate $\\Delta$R(d,d)')))
#     dphi            = H((100, -3.2, 3.2, ("dphi",        'Diboson Candidate $\\Delta$R(d,d)')))
#     deta            = H((100,   -5, 5,   ("deta",        'Diboson Candidate $\\Delta$R(d,d)')))
#     xZZ             = H((100, 0, 10,     ("xZZ",         'Diboson Candidate zZZ')))
#     xZH             = H((100, 0, 10,     ("xZH",         'Diboson Candidate zZH')))
#     xHH             = H((100, 0, 10,     ("xHH",         'Diboson Candidate zHH')))

#     lead_vs_subl_m   = H((50, 0, 250, ('lead.mass', 'Lead Boson Candidate Mass')),
#                          (50, 0, 250, ('subl.mass', 'Subl Boson Candidate Mass')))

#     close_vs_other_m = H((50, 0, 250, ('close.mass', 'Close Boson Candidate Mass')),
#                          (50, 0, 250, ('other.mass', 'Other Boson Candidate Mass')))

#     lead            = LorentzVector.plot_pair(('...', R'Lead Boson Candidate'),  'lead',  skip=['n'])
#     subl            = LorentzVector.plot_pair(('...', R'Subl Boson Candidate'),  'subl',  skip=['n'])
#     close           = LorentzVector.plot_pair(('...', R'Close Boson Candidate'), 'close', skip=['n'])
#     other           = LorentzVector.plot_pair(('...', R'Other Boson Candidate'), 'other', skip=['n'])


class analysis(processor.ProcessorABC):
    def __init__(self, JCM='', threeTag = True, corrections_metadata: dict = None, run_systematics=[], SRno = '4', make_classifier_input: str = ''):
        logging.debug('\nInitialize Analysis Processor')
        self.histCuts = ['passPreSel']
        self.tags = ['threeTag', 'fourTag'] if threeTag else ['fourTag']
        # self.JCM = jetCombinatoricModel(JCM)
        self.corrections_metadata = corrections_metadata
        self.run_systematics = run_systematics
        self.m4jBinEdges = np.array([[0, 361], [361, 425], [425, 479], [479, 533], [533, 591], [591, 658], [658, 741], [741, 854], [854, 1044], [1044, 1800]])
        self.SRno = int(SRno)
        self.m4j_SR = self.m4jBinEdges[self.SRno]
        self.m4j_lowSB = self.m4jBinEdges[int(self.SRno-1)]
        self.m4j_highSB = self.m4jBinEdges[int(self.SRno+1)]
        self.make_classifier_input = make_classifier_input
        logging.info(f'Region SR at {self.SRno} with low SB at {int(self.SRno-1)} and high SB at {int(self.SRno+1)}')
        logging.info(f'm4js are {self.m4j_lowSB[0]}, {self.m4j_SR[0]}, {self.m4j_SR[1]}, {self.m4j_highSB[1]} GeV.')

    def process(self, event):
        tstart = time.time()
        fname   = event.metadata['filename']

        logging.info(f'File path: {fname}')

        dataset = event.metadata['dataset']
        fileuuild = event.metadata['fileuuid']

        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        era     = event.metadata.get('era', '')
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        isMixedData = not (dataset.find("mix_v") == -1)
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        nEvent = len(event)

        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = nEvent
        self._cutFlow = cutflow_4b()

        ###############################################
        ###### Reading 3to4, DtoM friend trees ########
        ###############################################


        path = fname.replace(fname.split('store/user')[-1], '')


        if 'picoAOD_3b_wJCM_newSBDef' in fname:
            fname_w3to4 = f"/smurthy/condor/unsupervised4b/randPair/w3to4hist/data20{year[2:4]}_picoAOD_3b_wJCM_newSBDef_w3to4_hist.root"
            fname_wDtoM = f"/smurthy/condor/unsupervised4b/randPair/wDtoMwJMC/data20{year[2:4]}_picoAOD_3b_wJCM_newSBDef_wDtoM.root"
            event['w3to4'] = NanoEventsFactory.from_root(f'{path}{fname_w3to4}',
                            entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().w3to4.w3to4

            event['wDtoM'] = NanoEventsFactory.from_root(f'{path}{fname_wDtoM}',
                            entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().wDtoM.wDtoM

            #### event['w3to4', 'frac_err'] = event['w3to4'].std / event['w3to4'].w3to4
            # ####### Fix this!!! Giving errors
            # if not ak.all(event.w3to4.event == event.event):
            #     logging.error('ERROR: w3to4 events do not match events ttree')
            #     return
            # if not ak.all(event.wDtoM.event == event.event):
            #     logging.error('ERROR: wDtoM events do not match events ttree')
            #     return


        ##############################################
        ### general event weights
        if isMC:
            ### genWeight
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)
            logging.debug(f"event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw) = {event.genWeight[0]} * ({lumi} * {xs} * {kFactor} / {genEventSumw}) = {event.weight[0]}\n")

            ### trigger Weight (to be updated)
            ###event['weight'] = event.weight * event.trigWeight.Data

            ###puWeight
            if ('Pileup' in event.fields):
                puWeight = list(correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['PU']).values())[0]
                for var in ['nominal', 'up', 'down']:
                    event[f'PU_weight_{var}'] = puWeight.evaluate(event.Pileup.nTrueInt.to_numpy(), var)
                event['weight'] = event.weight * event.PU_weight_nominal
            else:
                logging.info('Pileup weights not found')

            ### L1 prefiring weight
            if ('L1PreFiringWeight' in event.fields):   #### AGE: this should be temprorary (field exists in UL)
                event['weight'] = event.weight * event.L1PreFiringWeight.Nom
        else:
            event['weight'] = 1


        logging.debug(f"event['weight'] = {event.weight}")

        ### Event selection (function only adds flags, not remove events)
        event = apply_event_selection( event, isMC, self.corrections_metadata[year])

        self._cutFlow.fill("all",  event[event.lumimask], allTag=True)
        self._cutFlow.fill("passNoiseFilter",  event[ event.lumimask & event.passNoiseFilter], allTag=True)
        self._cutFlow.fill("passHLT",  event[ event.lumimask & event.passNoiseFilter & event.passHLT], allTag=True)

        ### Apply object selection (function does not remove events, adds content to objects)
        event =  apply_4b_selection( event, year, isMC, dataset, self.corrections_metadata[year], isMixedData=isMixedData  )
        self._cutFlow.fill("passJetMult",  event[ event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult ], allTag=True)

        selections = []
        filters = event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult
        selections.append(filters & event.passPreSel)

        ## Filtering object and event selection
        selev = event[filters]

        ##### Calculate and apply btag scale factors
        if isMC:
            btagSF = correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['btagSF'])['deepJet_shape']
            btag_SF_weights = apply_btag_sf(selev.selJet,
                                            correction_file=self.corrections_metadata[year]['btagSF'],
                                            btag_uncertainties= None )['btagSF_central']
            selev['weight'] = selev.weight * btag_SF_weights

            self._cutFlow.fill("passJetMult_btagSF",  selev, allTag=True)

        ## Preselection: keep only three or four tag events
        selev = selev[selev.passPreSel]

        selev['wNo3to4DtoM'] = selev.weight
        selev['wNo3to4'] = selev.weight

        if self.make_classifier_input is None:
            logging.info('Reweighting w3to4')
            if 'picoAOD_3b_wJCM_newSBDef' in fname:
                selev['wNo3to4'] = selev.wNo3to4DtoM * selev.wDtoM
                selev['weight'] = selev.wNo3to4 * selev.w3to4


        ############################################
        ############## Unsup 4b code ###############
        ############################################

        #### Calculate hT (scalar sum of jet pts) (previously known as St)
        selev['hT']          = ak.sum(selev.Jet[selev.Jet.selected_loose].pt, axis=1)
        selev['hT_selected'] = ak.sum(selev.Jet[selev.Jet.selected      ].pt, axis=1)


        ### Build and select boson candidate jets with bRegCorr applied
        sorted_idx = ak.argsort(selev.Jet.btagScore * selev.Jet.selected, axis=1, ascending=False)
        canJet_idx    = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]
        canJet = selev.Jet[canJet_idx]

        ### apply bJES to canJets
        canJet = canJet * canJet.bRegCorr
        canJet['bRegCorr'] = selev.Jet.bRegCorr[canJet_idx]
        canJet['btagScore'] = selev.Jet.btagScore[canJet_idx]
        canJet['puId'] = selev.Jet.puId[canJet_idx]
        canJet['jetId'] = selev.Jet.puId[canJet_idx]
        if isMC:
            canJet['hadronFlavour'] = selev.Jet.hadronFlavour[canJet_idx]
        if not isMixedData:
            canJet['calibration'] = selev.Jet.calibration[canJet_idx]

        ### pt sort canJets
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        selev['canJet'] = canJet

        ###  Declare candidate jets
        selev['canJet0'] = canJet[:, 0]
        selev['canJet1'] = canJet[:, 1]
        selev['canJet2'] = canJet[:, 2]
        selev['canJet3'] = canJet[:, 3]
        selev['v4j'] = canJet.sum(axis=1)
        selev['m4j'] = selev.v4j.mass


        ### Compute Regions: SR, SB
        selev['SR'] = (self.m4j_SR[0] <= selev.m4j) & (selev.m4j < self.m4j_SR[1])
        selev['SB_low'] = (self.m4j_lowSB[0] <= selev.m4j) & (selev.m4j < self.m4j_lowSB[1])
        selev['SB_high'] = (self.m4j_highSB[0] <= selev.m4j) & (selev.m4j < self.m4j_highSB[1])
        selev['SB'] = selev.SB_low | selev.SB_high
        selev['SRSB'] = selev.SR | selev.SB
        selev['notSRSB'] = ~selev.SRSB


        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        notCanJet['isSelJet'] = 1 * ((notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4))     # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev['notCanJet_coffea'] = notCanJet
        selev['nNotCanJet'] = ak.num(selev.notCanJet_coffea)

        ### Build diJets, indexed by diJet[event,pairing,0/1]
        canJet = selev['canJet']
        pairing = [([0, 2], [0, 1], [0, 1]),
                   ([1, 3], [2, 3], [3, 2])]
        diJet       = canJet[:, pairing[0]]     +   canJet[:, pairing[1]]
        diJet['st'] = canJet[:, pairing[0]].pt  +   canJet[:, pairing[1]].pt
        diJet['dr'] = canJet[:, pairing[0]].delta_r(canJet[:, pairing[1]])
        diJet['dphi'] = canJet[:, pairing[0]].delta_phi(canJet[:, pairing[1]])
        diJet['lead'] = canJet[:, pairing[0]]
        diJet['subl'] = canJet[:, pairing[1]]

        # Sort diJets within views to be lead st, subl st
        diJet   = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
        diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]
        # # Now indexed by diJet[event,pairing,lead/subl st]

        #### Do I want this???
        # Compute diJetMass cut with independent min/max for lead/subl
        minDiJetMass = np.array([[[ 0,  0]]])
        maxDiJetMass = np.array([[[1000, 1000]]])
        diJet['passDiJetMass'] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

        ##### Build quadJets
        seeds = np.array(event.event)[[0, -1]].view(np.ulonglong)
        randomstate = np.random.Generator(np.random.PCG64(seeds))  ###
        quadJet = ak.zip({'lead': diJet[:, :, 0],
                          'subl': diJet[:, :, 1],
                          'close': diJetDr[:, :, 0],
                          'other': diJetDr[:, :, 1],
                          'passDiJetMass': ak.all(diJet.passDiJetMass, axis=2),
                          'random': randomstate.uniform(low=0.1, high=0.9, size=(diJet.__len__(), 3))})

        quadJet['dr']   = quadJet['lead'].delta_r(quadJet['subl'])
        quadJet['dphi'] = quadJet['lead'].delta_phi(quadJet['subl'])
        quadJet['deta'] = quadJet['lead'].eta - quadJet['subl'].eta

        ### pick quadJet at random
        quadJet['rank'] = quadJet.random
        quadJet['selected'] = quadJet.rank == np.max(quadJet.rank, axis=1)

        selev['diJet'] = diJet
        selev['quadJet'] = quadJet
        selev['quadJet_selected'] = quadJet[quadJet.selected][:, 0]
        selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)
        selev['leadStM'] = selev.quadJet_selected.lead.mass
        selev['sublStM'] = selev.quadJet_selected.subl.mass


        selev['region'] = selev.SR*0b01  +selev.notSRSB*0b100 + selev.SB_low*0b10 + selev.SB_high*0b11 # + selev.SB*0b01

        # logging.info(f'{selev.SR}')
        # logging.info(f'{selev.SB}')
        # logging.info(f'{selev.region}')
        # logging.info(f'{selev.region}')
        # logging.info(f'{selev.region}')

        ###  Build the top Candiates
        ### sort the jets by btagging
        selev.selJet  = selev.selJet[ak.argsort(selev.selJet.btagScore, axis=1, ascending=False)]
        selev["nSelJets"] = ak.num(selev.selJet)
        top_cands     = find_tops(selev.selJet)
        rec_top_cands = buildTop(selev.selJet, top_cands)
        # selev["top_cand"] = rec_top_cands[:, 0]
        # selev["xbW_reco"] = selev.top_cand.xbW
        # selev["xW_reco"]  = selev.top_cand.xW
        # selev["delta_xbW"] = selev.xbW - selev.xbW_reco
        # selev["delta_xW"] = selev.xW - selev.xW_reco

        # ####### Fix this!!! Giving errors
        ### Blind data in fourTag SR
        # if not (isMC or 'mixed' in dataset) and self.blind:
        #     selev = selev[~(selev['quadJet_selected'].SR & selev.fourTag)]


        ###########################################################
        ######################### Hists ##########################
        #########################################################

        fill = Fill(process=processName, year=year, weight='weight')

        hist = Collection(process = [processName],
                          year    = [year],
                          tag     = [3, 4, 0],    # 3 / 4/ Other
                          region  = [4, 3, 2, 1, 0],    # SR / SB / Other
                          **dict((s, ...) for s in self.histCuts))

        fill += hist.add('nPVs',     (101, -0.5, 100.5, ('PV.npvs',     'Number of Primary Vertices')))
        fill += hist.add('nPVsGood', (101, -0.5, 100.5, ('PV.npvsGood', 'Number of Good Primary Vertices')))

        fill += hist.add('nJet_selected', (16, 0, 15, ('nJet_selected', 'nJet_selected')))
        fill += hist.add('hT',          (100,  0,   1000,  ('hT',          'H_{T} [GeV}')))
        fill += hist.add('hT_selected', (100,  0,   1000,  ('hT_selected', 'H_{T} (selected jets) [GeV}')))
        fill += LorentzVector.plot_pair(('v4j'), 'v4j', skip=['n', 'dr', 'dphi', 'st'], bins={'mass': (120, 0, 1200)})   ###  Make quad jet hists
        fill += QuadJetHistsUnsup(('quadJet_selected', 'Selected Quad Jet'), 'quadJet_selected')  #### Build a new template

        fill += hist.add('nJet_selected_no3to4', (16, 0, 15, ('nJet_selected', 'nJet_selected no3to4')), weight="wNo3to4")
        fill += hist.add('hT_no3to4',          (100,  0,   1000,  ('hT',          'H_{T} [GeV}')), weight="wNo3to4")
        fill += hist.add('hT_selected_no3to4', (100,  0,   1000,  ('hT_selected', 'H_{T} (selected jets) [GeV}')), weight="wNo3to4")
        fill += LorentzVector.plot_pair(('v4j_no3to4'), 'v4j', skip=['n', 'dr', 'dphi', 'st'], bins={'mass': (120, 0, 1200)}, weight="wNo3to4")
        fill += QuadJetHistsUnsup(('quadJet_selected_no3to4', 'Selected Quad Jet no3to4'), 'quadJet_selected', weight = "wNo3to4")

        fill += hist.add('nJet_selected_no3to4DtoM', (16, 0, 15, ('nJet_selected', 'nJet_selected no3to4DtoM')), weight="wNo3to4DtoM")
        fill += hist.add('hT_no3to4DtoM',          (100,  0,   1000,  ('hT',          'H_{T} [GeV}')), weight="wNo3to4DtoM")
        fill += hist.add('hT_selected_no3to4DtoM', (100,  0,   1000,  ('hT_selected', 'H_{T} (selected jets) [GeV}')), weight="wNo3to4DtoM")
        fill += LorentzVector.plot_pair(('v4j_no3to4DtoM'), 'v4j', skip=['n', 'dr', 'dphi', 'st'], bins={'mass': (120, 0, 1200)}, weight="wNo3to4DtoM")
        fill += QuadJetHistsUnsup(('quadJet_selected_no3to4DtoM', 'Selected Quad Jet no3to4DtoM'), 'quadJet_selected', weight = "wNo3to4DtoM")


        fill += Jet.plot(('selJets', 'Selected Jets'),        'selJet',           skip=['deepjet_c'])
        fill += Jet.plot(('tagJets', 'Tag Jets'),             'tagJet',           skip=['deepjet_c'])
        fill += Jet.plot(('othJets', 'Other Jets'),           'notCanJet_coffea', skip=['deepjet_c'])
        fill += Jet.plot(('canJets', 'Candidate Jets'),        'canJet',           skip=['deepjet_c'])


        ### fill histograms ###
        # fill.cache(selev)
        fill(selev)


        ### CutFlow ###
        self._cutFlow.fill("passPreSel", selev)
        self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
        self._cutFlow.fill("passThreeTag", selev[selev.threeTag])
        self._cutFlow.fill("passFourTag", selev[selev.fourTag])
        self._cutFlow.fill("SR",            selev[(selev.passDiJetMass & selev.SR)])
        self._cutFlow.fill("SB",            selev[(selev.passDiJetMass & selev.SB)])

        garbage = gc.collect()

        ### Done ###
        elapsed = time.time() - tstart
        logging.debug(f'{chunk}{nEvent/elapsed:,.0f} events/s New')
        self._cutFlow.addOutput(processOutput, event.metadata['dataset'])

        friends = {}
        if self.make_classifier_input is not None:
            from coffea4bees.analysis.helpers.dump_friendtrees import dump_unsup_friend
            friends["friends"] = dump_unsup_friend(
                selev,
                self.make_classifier_input,   #### output file: return pathlike
                "classifier_input",
                *selections,
                )

        return hist.output | processOutput | friends




    def postprocess(self, accumulator):
        ...
