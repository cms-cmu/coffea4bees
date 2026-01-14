# Specialized subclass for 4b analysis
from hist import Hist
import numpy as np
import awkward as ak
from src.skimmer.cutflow import cutflow

class cutflow_4b(cutflow):
    def __init__(self, do_truth_hists=False):

        self._cutFlowTwoTag = {}
        self._cutFlowThreeTag = {}
        self._cutFlowFourTag  = {}

        if do_truth_hists:
            self._hists  = {}
        else:
            self._hists  = None

    def fill(self, cut, event, allTag=False, wOverride=None):
        if cut not in self._cutFlowFourTag:
            self._cutFlowTwoTag  [cut] = (0, 0)    # weighted, raw
            self._cutFlowThreeTag[cut] = (0, 0)    # weighted, raw
            self._cutFlowFourTag [cut] = (0, 0)    # weighted, raw
        if allTag:
            if self._hists is not None:
                m4b = event.truth_v4b.mass
            if isinstance(wOverride, ak.Array):
                sumw = float(np.sum(wOverride))
                m4b_weights = wOverride
            else:
                sumw = float(np.sum(event.weight))
                m4b_weights = event.weight
            sumn_2, sumn_3, sumn_4 = len(event), len(event), len(event)
            sumw_2, sumw_3, sumw_4 = sumw, sumw, sumw
        else:
            e2, e3, e4 = event[event.twoTag], event[event.threeTag], event[event.fourTag]
            if isinstance(wOverride, ak.Array):
                e2.weight = wOverride[event.twoTag]
                e3.weight = wOverride[event.threeTag]
                e4.weight = wOverride[event.fourTag]
            if self._hists is not None:
                m4b = e4.truth_v4b.mass
            m4b_weights      = e4.weight
            sumw_2 = float(np.sum(e2.weight))
            sumn_2 = len(e2.weight)
            sumw_3 = float(np.sum(e3.weight))
            sumn_3 = len(e3.weight)
            sumw_4 = float(np.sum(e4.weight))
            sumn_4 = len(e4.weight)

        self._cutFlowTwoTag  [cut] = (sumw_2, sumn_2)     # weighted, raw
        self._cutFlowThreeTag[cut] = (sumw_3, sumn_3)     # weighted, raw
        self._cutFlowFourTag [cut] = (sumw_4, sumn_4)     # weighted, raw
        if self._hists is not None:
            self._hists[cut] = Hist.new.Reg(120, 0, 1200, name="mass", label="Values").Weight()
            self._hists[cut].fill(mass=m4b, weight=m4b_weights)

    def addOutput(self, o, dataset):
        o["cutFlowFourTag"] = {}
        o["cutFlowFourTagUnitWeight"] = {}
        o["cutFlowFourTag"][dataset] = {}
        o["cutFlowFourTagUnitWeight"][dataset] = {}
        for k, v in  self._cutFlowFourTag.items():
            o["cutFlowFourTag"][dataset][k] = v[0]
            o["cutFlowFourTagUnitWeight"][dataset][k] = v[1]

        o["cutFlowThreeTag"] = {}
        o["cutFlowThreeTagUnitWeight"] = {}
        o["cutFlowThreeTag"][dataset] = {}
        o["cutFlowThreeTagUnitWeight"][dataset] = {}
        for k, v in  self._cutFlowThreeTag.items():
            o["cutFlowThreeTag"][dataset][k] = v[0]
            o["cutFlowThreeTagUnitWeight"][dataset][k] = v[1]


        o["cutFlowTwoTag"] = {}
        o["cutFlowTwoTagUnitWeight"] = {}
        o["cutFlowTwoTag"][dataset] = {}
        o["cutFlowTwoTagUnitWeight"][dataset] = {}
        for k, v in  self._cutFlowTwoTag.items():
            o["cutFlowTwoTag"][dataset][k] = v[0]
            o["cutFlowTwoTagUnitWeight"][dataset][k] = v[1]


        if self._hists is not None:
            o["cutflow_hists"] = {}
            o["cutflow_hists"][dataset] = {}
            for k, v in  self._hists.items():
                o["cutflow_hists"][dataset][k] = v
        return

    def addOutputSkim(self, o, dataset):
        o[dataset]["cutFlowFourTag"]           = {}
        o[dataset]["cutFlowFourTagUnitWeight"] = {}
        for k, v in  self._cutFlowFourTag.items():
            o[dataset]["cutFlowFourTag"][k]           = v[0]
            o[dataset]["cutFlowFourTagUnitWeight"][k] = v[1]

        o[dataset]["cutFlowThreeTag"] = {}
        o[dataset]["cutFlowThreeTagUnitWeight"] = {}
        for k, v in  self._cutFlowThreeTag.items():
            o[dataset]["cutFlowThreeTag"][k] = v[0]
            o[dataset]["cutFlowThreeTagUnitWeight"][k] = v[1]

        o[dataset]["cutFlowTwoTag"] = {}
        o[dataset]["cutFlowTwoTagUnitWeight"] = {}
        for k, v in  self._cutFlowTwoTag.items():
            o[dataset]["cutFlowTwoTag"][k] = v[0]
            o[dataset]["cutFlowTwoTagUnitWeight"][k] = v[1]

        return
