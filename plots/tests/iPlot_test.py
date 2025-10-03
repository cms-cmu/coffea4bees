import unittest
import argparse
from coffea.util import load
import yaml
import sys


import os
sys.path.insert(0, os.getcwd())

from coffea4bees.plots.iPlot import plot, plot2d, cfg
from coffea4bees.plots.plots import load_config_4b
from src.plotting.plots import load_hists, read_axes_and_cuts
from coffea4bees.analysis.tests.parser import wrapper


class iPlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.inputFile = wrapper.args["inputFile"]


    def do_plots(self):

        args     = {"var": "v4j.*", "region": "SR", "cut": "passPreSel"}
        doRatio  = {"doRatio": 1}
        norm     = {"norm": 1}
        logy     = {"yscale": "log"}
        rlim     = {"rlim": [0, 2]}
        rebin    = {"rebin": 4}
        add_flow = {"add_flow": True}

        print(f"plot with {args}")
        plot(**args)
        args["var"] = "v4j.mass"

        print(f"plot with {args}")
        plot(**args)

        print(f"plot with {args | doRatio}")
        plot(**(args | doRatio))

        print(f"plot with {args | doRatio| add_flow}")
        plot(**(args | doRatio| add_flow))


        print(f"plot with {args | norm}")
        plot(**(args | norm))

        print(f"plot with {args | logy}")
        plot(**(args | logy))

        print(f"plot with {args | rebin}")
        plot(**(args | rebin))

        print(f"plot with {args | doRatio | norm}")
        plot(**(args | doRatio | norm))

        print(f"plot with {args | doRatio | norm | rlim}")
        plot(**(args | doRatio | norm | rlim))

        print(f"plot with {args | doRatio | norm | rlim | rebin}")
        plot(**(args | doRatio | norm | rlim | rebin))

        manyCuts = {"cut": ["passPreSel", "failSvB", "passSvB"],
                    "process": "data"}
        print(f"plot with {args | doRatio | norm | rlim | manyCuts}")
        plot(**(args | doRatio | norm | rlim | manyCuts))

        manyRegions = {"cut": ["passPreSel", "failSvB", "passSvB"],
                       "process": "data"}
        print(f"plot with {args | doRatio | norm | rlim | manyRegions}")
        plot(**(args | doRatio | norm | rlim | manyRegions))

        args2d = {"var": "quadJet_min_dr.close_vs_other_m", "region": "SR",
                  "cut": "passPreSel", "process": "data"}
        full = {"full": True}
        print(f"plot with {args2d}")
        plot2d(**args2d)

        print(f"plot with {args2d | full}")
        plot2d(**(args2d | full))

        manyProcs = {"cut": "passPreSel",
                     "process": ["data","HH4b","TTbar"]}
        print(f"plot with {args | doRatio | norm | rlim | manyProcs}")
        plot(**(args | doRatio | norm | rlim | manyProcs))

        manyVars = {"cut": "passPreSel",
                    "var": ["canJet0.pt","canJet1.pt","canJet2.pt","canJet3.pt"],
                    "process": "data"}
        print(f"plot with {args | doRatio | norm | rlim | manyVars}")
        plot(**(args | doRatio | norm | rlim | manyVars))


        manyYears = {"cut": "passPreSel",
                     "var": "canJet0.pt",
                     "year":  ["UL16_preVFP","UL16_postVFP","UL17","UL18"],
                     "process": "data"}
        print(f"plot with {args | doRatio | norm | rlim | manyYears}")
        plot(**(args | doRatio | norm | rlim | manyYears))


        args["var"] = "v4j.mass"
        invalid_region = {"region": "InvalidRegion"}
        print(f"plot with {args | invalid_region}")
        self.assertIsNone(plot(**(args | invalid_region)))

        invalid_cut    = {"cut": "InvalidCut"}
        print(f"plot with {args | invalid_cut}")
        self.assertIsNone(plot(**(args | invalid_cut)))


    def test_singleFile(self):

        metadata = "coffea4bees/plots/metadata/plotsAll.yml"
        cfg.plotConfig = load_config_4b(metadata)

        input_files = [self.inputFile]
        cfg.hists = load_hists(input_files)

        cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists,
                                                                 cfg.plotConfig)
        cfg.set_hist_key("hists")

        self.do_plots()



    def test_multipleFiles(self):

        metadata = "coffea4bees/plots/metadata/plotsAll.yml"
        cfg.plotConfig = load_config_4b(metadata)

        input_files = [self.inputFile, self.inputFile]
        cfg.hists = load_hists(input_files)
        cfg.fileLabels = ["file1", "file2"]

        cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists,
                                                                 cfg.plotConfig)
        cfg.set_hist_key("hists")

        args    = {"var": "v4j.*", "region": "SR",
                   "cut": "passPreSel", "process": "data"}

        doRatio = {"doRatio": 1}
        norm    = {"norm": 1}
        logy    = {"yscale": "log"}
        rlim    = {"rlim": [0, 2]}

        print(f"plot with {args}")
        plot(**args)
        args["var"] = "v4j.mass"

        print(f"plot with {args}")
        plot(**args)

        print(f"plot with {args | doRatio}")
        plot(**(args | doRatio))

        print(f"plot with {args | norm}")
        plot(**(args | norm))

        print(f"plot with {args | logy}")
        plot(**(args | logy))

        print(f"plot with {args | doRatio | norm}")
        plot(**(args | doRatio | norm))

        print(f"plot with {args | doRatio | norm | rlim}")
        plot(**(args | doRatio | norm | rlim))



    def test_NoFvT(self):

        metadata = "coffea4bees/plots/metadata/plotsAllNoFvT.yml"
        cfg.plotConfig = load_config_4b(metadata)

        input_files = [self.inputFile]
        cfg.hists = load_hists(input_files)

        cfg.axisLabelsDict, cfg.cutListDict = read_axes_and_cuts(cfg.hists,
                                                                 cfg.plotConfig)
        cfg.set_hist_key("hists")

        self.do_plots()



if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)
