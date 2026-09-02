import unittest
import sys
import os
sys.path.insert(0, os.getcwd())
from coffea4bees.stats_analysis.tests.parser import wrapper
import ROOT
import numpy as np
import yaml

class TestRunTwoStageClosure(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        #
        # make output
        #

        # This is the output from runTwoStageClosure
        #    time python3 runTwoStageClosure.py
        #
        self.output_path = wrapper.args["output_path"]
        inputFileName = wrapper.args["inputFile"] 
        #inputFileName = self.output_path+"/hists_closure_3bDvTMix4bDvT_SvB_MA_ps_hh_rebin20.root"
        self.inputROOTFile = ROOT.TFile(inputFileName, "READ")

        #  Make these numbers with:
        #  >  python3     coffea4bees/stats_analysis/tests/dumpTwoStageInputs.py --input [inputFileName] -o [outputFielName]
        #    (python3 coffea4bees/stats_analysis/tests/dumpTwoStageInputs.py --input coffea4bees/stats_analysis/hists_closure_3bDvTMix4bDvT_New.root --output coffea4bees/stats_analysis/tests/twoStageClosureInputsCounts.yml)
        knownCountFile = wrapper.args["knownCounts"] 
        self.knownCounts = yaml.safe_load(open(knownCountFile, 'r'))
        print(self.knownCounts.keys())




    def test_input_counts(self):


        for k, v  in self.knownCounts.items():
            channel = v["channel"]
            process = v["process"]
            expected_counts  =  v["counts"]


            isMixed = "mix" in v
            if isMixed: 
                mix = v["mix"]
                print(f"checking .. {mix}/{channel}/{process}")

                input_hist = self.inputROOTFile.Get(f"{mix}/{channel}/{process}")

            else:
                print(f"checking.. {channel}/{process}")
                input_hist = self.inputROOTFile.Get(f"{channel}/{process}")


            intput_counts = []
            for ibin in range(input_hist.GetSize()):
                intput_counts.append(input_hist.GetBinContent(ibin))

            np.testing.assert_array_equal(intput_counts, expected_counts)



    def check_dict_for_differences(self, lhs, rhs, k):
        delta = 2.0 if k == "chi2" else 0.001
        if type(lhs[k]) == list:
            for listIdx in range(len(lhs[k])):
                self.assertAlmostEqual(lhs[k][listIdx], rhs[k][listIdx], delta=0.001, msg=f"Failed match {k}... {lhs[k]} vs {rhs[k]} ... Diff: {lhs[k][listIdx]-rhs[k][listIdx]}")
        else:
            self.assertAlmostEqual(lhs[k], rhs[k], delta=delta, msg=f"Failed match {k}... {lhs[k]} vs {rhs[k]} ... Diff: {lhs[k]-rhs[k]}")





    def test_yaml_content(self):
        
        for test_pair in [
                ('coffea4bees/stats_analysis/tests/known_0_variance_results_SvB_MA_ps_hh.yml', f'{self.output_path}/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/0_variance_results.yml'),
                ('coffea4bees/stats_analysis/tests/known_1_bias_results_SvB_MA_ps_hh.yml',     f'{self.output_path}/3bDvTMix4bDvT/SvB_MA/rebin1/SR/hh/1_bias_results.yml')
        ]:
            
            reference_file = test_pair[0]
            test_file      = test_pair[1]
            print("\ntesting",test_file, "vs", reference_file)
            
            # Load the content of the test YAML file
            with open(test_file, 'r') as file:
                test_data = yaml.safe_load(file)
        
            # Load the content of the reference YAML file
            with open(reference_file, 'r') as file:
                reference_data = yaml.safe_load(file)

            for k, v in test_data.items():

                if type(v) is dict:
                    for k1, v1 in v.items():                    
                        self.check_dict_for_differences(test_data[k], reference_data[k], k1)

                else:
                    self.check_dict_for_differences(test_data, reference_data, k)


if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)


