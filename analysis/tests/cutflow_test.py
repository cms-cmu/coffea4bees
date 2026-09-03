import unittest
import argparse
from coffea.util import load
import yaml
from parser import wrapper
import sys

#_PERCENT_ERROR_THRESHOLD =  0.001
#_PERCENT_ERROR_THRESHOLD =  0.01

#
# python3 analysis/tests/cutflow_test.py   --inputFile output/test.coffea --knownCounts analysis/tests/known_Counts.yml
#
class CutFlowTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.error_threshold = float(wrapper.args["error_threshold"])

        inputFile = wrapper.args["inputFile"]
        hists = load(inputFile)

        self.cf4      = hists["cutFlowFourTag"]
        self.cf4_unit = hists["cutFlowFourTagUnitWeight"]
        self.cf3      = hists["cutFlowThreeTag"]
        self.cf3_unit = hists["cutFlowThreeTagUnitWeight"]
        self.obs_counts = [self.cf4, self.cf3, self.cf4_unit, self.cf3_unit]


        #  Make these numbers with:
        #  >  python     analysis/tests/dumpCutFlow.py --input [inputFileName] -o [outputFielName]
        #       (python analysis/tests/dumpCutFlow.py --input output/histAll.coffea -o analysis/tests/histAllCounts.yml )
        #
        knownCountFile = wrapper.args["knownCounts"]
        self.knownCounts = yaml.safe_load(open(knownCountFile, 'r'))
        self.cf_names = ["counts4", "counts3", "counts4_unit", "counts3_unit"]

    def get_failures(self, expected, observed):
        failures = []
        for datasetAndEra in expected.keys():
            if not expected.get(datasetAndEra):
                continue
            if datasetAndEra not in observed:
                continue
            for cut, exp in expected[datasetAndEra].items():
                if cut not in observed[datasetAndEra]:
                    continue
                obs = round(float(observed[datasetAndEra][cut]), 2)
                diff = obs - exp
                percent_diff = diff / exp if exp else 0
                if abs(percent_diff) > self.error_threshold:
                    failures.append( (datasetAndEra, cut, obs, exp) )
        return failures


    def print_test_results(self, failures):
        print()
        print()
        for k, v in failures.items():
            if len(v):
                print(f'{"":40} {"cut":^20} {"observed":^20} {"expected":^20} {"Fail fraction":^20}  {"Absolute Difference":^20} ')
                print(f"Failed {k}:")
                for datasetAndEra, cut, obs, exp in v:
                    percentFail = (obs - exp) / exp if exp else -1
                    print(f"\t{datasetAndEra:^40} {cut:^20} {obs:^10} {exp:^10} {str(round(percentFail,4)):^20} {str(round(obs - exp,2)):^20} ")
        print()
        print()

    def test_counts(self):
        """
        Test the cutflow for all events
        """
        failures = {}
        for name, count in zip(self.cf_names, self.obs_counts):
            failures[name] = self.get_failures(self.knownCounts[name], count)

        self.print_test_results(failures)

        #
        # Pass/Fail test
        #
        for name in self.cf_names:
            self.assertIs(len(failures[name]),0,f'incorrect number of {name} for {failures[name]} ')


if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)
