import unittest
import sys
import os
#sys.path.insert(0, os.getcwd())

#from src.JCMTools import ncr  # Replace 'your_module' with the name of your Python file without the .py extension
#
#class TestNCR(unittest.TestCase):
#    def test_known_values(self):
#        self.assertEqual(ncr(5, 2), 10)
#        self.assertEqual(ncr(6, 3), 20)
#        self.assertEqual(ncr(10, 0), 1)
#        self.assertEqual(ncr(0, 0), 1)
#
#    def test_symmetry(self):
#        n = 10
#        for r in range(n+1):
#            self.assertEqual(ncr(n, r), ncr(n, n-r))
#
#    def test_negative_values(self):
#        with self.assertRaises(ValueError):  # Assuming you want to raise a ValueError for negative inputs
#            ncr(-1, 5)
#        with self.assertRaises(ValueError):  # Adjust according to the behavior you expect
#            ncr(5, -1)


import yaml
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--path', type=str, default="output/analysis-make-weights-job",  help='Path to the files')
    return parser
        
class TestJCM(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        #
        # make output
        #
        #os.system("python analysis/make_weights_ROOT.py -o testJCM_ROOT_tests -c passPreSel -r SB --ROOTInputs")
        #os.system("python analysis/make_weights_ROOT.py -o testJCM_Coffea_tests -c passPreSel -r SB ")

        #
        #  > From python python/analysis/tests/dumpROOTToHist.py -o python/analysis/tests/HistsFromROOTFile.coffea -c passPreSel -r SB
        #
        parser = create_parser()
        self.args = parser.parse_args()

    def test_yaml_content(self):

        base_path = self.args.path

        for test_pair in [#('analysis/testJCM_ROOT_tests/jetCombinatoricModel_SB_.yml', 'analysis/tests/jetCombinatoricModel_SB_ROOT.yml'),
                            #('analysis/testJCM_Coffea_tests/jetCombinatoricModel_SB_.yml', 'analysis/tests/jetCombinatoricModel_SB_Coffea.yml'),
                            (f'{base_path}/testJCM_ROOT/jetCombinatoricModel_SB_.yml',   'coffea4bees/analysis/tests/jetCombinatoricModel_SB_ROOT_new.yml'),
                            (f'{base_path}/testJCM_Coffea/jetCombinatoricModel_SB_.yml', 'coffea4bees/analysis/tests/jetCombinatoricModel_SB_Coffea_new.yml'),
                        ]:

            test_file      = test_pair[0]
            reference_file = test_pair[1]
            print("testing",test_file)

            # Load the content of the test YAML file
            with open(test_file, 'r') as file:
                test_data = yaml.safe_load(file)

            # Load the content of the reference YAML file
            with open(reference_file, 'r') as file:
                reference_data = yaml.safe_load(file)

            for k, v in test_data.items():
                if k not in reference_data:
                    continue  # skip keys not in reference (new outputs from updated code)
                if type(v) == list:
                    for listIdx in range(len(v)):
                        try:
                            diff = v[listIdx] - reference_data[k][listIdx]
                            self.assertAlmostEqual(v[listIdx], reference_data[k][listIdx], delta=0.001, msg=f"Failed match {k}... {v} vs {reference_data[k]} ... Diff: {diff}")
                        except TypeError:
                            pass  # skip non-numeric values (e.g. strings)
                else:
                    try:
                        diff = v - reference_data[k]
                        self.assertAlmostEqual(v, reference_data[k], delta=0.001, msg=f"Failed match {k}... {v} vs {reference_data[k]} ... Diff: {diff}")
                    except TypeError:
                        pass  # skip non-numeric values (e.g. strings)


if __name__ == '__main__':
    parser = create_parser()
    args, unknown = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0]] + unknown)

