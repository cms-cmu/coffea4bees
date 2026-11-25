"""Regression test comparing JCM ROOT-input fits against the reference YAML."""

import os
import subprocess
import sys
import tempfile
import unittest

import yaml


class JCMRootRegressionTestCase(unittest.TestCase):
    """Run make_jcm_weights --ROOTInputs and compare output to checked-in reference."""

    def test_root_inputs_regression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = os.path.join(tmpdir, "testJCM_ROOT")
            subprocess.run(
                [
                    sys.executable,
                    "coffea4bees/analysis/jcm_tools/make_jcm_weights.py",
                    "-o",
                    outdir,
                    "-r",
                    "SB",
                    "--ROOTInputs",
                    "--jcm_config",
                    "coffea4bees/analysis/jcm_tools/metadata/test_jcm_config.yml",
                    "-i",
                    "coffea4bees/analysis/tests/HistsFromROOTFile.coffea",
                ],
                check=True,
            )

            out_yml = os.path.join(outdir, "jetCombinatoricModel_SB_.yml")
            ref_yml = "coffea4bees/analysis/tests/jetCombinatoricModel_SB_ROOT_new.yml"
            with open(out_yml) as f:
                test_data = yaml.safe_load(f)
            with open(ref_yml) as f:
                reference_data = yaml.safe_load(f)

            for k, v in test_data.items():
                if k not in reference_data:
                    continue
                if isinstance(v, list):
                    for idx, item in enumerate(v):
                        if isinstance(item, (int, float)) and not isinstance(item, bool):
                            self.assertAlmostEqual(
                                item,
                                reference_data[k][idx],
                                delta=0.001,
                                msg=f"Failed match {k}[{idx}]: {item} vs {reference_data[k][idx]}",
                            )
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    self.assertAlmostEqual(
                        v,
                        reference_data[k],
                        delta=0.001,
                        msg=f"Failed match {k}: {v} vs {reference_data[k]}",
                    )


if __name__ == "__main__":
    unittest.main()
