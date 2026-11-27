"""Regression test for two-stage closure variance, bias, and status outputs."""

import argparse
import json
import os
import sys
import unittest

import yaml


ARGS = None


def parse_args():
    """Parse CLI options while preserving remaining args for unittest."""
    parser = argparse.ArgumentParser(
        description="Verify runTwoStageClosure variance/bias YAML and status JSON outputs."
    )
    parser.add_argument(
        "--closure-dir",
        required=True,
        help="Path to closure_fits/<mix>/<classifier>/rebin<N> containing <region>/<channel>/ outputs.",
    )
    parser.add_argument(
        "--ref-dir",
        default="coffea4bees/stats_analysis/tests",
        help="Directory containing known_0_variance / known_1_bias reference YAML files.",
    )
    parser.add_argument("--channel", default="HH4b", help="Analysis channel subdirectory name.")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["SR", "SB"],
        help="Regions to verify against reference files.",
    )
    parser.add_argument(
        "--chi2-delta",
        type=float,
        default=2.0,
        help="Allowed absolute tolerance for chi2 values.",
    )
    parser.add_argument(
        "--float-delta",
        type=float,
        default=0.01,
        help="Allowed absolute tolerance for non-chi2 numeric values.",
    )
    ns, remaining = parser.parse_known_args()
    return ns, [sys.argv[0]] + remaining


class TestRunTwoStageClosure(unittest.TestCase):
    """Compare runTwoStageClosure outputs against checked-in CI references."""

    def _compare_values(self, actual, expected, context, key_name=None):
        """Recursively compare numeric, list, and dict values with key-aware tolerances."""
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict, msg=f"{context}: expected dict, got {type(actual).__name__}")
            self.assertEqual(
                set(actual.keys()),
                set(expected.keys()),
                msg=f"{context}: dict keys differ ({sorted(map(str, actual.keys()))} vs {sorted(map(str, expected.keys()))})",
            )
            for k in expected:
                self._compare_values(actual[k], expected[k], f"{context}[{k!r}]", key_name=k)
            return

        if isinstance(expected, list):
            self.assertIsInstance(actual, list, msg=f"{context}: expected list, got {type(actual).__name__}")
            self.assertEqual(
                len(actual),
                len(expected),
                msg=f"{context}: list length differs ({len(actual)} vs {len(expected)})",
            )
            for idx, (a_item, e_item) in enumerate(zip(actual, expected)):
                self._compare_values(a_item, e_item, f"{context}[{idx}]", key_name=key_name)
            return

        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            self.assertIsInstance(
                actual,
                (int, float),
                msg=f"{context}: expected numeric value, got {type(actual).__name__}",
            )
            self.assertFalse(isinstance(actual, bool), msg=f"{context}: unexpected bool value")
            delta = ARGS.chi2_delta if key_name == "chi2" else ARGS.float_delta
            self.assertAlmostEqual(
                actual,
                expected,
                delta=delta,
                msg=f"{context}: {actual} != {expected} (diff={actual - expected}, delta={delta})",
            )
            return

        self.assertEqual(actual, expected, msg=f"{context}: {actual!r} != {expected!r}")

    def test_variance_and_bias_yaml(self):
        """Check 0_variance_results.yml and 1_bias_results.yml match reference YAMLs."""
        for region in ARGS.regions:
            pairs = (
                (
                    "0_variance_results.yml",
                    f"known_0_variance_results_HH4b_bkg_syst_ci_{region}.yml",
                ),
                (
                    "1_bias_results.yml",
                    f"known_1_bias_results_HH4b_bkg_syst_ci_{region}.yml",
                ),
            )
            for out_name, ref_name in pairs:
                test_path = os.path.join(ARGS.closure_dir, region, ARGS.channel, out_name)
                ref_path = os.path.join(ARGS.ref_dir, ref_name)
                self.assertTrue(os.path.exists(test_path), msg=f"Missing test output: {test_path}")
                self.assertTrue(os.path.exists(ref_path), msg=f"Missing reference file: {ref_path}")

                with open(test_path, "r") as f_test:
                    test_data = yaml.safe_load(f_test)
                with open(ref_path, "r") as f_ref:
                    ref_data = yaml.safe_load(f_ref)

                self._compare_values(test_data, ref_data, f"{region}/{out_name}")

    def test_closure_status(self):
        """Verify closure_status.json quality flags and selected basis."""
        for region in ARGS.regions:
            status_path = os.path.join(ARGS.closure_dir, region, ARGS.channel, "closure_status.json")
            self.assertTrue(os.path.exists(status_path), msg=f"Missing status file: {status_path}")

            with open(status_path, "r") as f_status:
                status = json.load(f_status)

            ctx = f"{region}/closure_status.json"
            self.assertEqual(status.get("channel"), ARGS.channel, msg=f"{ctx}: unexpected channel")
            self.assertEqual(status.get("var"), "SvB_MA_ps_hh_fine", msg=f"{ctx}: unexpected var")
            self.assertEqual(status.get("rebin"), 12, msg=f"{ctx}: unexpected rebin")
            self.assertIs(status.get("passed"), True, msg=f"{ctx}: passed is not True")
            self.assertIs(status.get("variance_passed"), True, msg=f"{ctx}: variance_passed is not True")
            self.assertIs(status.get("bias_passed"), True, msg=f"{ctx}: bias_passed is not True")
            self.assertIs(
                status.get("spurious_signal_passed"),
                True,
                msg=f"{ctx}: spurious_signal_passed is not True",
            )
            self.assertEqual(status.get("failed_steps"), [], msg=f"{ctx}: unexpected failed_steps")
            self.assertEqual(status.get("selected_basis"), 0, msg=f"{ctx}: unexpected selected_basis")


if __name__ == "__main__":
    ARGS, unittest_argv = parse_args()
    unittest.main(argv=unittest_argv, verbosity=2)
