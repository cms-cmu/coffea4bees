"""Pure unit tests for plotting helpers (migrated from plots_test.py)."""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.getcwd())

from coffea4bees.plots.plots import load_config_4b
import src.plotting.helpers as plot_helpers
from src.plotting.helpers_make_plot_dict import get_values_variances_centers_from_dict


class PlotHelpersTestCase(unittest.TestCase):
    """Unit tests for pure plotting helpers that need no .coffea input."""

    def test_get_value_nested_dict(self):
        test_dict = {
            "dA0": {"dA1": {"dA2": {"ka0": "va0", "ka1": "va1"}}, "ka2": "va2"},
            "dB0": {"kb0": "vb0"},
            "kb1": "vb1",
        }
        self.assertEqual(plot_helpers.get_value_nested_dict(test_dict, "ka0"), "va0")
        self.assertEqual(plot_helpers.get_value_nested_dict(test_dict, "ka1"), "va1")
        self.assertEqual(plot_helpers.get_value_nested_dict(test_dict, "ka2"), "va2")
        self.assertEqual(plot_helpers.get_value_nested_dict(test_dict, "kb0"), "vb0")
        self.assertEqual(plot_helpers.get_value_nested_dict(test_dict, "kb1"), "vb1")

        plot_config = load_config_4b("coffea4bees/plots/metadata/plotsAll.yml")
        self.assertEqual(plot_helpers.get_value_nested_dict(plot_config, "fillcolor"), "k")

    def test_get_cut_dict(self):
        cut = "cut1"
        cut_list = ["cut1", "cut2", "cut3"]
        result = plot_helpers.get_cut_dict(cut, cut_list)

        self.assertIsInstance(result, dict, "The result should be a dictionary.")
        self.assertTrue(all(c in result for c in cut_list), "All cuts should be keys in the result dictionary.")
        self.assertTrue(result[cut], "The specified cut should be True.")
        for c in cut_list:
            if c != cut:
                self.assertTrue(result[c] == sum, f"{c} should be marked as sum.")

    def test_get_values_variances_centers_from_dict_hists_type(self):
        hist_config = {
            "name": "hist1",
            "values": [1, 2, 3],
            "variances": [4, 5, 6],
            "centers": [0.5, 1.5, 2.5],
        }
        plot_data = {"hists": {"hist1": hist_config}, "stack": {}}
        values, variances, centers = get_values_variances_centers_from_dict(
            {"type": "hists", "key": "hist1"},
            plot_data,
        )
        np.testing.assert_array_equal(values, np.array([1, 2, 3]))
        np.testing.assert_array_equal(variances, np.array([4, 5, 6]))
        np.testing.assert_array_equal(centers, np.array([0.5, 1.5, 2.5]))

    def test_make_ratio(self):
        num = np.array([2.0, 4.0])
        num_vars = np.array([1.0, 1.0])
        den = np.array([1.0, 2.0])
        den_vars = np.array([1.0, 1.0])
        ratios, _ = plot_helpers.make_ratio(num.copy(), num_vars.copy(), den.copy(), den_vars.copy())
        np.testing.assert_allclose(ratios, [2.0, 2.0])

        den_zero = np.array([0.0, 0.0])
        ratios_zero, _ = plot_helpers.make_ratio(num.copy(), num_vars.copy(), den_zero.copy(), den_vars.copy())
        self.assertTrue(np.all(np.isnan(ratios_zero)))

        num_n = np.array([6.0, 3.0])
        den_n = np.array([1.0, 1.0])
        ratios_raw, _ = plot_helpers.make_ratio(num_n.copy(), np.ones(2), den_n.copy(), np.ones(2))
        ratios_norm, _ = plot_helpers.make_ratio(num_n.copy(), np.ones(2), den_n.copy(), np.ones(2), norm=True)
        expected_norm = ratios_raw * (np.sum(den_n) / np.sum(num_n))
        np.testing.assert_allclose(ratios_norm, expected_norm, rtol=1e-5)

    def test_stack_type(self):
        hist_mock = MagicMock()
        hist_mock.values.return_value = np.array([1, 2, 3])
        hist_mock.axes[0].centers = np.array([0.5, 1.5, 2.5])
        stack_dummy = {
            "values": [1, 2, 3],
            "variances": [4, 5, 6],
            "centers": [0.5, 1.5, 2.5],
        }
        plot_data = {
            "hists": {},
            "stack": {"stack1": stack_dummy, "stack2": stack_dummy},
        }
        values, variances, centers = get_values_variances_centers_from_dict({"type": "stack"}, plot_data)
        np.testing.assert_array_equal(values, np.array([2, 4, 6]))
        np.testing.assert_array_equal(variances, np.array([8, 10, 12]))
        np.testing.assert_array_equal(centers, np.array([0.5, 1.5, 2.5]))


if __name__ == "__main__":
    unittest.main()
