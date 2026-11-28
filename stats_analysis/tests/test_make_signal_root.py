"""Unit tests for coffea4bees/stats_analysis/make_signal_root.py."""

import os
import sys
import unittest

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "coffea4bees"))

from coffea4bees.stats_analysis.make_signal_root import find_key


class TestFindKey(unittest.TestCase):
    """Verify SvB vs SvB_MA key resolution in make_signal_root.find_key."""

    def test_svb_does_not_match_svb_ma(self):
        self.assertIsNone(find_key("SvB_ps_ttHbb", ["SvB_MA.ps_ttHbb"]))
        self.assertIsNone(find_key("SvB_ps", ["SvB_MA.ps"]))

    def test_svb_matches_exact_svb(self):
        self.assertEqual(
            find_key("SvB_ps_ttHbb", ["SvB_MA.ps_ttHbb", "SvB.ps_ttHbb"]),
            "SvB.ps_ttHbb",
        )
        self.assertEqual(
            find_key("SvB_ps", ["SvB_MA.ps", "SvB.ps"]),
            "SvB.ps",
        )

    def test_svb_ma_matches_exact_svb_ma(self):
        self.assertEqual(
            find_key("SvB_MA_ps_ttHbb", ["SvB_MA.ps_ttHbb", "SvB.ps_ttHbb"]),
            "SvB_MA.ps_ttHbb",
        )


if __name__ == "__main__":
    unittest.main()
