from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("coffea4bees"))

from coffea4bees.classifier.config.dataset.HCR.FvT import Train, _d3_selection
from src.classifier.config.setting.df import Columns
from src.classifier.df.tools import prescale


def test_d3_prescale_cli_option():
    default_ds = Train.__new__(Train)
    default_ds.opts = Train.argparser.parse_args([])
    assert default_ds.opts.d3_prescale == "1"

    scaled_ds = Train.__new__(Train)
    scaled_ds.opts = Train.argparser.parse_args(["--d3-prescale", "5"])
    assert scaled_ds.opts.d3_prescale == "5"


def test_d3_prescale_preserves_weights_and_fourtag():
    rng = np.random.default_rng(12345)
    n_d3 = 5000
    n_d4 = 500
    is_three = np.concatenate([np.ones(n_d3, dtype=bool), np.zeros(n_d4, dtype=bool)])
    is_four = ~is_three
    events = np.arange(1, n_d3 + n_d4 + 1, dtype=np.uint64)
    weights = np.concatenate([
        rng.uniform(0.05, 0.25, size=n_d3),
        np.ones(n_d4),
    ])

    df = pd.DataFrame({
        Columns.event: events,
        Columns.weight: weights,
        "threeTag": is_three,
        "fourTag": is_four,
    })

    proc_a = prescale(scale="5", selection=_d3_selection, seed=("d3", 0))
    proc_b = prescale(scale="5", selection=_d3_selection, seed=("d3", 0))

    out_a = proc_a(df.copy())
    out_b = proc_b(df.copy())

    pd.testing.assert_frame_equal(out_a, out_b)

    assert int(out_a["fourTag"].sum()) == n_d4
    assert np.isclose(out_a.loc[out_a["fourTag"], Columns.weight].sum(), df.loc[df["fourTag"], Columns.weight].sum())

    kept_d3 = int(out_a["threeTag"].sum())
    assert 850 < kept_d3 < 1150
    assert np.isclose(out_a.loc[out_a["threeTag"], Columns.weight].sum(), df.loc[df["threeTag"], Columns.weight].sum())
