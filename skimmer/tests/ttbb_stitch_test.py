"""
Tests for the tt+bb stitching.

Run from the barista root, inside the container::

    ./run_container python -m pytest coffea4bees/skimmer/tests/ttbb_stitch_test.py -v

The central assertion is the one that defines the stitching: replacing the tt+B
events of the inclusive ttbar sample with the tt+B events of the dedicated TTbb
sample, rescaled by k = sumw_ttB(inclusive) / sumw_ttB(TTbb), leaves the total
sum of genWeight unchanged.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from coffea4bees.analysis.helpers.ttbar_categories import (  # noqa: E402
    is_ttB,
    is_ttC,
    is_ttLF,
    ttbar_category_mask,
)
from coffea4bees.skimmer.tools.make_stitched_dataset import build  # noqa: E402

FACTORS = Path("coffea4bees/skimmer/metadata/ttbb_stitch_factors.json")

# Every genTtbarId value that can carry meaning, plus the boundaries just outside
# the two heavy-flavour windows and the negative mirror of each.
GEN_TTBAR_IDS = np.array(
    sorted(
        {0, 1, 40, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56, 60, 99}
        | {100, 141, 151, 155, 156, 200, 251, 255}
        | {-0, -41, -45, -46, -51, -55, -56, -151, -255}
    ),
    dtype=np.int64,
)


def reference_mask(genTtbarId, ttBId):
    """Verbatim transcription of the analysis' reference categorization.

    Kept independent of the implementation under test on purpose, so that the
    helper is checked against the definition rather than against itself.
    """
    if ttBId == "ttB":
        return (
            ((abs(genTtbarId) % 100) == 51)
            | ((abs(genTtbarId) % 100) == 52)
            | ((abs(genTtbarId) % 100) == 53)
            | ((abs(genTtbarId) % 100) == 54)
            | ((abs(genTtbarId) % 100) == 55)
        )
    if ttBId == "ttC":
        return (
            ((abs(genTtbarId) % 100) == 41)
            | ((abs(genTtbarId) % 100) == 42)
            | ((abs(genTtbarId) % 100) == 43)
            | ((abs(genTtbarId) % 100) == 44)
            | ((abs(genTtbarId) % 100) == 45)
        )
    if ttBId == "ttLF":
        return ~(
            ((abs(genTtbarId) % 100) == 41)
            | ((abs(genTtbarId) % 100) == 42)
            | ((abs(genTtbarId) % 100) == 43)
            | ((abs(genTtbarId) % 100) == 44)
            | ((abs(genTtbarId) % 100) == 45)
            | ((abs(genTtbarId) % 100) == 51)
            | ((abs(genTtbarId) % 100) == 52)
            | ((abs(genTtbarId) % 100) == 53)
            | ((abs(genTtbarId) % 100) == 54)
            | ((abs(genTtbarId) % 100) == 55)
        )
    raise AssertionError(ttBId)


# --------------------------------------------------------------------------
# categorization
# --------------------------------------------------------------------------


@pytest.mark.parametrize("category", ["ttB", "ttC", "ttLF"])
def test_matches_reference_categorization(category):
    np.testing.assert_array_equal(
        ttbar_category_mask(GEN_TTBAR_IDS, category),
        reference_mask(GEN_TTBAR_IDS, category),
        err_msg=f"{category} disagrees with the reference definition",
    )


def test_categories_partition_the_sample():
    b, c, lf = (f(GEN_TTBAR_IDS) for f in (is_ttB, is_ttC, is_ttLF))
    assert not (b & c).any()
    assert not (b & lf).any()
    assert not (c & lf).any()
    np.testing.assert_array_equal(b | c | lf, np.ones_like(b))


def test_notB_is_the_complement_of_ttB():
    np.testing.assert_array_equal(
        ttbar_category_mask(GEN_TTBAR_IDS, "notB"), ~is_ttB(GEN_TTBAR_IDS)
    )
    # notB is what the inclusive side keeps, so it must be exactly ttC | ttLF.
    np.testing.assert_array_equal(
        ttbar_category_mask(GEN_TTBAR_IDS, "notB"),
        is_ttC(GEN_TTBAR_IDS) | is_ttLF(GEN_TTBAR_IDS),
    )


def test_abs_is_applied_before_the_modulo():
    """``genTtbarId % 100`` is *not* a valid substitute for ``abs(...) % 100``."""
    negative = np.array([-51, -52, -55], dtype=np.int64)
    assert is_ttB(negative).all()
    # numpy's modulo maps -51 -> 49, which would land outside the ttB window.
    naive = np.isin(negative % 100, [51, 52, 53, 54, 55])
    assert not naive.any()


def test_known_values():
    assert is_ttB(np.array([51, 52, 53, 54, 55])).all()
    assert is_ttC(np.array([41, 42, 43, 44, 45])).all()
    assert is_ttLF(np.array([0, 40, 46, 50, 56, 99])).all()
    # the leading digits encode the ttbar decay, only the last two matter here
    assert is_ttB(np.array([151, 255])).all()
    assert is_ttC(np.array([141, 241])).all()


def test_unknown_category_rejected():
    with pytest.raises(ValueError):
        ttbar_category_mask(GEN_TTBAR_IDS, "ttbb")


# --------------------------------------------------------------------------
# genWeight bookkeeping
# --------------------------------------------------------------------------


def _synthetic_sample(n, seed, ttbar_ids, weight_scale):
    rng = np.random.default_rng(seed)
    gid = rng.choice(ttbar_ids, size=n)
    gw = rng.normal(loc=weight_scale, scale=0.1 * weight_scale, size=n)
    return gid, gw


def test_stitching_conserves_sumw_on_synthetic_samples():
    """The construction must preserve the inclusive sum of genWeight exactly."""
    incl_gid, incl_gw = _synthetic_sample(
        200_000, seed=1, ttbar_ids=[0, 41, 45, 51, 55], weight_scale=300.0
    )
    # A dedicated sample: mostly ttB, far fewer events, much smaller weights.
    ttbb_gid, ttbb_gw = _synthetic_sample(
        40_000, seed=2, ttbar_ids=[0, 45, 51, 52, 55], weight_scale=20.0
    )

    incl_B = is_ttB(incl_gid)
    ttbb_B = is_ttB(ttbb_gid)

    k = incl_gw[incl_B].sum() / ttbb_gw[ttbb_B].sum()

    sumw_before = incl_gw.sum()
    sumw_after = incl_gw[~incl_B].sum() + k * ttbb_gw[ttbb_B].sum()

    assert sumw_after == pytest.approx(sumw_before, rel=1e-12)
    # ... and the tt+B piece alone carries the removed weight.
    assert (k * ttbb_gw[ttbb_B].sum()) == pytest.approx(
        incl_gw[incl_B].sum(), rel=1e-12
    )


def test_float32_storage_preserves_sumw():
    """genWeight is written back as float32; the sum must survive that."""
    rng = np.random.default_rng(3)
    gw = rng.normal(loc=20.0, scale=2.0, size=1_000_000)
    k = 25.4185
    exact = (gw.astype(np.float64) * k).sum()
    stored = (gw * k).astype(np.float32).astype(np.float64).sum()
    assert stored == pytest.approx(exact, rel=1e-6)


def test_a_wrong_scale_breaks_the_closure():
    """Guard against the closure check silently passing on a bad scale factor."""
    incl_gid, incl_gw = _synthetic_sample(
        50_000, seed=4, ttbar_ids=[0, 41, 51, 55], weight_scale=300.0
    )
    ttbb_gid, ttbb_gw = _synthetic_sample(
        10_000, seed=5, ttbar_ids=[51, 52, 55], weight_scale=20.0
    )
    incl_B, ttbb_B = is_ttB(incl_gid), is_ttB(ttbb_gid)
    bad_k = 1.05 * incl_gw[incl_B].sum() / ttbb_gw[ttbb_B].sum()
    sumw_after = incl_gw[~incl_B].sum() + bad_k * ttbb_gw[ttbb_B].sum()
    assert sumw_after != pytest.approx(incl_gw.sum(), rel=1e-6)


# --------------------------------------------------------------------------
# the measured factors and the stage-3 merge
# --------------------------------------------------------------------------


@pytest.mark.skipif(not FACTORS.exists(), reason=f"{FACTORS} not produced yet")
def test_measured_factors_close_on_real_picoaods():
    """Closure of the real, measured picoAOD genWeight sums per channel and era."""
    factors = json.load(open(FACTORS))
    assert factors["stitch"], "no (channel, era) entries in the factors file"

    for channel, eras in factors["stitch"].items():
        for era, spec in eras.items():
            where = f"{channel}/{era}"

            # k is exactly the ratio of the two tt+B genWeight sums ...
            assert spec["scale"] == pytest.approx(
                spec["sumw_inclusive_ttB"] / spec["sumw_ttbb_ttB"], rel=1e-12
            ), where

            # ... and the rescaled tt+B piece restores the removed weight.
            assert spec["scale"] * spec["sumw_ttbb_ttB"] == pytest.approx(
                spec["sumw_inclusive_ttB"], rel=1e-12
            ), where

            # sum of genWeight before vs after stitching
            assert spec["sumw_stitched_expected"] == pytest.approx(
                spec["sumw_inclusive_total"], rel=1e-12
            ), where

            # the two kept pieces must add up to the inclusive total
            assert (
                spec["sumw_inclusive_notB"] + spec["sumw_inclusive_ttB"]
            ) == pytest.approx(spec["sumw_inclusive_total"], rel=1e-12), where

            assert spec["scale"] > 0, where
            assert spec["sumw_inclusive_notB"] > 0, where


@pytest.mark.skipif(not FACTORS.exists(), reason=f"{FACTORS} not produced yet")
def test_stage3_merge_accepts_a_consistent_skim_output():
    """build() must accept a closing stage-2 output and merge both file lists."""
    factors = json.load(open(FACTORS))
    datasets = {}
    skim_out = {}

    for channel, eras in factors["stitch"].items():
        for era, spec in eras.items():
            incl, ttbb = spec["inclusive"], spec["ttbb"]
            datasets.setdefault(incl, {"xs": spec["xs"]})[era] = {
                "picoAOD": {
                    "sumw": spec["metadata_sumw_inclusive"],
                    "sumw2": 1.0,
                    "total_events": 10,
                    "count": 10,
                }
            }
            skim_out[f"{incl}_{era}"] = {
                "files": [f"{incl}_{era}_chunk0.root"],
                "stitch_n_in": 100,
                "stitch_n_out": spec["n_inclusive_notB"],
                "stitch_sumw_in": spec["sumw_inclusive_total"],
                "stitch_sumw_selected_scaled": spec["sumw_inclusive_notB"],
            }
            skim_out[f"{ttbb}_{era}"] = {
                "files": [f"{ttbb}_{era}_chunk0.root"],
                "stitch_n_in": 100,
                "stitch_n_out": spec["n_ttbb_ttB"],
                "stitch_sumw_in": 1.0,
                "stitch_sumw_selected_scaled": spec["scale"] * spec["sumw_ttbb_ttB"],
            }

    out, report, failures = build(factors, skim_out, datasets, "_stitched")

    assert not failures, failures
    assert report
    for name, entry in out.items():
        assert name.endswith("_stitched")
        assert entry["xs"] not in (None, 1), f"{name} did not inherit the inclusive xs"
        for era in (e for e in entry if e != "xs"):
            pico = entry[era]["picoAOD"]
            # the union of both sources, and the inclusive generator-level sumw
            assert len(pico["files"]) == 2
            assert pico["sumw"] > 0
            assert pico["saved_events"] == (
                pico["stitch"]["n_from_inclusive"] + pico["stitch"]["n_from_ttbb"]
            )
            assert abs(pico["stitch"]["closure_rel"]) < 1e-9


@pytest.mark.skipif(not FACTORS.exists(), reason=f"{FACTORS} not produced yet")
def test_stage3_merge_rejects_a_broken_skim_output():
    """A stage-2 output whose sums do not close must be refused, not written."""
    factors = json.load(open(FACTORS))
    channel = next(iter(factors["stitch"]))
    era = next(iter(factors["stitch"][channel]))
    spec = factors["stitch"][channel][era]
    incl, ttbb = spec["inclusive"], spec["ttbb"]

    one = {channel: {era: spec}}
    datasets = {
        incl: {
            "xs": spec["xs"],
            era: {"picoAOD": {"sumw": spec["metadata_sumw_inclusive"]}},
        }
    }
    skim_out = {
        f"{incl}_{era}": {
            "files": ["a.root"],
            "stitch_n_in": 100,
            "stitch_n_out": 10,
            "stitch_sumw_in": spec["sumw_inclusive_total"],
            "stitch_sumw_selected_scaled": spec["sumw_inclusive_notB"],
        },
        f"{ttbb}_{era}": {
            "files": ["b.root"],
            "stitch_n_in": 100,
            "stitch_n_out": 10,
            "stitch_sumw_in": 1.0,
            # 10% off: forgot to apply the scale correctly
            "stitch_sumw_selected_scaled": 1.1 * spec["scale"] * spec["sumw_ttbb_ttB"],
        },
    }

    _out, _report, failures = build({"stitch": one}, skim_out, datasets, "_stitched")
    assert failures, "a 10% normalisation error was not caught"
