"""Exact-reproduction test of the processor-side JCM loader for the f(nJet) x r^n model.

The Run 3 ttHbb fNjetR JCM files carry pseudoTagProbSlope (alpha) and looseVetoFactor (r), plus the
fit's own JCM_weights list (weight for nSelJets 4..20). The loader must reproduce that list; a file
without the two keys must give bit-identical weights and pseudo-tag counts to the loader before the
change. That reference is pinned at REF_REV, the last commit that touched the loader before this
change, not at HEAD: once the change is committed, HEAD would compare the new code with itself.

maxPseudoTags: the fit sums every pseudo-tag count, the loader's default keeps npt <= 12. With
maxPseudoTags = 17 the loader reproduces all 17 stored entries (the formula itself); with the default
cap it is exact up to nSelJets 15, and for nSelJets 16..20 (n_light >= 13) the dropped npt > 12 terms
leave a relative residual below 2e-8 (largest for 2024). 3b events with nSelJets >= 16 are < 3e-7 of
the sample, and keeping the default keeps files without alpha/r bit-identical (orchestrator ruling A).

Run from the barista root:  python -m pytest coffea4bees/analysis/tests/test_jcm_fnjetr_loader.py -q
"""
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.special import comb

from coffea4bees.analysis.helpers.jetCombinatoricModel import jetCombinatoricModel

C4B = Path(__file__).resolve().parents[2]
JCM_DIR = C4B / "metadata/weights/JCM/Run3_ttHbb"
FNJETR = [
    "jetCombinatoricModel_SB_run3_2022_preEE_fNjetR_v1.yml",
    "jetCombinatoricModel_SB_run3_2022_EE_fNjetR_v1.yml",
    "jetCombinatoricModel_SB_run3_2023_preBPix_pt25_fNjetR_v1.yml",
    "jetCombinatoricModel_SB_run3_2023_BPix_pt25_fNjetR_v1.yml",
    "jetCombinatoricModel_SB_run3_2024_pt25_fNjetR_v1.yml",
]
BASELINE = "jetCombinatoricModel_SB_run3_2022_v1.yml"  # no alpha / r keys
REF_REV = "c24101b4"  # last commit touching analysis/helpers/jetCombinatoricModel.py before the alpha/r change
RTOL = 1e-9
N_LIGHT = np.arange(1, 18)  # n_light 1..17 = nSelJets 4..20
EVENTS = np.arange(1000, 1000 + len(N_LIGHT), dtype=np.uint64)


def _stored(path):
    with open(path) as f:
        return np.asarray(yaml.safe_load(f)["JCM_weights"], dtype=float)


def _rel_diff(w, stored):
    n = len(stored)
    return np.abs(w[:n] - stored) / np.abs(stored)


@pytest.fixture(scope="module")
def ref_class(tmp_path_factory):
    """The loader as it was at REF_REV, imported from `git show` into a temporary module."""
    src = subprocess.run(["git", "-C", str(C4B), "show", f"{REF_REV}:analysis/helpers/jetCombinatoricModel.py"],
                         check=True, capture_output=True, text=True).stdout
    assert "pseudoTagProbSlope" not in src, "REF_REV already contains the alpha/r change"
    path = tmp_path_factory.mktemp("jcm_ref") / "jcm_ref.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("jcm_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.jetCombinatoricModel


@pytest.mark.parametrize("name", FNJETR + [BASELINE])
def test_reproduces_stored_weights(name):
    """The formula: every stored entry to rtol 1e-9, with no pseudo-tag cap below n_light (fNjetR: 17)."""
    path = JCM_DIR / name
    stored = _stored(path)
    if name != BASELINE:
        jcm = jetCombinatoricModel(str(path), cut=None, maxPseudoTags=17)
        assert jcm.alpha != 0.0 and jcm.r != 1.0, (jcm.alpha, jcm.r)
        assert len(stored) == 17
    else:
        jcm = jetCombinatoricModel(str(path), cut=None)
        assert (jcm.alpha, jcm.r) == (0.0, 1.0)
        assert len(stored) == 12
    w, npt = jcm(N_LIGHT, event=EVENTS)
    rd = _rel_diff(w, stored)
    print(f"{name}: {len(stored)} entries, max relative difference {rd.max():.3e}")
    assert np.all(rd <= RTOL), rd


TRUNCATION_BOUND = 2e-8  # default maxPseudoTags = 12 at nSelJets 16..20 (see the module docstring)


@pytest.mark.parametrize("name", FNJETR)
def test_default_cap_reproduces_stored_weights(name):
    """As the processors use it (default maxPseudoTags = 12): exact up to nSelJets 15, bounded truncation above."""
    path = JCM_DIR / name
    stored = _stored(path)
    w, _ = jetCombinatoricModel(str(path), cut=None)(N_LIGHT, event=EVENTS)
    rd = _rel_diff(w, stored)
    print(f"{name} (default cap): nSelJets 4..15 max {rd[:12].max():.3e}; nSelJets 16..20 {np.array2string(rd[12:], precision=2)}")
    assert np.all(rd[:12] <= RTOL), rd[:12]
    assert np.all(rd[12:] < TRUNCATION_BOUND), rd[12:]
    assert np.all(w[12:] <= stored[12:] * (1 + RTOL)), "the truncation can only remove (positive) npt > 12 terms"


def test_baseline_bit_identical_to_reference(ref_class):
    path = str(JCM_DIR / BASELINE)
    rng = np.random.default_rng(12345)
    n_light = rng.integers(1, 13, size=50000)
    events = rng.integers(0, 2**63, size=50000, dtype=np.uint64)
    for kwargs in ({}, {"zero_npt": True}):
        w_new, npt_new = jetCombinatoricModel(path, cut=None, **kwargs)(n_light, event=events)
        w_ref, npt_ref = ref_class(path, cut=None, **kwargs)(n_light, event=events)
        assert w_new.dtype == w_ref.dtype and npt_new.dtype == npt_ref.dtype
        assert np.array_equal(w_new, w_ref), kwargs  # bit-identical, not approximately equal
        assert np.array_equal(npt_new, npt_ref), kwargs


@pytest.mark.parametrize("key, value", [("looseVetoFactor", 1.0), ("pseudoTagProbSlope", 0.0)])
def test_negative_control_detects_ignored_key(tmp_path, key, value):
    """A loader that ignored r (or alpha) would compute these weights; they must NOT match the stored list.
    maxPseudoTags = 17, as in test_reproduces_stored_weights, so the unmutated file matches and the key is the only difference."""
    src = JCM_DIR / FNJETR[-1]
    data = yaml.safe_load(open(src))
    stored = np.asarray(data["JCM_weights"], dtype=float)
    data[key] = value
    mutated = tmp_path / f"mutated_{key}.yml"
    yaml.safe_dump(data, open(mutated, "w"))
    w, _ = jetCombinatoricModel(str(mutated), cut=None, maxPseudoTags=17)(N_LIGHT, event=EVENTS)
    rd = _rel_diff(w, stored)
    print(f"negative control {key}={value}: max relative difference {rd.max():.3e}")
    assert not np.all(rd <= RTOL)


@pytest.mark.parametrize("name", FNJETR)
def test_npt_range_and_distribution(name):
    path = JCM_DIR / name
    data = yaml.safe_load(open(path))
    jcm = jetCombinatoricModel(str(path), cut=None)
    # range over every n_light
    nl = np.repeat(N_LIGHT, 2000)
    _, npt = jcm(nl, event=np.arange(len(nl), dtype=np.uint64))
    assert np.all(npt >= 1) and np.all(npt <= np.minimum(nl, jcm.maxPseudoTags))
    # distribution at nSelJets 6 (n_light 3) vs the per-npt probabilities, computed independently here
    n_ev, n_light, nbt = 200000, 3, 3
    p = data["pseudoTagProb"] * ((n_light + nbt) / 4.0) ** data["pseudoTagProbSlope"]
    e, d, t = data["pairEnhancement"], data["pairEnhancementDecay"], data["threeTightTagFraction"]
    wk = np.array([t * comb(n_light, k, exact=True) * p**k * (1 - p) ** (n_light - k) * ((1 + e / n_light**d) if (nbt + k) % 2 == 0 else 1.0)
                   for k in range(1, n_light + 1)])
    prob = wk / wk.sum()
    _, npt = jcm(np.full(n_ev, n_light), event=np.arange(10**6, 10**6 + n_ev, dtype=np.uint64))
    counts = np.array([(npt == k).sum() for k in range(1, n_light + 1)])
    assert counts.sum() == n_ev
    sigma = np.sqrt(n_ev * prob * (1 - prob))
    pulls = (counts - n_ev * prob) / sigma
    print(f"{name}: npt fractions {counts / n_ev} vs {prob}, pulls {pulls}")
    assert np.all(np.abs(pulls) < 3), pulls
