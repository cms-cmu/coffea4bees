"""
Parity of the generic CoReNet copy in barista (``src.classifier.nn.corenet``) with the hard-coded
HH -> 4b HCR network (``coffea4bees.classifier.nn.blocks.HCR.HCR``).

CoReNet configured for HH -> 4b (``b: jets btag 4``, ``other: jets rest 8``, ``HH: {H1: [b, b],
H2: [b, b], context: other}``) is loaded with HCR's weights through a fixed renaming of the
state-dict keys (``NAME_MAP``, from CoReNet's tests/reference/test_parity_reference.py) and must
give bit-identical (torch.equal) class and view logits, running statistics, gradients and weights
after optimiser steps, on synthetic events, in training (ghost batches) and eval mode.

Run from the barista root inside the classifier container:

    python -m coffea4bees.classifier.tests.test_corenet_parity
    python -m pytest -q coffea4bees/classifier/tests/test_corenet_parity.py
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from coffea4bees.classifier.nn.blocks.HCR import HCR
from src.classifier.nn.corenet.config import parse_config
from src.classifier.nn.corenet.network import CoReNet

DEVICE = "cpu"
N_EVENTS = 2048
GHOST_BATCHES = 16
LABELS = ["d4", "d3", "t4", "t3"]  # HCR's four classes, in label order
N_CLASSES = len(LABELS)
ANCILLARY = ["xW", "nSelJets", "xbW", "year"]
CONFIGS = [  # (d, attention gate, kv projection)
    (8, True, True),
    (8, False, False),
    (4, True, False),
]

# test_backward tolerances. The two networks assemble the reinforce input differently (HCR slices,
# the generic code gathers by index), so autograd sums the per-pixel contributions in a different
# order; forward values are exact, gradients agree to float32 rounding (observed max relative ~1e-6,
# absolute ~2.4e-8).
GRAD_RTOL = 2e-6
GRAD_ATOL = 1e-8
WEIGHT_RTOL = 2e-6
WEIGHT_ATOL = 1e-7

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

# HCR state-dict prefix -> CoReNet state-dict prefix (CoReNet tests/reference/test_parity_reference.py)
from coffea4bees.classifier.model.corenet_from_hcr import NAME_MAP  # noqa: E402


def make_spec(d, gate, kv, context=True):
    hh = {"H1": ["b", "b"], "H2": ["b", "b"]}
    if context:
        hh["context"] = "other"
    return {
        "objects": {
            "jets": {"features": ["pt", "eta", "phi", "mass"], "max": 12},
            "b": "jets btag 4",
            "other": "jets rest 8",
        },
        "event": list(ANCILLARY),
        "hypotheses": {"HH": hh},
        "labels": list(LABELS),
        "outputs": {"d4": "HH"},  # the output name must be a label; no further outputs -> no aux heads
        "combine": ["HH"],
        "training": {
            "features": d,
            "attention_heads": 2,
            "attention_gate": gate,
            "kv_projection": kv,
            "phase_symmetric": True,
        },
    }


# module-level checks: the spec parses, the class count comes from ``labels:``, no aux heads
_spec = parse_config(make_spec(8, True, True))
_model = CoReNet(_spec, device=DEVICE)
assert _spec.classes == LABELS and _spec.main_class == 0, (_spec.classes, _spec.main_class)
assert _model.nC == N_CLASSES == 4, _model.nC
assert len(_model.aux_heads) == 0, list(_model.aux_heads)
del _spec, _model


# ----------------------------------------------------------------------------------- helpers
def map_key(k):
    src, dst = next((s, d) for s, d in NAME_MAP if k.startswith(s))
    return dst + k[len(src) :]


def build_models(d, gate, kv, context=True, seed=0):
    torch.manual_seed(seed)
    hcr = HCR(
        dijetFeatures=d,
        quadjetFeatures=d,
        ancillaryFeatures=list(ANCILLARY),
        canJetFeatures=("pt", "eta", "phi", "mass"),
        useOthJets="attention" if context else "",
        device=DEVICE,
        nClasses=N_CLASSES,
        use_attention_gate=gate,
        use_kv_proj=kv,
    )
    torch.manual_seed(seed)
    cor = CoReNet(parse_config(make_spec(d, gate, kv, context)), device=DEVICE)
    assert cor.nC == N_CLASSES and len(cor.aux_heads) == 0
    return hcr, cor


def make_data(seed=0, n=N_EVENTS):
    rng = np.random.default_rng(seed)
    # candidate jets, pt-ordered
    pt = rng.uniform(40, 1000, (n, 4))
    eta = rng.uniform(-2.4, 2.4, (n, 4))
    phi = rng.uniform(-math.pi, math.pi, (n, 4))
    mass = pt * rng.uniform(0.03, 0.15, (n, 4))
    order = np.argsort(-pt, axis=1)
    pt, eta, phi, mass = (np.take_along_axis(x, order, 1) for x in (pt, eta, phi, mass))
    j = np.stack([pt, eta, phi, mass], axis=1)  # (n, 4 features, 4 jets)
    # other jets: k present (pt-ordered), the rest padded with -1 in all channels
    k = rng.integers(0, 9, n)
    opt = rng.uniform(20, 1000, (n, 8))
    oeta = rng.uniform(-4.7, 4.7, (n, 8))
    ophi = rng.uniform(-math.pi, math.pi, (n, 8))
    omass = opt * rng.uniform(0.03, 0.15, (n, 8))
    order = np.argsort(-opt, axis=1)
    opt, oeta, ophi, omass = (
        np.take_along_axis(x, order, 1) for x in (opt, oeta, ophi, omass)
    )
    issel = ((opt >= 40) & (np.abs(oeta) < 2.4)).astype(np.float64)
    o = np.stack([opt, oeta, ophi, omass, issel], axis=1)  # (n, 5, 8)
    present = np.arange(8)[None, :] < k[:, None]
    o = np.where(present[:, None, :], o, -1.0)
    flags = o[:, 4, :][present]
    assert (flags == 0).any() and (flags == 1).any(), "isSelJet must take both 0 and 1"
    # ancillary
    xW = rng.uniform(0, 1, n)
    xbW = rng.uniform(0, 1, n)
    nsel = 4 + ((o[:, 4, :] == 1) & present).sum(1)
    year = rng.choice([2016.0, 2017.0, 2018.0], n)
    a = np.stack([xW, nsel, xbW, year], axis=1)
    t = {
        "j": torch.from_numpy(j.astype(np.float32)),
        "o": torch.from_numpy(o.astype(np.float32)),
        "a": torch.from_numpy(a.astype(np.float32)),
    }
    return t


def hcr_args(data):
    n = data["j"].shape[0]
    return (
        data["j"].clone().reshape(n, 16),
        data["o"].clone().reshape(n, 40),
        data["a"].clone(),
    )


def cor_inputs(data):
    j = data["j"].clone()
    return {
        "b": torch.cat([j, torch.ones_like(j[:, :1])], dim=1),
        "other": data["o"].clone(),
        "event": data["a"].clone(),
    }


def assert_equal(x, y, what):
    assert x.shape == y.shape, f"{what}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
    if not torch.equal(x, y):
        diff = (x - y).abs().max().item()
        print(f"    FAIL {what}: max |diff| = {diff:.3e}")
        raise AssertionError(f"{what}: max |diff| = {diff:.3e}")


def max_rel_diff(x, y):
    """Largest |x - y| / |y| over the elements where x and y differ (0.0 if none)."""
    diff = (x - y).abs()
    mask = diff > 0
    if not mask.any():
        return 0.0
    return (diff[mask] / y.abs()[mask]).max().item()


def assert_close(x, y, what, rtol, atol):
    """torch.allclose(x, y, rtol, atol); returns the max relative difference."""
    assert x.shape == y.shape, f"{what}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
    rel = max_rel_diff(x, y)
    if not torch.allclose(x, y, rtol=rtol, atol=atol):
        diff = (x - y).abs().max().item()
        print(f"    FAIL {what}: max |diff| = {diff:.3e}, max rel diff = {rel:.3e}")
        raise AssertionError(f"{what}: max |diff| = {diff:.3e}, max rel diff = {rel:.3e}")
    return rel


def check_mapping(hcr, cor):
    """Bijection of the state dicts through NAME_MAP with equal shapes; returns the key map."""
    hs, cs = hcr.state_dict(), cor.state_dict()
    keymap = {k: map_key(k) for k in hs}
    targets = list(keymap.values())
    assert len(set(targets)) == len(targets), "two HCR keys map to the same CoReNet key"
    missing = sorted(set(cs) - set(targets))
    extra = sorted(set(targets) - set(cs))
    assert not missing and not extra, (
        f"not a bijection: CoReNet keys without HCR source {missing}; mapped keys absent in CoReNet {extra}"
    )
    for hk, ck in keymap.items():
        assert hs[hk].shape == cs[ck].shape, (
            f"{hk} -> {ck}: {tuple(hs[hk].shape)} vs {tuple(cs[ck].shape)}"
        )
    hp, cp = dict(hcr.named_parameters()), dict(cor.named_parameters())
    assert {keymap[k] for k in hp} == set(cp), "parameter sets differ under the map"
    n_h = sum(p.numel() for p in hcr.parameters())
    n_c = sum(p.numel() for p in cor.parameters())
    assert n_h == n_c, (n_h, n_c)
    return keymap, n_h


def load_mapped(hcr, cor, keymap):
    cor.load_state_dict(
        {keymap[k]: v.clone() for k, v in hcr.state_dict().items()}, strict=True
    )


def same_state(hcr, cor, keymap, what):
    hs, cs = hcr.state_dict(), cor.state_dict()
    for hk, ck in keymap.items():
        assert_equal(hs[hk], cs[ck], f"{what}: {hk} / {ck}")


def setup(d, gate, kv, context=True, seed=0):
    """Models with HCR's weights in CoReNet, statistics frozen from the full batch, 16 ghost batches."""
    hcr, cor = build_models(d, gate, kv, context, seed)
    keymap, _ = check_mapping(hcr, cor)
    load_mapped(hcr, cor, keymap)
    data = make_data(seed)
    hcr.train()
    cor.train()
    with torch.no_grad():
        hcr.updateMeanStd(*hcr_args(data))
        cor.updateMeanStd(cor_inputs(data))
    hcr.initMeanStd()
    cor.initMeanStd()
    same_state(hcr, cor, keymap, "after initMeanStd")
    hcr.setGhostBatches(GHOST_BATCHES, False)
    cor.setGhostBatches(GHOST_BATCHES)
    same_state(hcr, cor, keymap, "after setGhostBatches")
    return hcr, cor, keymap, data


def run_both(hcr, cor, data):
    c_logits, q_logits = hcr(*hcr_args(data))
    out = cor(cor_inputs(data))
    return c_logits, q_logits, out


def forward_parity(hcr, cor, keymap, data, tag):
    for i, mode in enumerate(("train", "train", "eval")):
        getattr(hcr, mode)()
        getattr(cor, mode)()
        with torch.no_grad():
            c_logits, q_logits, out = run_both(hcr, cor, data)
        assert_equal(c_logits, out["logits"], f"{tag} [{mode} {i}] class logits")
        assert_equal(q_logits, out["views"]["HH"], f"{tag} [{mode} {i}] view logits")
        assert out["heads"] == {}, out["heads"].keys()
        same_state(hcr, cor, keymap, f"{tag} [{mode} {i}] state")


def tag_of(d, gate, kv):
    return f"d={d} gate={gate} kv={kv}"


# ----------------------------------------------------------------------------------- tests
def test_mapping():
    for d, gate, kv in CONFIGS:
        hcr, cor = build_models(d, gate, kv)
        keymap, n = check_mapping(hcr, cor)
        print(
            f"    mapping {tag_of(d, gate, kv)}: bijection over {len(keymap)} state-dict tensors, "
            f"{n} parameters in both"
        )


def test_forward():
    for d, gate, kv in CONFIGS:
        torch.manual_seed(0)
        hcr, cor, keymap, data = setup(d, gate, kv)
        forward_parity(hcr, cor, keymap, data, tag_of(d, gate, kv))
        print(
            f"    forward {tag_of(d, gate, kv)}: train x2 / eval, class + view logits and state torch.equal"
        )


def test_backward():
    for d, gate, kv in CONFIGS:
        tag = tag_of(d, gate, kv)
        hcr, cor, keymap, data = setup(d, gate, kv)
        hcr.train()
        cor.train()
        labels = torch.from_numpy(
            np.random.default_rng(5).integers(0, N_CLASSES, N_EVENTS)
        ).long()
        opt_h = torch.optim.Adam(hcr.parameters(), lr=1e-2)
        opt_c = torch.optim.Adam(cor.parameters(), lr=1e-2)
        cp = dict(cor.named_parameters())
        grad_rel = 0.0
        weight_rel = 0.0
        for step in range(5):
            c_logits, q_logits = hcr(*hcr_args(data))
            loss_h = F.cross_entropy(c_logits, labels) + (q_logits**2).mean()
            out = cor(cor_inputs(data))
            loss_c = F.cross_entropy(out["logits"], labels) + (out["views"]["HH"] ** 2).mean()
            opt_h.zero_grad()
            opt_c.zero_grad()
            loss_h.backward()
            loss_c.backward()
            assert_equal(loss_h.detach(), loss_c.detach(), f"{tag} step {step} loss")
            for k, p in hcr.named_parameters():
                q = cp[keymap[k]]
                assert (p.grad is None) == (q.grad is None), f"{tag} step {step}: grad presence of {k}"
                if p.grad is not None:
                    grad_rel = max(
                        grad_rel,
                        assert_close(
                            p.grad, q.grad, f"{tag} step {step}: grad of {k}", GRAD_RTOL, GRAD_ATOL
                        ),
                    )
            opt_h.step()
            opt_c.step()
        hs, cs = hcr.state_dict(), cor.state_dict()
        for hk, ck in keymap.items():
            weight_rel = max(
                weight_rel,
                assert_close(
                    hs[hk], cs[ck], f"{tag} after 5 Adam steps: {hk} / {ck}", WEIGHT_RTOL, WEIGHT_ATOL
                ),
            )
        print(
            f"    backward {tag}: max rel diff over all gradients = {grad_rel:.3e}, "
            f"over all post-step parameters = {weight_rel:.3e}"
        )
        hcr.eval()
        cor.eval()
        with torch.no_grad():
            c_logits, q_logits, out = run_both(hcr, cor, data)
        assert_close(
            c_logits, out["logits"], f"{tag} eval class logits after training", WEIGHT_RTOL, WEIGHT_ATOL
        )
        assert_close(
            q_logits, out["views"]["HH"], f"{tag} eval view logits after training", WEIGHT_RTOL, WEIGHT_ATOL
        )
        print(
            f"    backward {tag}: losses torch.equal; gradients, 5 Adam steps, weights and eval outputs "
            f"allclose (final loss {loss_c.item():.6f})"
        )


def test_finetune_freeze_set():
    for d, gate, kv in CONFIGS:
        tag = tag_of(d, gate, kv)
        hcr, cor = build_models(d, gate, kv)
        keymap, _ = check_mapping(hcr, cor)
        hcr.layers.setLayerRequiresGrad(
            requires_grad=False, index=hcr.embedding_layers()
        )
        for name, p in cor.named_parameters():
            if not name.startswith("head_main."):
                p.requires_grad = False
        h_on = {keymap[k] for k, p in hcr.named_parameters() if p.requires_grad}
        h_off = {keymap[k] for k, p in hcr.named_parameters() if not p.requires_grad}
        c_on = {k for k, p in cor.named_parameters() if p.requires_grad}
        c_off = {k for k, p in cor.named_parameters() if not p.requires_grad}
        assert h_on == c_on, (
            f"{tag}: trainable sets differ: HCR only {sorted(h_on - c_on)}, CoReNet only {sorted(c_on - h_on)}"
        )
        assert h_off == c_off, f"{tag}: frozen sets differ"
        assert all(k.startswith("head_main.") for k in c_on) and c_on, sorted(c_on)
        n_train = sum(p.numel() for p in cor.parameters() if p.requires_grad)
        print(
            f"    finetune {tag}: trainable = {sorted(c_on)} ({n_train} parameters), identical in both"
        )


def test_flag_is_used():
    n_flip = 100
    for d, gate, kv in CONFIGS:
        tag = tag_of(d, gate, kv)
        hcr, cor, keymap, data = setup(d, gate, kv)
        hcr.eval()
        cor.eval()
        flipped = copy.deepcopy(data)
        o = flipped["o"]
        sub = o[:n_flip, 4, :]
        changed = (sub == 0).any(1)  # events of the first 100 with at least one 0 -> 1 flip
        sub[sub == 0] = 1.0
        assert int(changed.sum()) > 0
        with torch.no_grad():
            ref = cor(cor_inputs(data))["logits"]
            new = cor(cor_inputs(flipped))["logits"]
            c_h, _ = hcr(*hcr_args(flipped))
        assert_equal(c_h, new, f"{tag} flipped-flag class logits HCR vs CoReNet")
        rows_eq = (ref == new).all(1)
        assert not rows_eq[:n_flip][changed].any(), (
            f"{tag}: {int(rows_eq[:n_flip][changed].sum())} flipped events with unchanged logits"
        )
        assert rows_eq[:n_flip][~changed].all(), f"{tag}: unflipped events among the first 100 changed"
        assert rows_eq[n_flip:].all(), f"{tag}: events beyond the first {n_flip} changed"
        print(
            f"    flag {tag}: {int(changed.sum())}/{n_flip} events flipped 0->1, all of them change, "
            f"{N_EVENTS - int(changed.sum())} others unchanged"
        )


def test_no_attention():
    d = 8
    for gate, kv in ((True, True),):  # gate / kv are irrelevant without attention
        tag = f"no attention d={d}"
        hcr, cor = build_models(d, gate, kv, context=False)
        keymap, n = check_mapping(hcr, cor)
        assert not any(k.startswith("context.") for k in keymap.values())
        assert not any(k.startswith(("attention_", "inputEmbed.othJet", "inputEmbed.MdPhi")) for k in keymap)
        hcr, cor, keymap, data = setup(d, gate, kv, context=False)
        forward_parity(hcr, cor, keymap, data, tag)
        print(f"    {tag}: bijection over {len(keymap)} tensors, {n} parameters; forward torch.equal")


TESTS = [
    test_mapping,
    test_forward,
    test_backward,
    test_finetune_freeze_set,
    test_flag_is_used,
    test_no_attention,
]


def main(argv=None):
    n = 0
    for t in TESTS:
        print(f"{t.__name__}")
        t()
        n += 1
    print(f"corenet_parity: {n} tests passed")


if __name__ == "__main__":
    main()
