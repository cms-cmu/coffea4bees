import pandas as pd
import pytest
import torch

from coffea4bees.classifier.config.dataset.HCR._group import (
    _ERA_ONEHOT_NAMES,
    add_year,
)
from coffea4bees.classifier.nn.blocks.HCR import HCR


RUN2_ERAS = ("UL16_preVFP", "UL16_postVFP", "UL17", "UL18")
CAT_ANCILLARY = ["xW", "nSelJets", "xbW"] + [f"year_{e}" for e in RUN2_ERAS]


@pytest.mark.parametrize(
    "era,expected_int",
    [
        ("UL16_preVFP", 15),
        ("UL16_postVFP", 16),
        ("UL17", 17),
        ("UL18", 18),
    ],
)
def test_add_year_emits_int_and_onehot(era, expected_int):
    processors = list(add_year()(frozenset({f"year:{era}", "label:data"})))
    assert len(processors) == 1
    df = pd.DataFrame({"event": [1, 2, 3]})
    df = processors[0](df)

    assert (df["year"] == expected_int).all()
    for name in _ERA_ONEHOT_NAMES:
        col = f"year_{name}"
        assert col in df.columns
        expected = 1.0 if name == era else 0.0
        assert (df[col] == expected).all()


def _make_dummy_batch(batch_size: int, n_ancillary: int, with_onehot: bool):
    torch.manual_seed(0)
    j = torch.rand(batch_size, 16) * 50.0 + 20.0
    o = torch.rand(batch_size, 40) * 50.0 + 20.0
    a = torch.zeros(batch_size, n_ancillary)
    a[:, 0] = torch.rand(batch_size) * 2.0  # xW
    a[:, 1] = torch.randint(4, 10, (batch_size,)).float()  # nSelJets >= 4
    a[:, 2] = torch.rand(batch_size) * 2.0  # xbW
    if with_onehot:
        for i in range(batch_size):
            a[i, 3 + (i % 4)] = 1.0
    else:
        a[:, 3] = torch.tensor([15, 16, 17, 18] * (batch_size // 4), dtype=torch.float32)
    return j, o, a


def test_hcr_categorical_year_forward_and_layers():
    model = HCR(
        dijetFeatures=8,
        quadjetFeatures=8,
        ancillaryFeatures=CAT_ANCILLARY,
        useOthJets="attention",
        device="cpu",
        nClasses=4,
    )
    assert hasattr(model, "era_out")
    assert model._era_indices == [3, 4, 5, 6]
    # era_out shares the output layer index so FinetuneStep keeps it trainable
    assert model.era_out.index in model.output_layers()
    assert model.era_out.index not in model.embedding_layers()

    j, o, a = _make_dummy_batch(64, len(CAT_ANCILLARY), with_onehot=True)
    model.setGhostBatches(0)
    model.updateMeanStd(j.clone(), o.clone(), a.clone())
    model.initMeanStd()
    model.eval()

    with torch.no_grad():
        c0, q0 = model(j.clone(), o.clone(), a.clone())
    assert c0.shape == (64, 4)
    assert q0.shape == (64, 3)
    assert torch.isfinite(c0).all()
    assert torch.isfinite(q0).all()

    # Non-zero era_out weights shift logits exactly per era
    with torch.no_grad():
        shift = torch.tensor(
            [
                [0.1, -0.2, 0.3, -0.4],
                [0.5, 0.0, -0.5, 0.2],
                [-0.3, 0.4, 0.1, 0.0],
                [0.0, -0.1, 0.2, 0.6],
            ],
            dtype=torch.float32,
        )
        model.era_out.module.weight.copy_(shift.T)
        c1, _ = model(j.clone(), o.clone(), a.clone())

    for era_idx in range(4):
        diff = c1[era_idx::4] - c0[era_idx::4]
        assert torch.allclose(diff, shift[era_idx].unsqueeze(0).expand_as(diff), atol=1e-5)


def test_hcr_legacy_scalar_year_backward_compatible():
    model = HCR(
        dijetFeatures=8,
        quadjetFeatures=8,
        ancillaryFeatures=["xW", "nSelJets", "xbW", "year"],
        useOthJets="attention",
        device="cpu",
        nClasses=4,
    )
    assert not hasattr(model, "era_out")
    assert model._era_indices == []
