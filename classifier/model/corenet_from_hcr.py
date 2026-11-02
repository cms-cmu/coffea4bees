"""Convert HCR checkpoints (``src.classifier.ml.models.HCR``) into CoReNet checkpoints
(``src.classifier.ml.models.CoReNet``) with the same weights.

Usage (from the barista root, inside the classifier container)::

    python -m coffea4bees.classifier.model.corenet_from_hcr \\
        --result output/main_train/result.json \\
        --out output/corenet_from_hcr \\
        --network coffea4bees/classifier/test/config/corenet/HH4b_test.yml

Inputs
  * ``--result``: the result.json of an HCR training; every Output stage it references is converted.
  * ``--network``: the CoReNet event description of the HH -> 4b network (roles ``b`` and ``other``,
    hypothesis ``HH: {H1: [b, b], H2: [b, b], context: other}``); ``n_features``, ``attention``,
    ``use_attention_gate`` and ``use_kv_proj`` are taken from the HCR checkpoint.
  * ``--inputs`` (optional): role -> batch key, default ``{b: CanJet, other: NotCanJet}``.

Outputs (in ``--out``)
  * one ``<name>__<uuid>.pkl`` per HCR checkpoint: the CoReNet checkpoint (same ``metadata``, ``uuid``,
    ``label``, ``input``; ``model`` = the HCR state dict renamed through ``NAME_MAP``; ``arch`` =
    ``CoReNetArch.save()``), read by ``CoReNet.SparseDenseEval`` / ``CoReNetEvaluation``.
  * ``result.json``: a copy of the HCR result with the Output-stage paths pointing (relative) to the
    converted checkpoints, so that ``--models Final <out>/result.json`` finds them.

``NAME_MAP`` (HCR state-dict prefix -> CoReNet state-dict prefix) is the renaming proven by
``coffea4bees/classifier/tests/test_corenet_parity.py`` (bit-identical outputs).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

# HCR state-dict prefix -> CoReNet state-dict prefix (CoReNet tests/reference/test_parity_reference.py)
NAME_MAP = [
    ("inputEmbed.ancillaryEmbed.", "event_embed."),
    ("inputEmbed.jetEmbed.", "leaf_embed.b.embed."),
    ("inputEmbed.jetConv.", "leaf_embed.b.conv."),
    ("inputEmbed.dijetEmbed.", "node_embed.H1.embed."),
    ("inputEmbed.dijetConv.", "node_embed.H1.conv."),
    ("inputEmbed.quadjetEmbed.", "node_embed.HH.embed."),
    ("inputEmbed.quadjetConv.", "node_embed.HH.conv."),
    ("inputEmbed.othJetEmbed.", "context.HH.embed.embed."),
    ("inputEmbed.othJetConv.", "context.HH.embed.conv."),
    ("inputEmbed.MdPhi_embed.", "context.HH.pair.embed."),
    ("inputEmbed.MdPhi_conv.", "context.HH.pair.conv."),
    ("attention_oo.", "context.HH.self_attention."),
    ("attention_do.", "context.HH.attention."),
    ("dijetResNetBlock.", "resnet.H1."),
    ("quadjetResNetBlock.", "resnet.HH."),
    ("select_q.", "selector.HH."),
    ("out.", "head_main."),
]

_ARCH_KEYS = ("n_features", "attention", "use_attention_gate", "use_kv_proj")


def map_key(key: str) -> str:
    """The CoReNet state-dict key of an HCR state-dict key."""
    for src, dst in NAME_MAP:
        if key.startswith(src):
            return dst + key[len(src) :]
    raise KeyError(f"HCR state-dict key {key!r} matches no NAME_MAP prefix")


def map_state_dict(hcr_state_dict: dict) -> dict:
    """The HCR state dict under CoReNet names; asserts that the renaming is one to one."""
    out = {map_key(k): v for k, v in hcr_state_dict.items()}
    assert len(out) == len(hcr_state_dict), "two HCR keys map to the same CoReNet key"
    return out


def _read_yaml(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _inputs(inputs: dict | None) -> dict:
    from src.classifier.config.setting.HCR import Input

    inputs = inputs or {"b": "CanJet", "other": "NotCanJet"}
    return {r: getattr(Input, k) if hasattr(Input, k) else k for r, k in inputs.items()}


def convert_checkpoint(hcr_pkl, out_pkl, network_yml, inputs: dict | None = None) -> None:
    """Load an HCR checkpoint, build the CoReNet of ``network_yml`` with the HCR architecture, load the
    renamed weights (strict) and save a CoReNet checkpoint to ``out_pkl``."""
    import fsspec
    import torch

    from src.classifier.ml.models.CoReNet import CoReNetArch, resolve_spec
    from src.classifier.nn.corenet import CoReNet

    with fsspec.open(str(hcr_pkl), "rb") as f:
        saved = torch.load(f, map_location=torch.device("cpu"), weights_only=False)
    architecture = {k: saved["arch"][k] for k in _ARCH_KEYS if k in saved["arch"]}
    arch = CoReNetArch(
        network=_read_yaml(network_yml),
        inputs=_inputs(inputs),
        architecture=architecture,
    )
    spec, resolved = resolve_spec(
        arch, labels=saved["label"], event=saved["input"]["feature_ancillary"]
    )
    arch.resolved = resolved
    net = CoReNet(spec, device="cpu")
    net.load_state_dict(map_state_dict(saved["model"]), strict=True)
    with fsspec.open(str(out_pkl), "wb") as f:
        torch.save(
            {
                "model": net.state_dict(),
                "metadata": saved["metadata"],
                "uuid": saved["uuid"],
                "label": saved["label"],
                "arch": arch.save(),
                "input": saved["input"],
            },
            f,
        )
    logging.info(f"converted {hcr_pkl} -> {out_pkl}")


def convert_result_json(hcr_result_json, out_dir, network_yml, inputs: dict | None = None) -> Path:
    """Convert every Output-stage checkpoint of an HCR result.json into ``out_dir`` and write
    ``out_dir/result.json`` with relative paths to the converted checkpoints. Returns that path."""
    from src.classifier.config.setting import ResultKey

    src = Path(hcr_result_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(src) as f:
        data = json.load(f)
    n = 0
    for result in data.get(ResultKey.models, []):
        for stage in result.get("history", []):
            if stage.get("stage") != "Output":
                continue
            path = Path(stage["path"])
            if stage.get("relative", False):
                path = src.parent / path
            name = path.name
            convert_checkpoint(path, out_dir / name, network_yml, inputs)
            stage["path"] = name
            stage["relative"] = True
            n += 1
    out = out_dir / "result.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"corenet_from_hcr: converted {n} checkpoints from {src} into {out_dir}; wrote {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--result", required=True, help="HCR result.json")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--network", required=True, help="CoReNet event description (.yml)")
    parser.add_argument(
        "--inputs",
        default=None,
        help="role -> batch key as YAML, default '{b: CanJet, other: NotCanJet}'",
    )
    args = parser.parse_args(argv)
    inputs = None
    if args.inputs:
        import yaml

        inputs = yaml.safe_load(args.inputs)
    convert_result_json(args.result, args.out, args.network, inputs)


if __name__ == "__main__":
    main()
