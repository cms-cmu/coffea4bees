#!/usr/bin/env python3
"""Validate Phase D (SvB) outputs: yields, trained models, and evaluated friend trees."""

import argparse
import json
import math
import os
from pathlib import Path


def _is_finite(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def validate_yields(yields_path, lines):
    if not os.path.exists(yields_path):
        raise AssertionError(f"Missing yields.json: {yields_path}")
    with open(yields_path) as f:
        data = json.load(f)

    regions = data.get("regions", {})
    if not regions:
        raise AssertionError(f"No regions found in {yields_path}")

    lines.append("=== Phase D Yields ===")
    lines.append(f"{'Region':10s} {'Class':12s} {'Count':>10s} {'Weight':>14s} Status")
    lines.append("-" * 58)
    for reg, classes in sorted(regions.items()):
        if not isinstance(classes, dict):
            continue
        for cls_name, entry in sorted(classes.items()):
            cnt = entry.get("count", 0)
            w = entry.get("weight", float("nan"))
            ok = cnt > 0 and _is_finite(w) and w > 0
            lines.append(f"{reg:10s} {cls_name:12s} {cnt:10d} {w:14.4f} {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise AssertionError(f"Invalid yield for {reg}/{cls_name}: {entry}")


def validate_models(model_result_path, lines):
    if not os.path.exists(model_result_path):
        raise AssertionError(f"Missing model result.json: {model_result_path}")
    with open(model_result_path) as f:
        data = json.load(f)

    models = data.get("models", [])
    if not models:
        raise AssertionError(f"No trained models found in {model_result_path}")

    lines.append("")
    lines.append(f"=== Phase D Models ({len(models)} folds) ===")
    lines.append(f"{'Fold':6s} {'BaselineLoss':>14s} {'TrainLoss':>12s} {'FinetuneLoss':>14s} {'Final':>7s} Status")
    lines.append("-" * 64)

    for idx, mod in enumerate(models):
        history = mod.get("history", [])
        baseline_loss = None
        train_loss = None
        finetune_loss = None
        has_final = False

        for h in history:
            stage = h.get("stage")
            name = h.get("name")
            if stage == "Benchmark" and name == "Baseline":
                baseline_loss = h.get("benchmarks", {}).get("training", {}).get("scalars", {}).get("loss")
            elif name == "Training" and "training" in h and h["training"]:
                train_loss = h["training"][-1].get("benchmarks", {}).get("training", {}).get("scalars", {}).get("loss")
            elif name == "Finetuning" and "training" in h and h["training"]:
                finetune_loss = h["training"][-1].get("benchmarks", {}).get("training", {}).get("scalars", {}).get("loss")
            elif stage == "Output" and name == "Final":
                has_final = True

        last_loss = finetune_loss if finetune_loss is not None else train_loss
        ok = (
            has_final
            and _is_finite(baseline_loss)
            and _is_finite(train_loss)
            and _is_finite(last_loss)
            and train_loss > 0
            and last_loss > 0
        )
        ft_str = f"{finetune_loss:14.4f}" if finetune_loss is not None else f"{'N/A':>14s}"
        lines.append(
            f"{idx:<6d} {baseline_loss:14.4f} {train_loss:12.4f} {ft_str} "
            f"{str(has_final):>7s} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            raise AssertionError(
                f"Model fold {idx} failed check: has_final={has_final}, "
                f"baseline={baseline_loss}, train={train_loss}, finetune={finetune_loss}"
            )


def validate_friends(friend_result_path, lines):
    import numpy as np
    import uproot

    if not os.path.exists(friend_result_path):
        raise AssertionError(f"Missing friend result.json: {friend_result_path}")
    with open(friend_result_path) as f:
        data = json.load(f)

    analysis = data.get("analysis", [])
    if not analysis or "merged" not in analysis[0]:
        raise AssertionError(f"Missing analysis[0].merged in {friend_result_path}")

    merged = analysis[0]["merged"]
    branches = merged.get("branches", [])
    if not branches:
        raise AssertionError(f"No merged SvB branches listed in {friend_result_path}")

    chunk_paths = []
    for item in merged.get("data", []):
        if len(item) >= 2 and isinstance(item[1], list):
            for seg in item[1]:
                p = seg.get("chunk", {}).get("path")
                if p:
                    chunk_paths.append((p, seg.get("chunk", {}).get("name", "Events")))

    if not chunk_paths:
        raise AssertionError(f"No merged friend tree chunks listed in {friend_result_path}")

    lines.append("")
    lines.append(f"=== Phase D Friend Trees ({len(chunk_paths)} file(s), {len(branches)} branches) ===")
    lines.append(f"{'File':55s} {'Entries':>8s} {'Finite':>8s} {'Infs':>5s} Status")
    lines.append("-" * 86)

    for path, tree_name in chunk_paths:
        if not os.path.exists(path):
            raise AssertionError(f"Missing SvB friend ROOT file: {path}")
        with uproot.open(path) as rf:
            tree = rf[tree_name]
            n_entries = tree.num_entries
            arrays = tree.arrays(library="np")

        n_infs = 0
        nan_mask = None
        consistent_nans = True
        for b, arr in arrays.items():
            if np.issubdtype(arr.dtype, np.floating):
                n_infs += int(np.isinf(arr).sum())
                cur_nan = np.isnan(arr)
                if nan_mask is None:
                    nan_mask = cur_nan
                elif not np.array_equal(nan_mask, cur_nan):
                    consistent_nans = False

        n_finite = int((~nan_mask).sum()) if nan_mask is not None else n_entries
        ok = n_entries > 0 and n_finite > 0 and consistent_nans and n_infs == 0
        short = path[-55:] if len(path) > 55 else path
        lines.append(f"{short:55s} {n_entries:8d} {n_finite:8d} {n_infs:5d} {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError(
                f"Invalid SvB friend tree {path}: entries={n_entries}, finite={n_finite}, "
                f"consistent_nans={consistent_nans}, infs={n_infs}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yields", required=True, help="Path to Phase D yields.json")
    parser.add_argument("--model-result", required=True, help="Path to Phase D model/result.json")
    parser.add_argument("--friend-result", required=True, help="Path to Phase D friend/result.json")
    parser.add_argument("--output", default=None, help="Optional output validation summary text file")
    args = parser.parse_args()

    lines = []
    validate_yields(args.yields, lines)
    validate_models(args.model_result, lines)
    validate_friends(args.friend_result, lines)
    lines.append("")
    lines.append("Phase D validation PASSED.")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
