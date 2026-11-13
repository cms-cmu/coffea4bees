#!/usr/bin/env python3
"""Validate Phase C (FvT) outputs: yields, trained models, and evaluated friend trees."""

import argparse
import json
import math
import os
import sys
from pathlib import Path


def _is_finite(val):
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)


def validate_yields(yields_path, lines):
    if not os.path.exists(yields_path):
        raise AssertionError(f"Missing yields.json: {yields_path}")
    with open(yields_path) as f:
        data = json.load(f)

    regions = data.get("regions", {})
    required_regions = ("SB", "HHSR", "ZHSR", "ZZSR")
    for reg in required_regions:
        if reg not in regions:
            raise AssertionError(f"Missing region '{reg}' in {yields_path}")

    lines.append("=== Phase C Yields (SB) ===")
    lines.append(f"{'Class':10s} {'Count':>10s} {'Weight':>14s} Status")
    lines.append("-" * 46)
    sb = regions["SB"]
    for cls_name in ("d3", "d4", "t3", "t4"):
        entry = sb.get(cls_name)
        if not entry:
            raise AssertionError(f"Missing class '{cls_name}' in SB region of {yields_path}")
        cnt = entry.get("count", 0)
        w = entry.get("weight", float("nan"))
        ok = cnt > 0 and _is_finite(w) and w > 0
        lines.append(f"{cls_name:10s} {cnt:10d} {w:14.4f} {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise AssertionError(f"Invalid SB yield for {cls_name}: {entry}")


def validate_models(model_result_path, lines):
    if not os.path.exists(model_result_path):
        raise AssertionError(f"Missing model result.json: {model_result_path}")
    with open(model_result_path) as f:
        data = json.load(f)

    models = data.get("models", [])
    if not models:
        raise AssertionError(f"No trained models found in {model_result_path}")

    lines.append("")
    lines.append(f"=== Phase C Models ({len(models)} folds) ===")
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
            and last_loss < baseline_loss
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
    required_branches = {"FvT", "p_d4", "p_d3", "p_t4", "p_t3", "p_4b", "p_3b", "p_data", "p_ttbar"}
    missing = required_branches - set(branches)
    if missing:
        raise AssertionError(f"Missing expected FvT friend branches: {sorted(missing)}")

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
    lines.append(f"=== Phase C Friend Trees ({len(chunk_paths)} file(s), {len(branches)} branches) ===")
    lines.append(f"{'File':50s} {'Entries':>8s} {'Finite':>8s} {'Infs':>5s} {'Mean(FvT)':>10s} Status")
    lines.append("-" * 93)

    for path, tree_name in chunk_paths:
        if not os.path.exists(path):
            raise AssertionError(f"Missing FvT friend ROOT file: {path}")
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

        fvt_arr = arrays["FvT"]
        finite_fvt = fvt_arr[~np.isnan(fvt_arr)]
        n_finite = int(len(finite_fvt))
        mean_fvt = float(np.mean(finite_fvt)) if n_finite > 0 else float("nan")
        ok = (
            n_entries > 0
            and n_finite > 0
            and consistent_nans
            and n_infs == 0
            and math.isfinite(mean_fvt)
            and mean_fvt > 0
        )
        short = path[-50:] if len(path) > 50 else path
        lines.append(
            f"{short:50s} {n_entries:8d} {n_finite:8d} {n_infs:5d} {mean_fvt:10.4f} {'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            raise AssertionError(
                f"Invalid FvT friend tree {path}: entries={n_entries}, finite={n_finite}, "
                f"consistent_nans={consistent_nans}, infs={n_infs}, mean={mean_fvt}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yields", required=True, help="Path to Phase C yields.json")
    parser.add_argument("--model-result", required=True, help="Path to Phase C model/result.json")
    parser.add_argument("--friend-result", required=True, help="Path to Phase C friend/result.json")
    parser.add_argument("--output", default=None, help="Optional output validation summary text file")
    args = parser.parse_args()

    lines = []
    validate_yields(args.yields, lines)
    validate_models(args.model_result, lines)
    validate_friends(args.friend_result, lines)
    lines.append("")
    lines.append("Phase C validation PASSED.")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report)


if __name__ == "__main__":
    main()
