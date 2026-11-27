# ==============================================================================
# coffea4bees/workflows/helpers/fvt_opt_common.smk
# Shared helpers for the FvT optimization sweep workflow (OPT_C -> OPT_F -> OPT_SCORE).
# ==============================================================================

import copy
import os
import re

if not globals().get("_FVT_OPT_COMMON_INCLUDED", False):
    _FVT_OPT_COMMON_INCLUDED = True

    include: "bkg_syst_common.smk"

    # Never inherit an upstream FvT handoff (e.g. fvt_upstream / fvt_handoff_eos from a base
    # bkg_syst config). OPT_F must always consume this roast's own newly trained FvT friends.
    FVT_HANDOFF_EOS = HANDOFF_EOS
    config["fvt_handoff_eos"] = HANDOFF_EOS
    UPSTREAM_FVT_ID = None
    UPSTREAM_FVT_DIR = None

    fvt_opt_cfg = config.get("fvt_opt") or {}
    OPT_MIXES = [str(m) for m in fvt_opt_cfg.get("mixes", SUBSAMPLES)]
    # Keep bkg_syst_common SUBSAMPLES aligned with fvt_opt.mixes
    SUBSAMPLES = list(OPT_MIXES)
    N_SUBSAMPLES = len(SUBSAMPLES)

    OPT_REFERENCE = str(fvt_opt_cfg.get("reference", "ref"))
    OPT_SEEDS = {str(k): int(v) for k, v in (fvt_opt_cfg.get("seeds") or {}).items()}
    OPT_VARIANTS = dict(fvt_opt_cfg.get("variants") or {"ref": {}})

    OPT_RUNS = []
    OPT_RUN_SPECS = {}
    for _vname, _vspec in OPT_VARIANTS.items():
        _vname = str(_vname)
        _spec_base = dict(_vspec or {})
        _n_seeds = max(1, int(OPT_SEEDS.get(_vname, 1)))
        if _n_seeds > 1:
            for _s in range(_n_seeds):
                _run_id = f"{_vname}_s{_s}"
                if not re.fullmatch(r"[A-Za-z0-9_]+", _run_id):
                    raise ValueError(f"fvt_opt run name {_run_id!r} must match [A-Za-z0-9_]+")
                OPT_RUNS.append(_run_id)
                OPT_RUN_SPECS[_run_id] = {**_spec_base, "variant": _vname, "seed_offset": _s}
        else:
            if not re.fullmatch(r"[A-Za-z0-9_]+", _vname):
                raise ValueError(f"fvt_opt variant name {_vname!r} must match [A-Za-z0-9_]+")
            OPT_RUNS.append(_vname)
            OPT_RUN_SPECS[_vname] = {**_spec_base, "variant": _vname, "seed_offset": 0}

    OPT_SCORE_CFG = dict(fvt_opt_cfg.get("score") or {})

    out_opt_c = f"{out}fvt_opt_C/"
    out_opt_f = f"{out}fvt_opt_F/"
    out_opt_score = f"{out}fvt_opt_SCORE/"
    os.makedirs(out_opt_c, exist_ok=True)
    os.makedirs(out_opt_f, exist_ok=True)
    os.makedirs(out_opt_score, exist_ok=True)


def apply_variant_to_model_wf(train_wf: dict, spec: dict) -> dict:
    """Apply a single fvt_opt variant spec onto a staged fvt_train_workflow dict."""
    wf = copy.deepcopy(train_wf)
    spec = spec or {}
    seed_offset = int(spec.get("seed_offset", 0))
    train_kwarg_keys = ("epoch", "bs_init", "bs_scale", "bs_milestones", "lr_init", "lr_scale", "lr_milestones", "weight_decay")

    for mod in (wf.get("model") or []):
        if not isinstance(mod, dict) or not isinstance(mod.get("option"), list):
            continue
        opts = list(mod["option"])

        # 1. Seed offset
        new_opts = []
        found_seed_offset = False
        for o in opts:
            if isinstance(o, str) and o.strip().startswith("--kfold-seed-offsets"):
                new_opts.append(f"--kfold-seed-offsets {seed_offset}")
                found_seed_offset = True
            else:
                new_opts.append(o)
        if not found_seed_offset:
            new_opts.append(f"--kfold-seed-offsets {seed_offset}")
        opts = new_opts

        # 2. Architecture overrides
        arch_over = spec.get("architecture")
        if isinstance(arch_over, dict) and arch_over:
            arch_idx = next(
                (i for i, o in enumerate(opts) if isinstance(o, str) and o.strip() == "--architecture"),
                None,
            )
            if arch_idx is not None:
                if arch_idx + 1 < len(opts) and isinstance(opts[arch_idx + 1], dict):
                    opts[arch_idx + 1].update(arch_over)
                else:
                    opts.insert(arch_idx + 1, dict(arch_over))
            else:
                opts.append("--architecture")
                opts.append(dict(arch_over))

        # 3. Training scheduler class & kwargs
        tr_idx = next(
            (i for i, o in enumerate(opts) if isinstance(o, str) and o.strip().startswith("--training")),
            None,
        )
        if spec.get("training"):
            if tr_idx is not None:
                opts[tr_idx] = f"--training {spec['training']}"
            else:
                opts.append(f"--training {spec['training']}")
                tr_idx = len(opts) - 1

        tr_updates = {k: spec[k] for k in train_kwarg_keys if k in spec}
        if tr_updates:
            if tr_idx is None:
                opts.append("--training FixedStep")
                tr_idx = len(opts) - 1
            if tr_idx + 1 < len(opts) and isinstance(opts[tr_idx + 1], dict):
                opts[tr_idx + 1].update(tr_updates)
            else:
                opts.insert(tr_idx + 1, tr_updates)

        # 4. Finetuning removal if finetuning: null
        if "finetuning" in spec and spec["finetuning"] is None:
            cleaned = []
            i = 0
            while i < len(opts):
                o = opts[i]
                if isinstance(o, str) and o.strip().startswith("--finetuning"):
                    i += 1
                    while i < len(opts) and (
                        isinstance(opts[i], dict)
                        or (isinstance(opts[i], str) and not opts[i].strip().startswith("-"))
                    ):
                        i += 1
                    continue
                cleaned.append(o)
                i += 1
            opts = cleaned

        mod["option"] = opts
    return wf
