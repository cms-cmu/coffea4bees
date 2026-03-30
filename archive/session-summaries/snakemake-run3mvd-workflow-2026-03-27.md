# Session — snakemake-run3mvd-workflow (2026-03-27)

## What we did
- Fixed `mprof` being called unconditionally in `merging_coffea_files` rule even when `run_performance=False`
- Added `mode` config key to `Snakefile_classifier_inputs_Run3MvD.smk` (nominal/quadjet_run2) to select configs and output labels
- Added JCM fitting rule (`make_JCM_Run3MvD`) depending on merged histograms
- Added wJCM histogram step: `create_histogram_config_wJCM` (patches JCM_file, apply_MvD, apply_MvD_weight via sed), reruns mixeddata_all only, merges with TTbar+data flat, plots with `make_plots_wJCM`
- Added `apply_MvD: false` and `apply_MvD_weight: false` to both `HH4b_run_fastTopReco_Run3.yml` and `_quadjet_run2.yml`
- Fixed sed wildcard ambiguity (`apply_MvD.*` matching `apply_MvD_weight`) by using `apply_MvD:[^_]` pattern and processing `apply_MvD_weight` first
- Added `use-singularity: true` to lpc profile so `container:` directives in merge/plot rules are honoured
- Made `dashboard_address` configurable via `--dashboard-address` CLI arg in `runner.py`; default stays 10200, Snakemake rules set 0; also set `scheduler_options: port: 0` to fix scheduler port conflict
- Created `Snakefile_classifier_training_Run3MvD.smk` with train/analyze/evaluate rules, `create_train_yml` rule that patches JCM and friends paths into train.yml via sed
- Converted `JCM_INSTALL_PATH` and `CLASSIFIER_INPUTS_INSTALL_PATH` from Python variables to config keys so they flow through module imports to the training Snakefile
- Added `install_JCM` and `install_classifier_inputs` rules; `all_with_training` imports training rules via `module`
- Created `lpc_gpu` Snakemake profile with `--nv` for GPU passthrough on cmslpcgpu1
- Fixed proxy inside container for `analyze` rule by adding `voms-proxy-init` check in the shell block and setting `X509_USER_PROXY`
- Successfully ran full training pipeline (train 2h19m, evaluate, analyze) on cmslpcgpu1

## Decisions
- Flat merge for wJCM histograms (TTbar+data+mix_wJCM in one step) — not worth hierarchical merging since wJCM step runs infrequently
- JCM file goes in git repo (versioned), not EOS — small YAML, represents a deliberate analysis choice
- Training Snakefile split from data prep Snakefile — physical/environment boundary (GPU machine); other steps kept in one file
- `all_with_training` uses Snakemake `module` import so both workflows can run together or separately
- `rule all` does not include training by default; use `all_with_training` target to include it
- Proxy handling in `analyze` mirrors `run_container` pattern: local `./proxy/x509_proxy` file bound into container

## Files changed
- `coffea4bees/workflows/Snakefile_classifier_inputs_Run3MvD.smk` — mode-based config, JCM/wJCM/plots/install/training rules
- `coffea4bees/workflows/Snakefile_classifier_training_Run3MvD.smk` — NEW: train/analyze/evaluate pipeline for MvD
- `coffea4bees/workflows/rules/analysis.smk` — fixed mprof conditional, added `dashboard_address` param, fixed `run_container_wrapper` trailing comma
- `coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml` — added `apply_MvD: false`, `apply_MvD_weight: false`
- `coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml` — same additions
- `coffea4bees/scripts/run-analysis-processor.sh` — added `--dashboard-address` flag
- `runner.py` — added `--dashboard-address` CLI arg, apply override after `setup_config_defaults`, `scheduler_options port: 0` when dashboard_address=0, default stays 10200
- `software/snakemake/profiles/lpc/config.yaml` — added `use-singularity: true`
- `software/snakemake/profiles/lpc_gpu/config.yaml` — NEW: GPU profile with `--nv`

## Open threads
- `jetCombinatoricModel_SB_.yml` has trailing underscore in name (from empty `-w` weightSet arg) — cosmetic, works correctly
- The `output/Run3_MvD/train.yml` sentinel path is hardcoded (not mode-aware) — could conflict if nominal and quadjet_run2 training run simultaneously
