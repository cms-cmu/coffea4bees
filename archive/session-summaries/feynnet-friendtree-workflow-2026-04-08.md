# Session — feynnet-friendtree-workflow (2026-04-08)

## What we did
- Created Snakemake workflow for SvB_FeynNet friend tree creation (data, mixeddata_all, TTBar MC)
- Created config `HH4b_make_friend_SvBFeynNet_Run3.yml` for FeynNet-only evaluation (SvB/SvB_MA disabled, ONNX model)
- Created test script `SvBFeynNet-friendtree-test.sh` for local validation before condor
- Fixed `runner.py` port collision: unique UUID log directory per invocation (dask_jobqueue calls `os.makedirs` without `exist_ok`)
- Fixed `runner.py` scheduler port collision: added `'port': 0` to `scheduler_options`
- Added `env_extra: PYTHONUNBUFFERED=1` to condor workers in `runner.py`
- Added `client.dashboard_link` logging to the condor cluster path in `runner.py` (was only logged for LocalCluster)
- Built `src/tools/runner_monitor.py`: live Dask task monitor that discovers dashboard URLs from log files, handles `/proxy/PORT` URLs, falls back to TCP scheduler connection, shows pending jobs from Snakemake log, queries Prometheus `/metrics` endpoint
- Wrote `src/tools/runner_monitor.md` README

## Decisions
- Separate Snakemake file for FeynNet (`Snakefile_SvBFeynNet_friendtrees_Run3.smk`) rather than adding to existing SvB file — different config, model lifecycle, merge target
- TTBar datasets run as separate jobs per dataset per year (3×4=12 jobs) via `{tt_dataset}` wildcard — better parallelism and fault isolation
- `top_reconstruction: null` in FeynNet config — not needed for friend tree creation
- Workflow runs on data + mixeddata_all + TTBar (no pre-existing FeynNet data JSON unlike SvB)
- runner_monitor uses Prometheus `/metrics` endpoint (not dashboard HTML) — machine-readable, no parsing needed
- Pending jobs sourced from `.snakemake/log/` latest file (parses `log:` field) — generic, no hardcoded dataset lists

## Files changed
- `coffea4bees/workflows/Snakefile_SvBFeynNet_friendtrees_Run3.smk` — new Snakemake workflow for FeynNet friend trees
- `coffea4bees/analysis/metadata/HH4b_make_friend_SvBFeynNet_Run3.yml` — new runner config for FeynNet friend tree creation
- `coffea4bees/scripts/SvBFeynNet-friendtree-test.sh` — new local test script
- `runner.py` — UUID log directory, `port: 0` scheduler, `PYTHONUNBUFFERED`, `dashboard_link` logging for condor path
- `src/tools/runner_monitor.py` — new Dask job monitor script
- `src/tools/runner_monitor.md` — README for runner_monitor

## Open threads
- runner_monitor dashboard URL only present in logs from jobs run after the `dashboard_link` fix — older logs fall back to TCP (which may fail if scheduler is gone)
- `env_extra: PYTHONUNBUFFERED=1` accepted without error but worker stdout content not verified in condor `.out` files
- `SvBFeynNetfriend_mixeddata_data.json` not yet installed to metadata (workflow still running)
- `friends_HH4b.yml` not yet updated to include `SvB_FeynNet` entry pointing to the new JSON
