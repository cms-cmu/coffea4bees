# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**coffea4bees** is the main analysis package for the HH→4b (di-Higgs to 4 b-jets) search, built on top of the **barista** base class library (`src/`). See the barista `CLAUDE.md` in the repository root for framework-level documentation (containers, runner.py, CI/CD, base class modules).

Repository: `gitlab.cern.ch/cms-cmu/coffea4bees` (CERN GitLab)

## Running Tests

Test scripts live in `scripts/` and are run inside the analysis container from the barista root directory:

```bash
./run_container bash coffea4bees/scripts/<script>.sh
```

Most scripts accept `--output-base DIR` (default: `output/`).

Key test scripts:
- `analysis-test.sh` - Run 2 analysis processor test (data + TT backgrounds)
- `analysis-test-Run3.sh` - Run 3 analysis processor test
- `analysis-systematics-test.sh` - Systematics test
- `analysis-signals-test.sh` - Signal samples test
- `skimmer-test.sh` - Skimmer test (NanoAOD→PicoAOD)
- `analysis-test-mixed.sh` - Mixed data analysis test
- `code-analysis-helpers.sh`, `code-jet-clustering.sh`, `code-hemisphere-mixing.sh` - Unit-level code tests
- `code-trig-emulator.sh`, `code-plot-test.sh` - Trigger emulator and plotting tests

### Running CI Locally

From the barista root:

```bash
cd coffea4bees/
source scripts/run-local-ci.sh NAME_OF_CI_JOB
```

This runs the `Snakefile_testCI` workflow, which mirrors the GitLab CI pipeline.

## Architecture

## processor_HH4b.py Key Features

For detailed implementation notes on TTbar estimation from data and the cutflow structure, see `analysis/processors/ttbar-estimation.md`.

For MvD classifier application to `mixeddata_all` (weight formula, JCM interaction, friend tree format, config), see `analysis/processors/apply_MvD.md`.

## Hemisphere Mixing / Synthetic Datasets

For physics motivation, matching algorithm details, and implementation notes (including the optional boost-corrected matching mode), see `hemisphere_mixing/README.md`.
