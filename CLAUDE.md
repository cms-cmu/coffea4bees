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

### Module Structure

| Module | Purpose |
|--------|---------|
| `analysis/processors/` | Coffea processor implementations (`processor_HH4b.py` is the main one) |
| `analysis/helpers/` | Event selection, weighting, cutflow tracking |
| `analysis/weights/` | Weight computation (MC, trigger, resonance) |
| `analysis/trigger_emulator/` | Trigger simulation |
| `skimmer/` | NanoAOD→PicoAOD conversion |
| `classifier/` | ML model training (FvT, SvB classifiers) |
| `jet_clustering/` | Jet clustering algorithms |
| `stats_analysis/` | Statistical analysis with CMS Combine |
| `workflows/` | Snakemake workflow files (11+ `.smk` files) |
| `metadata/` | Dataset configs, luminosity values |
| `plots/` | Plotting scripts and styles |
| `scripts/` | Test and job execution scripts |

### Analysis Pipeline

1. **Skimming**: NanoAOD ROOT files → PicoAOD (reduced event trees)
2. **Friend Trees**: Attach external data (FvT/SvB ML scores, JCM, top reconstruction)
3. **Analysis Processing**: Event selection (4-b-jet tagging), object corrections, weight calculation
4. **Histograms**: Fill nominal + systematic variations
5. **Statistical Analysis**: CMS Combine for limits/significance

### Running Analysis

```bash
# HH4b analysis example
./run_container python runner.py \
    -p coffea4bees/analysis/processors/processor_HH4b.py \
    -c coffea4bees/analysis/metadata/HH4b.yml \
    -m coffea4bees/metadata/datasets_HH4b.yml \
    -y UL18 -t

# Snakemake workflow example
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_testCI.smk --cores 4
```

Dataset metadata is organized by run period: `metadata/datasets_HH4b_Run2/`, `metadata/datasets_HH4b_Run3/`.

## processor_HH4b.py Key Features

### TTbar Estimation from Data (`plot_ttbar_with_weights`)

When `plot_ttbar_with_weights=True`, the processor estimates ttbar contamination by reweighting 3b data events using FvT classifier outputs (`FvT.d3_to_t4` and `FvT.d3_to_t3`). These weights are computed in `analysis/helpers/event_weights.py` inside `include_pseudotag_in_weight` and stored on the event array as `weight_d3_to_t4` and `weight_d3_to_t3`. They are only available when `apply_FvT=True` and `isDataForMixed=False`.

**Histograms**: `filling_nominal_histograms` is called twice more with `processName="TTbar4b_from_d3"` / `"TTbar3b_from_d3"` and `weight_name="weight_d3_to_t4"` / `"weight_d3_to_t3"`.

**Cutflow**: A single `_cutFlow_ttbar` object is filled with a custom scheme — both tag bins come from the same threeTag events but with different weights:
- `_cutFlowFourTag[cut]` = `sum(weight_d3_to_t4)` over threeTag events (ttbar → 4b contribution)
- `_cutFlowThreeTag[cut]` = `sum(weight_d3_to_t3)` over threeTag events (ttbar → 3b contribution)
- `_cutFlowTwoTag[cut]` = 0

Output key per dataset: `f"TTbar_from_d3_{era}"` where `era = dataset.removeprefix("data_")`, e.g. `"TTbar_from_d3_2023_preBPixC12"`. This is filled only for nominal (not systematic shifts) and only for the detailed post-candidate cuts (same set as `fill_detailed_cutflows`).

**Important**: `cutflow_4b.addOutput` uses `setdefault` so multiple `addOutput` calls per chunk (nominal data + TTbar) do not overwrite each other.

### Cutflow Structure (`analysis/helpers/cutflow.py`)

`cutflow_4b` tracks three tag regions: `_cutFlowTwoTag`, `_cutFlowThreeTag`, `_cutFlowFourTag`. Each stores `(weighted_sum, raw_count)` per cut name. Early cuts (HLT, jet multiplicity, etc.) are filled in `build_cutflow` with `allTag=True`. Detailed cuts after candidate building are filled in `fill_detailed_cutflows` split by actual tag. The `fill_cutflow_with_and_without_trig` helper always also fills a `{cut}_woTrig` variant excluding the trigger SF weight.
