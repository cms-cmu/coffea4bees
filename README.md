
# Coffea4bees Python Package

[![pipeline status](https://gitlab.cern.ch/cms-cmu/coffea4bees/badges/master/pipeline.svg)](https://gitlab.cern.ch/cms-cmu/coffea4bees/-/commits/master)

This directory contains the main Python code for the Coffea4bees project, built on top of the [barista](https://gitlab.cern.ch/cms-cmu/barista) framework for high-energy physics analyses.

## Purpose

This folder provides all analysis, skimming, machine learning, plotting, and workflow automation tools for 4b physics analyses. It is the main entry point for running and developing new analysis features.

## Quickstart

To run analysis or skimming from this directory:

1. Clone the `barista` repository and then this repository.

```bash
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/barista.git
cd barista
git clone ssh://git@gitlab.cern.ch:7999/cms-cmu/coffea4bees.git
```

2. Run the main analysis script:

```bash
python runner.py --help
```

3. Explore subfolders for specialized tasks (skimming, ML, plotting, etc.).


## Folder Overview

- **analysis/**: Main analysis processors, helpers, metadata, tools, and tests for physics analysis.
- **analysis_dask/**: Dask-based analysis modules and configuration for distributed processing.
- **archive/**: Archived datasets, plots, and skims from previous runs.
- **classifier/**: Machine learning models, utilities, and scripts for classification tasks.
- **examples/**: Example scripts for analysis and meta-data rescue.
- **jet_clustering/**: Jet clustering algorithms, studies, and synthetic data generation.
- **metadata/**: Central metadata configurations including unified datasets, cross-sections, triggers, and friend trees.
- **plots/**: Plotting scripts, styles, and metadata for visualizations.
- **scripts/**: Shell scripts for running, testing, and automating analysis jobs.
- **skimmer/**: Processors for filtering NanoAOD files and saving skimmed (picoAOD) files.
- **stats_analysis/**: Statistical analysis scripts and Combine framework integration.
- **workflows/**: Snakemake workflows and rules for automating analysis pipelines.

For more details about each component, refer to the `README.md` file in the respective folder.

## Metadata & Dataset Structure

The `metadata/` directory has been reorganized into a centralized structure to support seamless, unified execution across different runs:

### 1. Unified Datasets (`metadata/datasets/`)
Contains all data and MC dataset definition YAML files (e.g. `TT.yml`, `GluGluToHHTo4B.yml`, `data.yml`). 
- **Unified Cross-Sections**: To prevent key collisions and support running Run 2 and Run 3 analysis scripts under the same dataset keys, all cross-sections (`xs`) are defined as run-dependent dictionaries:
  ```yaml
  xs:
    Run2: <run2_cross_section_value>
    Run3: <run3_cross_section_value>
  ```
  If a dataset is specific to only one Run, the other Run's cross-section is set to a placeholder `1`.
- **Dataset Archive (`metadata/datasets/archive/`)**: Holds older dataset definitions and versions (e.g. `Run2_2024_v1`, `Run2_2024_v2`, `Run3_archive`).

### 2. Friend Trees & Trigger Weights (`metadata/friends/`)
Contains friend tree configurations (`friends_HH4b.yml`, `friends_empty.yml`) and their corresponding active JSON lookups:
- Active JSON files for trigger weights and classifier friend trees (e.g. `trigweights_2024_v1p2.json`, `data_SvBfriend.json`, etc.) live directly in `metadata/friends/`.
- Unused/legacy friend trees are placed in `metadata/friends/archive/`.

## REANA Integration

[![Launch with Snakemake on REANA](https://www.reana.io/static/img/badges/launch-on-reana.svg)](https://reana.cern.ch/launch?name=Coffea4bees&specification=reana.yml&url=https%3A%2F%2Fgitlab.cern.ch%2Fcms-cmu%2Fcoffea4bees)

This package supports running workflows on [REANA](https://reana.cern.ch/). The REANA workflow is triggered manually via the GitLab CI pipeline or automatically every Saturday.

Workflow outputs (plots, files) are available at [https://plotsalgomez.webtest.cern.ch/HH4b/reana/](https://plotsalgomez.webtest.cern.ch/HH4b/reana/).

Each output folder is named with the REANA job execution date and the corresponding Git commit hash. Folders are only copied here if the REANA job completes successfully.