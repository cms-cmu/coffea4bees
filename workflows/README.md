# Coffea4bees Analysis Workflows

This directory contains the Snakemake workflows orchestrated for the **HH $\to$ 4b** (and **ttH(bb)**) analysis pipelines in `coffea4bees`.

The pipeline is organized into modular **Phases (A through F)** reflecting the full analysis lifecycle, from raw nanoAOD ntuples to final Combine statistical interpretation and limit extraction.

---

## 1. Execution Environments & Target Machines

| Phase | Purpose | Target Execution Environment | Compute Requirements |
| :--- | :--- | :--- | :--- |
| **Phase A** | Skimmer & Trigger Weights | **`cmslpc`** | CPU (Condor / Dask batching) |
| **Phase B** | **Phase B.1**: Compute JCM *(New Analysis Only)*<br>**Phase B.2**: Make Classifier Friend Trees *(Required)* | **`cmslpc`** | CPU (Condor / Dask batching) |
| **Phase C** | **Phase C.1 / C.2**: Plot Inputs & Train *(Optional)*<br>**Phase C.3**: Evaluate *(If Bkg Model Changed)* | **`falcon`** (GPU cluster) / **PSC Bridges-2** | GPU (NVIDIA MPS/CUDA for training & inference) |
| **Phase D** | **Phase D.1 / D.2**: Plot Inputs & Train *(Optional)*<br>**Phase D.3**: Evaluate *(Required)* | **`falcon`** (GPU cluster) / **PSC Bridges-2** | GPU (NVIDIA MPS/CUDA for training & inference) |
| **Phase E** | Background Uncertainties *(One-Time / Optional)* | **`cmslpc`** | CPU (Condor / Dask batching) |
| **Phase F** | Analysis Processor & CMS Combine Stats | **`cmslpc`** | CPU (Condor + Dask + Combine container) |

> [!TIP]
> **Configuration Best Practice**: While Snakemake supports direct command-line parameter overrides (e.g. `--config dataset=ttHbb year=UL18`), the **recommended and reproducible approach** is to run workflows using a centralized YAML configuration file passed via `--configfile` (e.g. `--configfile coffea4bees/workflows/config/nominal_run2.yml` or `coffea4bees/workflows/config/analysis_ttHbb.yml`). CLI `--config` flags should only be used for temporary or targeted overrides (such as `--config test=true` or single-dataset evaluation).

---

## 2. High-Level Analysis Pipeline Architecture

```mermaid
flowchart TD
    subgraph PhaseA["Phase A: Pre-processing & Corrections (cmslpc)"]
        A1["PhaseA_1_skimmer.smk\n(PicoAOD Skimming)"] --> A2["PhaseA_2_trigWeights.smk\n(Trigger Efficiency Weights)"]
    end

    subgraph PhaseB["Phase B: Calibration & Classifier Inputs (cmslpc)"]
        B1["PhaseB_1_computeJCM.smk\n(JCM Weights — [New Analysis Only])"]
        B2["PhaseB_2_make_classifier_friendtree.smk\n(Classifier Input Friend Trees — [Required])"]
    end

    subgraph PhaseC["Phase C: FvT Classifier Pipeline (falcon / PSC)"]
        C1["PhaseC_1_plot_inputs.smk\nPhaseC_2_train.smk\n(Diagnostics & Training — [Optional])"]
        C2["PhaseC_3_evaluate.smk\n(FvT Inference — [If Bkg Model Changed])"]
    end

    subgraph PhaseD["Phase D: SvB Classifier Pipeline (falcon / PSC)"]
        D1["PhaseD_1_plot_inputs.smk\nPhaseD_2_train.smk\n(Diagnostics & Training — [Optional])"]
        D2["PhaseD_3_evaluate.smk\n(SvB Inference — [Required])"]
    end

    subgraph PhaseE["Phase E: Background Uncertainties (cmslpc) — [One-Time / Optional]"]
        E1["PhaseE.smk\n(Mixed Data Closure & Systematics)"]
    end

    subgraph PhaseF["Phase F: Analysis & Statistical Interpretation (cmslpc)"]
        F1["PhaseF_1_analysis.smk\n(Main Processor, Cutflows, Plots)"] --> F2["PhaseF_2_stats.smk\n(Datacards, Workspaces, Fits & Limits)"]
    end

    PhaseA --> PhaseB
    PhaseB --> PhaseC
    PhaseC --> PhaseD
    PhaseD --> PhaseE
    PhaseE --> PhaseF
```

> [!NOTE]
> *See the [Phase-by-Phase Technical Specifications](#3-phase-by-phase-technical-specification) below for detailed conditions regarding optional (one-time setup or retraining) versus routine execution steps.*

---

## 3. Phase-by-Phase Technical Specification

### Phase A: Skimming & Trigger Weights
* **Target Machine:** **`cmslpc`** (CPU batching via HTCondor)
* **Coordinator:** `Snakefile_PhaseA.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseA_1_skimmer.smk`: Runs `skimmer_4b.py` on nanoAOD datasets, applies baseline object selections, and writes skimmed picoAOD root files.
  * `Snakefile_PhaseA_2_trigWeights.smk`: Calculates trigger efficiency scale factors and outputs a trigger weights friend tree JSON.

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseA.svg" alt="Phase A Rulegraph DAG" width="380"/>
</p>

**Execution Examples (using recommended config file):**
```bash
# Run full Phase A (Skim + Trigger Weights) via config file on cmslpc
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseA.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml

# Run only the skimmer
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseA_1_skimmer.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml

# Run only trigger weights calculation
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseA_2_trigWeights.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml
```

---

### Phase B: Calibration & Classifier Input Preparation
* **Target Machine:** **`cmslpc`**
* **Coordinator:** `Snakefile_PhaseB.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseB_1_computeJCM.smk`: **[New Analysis Only / One-Time]** Derives jet combinatoric model weights by running the Coffea processor with `apply_JCM: false` and fitting the resulting histograms to compute transfer factors between jet multiplicities. Done once when establishing a new analysis baseline, and reused thereafter.
  * `Snakefile_PhaseB_2_make_classifier_friendtree.smk`: **[Required for All Analyses]** Runs the Coffea processor (`processor_HH4b.py make_classifier_input`) on datasets to create the ROOT friend tree files used as input features for classifier training and evaluation.

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseB.svg" alt="Phase B Rulegraph DAG" width="480"/>
</p>

**Execution Examples (on cmslpc):**
```bash
# Run full Phase B (JCM + Classifier Inputs) on cmslpc
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseB.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run only Phase B.1 JCM computation (New analysis only)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseB_1_computeJCM.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 4

# Run only Phase B.2 Classifier friend tree inputs creation (All analyses)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseB_2_make_classifier_friendtree.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8
```

---

### Phase C: FvT (Four-vs-Three) Classifier Pipeline
* **Target Machine:** **`falcon`** (GPU cluster) or **PSC Bridges-2** (Entire Phase on GPU)
* **Coordinator:** `Snakefile_PhaseC.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseC_1_plot_inputs.smk`: *(Optional / Diagnostics)* Generates raw feature distributions, preprocessed data distributions, and learned event weights plots (`plot_inputs_raw`, `plot_inputs_dataprep`, `plot_weights`).
  * `Snakefile_PhaseC_2_train.smk`: *(Optional — If Retraining FvT)* Trains multi-fold FvT neural networks using PyTorch/HCR (`train`) and produces training loss and ROC curve diagnostics (`analyze`).
  * `Snakefile_PhaseC_3_evaluate.smk`: *(Run If Background Estimation / JCM Changed)* Evaluates trained models on datasets to produce FvT friend tree ntuples (`evaluate`). Flexible to run standalone or chained after training.

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseC.svg" alt="Phase C Rulegraph DAG" width="400"/>
</p>

**Execution Examples (on falcon / PSC):**
```bash
# Run full Phase C master pipeline
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run only Phase C.1 Diagnostic plotting
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_1_plot_inputs.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 4

# Run only Phase C.2 Model training & loss/ROC analysis (If retraining FvT)
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_2_train.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase C.3 evaluation on ALL datasets (If background estimation changed)
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_3_evaluate.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase C.3 evaluation on a SINGLE dataset (e.g. ttHbb)
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_3_evaluate.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config dataset=ttHbb \
    --cores 4
```

---

### Phase D: SvB (Signal-vs-Background) Classifier Pipeline
* **Target Machine:** **`falcon`** (GPU cluster) or **PSC Bridges-2** (Entire Phase on GPU)
* **Coordinator:** `Snakefile_PhaseD.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseD_1_plot_inputs.smk`: *(Optional / Diagnostics)* Generates raw and preprocessed feature distributions and weight plots.
  * `Snakefile_PhaseD_2_train.smk`: *(Optional — If Retraining SvB)* Trains multiclass / binary SvB classifiers (`train`) and generates ROC curves (`analyze`).
  * `Snakefile_PhaseD_3_evaluate.smk`: **[Required for All Analyses]** Evaluates trained SvB models across all analysis datasets to generate the final `SvB_nominal.root` / `.json` friend tree ntuples (`evaluate`) required for downstream event selection in Phase F.

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseD.svg" alt="Phase D Rulegraph DAG" width="400"/>
</p>

**Execution Examples (on falcon / PSC):**
```bash
# Run full Phase D master pipeline
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run only Phase D.1 Diagnostic plotting
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD_1_plot_inputs.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 4

# Run only Phase D.2 SvB training & analysis (If retraining SvB)
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD_2_train.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase D.3 SvB evaluation on ALL datasets (Required for All Analyses)
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD_3_evaluate.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase D.3 SvB evaluation on a single dataset (e.g. ttHbb)
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD_3_evaluate.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config dataset=ttHbb \
    --cores 4
```

---

### Phase E: Background Uncertainties & Systematics
* **Target Machine:** **`cmslpc`**
* **Coordinator:** `Snakefile_PhaseE.smk`
* **Note on Scope:** **Phase E is optional/one-off.** It is executed when establishing background model closure, hemisphere mixing, and transfer factor systematic uncertainty derivations. Once the background uncertainty inputs are generated, they are stored and reused in subsequent analysis runs.

---

### Phase F: Analysis & Statistical Interpretation
* **Target Machine:** **`cmslpc`** (CPU batching with HTCondor/Dask + Combine container)
* **Coordinator:** `Snakefile_PhaseF.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseF_1_analysis.smk`: Runs main Coffea analysis processor (`processor_HH4b.py`), merges histogram files, checks cutflow agreement against reference counts, and generates data/MC comparison plots.
  * `Snakefile_PhaseF_2_stats.smk`: Converts histogram distributions to Combine JSON format, generates CMS Combine datacards, builds workspaces, and computes expected limits, signal significance, and likelihood profile scans.

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseF.svg" alt="Phase F Rulegraph DAG" width="550"/>
</p>

**Execution Examples (on cmslpc):**
```bash
# Run full Phase F (Analysis + Stats) on cmslpc
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseF.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 16

# Run only Phase F.1 Analysis processor & plotting
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseF_1_analysis.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 16

# Run only Phase F.2 Combine statistical fits
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseF_2_stats.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 8
```

---

## 4. General Running Guide & Common Options

### Local & Dry-Run Mode
Always dry-run (`-n` or `-np`) before launching large jobs:
```bash
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseF_1_analysis.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config test=true \
    -np
```
