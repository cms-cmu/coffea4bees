# Coffea4bees Analysis Workflows

This directory contains the Snakemake workflows orchestrated for the **HH $\to$ 4b** (and **ttH(bb)**) analysis pipelines in `coffea4bees`.

The pipeline is organized into modular **Phases (A through E)** reflecting the full analysis lifecycle, from raw nanoAOD ntuples to final Combine statistical interpretation and limit extraction.

---

## 1. Execution Environments & Target Machines

| Phase | Purpose | Target Execution Environment | Compute Requirements |
| :--- | :--- | :--- | :--- |
| **Phase A** | Skimmer & Trigger Weights | **`cmslpc`** | CPU (Condor / Dask batching) |
| **Phase B** | JCM Computation *(One-time / Initial)* | **`cmslpc`** | CPU (Multiprocessing / Dask) |
| **Phase C** | FvT Classifier (Inputs $\to$ Train $\to$ Evaluate) | **`falcon`** (GPU cluster) / **PSC Bridges-2** | GPU (NVIDIA MPS/CUDA for training & inference) |
| **Phase D** | SvB Classifier (Inputs $\to$ Train $\to$ Evaluate) | **`falcon`** (GPU cluster) / **PSC Bridges-2** | GPU (NVIDIA MPS/CUDA for training & inference) |
| **Phase E** | Analysis Processor & CMS Combine Stats | **`cmslpc`** | CPU (Condor + Dask + Combine container) |

> [!TIP]
> **Configuration Best Practice**: While Snakemake supports direct command-line parameter overrides (e.g. `--config dataset=ttHbb year=UL18`), the **recommended and reproducible approach** is to run workflows using a centralized YAML configuration file passed via `--configfile` (e.g. `--configfile coffea4bees/workflows/config/nominal_run2.yml` or `coffea4bees/workflows/config/analysis_ttHbb.yml`). CLI `--config` flags should only be used for temporary or targeted overrides (such as `--config test=true` or single-dataset evaluation).

---

## 2. High-Level Analysis Pipeline Architecture

```mermaid
flowchart TD
    subgraph PhaseA["Phase A: Pre-processing & Corrections (cmslpc)"]
        A1["PhaseA_1_skimmer.smk\n(PicoAOD Skimming)"] --> A2["PhaseA_2_trigWeights.smk\n(Trigger Efficiency Weights)"]
    end

    subgraph PhaseB["Phase B: Jet Combinatoric Model (JCM) — [One-time / Optional] (cmslpc)"]
        B1["PhaseB_computeJCM.smk\n(Calculate & Fit JCM Parameters)"]
    end

    subgraph PhaseC["Phase C: FvT Classifier (Entirely on falcon / PSC)"]
        C1["PhaseC_1_inputs.smk\n(Classifier Inputs)"] --> C2["PhaseC_2_train.smk\n(FvT Model Training)"]
        C2 --> C3["PhaseC_3_evaluate.smk\n(FvT Inference & Friends)"]
    end

    subgraph PhaseD["Phase D: SvB Classifier (Entirely on falcon / PSC)"]
        D1["PhaseD_1_inputs.smk\n(SvB Inputs with FvT)"] --> D2["PhaseD_2_train.smk\n(SvB Model Training)"]
        D2 --> D3["PhaseD_3_evaluate.smk\n(SvB Inference & Friends)"]
    end

    subgraph PhaseE["Phase E: Analysis & Statistical Interpretation (cmslpc)"]
        E1["PhaseE_1_analysis.smk\n(Main Processor, Cutflows, Plots)"] --> E2["PhaseE_2_stats.smk\n(Datacards, Workspaces, Fits & Limits)"]
    end

    PhaseA -->|New analysis / dataset baseline| PhaseB
    PhaseA -->|Routine analysis using existing JCM| PhaseC
    PhaseB --> PhaseC
    PhaseC --> PhaseD
    PhaseD --> PhaseE
```

---

## 3. Phase-by-Phase Technical Specification

### Phase A: Skimming & Trigger Weights
* **Target Machine:** **`cmslpc`** (CPU batching via HTCondor)
* **Coordinator:** `Snakefile_PhaseA.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseA_1_skimmer.smk`: Runs `skimmer_4b.py` on nanoAOD datasets, applies baseline object selections, and writes skimmed picoAOD root files.
  * `Snakefile_PhaseA_2_trigWeights.smk`: Calculates trigger efficiency scale factors and outputs a trigger weights friend tree JSON.

```mermaid
flowchart LR
    nanoAOD["nanoAOD Ntuples"] --> Skim["Phase A.1: Skimmer\n(processor_4b.py)"]
    Skim --> picoAOD["picoAOD Files +\nmodified_datasets.yml"]
    picoAOD --> Trig["Phase A.2: Trigger Weights\n(processor_trigger_weights.py)"]
    Trig --> TrigJSON["trigger_weights_friends.json"]
```

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

### Phase B: Jet Combinatoric Model (JCM) — *(One-time / Initial Calibration)*
* **Target Machine:** **`cmslpc`**
* **Workflow:** `Snakefile_PhaseB_computeJCM.smk` (also aliased by `Snakefile_computeJCM.smk`)
* **Note on Scope:** **Phase B is optional/one-off.** It is executed when setting up a new analysis or establishing a new dataset parameterization baseline. Once the JCM transfer function parameters (`jetCombinatoricModel_SB_<tag>.yml`) are fitted, they are stored and reused in subsequent analysis runs without needing to re-run Phase B every time.
* **Purpose:** Derives jet combinatoric model weights by running the Coffea processor with `apply_JCM: false` and fitting the resulting histograms to compute transfer factors between jet multiplicities.
* **Outputs:** `jetCombinatoricModel_SB_<tag>.yml` and validation plots.

```mermaid
flowchart LR
    picoAOD["picoAOD + TrigWeights"] --> NoJCM["Processor (No JCM)\nData & TTbar"]
    NoJCM --> Merge["Merge coffea files"]
    Merge --> FitJCM["make_new_JCM\n(Fit JCM SB)"]
    Merge --> Plots["make_plots_noJCM"]
    FitJCM --> JCMYaml["jetCombinatoricModel_SB.yml"]
```

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseB.svg" alt="Phase B Rulegraph DAG" width="380"/>
</p>

**Execution Examples:**
```bash
# Dry run
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseB_computeJCM.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config test=true \
    -n

# Run production JCM computation on cmslpc
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseB_computeJCM.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 4
```

---

### Phase C: FvT (Four-vs-Three) Classifier Pipeline
* **Target Machine:** **`falcon`** (GPU cluster) or **PSC Bridges-2** (Entire Phase: inputs, training, and evaluation)
* **Coordinator:** `Snakefile_PhaseC.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseC_1_inputs.smk` (aliased by `Snakefile_make_classifier_friendtree.smk`): Generates input ROOT friend trees with event features and JCM weights for training.
  * `Snakefile_PhaseC_2_train.smk`: Trains multi-fold FvT neural networks using PyTorch/HCR, producing checkpoints and loss/weight diagnostics.
  * `Snakefile_PhaseC_3_evaluate.smk`: Evaluates trained models on datasets to produce FvT friend tree ntuples.

```mermaid
flowchart TD
    picoAOD["picoAOD + JCM"] --> C1["Phase C.1: Inputs (falcon/PSC)\n(processor_HH4b.py make_classifier_input)"]
    C1 --> CIFriends["classifier_inputs_friends.json"]
    CIFriends --> C2["Phase C.2: Training (falcon/PSC)\n(src.classifier.task.main train)"]
    C2 --> Model["Trained Checkpoints &\nLoss/Weight Plots"]
    Model --> C3["Phase C.3: Evaluation (falcon/PSC)\n(src.classifier.task.main evaluate)"]
    C3 --> FvTFriends["FvT Friend Trees\n(FvT_nominal.root / .json)"]
```

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseC.svg" alt="Phase C Rulegraph DAG" width="400"/>
</p>

**Execution Examples (on falcon / PSC):**
```bash
# Run full Phase C master pipeline on falcon / PSC
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC.smk \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/FvT/workflow_config.yml \
    --cores 8

# Run only classifier inputs generation
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_1_inputs.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 8

# Run only training & diagnostic validation plots
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_2_train.smk \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/FvT/workflow_config.yml \
    --cores 8

# Run evaluation on ALL datasets
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_3_evaluate.smk \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/FvT/workflow_config.yml \
    --cores 8

# Run evaluation on a SINGLE dataset (e.g. ttHbb)
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseC_3_evaluate.smk \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/FvT/workflow_config.yml \
    --config dataset=ttHbb \
    --cores 4
```

---

### Phase D: SvB (Signal-vs-Background) Classifier Pipeline
* **Target Machine:** **`falcon`** (GPU cluster) or **PSC Bridges-2** (Entire Phase: inputs, training, and evaluation)
* **Coordinator:** `Snakefile_PhaseD.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseD_1_inputs.smk`: Generates SvB classifier inputs with FvT weights applied.
  * `Snakefile_PhaseD_2_train.smk`: Trains multiclass / binary SvB classifiers (e.g. HH4b vs ttbar/multijet, or ttHbb vs ttbar).
  * `Snakefile_PhaseD_3_evaluate.smk`: Evaluates trained SvB models across all analysis datasets.

```mermaid
flowchart TD
    picoAOD["picoAOD + FvT Friends"] --> D1["Phase D.1: Inputs (falcon/PSC)\n(make_classifier_input + apply_FvT)"]
    D1 --> SvBInputs["SvB classifier_inputs_friends.json"]
    SvBInputs --> D2["Phase D.2: Training (falcon/PSC)\n(src.classifier.task.main train)"]
    D2 --> SvBModel["Trained SvB Checkpoints &\nROC Curves"]
    SvBModel --> D3["Phase D.3: Evaluation (falcon/PSC)\n(src.classifier.task.main evaluate)"]
    D3 --> SvBFriends["SvB Friend Trees\n(SvB_nominal.root / .json)"]
```

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseD.svg" alt="Phase D Rulegraph DAG" width="400"/>
</p>

**Execution Examples (on falcon / PSC):**
```bash
# Run full Phase D master pipeline on falcon / PSC
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD.smk \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/workflow_config.yml \
    --cores 8

# Run only classifier inputs generation
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD_1_inputs.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 8

# Run only SvB training
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD_2_train.smk \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/workflow_config.yml \
    --cores 8

# Run SvB evaluation on a specific dataset or list of datasets
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseD_3_evaluate.smk \
    --configfile coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/workflow_config.yml \
    --config dataset=ttHbb \
    --cores 4
```

---

### Phase E: Analysis & Statistical Interpretation
* **Target Machine:** **`cmslpc`** (CPU batching with HTCondor/Dask + Combine container)
* **Coordinator:** `Snakefile_PhaseE.smk` (aliased by `Snakefile_full_analysis.smk`)
* **Sub-workflows:**
  * `Snakefile_PhaseE_1_analysis.smk` (aliased by `Snakefile_analysis.smk`): Runs main Coffea analysis processor (`processor_HH4b.py`), merges histogram files, checks cutflow agreement against reference files, and generates data/MC comparison plots.
  * `Snakefile_PhaseE_2_stats.smk` (aliased by `Snakefile_stats.smk`): Converts histogram distributions to Combine JSON format, generates CMS Combine datacards, builds workspaces, and computes expected limits, signal significance, and likelihood profile scans.

```mermaid
flowchart TD
    subgraph E1["Phase E.1: Analysis Processor (cmslpc)"]
        In["picoAOD + Trig + JCM + FvT + SvB"] --> Proc["runner.py (processor_HH4b.py)\nData & Signals"]
        Proc --> Merge["Merge coffea files\n(histAll_<label>.coffea)"]
        Merge --> Cutflow["Check Cutflow Test\n(dumpCutFlow.py)"]
        Merge --> Plots["Make Analysis Plots\n(makePlots.py)"]
    end

    subgraph E2["Phase E.2: Statistical Interpretation (cmslpc)"]
        Merge --> H2J["Convert Hist to JSON\n(convert_hist_to_json.py)"]
        H2J --> Cards["Make Combine Inputs\n(make_combine_inputs.py)"]
        Cards --> WS["text2workspace.py\n(Combine Datacards)"]
        WS --> Limits["Combine Limits (AsymptoticLimits)"]
        WS --> Sig["Combine Significance (Significance)"]
        WS --> Scans["Likelihood Scans (MultiDimFit)"]
        WS --> Postfit["Postfit Plots (FitDiagnostics)"]
    end

    E1 --> E2
```

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseE.svg" alt="Phase E Rulegraph DAG" width="550"/>
</p>

**Execution Examples (on cmslpc):**
```bash
# Run full Phase E (Analysis + Stats) on cmslpc
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 16

# Run only Phase E.1 Analysis processor & plotting
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_1_analysis.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 16

# Run only Phase E.2 Combine statistical fits
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_2_stats.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 8
```

---

## 4. General Running Guide & Common Options

### Local & Dry-Run Mode
Always dry-run (`-n` or `-np`) before launching large jobs:
```bash
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseE_1_analysis.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config test=true \
    -np
```

### Distributed Execution (Condor & Dask on cmslpc)
For full-scale analysis runs on `cmslpc`, run within the container wrapper using Condor and Dask batching:
```bash
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE.smk \
    --configfile coffea4bees/workflows/config/nominal_run2.yml \
    --cores 16
```

### Generating Snakemake DAG & Rulegraph Visualizations
To generate the visual DAG diagram for any phase locally:
```bash
# Generate SVG rulegraph
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseA.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --rulegraph | dot -Tsvg -o rulegraph_PhaseA.svg

# Generate full detailed job DAG
pixi run snakemake -s coffea4bees/workflows/Snakefile_PhaseE_1_analysis.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config test=true \
    --dag | dot -Tpng -o job_dag_PhaseE1.png
```

### Backwards Compatibility
Existing legacy workflow entry points are preserved as thin wrappers:
* `Snakefile_computeJCM.smk` $\to$ includes `Snakefile_PhaseB_computeJCM.smk`
* `Snakefile_make_classifier_friendtree.smk` $\to$ includes `Snakefile_PhaseC_1_inputs.smk`
* `Snakefile_analysis.smk` $\to$ includes `Snakefile_PhaseE_1_analysis.smk`
* `Snakefile_stats.smk` $\to$ includes `Snakefile_PhaseE_2_stats.smk`
* `Snakefile_full_analysis.smk` $\to$ includes `Snakefile_PhaseE.smk`

Legacy standalone exploratory files (e.g. `Snakefile_lowpt.smk`, `Snakefile_combinations_ZZ_ZH.smk`, `Snakefile_ZZ_ZH.smk`, `Snakefile_addingVBF.smk`) have been archived into `coffea4bees/workflows/archive/`.
