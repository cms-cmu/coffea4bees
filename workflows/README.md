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
| **Phase E** | Background Uncertainties & Closure *(Optional — Skip if Stat-Only; Requires Phase F.1 Singlefiles)* | **`cmslpc`** | CPU (Condor / Dask batching) |
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

    subgraph PhaseE["Phase E: Background Uncertainties (cmslpc) — [Optional / Skip if Stat-Only]"]
        E1["PhaseE.smk\n(Mixed Data Closure & Systematics — [Needs Phase F.1])"]
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
> *See the [Phase-by-Phase Technical Specifications](#3-phase-by-phase-technical-specification) below for detailed conditions regarding optional (one-time setup or retraining) versus routine execution steps, along with intermediate metadata updates.*

---

## 3. Phase-by-Phase Technical Specification

### Phase A: Skimming & Trigger Weights
* **Target Machine:** **`cmslpc`** (CPU batching via HTCondor)
* **Coordinator:** `Snakefile_PhaseA.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseA_1_skimmer.smk`: Runs `skimmer_4b.py` on nanoAOD datasets, applies baseline object selections, and writes skimmed picoAOD root files.
  * `Snakefile_PhaseA_2_trigWeights.smk`: Calculates trigger efficiency scale factors and outputs a trigger weights friend tree JSON.

#### Key Artifacts & Required Downstream Updates:
* **After Phase A.1 (Skimmer)**:
  * *Outputs Produced*: Skimmed picoAOD ROOT files (`picoAOD_*.root`) on EOS.
  * *Files to Add/Modify*: Create or update dataset manifest files in `metadata/datasets/` (e.g. `picoaod_datasets_<analysis>.yml`) so downstream processors find the skimmed files.
* **After Phase A.2 (Trigger Weights)**:
  * *Outputs Produced*: Trigger weight friend trees and manifest `trigWeights_nominal.json` on EOS.
  * *Files to Add/Modify*: Register the trigger weight friend path in the analysis friend file (e.g. `friends_<analysis>.yml`):
    ```yaml
    trigWeight:
      - path: 'root://cmseos.fnal.gov//store/user/.../trigWeights_nominal.json'
        name: Final
    ```

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseA.svg" alt="Phase A Rulegraph DAG" width="380"/>
</p>

**Execution Examples:**
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
  * `Snakefile_PhaseB_1_computeJCM.smk`: **[New Analysis Only / One-Time]** Derives jet combinatoric model weights by running the Coffea processor with `apply_JCM: false` and fitting the resulting histograms to compute transfer factors between jet multiplicities. Once the fit is done, the same datasets (data + ttbar) are **rerun with the fitted JCM applied** (`apply_JCM: true`, `JCM_file` → the new fit; still no FvT/SvB) to produce `histAll_wJCM.coffea`, and plots are made from that file only, using the NoFvT plot config (`jcm_plot_config`, default `plots/metadata/plotsAllNoFvT.yml`). Done once when establishing a new analysis baseline, and reused thereafter.
  * `Snakefile_PhaseB_2_make_classifier_friendtree.smk`: **[Required for All Analyses]** Runs the Coffea processor (`processor_HH4b.py make_classifier_input`) on datasets to create the ROOT friend tree files used as input features for classifier training and evaluation.

#### Key Artifacts & Required Downstream Updates:
* **After Phase B.1 (computeJCM)**:
  * *Outputs Produced*: Fitted JCM YAML file (`jetCombinatoricModel_SB_<tag>.yml`), the JCM-applied histograms `histAll_wJCM.coffea`, the NoFvT plots in `plots_wJCM/` (with an `index.html` gallery), and cutflow dumps `cutflow_NoJCM.yml` / `cutflow_wJCM.yml`. The cutflows are compared (like the CI known counts) against `analysis/tests/known_fullCounts_JCM_<pass>.yml` (`known_Counts_JCM_<pass>.yml` in test mode; override with `jcm_known_counts_<pass>[_test]` in the config). If the reference is missing the cutflow is only dumped — bless the dump as the new reference when a baseline changes on purpose. The comparison verdict (observed vs expected table) is written to `cutflow_validation_<pass>_result.txt` next to the yml; on a mismatch the rule fails (Phase B stops) but that file and a `cutflow_<pass>_failed.yml` copy of the counts survive and are published by roast for debugging. `cutflow_<pass>.html` (and `cutflow_<pass>_table.txt`) is the background-closure view of the same numbers (`src/tools/cutflow_closure.py`): cuts as rows, data 3b | tt 3b | Multijet | tt 4b | Bkg | data 4b | data/Bkg, summed over all years and per year, with a toggle for the ttbar components and the tt 3b / data 3b fraction.
  * *Files to Add/Modify*: Store the JCM file in `metadata/weights/JCM/<analysis>/` and update `weights_<analysis>.yml` (e.g. `weights_ttHbb.yml` or `weights_HH4b.yml`) to point `JCM_file:` to this new file.
* **After Phase B.2 (make_classifier_friendtree)**:
  * *Outputs Produced*: Classifier input friend tree ROOT files on EOS + `classifier_inputs_friends.json` manifest.
  * *Files to Add/Modify*: Create the classifier dataset manifest in `metadata/datasets/.../classifier_inputs_<analysis>.json` with the `@@HCR_input` / `@@HCR_input_lowpt` dataset configuration block for PyTorch dataloaders. Ensure `fvt.metadata` and `svb.metadata` in the master config point to this JSON.

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

#### Key Artifacts & Required Downstream Updates:
* **After Phase C.2 (Train)**:
  * *Outputs Produced*: PyTorch checkpoint weights (`model.pt`, `model_fold_*.pt`) and `result.json` on EOS.
* **After Phase C.3 (Evaluate)**:
  * *Outputs Produced*: Evaluated FvT friend tree ROOT files (`FvT_*.root`) on EOS and manifest `result.json`.
  * *Files to Add/Modify*: Register the FvT friend tree path in `weights_<analysis>.yml`:
    ```yaml
    FvT:
      - path: 'root://cmseos.fnal.gov//store/user/.../classifier/FvT_<label>/result.json'
        name: Final
    ```
    Ensure `svb.train_template` in the master config references this FvT friend path.

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseC.svg" alt="Phase C Rulegraph DAG" width="400"/>
</p>

**Execution Examples (on falcon / PSC):**
```bash
# Run full Phase C master pipeline
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseC.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run only Phase C.1 Diagnostic plotting
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseC_1_plot_inputs.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 4

# Run only Phase C.2 Model training & loss/ROC analysis (If retraining FvT)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseC_2_train.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase C.3 evaluation on ALL datasets (If background estimation changed)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseC_3_evaluate.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase C.3 evaluation on a SINGLE dataset (e.g. ttHbb)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseC_3_evaluate.smk \
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

#### Key Artifacts & Required Downstream Updates:
* **After Phase D.2 (Train)**:
  * *Outputs Produced*: PyTorch checkpoint weights (`model.pt`, `model_fold_*.pt`) and `result.json` on EOS.
* **After Phase D.3 (Evaluate)**:
  * *Outputs Produced*: Evaluated SvB friend tree ROOT files (`SvB_*.root`) on EOS and manifest `result.json`.
  * *Files to Add/Modify*: Register the SvB friend tree path in `weights_<analysis>.yml`:
    ```yaml
    SvB_MA:
      - path: 'root://cmseos.fnal.gov//store/user/.../classifier/SvB_<label>/result.json'
        name: Final
    ```

#### Snakemake Rulegraph DAG:
<p align="center">
  <img src="docs/figures/rulegraph_PhaseD.svg" alt="Phase D Rulegraph DAG" width="400"/>
</p>

**Execution Examples (on falcon / PSC):**
```bash
# Run full Phase D master pipeline
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseD.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run only Phase D.1 Diagnostic plotting
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseD_1_plot_inputs.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 4

# Run only Phase D.2 SvB training & analysis (If retraining SvB)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseD_2_train.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase D.3 SvB evaluation on ALL datasets (Required for All Analyses)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseD_3_evaluate.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --cores 8

# Run Phase D.3 SvB evaluation on a single dataset (e.g. ttHbb)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseD_3_evaluate.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config dataset=ttHbb \
    --cores 4
```

---

### Phase E: Background Systematics & Two-Stage Closure
* **Target Machine:** **`cmslpc`** (Steps 1, 2, 3, 5, 6: CPU batching with HTCondor/Dask + Combine container) / **`falcon`** (Step 4: GPU cluster for FvT training & evaluation)
* **Coordinator:** `Snakefile_PhaseE.smk`
* **Dedicated In-Depth Guide:** See [PhaseE_MixedData_Closure.md](docs/PhaseE_MixedData_Closure.md) for full physics motivation, architecture, step-by-step instructions, and troubleshooting.
* **Sub-workflows:**
  * `Snakefile_PhaseE_1_make_mixeddata.smk`: Step 1 — Generates mixed-data picoAODs from collision data using the hemisphere mixing library.
  * `Snakefile_PhaseE_2_make_subsamples.smk`: Step 2 — Slices mixed data into 15 statistically independent subsamples (`mix_v0` .. `mix_v14`), fits subsample-specific JCM weights, creates classifier inputs, and evaluates SvB friend trees.
  * `Snakefile_PhaseE_3_make_ttbar_psdata.smk`: Step 3 — Generates pseudo-data from $t\bar{t}$ MC to validate hemisphere subtraction.
  * `Snakefile_PhaseE_4_FvT_training.smk`: Step 4 — Trains 15 distinct FvT models (one per subsample) on Falcon GPU and evaluates friend trees on 3-tag collision data.
  * `Snakefile_PhaseE_5_analysis.smk`: Step 5 — Executes Coffea analysis processor (`processor_ttHbb.py`) over all 15 mixed-data subsamples across 4 Run 2 eras (`UL16_preVFP`, `UL16_postVFP`, `UL17`, `UL18`).
  * `Snakefile_PhaseE_6_closure.smk`: Step 6 — Builds 3-tag background ROOT histograms with `coffea4bees/stats_analysis/make_fvt_data3b_hists.py` ($SF = 1.4508$, 30 variable bins), converts mixed-data and signal to ROOT, and executes `runTwoStageClosure.py`.

#### Required Datasets and JCM Models per Step:
Because Phase E mixes multiple data-driven and MC components, ensuring the correct dataset and JCM file at each step is critical:

| Step | Sub-Workflow | Target Machine | Input Datasets | Applied JCM Calibration | Outputs Produced |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Step 1** | `PhaseE_1_make_mixeddata.smk` | `cmslpc` (CPU) | Collision Data 3-tag/4-tag | Nominal inclusive JCM (`jetCombinatoricModel_SB_2024_v2_stitched.yml`) | `mixeddata_ttHbb_stitched_rank0_0.yml` & `JCM_mixeddata_inclusive/` |
| **Step 2** | `PhaseE_2_make_subsamples.smk` | `cmslpc` (CPU) | `mixeddata_ttHbb_stitched_rank0_0.yml` | Subsample JCMs (`jetCombinatoricModel_SB_mix_v{m}.yml` fitted per subsample) | `mixeddata_4b.yml` (15 subsamples), `classifier_inputs_mixeddata_ttHbb.json`, `friends_ttHbb_mixeddata_4b.json` |
| **Step 3** | `PhaseE_3_make_ttbar_psdata.smk` | `cmslpc` (CPU) | $t\bar{t}$ stitched MC (`TTTo2L2Nu_stitched`, `TTToSemiLeptonic_stitched`, `TTToHadronic_stitched`) | Nominal inclusive JCM | `ttbar_PSData_stitched.yml` |
| **Step 4** | `PhaseE_4_FvT_training.smk` | `falcon` (GPU) | Subsample classifier inputs + 3-tag collision data (`data_3b_for_mixed`) | Subsample JCMs (`jetCombinatoricModel_SB_mix_v{m}.yml`) | 15 FvT model weights & friend JSONs (`friends_FvT_ttHbb_mixeddata_stitched_v{m}.json`) |
| **Step 5** | `PhaseE_5_analysis.smk` | `cmslpc` (CPU) | `mixeddata_4b` (15 subsamples across 4 Run 2 eras) | Subsample JCMs + SvB friend trees from Step 2 | `histAll_ttHbb_mixeddata_stitched.coffea`, comparison & analysis plots |
| **Step 6** | `PhaseE_6_closure.smk` | `cmslpc` (CPU) | 4-tag Mixed Data (`histAll_ttHbb_mixeddata_stitched.root`), 3-tag Data with 15 FvT friends (`histMixedBkg_data_3b_for_mixed.root`), Signal (`hist_signal_ttHbb.root`) | Subsample JCMs (used inside `stats_analysis/make_fvt_data3b_hists.py`) | Closure results `.pkl`, 30-bin diagnostic fit plots, subsample overlay plots |

#### Key Artifacts & Downstream Use:
* **Outputs Produced**: Background closure fit histograms, diagnostic plots, and systematic uncertainty pickle file (`hists_closure_*.pkl`).
* **Validation Criteria**:
  * **Multijet Ensemble Variance**: Minimizes adjacent bin pull correlation at Basis 3 ($r = 0.834$).
  * **Spurious Signal Test**: Passes at $\zeta = -0.17 \pm 0.05$ ($< 2\sigma$ from zero).
  * **Subsample Shape Consistency**: 15-subsample overlay confirms $< 3\%$ shape variance across all 30 bins.

**Execution Examples:**
```bash
# Run full Phase E pipeline (Steps 1 through 6) on cmslpc
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml \
    --cores 8

# Step 1: Make mixed data picoAODs
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_1_make_mixeddata.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml

# Step 2: Split 15 subsamples, compute JCMs, evaluate SvB friends
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_2_make_subsamples.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml

# Step 3: Make ttbar pseudo-data (MC subtraction check)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_3_make_ttbar_psdata.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml

# Step 4: Train 15 FvT models on Falcon GPU & evaluate friend trees
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_4_FvT_training.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml

# Step 5: Run Coffea analysis processor over 15 subsamples
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_5_analysis.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml

# Step 6: Two-stage closure fit, spurious signal test & plotting
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_6_closure.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb_mixeddata.yml
```

---


### Phase F: Analysis & Statistical Interpretation
* **Target Machine:** **`cmslpc`** (CPU batching with HTCondor/Dask + Combine container)
* **Coordinator:** `Snakefile_PhaseF.smk`
* **Sub-workflows:**
  * `Snakefile_PhaseF_1_analysis.smk`: Runs main Coffea analysis processor (`processor_HH4b.py`), merges histogram files, checks cutflow agreement against reference counts, and generates data/MC comparison plots.
  * `Snakefile_PhaseF_2_stats.smk`: Converts histogram distributions to Combine JSON format, generates CMS Combine datacards, builds workspaces, and computes expected limits, signal significance, and likelihood profile scans.

#### Key Artifacts & Required Downstream Updates:
* **After Phase F.1 (Analysis Processor)**:
  * *Outputs Produced*: `histAll_<label>.coffea`, cutflow summary `cutflow_<label>.yml`, and data/MC plots on EOS/web.
  * *Files to Add/Modify*: Update reference cutflow counts file (`known_Counts_<analysis>.yml`) if this run establishes a new validated baseline.
* **After Phase F.2 (Combine Stats)**:
  * *Outputs Produced*: Combine JSON histograms, datacards (`datacard_*.txt`), workspaces (`datacard_*.root`), limit outputs (`datacard_limits_*.json`), and likelihood profile scan PDFs/ROOT snapshots.

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
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseF_1_analysis.smk \
    --configfile coffea4bees/workflows/config/analysis_ttHbb.yml \
    --config test=true \
    -np
```
