# Phase E: Run 2 $t\bar{t}H(b\bar{b})$ Two-Stage Mixed-Data Background Closure

## 1. Overview & Physics Motivation

In the CMS Run 2 search for $t\bar{t}H(b\bar{b})$ in the fully hadronic (and semi-leptonic) final states, non-resonant multi-jet production ($t\bar{t}$ + heavy-flavor jets and pure QCD multi-jet) represents the dominant background. Because simulating multi-jet QCD to the required precision is intractable, we employ a data-driven **Mixed Data** technique paired with a **Two-Stage Closure** statistical test.

### Core Principles
1. **Mixed Data Generation**: We decouple jet kinematics from correlations by mixing hemispheres from 3-tag events ($3b$) to generate synthetic 4-tag multi-jet events without true signal contamination.
2. **$t\bar{t}$ Contamination & Subtraction**: Genuine $t\bar{t}$ events are present in both the 3-tag and 4-tag regions. We use a stitched $t\bar{t}$ MC sample across all Run 2 eras (2016_preVFP, 2016_postVFP, 2017, 2018) to model $t\bar{t}$ pseudo-data and subtract $t\bar{t}$ components from the data-driven mixture.
3. **15 Independent Subsamples**: To reduce statistical uncertainty and systematically probe variance, mixed data is partitioned into 15 disjoint subsamples (`subsample_0` through `subsample_14`).
4. **Two-Stage Reweighting & Closure**:
   - **Stage 1**: Correct kinematic discrepancies between $3b$ and $4b$ using the Jet Combinatoric Model (JCM).
   - **Stage 2**: Train a deep Four-vs-Three (FvT) classifier (HCR architecture) to reweight the mixed data + $t\bar{t}$ background model into the signal region ($4b$).
   - **Statistical Validation**: A binned maximum-likelihood fit over the SvB classifier score evaluates spurious signal bias ($\zeta$) and background closure variance ($r$).

---

## 2. Dataset and JCM Calibration Contract

The table below explicitly maps out the input datasets, JCM weights, and outputs for every sub-step of Phase E. **Refer to this matrix before launching any workflow stage.**

| Sub-workflow | Stage Name | Input Datasets Required | JCM Weights File | Primary Outputs | Execution Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase E.1** | Hemisphere Mixing | Run 2 Data (3-tag selected, all eras: `2016_preVFP`, `2016_postVFP`, `2017`, `2018`) | N/A (kinematic hemisphere pairing) | `output/mixeddata/Run2/` (PicoAOD root trees) | `cmslpc` (CPU) |
| **Phase E.2** | Subsamples & Friends | `mixeddata/Run2` trees, Stitched $t\bar{t}$ MC (`TTToHadronic`, `TTToSemiLeptonic`, `TTTo2L2Nu`) | `jetCombinatoricModel_SB_2024_v2_stitched.yml` | 15 subsample trees (`subsample_0` .. `subsample_14`) + friend trees (`SvB`, `FvT`) | `cmslpc` / `falcon` |
| **Phase E.3** | $t\bar{t}$ Pseudo-data | Stitched $t\bar{t}$ MC (inclusive Run 2) | `jetCombinatoricModel_SB_2024_v2_stitched.yml` | `output/ttbar_PSData/` | `cmslpc` (CPU) |
| **Phase E.4** | FvT Reweighting Training | Data $3b$, Mixed Data subsamples, Stitched $t\bar{t}$ $3b$ | `jetCombinatoricModel_SB_2024_v2_stitched.yml` | Trained FvT PyTorch weights & checkpoints | `falcon` (GPU) |
| **Phase E.5** | Analysis Histogramming | 15 Mixed Data subsamples, Stitched $t\bar{t}$ MC, $t\bar{t}H(b\bar{b})$ Signal MC (all eras) | `jetCombinatoricModel_SB_2024_v2_stitched.yml` | `output/ttHbb_mixeddata_stitched_closure_varbins/` (`histAll_*.coffea`) | `cmslpc` (CPU) |
| **Phase E.6** | Statistical Closure Fit | Analysis histograms from Phase E.5 (`coffea` files) | Embedded in histograms | ROOT datacards (`.root`), fit logs, pull & closure plots | `cmslpc` (CPU) |

---

## 3. Workflow Architecture & Step-by-Step Execution

All Snakefiles are located in `coffea4bees/workflows/`. Execute commands using the analysis container via `./run_container snakemake` on `cmslpc`.

```
                    ┌──────────────────────────────────────────────┐
                    │ Phase E.1: Snakefile_PhaseE_1_make_mixeddata  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Phase E.2: Snakefile_PhaseE_2_make_subsamples │
                    └──────────────────────┬───────────────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        ▼                                     ▼
     ┌────────────────────────────────────┐ ┌────────────────────────────────────┐
     │ Phase E.3: make_ttbar_psdata.smk   │ │ Phase E.4: FvT_training.smk        │
     └──────────────────┬─────────────────┘ └──────────────────┬─────────────────┘
                        │                                      │
                        └──────────────────┬───────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Phase E.5: Snakefile_PhaseE_5_analysis       │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │ Phase E.6: Snakefile_PhaseE_6_closure        │
                    └──────────────────────────────────────────────┘
```

### Phase E.1: Generate Mixed Data
Pairs hemispheres from Run 2 3-tag data to synthesize multi-jet kinematics.
```bash
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_1_make_mixeddata.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -j 8
```

### Phase E.2: Split Subsamples & Evaluate Friends
Splits the mixed data into 15 equal-luminosity subsamples and evaluates SvB/FvT friend trees.
- **JCM Requirement**: `jetCombinatoricModel_SB_2024_v2_stitched.yml`
```bash
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_2_make_subsamples.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -j 8
```

### Phase E.3: Generate $t\bar{t}$ Pseudo-data
Extracts and weights stitched $t\bar{t}$ components to combine with mixed data for full background subtraction.
```bash
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_3_make_ttbar_psdata.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -j 8
```

### Phase E.4: Train FvT Reweighting Classifier
Trains the deep neural network (HCR) on `falcon` GPU to predict the kinematic transfer function from 3-tag to 4-tag.
```bash
# On falcon (GPU cluster)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_4_FvT_training.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -j 4
```

### Phase E.5: Analysis Processing & Histogramming
Runs `processor_ttHbb.py` across all 15 subsamples, stitched $t\bar{t}$ MC, and $t\bar{t}H(b\bar{b})$ signal MC.
- **Config**: `workflows/config/analysis_ttHbb_mixeddata.yml`
- **Output**: Generates `histAll_ttHbb_stitched.coffea` and subsample histogram files.
```bash
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_5_analysis.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -j 8
```

### Phase E.6: Statistical Closure Fit & Validation
Converts the `.coffea` histograms to ROOT histograms with a 30-bin variable scheme (optimized for signal significance and statistical control) and runs the two-stage closure fit.
- **Script**: `stats_analysis/runTwoStageClosure.py`
```bash
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE_6_closure.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -j 4
```

---

## 4. Physics Validation Benchmarks

The completed pipeline was validated against the full Run 2 dataset. Key metrics that must be verified in the output plots and fit summaries:

1. **Variance Ratio ($r$)**:
   - Metric: $r = \sigma_{\text{fit}}^2 / \sigma_{\text{expected}}^2$
   - Baseline Benchmark: $r = 0.834$ (well within the theoretical requirement $r < 1.0$, confirming that mixed data statistical fluctuations do not blow up the background uncertainty).
2. **Spurious Signal Bias ($\zeta$)**:
   - Metric: Fitted signal strength in background-only pseudo-data.
   - Benchmark: $\zeta = -0.17 \pm 0.05$ (compatible with small nuisance pull, demonstrating unbiased background modeling).
3. **Subsample Variance Envelope**:
   - The 15 individual subsample curves must lie within a $\pm 3\%$ envelope across the entire SvB discriminative spectrum $[0, 1]$.
   - In `plots_closure/`, inspect `closure_subsamples_overlay.pdf` to verify no localized shape divergence.

---

## 5. Master Pipeline Execution

To run or dry-run the complete end-to-end pipeline in a single command:
```bash
# Dry-run
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -n

# Full Execution (ensure GPU training in Phase E.4 is executed beforehand or scheduled on falcon)
./run_container snakemake -s coffea4bees/workflows/Snakefile_PhaseE.smk \
    --config output_path=output/ttHbb_mixeddata_stitched_closure_varbins -j 12
```
