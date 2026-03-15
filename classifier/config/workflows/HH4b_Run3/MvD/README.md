# MvD Classifier — Training & Evaluation Workflow

MvD (Mixed vs Detector) is the classifier used to reweight hemisphere-mixed `mixeddata_all` events to look like real detector 4b data, as an alternative to FvT.

For documentation on how MvD is applied in the analysis processor, see `coffea4bees/analysis/processors/apply_MvD.md`.

## Classification Task

The classifier distinguishes three classes of **fourTag** events:

| Label  | Source              | Selection                                        |
|--------|---------------------|--------------------------------------------------|
| `mix4` | `mixeddata_all`     | `(SB \| SR) & fourTag`                           |
| `d4`   | detector data       | `(SB \| SR) & fourTag & ~(SR & fourTag) & passHLT` (SR 4b blinded) |
| `t4`   | ttbar MC            | `(SB \| SR) & fourTag & passHLT`                 |

**Weight formula** (applied at analysis time to `mixeddata_all` fourTag events):
```
MvD = (p_d4 - p_t4) / p_mix4
```

## Model Architecture

HCR (Heterogeneous Classifier with Residuals) ensemble — same architecture as FvT. Inputs per event:

- **CanJet** (4 jets): `pt, eta, phi, mass`
- **NotCanJet** (up to 8 jets): `pt, eta, phi, mass, isSelJet`
- **Ancillary**: `year, nSelJets, xW, xbW`

Defined in `src/classifier/config/setting/HCR.py`.

## Training (`train.yml`)

**Dataset module:** `HCR.MvD.TrainBaseline`
- Loads picoAOD data from `coffea4bees/metadata/datasets_HH4b_Run3/archive/datasets_HH4b_Run3_2025_Run3_skims`
- Data sources: `mixed_all` (→ `mix4`), `detector` (→ `d4`), plus ttbar-labeled files (→ `t4`)
- HCR input features loaded via `coffea4bees/metadata/datasets_HH4b_Run3/classifier_inputs_MvD_Run3.json@@HCR_input`

**JCM weight** (applied to `mixeddata_all` only):
```
--JCM-weight "source:mixed_all" coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_.yml@@JCM_weights
```
The weight table has 12 entries indexed by `nSelJets` (starting at index 4). It corrects for the jet combinatoric probability of hemisphere-mixed 4b events. Applied to `fourTag` events only (contrast with FvT which applies JCM to `threeTag` events).

**Model module:** `HCR.MvD.baseline.Train`
- 3 kfolds, seed `MvD random`, offset 0
- Training: `FixedStep`, 20 epochs, batch size 1024
- Fine-tuning: `FixedStep`, 1 epoch, batch size 16384

**Output:** `{model}` (path provided as template parameter)

## Evaluation (`evaluate.yml`)

**Dataset module:** `HCR.MvD.Eval`
- Same picoAOD metadata and HCR inputs as training
- Data sources: `detector`, `mixed_all` (no JCM weight applied at eval time)

**Model module:** `HCR.MvD.baseline.Eval`
- Evaluates the `Final` checkpoint from each kfold: `{model}/result.json`

**Post-processing:** `kfold.Merge`
- Merges 3 kfold predictions into a single output named `MvD`
- Step size 100000, 5 workers

**Output:** `{MvD}` — friend tree registered in `coffea4bees/metadata/friends_HH4b.yml` as:
```yaml
MvD: root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_v2/friend/MvD/result.json@@analysis.0.merged
```

## Running the Workflow

```bash
# Training (from barista root, inside classifier container)
./run_container classifier python classifier.py \
    --workflow coffea4bees/classifier/config/workflows/HH4b_Run3/MvD/train.yml \
    --model <output_model_path>

# Evaluation
./run_container classifier python classifier.py \
    --workflow coffea4bees/classifier/config/workflows/HH4b_Run3/MvD/evaluate.yml \
    --model <output_model_path> \
    --MvD <output_friend_path>
```

## Key Code Locations

| Component                  | File                                                                 |
|----------------------------|----------------------------------------------------------------------|
| Dataset config (MvD)       | `src/classifier/config/dataset/HCR/MvD.py`                          |
| JCM weight application     | `coffea4bees/classifier/compatibility/JCM/fit.py`                    |
| JCM weight table           | `coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_.yml` |
| HCR input settings         | `src/classifier/config/setting/HCR.py`                               |
| Analysis application       | `coffea4bees/analysis/helpers/event_weights.py`                      |
| Analysis config example    | `coffea4bees/analysis/metadata/HH4b_MvD.yml`                         |
