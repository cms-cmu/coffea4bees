# Session — apply-MvD-jcm-without-weight (2026-03-27)

## What we did
- Added `apply_MvD_weight` boolean to decouple "use MvD JCM path" from "apply MvD classifier score as event weight"
- Updated `HH4b_MvD.yml` to set `apply_MvD_weight: false` and cleared `hist_cuts` (removed `"highMvD"`)
- Gated MvD friend tree loading in processor on `apply_MvD and apply_MvD_weight`
- Gated `event["highMvD"]`, `weight_mix4_to_t4_MvD`, and `weight_mix4_to_t4_MvD_noMvD` computation on `apply_MvD_weight`
- Gated `MvDHists` and `MvD_noMvD` histogram on `apply_MvD_weight` in `filling_histograms.py`
- Gated `plot_ttbar_with_MvD_weights` TTbar4b_from_MvD block on `self.apply_MvD_weight`

## Decisions
- Keep weight entry named `"MvD"` in coffea Weights object even when only JCM is applied — avoids downstream histogram key changes
- When `apply_MvD_weight=False`, weight for fourTag events = `jcm_weight` (not `jcm_weight * event.MvD.MvD`)
- `selJets_noMvD` histogram kept under `apply_MvD` gate (not `apply_MvD_weight`) since `weight_noMvD` is still set in both cases
- `hist_cuts: []` in YAML when `apply_MvD_weight: false` — `highMvD` field is not set so it can't be used as a histogram cut axis

## Files changed
- `coffea4bees/analysis/metadata/HH4b_MvD.yml` — added `apply_MvD_weight: false`, changed `hist_cuts` from `["highMvD"]` to `[]`
- `coffea4bees/analysis/helpers/event_weights.py` — added `apply_MvD_weight` param to `add_pseudotagweights`; conditioned MvD score multiplication, `highMvD`, and `weight_mix4_to_t4_MvD` on it
- `coffea4bees/analysis/processors/processor_HH4b.py` — added `apply_MvD_weight` init param and `self.apply_MvD_weight`; gated `load_MvD()`, `plot_ttbar_with_MvD_weights` block, and passed param through to `add_pseudotagweights` and `filling_nominal_histograms`
- `coffea4bees/analysis/helpers/filling_histograms.py` — added `apply_MvD_weight` param; gated `MvDHists` and `MvD_noMvD` histogram on it

## Open threads
- MvD weights have not yet been fit; once fit, set `apply_MvD_weight: true` in `HH4b_MvD.yml` and restore `hist_cuts: ["highMvD"]` to re-enable full MvD weight application
