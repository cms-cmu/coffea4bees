# TTbar Estimation from Data (`plot_ttbar_with_weights`)

When `plot_ttbar_with_weights=True`, the processor estimates ttbar contamination by reweighting 3b data events using FvT classifier outputs (`FvT.d3_to_t4` and `FvT.d3_to_t3`). These weights are computed in `analysis/helpers/event_weights.py` inside `add_pseudotagweights` and stored on the event array as `weight_d3_to_t4` and `weight_d3_to_t3`. They are only available when `apply_FvT=True` and `isDataForMixed=False`.

**Histograms**: `filling_nominal_histograms` is called twice more with `processName="TTbar4b_from_d3"` / `"TTbar3b_from_d3"` and `weight_name="weight_d3_to_t4"` / `"weight_d3_to_t3"`.

**Cutflow**: A single `_cutFlow_ttbar` object is filled with a custom scheme — both tag bins come from the same threeTag events but with different weights:
- `_cutFlowFourTag[cut]` = `sum(weight_d3_to_t4)` over threeTag events (ttbar → 4b contribution)
- `_cutFlowThreeTag[cut]` = `sum(weight_d3_to_t3)` over threeTag events (ttbar → 3b contribution)
- `_cutFlowTwoTag[cut]` = 0

Output key per dataset: `f"TTbar_from_d3_{era}"` where `era = dataset.removeprefix("data_")`, e.g. `"TTbar_from_d3_2023_preBPixC12"`. This is filled only for nominal (not systematic shifts) and only for the detailed post-candidate cuts (same set as `fill_detailed_cutflows`).

**Important**: `cutflow_4b.addOutput` uses `setdefault` so multiple `addOutput` calls per chunk (nominal data + TTbar) do not overwrite each other.

# TTbar Estimation from MvD (`plot_ttbar_with_MvD_weights`)

When `plot_ttbar_with_MvD_weights=True` and the dataset is `mixeddata_all`, the processor fills a TTbar estimate using the MvD classifier to project mix4 events into the t4 (4b data) space. This is the MvD analogue of `plot_ttbar_with_weights`.

**Weight**: `weight_mix4_to_t4_MvD`, computed in `add_pseudotagweights` (MvD path):
```
w = base_weight * jcm_weight * (p_t4 / p_mix4)   for fourTag events
w = base_weight                                    otherwise
```
where `base_weight = weights.partial_weight(exclude=["MvD"])` (all weights except the MvD
classifier) and `jcm_weight` is the 4b JCM pseudo-tag weight.

**Histograms**: `filling_nominal_histograms` is called with `processName="TTbar4b_from_MvD"`, `tag_list=["fourTag"]`, and `weight_name="weight_mix4_to_t4_MvD"`.

# `_noFvT` / `_noMvD` Histograms in TTbar Filling

Several histograms in `filling_nominal_histograms` are defined with a hardcoded weight
override to show distributions without the classifier reweighting:

| Histogram        | Default override weight | Purpose                              |
|------------------|------------------------|--------------------------------------|
| `FvT_noFvT`      | `weight_noFvT`         | FvT score distribution, no FvT weight |
| `SvB_MA_noFvT`   | `weight_noFvT`         | SvB MA, no FvT weight                |
| `MvD_noMvD`      | `weight_noMvD`         | MvD score distribution, no MvD weight |
| `selJets_noMvD`  | `weight_noMvD`         | Jet kinematics, no MvD weight        |

For TTbar filling these overrides would replace the X_to_t4 conversion weight with the
plain `_noX` weight (which lacks the conversion factor). To fix this, `filling_nominal_histograms`
accepts two optional parameters:

```python
weight_noFvT_override: str = None   # replaces "weight_noFvT" for _noFvT histograms
weight_noMvD_override: str = None   # replaces "weight_noMvD" for _noMvD histograms
```

The processor passes these when calling for TTbar:

| Call                     | `weight_name`            | `weight_noFvT_override`     | `weight_noMvD_override`        |
|--------------------------|--------------------------|-----------------------------|--------------------------------|
| `TTbar4b_from_d3`        | `weight_d3_to_t4`        | `weight_d3_to_t4_noFvT`     | —                              |
| `TTbar3b_from_d3`        | `weight_d3_to_t3`        | `weight_d3_to_t3_noFvT`     | —                              |
| `TTbar4b_from_MvD`       | `weight_mix4_to_t4_MvD`  | —                           | `weight_mix4_to_t4_MvD_noMvD`  |

The `_noFvT` and `_noMvD` suffix aliases are set in `event_weights.py` immediately after
their source weights. They are numerically identical to the source weights (the X_to_t4
weights already exclude the classifier), but the distinct names make the intent explicit
and allow `filling_histograms.py` to select the right weight per-histogram.

# Cutflow Structure (`analysis/helpers/cutflow.py`)

`cutflow_4b` tracks three tag regions: `_cutFlowTwoTag`, `_cutFlowThreeTag`, `_cutFlowFourTag`. Each stores `(weighted_sum, raw_count)` per cut name. Early cuts (HLT, jet multiplicity, etc.) are filled in `build_cutflow` with `allTag=True`. Detailed cuts after candidate building are filled in `fill_detailed_cutflows` split by actual tag. The `fill_cutflow_with_and_without_trig` helper always also fills a `{cut}_woTrig` variant excluding the trigger SF weight.
