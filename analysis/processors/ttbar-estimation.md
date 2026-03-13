# TTbar Estimation from Data (`plot_ttbar_with_weights`)

When `plot_ttbar_with_weights=True`, the processor estimates ttbar contamination by reweighting 3b data events using FvT classifier outputs (`FvT.d3_to_t4` and `FvT.d3_to_t3`). These weights are computed in `analysis/helpers/event_weights.py` inside `include_pseudotag_in_weight` and stored on the event array as `weight_d3_to_t4` and `weight_d3_to_t3`. They are only available when `apply_FvT=True` and `isDataForMixed=False`.

**Histograms**: `filling_nominal_histograms` is called twice more with `processName="TTbar4b_from_d3"` / `"TTbar3b_from_d3"` and `weight_name="weight_d3_to_t4"` / `"weight_d3_to_t3"`.

**Cutflow**: A single `_cutFlow_ttbar` object is filled with a custom scheme — both tag bins come from the same threeTag events but with different weights:
- `_cutFlowFourTag[cut]` = `sum(weight_d3_to_t4)` over threeTag events (ttbar → 4b contribution)
- `_cutFlowThreeTag[cut]` = `sum(weight_d3_to_t3)` over threeTag events (ttbar → 3b contribution)
- `_cutFlowTwoTag[cut]` = 0

Output key per dataset: `f"TTbar_from_d3_{era}"` where `era = dataset.removeprefix("data_")`, e.g. `"TTbar_from_d3_2023_preBPixC12"`. This is filled only for nominal (not systematic shifts) and only for the detailed post-candidate cuts (same set as `fill_detailed_cutflows`).

**Important**: `cutflow_4b.addOutput` uses `setdefault` so multiple `addOutput` calls per chunk (nominal data + TTbar) do not overwrite each other.

# Cutflow Structure (`analysis/helpers/cutflow.py`)

`cutflow_4b` tracks three tag regions: `_cutFlowTwoTag`, `_cutFlowThreeTag`, `_cutFlowFourTag`. Each stores `(weighted_sum, raw_count)` per cut name. Early cuts (HLT, jet multiplicity, etc.) are filled in `build_cutflow` with `allTag=True`. Detailed cuts after candidate building are filled in `fill_detailed_cutflows` split by actual tag. The `fill_cutflow_with_and_without_trig` helper always also fills a `{cut}_woTrig` variant excluding the trigger SF weight.
