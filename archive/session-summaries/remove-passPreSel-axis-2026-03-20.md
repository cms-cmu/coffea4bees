# Session — remove-passPreSel-axis (2026-03-20)

## What we did
- Removed `passPreSel` from `hist_cuts` in 7 YAML metadata configs (changed `[passPreSel]` → `[]`)
- Removed `hist_cuts = ['passPreSel']` default from 5 processor files
- Updated 6 plotting scripts to replace `cut="passPreSel"` with `cut=None`
- Updated 4 CI shell scripts to remove `passPreSel/` from output path checks and remove `-c passPreSel` from JCM tool invocations
- Updated `make_jcm_weights.py` to make `passPreSel` axis conditional (fallback when axis absent)
- Updated 3 test reference files (JCM YAML, known plot counts) to remove `_passPreSel` key suffixes
- Fixed 6 CI failures across 3 push cycles: axis index shift in `convert_hist_to_json_closure.py`, `cut=None` crash in `jcm_tools/helpers.py`, key lookup fallback in `jetCombinatoricModel.py`, empty-cut list in `makePlots_unsup.py`, `dumpPlotCounts.py` test vectors, and unsup CI path check
- Pipeline #14334431 passed (MR #647)

## Decisions
- `cut=["failSvB","passSvB"]` in `makePlots_unsup.py` comparison section — `cut=None` routes to config-based dispatch which can't find unsup processes by name; list dispatch queries histogram directly
- `convert_hist_to_json_closure.py` axis indices updated 3→2 and 4→3 — passPreSel was axis[2], removing it shifted tag and region down one
- `jetCombinatoricModel.py` uses `.get()` with fallback to unsuffixed key — allows JCM files without a cut suffix (new format) alongside old files with `_passPreSel` suffix
- `tools_memory_test` failure (1763 MB vs 1760 MB threshold) classified as pre-existing/flaky — `allow_failure: true` in CI, unrelated to passPreSel removal

## Files changed
- `coffea4bees/analysis/metadata/HH4b.yml` — `hist_cuts: []`
- `coffea4bees/analysis/metadata/HH4b_dataUL17B.yml` — `hist_cuts: []`
- `coffea4bees/analysis/metadata/HH4b_event_displays.yml` — `hist_cuts: []`
- `coffea4bees/analysis/metadata/HH4b_mixed_data_new.yml` — `hist_cuts: []`
- `coffea4bees/analysis/metadata/HH4b_nottcheck.yml` — `hist_cuts: []`
- `coffea4bees/analysis/metadata/HH4b_signals.yml` — `hist_cuts: []`
- `coffea4bees/analysis/metadata/HH4b_signals_Run.yml` — `hist_cuts: []`
- `coffea4bees/analysis/processors/processor_HH4b.py` — `hist_cuts = []`
- `coffea4bees/analysis/processors/processor_HH4b_resonant.py` — `hist_cuts = []`
- `coffea4bees/analysis/processors/processor_HH4b_2Dplots.py` — `self.histCuts = []`
- `coffea4bees/analysis/processors/processor_unsup.py` — `hist_cuts = []`
- `coffea4bees/analysis/processors/processor_unsup_friend.py` — `hist_cuts = []`
- `coffea4bees/analysis/helpers/jetCombinatoricModel.py` — fallback to unsuffixed param keys when cut is None
- `coffea4bees/analysis/jcm_tools/helpers.py` — guard `get_cut_dict` for `cut=None`
- `coffea4bees/analysis/jcm_tools/make_jcm_weights.py` — conditional passPreSel axis handling, default `-c None`
- `coffea4bees/analysis/tests/jetCombinatoricModel_SB_Coffea_new.yml` — removed `_passPreSel` key suffixes
- `coffea4bees/analysis/tests/jetCombinatoricModel_SB_ROOT_new.yml` — removed `_passPreSel` key suffixes
- `coffea4bees/jet_clustering/make_jet_splitting_PDFs.py` — replace passPreSel cut_dict with `{}`
- `coffea4bees/plots/iPlot.py` — example docstrings updated
- `coffea4bees/plots/makePlots.py` — `cut=None`, `["failSvB","passSvB"]` cut list
- `coffea4bees/plots/makePlotsMixed.py` — `cut=None` (2 locations)
- `coffea4bees/plots/makePlotsMixedVsDataVs3b.py` — `cut=None` (2 locations)
- `coffea4bees/plots/makePlots_unsup.py` — `cut=None` for main plots, `cut=["failSvB","passSvB"]` for comparison
- `coffea4bees/plots/makeRocPlot.py` — `cut=None`
- `coffea4bees/plots/tests/iPlot_test.py` — `cut=None` throughout
- `coffea4bees/plots/tests/known_PlotCounts.yml` — keys renamed, `cut: null`
- `coffea4bees/scripts/analysis-plot.sh` — removed `passPreSel/` from path checks
- `coffea4bees/scripts/analysis-test.sh` — sed pattern updated to `hist_cuts: []`
- `coffea4bees/scripts/analysis-unsup-plot.sh` — path check updated to `failSvB_vs_passSvB/region_SR/...`
- `coffea4bees/scripts/tools-make-jcm-weights.sh` — removed `-c passPreSel`
- `coffea4bees/stats_analysis/convert_hist_to_json_closure.py` — axes[3/4]→axes[2/3], removed `'passPreSel': True`
- `coffea4bees/tests/dumpPlotCounts.py` — test_vectors use `cut=None`, key generation handles `None`
- `src/plotting/plots.py` — docstring examples updated

## Open threads
- (none)
