# Session — year-override-histograms (2026-03-16)

## What we did
- Added `year_override` bool option to collapse per-year histogram bins into `Run2`/`Run3` eras
- Implemented `YEAR_OVERRIDE_MAP` and `_apply_year_override()` helper in `filling_histograms.py`
- Threaded `year_override` param through `filling_nominal_histograms`, `filling_syst_histograms`, and all call sites in `processor_HH4b.py`
- Added `year_override: true` to `HH4b_MvD.yml` config (user set to `true` after initial `false`)
- Pushed `MvD` branch to coffea4bees remote and opened MR #639
- Ran CI pipeline #14296301 — passed (52 jobs; 1 pre-existing `allow_failure` failure)

## Decisions
- Mapping: `201*` → `Run2`, `202*` → `Run3`; years not matching either prefix pass through unchanged — handles legacy `UL18`-style labels safely
- `year_override=False` default — fully backward-compatible, no existing call sites changed
- Config key lives under `config:` in the YAML so it is unpacked directly as a kwarg to the processor `__init__` by `runner.py`

## Files changed
- `coffea4bees/analysis/helpers/filling_histograms.py` — added `YEAR_OVERRIDE_MAP`, `_apply_year_override()`, `year_override` param to both fill functions
- `coffea4bees/analysis/processors/processor_HH4b.py` — added `year_override` param to `__init__`, stored as `self.year_override`, passed to all 5 fill calls
- `coffea4bees/analysis/metadata/HH4b_MvD.yml` — added `year_override: true`

## Open threads
- (none)
