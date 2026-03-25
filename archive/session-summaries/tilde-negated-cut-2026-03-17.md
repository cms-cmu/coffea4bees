# Session — tilde-negated-cut (2026-03-17)

## What we did
- Implemented `~` prefix notation for negated cuts in the plotting API
- Modified `get_cut_dict` to strip `~` and select `False` for the named axis
- Added `cut_to_label` helper to produce readable labels ("pass X" / "fail X")
- Fixed cut validation in `get_plot_dict_from_config` to strip `~` before checking `cfg.cutList`
- Updated `_handle_cut_list` to use `cut_to_label` for auto-generated legend entries
- Added pass/fail example to `iPlot.py` `examples()`

## Decisions
- `~` prefix chosen over new parameters — keeps all cut arguments as plain strings with no API changes
- `cut_to_label` produces "pass X" / "fail X" so legend reads "data pass passPreSel" / "data fail passPreSel" rather than "data passPreSel" / "data ~passPreSel"

## Files changed
- `src/plotting/helpers.py` — `get_cut_dict` updated for `~` prefix; `cut_to_label` added
- `src/plotting/helpers_make_plot_dict.py` — validation strips `~`; `_handle_cut_list` uses `cut_to_label`
- `coffea4bees/plots/iPlot.py` — pass/fail example added to `examples()`

## Open threads
- CI not yet run; `/test-barista` should be executed to confirm no regressions
