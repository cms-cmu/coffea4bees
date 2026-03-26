# Session — btag-sf-run3-fixes (2026-03-25)

## What we did
- Diagnosed `RuntimeError: Index below bounds in Binning for input argument 4 value: -1.000000` in `apply_btag_sf` — caused by jets with `btagScore = -1` reaching correctionlib
- Added debug logging in `get_sf` to print pt, eta, phi, btagScore, jetId, btagPNetB, correction_file, and correction_type for any jets with `btagScore < 0`
- Added clipping of `btagScore` to `[0, 1]` before passing to correctionlib
- Added `isRun3` parameter to `add_btagweights` in `event_weights.py`; selects `correction_type="particleNet_shape"` for Run3, `"deepJet_shape"` for Run2
- Propagated `isRun3` to all five call sites: `processor_HH4b.py`, `processor_HH4b_resonant.py`, `sub_sample_MC.py`, `make_mixed_data.py`, `make_declustered_data_4b.py`

## Decisions
- Clip btagScore to [0,1] rather than filtering jets — a score of 0 gives the untagged SF, which is correct for edge-case jets that couldn't be scored
- Use `config["isRun3"]` (not `year`-based logic) to select correction type, consistent with existing config patterns

## Files changed
- `src/physics/common.py` — added debug logging for btagScore<0 jets; added np.clip on btagScore before correctionlib evaluate
- `coffea4bees/analysis/helpers/event_weights.py` — added `isRun3: bool = False` param; conditioned correction_type on isRun3
- `coffea4bees/analysis/processors/processor_HH4b.py` — pass `isRun3=self.config["isRun3"]` to `add_btagweights`
- `coffea4bees/analysis/processors/processor_HH4b_resonant.py` — pass `isRun3=self.config["isRun3"]` to `add_btagweights`
- `coffea4bees/skimmer/processor/sub_sample_MC.py` — pass `isRun3=config["isRun3"]` to `add_btagweights`
- `coffea4bees/skimmer/processor/make_mixed_data.py` — pass `isRun3=config["isRun3"]` to `add_btagweights`
- `coffea4bees/skimmer/processor/make_declustered_data_4b.py` — pass `isRun3=config["isRun3"]` to `add_btagweights`

## Open threads
- Debug logging in `src/physics/common.py` still present — should be removed once root cause of btagScore=-1 jets is understood
