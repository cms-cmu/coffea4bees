# MvD Classifier in processor_HH4b.py (`apply_MvD`)

## What it does

MvD (Mixed vs Data) is an alternative to FvT that reweights `mixeddata_all` (4b) events to look like detector data 4b. It applies **only** to `mixeddata_all` datasets — not `mix_v`, `mix_noTT_v`, or `mix_pz_v`.

**Weight formula**: `MvD = (p_d4 - p_t4) / p_mix4`

Applied to fourTag `mixeddata_all` events. JCM (4b) accounts for the jet combinatoric probability of the hemisphere-mixed events.

## Key implementation details

**`isMixedDataAll` flag** (`analysis/helpers/processor_config.py`):
```python
config["isMixedDataAll"] = "mixeddata_all" in dataset
```
This narrows the existing `isMixedData` flag to only the `mixeddata_all` type.

**Loading the friend tree** (`analysis/helpers/load_friend.py`):
`read_MvD_friend()` loads the MvD friend tree. In the processor, `load_MvD` is only called when `apply_MvD=True` and `isMixedDataAll=True`, so the friend tree entry in `friends_HH4b.yml` does not affect data/ttbar/signal processing.

**Weight application** (`analysis/helpers/event_weights.py`):
The MvD block in `add_pseudotagweights` runs when `apply_MvD and isMixedDataAll`:
- JCM is applied to **fourTag** events (contrast with FvT JCM which applies to threeTag)
- `mvd_weight = np.where(fourTag, event.MvD.MvD, 1.0)`
- The block returns early, so FvT/3b JCM path is never reached for `mixeddata_all`

FvT JCM (3b) and MvD JCM (4b) are **mutually exclusive by design** — the existing `JCM` parameter is reused for both, so they are never run simultaneously.

## Friend tree format

The MvD evaluate workflow produces a friend tree containing:
- `MvD` — per-event weight `(p_d4 - p_t4) / p_mix4`
- `p_mix4`, `p_d4`, `p_t4` — class probabilities

Registered in `coffea4bees/metadata/friends_HH4b.yml` under the `friends_Run3` anchor:
```yaml
MvD: root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_v2/friend/MvD/result.json@@analysis.0.merged
```
The `@@analysis.0.merged` key matches the `kfold.Merge` output structure, same as FvT.

## Config usage

```yaml
apply_MvD: true
apply_FvT: false     # don't apply FvT to mixeddata_all
apply_JCM: true
JCM_file: coffea4bees/analysis/weights/JCM/Run3_MvD/jetCombinatoricModel_SB_.yml
```

## Files

| File                                    | What changed                                                        |
|-----------------------------------------|---------------------------------------------------------------------|
| `analysis/helpers/processor_config.py`  | Added `isMixedDataAll` flag                                         |
| `analysis/helpers/load_friend.py`       | Added `read_MvD_friend()`                                           |
| `analysis/helpers/event_weights.py`     | Added `apply_MvD`/`isMixedDataAll` params and MvD weight block      |
| `analysis/processors/processor_HH4b.py` | Added `apply_MvD` param, `load_MvD` method, `load_friend_MvD` stage |
| `metadata/friends_HH4b.yml`             | Added MvD friend tree entry for Run3                                |
