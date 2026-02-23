# Hemisphere Mixing

## Physics Motivation

Hemisphere mixing is a data-driven technique to create synthetic background samples for the HH→4b analysis.

### Why It Works

The dominant 4b background comes from QCD multi-jet production, primarily through 2→2 gluon scattering. In these events, the two "hemispheres" (roughly, the two sides of the event separated by the plane perpendicular to the thrust axis) are essentially **independent** - they result from separate parton fragmentation chains.

In contrast, **signal events (HH→4b)** have **correlated structure** between hemispheres: each Higgs boson decays to a pair of b-jets, creating characteristic di-jet mass peaks. The two Higgs candidates span both hemispheres, introducing inter-hemisphere correlations.

### The Mixing Procedure

1. **Build a hemisphere library**: Split background events into hemispheres and store them with relevant kinematic properties.

2. **Mix hemispheres**: For each event, replace one or both hemispheres with pre-computed hemispheres from the library, matched on appropriate kinematic variables.

3. **Result**: The mixed events preserve single-hemisphere kinematics (jet pT, η, multiplicity) while **breaking inter-hemisphere correlations** - effectively creating a signal-depleted background model.

### Key Properties

- **Preserves**: Overall background kinematics, jet distributions, single-hemisphere structure
- **Suppresses**: HH→4b signal (correlations between hemispheres are destroyed)
- **Use case**: Data-driven background estimation, closure tests, systematic studies

### Hemisphere Matching Algorithm

When replacing a hemisphere from a 3-tag event, we need to find the "best" replacement from the hemisphere library. This is done via nearest-neighbor search in a 4-dimensional space of **hemisphere summary variables**:

| Variable | Description |
|----------|-------------|
| `pz` | Combined p_z of all jets in the hemisphere |
| `sumPt_T` | Combined p_T projection **parallel** to the thrust axis |
| `sumPt_T_minor` | Combined p_T projection **perpendicular** to the thrust axis |
| `combinedMass` | Combined invariant mass of all jets in the hemisphere |

**Matching constraints**:
- Hemispheres are matched only within the same **jet multiplicity bin** (e.g., a 2-jet hemisphere only matches other 2-jet hemispheres)
- Distance metric: **Euclidean distance** in the 4D summary variable space
- Implementation uses **k-d trees** for efficient nearest-neighbor lookup

## Implementation

### Step 1: Build the Hemisphere Library

**Processor**: `coffea4bees/analysis/processors/processor_make_hemi_library.py`

**Example script**: `coffea4bees/scripts/mixeddata-cluster-Run3.sh`

**Config**: `coffea4bees/analysis/metadata/make_hemi_library_4b.yml`

The processor:
1. Applies event selection (lumimask, noise filters, HLT, jet multiplicity)
2. Selects 4-tag events (`fourTag` requirement)
3. Optionally subtracts TTbar contribution using FvT weights (`subtract_ttbar_with_weights`)
4. Splits selected events into hemispheres via `split_events_into_hemispheres()` from `mixing_helpers.py`
5. Writes both positive and negative hemispheres to ROOT files

Output: ROOT files containing hemisphere data, stored at `{base_path}/{dataset}/hemisphereLib_{chunk_uuid}_{start}_{stop}.root`

```bash
# Run for Run 3 data
./run_container bash coffea4bees/scripts/mixeddata-cluster-Run3.sh --output-base output/
```

### Step 2: Mix Hemispheres (Create Synthetic Data)

**Processor**: `coffea4bees/skimmer/processor/make_mixed_data.py` (class `HemiMixer`)

**Example script**: `coffea4bees/scripts/mixeddata-make-dataset-Run3.sh`

**Config**: `coffea4bees/skimmer/metadata/mixeddata_Run3.yml`

The processor:
1. Loads the hemisphere library (from Step 1) and builds k-d trees for efficient nearest-neighbor lookup
2. Applies event selection on **3-tag events** (not 4-tag)
3. Optionally subtracts TTbar using FvT weights
4. Applies JCM (Jet Combinatoric Model) pseudo-tag weights
5. Splits each 3-tag event into positive/negative hemispheres
6. For each hemisphere, finds the best match from the library using k-d tree search on matching variables:
   - `sumPt_T_minor`, `sumPt_T`, `combinedMass`, `pz`
7. Replaces event hemispheres with matched library hemispheres
8. Outputs mixed PicoAOD with new jets and metadata (original/new hemisphere info, match distance)

```bash
# Run for Run 3 data
./run_container bash coffea4bees/scripts/mixeddata-make-dataset-Run3.sh --output-base output/
```

## Boost-Corrected Matching (Optional Mode)

### Motivation

The current matching algorithm penalizes hemispheres with different `pz` values, even if their transverse structure is identical. This is suboptimal because:

1. **The z-axis is special in hadron colliders**: The partonic center-of-mass frame is boosted along the beam axis (z) by an unknown amount that depends on the parton momentum fractions (x₁, x₂). Two hemispheres that are physically identical in their rest frames can have very different lab-frame pz values simply due to different initial-state boosts.

2. **Lorentz boosts along z preserve transverse properties**: The transverse momentum projections (sumPt_T, sumPt_T_minor) and invariant mass are either unchanged or simply related under longitudinal boosts. These variables capture the "intrinsic" hemisphere structure.

3. **Limited effective library statistics**: By requiring similar pz, we reject many hemispheres that would otherwise be excellent matches in their transverse properties. This effectively reduces the usable library size.

### Proposed Change

**Match on 3 variables instead of 4**, then **correct pz with a Lorentz boost**:

1. **Modified matching**: Use only transverse variables for k-d tree lookup:
   - `sumPt_T` (pT projection parallel to thrust axis)
   - `sumPt_T_minor` (pT projection perpendicular to thrust axis)
   - `combinedMass` (invariant mass)

2. **Post-match boost correction**: After finding the best match, apply a Lorentz boost along z to the matched hemisphere's jets so that the combined pz matches the target:
   - Compute rapidity of matched hemisphere (from library): y_match = 0.5 × ln((E + pz_match)/(E − pz_match))
   - Compute rapidity of target hemisphere (from 3-tag event): y_target = 0.5 × ln((E + pz_target)/(E − pz_target))
   - Boost each jet in the matched hemisphere by Δy = y_target − y_match (i.e., β_z = tanh(Δy))
   - If pz values are already similar, Δy ≈ 0 and the boost is negligible

### Expected Benefits

- **Increased effective library statistics**: Hemispheres that were previously "far away" due to pz differences become viable matches
- **Better transverse matching**: By not penalizing pz differences, the algorithm can focus on finding the best transverse-structure match
- **Physically motivated**: The correction respects Lorentz invariance - we're matching on boost-invariant properties and then applying the appropriate boost

### Usage

The boost-corrected matching is implemented as an **optional mode**, preserving the existing 4D matching as the default. This allows:

- **A/B comparison**: Run both methods on the same inputs to evaluate the impact on synthetic data quality
- **Backward compatibility**: Existing workflows continue to work unchanged
- **Validation**: Compare closure tests, kinematic distributions, and signal suppression between methods

**Configuration**: Set `use_boost_corrected_matching: True` in `coffea4bees/skimmer/metadata/mixeddata_Run3.yml` to enable boost-corrected matching (default: `False`)

### Implementation Details

#### Files to Modify

| File | Changes |
|------|---------|
| `coffea4bees/skimmer/processor/make_mixed_data.py` | Add config flag to `HemiMixer.__init__()`, pass to replacement function |
| `coffea4bees/hemisphere_mixing/mixing_helpers.py` | Add boost function, modify `replace_hemis_load_kdTrees()` |
| `coffea4bees/skimmer/metadata/mixeddata_Run3.yml` | Expose `use_boost_corrected_matching` option |

#### Implementation Summary

1. **Add configuration flag** (`make_mixed_data.py`)
   - New parameter: `use_boost_corrected_matching: bool = False` in `HemiMixer.__init__()`
   - Conditionally set `hemi_summary_vars`:
     - `False` (default): `["sumPt_T_minor", "sumPt_T", "combinedMass", "pz"]` (current behavior)
     - `True`: `["sumPt_T_minor", "sumPt_T", "combinedMass"]` (3D matching)

2. **Add Lorentz boost function** (`mixing_helpers.py`)
   - New function: `boost_jets_to_target_pz(jets, pz_matched, pz_target)`
   - Computes rapidity difference between matched and target hemispheres
   - Applies z-boost to each jet's 4-momentum (E, px, py, pz) → (E', px, py, pz')
   - Returns boosted jets with corrected pz

3. **Modify replacement logic** (`mixing_helpers.py`)
   - Update `replace_hemis_load_kdTrees()` to accept `use_boost_corrected_matching` flag
   - After constructing `new_Jets`, if boost mode enabled:
     - Get target hemisphere pz from `subset_hemis`
     - Get matched hemisphere pz from library
     - Apply `boost_jets_to_target_pz()` to `new_Jets`
   - Existing phi rotation for thrust alignment remains unchanged

4. **Optional: Store diagnostic info**
   - Add `boost_delta_rapidity` to output for validation/debugging

#### What Remains Unchanged

- Hemisphere library creation (Step 1) - no changes needed
- Thrust axis calculation
- Phi rotation to align thrust axes
- Jet multiplicity binning for k-d tree lookup
- All event selection and weighting logic
