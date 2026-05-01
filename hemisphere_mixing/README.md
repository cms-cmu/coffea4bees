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

### Difference

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

### Benefits

- **Increased effective library statistics**: Hemispheres that were previously "far away" due to pz differences become viable matches
- **Better transverse matching**: By not penalizing pz differences, the algorithm can focus on finding the best transverse-structure match
- **Physically motivated**: The correction respects Lorentz invariance - we're matching on boost-invariant properties and then applying the appropriate boost

### Usage

The boost-corrected matching is implemented as an **optional mode**, preserving the existing 4D matching as the default. This allows:

- **A/B comparison**: Run both methods on the same inputs to evaluate the impact on synthetic data quality
- **Backward compatibility**: Existing workflows continue to work unchanged
- **Validation**: Compare closure tests, kinematic distributions, and signal suppression between methods

**Configuration**: Set `use_boost_corrected_matching: True` in `coffea4bees/skimmer/metadata/mixeddata_Run3.yml` to enable boost-corrected matching (default: `False`)


## Top-K Neighbor Matching with Rank Selection (Optional Mode)

### Motivation

The default nearest-neighbor (k=1) matching has two limitations:

1. **Same-source-event collisions.** When the k-d tree picks the same library 4-tag event for both the positive and negative replacement hemispheres of a single 3-tag input, the mixed event degenerately reconstructs a real 4-tag event. The inter-hemisphere correlations the procedure is meant to break are restored — exactly what mixing is supposed to suppress.

2. **Limited synthetic dataset size.** Each 3-tag input contributes exactly one mixed event. To grow the synthetic dataset, we'd want each input event to contribute multiple statistically-independent pseudo-events.

Both reduce to the same primitive: **query the top-K nearest neighbors per hemisphere and choose a rank per hemisphere.** With that primitive:

- Same-event collisions can be resolved by walking down the rank list to the next non-colliding pair.
- The dataset can be expanded by running multiple times with `default_rank = 0, 1, 2, …` and recording the rank used in each output picoAOD.

### Difference

Two new knobs on top of the standard 4D / boost-corrected matching:

| Knob | Meaning |
|------|---------|
| `k_neighbors` (K) | Query the top-K neighbors instead of just the nearest. K=10 is plenty for collision retry; bump higher for dataset expansion. |
| `default_rank` (R) | Select the R-th nearest neighbor as the primary pick. R=0 reproduces nearest-neighbor; R>0 picks further neighbors (future dataset-expansion knob). |
| `collision_mode` | Policy when pos/neg hemispheres of an input collide on the library source event: `ignore` (keep), `drop` (discard the input event), or `retry` (rank-walk to a non-colliding pair). |

When `collision_mode: retry` and a pos/neg pair collide at rank R, the algorithm searches `(rp, rn) ∈ [R..K-1]²` for the smallest-distance non-colliding pair. If none exists within K (vanishingly rare in practice), the event is dropped.

The chosen rank is written to the output as `posHemiNew_match_rank` / `negHemiNew_match_rank` so downstream code can stratify by rank.

### Benefits

- **No bias from same-event collisions** — degenerate hemisphere pairs no longer leak through.
- **Future dataset expansion** — the same primitive serves both retry now and the planned multi-rank emission mode later.
- **A/B-friendly** — legacy `replace_hemis_load_kdTrees` is left untouched; the new path is selected via a single config flag, so the two modes can be compared on the same input.

### Usage

In `coffea4bees/skimmer/metadata/mixeddata_Run3.yml`:

```yaml
config:
  use_topk_matching: True   # opt in to top-K matching
  k_neighbors: 10           # K for the kd-tree query
  collision_mode: retry     # ignore | drop | retry
  default_rank: 0           # 0 = nearest; >0 picks further neighbors
```

Defaults: `use_topk_matching: False` (legacy nearest-neighbor; events whose pos/neg hemispheres collide on the library source event are dropped).

`use_topk_matching` is independent of `use_boost_corrected_matching` — both can be enabled together.

**Implementation**: `coffea4bees/hemisphere_mixing/mixing_helpers.py` → `replace_hemis_topk_kdTrees`. Internally three stages: per-bin top-K query → per-event rank resolution → per-bin build at chosen rank. Stage 1 fetches only summary fields (event/run/lumi/distance) for collision detection; Stage 3 fetches the full jet payload only at the chosen rank, keeping memory footprint flat in K.
