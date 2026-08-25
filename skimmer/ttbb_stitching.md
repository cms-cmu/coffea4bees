# tt+bb stitching

Builds a stitched ttbar sample that takes its **tt+LF and tt+C** events from the
inclusive ttbar samples and its **tt+B** events from the dedicated 4FS TTbb
samples, while keeping the inclusive sample's total cross section and total sum
of genWeight.

Channels are stitched independently, for all four Run 2 eras
(`UL16_preVFP`, `UL16_postVFP`, `UL17`, `UL18`):

| channel      | inclusive          | dedicated TTbb      | stitched entry               |
|--------------|--------------------|---------------------|------------------------------|
| dilepton     | `TTTo2L2Nu`        | `TTbb_2L2Nu`        | `TTTo2L2Nu_stitched`         |
| semileptonic | `TTToSemiLeptonic` | `TTbb_SemiLeptonic` | `TTToSemiLeptonic_stitched`  |
| hadronic     | `TTToHadronic`     | `TTbb_Hadronic`     | `TTToHadronic_stitched`      |

## Normalisation

Events are categorised with `abs(genTtbarId) % 100`
(`analysis/helpers/ttbar_categories.py`): `51..55` is tt+B, `41..45` is tt+C,
everything else is tt+LF. Note the `abs()` before the modulo — for a negative
`genTtbarId`, plain `%` gives a value outside the category windows.

The tt+B events taken from TTbb have their `genWeight` scaled by

    k = sumw_ttB(inclusive) / sumw_ttB(TTbb)

so the tt+B piece carries exactly the genWeight sum of the tt+B piece removed
from the inclusive sample. The total is therefore preserved:

    sumw(stitched) = sumw_notB(inclusive) + k * sumw_ttB(TTbb)
                   = sumw_notB(inclusive) + sumw_ttB(inclusive)
                   = sumw(inclusive)

The stitched dataset entry inherits the inclusive `xs` and the inclusive
generator-level `sumw`, so

    w = genWeight * lumi * xs / genEventSumw

gives the same total yield as running on the unmodified inclusive sample, with
the tt+B shape supplied by TTbb.

**Both sums are taken at picoAOD (post-skim) level.** This is the level at which
the stitched files are built and at which the closure is measurable. It preserves
the inclusive tt+B yield *within the skim acceptance*; it does not preserve the
generator-level tt+B cross section, which would require a pass over nanoAOD.

> **Statistics caveat.** The TTbb samples have far fewer generated events than
> the inclusive ones (e.g. 10.4M vs 474M for semileptonic UL18). In this
> b-enriched skim they deliver *fewer* effective tt+B events than the inclusive
> sample — roughly 0.5x for hadronic and 0.4-0.6x for semileptonic, 0.7-1.1x for
> dilepton. Stitching is a modelling improvement here, not a statistical one.

## Running it

All commands are run from the barista root.

### Stage 1 - measure the scale factors

Reads only `genWeight` and `genTtbarId` from the existing picoAODs of all six
datasets and writes the per (channel, era) factors.

```bash
./run_container python coffea4bees/skimmer/tools/ttbb_stitch_factors.py \
    -m coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT.yml \
    -o coffea4bees/skimmer/metadata/ttbb_stitch_factors.json
```

### Stage 2 - write the stitched picoAODs

One skimmer pass over both sides. `StitchTTbb` keeps the requested genTtbarId
category and, on the TTbb side, overrides `genWeight` with `genWeight * k`.
Every other branch is copied through untouched.

```bash
./run_container python runner.py -s \
    -p coffea4bees/skimmer/processor/stitch_ttbb.py \
    -c coffea4bees/skimmer/metadata/stitch_ttbb.yml \
    -m coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT.yml \
    -y UL16_preVFP UL16_postVFP UL17 UL18 \
    -d TTTo2L2Nu TTToSemiLeptonic TTToHadronic \
       TTbb_2L2Nu TTbb_SemiLeptonic TTbb_Hadronic \
    --friends none --weights none \
    -o picoaod_datasets_ttbb_stitched.yml \
    -op coffea4bees/skimmer/metadata/ \
    --dask --condor
```

Set `base_path` in `skimmer/metadata/stitch_ttbb.yml` to the output location.
Add `-t` for a two-file, one-chunk smoke test first.

`campaign` in that config is pinned rather than auto-generated, so re-running the
same command after a failure skips the chunks that were already written instead
of redoing them. Change it only to force a full reprocessing.

### Stage 3 - build the dataset entry and check the closure

Merges each channel's two file lists into one dataset entry and **refuses to
write** unless `sumw(stitched) == sumw(inclusive)`.

```bash
./run_container python coffea4bees/skimmer/tools/make_stitched_dataset.py \
    -i coffea4bees/skimmer/metadata/picoaod_datasets_ttbb_stitched.yml \
    -f coffea4bees/skimmer/metadata/ttbb_stitch_factors.json \
    -m coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT.yml \
    -o coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT_stitched.yml
```

Because the ttHbb config points `dataset_location` at that directory, dropping
`TT_stitched.yml` there makes the entries available to the analysis.

### Verify against the produced files

Independent of the counters above: re-reads every stitched file, checks each is
pure in its genTtbarId category, and re-derives the closure.

```bash
./run_container python coffea4bees/skimmer/tools/ttbb_stitch_verify.py \
    -s coffea4bees/metadata/datasets/archive/Run2_2024_v2/TT_stitched.yml
```

### Unit tests

```bash
./run_container python -m pytest coffea4bees/skimmer/tests/ttbb_stitch_test.py -v
```

Checks the categorization against the reference definition, that the three
categories partition the sample, that the genWeight sum is conserved, that
float32 storage preserves it, and that a wrong scale factor is actually caught.

## After stitching

* **Friend trees must be regenerated.** They are keyed by picoAOD file UUID and
  entry range (`src/friendtrees/dump_friend.py`), so the existing `trigWeight`
  and `FvT` friends do not apply to the new files.
* **Group the stitched entries as one process** in the plot and stats configs,
  the same way the inclusive channels are grouped today, e.g.
  `process: [TTToHadronic_stitched, TTToSemiLeptonic_stitched, TTTo2L2Nu_stitched]`.
* Do not also include the original `TT*` or `TTbb_*` entries, or the tt+B
  component would be double counted.
