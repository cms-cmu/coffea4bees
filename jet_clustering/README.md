# Jet Declustering (Synthetic Datasets)

## Physics Motivation

Jet declustering is a data-driven technique to create synthetic background
samples for the HH→4b analysis. It is complementary to
[hemisphere mixing](../hemisphere_mixing/README.md): both build a
signal-depleted 4b background model from real data, but they break the
signal correlations at a different level — hemisphere mixing swaps whole
hemispheres, while declustering re-generates the *internal splitting
kinematics* of the jets.

### Why It Works

The dominant 4b background is QCD multi-jet production. Its final-state jets are
built up by a sequence of approximately **independent 1→2 parton splittings**
(the QCD shower). The kinematics of an individual splitting — its opening angle,
momentum sharing, and sub-jet masses — follow largely **universal** distributions
that depend on the splitting's own scale (jet pT, flavor) but *not* on the rest
of the event.

In contrast, **signal events (HH→4b)** carry **correlated structure**: two b-jets
come from each Higgs and reconstruct a characteristic di-jet mass peak. That
correlation lives in the *relationship between* the jets, not in any single
splitting.

If we take a real 4b event, decompose it into its splitting tree, and then
**re-generate each splitting by sampling fresh kinematics from an inclusive
library of splittings**, we retain the per-splitting QCD behavior while
**washing out the inter-jet correlations** — exactly the correlations the HH
signal relies on. The result is a synthetic, signal-depleted background model.

### The Declustering Procedure

1. **Cluster**: For each 4-tag data event, run an exclusive jet-clustering
   algorithm on the four candidate jets to reconstruct the pairwise **splitting
   tree** (which jets combine, and in what order). For every 1→2 node, compute a
   set of kinematic **splitting variables**.

2. **Build PDFs**: Histogram the splitting variables — binned in the combined
   jet pT and grouped by **splitting type** (see below) — and distill them into
   template probability-density files (`clustering_pdfs_vs_pT_{era}.yml`).

3. **Decluster**: For each event, walk its splitting tree from the top down. At
   each node, **re-generate** the two children by sampling new splitting
   variables from the PDFs (seeded by `declustering_rand_seed`), then boost/rotate
   them back into the lab frame. The output is a **synthetic event** with the same
   jet multiplicity and flavor structure but statistically independent kinematics.

### Key Properties

- **Preserves**: jet multiplicity and flavor composition, single-splitting
  kinematics (opening angle, momentum sharing, sub-jet masses), overall event
  pT/η
- **Suppresses**: HH→4b signal (the di-jet mass correlations are broken by
  resampling each splitting independently)
- **Use case**: data-driven background estimation, closure tests, systematic
  studies

### Splitting Variables

Each 1→2 splitting node combines two objects `part_A` and `part_B` into a
combined jet. The declustering variables (computed in `declustering.py`,
`compute_decluster_variables`; sampled back in `decluster_combined_jets`) are
defined in the frame where the combined jet is boosted to `pz = 0`:

| Variable | Description |
|----------|-------------|
| `zA` | Longitudinal momentum fraction carried by `part_A` (momentum sharing of the splitting) |
| `thetaA` | Opening angle of `part_A` relative to the combined jet direction |
| `mA`, `mB` | Invariant masses of the two children (`rhoA = mA/ptA`, `rhoB = mB/ptB`) |
| `decay_phi` | Azimuthal orientation of the decay plane about the combined-jet axis |
| `abs_eta` | \|η\| of the combined jet |

The combined jet pT is used to **bin** the PDFs, so the sampled kinematics
depend on the scale of the splitting.

### Splitting Types

Splittings are labeled by the **flavor and multiplicity** of their two children,
where `b` = b-tagged jet and `j` = light jet. The recursive parenthesized
notation (e.g. `(bj)(bj)`, `((jj)b)b`) encodes the full sub-tree; a compact
name is derived by `get_splitting_name()`:

- Low-multiplicity nodes keep the explicit flavor count, e.g. `1b0j/1b0j`
  (a b–b splitting), `1b1j/...`
- Higher-multiplicity nodes collapse to `nA/nB` counts, e.g. `3/2`, `3/3`,
  `4/1`, and `X/X` / `X/2` for the largest (`X` = "≥5").

Separate PDFs are built per splitting type so that, e.g., a `b/b` splitting is
resampled from `b/b` splittings only.

### Clustering / Recombination Metric

The clustering (`clustering.py`) uses a kT-like exclusive recombination
distance:

```
dij  = min(ptA², ptB²) · ΔR(A,B)²      # optionally / R²
diB  = ptA²                             # beam distance (only if R > 0)
```

With `R = 0` this is **exclusive** clustering: the pair with the smallest `dij`
is recombined at every step until the requested number of jets remains,
yielding a unique binary splitting tree per event. A numba-jitted core
(`get_min_indicies_numba_core`) does the heavy lifting.

### Notes on the Declustering Step

- **b-jet handling**: b-tagged objects below `b_pt_threshold` (default 40 GeV)
  are treated specially so the synthetic event keeps a consistent b-jet content.
- **ΔR separation**: after declustering, output jets are required to be
  separated by `dr_threshold` (default 0.4). Events failing the check are
  re-sampled with a perturbed seed, up to a retry cap, then kept.
- **Seeds**: `declustering_rand_seed` seeds all the PDF sampling, so different
  seeds produce **independent synthetic replicas** of the same input data.

## Implementation

### Step 1: Cluster (learn the splittings)

**Processor**: `coffea4bees/analysis/processors/processor_cluster_4b.py`
(extends the standard `HH4bBaseProcessor`)

**Example script**: `coffea4bees/scripts/synthetic-dataset-cluster-Run3-all.sh`

**Config**: `coffea4bees/analysis/metadata/cluster_4b_Run3.yml`

The processor:
1. Applies the standard event selection and selects **4-tag** data events
2. Optionally subtracts the TTbar contribution using FvT weights
   (`subtract_ttbar_with_weights`)
3. Clusters the candidate jets into the splitting tree (`cluster_bs`)
4. Fills `ClusterHists` of the splitting variables, binned in pT and grouped by
   splitting type

Output: `synthetic_datasets_Run3_nott.coffea` (histograms of the splittings)

```bash
./run_container bash coffea4bees/scripts/synthetic-dataset-cluster-Run3-all.sh --output-base output/
```

### Step 1b: Make the clustering PDFs

**Tool**: `coffea4bees/jet_clustering/make_jet_splitting_PDFs.py`

Distills the splitting histograms from Step 1 into per-era PDF template YAMLs
(`clustering_pdfs_vs_pT_{era}.yml`), one probability table per splitting
variable and splitting type.

```bash
python jet_clustering/make_jet_splitting_PDFs.py \
    output/.../synthetic_datasets_Run3_nott.coffea \
    --out jet_clustering/jet-splitting-PDFs-00-11-01/ \
    --years Run3
```

### Step 2: Decluster (create the synthetic dataset)

**Processor**: `coffea4bees/skimmer/processor/make_declustered_data_4b.py`
(class `DeClusterer`)

**Example script**: `coffea4bees/scripts/synthetic-dataset-make-dataset-Run3-all.sh`

**Config**: `coffea4bees/skimmer/metadata/declustering_Run3.yml`

The processor:
1. Loads the clustering PDFs (`clustering_pdfs_file`, with `XXX` → era)
2. Applies event selection and re-clusters each 4b event into its splitting tree
3. Walks the tree top-down, re-generating each splitting from the PDFs via
   `make_synthetic_event()` seeded by `declustering_rand_seed`
4. Enforces the ΔR separation of the output jets (with re-sampling on failure)
5. Writes a synthetic PicoAOD, `picoAOD_seed{seed}.root`, per input file

Output: declustered PicoAODs on EOS
(`{base_path}/{dataset}/picoAOD_seed{seed}...root`) plus a picoAOD registry YAML.

```bash
./run_container bash coffea4bees/scripts/synthetic-dataset-make-dataset-Run3-all.sh --output-base output/
```

### Step 3: Analyze

**Processor**: `coffea4bees/analysis/processors/processor_HH4b.py`

**Example script**: `coffea4bees/scripts/synthetic-dataset-analyze-Run3-all.sh`

**Config**: `coffea4bees/analysis/metadata/HH4b_mixed_data_new.yml`

Runs the standard HH4b analysis over the installed synthetic dataset (all seeds
are read via the dataset's `nSamples` / `files_template`), producing
`histDeClusteredDataRun3_noTT.coffea`.

```bash
./run_container bash coffea4bees/scripts/synthetic-dataset-analyze-Run3-all.sh output/
```

## Snakemake Workflow (end-to-end)

`coffea4bees/workflows/Snakefile_Run3_make_synthetic.smk` wires all four stages
into a single DAG (one HTCondor job per year, PDFs joining cluster → decluster):

```
cluster (×year) → merge → make_pdfs
    → [patch config → decluster (×seed×year) → merge registry] (×seed)
    → install dataset → analyze (×year) → merge
```

```bash
snakemake --profile software/snakemake/profiles/lpc \
    --snakefile coffea4bees/workflows/Snakefile_Run3_make_synthetic.smk \
    --cores 4
# multiple seeds:
#   ... --config n_seeds=5
```

Key points:

- **Seeds** — `--config n_seeds=N` produces the contiguous replicas
  `0..N-1`. The analyze step reads them all via `nSamples`, so the seeds
  **must** stay contiguous from 0 (runner.py expands the `seedXXX` template over
  `range(nSamples)`). `n_seeds` becomes `nSamples` in the installed dataset.
- **2023 pt threshold (25 GeV)** — all three processors read
  `coffea4bees/analysis/metadata/object_selection_thresholds.yml`, whose
  `era_overrides: "2023"` lowers the selected-jet `pt_min` to 25 GeV for the
  2023 eras. It is declared as an `input:` on every processing stage, so the DAG
  fails fast if it is ever missing and the pt25 selection is guaranteed to be
  applied.
- **Cluster + PDFs are seed-independent** and live in a shared (tag-keyed) dir
  that every `n_seeds` run reuses. The generated PDFs are written to a
  workflow-local directory so the committed `jet-splitting-PDFs-*/` is left
  untouched.
- **Install** — `workflows/scripts/install_synthetic_dataset.py` converts the
  per-seed registries into a `files_template` dataset YAML installed under
  `coffea4bees/metadata/datasets_HH4b_Run3/` (commit it to version the dataset).

---

## Reference: manual commands & debugging

### Unit tests

All:

```bash
python -m unittest jet_clustering.tests.test_clustering.clusteringTestCase
```

One test:

```bash
python -m unittest jet_clustering.tests.test_clustering.clusteringTestCase.test_synthetic_datasets_bbjjets
```

### Compare synthetic and nominal datasets

```bash
python jet_clustering/compare_datasets.py \
    --combine_input_files analysis/hists/histData.coffea analysis/hists/test_synthetic_data_seedXXX.coffea \
    --out analysis/plots_synthetic_datasets_all_00-07-02-wPSData \
    -m analysis/metadata/plotsSyntheticVsData2.yml
```

### Debugging splittings

Compare splittings:

```bash
python jet_clustering/splitting_comparison_plots.py \
    analysis/hists/test_synthetic_datasets_4j_and_5j.coffea \
    --out jet_clustering/jet-splitting-PDFs-00-02-00/comparison
```

Check the reclustered splittings (needs `processor_cluster_4b.py` run with the
`cluster_and_decluster.yml` configuration):

```bash
python jet_clustering/check_reclusted_splittings.py \
    analysis/hists/test_synthetic_datasets_4j_and_5j.coffea \
    --out jet_clustering/jet-splitting-PDFs-00-02-00/reclustering
```

### Make pdflatex slides

```bash
awk -f makeslides.awk nominal.config plots_jet_clustering.config > testTexSlides.tex
pdflatex testTexSlide.tex
```

where the awk files and config are in the repo `git@github.com:johnalison/lab.git`.
