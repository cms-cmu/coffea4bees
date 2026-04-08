# Snakefile_classifier_inputs_Run3MvD

Generates Run3 histograms (for JCM fitting) and classifier input friend trees (for MvD training), for all Run3 datasets. Replaces the shell scripts:
- `coffea4bees/scripts/analysis-runAll-Run3.sh`
- `coffea4bees/scripts/classifier-inputs-Run3.sh`
- `coffea4bees/scripts/classifier-inputs-mixeddata-all-Run3.sh`

## What it does

Two independent branches, each submitting one condor job per dataset per year (**20 jobs each**, 5 datasets × 4 years):

| Branch | Config | Output |
|--------|--------|--------|
| Histograms | `HH4b_run_fastTopReco_Run3.yml` | `histAll_Run3MvD.coffea` (merged) |
| Classifier inputs | `HH4b_classifier_inputs_Run3.yml` | `classifier_inputs_Run3MvD.json` (merged) |

**Datasets:** TTToSemiLeptonic, TTToHadronic, TTTo2L2Nu, data, mixeddata_all
**Years:** 2022_EE, 2022_preEE, 2023_BPix, 2023_preBPix
**Friends:** `friends_HH4b.yml` for all datasets

## Running

From the barista root:

```bash
# Dry run (validate rule graph)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_classifier_inputs_Run3MvD.smk --cores 4 --dry-run --printshellcmds

# Run everything (histograms + classifier inputs)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_classifier_inputs_Run3MvD.smk --cores 4

# Run only histograms
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_classifier_inputs_Run3MvD.smk \
    --until merge_histograms --cores 4

# Run only classifier inputs
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_classifier_inputs_Run3MvD.smk \
    --until merge_json_classifier_inputs --cores 4
```

## Configuration

All settings can be overridden with `--config key=value`:

| Key | Default | Description |
|-----|---------|-------------|
| `output_path` | `output/classifier_inputs_Run3MvD/` | Base output directory |
| `label` | `''` | Tag appended to output path and final outputs (see variants below) |
| `dataset_location` | `coffea4bees/metadata/datasets_HH4b_Run3/` | Run3 datasets metadata directory |
| `datasets` | `['TTToSemiLeptonic','TTToHadronic','TTTo2L2Nu','data','mixeddata_all']` | All datasets to process |
| `years` | `['2022_EE','2022_preEE','2023_BPix','2023_preBPix']` | Years to process |
| `histogram_config` | `coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml` | Config for histogram step |
| `classifier_config` | `coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3.yml` | Config for classifier input step |
| `analysis_container` | barista:latest on cvmfs | Container for merge steps |

## Running Config Variants

Use `label` to tag a variant run so its outputs coexist with the nominal:

```
output/classifier_inputs_Run3MvD/              ← nominal (label='')
output/classifier_inputs_Run3MvD_quadjet_run2/ ← variant (label='_quadjet_run2')
```

### Example: quadjet selection mode run2

Pre-made variant config files:
- `coffea4bees/analysis/metadata/candidates_selection_thresholds_quadjet_run2.yml` — thresholds with `quadjet_selection: mode: run2`
- `coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3_quadjet_run2.yml` — classifier input config
- `coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml` — histogram config

```bash
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_classifier_inputs_Run3MvD.smk \
    --config \
        label="_quadjet_run2" \
        classifier_config="coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3_quadjet_run2.yml" \
        histogram_config="coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml" \
    --cores 4
```

### Example: single year/dataset for testing

```bash
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_classifier_inputs_Run3MvD.smk \
    --config years='["2022_EE"]' datasets='["TTToSemiLeptonic"]' label='_test' \
    --cores 4 --dry-run
```

## Outputs

```
output/classifier_inputs_Run3MvD{label}/
├── histograms/
│   ├── hist_TTToSemiLeptonic_2022_EE.coffea
│   ├── hist_data_2022_EE.coffea
│   ├── hist_mixeddata_all_2022_EE.coffea
│   └── ...  (5 datasets × 4 years = 20 files)
├── classifier_inputs/
│   ├── classifier_inputs_TTToSemiLeptonic_2022_EE.coffea  # + .json (written to EOS)
│   ├── classifier_inputs_data_2022_EE.coffea
│   └── ...  (20 files)
├── histAll_Run3MvD{label}.coffea          ← merged histograms for JCM fitting
├── classifier_inputs_Run3MvD{label}.json  ← merged classifier input metadata
└── logs/
```
