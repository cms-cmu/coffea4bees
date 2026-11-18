# CoReNet workflow for boosted HH → 4b studies

An optional Snakemake workflow that runs the analysis chain with **CoReNet** as the classifier
for both the FvT (multijet reweighting) and the SvB (signal vs background):

```
3b data × JCM ──► FvT (CoReNet) ──► SvB (CoReNet) ──► histograms ──► fit on the SvB histogram
                  train, evaluate    train, evaluate   runner.py      datacards, limits
```

It is opt-in. Nothing includes it, no default config points at it, and it is not part of CI or
of `bin/roast`. It runs only when you name its Snakefile.

All commands are run from the barista root directory.

## Status: read this first

| Part | State |
| --- | --- |
| Workflow DAG (all four steps) | Resolves in a dry run with the placeholder config (49 jobs). |
| FvT and SvB commands, train and evaluate | Arguments parse, the trainers and the network are built. **No training or evaluation has been run**, for either classifier, with this configuration. |
| CoReNet SvB | The model config existed but had not been used in production. It parses and builds like the FvT. Its first real training is yours: treat it as untested. |
| Inputs | **Placeholders.** No boosted classifier-input friend trees, JCM fit or dataset lists exist in the repository. |
| Histogramming for a boosted selection | **Missing.** The `analysis` step points at the resolved processor as a placeholder (see [What is not there yet](#what-is-not-there-yet)). |

## Files

| File | Purpose |
| --- | --- |
| `coffea4bees/workflows/Snakefile_CoReNet.smk` | The workflow. |
| `coffea4bees/workflows/config/corenet_boosted.yml` | Its config: inputs, network, classifier options, analysis and stats settings. |
| `coffea4bees/classifier/config/corenet/HH4b_boosted.yml` | CoReNet event description: objects, decay hypothesis, network size. |
| `coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/common.yml` | Classifier settings: eras, input branches, event scalars. |
| `coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/{FvT,SvB}/{train,evaluate}.yml` | The four classifier jobs, as templates. Used by the workflow and by hand. |
| `coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/paths.yml` | Inputs and outputs for the jobs run by hand. |

## The decay chain file

`coffea4bees/classifier/config/corenet/HH4b_boosted.yml` tells CoReNet what the event is made of
and how it decays. It is a sample to start from; `corenet.network` in the config (or `network` in
`paths.yml`) selects which file is used.

| Block | Sample value | Meaning |
| --- | --- | --- |
| `objects.subjets`, `objects.jets` | `{features: [pt, eta, phi, mass], max: 4}` / `max: 8` | Object collections and their per-object features. |
| `objects.b` | `subjets pt 4` | Role `b`: four subjets, ordered by pt. Read from the `CanJet_*` branches: slots 0, 1 are the subjets of the first AK8 jet, slots 2, 3 of the second. |
| `objects.other` | `jets pt 8` | Role `other`: up to eight further jets, read from `NotCanJet_*`. |
| `hypotheses.HH` | `H1: [b, b]`, `H2: [b, b]`, `context: other` | The decay chain: HH → two Higgs candidates, each → two `b` objects. `other` is attention context. |
| `outputs`, `combine` | `main: HH`, `[HH]` | Which hypothesis feeds the classifier output. |
| `training` | `features: 8`, `attention_heads: 2`, ... | Network size and switches. |

The network scores all three pairings of the four `b` objects (views `q_1234`, `q_1324`,
`q_1423`); `q_1234` is the pairing as written in the friend tree.

## What you have to provide

1. **Dataset metadata**: a directory of YAMLs listing the picoAOD files, in the layout of
   `coffea4bees/metadata/datasets/` (`data.<year>.picoAOD.<era>.files`,
   `<MC sample>.<year>.picoAOD.files`).
2. **Classifier-input friend trees** and their manifest (`<file>.json@@<key>`). The loader is the
   one of the resolved analysis, so the friend tree must follow its branch names, whatever the
   objects are:

   | Branch | Content |
   | --- | --- |
   | `CanJet_pt`, `_eta`, `_phi`, `_mass` | The candidate objects, exactly `n_CanJet` per event. Default here: the four subjets of the two Higgs-candidate AK8 jets, slots 0, 1 from the first jet and 2, 3 from the second. |
   | `NotCanJet_pt`, `_eta`, `_phi`, `_mass`, `_isSelJet` | Other objects (context), any number; padded or truncated to `n_NotCanJet`. |
   | `fourTag`, `threeTag` | The two tag regions the FvT maps between (your "pass" and "fail"). |
   | `SR`, `SB`, `ZZSR`, `ZHSR`, `HHSR` | Mass regions. The FvT trains on `SB` and `SR` (never on four-tag `SR` data), the SvB on `SR`. |
   | `passHLT`, `weight` | Trigger decision and event weight. |
   | event scalars | One branch per name in `feature_ancillary` (`year` is added by the loader). |

   `coffea4bees/analysis/helpers/dump_friendtrees.py::dump_input_friend` writes this format; its
   `CanJet` / `NotCanJet` arguments name the event collections to dump.
3. **JCM weights** (`<file>.yml@@JCM_weights`): the 3b → 4b normalisation applied to three-tag
   data. Optional: set `inputs.jcm_weights: ""` to train without.
4. **An output location** for models and friend trees, visible to the GPU jobs.

## Running the workflow

Put the four inputs in the `inputs:` block of the config (or copy the config and edit the copy).

```bash
# dry run: lists the jobs, submits nothing
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml -n

# add -p to also print the commands of each job
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml -np

# run (inside tmux: the scheduler stays alive until the last job ends)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml
```

Run part of the chain by target or by step:

```bash
# one target: corenet_fvt, corenet_svb, corenet_analysis, corenet_stats, corenet_all (default)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml corenet_svb

# classifiers only, without loading the analysis and stats rules
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml --config steps=fvt,svb
```

A real run stops at its first rule while any input still contains `PLACEHOLDER`.

The dry run with the placeholder config gives this job list:

| Step | Rules (count) |
| --- | --- |
| setup | `corenet_check_inputs` (1) |
| fvt, svb | `corenet_stage_fvt` (1), `corenet_stage_svb` (1), `train` (2), `evaluate` (2), `yields` (2), `analyze` (2), `classifier_gallery` (2) |
| analysis | `corenet_analysis_config` (1), `analysis_data` (15), `analysis_dataset` (4), `merging_files` (1), `make_plots` (1), `check_cutflow` (1) |
| stats | `convert_hist_to_json` (1), `make_combine_inputs` (1), `workspace` (1), `limits` (1), `significance` (1), `fit_diagnostics_sb` (1), `postfit` (1), `likelihood_scan_snapshot` (1), `likelihood_scan_chunk` (2), `likelihood_scan` (1), `stat_summary` (1) |
| | `corenet_all` (1); 49 in total |

Order: the SvB config is staged only once the FvT friend tree exists, and the analysis config is
written only once both friend trees exist, so each step waits for the one before it.

### Where things go

| What | Where |
| --- | --- |
| Models, `result.json` | `<inputs.output_base>/classifier/<fvt.label or svb.label>/` |
| Friend trees | `<inputs.output_base>/friend/<label>/` (`result.json@@analysis.0.merged`) |
| Staged configs, logs, flag files, training plots | `<output_path>classifiers/<label>/` |
| Histograms, plots, cutflow | `<output_path>PhaseF_1_analysis/` |
| Datacards and Combine results | `<output_path>PhaseF_2_stats/` |

The exact classifier configs a run used are `<output_path>classifiers/<label>/wfs/train.yml`,
`wfs/evaluate.yml` and `common.yml`.

### Cluster notes

- `./run_container snakemake` picks the Snakemake profile of the host. On falcon the `train` and
  `evaluate` rules become SLURM GPU jobs (default `mps:25`, QOS `light`, 24 GB, 8 h; change with
  `gpu_gres`, `gpu_qos`, `gpu_mem`, `train_runtime`, `eval_runtime` in the config).
- On lxplus the GPU profile is chosen from the Snakefile name, and this file is not in that
  list. Pass `--profile software/snakemake/profiles/lxplus_gpu` yourself.
- On Bridges-2 the profile's own resources for `train` and `evaluate` win over the config.
- The analysis jobs call `./run_container python runner.py ...`; add `--condor` or `--slurm`
  through `additional_parameters` to spread each one over batch workers.

## Config reference

`coffea4bees/workflows/config/corenet_boosted.yml`:

| Key | Meaning |
| --- | --- |
| `label` | Name of the run. Used in file names and as `{roast_id}`. Change it to keep two variants apart. |
| `output_path` | Local directory of this run. |
| `steps` | Subset of `[fvt, svb, analysis, stats]`. A classifier step that is left out is replaced by `inputs.fvt_friend` / `inputs.svb_friend`. |
| `inputs.dataset_metadata` | Dataset metadata directory for the classifier jobs (trailing `/`). |
| `inputs.classifier_inputs` | Friend-tree manifest, `<file>.json@@<key>`. |
| `inputs.jcm_weights` | JCM weights, `<file>.yml@@JCM_weights`; `""` removes them. |
| `inputs.output_base` | Where models and friend trees are written. |
| `inputs.fvt_friend`, `inputs.svb_friend` | Existing friend directories, read only when the step is not run. |
| `corenet.network` | Event description file. |
| `corenet.inputs` | Role → batch key (`CanJet`, `NotCanJet`), both classifiers. |
| `corenet.architecture` | `n_features`, `attention`, `use_attention_gate`, `use_kv_proj`, both classifiers. |
| `classifier_setting` | Replaces the `setting:` list of `common.yml` (eras, input branches, event scalars). |
| `fvt.label`, `svb.label` | Names of the model and friend directories. |
| `fvt.wfs_base`, `svb.wfs_base` | Directory holding `train.yml` and `evaluate.yml` (and `../common.yml`). |
| `fvt.model_options`, `svb.model_options` | Overrides of the model options of `train.yml`, by flag (below). |
| `gpu_gres`, `gpu_qos`, `gpu_mem`, `train_runtime`, `eval_runtime`, `retries` | Resources of the GPU jobs. `retries` is 0 here on purpose (see rough edges). |
| `year_eras`, `dataset` | Years, eras and datasets of the histogramming step. |
| `additional_parameters` | Extra `runner.py` arguments. |
| `analysis_config` | Processor, its metadata files and options. The two friend trees and `JCM_file` are filled in from this run unless you set them. |
| `analysis_friend_names` | Names under which the processor reads the FvT and SvB friends. |
| `plot_config`, `known_counts` | Plot list; reference cutflow (none). |
| `channels`, `combine_flags`, `make_combine_inputs`, `metadata_template` | Fit: histogram, signal, stat-only, blinding. Same meaning as in `nominal_run2.yml`. |

`{roast_id}` (the `label`) and `{eos_prod}` are filled in anywhere in the config. `{eos_prod}`
comes from `--config eos_prod=...`, from `~/.config/coffea4bees/profile.yml`, or from your account
name.

### `model_options` values

Each key is a command-line flag of the model module in `train.yml`.

| Value | Effect |
| --- | --- |
| scalar, e.g. `--kfolds: 5` | Replaces the flag's value. |
| list, e.g. `--training: [FixedStep, {epoch: 30}]` | Several arguments; a mapping is passed as embedded options. |
| `true` | Adds a bare flag. |
| `null` | Removes the flag, e.g. `--finetuning: null`. |

### Optional config blocks

These are not in the checked-in config; add them when needed.

```yaml
corenet:
  inputs: {b: CanJet, other: NotCanJet}   # role -> batch key; the event description's first role first
  architecture: {n_features: 16}          # also: attention, use_attention_gate, use_kv_proj

classifier_setting:                       # replaces the whole `setting:` list of common.yml
  - module: cms.CollisionData
    option:
      - eras:
          UL18: [A, B, C, D]
  - module: HCR.InputBranch
    option:
      - feature_CanJet: [pt, eta, phi, mass]
        n_CanJet: 4
        feature_NotCanJet: [pt, eta, phi, mass, isSelJet]
        n_NotCanJet: 8
        feature_ancillary: [nOtherJets]

gpu_gres: "mps:25"                        # defaults shown
gpu_qos: "light"
gpu_mem: 24000
train_runtime: 480                        # minutes
eval_runtime: 480
```

## How to change things

### Inputs

Edit `inputs:` in the config, or override one on the command line by supplying a second config
file with only that block (Snakemake merges `--configfile a.yml b.yml`). The top of
`Snakefile_CoReNet.smk` has the same values as named variables (`DATASET_METADATA`,
`CLASSIFIER_INPUTS`, `JCM_WEIGHTS`, `OUTPUT_BASE`, `NETWORK`).

Paths may not contain spaces, commas, quotes or braces.

### Network size

Three ways, from broadest to narrowest:

- `training.features` in `corenet/HH4b_boosted.yml`: the embedding width of every layer. It must
  be even and divisible by `attention_heads`.
- `corenet.architecture: {n_features: 16}` in the config: the same, for this run, both classifiers.
- `svb.model_options: {"--architecture": [{n_features: 16}]}`: one classifier only.

For scale: 8 features is about 1.5k parameters, 16 about 5k.

`attention: false` under `architecture` drops the context objects (`other`).

### Network inputs

Three places must agree; the training stops with a shape error when they do not.

| To change | Edit |
| --- | --- |
| Per-object features (e.g. add a b-tag score to the subjets) | `feature_CanJet` in `common.yml`, and the `features` list of the `subjets` collection in the event description, same order. The friend tree needs `CanJet_<name>`. |
| Features of the context objects | `feature_NotCanJet` (keep `isSelJet` last) and the `jets` collection. |
| Number of objects | `n_CanJet` / `n_NotCanJet`, and `max` plus the role size in the event description. |
| Event scalars | `feature_ancillary` in `common.yml`. Nothing to change in the event description. |
| Which objects are which | The friend tree: what you write to `CanJet_*` and `NotCanJet_*`. |

Rules for the event description (`corenet/HH4b_boosted.yml`):

- Do not add `event:` or `labels:`; they are filled in from `feature_ancillary` and from the
  dataset.
- The role listed first (`b`) must be the first key of the model's `--inputs` mapping. Its leading
  slot fixes the eta reflection.
- `outputs.main` is only a name; the classes come from the dataset.
- `other` is used as attention context only.

Traps:

- Remove `year` from `feature_ancillary` when the dataset has a single year: its variance is
  zero and the input becomes NaN.
- A scalar whose name contains `nSelJets` is fed as `log(nSelJets - (min_jets - 1))`, with
  `min_jets: 4` unless the event description sets it (minimum 2). A boosted event with fewer
  jets than that gives NaN. Use another branch name or set `min_jets`.
- The decay hypothesis is `HH: {H1: [b, b], H2: [b, b]}` over one role of four objects. The
  network scores all three pairings; view `q_1234` is the pairing you wrote. This is the only
  topology the existing FvT and SvB output definitions accept: they write the columns `q_1234`,
  `q_1324`, `q_1423` by name. A hypothesis with two roles builds and runs forward, but names its
  views `q_HH_<n>`, so the evaluation would need a new model config. "HH = AK8 + AK8" with the
  two AK8 jets as the only objects cannot be written at all: a hypothesis needs at least two
  composite children.

### Folds and seeds

In `model_options` of `fvt` or `svb`, or in the model block of the templates:

| Flag | Effect |
| --- | --- |
| `--kfolds: 3` | Number of folds. One model per fold; each event is evaluated by the model that did not train on it. |
| `--kfold-seed: [FvT, random]` | Base of the seed that assigns events to folds. |
| `--kfold-seed-offsets: "0"` | Which fold assignments to train. `"0-2"` trains three sets of `kfolds` models. |
| `--kfold-offsets: "0"` | Train only these folds. |
| `--seed: 2` | Base seed of the weight initialisation and the batch order. |

A training with the same seeds is not bit-reproducible (measured on the resolved FvT: losses
differ by up to a few 1e-3 after one epoch). Compare variants over several seeds.

### Training schedule

```yaml
fvt:
  model_options:
    --training: [FixedStep, {epoch: 30, bs_init: 1024, lr_init: 0.005}]
    --finetuning: null          # drop the fine-tuning stage
```

`FixedStep` (`src/classifier/config/scheduler/basic.py`): `epoch`, `bs_init` (batch size, doubled
at epochs 1, 3, 6, 10 = `bs_milestones`), `lr_init` (0.01), `lr_scale` (0.25, applied at every
epoch of `lr_milestones`, default 15 to 24). The milestones are absolute epoch numbers: move them
when you change `epoch`. `FinetuneStep` trains the output layer only, at `lr_init: 1e-4`.

`ClosureGatedStep` does not work with CoReNet (it needs the fit-quality benchmark).

### Signal and normalisation of the SvB

In `SvB/train.yml`, `--norm 3` is the total background weight and each signal process (ZZ, ZH,
ggF) is normalised to 1. To train on the HH signal only, add `--mc-processes ggF` to the
`HCR.SvB.Signal` block and set `--norm 1`. To train either classifier without the 3b → 4b
normalisation, delete its `--JCM-weight` line (the workflow does this when `inputs.jcm_weights`
is `""`).

### Anything else

Dataset options (`--mc-processes`, `--regions`, `--no-SR`, `--norm`, signal naming such as
`--ggf-signal-pattern`) are in the `dataset:` blocks of the templates. Copy the template
directory, edit the copy, and point `fvt.wfs_base` / `svb.wfs_base` at it. `common.yml` is read
from one level above `wfs_base`.

The templates go through Python's `str.format`: the only braces allowed in them, comments
included, are the template keys. Write a literal brace as `{{` or `}}`.

## Running the jobs directly, without the workflow

Copy `paths.yml`, fill in the inputs and outputs, and run the four jobs in this order. `T` is
only shorthand for this page.

```bash
T=coffea4bees/classifier/config/workflows/HH4b_boosted_corenet
cp $T/paths.yml my_paths.yml        # then edit my_paths.yml

# 1. FvT training
./run_container classifier env CLASSIFIER_CONFIG_PATHS=coffea4bees \
    python -m src.classifier.task.main template file:##my_paths.yml \
    $T/FvT/train.yml -from $T/common.yml -flag debug

# 2. FvT evaluation: writes the FvT friend tree
./run_container classifier env CLASSIFIER_CONFIG_PATHS=coffea4bees \
    python -m src.classifier.task.main template file:##my_paths.yml \
    $T/FvT/evaluate.yml -from $T/common.yml -flag debug

# 3. SvB training: reads the FvT friend tree
./run_container classifier env CLASSIFIER_CONFIG_PATHS=coffea4bees \
    python -m src.classifier.task.main template file:##my_paths.yml \
    $T/SvB/train.yml -from $T/common.yml -flag debug

# 4. SvB evaluation: writes the SvB friend tree
./run_container classifier env CLASSIFIER_CONFIG_PATHS=coffea4bees \
    python -m src.classifier.task.main template file:##my_paths.yml \
    $T/SvB/evaluate.yml -from $T/common.yml -flag debug
```

Template keys, all set in your copy of `paths.yml`:

| Key | Meaning |
| --- | --- |
| `metadata` | Dataset metadata directory (trailing `/`). |
| `inputs` | Classifier-input friend-tree manifest, `<file>.json@@<key>`. |
| `jcm` | JCM weights, `<file>.yml@@JCM_weights`. |
| `network` | CoReNet event description. |
| `fvt_model`, `svb_model` | Where the trained models and `result.json` are written. |
| `fvt_friend`, `svb_friend` | Where the friend trees are written; the SvB training reads `fvt_friend`. |

Outputs can be local directories or `root://` URLs.

What the pieces mean:

- `./run_container classifier <command>` runs the command in the classifier (GPU) image. On
  falcon it submits it as a SLURM GPU job and returns at once (log:
  `slurm_logs/classifier_batch_<jobid>.out`); wait for each job before starting the next. On
  lxplus it submits an HTCondor GPU job. Elsewhere it runs in place.
- `CLASSIFIER_CONFIG_PATHS=coffea4bees` makes the `HCR.*` and `CoReNet.*` modules visible.
- `template file:##my_paths.yml <job>.yml` fills the template keys of the job file from your
  paths file. Keep the paths in a file: `run_container` re-joins its arguments, so an inline
  mapping with spaces does not survive the quoting.
- `-from $T/common.yml` adds the shared settings.
- `-flag debug` prints the loaded yields per class and region.

To change a training without the workflow, edit the job file (folds, seeds, schedule, datasets)
or `common.yml` (input branches, eras). The options of a model module are listed by `help`
(`--network x` only satisfies the required argument; nothing is read or run):

```bash
./run_container classifier env CLASSIFIER_CONFIG_PATHS=coffea4bees \
    python -m src.classifier.task.main help -model CoReNet.FvT.baseline.Train --network x
```

The workflow runs exactly these four commands, with the staged copies of the job files and the
template keys taken from the config; `-np` prints them.

Loss curves and ROCs are plotted by the workflow's `analyze` rule
(`src/classifier/workflow/Snakefile`); the per-epoch numbers are in `<model>/result.json`.

## Known rough edges of CoReNet

Not fixed here. Source: the CoReNet / HCR comparison of 2026-11-16.

- **Resuming a training.** CoReNet checkpoints its Initialization stage, the pass that measures
  the input means and widths. A job that resumes after it can skip that pass and train with
  unnormalised inputs. The workflow therefore sets `retries: 0`. If a training dies, delete the
  `checkpoint_*` files (in `<output_path>classifiers/<label>/` for the workflow, in the model
  directory for a job run by hand) and start it again from scratch.
- **No fit-quality benchmark.** CoReNet trainings report losses and ROCs but not the sideband
  closure χ² that HCR FvT trainings report, so there is no automatic FvT quality gate in this
  workflow. Judge the FvT from the histograms.
- **Warm start.** `--pretrained-weights <directory>` does not find the saved models (it looks
  for `_offsetN_`, the files are named `__offset_N__`). A direct path to one `.pkl` is accepted,
  but then every fold starts from that same file.
- **Input plots.** `plot_inputs` and `plot_weights` of the classifier workflow rebuild an HCR
  network from the checkpoint and cannot read a CoReNet one. They are not scheduled here.
- **Reproducibility.** See [Folds and seeds](#folds-and-seeds).

## What is not there yet

- **A boosted histogramming processor.** `analysis/processors/processor_boosted_HH4b.py` counts
  events passing the resolved and the boosted selection; it fills no histograms, applies no FvT
  and reads no SvB. The `analysis` step is configured with `processor_HH4b.py`, the resolved
  processor, so that the chain is complete and each piece can be swapped. Until a boosted
  processor exists, use `steps: [fvt, svb]`.
- **A processor option that writes boosted classifier inputs.** The resolved processor writes
  them with `make_classifier_input`; nothing does it for AK8 jets and subjets.
- **A boosted JCM, FvT closure check, background systematics and fit metadata.** The `stats`
  step is stat-only and reuses the resolved `HH4b` channel definition.
- **Signal samples other than the ones the loader knows** (`ggF`, `ZZ`, `ZH`, with Run 2 dataset
  names by default; see `--ggf-signal-pattern` for Run 3 names).

## Using it with roast

Not wired in, and not needed for studies. `bin/roast new` accepts a step outside its phase table
as `--step <host>:<snakefile>[:targets]`, so a roast of this workflow would be

```bash
bin/roast new --config coffea4bees/workflows/config/corenet_boosted.yml --label <label> \
    --step falcon:coffea4bees/workflows/Snakefile_CoReNet.smk:corenet_all
```

That has not been tried. Two things would need attention first: the GPU steps and the analysis
step usually run on different hosts, so the single Snakefile would have to be split into one
step per host with `steps` set per step (`--config steps=fvt,svb` on the GPU host,
`steps=analysis,stats` plus `inputs.fvt_friend` / `inputs.svb_friend` on the other); and
`inputs.output_base` would follow the roast convention `{eos_prod}/{roast_id}`.
