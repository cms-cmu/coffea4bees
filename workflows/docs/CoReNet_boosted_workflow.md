# CoReNet workflow for boosted HH → 4b studies

An optional Snakemake workflow that runs the HH → 4b chain with the **CoReNet** network as the
classifier for both FvT and SvB:

```
3b data × JCM ──► FvT (CoReNet) ──► SvB (CoReNet) ──► histograms ──► fit on the SvB histogram
                  train, evaluate    train, evaluate   runner.py      datacards, limits
```

Terms used on this page:

- **picoAOD**: the skimmed ntuple written from NanoAOD, read by the analysis and the classifiers.
- **Friend tree**: extra branches for the events of a picoAOD, in a separate file matched event by
  event. Classifier inputs and outputs are stored as friend trees.
- **FvT** (four-tag vs three-tag): a classifier trained on data to tell four-b-tag events from
  three-b-tag events. Its output reweights three-tag data into a model of the four-tag multijet
  background.
- **SvB** (signal vs background): a classifier that separates HH (and ZZ, ZH) from multijet and
  ttbar. Its output histogram is what the fit uses.
- **JCM** (jet combinatoric model): a per-event weight for three-tag data, one value per jet
  multiplicity (`nSelJets`), that normalises three-tag to four-tag before the FvT.
- **Template key**: a `{name}` placeholder in a classifier job file, filled in when the job runs.

The workflow is opt-in: no default config, CI job or `bin/roast` phase uses it. All commands are
run from the barista root directory.

## Status

| Part | State |
| --- | --- |
| Workflow | The dry run resolves: 49 jobs for the full chain, 14 with `--config steps=fvt,svb`. |
| Boosted classifier jobs | The four job files parse and the network builds (about 1.5k parameters). **Nothing has been trained or evaluated with this boosted configuration.** |
| CoReNet package | The network is no longer copied into barista. It comes from the CoReNet repository (github.com/cms-cmu/CoReNet), cloned next to the code: see [Setup](#setup). |
| CoReNet itself | CoReNet FvT and SvB have been trained and evaluated in production for the resolved Run 2 analysis (full chain through limits, 2026-11-17). Those runs used the network copy that was in barista at the time; the cloned package gives the same results in the small test described under Setup. |
| Inputs | **Placeholders.** There are no boosted classifier-input friend trees and no boosted JCM fit in the repository. |
| Histogramming step | Uses the resolved processor `processor_HH4b.py` as a placeholder. No existing boosted processor fills analysis histograms with FvT and SvB; see [Boosted code that exists, and what is missing](#boosted-code-that-exists-and-what-is-missing). Until one exists, run `steps: [fvt, svb]`. |

## Setup

Three repositories are needed in one place: barista (the framework), coffea4bees (the analysis)
and CoReNet (the network). CoReNet's code is not copied into barista; it is used as a Python
package named `corenet`.

| Piece | Where | You edit it? |
| --- | --- | --- |
| The network | CoReNet repository, https://github.com/cms-cmu/CoReNet | No |
| The barista model module (feeds batches to the network, ghost batches, fine-tuning, checkpoints, output columns) | `src/classifier/ml/models/CoReNet.py` in barista | No |
| The model names `CoReNet.FvT.baseline` and `CoReNet.SvB.ggF.baseline` (loss and classes per classifier) | `coffea4bees/classifier/config/model/CoReNet/` | No |
| The decay-chain file and the job files | see [Files to edit before running](#files-to-edit-before-running) | Yes |

### Step by step

```bash
# 1. barista and coffea4bees, both on the corenet branch (until it is merged)
git clone -b corenet https://gitlab.cern.ch/cms-cmu/barista.git
cd barista
git clone -b corenet https://gitlab.cern.ch/cms-cmu/coffea4bees.git

# 2. CoReNet, at the tested commit, and a link so that Python finds the package
git clone https://github.com/cms-cmu/CoReNet.git
git -C CoReNet checkout 6c260c3
ln -s CoReNet/src/corenet corenet

# 3. h5py, which the containers do not have, installed next to the code
./run_container classifier python -m pip install --no-deps --target pkgs h5py
ln -s pkgs/h5py pkgs/h5py.libs .

# 4. check: no error means CoReNet and h5py load (a RuntimeWarning from runpy is harmless)
./run_container classifier python -s -m corenet.models.network

# 5. grid proxy, if any input or output is on root://
./run_container voms-proxy-init -voms cms -rfc --valid 168:00 -out proxy/x509_proxy
```

**On falcon, run steps 3 and 4 inside a CPU job.** Typed on the falcon login node,
`./run_container classifier <command>` submits a batch job that asks for a share of a GPU and
returns at once. Neither step needs a GPU, and when the GPUs are busy the job can wait for hours.
Inside a job, `run_container` runs the command directly, so submit the two steps as one CPU job
instead:

```bash
sbatch -p work -q cpu_light -c 4 --mem 8G -t 30:00 -o setup_%j.out --wrap \
  "./run_container classifier python -m pip install --no-deps --target pkgs h5py && \
   ./run_container classifier python -s -m corenet.models.network"
ln -s pkgs/h5py pkgs/h5py.libs .
```

The links can be made before the job has finished. When it has, `setup_<jobid>.out` should end
without a `ModuleNotFoundError`.

Then fill in the files listed under [Files to edit before running](#files-to-edit-before-running)
and start the trainings as described under [Running the workflow](#running-the-workflow).

Why each step is needed:

- **The link `corenet`.** Classifier jobs are started from the barista directory, which Python
  searches for packages. The repository keeps its code in `CoReNet/src/corenet/`, so the link
  puts it where `import corenet` finds it. No `pip install` of CoReNet is needed: it is pure
  Python.
- **h5py.** Importing `corenet` loads CoReNet's HDF5 reader, which imports `h5py`. Barista never
  uses that reader, but without `h5py` the import fails.
- **`./run_container classifier` for step 3.** Training and evaluation run in the classifier
  container (Python 3.11). An `h5py` installed with the analysis container (`./run_container
  python`, Python 3.12) does not load there.
- **`-s` in step 4.** It makes Python ignore packages in your home directory
  (`~/.local/lib/pythonX.Y/site-packages`), which both containers otherwise load. A package
  found there makes a job work for you and fail for someone else.

`CoReNet`, `corenet`, `pkgs`, `h5py` and `h5py.libs` show up as untracked files in `git status`
of barista. Do not commit them.

If the setup is missing, HCR jobs are not affected; a `CoReNet.*` job stops at start-up with
`ModuleNotFoundError: No module named 'corenet'` (or `'h5py'`).

### What was tested

On falcon, 2026-11-19, with home-directory packages switched off, in a fresh clone of the
`corenet` branches of barista and coffea4bees (cloned from a local repository, not from GitLab):
steps 2 to 4, including the CPU-job form of steps 3 and 4, then a small CoReNet training and
evaluation on CPU (two folds, two epochs) in the classifier container, and the workflow's
`analyze` step in the analysis container. All ran. The `analyze` step needs neither CoReNet nor
`h5py`.

The same small training gives the same losses and AUCs as the network copy that used to be in
barista: identical in the analysis container, and equal to about seven digits between the two
containers, which have different torch versions.

Not tested with this setup: the GitLab addresses in step 1, a production-size training on a GPU,
hosts other than falcon, and the boosted job files.

### Other hosts and roast

The clone, the two links and `pkgs/` must exist in every barista directory a job runs from.
`bin/roast checkout` creates its own barista and coffea4bees checkouts on the remote host and
does not create these, and roast does not record the CoReNet commit. For runs through roast,
install CoReNet in the container instead.

### Alternative: CoReNet in the container image

Add this to the build of the classifier container, and steps 2 and 3 are not needed:

```bash
pip install "git+https://github.com/cms-cmu/CoReNet.git@6c260c3"
```

- It installs as `CoReNet-1.1.0` and brings `h5py` with it.
- What the two images had on 2026-11-18, without home-directory packages:

  | | Classifier container (`./run_container classifier`, `barista:classifier_latest`) | Analysis container (`./run_container python`, `barista:latest`) |
  | --- | --- | --- |
  | Python | 3.11 | 3.12 |
  | torch, numpy, pyyaml, matplotlib | present | present |
  | scikit-learn | present | missing |
  | h5py | missing | missing |
  | setuptools | 84 | 75.8, too old to build the package with `--no-build-isolation` |

- Only `corenet.models` (the network and its config parser) is used. CoReNet's own commands
  (`corenet-train`, `corenet-evaluate`, ...) read HDF5 files and are not part of this workflow.
- This route has been tested only as an install into a side directory, not as part of an image.

## Files to edit before running

In this order. Copy a file before editing it if you want to keep the sample intact, and point the
config at the copy.

### 1. Workflow config: `coffea4bees/workflows/config/corenet_boosted.yml`

Must change (they contain `PLACEHOLDER`):

| Key | Value |
| --- | --- |
| `inputs.dataset_metadata` | Directory of dataset YAMLs for the classifier jobs, with a trailing `/` (see [Dataset metadata](#dataset-metadata)). |
| `inputs.classifier_inputs` | The classifier-input friend trees, as `<manifest>.json@@<key>` (see [Classifier-input friend trees](#classifier-input-friend-trees)). |
| `inputs.jcm_weights` | JCM weights, `<file>.yml@@JCM_weights`, or `""` to train without JCM. |
| `inputs.output_base` | Where models and friend trees are written, local or `root://`. Must be readable by the GPU jobs. |
| `analysis_config.dataset_location` | Dataset metadata for the histogramming step. Only needed with the `analysis` step. |

The workflow's first rule, `corenet_check_inputs`, stops a real run if a value under `inputs:` or
`corenet.network` still contains `PLACEHOLDER`. It does not check `analysis_config`, and it is
not run when both classifier steps are skipped.

Optional: `label` and `output_path` (name and local directory of the run), `steps`,
`corenet.network`, the `fvt:` / `svb:` blocks, `year_eras` and `dataset` (what the histogramming
step runs over), `analysis_config`, `channels`, and the [optional blocks](#optional-config-blocks).
Every key is listed in the [config reference](#config-reference).

### 2. Decay-chain file: `coffea4bees/classifier/config/corenet/HH4b_boosted.yml`

No placeholders. It tells CoReNet what objects the event contains and how HH decays into them.
Edit it only if you change the objects; see [Network inputs](#network-inputs).

| Block | Sample value | Meaning |
| --- | --- | --- |
| `objects.subjets`, `objects.jets` | `{features: [pt, eta, phi, mass], max: 4}`, `max: 8` | Object collections and their per-object features. |
| `objects.b` | `subjets pt 4` | Role `b`: four subjets, ordered by pt, read from the `CanJet_*` branches. Slots 0 and 1 are the subjets of the first AK8 jet, slots 2 and 3 those of the second. |
| `objects.other` | `jets pt 8` | Role `other`: up to eight further jets, read from `NotCanJet_*`. |
| `hypotheses.HH` | `H1: [b, b]`, `H2: [b, b]`, `context: other` | HH → two Higgs candidates, each → two `b` objects. `other` is used only as attention context. |
| `outputs`, `combine` | `main: HH`, `[HH]` | Which hypothesis feeds the classifier output. `main` is only a name; the classes come from the dataset. |
| `training` | `features: 8`, `attention_heads: 2`, ... | Network width and switches. |

The network scores all three ways of pairing the four `b` objects into two Higgs candidates
(outputs `q_1234`, `q_1324`, `q_1423`); `q_1234` is the pairing as written in the friend tree.

### 3. Classifier settings: `coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/common.yml`

No placeholders. Shared by the FvT and SvB jobs. It holds the eras per year (`cms.CollisionData`;
these must exist in your dataset metadata) and the input branches (`HCR.InputBranch`):
`feature_CanJet`, `n_CanJet`, `feature_NotCanJet`, `n_NotCanJet`, and `feature_ancillary` (event
scalars fed to the network). Instead of editing it you can replace its whole `setting:` list from
the workflow config with `classifier_setting` ([optional blocks](#optional-config-blocks)).

These eras are separate from `year_eras` in the workflow config, which drives only the
histogramming step (UL17 is B–F here and C–F there, as in the resolved Run 2 configs).

### 4. Job templates: `HH4b_boosted_corenet/{FvT,SvB}/{train,evaluate}.yml`

No placeholders: the paths in them are template keys, filled by the workflow from `inputs:` and
`corenet.network` (or from `paths.yml` when run by hand). Edit them to change dataset options
(`dataset:` blocks) or training options (`model:` block). The usual training options can be
changed from the config instead, through `fvt.model_options` / `svb.model_options`.

To keep the samples intact, copy the whole `HH4b_boosted_corenet` directory, edit the copy, and
set `fvt.wfs_base` / `svb.wfs_base` to `<copy>/FvT` and `<copy>/SvB`. `common.yml` is always read
from one level above `wfs_base`.

The templates are filled with Python's `str.format`, so the only braces allowed anywhere in them,
comments included, are the template keys: `metadata`, `inputs`, `jcm`, `network`, `fvt_model`,
`fvt_friend`, `svb_model`, `svb_friend`. Write a literal brace as `{{` or `}}`.

### 5. `HH4b_boosted_corenet/paths.yml` (only to run the jobs by hand)

Not used by the workflow. All values except `network` are placeholders; see
[Running the classifier jobs without the workflow](#running-the-classifier-jobs-without-the-workflow).

## Inputs you have to provide

### Dataset metadata

A directory of YAMLs in the layout of `coffea4bees/metadata/datasets/`. The classifier jobs look
up these keys:

| Needed by | Keys |
| --- | --- |
| FvT and SvB (data) | `data.<year>.picoAOD.<era>.files`, for every year and era in `common.yml` |
| FvT and SvB (ttbar) | `<sample>.<year>.picoAOD.files` for `TTToSemiLeptonic`, `TTToHadronic`, `TTTo2L2Nu` |
| SvB (signal) | `ZZ4b`, `ZH4b`, `ggZH4b`, and `GluGluToHHTo4B_cHHH{0,1,2p45,5}` |

To load fewer processes, add `--mc-processes <list>` to a `dataset:` block (e.g. `ggF` in the
`HCR.SvB.Signal` block), `--ggf-load-kl 1` for the SM signal only, and `--ggf-signal-pattern` for
Run 3 signal names (`GluGlutoHHto4B_kl-{kl}_kt-{kt}_c2-{c2}` style). The SvB training uses the
ggF sample with kl = 1; the other kl values are loaded but not trained on.

### Classifier-input friend trees

Given as `<manifest>.json@@<key>`. The loader is the one of the resolved analysis, so the friend
tree must use its branch names, whatever objects they hold:

| Branch | Content |
| --- | --- |
| `CanJet_pt`, `_eta`, `_phi`, `_mass` | The `b` objects, exactly `n_CanJet` per event. Here: the four subjets of the two Higgs-candidate AK8 jets, slots 0, 1 from the first jet, 2, 3 from the second. |
| `NotCanJet_pt`, `_eta`, `_phi`, `_mass`, `_isSelJet` | The `other` objects, any number; padded or cut to `n_NotCanJet`. |
| `fourTag`, `threeTag` | The two tag categories the FvT maps between. |
| `SR`, `SB`, `ZZSR`, `ZHSR`, `HHSR` | Mass regions. The FvT trains on `SB` and `SR` events except four-tag `SR` data; the SvB trains on `SR`. |
| `passHLT`, `weight` | Trigger decision (required for data in the FvT) and event weight. |
| `nSelJets` | Jet multiplicity used to look up the JCM weight. |
| event scalars | One branch per name in `feature_ancillary`; `year` comes from the dataset. |

`coffea4bees/analysis/helpers/dump_friendtrees.py::dump_input_friend` writes this format; its
`CanJet` / `NotCanJet` arguments name the event collections to write.

**JCM caveat.** The JCM weight is applied only if `nSelJets` is loaded, and the loader reads only
the branches it is told about. The sample `common.yml` does not include `nSelJets`
(`feature_ancillary: [nOtherJets, year]`), so with the files as they are the JCM weights are
**silently not applied**. Add `--branch nSelJets` to the `dataset:` blocks that carry
`--JCM-weight`, or set `inputs.jcm_weights: ""` to train without JCM on purpose.

### JCM weights

`<file>.yml@@JCM_weights`: a list of weights indexed by `nSelJets` (index 4 is the first entry).
No boosted fit exists.

## Running the workflow

```bash
# dry run: lists the jobs, submits nothing (add -p to print each job's command)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml -n

# run (start it inside tmux: Snakemake must stay alive until the last job ends)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml
```

Run part of the chain:

```bash
# one target: corenet_fvt, corenet_svb, corenet_analysis, corenet_stats, corenet_all (default)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml corenet_svb

# classifiers only; the analysis and stats rules are not loaded
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml --config steps=fvt,svb

# FvT only
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml --config steps=fvt

# SvB only: first set inputs.fvt_friend in the config to an existing FvT friend directory
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml --config steps=svb
```

`steps` can also be set in the config file (`steps: [svb]`). Write `steps=svb` without spaces.
The SvB trains on three-tag data weighted by the FvT, so an SvB-only run stops with an error if
`inputs.fvt_friend` is empty. The two single-step cases have not been dry-run.

A step left out of `steps` is replaced by an existing friend tree: `inputs.fvt_friend` when `fvt`
is skipped, `inputs.svb_friend` when `svb` is skipped. A second config file with only some keys
overrides those keys (`--configfile a.yml b.yml`).

What runs, in order:

1. `corenet_check_inputs` (placeholder check).
2. FvT: stage the job files, `train`, `evaluate` (writes the FvT friend tree), plus `yields`,
   `analyze` (loss curves, ROCs) and `classifier_gallery`.
3. SvB: the same; its job files are staged only after the FvT friend tree exists.
4. Histogramming (`runner.py`, one job per data era and per MC sample and year), merging, plots,
   cutflow check. Its config is written only after both friend trees exist.
5. Fit: datacards, workspace, limits, significance, fit diagnostics, likelihood scan.

### Where things go

| What | Where |
| --- | --- |
| Models, `result.json` (per-epoch losses) | `<inputs.output_base>/classifier/<fvt.label or svb.label>/` |
| Friend trees | `<inputs.output_base>/friend/<label>/`, read as `result.json@@analysis.0.merged` |
| Staged job files (`wfs/train.yml`, `wfs/evaluate.yml`, `common.yml`: what the run actually used), logs, flag files, training plots | `<output_path>classifiers/<label>/` |
| Histograms, plots, cutflow | `<output_path>PhaseF_1_analysis/` |
| Datacards and Combine results | `<output_path>PhaseF_2_stats/` |

### Cluster notes

- `./run_container snakemake` adds the Snakemake profile of the host. On falcon every rule except
  the small bookkeeping ones becomes a SLURM job; `train` and `evaluate` request a GPU (default
  `mps:25`, QOS `light`, 24 GB, 8 h; see [optional blocks](#optional-config-blocks)).
- On lxplus the GPU profile is picked from the Snakefile name, and this Snakefile is not in that
  list: pass `--profile software/snakemake/profiles/lxplus_gpu` yourself.
- On Bridges-2 the profile's resources for `train` and `evaluate` win over the config.
- The histogramming jobs run `runner.py` locally in each job (`additional_parameters: ""`). Add
  `--condor` or `--slurm` there to spread each one over batch workers.

## Optional config blocks

These are top-level keys of `coffea4bees/workflows/config/corenet_boosted.yml`. They are not in
the checked-in file. Add them next to the existing blocks, for example:

```yaml
corenet:
  network: "coffea4bees/classifier/config/corenet/HH4b_boosted.yml"   # already there
  inputs: {b: CanJet, other: NotCanJet}    # role -> branch prefix; the first role of the decay-chain file first
  architecture: {n_features: 16}           # also: attention, use_attention_gate, use_kv_proj

classifier_setting:                        # replaces the whole `setting:` list of common.yml
  - module: cms.CollisionData              # copy all four entries of common.yml and edit them;
    option:                                # only one is shown here
      - eras:
          UL18: [A, B, C, D]

gpu_gres: "mps:25"                         # GPU job resources; defaults shown
gpu_qos: "light"
gpu_mem: 24000                             # MB
train_runtime: 480                         # minutes
eval_runtime: 480
```

`corenet.inputs` and `corenet.architecture` apply to both classifiers. `classifier_setting` is
written out as the `common.yml` of both classifiers, so give the full list (`ml.DataLoader`,
`ROOT`, `cms.CollisionData`, `HCR.InputBranch`); entries you leave out are lost.

## Config reference

All keys of `corenet_boosted.yml`:

| Key | Meaning |
| --- | --- |
| `label` | Name of the run, used in file names. Also fills `{roast_id}`. |
| `output_path` | Local directory of the run. |
| `steps` | Subset of `[fvt, svb, analysis, stats]`. |
| `inputs.*` | See [Files to edit before running](#files-to-edit-before-running). `fvt_friend`, `svb_friend`: existing friend directories, read only when that step is skipped. |
| `corenet.network` | Decay-chain file. `corenet.inputs`, `corenet.architecture`: [optional](#optional-config-blocks). |
| `fvt.label`, `svb.label` | Names of the model and friend directories. |
| `fvt.wfs_base`, `svb.wfs_base` | Directory with `train.yml` and `evaluate.yml`; `common.yml` is read from `../`. |
| `fvt.model_options`, `svb.model_options` | Changes to the `model:` block of `train.yml`, by flag (below). |
| `retries` | Retries of the GPU jobs. 0 on purpose: see [Known rough edges](#known-rough-edges-of-corenet). |
| `year_eras`, `dataset` | Years, eras and datasets of the histogramming step. |
| `additional_parameters` | Extra `runner.py` arguments for the histogramming jobs. |
| `analysis_config` | Processor, dataset location, friend and weight files, runner and processor options. The FvT and SvB friend trees and `JCM_file` are filled in from this run unless you set them under `analysis_config.config`. |
| `analysis_friend_names` | Names under which the processor reads the FvT and SvB friend trees (`FvT`, `SvB_MA`). |
| `plot_config`, `known_counts` | Plot list; reference cutflow (none). |
| `channels`, `combine_flags`, `combine_outdir`, `make_combine_inputs`, `metadata_template` | Fit setup: histogram, signal, stat-only, blinding. Same meaning as in `nominal_run2.yml`. The fit is stat-only and reuses the resolved `HH4b` channel definition. |

`{roast_id}` and `{eos_prod}` are replaced anywhere in the config. `{eos_prod}` comes from
`--config eos_prod=...`, else `~/.config/coffea4bees/profile.yml`, else your account name.

Paths in `inputs:` and `corenet.network` may not contain spaces, commas, quotes, braces or `": "`.

### `model_options` values

Each key is a flag of the model module in `train.yml`:

| Value | Effect |
| --- | --- |
| scalar, e.g. `--kfolds: 5` | Replaces the flag's value. |
| list, e.g. `--training: [FixedStep, {epoch: 30}]` | Several arguments; a mapping is passed as embedded options. |
| `true` | Adds a bare flag. |
| `null` | Removes the flag, e.g. `--finetuning: null`. |

## How to change things

### Network size

From broadest to narrowest:

- `training.features` in the decay-chain file: the width of every layer. It must be divisible by
  `attention_heads`, even when `phase_symmetric: true`, and at least `2 × attention_heads` when
  `kv_projection: true`.
- `corenet.architecture: {n_features: 16}` in the config: the same for this run, both classifiers.
- `svb.model_options: {"--architecture": [{n_features: 16}]}`: one classifier only.

`attention: false` under `architecture` drops the context objects (`other`).

### Network inputs

Three places must agree, or training stops with a shape error:

| To change | Edit |
| --- | --- |
| Per-object features (e.g. add a b-tag score to the subjets) | `feature_CanJet` in `common.yml` and the `features` of the `subjets` collection in the decay-chain file, same order. The friend tree needs `CanJet_<name>`. |
| Features of the context objects | `feature_NotCanJet` (keep `isSelJet` last) and the `jets` collection. |
| Number of objects | `n_CanJet` / `n_NotCanJet`, and `max` plus the role size in the decay-chain file. |
| Event scalars | `feature_ancillary` in `common.yml`. Nothing to change in the decay-chain file. |
| Which physical objects are used | The friend tree: what you write to `CanJet_*` and `NotCanJet_*`. |

Rules for the decay-chain file:

- Do not add `event:` or `labels:`; they are filled from `feature_ancillary` and the dataset.
- The first role (`b`) must be the first key of `--inputs`. Its leading slot sets the eta
  reflection (eta is flipped so that slot has eta ≥ 0).
- A scalar whose name contains `nSelJets` enters the network as `log(nSelJets - (min_jets - 1))`,
  with `min_jets: 4` unless the file sets it (minimum 2). For events with fewer jets the value is
  not finite and the network replaces it with -1. Use another branch name or set `min_jets`.
- The FvT and SvB outputs write the columns `q_1234`, `q_1324`, `q_1423` by name, so they need the
  sample topology: one role of four objects paired as `H1: [b, b]`, `H2: [b, b]`. A hypothesis
  over two roles builds and runs but names its outputs `q_HH_<n>`, so evaluation would need a new
  output definition. HH as two AK8 jets with no further structure cannot be written: every child
  of a hypothesis must itself be built from at least two objects.

### Folds and seeds

In `model_options` or in the `model:` block of `train.yml`:

| Flag | Effect |
| --- | --- |
| `--kfolds: 3` | Number of folds. One model per fold; each event is evaluated by the model that did not train on it. |
| `--kfold-seed: [FvT, random]` | Seed of the assignment of events to folds. |
| `--kfold-seed-offsets: "0"` | Which fold assignments to train; `"0-2"` trains three sets of `kfolds` models. |
| `--kfold-offsets: "0"` | Train only these folds. |
| `--seed: 2` | Base seed of the weight initialisation and training. |

Seeded trainings are not reproducible run to run: two trainings with the same seed differ event by
event almost as much as a CoReNet training differs from an HCR one. Compare variants over several
trainings.

### Training schedule

```yaml
fvt:
  model_options:
    --training: [FixedStep, {epoch: 30, bs_init: 1024, lr_init: 0.005}]
    --finetuning: null          # drop the fine-tuning stage
```

`FixedStep` (`src/classifier/config/scheduler/basic.py`): `epoch` (20), `bs_init` (batch size,
1024, doubled at epochs `bs_milestones` = 1, 3, 6, 10), `lr_init` (0.01), `lr_scale` (0.25,
applied at each epoch in `lr_milestones` = 15 to 24). Milestones are absolute epoch numbers: move
them when you change `epoch`. `FinetuneStep` trains only the output layer, from `lr_init: 1e-4`.
`ClosureGatedStep` does not work with CoReNet (it needs the fit-quality benchmark, below).

### Signal and normalisation of the SvB

In `SvB/train.yml`, `--norm 3` sets the total background weight; each signal process (ZZ, ZH,
ggF) is normalised to 1. To train on HH only, add `--mc-processes ggF` to the `HCR.SvB.Signal`
block and set `--norm 1`. Other dataset options (`--regions`, `--no-SR`, ...) also go in the
`dataset:` blocks.

## Running the classifier jobs without the workflow

Copy `paths.yml`, fill in every value, and run the four jobs in the order FvT/train,
FvT/evaluate, SvB/train, SvB/evaluate, starting each only after the previous one has finished:

```bash
T=coffea4bees/classifier/config/workflows/HH4b_boosted_corenet
cp $T/paths.yml my_paths.yml        # then edit my_paths.yml

JOB=FvT/train                       # then FvT/evaluate, SvB/train, SvB/evaluate
./run_container classifier env CLASSIFIER_CONFIG_PATHS=coffea4bees \
    python -m src.classifier.task.main template file:##my_paths.yml \
    $T/$JOB.yml -from $T/common.yml -flag debug
```

FvT evaluation writes the FvT friend tree, which the SvB training reads; SvB evaluation writes the
SvB friend tree. Keys of `paths.yml`:

| Key | Meaning |
| --- | --- |
| `metadata` | Dataset metadata directory (trailing `/`). |
| `inputs` | Classifier-input friend trees, `<manifest>.json@@<key>`. |
| `jcm` | JCM weights, `<file>.yml@@JCM_weights`. To train without, delete the `--JCM-weight` lines of the two `train.yml`. |
| `network` | Decay-chain file. |
| `fvt_model`, `svb_model` | Where models and `result.json` are written (local or `root://`). |
| `fvt_friend`, `svb_friend` | Where friend trees are written. |

What the command does:

- `./run_container classifier <command>` runs the command in the classifier (GPU) container. On
  falcon it submits a SLURM GPU job and returns at once (log
  `slurm_logs/classifier_batch_<jobid>.out`); wait for the job before starting the next. On
  lxplus it submits an HTCondor GPU job. Elsewhere it runs in place.
- `CLASSIFIER_CONFIG_PATHS=coffea4bees` makes the `HCR.*` and `CoReNet.*` modules visible;
  `template file:##my_paths.yml` fills the template keys from your file (keep them in a file:
  `run_container` re-joins its arguments, so an inline mapping does not survive); `-from` adds the
  shared settings; `-flag debug` prints the loaded yields.

The options of a model module are listed by `help` (`--network x` only satisfies the required
argument):

```bash
./run_container classifier env CLASSIFIER_CONFIG_PATHS=coffea4bees \
    python -m src.classifier.task.main help -model CoReNet.FvT.baseline.Train --network x
```

The workflow runs the same four commands with its staged job files; `-np` prints them.

## Known rough edges of CoReNet

- **Resuming a training.** CoReNet checkpoints its first stage, the pass that measures the input
  means and widths (HCR does not). A job resumed after that stage can skip the pass and train on
  unnormalised inputs. This is why `retries` is 0. If a training dies, delete the `checkpoint_*`
  files (in `<output_path>classifiers/<label>/` for the workflow, in the model directory for a
  job run by hand) and start again.
- **No fit-quality benchmark.** CoReNet trainings report losses and ROCs but not the sideband
  closure χ² that HCR FvT trainings report, so there is no automatic FvT quality check.
- **Warm start.** `--pretrained-weights <directory>` does not find the saved models (it looks for
  `_offsetN_`, the files are named `__offset_N__`) and only reads local directories. A path to one
  `.pkl` works, but then every fold starts from that same file.
- **Low-pt jet multiplicity.** The network copy that used to be in barista took a scalar whose
  name contains both `nSelJets` and `lowpt` without the `min_jets - 1` offset, as HCR does. The
  package applies the offset to every `nSelJets` scalar. No CoReNet job file in the repository
  uses such a scalar; a low-pt CoReNet training would need this added to the package.
- **Version string.** The tested commit installs as `CoReNet-1.1.0`, but `corenet.__version__`
  still reads `1.0.0`. Go by the commit.
- **Input and weight plots.** The classifier workflow's `plot_inputs` and `plot_weights` rebuild an
  HCR network and cannot read a CoReNet model. They are off here.

## Boosted code that exists, and what is missing

Boosted code in coffea4bees (the same files on `master` and on this branch):

| File | What it does |
| --- | --- |
| `analysis/processors/processor_boosted_HH4b.py` | Applies the resolved and the boosted selection (`apply_boosted_4b_selection`) and returns event counts (boosted and resolved, only one, neither), the run/lumi/event lists of boosted-selected events, and a cutflow. No histograms. |
| `analysis/processors/processor_synthetic_boosted.py` | Histograms of AK8 jets and of their split into two subjets, for synthetic-data studies. All events are labelled four-tag SR. On `master` and this branch it uses `cutFlow` and `ClusterHistsDetailedBoosted` without importing them, so it fails on the first chunk; the `BoostedSynthetic` branch has the working imports. |
| `analysis/processors/processor_synthetic_boosted_PFcands.py` | The same kind of histograms for AK8 jets with pt > 300 GeV and two subjets. |
| `analysis/processors/processor_synthetic_boosted_stored.py` | Run 3 synthetic-data study: picks the two AK8 jets with the highest stored GloParT T_Xbb, computes the boosted BDT scores and fills histograms. Every event is marked fourTag and SR; it reads no FvT or SvB and writes no classifier inputs. The closest starting point for a boosted analysis step. |
| `analysis/processors/processor_overlap_study.py` | Overlap counts and simple histograms of the resolved, boosted, semi-resolved and low-pt selections, with generator matching for MC. |
| `analysis/processors/processor_hlt_study.py` | Per-event selection flags and HLT bits, for trigger studies. |
| `processor_HH4b.py`, option `apply_boosted_veto` | Run 2 UL: removes events selected by the boosted analysis, from the event lists `metadata/boosted_overlap_{signal,data}.coffea`. |
| `skimmer/processor/skimmer_4b_boosted.py` | NanoAOD → picoAOD skim keeping events with at least two AK8 jets (config `skimmer/metadata/HH4b_boosted.yml`, test `scripts/skimmer-boosted.sh`). |
| `skimmer/processor/skimmer_4b_forboosted.py` | Skims the events whose numbers are in a given `.coffea` event list. |
| `skimmer/processor/make_declustered_data_boosted_4b.py` | Makes synthetic boosted data by splitting the two Xbb-tagged AK8 jets into `SubJet_*` branches. |
| `metadata/datasets/archive/HIG-24-010/datasets_HH4b_2024_v2_boosted.yml` | Archived picoAOD lists of the boosted skim: UL18 data (eras A–D), SM ggF HH, synthetic data. |

What this workflow still needs:

- **Classifier inputs.** None of these processors writes the classifier-input friend tree
  (`CanJet_*`, `NotCanJet_*`, tag and region flags, `nSelJets`, weight). In the resolved analysis
  `processor_HH4b.py` does it with the option `make_classifier_input`, which calls
  `dump_input_friend`; nothing does it for AK8 jets and subjets. A boosted processor would also
  have to define the three-tag and four-tag categories and the SR/SB regions, which the boosted
  selection does not have.
- **Histogramming with FvT and SvB.** None of these processors reads the FvT and SvB friend trees
  or fills the histograms the plotting and fit steps read, so none can replace
  `analysis_config.processor`. It stays `processor_HH4b.py`, which applies the resolved selection.
- **Background model and fit inputs.** No boosted JCM fit, FvT closure check, background
  systematics or fit metadata. The `stats` step is stat-only with the resolved `HH4b` channel.

## Using it with roast

Not wired in and not tried. `bin/roast new` accepts a non-phase workflow as
`--step <host>:<snakefile>[:targets]`. Since the GPU and histogramming steps usually run on
different hosts, it would take two steps: `steps=fvt,svb` on the GPU host, and
`steps=analysis,stats` with `inputs.fvt_friend` / `inputs.svb_friend` on the other.
