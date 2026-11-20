# CoReNet for boosted HH → 4b: instructions

Trains and evaluates a classifier between the classes you give it (for example signal and
background), with the CoReNet network. You list the ROOT files of each class in one config file
and start the workflow. All commands are run from the barista directory.

## 1. Setup

```bash
# 1. barista and coffea4bees, both on the corenet branch
git clone -b corenet https://gitlab.cern.ch/cms-cmu/barista.git
cd barista
git clone -b corenet https://gitlab.cern.ch/cms-cmu/coffea4bees.git

# 2. CoReNet, and a link so that Python finds the package
git clone https://github.com/cms-cmu/CoReNet.git
git -C CoReNet checkout 6c260c3
ln -s CoReNet/src/corenet corenet

# 3. h5py, which the containers do not have, installed next to the code
./run_container classifier python -m pip install --no-deps --target pkgs h5py
ln -s pkgs/h5py pkgs/h5py.libs .

# 4. check: no error means CoReNet and h5py load (a RuntimeWarning from runpy is harmless)
./run_container classifier python -s -m corenet.models.network

# 5. grid proxy, only if an input or output is on root://
./run_container voms-proxy-init -voms cms -rfc --valid 168:00 -out proxy/x509_proxy
```

On falcon, run steps 3 and 4 as one CPU job. Typed on the login node, `./run_container
classifier` submits a GPU job, which can wait for hours when the GPUs are busy:

```bash
sbatch -p work -q cpu_light -c 4 --mem 8G -t 30:00 -o setup_%j.out --wrap \
  "./run_container classifier python -m pip install --no-deps --target pkgs h5py && \
   ./run_container classifier python -s -m corenet.models.network"
ln -s pkgs/h5py pkgs/h5py.libs .
```

When the job has finished, `setup_<jobid>.out` should end without a `ModuleNotFoundError`.

`CoReNet`, `corenet`, `pkgs`, `h5py` and `h5py.libs` show up as untracked in `git status` of
barista. Do not commit them.

## 2. Input files

One or more ROOT files per class. Each file has a tree `Events` with these branches:

| Branch | Content |
| --- | --- |
| `CanJet_pt`, `CanJet_eta`, `CanJet_phi`, `CanJet_mass` | The four b objects (subjets), four values per event. Entries 0 and 1 belong to the first Higgs candidate, 2 and 3 to the second. |
| `NotCanJet_pt`, `NotCanJet_eta`, `NotCanJet_phi`, `NotCanJet_mass`, `NotCanJet_isSelJet` | Up to eight further jets per event. |
| `nOtherJets`, `year` | Event scalars. |
| `weight` | Event weight used in the loss. |
| `event` | Event number, used to split the events into folds. |

The class of an event is the class its file is listed under; no label branch is needed. To use
other branches, see [Changing things](#4-changing-things).

## 3. Fill in the config and run

Edit `coffea4bees/workflows/config/corenet_boosted.yml`. Only the `PLACEHOLDER` values have to
change:

```yaml
inputs:
  classes:                       # any number of classes, any names
    signal: ["/path/to/signal_1.root", "/path/to/signal_2.root"]
    background: ["/path/to/background.root"]
  output_base: "/path/to/output" # models and evaluation output; local or root://
```

Paths may not contain spaces, commas, quotes or braces.

```bash
# dry run: lists the jobs, submits nothing
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml -n

# run (inside tmux: Snakemake must stay alive until the last job ends)
./run_container snakemake --snakefile coffea4bees/workflows/Snakefile_CoReNet.smk \
    --configfile coffea4bees/workflows/config/corenet_boosted.yml
```

On falcon the training and the evaluation are submitted as SLURM GPU jobs.

What it runs: a check that no placeholder is left, the training (3 folds), loss and ROC plots,
and the evaluation, in which each event is scored by the model that did not train on it.

| Output | Where |
| --- | --- |
| Models and `result.json` (losses and AUCs per epoch) | `<output_base>/classifier/boosted_corenet/` |
| Evaluation output: `p_<class>` per class, and `q_1234`, `q_1324`, `q_1423` for the three ways of pairing the four b objects | `<output_base>/friend/boosted_corenet/` |
| The job files the run used, logs, plots | `output/corenet_boosted/classifiers/boosted_corenet/` |

## 4. Changing things

Nothing here is required. Each row names the file to edit and an HCR file of the same kind to
compare with.

| To change | Edit | HCR example |
| --- | --- | --- |
| Classes and their files, output location | `coffea4bees/workflows/config/corenet_boosted.yml` | `coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/workflow_config.yml` |
| Training options: epochs, batch size, folds, seeds | `model:` block of `coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/train.yml` | `coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/train.yml` |
| Evaluation options | `coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/evaluate.yml` | `coffea4bees/classifier/config/workflows/HH4b_2024_v2/SvB/evaluate.yml` |
| Which branches are read | `coffea4bees/classifier/config/workflows/HH4b_boosted_corenet/common.yml` | `coffea4bees/classifier/config/workflows/HH4b_2024_v2/common.yml` |
| The network: objects, decay chain, width | `coffea4bees/classifier/config/corenet/HH4b_boosted.yml` | none: HCR has no such file |
| Which class counts as signal in the ROC curve | `--signal` in `train.yml` | |

**HCR instead of CoReNet.** Delete the `--network` line (and the `--inputs` lines under it) from
`train.yml`. The same job then trains the HCR network; nothing else changes.

**Training options from the config.** Instead of editing `train.yml`, set them for one run:

```yaml
classifier:
  model_options:
    --kfolds: 5
    --training: [FixedStep, {epoch: 30, bs_init: 1024}]
    --finetuning: null           # null removes an option
```

**Branches.** The branch lists in `common.yml` and the decay-chain file must agree, or the
training stops with a shape error:

| To change | Edit |
| --- | --- |
| Features of the b objects | `feature_CanJet` in `common.yml` and `features` of `subjets` in the decay-chain file, same order. The files need `CanJet_<name>`. |
| Features of the other jets | `feature_NotCanJet` (keep `isSelJet` last) and `features` of `jets`. |
| Number of objects | `n_CanJet` / `n_NotCanJet`, and `max` and the role size in the decay-chain file. |
| Event scalars | `feature_ancillary` in `common.yml` only. |

**Network width.** `training.features` in the decay-chain file. It must be divisible by
`attention_heads`.

**Histograms and fit.** The workflow has two further steps, `analysis` and `stats`, that are
empty: no boosted processor or fit setup is provided. They stay off unless added to `steps:`.
