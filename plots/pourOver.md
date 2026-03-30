# PourOver — Interactive Web-Based Plot Viewer

`pourOver.py` serves a local webpage with two complementary interfaces:

1. **Interactive form** — mirrors the `iPlot.py` `plot()` interface, letting you create custom plots on demand without leaving the browser
2. **Gallery** — a pre-generated grid of every variable × region combination, with PNG display and PDF download


---

## Installation

### Using the analysis container (recommended)

`flask` and `mplhep` are included in the analysis container (`barista:latest`). No extra setup needed — just run from the barista root:

```bash
./run_container pourover file.coffea -m meta.yml --new --label my-label
```

Then open **[http://localhost:5000](http://localhost:5000)** in your browser. The interactive plot form is the landing page; the gallery is at `/gallery-view`.

Add `--pregallery` to pre-generate PNG/PDF thumbnails for every variable × region on startup (~2 min for a typical file). Without it the gallery page is empty but the interactive form works immediately.

### Optional: shell alias and tab completion

If you want to type `pourover` instead of `./run_container pourover`, and get label tab-completion for `--load`/`--delete`/`--rename`, add to `~/.bashrc`:

```bash
source /path/to/barista/coffea4bees/plots/pourover-completion.sh
```

This is purely optional — `./run_container pourover` works without it.

### Using a local Python environment (no Apptainer)

Requires **Python 3.10** (coffea 0.7.x is not compatible with Python 3.11+).

```bash
python3.10 -m venv ~/python-environments/pourover
source ~/python-environments/pourover/bin/activate
pip install -r coffea4bees/plots/requirements-pourover.txt
```

When Apptainer is not available, the completion script alias falls back to `python coffea4bees/plots/pourOver.py` automatically.

---

## Main Commands

PourOver supports first-class session snapshots via `--new` and `--load`.

## Creating a snapshot (`--new`)

```bash
pourover file.coffea -m meta.yml --new --label ul18-sb-test
```

## Reloading a snapshot (`--load`)

```bash
pourover --load ul18-sb-test
```

## Listing archives (`--list`)

```bash
pourover --list
```

## Deleting archives (`--delete`)

```bash
pourover --delete ul18-sb-test
#   Removed .pourover_archives/pourover_archive_20260319_153045
# Deleted 1 archive(s) with label 'ul18-sb-test'
```

## Renaming archives (`--rename`)

```bash
pourover --rename ul18-sb-test ul18-sb-final
# Renamed 1 archive(s): 'ul18-sb-test' → 'ul18-sb-final'
```

---

## Command-Line Options

All standard `iPlot.py` / `makePlotsAll.py` arguments are accepted, plus:

| Option                | Default                                     | Description                                                                             |
|-----------------------|---------------------------------------------|-----------------------------------------------------------------------------------------|
| `inputFile`           | (required unless `--load`)                  | Path(s) to `.coffea` histogram file(s)                                                  |
| `-m / --metadata`     | `plotsAll.yml`                              | Metadata YAML defining processes, regions, colors                                       |
| `--modifiers`         | `plotModifiers.yml`                         | Per-variable plot options (xlim, rebin, 2d flag, etc.)                                  |
| `-o / --outputFolder` | `pourover_output`                           | Directory for generated PNG and PDF files                                               |
| `--port`              | `5000`                                      | Port to serve on                                                                        |
| `--pregallery`        | off                                         | Pre-generate the full plot gallery on startup                                           |
| `--new`               | off                                         | Snapshot inputs into a timestamped archive, register it, and serve                      |
| `--load LABEL`        | —                                           | Load a previously created archive by its label                                          |
| `--list`              | —                                           | List all archives in the registry and exit                                              |
| `--rename OLD NEW`    | —                                           | Rename all archives with label OLD to NEW and exit                                      |
| `--delete LABEL`      | —                                           | Delete all archives with this label (removes directories and registry entries) and exit |
| `--registry FILE`     | `.pourover_archives/pourover_registry.json` | Path to the archive registry JSON                                                       |
| `--label LABEL`       | (required with `--new`)                     | Label for this archive, e.g. `ul18-sb-test`                                             |

### Examples

```bash
# Standard run
./run_container pourover file.coffea -m metadata.yml

# Custom port
./run_container pourover file.coffea -m metadata.yml --port 8080

# Pre-generate the gallery (default is off — interactive form is available immediately)
./run_container pourover file.coffea -m metadata.yml --pregallery

# Multiple input files (overlaid)
./run_container pourover fileA.coffea fileB.coffea -m metadata.yml -l "Run2" "Run3"
```

---

## Session Management

PourOver supports first-class session snapshots via `--new` and `--load`.

### Creating a snapshot (`--new`)

```bash
pourover file.coffea -m meta.yml --new --label ul18-sb-test
```

What happens:
1. A timestamped archive directory is created (`.pourover_archives/pourover_archive_YYYYMMDD_HHMMSS/`)
2. Input coffea file(s) and metadata YAML are copied into `archive/inputs/`
3. A `manifest.json` is written recording the inputs, metadata path, timestamp, and label
4. An entry is appended to `.pourover_archives/pourover_registry.json`
5. The gallery is generated into `archive/gallery/` and the server starts

The archive is self-contained and relocatable — all paths in `manifest.json` are relative to the archive directory.

### Reloading a snapshot (`--load`)

```bash
pourover --load ul18-sb-test
```

What happens:
1. The label is looked up in the registry to find the archive directory
2. `manifest.json` is read to locate the coffea and metadata files (inside `inputs/`)
3. Histograms are loaded from the archived copies
4. The existing gallery PNGs are reused (no regeneration)
5. Flask starts immediately

No need to re-specify any inputs — everything comes from the archive. If the same label exists more than once, the most recent entry is used.

### Listing archives (`--list`)

```bash
pourover --list
```

Output is columnar: label, timestamp, archive directory path.

```
ul18-sb-test    2026-03-19T15:30:45  .pourover_archives/pourover_archive_20260319_153045
ul18-sr-full    2026-03-20T09:12:00  .pourover_archives/pourover_archive_20260320_091200
Run3-MvD        2026-03-21T11:05:33  .pourover_archives/pourover_archive_20260321_110533
```

Use `--registry` to point at a non-default registry file.

### Deleting archives (`--delete`)

```bash
pourover --delete ul18-sb-test
#   Removed .pourover_archives/pourover_archive_20260319_153045
# Deleted 1 archive(s) with label 'ul18-sb-test'
```

Removes every archive directory matching the label and purges the corresponding entries from the registry. If a directory is unexpectedly missing it prints a warning and continues. The operation is not reversible.

### Renaming archives (`--rename`)

```bash
pourover --rename ul18-sb-test ul18-sb-final
# Renamed 1 archive(s): 'ul18-sb-test' → 'ul18-sb-final'
```

Updates all matching entries in the registry and the `manifest.json` inside each archive directory. Labels must be kebab-case: letters, digits, and hyphens, no leading or trailing hyphens (e.g. `ul18-sb-final`, `Run3-MvD`).

### Shell tab completion

A bash completion script provides label completion for `--load` and `--rename`, plus file/directory completion for all other arguments.

Add to `~/.bashrc` (use the absolute path to your barista checkout):

```bash
source /path/to/barista/coffea4bees/plots/pourover-completion.sh
```

This defines a `pourover` alias. Then use `pourover` in place of `./run_container pourover`:

```
pourover --load ul18-<TAB>       # completes labels from registry
pourover --rename ul18-<TAB>     # same
pourover --delete ul18-<TAB>     # same
pourover file<TAB>               # completes .coffea files
pourover -m meta<TAB>            # completes .yml/.yaml files
```

### Archive directory layout

```
.pourover_archives/
  pourover_registry.json
  pourover_archive_YYYYMMDD_HHMMSS/
    manifest.json          ← created/label/inputs/metadata (relative paths)
    inputs/
      analysis.coffea      ← copy of input coffea file(s)
      metadata.yml         ← copy of metadata YAML
    gallery/
      v4j_mass_region_SR.png
      v4j_mass_region_SR.pdf
      ...
    interactive/           ← plots created during this session
    cli_history.json
    interactive_history.json
```

---

## Browser Interface

### Gallery Panel (left)

- Displays thumbnails for every 1D variable × region combination
- **Filter box**: type any part of a variable name to narrow the grid
- **Click** a thumbnail to open a full-size lightbox view
- **⬇ PDF** link on each thumbnail for high-resolution download

### Interactive Panel (right)

Controls mirror the `iPlot.py` `plot()` function:

| Control           | Equivalent `plot()` kwarg             |
|-------------------|---------------------------------------|
| Variable          | `var`                                 |
| Cut               | `cut`                                 |
| Region checkboxes | `region` (single or list for overlay) |
| Process           | `process`                             |
| Year              | `year`                                |
| doRatio           | `doRatio=1`                           |
| Normalize         | `norm=1`                              |
| yscale            | `yscale="log"` / `"linear"`           |
| Rebin             | `rebin=N`                             |
| xlim / ylim       | `xlim=[min, max]` / `ylim=[min, max]` |
| Add flow          | `add_flow=True`                       |
| 2D plot           | switches to `plot2d()`                |

Click **Plot it** to generate. The result appears inline with a PDF download link. Results stack and can be individually dismissed.

---

## Relation to Other Plotting Scripts

| Script            | Mode                       | Use when                                                                               |
|-------------------|----------------------------|----------------------------------------------------------------------------------------|
| `iPlot.py`        | Interactive Python REPL    | Exploratory, scripting, overlays                                                       |
| `makePlotsAll.py` | Batch, generates all plots | CI, producing full plot sets for notes/papers; supports `-j N` for parallel generation |
| `pourOver.py`     | Browser-based              | Browsing the full variable set, sharing results, quick comparisons                     |

`pourOver.py` reuses the same plotting backend (`src/plotting/`) as both scripts. Any plot you can make in `iPlot.py` can also be made in the interactive panel.

---

## Notes

- The server runs single-threaded (`threaded=False`) since matplotlib is not thread-safe. One plot is generated at a time; rapid successive clicks will queue.
- Gallery generation and interactive plots persist inside the archive directory. Use `--load` to restart instantly without regenerating.
- 2D plots (as flagged in `plotModifiers.yml`) are not included in the gallery but can be created via the interactive form by checking **2D plot** and selecting a process.
- ttbar control region cuts (`passMuMu`, `passElMu`) automatically switch to the `hists_ttbar` histogram key, matching `iPlot.py` behaviour.
