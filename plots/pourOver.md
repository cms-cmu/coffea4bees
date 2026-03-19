# PourOver — Interactive Web-Based Plot Viewer

`pourOver.py` serves a local webpage with two complementary interfaces:

1. **Gallery** — a pre-generated grid of every variable × region combination, with PNG display and PDF download
2. **Interactive form** — mirrors the `iPlot.py` `plot()` interface, letting you create custom plots on demand without leaving the browser

---

## Quick Start

### One-time setup

Requires **Python 3.10** (coffea 0.7.x is not compatible with Python 3.11+).

```bash
python3.10 -m venv ~/python-environments/pourover
source ~/python-environments/pourover/bin/activate
pip install -r coffea4bees/plots/requirements-pourover.txt
```

### Run (from barista root)

```bash
source ~/python-environments/pourover/bin/activate
python coffea4bees/plots/pourOver.py \
    coffea4bees/Run3_MvD/analysis_MvD_new.coffea \
    -m coffea4bees/plots/metadata/plotsAll_MvD_ttbar_weights.yml
```

Then open **http://localhost:5000** in your browser.

Gallery pre-generation runs at startup and prints progress to the terminal (~2 min for a typical file). The interactive form is available immediately.

---

## Command-Line Options

All standard `iPlot.py` / `makePlotsAll.py` arguments are accepted, plus:

| Option | Default | Description |
|--------|---------|-------------|
| `inputFile` | (required) | Path(s) to `.coffea` histogram file(s) |
| `-m / --metadata` | `plotsAll.yml` | Metadata YAML defining processes, regions, colors |
| `--modifiers` | `plotModifiers.yml` | Per-variable plot options (xlim, rebin, 2d flag, etc.) |
| `-o / --outputFolder` | `pourover_output` | Directory for generated PNG and PDF files |
| `--port` | `5000` | Port to serve on |
| `--no-pregallery` | off | Skip gallery pre-generation at startup |

### Examples

```bash
# Standard run
python coffea4bees/plots/pourOver.py file.coffea -m metadata.yml

# Custom port
python coffea4bees/plots/pourOver.py file.coffea -m metadata.yml --port 8080

# Skip gallery generation (fast startup, interactive form only)
python coffea4bees/plots/pourOver.py file.coffea -m metadata.yml --no-pregallery

# Multiple input files (overlaid)
python coffea4bees/plots/pourOver.py fileA.coffea fileB.coffea \
    -m metadata.yml -l "Run2" "Run3"

# Custom output directory
python coffea4bees/plots/pourOver.py file.coffea -m metadata.yml \
    -o my_plots/
```

---

## Output Directory Layout

```
pourover_output/
  gallery/
    v4j_mass_region_SR.png      ← pre-generated gallery plots
    v4j_mass_region_SR.pdf
    v4j_mass_region_SB.png
    ...
  interactive/
    20240101_120000_v4j_mass.pdf  ← plots created via the interactive form
    ...
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

| Control | Equivalent `plot()` kwarg |
|---------|--------------------------|
| Variable | `var` |
| Cut | `cut` |
| Region checkboxes | `region` (single or list for overlay) |
| Process | `process` |
| Year | `year` |
| doRatio | `doRatio=1` |
| Normalize | `norm=1` |
| yscale | `yscale="log"` / `"linear"` |
| Rebin | `rebin=N` |
| xlim / ylim | `xlim=[min, max]` / `ylim=[min, max]` |
| Add flow | `add_flow=True` |
| 2D plot | switches to `plot2d()` |

Click **Plot it** to generate. The result appears inline with a PDF download link. Results stack and can be individually dismissed.

### Archive Session

The **Archive Session** button (header) creates a timestamped, self-contained archive:

```
pourover_archive_YYYYMMDD_HHMMSS/
  reproduce.sh        ← shell command to reproduce this exact session
  inputs/
    analysis.coffea   ← copy of input coffea file(s)
    metadata.yml      ← copy of metadata file
  gallery/            ← copy of all pre-generated PNG + PDF files
  interactive/        ← copy of all interactive plots from this session
  index.html          ← standalone static page with all gallery plots embedded
```

The `index.html` is fully self-contained (images are base64-encoded inline) and can be opened directly in a browser with no server.

---

## Relation to Other Plotting Scripts

| Script | Mode | Use when |
|--------|------|----------|
| `iPlot.py` | Interactive Python REPL | Exploratory, scripting, overlays |
| `makePlotsAll.py` | Batch, generates all plots | CI, producing full plot sets for notes/papers; supports `-j N` for parallel generation |
| `pourOver.py` | Browser-based | Browsing the full variable set, sharing results, quick comparisons |

`pourOver.py` reuses the same plotting backend (`src/plotting/`) as both scripts. Any plot you can make in `iPlot.py` can also be made in the interactive panel.

---

## Notes

- The server runs single-threaded (`threaded=False`) since matplotlib is not thread-safe. One plot is generated at a time; rapid successive clicks will queue.
- Gallery generation and interactive plots are both written to `pourover_output/` and persist across server restarts. Use `--no-pregallery` to restart quickly without regenerating.
- 2D plots (as flagged in `plotModifiers.yml`) are not included in the gallery but can be created via the interactive form by checking **2D plot** and selecting a process.
- ttbar control region cuts (`passMuMu`, `passElMu`) automatically switch to the `hists_ttbar` histogram key, matching `iPlot.py` behaviour.
