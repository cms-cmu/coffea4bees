# Metadata

This directory houses the unified dataset configurations, trigger definitions, luminosities, and friend tree mappings used by the analysis framework.

## Directory Structure

- **`datasets/`**: The unified database containing all active dataset definition files (e.g. `TT.yml`, `data.yml`, `GluGluToHHTo4B.yml`).
  - **Cross-Section Format (`xs`)**: To support multi-run execution without key collisions, every dataset defines cross-sections in a run-dependent mapping structure:
    ```yaml
    xs:
      Run2: <run2_cross_section>
      Run3: <run3_cross_section>
    ```
    If a dataset only exists in a single Run (such as Run-specific signals), the other Run defaults to a placeholder `1`.
  - **`datasets/archive/`**: Contains legacy/archived dataset definitions (e.g. `HIG-24-010`, `Run3_archive`).
- **`friends/`**: Centralized repository for friend tree configs and lookup maps.
  - Houses active `friends_HH4b.yml`, `friends_empty.yml`, `friends_ttHbb.yml`, and `friends_HH4b_none.yml`.
  - Holds active trigger weights and classifier lookup JSON files (`trigweights_2024_v1p2.json`, `data_SvBfriend.json`, etc.).
  - **`friends/archive/`**: Contains legacy/unused friend tree JSON lookup files.
- **`triggers_HH4b.yml`** & **`boosted_triggers_HH4b.yml`**: Trigger path definitions per year/era.
- **`luminosities_HH4b.yml`**: Integrated luminosity values per year/era.

---

## Historical Context & Versions (Archived in `datasets/archive/`)

### HIG-24-010 (Archived in `datasets/archive/HIG-24-010/`)
- Baseline dataset manifests used in the `HIG-24-010` analysis.
- **`2024_v2`**: Upgraded to canonical datasets in `metadata/datasets/` providing production Run 2 picoAOD skims alongside Run 3.