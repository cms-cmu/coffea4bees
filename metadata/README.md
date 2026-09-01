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
  - **`datasets/archive/HIG-24-010/`**: Archived pre-2024 dataset definitions used for HIG-24-010 analysis.
  - **`datasets/archive/Run2_2024_v2/`**: Preserved reference copy of the Run2 2024_v2 datasets.
- **`friends/`**: Centralized repository for friend tree configs and lookup maps.
  - Houses active `friends_HH4b.yml`, `friends_ttHbb.yml`, `friends_empty.yml`, and `friends_HH4b_none.yml`.
  - Holds active trigger weights and classifier lookup JSON files (`trigweights_2024_v2.json`, `data_SvBfriend.json`, etc.).
  - **`friends/archive/`**: Contains legacy/unused friend tree JSON lookup files.
- **`triggers_HH4b.yml`** & **`boosted_triggers_HH4b.yml`**: Trigger path definitions per year/era.
- **`luminosities_HH4b.yml`**: Integrated luminosity values per year/era.

---

## Dataset Versions & Current Canonical Metadata

- **Run 2 (Canonical)**: Promoted `2024_v2` picoAODs with unified schema, updated trigger weights, and JCM configurations.
- **Run 3 (Canonical)**: Consolidated into 500k-event chunks under `/store/user/algomez/XX4b/Run3_nanov12/`.
- **Archived Versions**:
  - `HIG-24-010`: Preserved in `datasets/archive/HIG-24-010/`.
  - `Run2_2024_v2`: Preserved in `datasets/archive/Run2_2024_v2/` for backwards compatibility.