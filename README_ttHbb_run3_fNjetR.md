# ttHbb Run 3: per-era f(nJet) × r^n JCMs and the FvT trainings (branch `fvt_run3_allyears`)

Branch summary for review. Everything here is **add-only**: no existing file of coffea4bees is modified.
Plots: https://jgaytanv.web.cern.ch/plots/ttHbb/ (one `plots/` folder per run). Run reports and products:
`/eos/cms/store/group/phys_top/jgaytanv/ttHbb/4b/Run3_v1/ttHbb_stitched/{reports,classifier,friend}/`.

## 1. The JCM model

The default JCM (one constant per-jet pseudo-tag probability f) fails in Run 3: the 4b/3b ratio rises with nJet
faster than the model allows. Two additions:

- **f(nJet) = f0 · (nJet/4)^α**, α free in the fit.
- **3b loose-veto factor r^(nJet−3)**: threeTag requires exactly 3 Loose tags (all other jets fail Loose), fourTag
  requires ≥ 4 Medium tags, so the 3b sample loses events like p_failLoose^n; r = (1−p_M)/p_failLoose.

Per-era SB fits (χ²/ndf; repo-setup thresholds: 2022 at 30 GeV, 2023/2024 at 25 GeV selected jets):

| era | default JCM | f(nJet) × r^n | r | α |
|---|---|---|---|---|
| 2022_preEE | 73.3 | **0.70** | 1.055 | 0.33 |
| 2022_EE | 364.6 | **0.26** | 1.090 | 0.27 |
| 2023_preBPix | 1167 | **2.88** | 1.170 | 0.10 |
| 2023_BPix | 409 | **1.63** | 1.128 | 0.27 |
| 2024 | 11343 | 169.7 | 1.110 | 0.49 |

The 2024 χ² is statistics-driven (≈ 8× the 2022 4b events); at the skim-consistent 30 GeV it is worse (214), so the
25-vs-30 GeV mismatch is not the cause. Caveat for 2023/2024 at 25 GeV: the picoAODs were skimmed at 30 GeV, so 3b
events with a tag at 25–30 GeV are missing (2023: 3b N[25,30)/N[30,35) = 0.38 vs 4b 0.96); the JCM absorbs the
normalisation, not the kinematics.

Files: `metadata/weights/JCM/Run3_ttHbb/jetCombinatoricModel_SB_run3_<era>[_pt25]_fNjetR_v1.yml`. `JCM_weights`
covers nSelJets 4..20. Training reads only `JCM_weights`, so **no new code is needed to train**. The fit options
(`f_njet_power`, `loose_veto_factor`) and the r-aware processor loader live on the LPC branch `ttHbb_jcm_fnjet`
(not in this branch); master's `analysis/helpers/jetCombinatoricModel.py` reads only p, e, d, t and would silently
apply the default model to these files, so Phase C.4 / F need that branch.

## 2. Inputs

| year | B.2 classifier inputs | trigger weights |
|---|---|---|
| 2022 | Matteo's `classifier_inputs_ttHbb_run3.json` (1040 entries, 30 GeV) | v1 |
| 2023 | **new**: `classifier_inputs_ttHbb_run3_2023_trigv1.json` (850 entries, 25 GeV), produced on lxplus with the `classifier_inputs` block of `workflows/config/analysis_ttHbb_run3_fNjetR.yml` | v1 (pinned via the Run 3 keys of `metadata/friends/friends_ttHbb.yml`; master moved to v3 on 2026-10-07; v1 keeps all three years and all JCM fits consistent) |
| 2024 | Matteo's 25 GeV index, cleaned: `classifier_inputs_ttHbb_run3_2024_clean_v1.json` | v1 |

- 2023 eras: `2023_preBPix: [C01, C02, C11, C12, C3, C4]`, `2023_BPix: [D1, D2]` (the older defaults omit C3/C4 = 78% of preBPix).
- A 1-chunk canary of that block showed the current code + v1 reproduces Matteo's 2022/2024
  HCR_input friends exactly (float64 last-bit differences only).
- **2024 index defect**: a chunk with 0 selected events returns before the friend dump (`processor_HH4b.py`), and the
  merge names friend files by picoAOD only (`src/runner/orchestrator.py`), so discontiguous ranges of one picoAOD
  overwrite one file. `Chunk.integrity()` flagged 19 stale sub-chunks (0.6% of 2024 data); `clean_v1` drops them
  (`.dropped.json` lists them). 31 picoAODs (1.04% of 2024 data) have friends not covering [0, N);
  `clean_v1_fullcov.json` (+ `.excluded.json`) is the evaluation-only index without them, and
  `metadata/datasets/archive/datasets_eval_run3_fullcov/` the matching dataset list for `evaluate.yml::--metadata` (evaluate takes
  its targets from the dataset metadata, so dropping files from the friend index alone does not skip them).

## 3. Trainings (Phase C, lxplus HTCondor GPUs)

| run | config | result |
|---|---|---|
| 2022 only, combined 2022 JCM | `analysis_ttHbb_run3_fNjetR.yml` (on the earlier branch; same mechanics) | SB closure (Σ d3·JCM·FvT + Σ t4)/Σ d4 = 1.0029, ≤ 1.2σ in every nSelJets bin 4–10; baseline FvT (default JCM): 1.0184, +4..+7σ at nSelJets 5–8 |
| 2024 only | (2024-only fvt block, not kept) | trained (70 GB RAM, 6h15m, H100); evaluation not completed (see §2 defect) |
| **combined 2022+2023+2024** | `analysis_ttHbb_run3_fNjetR.yml` | trained (102.4 M loaded events, 128 GB RSS, 9h22m) and evaluated (196 GB RSS) |

Combined config: 5 JCM groups (`--JCM-weight year:<era> <yml>@@JCM_weights`), 3 friend indices in one `--friends ""`,
`year` added to `feature_ancillary`, `kfold.Merge --workers 2`.

Validation AUC ('4b vs 3b data') per fold: 2022-only 0.625–0.630, 2024-only 0.640–0.645, combined 0.637–0.643.

## 4. Main result: per-era closure of the combined model (SB, all nSelJets)

| era | JCM-only closure ((d3−t3)·JCM + t4)/d4 | JCM × FvT closure | pull | 2022-only model |
|---|---|---|---|---|
| 2022_preEE | 0.9999 | **1.030** | +8.2σ | 0.985 |
| 2022_EE | 1.0000 | **1.018** | +9.3σ | 1.008 |
| 2023_preBPix | 0.9999 | **1.033** | +19.9σ | — |
| 2023_BPix | 1.0000 | **1.035** | +15.0σ | — |
| 2024 | 0.9990 | 1.004 | +5.9σ | — |

The per-era JCMs close each era exactly on their own. The single FvT over-predicts the 2022/2023 SB 4b yields by
2–3.5%, concentrated at nSelJets 5–6 (pull up to +23) with under-prediction at nSelJets ≥ 9: its ⟨FvT⟩-vs-nJet shape
follows the 2024 target curve (2024 = 75% of the training events), and the 3-valued `year` input (22/23/24) does
not separate the eras. Options under discussion: per-year models (the 2022-only one closes at +0.3%), a retrain
with a 5-valued era input (framework change in `add_year`), or a downstream per-era correction.

## 5. Running on lxplus: things that bit us (no code changed here)

- `src/data_formats/root/io.py` uses `$_CONDOR_SCRATCH_DIR` (then `TMPDIR`) as the temp dir for friend writes; the
  `lxplus_gpu` per-rule container does not bind it. Fix on the command line:
  `--apptainer-args "<profile args> --env TMPDIR=/tmp --env _CONDOR_SCRATCH_DIR=/tmp"`.
- Evaluate errors (friend dump/load) hang the job instead of failing it.
- `--- Logging error ---` from `logging.info(msg, <rich Table>)` when the Monitor is disabled (diagnostics lost only).
- The CERN schedd derives `RequestCpus` from the memory request (≈ mem/3 GB); memory limits were not enforced.
- `kfold.Merge --clean` deletes fragments one by one over xrootd (slow), and cannot run in an EOS area without
  delete rights (e.g. the phys_top group area): evaluate there, then migrate.
- Re-running Snakemake after an abort resumes the recorded external job id and burns the rule's retry;
  `--rerun-triggers input` avoids re-training.
