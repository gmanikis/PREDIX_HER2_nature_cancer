# PREDIX HER2 — reproducibility package (Nature Cancer revision)

This package lets a reviewer verify, on a laptop, every number and figure in the
revised manuscript of the PREDIX HER2 multimodal pCR-prediction study — from the
deposited model artefacts down to the exact cell values of the deposited tables.
The centrepiece is `PREDIX_HER2_reproducibility.ipynb`, an **executed** notebook
whose code cells recompute each deposited quantity and assert equality
(tolerance 1e-9 on point estimates and CI bounds).

## Folder map

| path | contents |
|---|---|
| `PREDIX_HER2_reproducibility.ipynb` | the executed verification notebook (outputs embedded) |
| `run_notebook.py` | re-executes the notebook headless (`python run_notebook.py`) |
| `code/` | the six analysis scripts (`multimodal_pcr_pipeline.py`, `generate_report.py`, `revision_analyses.py`, `external_validation.py`, `cv_estimands.py`, `preflight.py`), `tests/test_statistics.py`, `requirements.txt`, and the production run scripts (`production_run_ubuntu.sh`, `production_run.ps1`) |
| `data/` | the canonical PREDIX input file (SHA-256 verified against provenance) and the two external cohort files (I-SPY2, NCT02326974) |
| `results/` | production model artefacts: discovery + consensus PKLs per scenario, CV splits, `run_provenance.json`, `methods_cv_statement.txt`, per-classifier signature CSVs |
| `results_rna_ispy2/`, `results_rna_nct/` | the **arm-matched** RNA-only locked-model runs used by the external validation |
| `results_rna_pooled_ispy2/`, `results_rna_pooled_nct/` | the **pooled** RNA-only locked-model runs used by the external validation |
| `report/` | the **deposited** figures and tables the notebook reproduces (`report/tables/revision/` is the citable set) |
| `report_pooled_external/` | the deposited pooled-model external validation (`external_validation_POOLED.xlsx`, `revfig06_external_validation_POOLED.pdf`) |
| `environment/` | `requirements.txt`, `pip_freeze_windows.txt` (notebook machine), `production_environment.json` (model-run machine, transcribed from `results/run_provenance.json`) |
| `MANIFEST_SHA256.txt` | SHA-256 of every file (written by notebook Section 13) |
| `_regenerated/` | **scratch, not part of the deposit.** Notebook Section 11 and Section 12 create it on demand and write a full second copy of the report into it for the cell-by-cell comparison. It is excluded from `MANIFEST_SHA256.txt` and from the distributed archive, so a clean package does not contain it. Nothing needs it to pre-exist: the analysis scripts create it, and an empty one left behind by an earlier run is harmless. Delete it freely. |

## Two reproduction levels

**Level A — the post-modelling computation, exactly reproducible from the
deposited fold-level outputs (this notebook; ≈ 15 min).**
All randomness downstream of the model PKLs (the cluster bootstraps) is seeded
(base 20240517, deterministic CRC32 offset per quantity), so every table and
figure reproduces exactly from `results/`. Run `python run_notebook.py` or open
the notebook and *Run All*. Measured runtimes on the notebook machine
(Windows 11, Python 3.14, consumer laptop): **14.3 min end to end** — test suite
12 s; Section 5 bootstrap CIs 2.0 min; Section 7 paired comparisons 1.5 min;
Section 8 calibration 11 s; Section 11 locked external-validation re-runs
0.9 min (arm-matched) + 1.4 min (pooled); Section 12 full regeneration 3.8 min
(`generate_report.py`) + 4.1 min (`revision_analyses.py`) + the cell-by-cell
comparison of 179,350 workbook cells in 6 s. The executed notebook shipped here
reports its own total in the last cell.

**Level B — full pipeline re-run (many CPU-hours).**
`code/production_run_ubuntu.sh` drives the whole thing: step −1 `preflight.py`
gates the input file, step 0 runs the test suite, step 1 re-trains everything
(5-fold × 200 repeats global, 5-fold × 100 per arm; seed 42;
`--training_data expanded`), steps 2–3 regenerate the report, and steps 4a–4e
produce the shared feature lists, the four RNA-only locked runs (two
arm-matched, two pooled) and both locked external validations. CV partitions
are fully determined by seed 42, but
classifier-internal randomness of the tree models is deliberately **unseeded**
(seeding it would correlate the CV repeats and understate variance — see
`reproducibility_note` in `results/run_provenance.json`). A Level B re-run
therefore reproduces linear-model numbers exactly and tree-model numbers
statistically; the deposited PKLs in `results/` are the archival record, and
Level A reproduces every published quantity from them exactly.

**What has been demonstrated, stated precisely.** The **post-modelling
computation** — consensus aggregation, metric computation, the seeded
patient-level cluster bootstrap, signature ranking and fusion summarisation —
reproduces bit-identically from the deposited fold-level outputs, verified
across Linux → Windows and a much newer software stack (179,350 workbook cells,
zero mismatches; both external validations at |diff| = 0). This is **not** a
demonstration that re-fitting the models reproduces bit-identically. The
pipeline deliberately leaves classifier-internal randomness unseeded
(`random_state=None`), and its own `reproducibility_note` records that the last
digit of a per-fold metric may vary between runs. Solver defaults, tie-breaking
and RNG consumption also change between scikit-learn majors, and this analysis
reports selection frequencies over 1,000 folds, which are sensitive to exactly
that. Say "exactly reproducible from the deposited fold-level outputs"; never
"bit-reproducible end to end". Treat a re-run on a newer stack as a re-analysis,
and report it as one.

## Software

`pip install -r environment/requirements.txt`, which **pins the exact versions
that produced the published results** (numpy 1.26.4, pandas 2.3.3,
scikit-learn 1.7.2, scipy 1.10.0, shap 0.49.1, joblib 1.5.3 on Python 3.10.12);
matplotlib, openpyxl and threadpoolctl stay as lower bounds because they affect
rendering and file I/O, not any computed value. Add `pymupdf` for inline figure
rendering and `nbformat nbclient ipykernel` for headless execution. The
production model run
used Python 3.10.12 / numpy 1.26.4 / pandas 2.3.3 / scikit-learn 1.7.2 /
scipy 1.10.0 / shap 0.49.1 / joblib 1.5.3 on Ubuntu 6.8.0 — those versions are
transcribed verbatim from the `environment` block of
`results/run_provenance.json` into `environment/production_environment.json`,
so the two can never disagree. The notebook was executed on
Windows 11 / Python 3.14.7 / numpy 2.5.2 / pandas 3.0.5 / scikit-learn 1.9.0 /
scipy 1.18.0 (`environment/pip_freeze_windows.txt`) — the post-modelling results
are identical across the two environments, which is the cross-platform claim
this package supports and the only one it supports (see "What has been
demonstrated" above).

> An earlier version of `requirements.txt` carried `numpy>=2.0` and
> `scipy>=1.14` — bounds that **exclude** the versions production actually used
> (numpy 1.26.4, scipy 1.10.0), so anyone following them would have installed an
> environment the results were never produced in while the file claimed to
> describe the published run. It now pins the real versions, and
> `results/run_provenance.json` records them independently.

## The cohort and the feature panel

The input file is **197 patients × 112 columns** (`patientID`, `pCR`, and 110
curated features: Clin 5, RNA 42, DNA 41, Prot 19, WSI 3).

- **Evaluation cohort: n = 110, 46 pCR events (DHP 59/24, T-DM1 51/22).** These
  are the patients complete across RNA, DNA, proteomics and WSI. Clinical
  covariates are recorded for everyone and imputed in-fold, so `Clin_` does not
  enter the completeness rule (`get_complete_case()`). Every model — unimodal or
  fused — is scored on exactly these 110, which is what makes every comparison
  paired.
- **Training cohorts are wider** because fusion is at the prediction level:
  Clin 197/88 events, DNA 189/84, RNA 185/84, WSI 169/71, Prot 137/61. The 87
  modality-incomplete patients contribute to training and never to scoring.
- **Feature panel: 110 metrics.** The outcome-blind deduplication list
  `TIER1_REMOVE` names **21** features, of which **18** are present in this
  delivery and are therefore actually removed, leaving **92 candidate features**
  entering the in-fold univariate screen. The three that the list names but the
  file does not carry are `DNA_TMB_uniform`, `DNA_TMB_clone` and `DNA_pTMB`, so
  on this delivery the panel carries clonal oncogenic TMB only — never write
  "total TMB". Separately, `RNA_ADC_trafficking` was withdrawn upstream by the
  authors before round 2 and is absent from the file altogether; it is not on the
  `TIER1_REMOVE` list. `RNA_FCGR3B` is present in the file and removed by
  `TIER1_REMOVE`, not by the delivery.
- The locked consensus signature is aggregated over the outer folds won by the
  modal classifier (`SIGNATURE_SOURCE = "winner_folds"`), so the reported
  classifier and the reported signature are one model rather than two.

One trap worth stating once: `Clin_TUMSIZE` and `Clin_prolifvalu` encode missing
as the **string** `"Unknown"` rather than as `NaN`. Both encode to `NaN` and are
median-imputed per fold, so modelling is unaffected — but only 104 rows are
literally complete on all 110 columns, and a script that reaches for `dropna()`
across every feature column is not computing the pipeline's cohort. Notebook
Section 2 uses the RNA/DNA/Prot/WSI rule and checks the token count explicitly.

## What each notebook section verifies

| § | reproduces | checked against |
|---|---|---|
| 1 | environment, input-file SHA-256, provenance, canonical CV statement | `results/run_provenance.json`, `methods_cv_statement.txt` |
| 2 | cohort: 197×112 file; evaluation cohort n = 110 (46 events; DHP 59/24, T-DM1 51/22); per-modality training cohorts; 110-metric panel with `TIER1_REMOVE` 21 listed / 18 present → 92 candidates | data file, `code/multimodal_pcr_pipeline.py` |
| 3 | statistics test suite (vs scipy / sklearn / R references) | `code/tests/test_statistics.py` exit 0 |
| 4 | CV design from the artefacts: 5×200/100 repeats, every patient predicted once per repeat | `results/*.pkl` |
| 5 | headline AUROC/AUPRC/Brier + cluster-bootstrap 95% CIs, both sources | `revision_performance_CI.xlsx`, `PREDIX_HER2_results.xlsx`, fig01 |
| 6 | why the estimand: per-repeat mean vs ensemble-mean artefact vs per-fold mean | (didactic; artefact asserted) |
| 7 | paired fused-vs-unimodal Δ, CIs, bootstrap p, per-repeat DeLong, verdicts | `revision_model_comparisons.xlsx`, revfig07 |
| 8 | calibration slope, recalibration intercept, calibration-in-the-large, Brier, ECE + CIs | `revision_calibration.xlsx`, revfig01 |
| 9 | selection stability, fusion-weight stability, per-fold EPV (3 spot recomputations) | `revision_stability.xlsx`, `revision_epv_per_fold.xlsx`, revfig02/08/03 |
| 10 | consensus signatures (K, winner classifiers) | `PREDIX_HER2_results.xlsx` Signatures, fig02/fig05 |
| 11 | locked external validation, **arm-matched and pooled**: both deposited tables, internal comparators from all four locked runs, and both full script re-runs | `external_validation.xlsx`, `external_validation_POOLED.xlsx`, revfig06 (both) |
| 12 | **entire deposited report regenerated** from the PKLs and compared cell-by-cell, strictly (one declared and printed exception: the `locked_from` CLI-path string) | every workbook under `report/tables/` |
| 13 | package manifest | `MANIFEST_SHA256.txt` |

The exploratory S1/S2/S3 biomarker-group analysis that used to occupy section 11
was withdrawn from the revision; the section, its workbook and its figure are gone
and the later sections were renumbered accordingly.

## The estimand (statement of record)

AUROC/AUPRC/Brier = in each CV repeat every patient has exactly one out-of-fold
prediction; the metric is computed on that complete out-of-fold vector and
averaged over the repeats (200 global / 100 arm). 95% CI = patient-level CLUSTER
bootstrap: 2,000 stratified resamples of PATIENTS, a resampled patient carrying
all its repeat predictions. Predictions are never averaged across repeats or
models. Paired Δ: same patient resample applied to both models and all repeats
(primary); DeLong per repeat summarised (descriptive). Verdict "not
distinguishable" whenever the paired 95% CI includes 0. Previously reported "±"
= SD of per-fold AUROC = NOT a CI.

Multiplicity: the Benjamini–Hochberg correction on the paired comparisons is
applied across the family that is actually published — all **15** comparisons
within a `source` (3 scenarios × 5 comparators) — not per scenario. Correcting
inside each scenario would ship six independent families of five while
presenting 30 comparisons; the published family is the wider one, and widening
it only ever makes q larger. `revision_model_comparisons.xlsx` records the
family size in `BH_family_size` and keeps the narrower per-scenario values in
`q_within_call_BH_m5` for comparison.

Every bootstrap in the package draws its seed from one function,
`cv_estimands.shared_seed(tag) = 20240517 + crc32(tag) % 2**31`.
`generate_report.py`, `revision_analyses.py`, `external_validation.py` and the
notebook all call it, so the same quantity gets the same resample stream
wherever it is computed and re-running reproduces the CI endpoints exactly. The
offset uses the full crc32 range: the earlier `% 10000` compressed the ~126 tags
these scripts generate into 10,000 slots and produced three real collisions —
pairs of analyses silently sharing a resample stream. Each CI was still
individually valid, but their Monte-Carlo errors were correlated and
undocumented.

## External validation (statement of record)

Two locked transcriptomic models were pre-specified, frozen on PREDIX and
applied once to each external cohort — nothing refitted, with a SHA-256
provenance guard confirming that the locking run and the validation run saw the
same input file. Both are reported, whichever way they fall; neither was chosen
after seeing its result.

| model | I-SPY2 (GSE194040), n = 44 / 26 events | NCT02326974 (GSE243375), n = 129 / 64 events |
|---|---|---|
| arm-matched (DHP → I-SPY2, T-DM1 → NCT; K = 5) | **0.774 (0.622–0.904)**, P = 0.0015, slope 1.07 (0.54–1.89) | **0.644 (0.547–0.737)**, P = 0.0035, slope 0.70 (0.28–1.29) |
| pooled (all 185 RNA-carrying patients; K = 8) | **0.801 (0.664–0.915)**, P < 0.001, slope 1.40 (0.78–2.70) | **0.669 (0.576–0.753)**, P < 0.001, slope 0.91 (0.46–1.55) |

Values are the pre-specified primary (`zscore`) harmonisation; the `rank`
harmonisation is deposited alongside every row and agrees (I-SPY2 0.793 / 0.774,
NCT02326974 0.617 / 0.707). **Both external cohorts transfer**: all eight
deposited rows are above chance on the one-sided test, and the pooled model's
calibration slope covers 1 on both cohorts.

The pooled model is the better transfer to NCT02326974, for an identifiable
reason rather than a mysterious one: `RNA_HER2DX_HER2_amplicon` is selected in
essentially every pooled fold but in under 2% of T-DM1 arm folds, so the
arm-matched T-DM1 signature carries no HER2-amplicon term while the pooled
signature leads with it. Quote the two rows together — the arm-matched table
from `report/tables/revision/external_validation.xlsx` and the pooled table
from `report_pooled_external/tables/revision/external_validation_POOLED.xlsx`.
The `_POOLED` suffix exists because an earlier run wrote the two analyses under
identical basenames in different directories, which is easy to misread.

## Two workbooks deliberately differ from the production run directory

Everything under `report/`, `results/`, `results_rna_*/` and
`report_pooled_external/` is byte-identical to the production run that produced
it, **with two deliberate, documented exceptions**. Both were regenerated from
the same deposited PKLs with the corrected `code/generate_report.py`, verified
deterministic, and diffed cell by cell against the production copies. **No
numeric cell moved in either** — 634 numeric cells checked in the headline
workbook and 120 in the pruning report, all identical.

**`report/tables/supplementary/supp_PREDIX_HER2_feature_pruning_report.xlsx`.**
The production copy's `Methodology` sheet described a pipeline that no longer
exists, in four specific ways, each independently verified against the shipped
code and the deposited artefacts:

1. It documented the **per-fold Tier-3 high-correlation filter as an active
   stage**. That stage was deleted: `CORR_FILTER_MODS` is the empty set.
   Redundancy is removed once, before any split, and never inside a fold.
2. It **omitted the in-fold univariate outcome screen entirely** — the single
   most important leakage control in the revision, and the stage the original
   submission was criticised for not having. It runs in every fold, fitted on
   training patients only.
3. It described calibration as "applied when the inner-loop calibration slope
   falls outside [0.80, 1.20]". Platt recalibration is **unconditional**:
   verified applied in **10,000 of 10,000** scenario × modality × fold
   combinations, so the deposited description contradicted what ran.
4. It carried **no disclosure of the univariate screen's floor of five**.
   Across the 6,000 folds that ran the screen the floor overrode the FDR gate in
   **795**, and in every one of those 795 folds *zero* features reached
   q ≤ 0.25 and exactly five were retained regardless — T-DM1 DNA 246/500
   (49.2%), DHP DNA 198/500 (39.6%), T-DM1 proteomics 135/500 (27.0%), pooled
   DNA 144/1000 (14.4%), DHP RNA 53/500 (10.6%), DHP proteomics 17/500 (3.4%),
   T-DM1 RNA 2/500 (0.4%). The response letter discloses this in detail; the
   regenerated sheet now matches that disclosure.

Separately, its `TIER1_REMOVE` listing had been hand-copied into
`generate_report.py` and was stuck at an earlier set of 11, naming
`RNA_NK-cells`, which `TIER1_REMOVE` deliberately **retains**, and omitting
`DNA_CDK12_CNA`, `DNA_FADD_CNA`, the three `DNA_coding_mutation_*` entries,
`Prot_ERBB2`, `Prot_GRB7` and `RNA_FCGR3B`. Every threshold in the regenerated
sheet is now read from `code/multimodal_pcr_pipeline.py` at generation time, so
it cannot drift again. Changes are confined to Methods prose and two
`Pruning_Statistics` column labels.

**`report/tables/PREDIX_HER2_results.xlsx`.** Exactly one cell: the `Signatures`
column header `E1`, relabelled from "Mean |SHAP importance|" to "Mean selection
rank (not |SHAP|)". The column holds a normalised cross-classifier selection
rank, not a SHAP magnitude; the old label named it as something it is not.

**Why ship a divergence rather than the production file.** This paper was
rejected over how its methods were described. A supplementary Methods table that
documents a filter which was deleted, omits the leakage control which was added,
contradicts the calibration procedure that ran, and hides a screen floor the
response letter discloses, is worse for reviewers than a divergence recorded in
the open — especially when not one reported number changes. Left unplaced, the
response letter, the shipped code and the deposited supplementary table would
have disagreed in public on a Methods point. Notebook Section 12 regenerates
both workbooks from the deposited artefacts and compares them **strictly**, with
no sheet exempt, so the divergence is from the production *directory* only, never
between this package's code and this package's tables.

## Known caveats

- **Classifier randomness is unseeded by design**, so Level B reproduces the
  tree-model numbers statistically, not bit-for-bit; Level A (from the deposited
  PKLs) is exact.
- **Training is "expanded"; evaluation is complete-case.** The data file carries
  197 patients, of whom 110 are complete across RNA, DNA, proteomics and WSI.
  Because fusion is at the *prediction* level, each unimodal model is fitted on
  every patient carrying its own modality — Clin 197 (88 events), DNA 189 (84),
  RNA 185 (84), WSI 169 (71), Prot 137 (61) — giving median pCR events per
  pooled training fold of 79 / 75 / 75 / 62 / 52 respectively. Every model is
  nevertheless *scored* on the same 110 patients, so all paired comparisons
  remain paired. The one-cell check that expanded training actually engaged is
  `median_n_events_train` in `revision_epv_per_fold.xlsx`: values that differ
  across modalities mean expanded, values identical across modalities mean
  complete-case.
- **Every unimodal model now meets five events per variable; the fusion layer
  inside the arms does not.** `pct_folds_epv_below_5` is 0.0 for all fifteen
  scenario × modality unimodal models (`revision_epv_per_fold.xlsx`); median
  realised EPV runs 5.2–20.7. The remaining breach is the arm-level fusion
  layer — DHP 39.2% and T-DM1 56.0% of folds below 5 — and it is *structural*
  rather than fixable: the second-stage combiner needs all five modality
  predictions for a patient, so it is complete-case by construction and trains
  on ~19 (DHP) or ~18 (T-DM1) events. Arm-level *integrated* results should be
  read accordingly; the pooled fusion layer is unaffected (0% of folds below 5).
- **The weakest models are the arm-level WSI and DNA models**, not any model
  below chance: the lowest deposited AUROC is T-DM1 WSI 0.541 (95% CI
  0.431–0.649), whose interval covers 0.5. No consensus model in the deposited
  table has a point estimate below chance.
- The `locked_from` column of `external_validation.xlsx` records the `--locked_*`
  path as typed on the production command line; re-runs from this package record
  the package-relative path instead. Printed in the Section 12 comparison, not
  skipped silently.
- Fixed during preparation of this package, recorded here because it shows what
  Section 13 is for: the `Feature_selection_stability` sheet of
  `revision_stability.xlsx` used to come out in a different **row order** on
  every run. `selection_frequency()` accumulated rows in the iteration order of
  a Python `set` of feature names — which varies with the per-process string
  hash seed — and the sort over selection frequency was not stable, so features
  tied at the same frequency permuted between runs. No value was ever affected
  (the two sheets were identical as a multiset of rows). The function now sorts
  by `(selection_freq_eligible, selection_freq, feature)` with a stable sort,
  `tests/test_statistics.py` checks that order, and Section 13 compares every
  sheet in its natural row order with no special handling.

## How to cite numbers

Quote numbers **only** from `report/tables/revision/` (and
`report/tables/PREDIX_HER2_results.xlsx` for the headline consensus table) — the
patient-level cluster-bootstrap set. The one exception is the pooled external
validation, which lives in
`report_pooled_external/tables/revision/external_validation_POOLED.xlsx` and
must always be quoted together with the arm-matched table. Discovery-phase
diagnostics under `report/tables/supplementary/` are per-fold descriptive
values, and any "±" found there is a fold SD, not a confidence interval.
