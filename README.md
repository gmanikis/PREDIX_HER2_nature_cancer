# PREDIX HER2 — multimodal prediction of pathological complete response

Analysis code, deposited cross-validation artefacts, and an executed
verification notebook for the multimodal pCR-prediction analysis of the
PREDIX HER2 randomised trial (clinical, transcriptomic, genomic, proteomic and
whole-slide-image data).

Everything reported can be re-derived from this repository. The centrepiece is
**`PREDIX_HER2_reproducibility.ipynb`**, shipped executed: it recomputes each
deposited quantity from the model artefacts, asserts equality, and finally
re-runs both post-processing scripts end to end and compares every regenerated
workbook with its deposited counterpart cell by cell.

The notebook prints its own totals — checks passed, workbook cells compared,
mismatches, figures regenerated — and this tree ships the executed copy, so read
the count there rather than here. (Run-5 rebuild: the earlier README hard-coded
"36 / 36 checks passed, 180,749 workbook cells compared, all 31 figures" from the
run-3 execution. Those numbers are not carried forward; re-execute
`run_notebook.py` against this tree and let its stored output be the record.)

## Quick start

```bash
pip install -r requirements.txt

python code/tests/test_statistics.py     # ~190 statistical checks, ~25 s
python run_notebook.py                   # re-executes the whole verification, ~20-40 min
```

Run both from the repository root — the notebook asserts that `code/` and
`results/` are in the working directory. To read the results without running
anything, open `PREDIX_HER2_reproducibility.ipynb`: the outputs are stored.

**[Read all the results here](RESULTS.md)** — every table and every figure, on
one page, generated from the deposited workbooks.

## Headline results

Cross-validated AUROC (95 % patient-level cluster-bootstrap CI), consensus
models. Source: `report/tables/revision/revision_performance_CI.xlsx`.

| Model | Pooled (n = 110, 46 pCR) | DHP (n = 59, 24) | T-DM1 (n = 51, 22) |
|---|---|---|---|
| Clinical | 0.61 (0.52–0.71) | 0.56 (0.44–0.69) | 0.58 (0.48–0.69) |
| Transcriptomic | 0.76 (0.67–0.84) | 0.80 (0.68–0.90) | 0.74 (0.59–0.87) |
| Genomic | 0.61 (0.54–0.69) | 0.68 (0.56–0.80) | 0.57 (0.49–0.65) |
| Proteomic | 0.74 (0.66–0.83) | 0.82 (0.71–0.91) | 0.68 (0.54–0.81) |
| Whole-slide image | 0.59 (0.49–0.70) | 0.60 (0.47–0.73) | 0.54 (0.43–0.65) |
| **Integrated (late fusion)** | **0.77 (0.69–0.85)** | **0.79 (0.69–0.89)** | **0.69 (0.57–0.80)** |

Integration is **not distinguishable** from the best single modality in any
scenario (pooled ΔAUROC vs transcriptomic +0.01, 95 % CI −0.03 to 0.05,
P = 0.62); in the pooled cohort it is higher than the clinical, genomic and WSI
models.

External validation of the locked transcriptomic models — **both cohorts
transfer, under both designs.** Arm-matched models
(`report/tables/revision/external_validation.xlsx`): I-SPY2 (GSE194040) AUROC
0.77 (0.62–0.90), calibration slope 1.07, P = 0.001; NCT02326974 (GSE243375)
AUROC 0.64 (0.55–0.74), slope 0.70, P = 0.003. Pooled model, refit on all 185
PREDIX patients carrying transcriptomics irrespective of arm
(`report_pooled_external/tables/revision/external_validation_POOLED.xlsx`):
I-SPY2 0.80 (0.66–0.92) and NCT02326974 0.67 (0.58–0.75), both P < 0.001.
Figures quoted are the z-score harmonisation; the rank scheme is reported beside
it in each workbook and agrees. Both analyses were pre-specified and must be
read together — neither replaces the other.

## Repository map

| path | contents |
|---|---|
| [`RESULTS.md`](RESULTS.md) | **every table and figure of the analysis, rendered for reading here** — generated from the workbooks, nothing typed by hand |
| `PREDIX_HER2_reproducibility.ipynb` | the executed verification notebook (outputs embedded) |
| `run_notebook.py` | re-executes the notebook headless |
| `code/` | the five analysis scripts, the test suite, and the production run scripts |
| `data/` | the PREDIX input file and the two external cohort files |
| `results/` | production model artefacts: discovery and consensus PKLs per scenario, CV splits, `run_provenance.json`, `methods_cv_statement.txt` |
| `results_rna_ispy2/`, `results_rna_nct/` | the RNA-only locked-model runs behind the **arm-matched** external validation |
| `results_rna_pooled_ispy2/`, `results_rna_pooled_nct/` | the same, for the **pooled** model refit on all transcriptomic patients irrespective of arm (added in run 5) |
| `report/figures/`, `report/tables/` | the deposited figures and tables the notebook regenerates |
| `report_pooled_external/` | the pooled-model external validation, kept in its own tree: `external_validation_POOLED.xlsx` and `revfig06_external_validation_POOLED.pdf` (added in run 5, after run 4 shipped the two external analyses under identical basenames) |
| `supplementary/` | the candidate feature panel (Table S-ML8) and the script that builds it |
| `docs/` | reproducibility guide, candidate-feature curation, the cross-validation statement and schematic |
| `environment/` | pinned requirements and the two environment records |
| `MANIFEST_SHA256.txt` | SHA-256 of every file in the repository |

## The analysis in one page

`docs/ED_Fig11a_CV_schematic.pdf` draws the whole design; it is generated from
`results/run_provenance.json` by `docs/build_ED_Fig11a_schematic.py`, so it
cannot drift away from what was run.

Pipeline order (`code/production_run_ubuntu.sh` drives all four steps):

1. `multimodal_pcr_pipeline.py` — trains the models, writes the PKLs in `results/`
2. `generate_report.py` — figures and tables
3. `revision_analyses.py` — confidence intervals, calibration, stability, EPV
4. `external_validation.py` — the locked-model validation in I-SPY2 and NCT02326974,
   run twice: arm-matched into `report/`, pooled into `report_pooled_external/`

`cv_estimands.py` is imported by steps 2–4 and is the single definition of the
performance estimand.

## The estimand (statement of record)

In each cross-validation repeat every patient has exactly one out-of-fold
prediction; AUROC, AUPRC and the Brier score are computed on that complete
out-of-fold vector and averaged over the repeats (200 pooled, 100 per arm). The
95 % interval is a **patient-level cluster bootstrap** — 2,000 stratified
resamples of patients, a resampled patient carrying all of its repeat
predictions. Paired comparisons use the same patient resample for both models
and all repeats. **Predictions are never averaged across repeats or across
models**: doing so scores a 200-model ensemble instead of the model, and is
badly biased for weak models.

Any "±" value in an earlier version of this work is a standard deviation of
per-fold AUROC, not a confidence interval.

## Two levels of reproduction

**Level A — post-processing, bit-for-bit (the notebook).** Everything downstream
of the model artefacts is seeded, so all tables and figures reproduce exactly
from `results/`.

**Level B — full pipeline re-run (many CPU-hours).** Step 1 of
`code/production_run_ubuntu.sh` re-trains everything (5-fold × 200 repeats
pooled, 5-fold × 100 per arm, seed 42). Cross-validation partitions are fully
determined by the seed, but classifier-internal randomness of the tree models is
deliberately **not** seeded — seeding it would correlate the repeats and
understate variance (see `reproducibility_note` in `results/run_provenance.json`).
Level B therefore reproduces linear-model numbers exactly and tree-model numbers
statistically; the deposited PKLs are the archival record that Level A verifies
exactly.

See `docs/REPRODUCIBILITY.md` for the section-by-section account of what the
notebook checks, measured runtimes, and the known caveats.

## Feature selection and leakage

The candidate panel and the leakage-free in-fold screen are documented in
`docs/CANDIDATE_FEATURE_CURATION.md`, with the complete 110-metric panel in
`supplementary/S-ML8_candidate_panel.xlsx`. In short: the a-priori biological
curation uses no outcome; the univariate association step, which in the original
submission had been applied across the whole cohort, is now performed inside
every training fold.

## Software

Python ≥ 3.10, `pip install -r requirements.txt`. The production model run used
Python 3.10.12 / numpy 1.26.4 / pandas 2.3.3 / scikit-learn 1.7.2 / scipy 1.10.0
/ shap 0.49.1 on Ubuntu — the run-5 versions, recorded verbatim under
`environment` in `results/run_provenance.json` (the run-3 README said numpy
2.2.6 / scipy 1.15.3). The notebook was executed on Windows 11 / Python 3.14.7 /
numpy 2.5.2 / pandas 3.0.5 / scikit-learn 1.9.0
(`environment/pip_freeze_windows.txt`). Post-processing results are identical
across the two environments.

## Repository size

≈ 155 MB, dominated by the deposited PKLs; the largest single file is ≈ 32 MB, so
a plain `git push` works and Git LFS is not required. (Run 5: was ≈ 95 MB. The
two pooled-model RNA result directories added in run 5 carry ≈ 23 MB each.) If
you prefer LFS, track `*.pkl` **before** the first commit:

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

## Verifying integrity

```bash
python - <<'PY'
import hashlib, pathlib
root = pathlib.Path('.')
bad = []
for line in (root / 'MANIFEST_SHA256.txt').read_text().splitlines():
    if not line.strip() or line.startswith('#'):
        continue
    digest, rel = line.split(None, 1)
    p = root / rel.strip()
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'MISSING'
    if h != digest:
        bad.append(rel.strip())
print('mismatched or missing:', bad or 'none')
PY
```

## Before making this repository public

- [ ] Confirm that ethics approval and patient consent permit releasing
      `data/clin_multiomics_curated_metrics_PREDIX_HER2_new.txt` (197 patients).
- [ ] Confirm the redistribution terms of the two external cohort files
      (GEO GSE194040 and GSE243375) and add attribution.
- [ ] Choose and write `LICENSE` (see the placeholder for the usual arrangement).
- [ ] Complete `CITATION.cff` and add the release DOI (Zenodo) to the article's
      Code availability statement.

## Known caveats

- Completeness is defined on the molecular modalities only — the clinical block
  never enters it — so the complete-case cohort is **n = 110** (46 pCR; DHP
  59/24, T-DM1 51/22), matching the submitted manuscript. (Run 5 closed the
  earlier n = 109 / DHP 58 discrepancy: it was an artefact of applying `dropna()`
  across all features, which the `Unknown` tokens in `Clin_TUMSIZE` and
  `Clin_prolifvalu` turn into six spurious drops. A naive `dropna()` over the
  whole table still returns 104 rows; use the pipeline's rule.)
- Models are trained on the **expanded** cohort — every patient carrying the
  modality, 197 for clinical down to 137 for proteomics — and evaluated only on
  the 110 complete cases, so the two counts differ by design and neither is a
  typo.
- The `locked_from` column of `external_validation.xlsx` records the command-line
  path as typed in the production run; a re-run from this repository records the
  local path instead. The notebook prints this difference rather than skipping it.

Generated 2026-08-21 from pipeline version 2.0.0-revision1, seed 42.
