# Results

Every table on this page is generated directly from the deposited workbooks under [`report/tables/`](report/tables) by [`docs/build_RESULTS_md.py`](docs/build_RESULTS_md.py), and every figure is the PNG rendering of the corresponding PDF in [`report/figures/`](report/figures). Nothing here is typed by hand.

Pipeline `2.0.0-revision1`, seed 42. Regenerated 2026-08-21.

> **How to read every number below.** In each cross-validation repeat every patient has exactly one out-of-fold prediction; the metric is computed on that complete out-of-fold vector and averaged over the repeats (200 pooled, 100 per arm). The 95% interval is a patient-level **cluster** bootstrap — 2,000 stratified resamples of patients, a resampled patient carrying all of its repeat predictions. Predictions are never averaged across repeats or across models. A comparison whose interval for ΔAUROC includes zero is reported as *not distinguishable*, however large the point difference.

## Contents

1. [Design and cohort](#1-design-and-cohort)
2. [Cross-validated performance](#2-cross-validated-performance)
3. [Is integration better than the best single modality?](#3-is-integration-better-than-the-best-single-modality)
4. [Calibration](#4-calibration)
5. [Events per variable](#5-events-per-variable)
6. [Feature-selection stability](#6-feature-selection-stability)
7. [Consensus signatures and fusion weights](#7-consensus-signatures-and-fusion-weights)
8. [External validation](#8-external-validation)
9. [Figures](#9-figures)

## 1. Design and cohort

| Cohort | Patients | pCR events | pCR rate | CV repeats | Outer evaluations |
|---|---|---|---|---|---|
| Pooled cohort | 110 | 46 | 41.8% | 200 | 1,000 |
| DHP arm | 59 | 24 | 40.7% | 100 | 500 |
| T-DM1 arm | 51 | 22 | 43.1% | 100 | 500 |

| Design element | Value |
|---|---|
| Outer resampling | stratified 5-fold `RepeatedStratifiedKFold` (no shuffle-split) |
| Inner resampling | 5-fold (pooled), 3-fold (per arm) |
| Candidate panel | 110 pre-defined metrics → 92 after the outcome-blind biological deduplication |
| Feature screen | in-fold Mann–Whitney AUROC, BH q ≤ 0.25, keep 5–40 |
| Classifier families | `ElasticNet_LR`, `RandomForest`, `ExtraTrees`, `HistGradBoost`, `SVM_Linear` |
| Signature size cap | at least 5 pCR events per selected variable |
| Fusion | elastic-net logistic regression (l1_ratio 0.5) over the five Platt-calibrated modality probability streams |
| Consensus finalisation | features above the stability threshold (0.6 pooled, 0.5 per arm); modal classifier |
| Signature aggregation | `winner_folds` — aggregated only over the outer folds the modal classifier won, so the reported classifier and signature are one model |
| Training cohort | `expanded` — each modality trains on every patient carrying it; evaluation is on the complete cases only |
| Random seed | 42 |

The full design is drawn in [`docs/ED_Fig11a_CV_schematic.pdf`](docs/ED_Fig11a_CV_schematic.pdf) and stated in [`docs/methods_cv_statement.txt`](docs/methods_cv_statement.txt), both generated from the run's own parameters.

## 2. Cross-validated performance

Consensus models — the frozen signature and classifier re-evaluated on the same outer splits. Source: `report/tables/revision/revision_performance_CI.xlsx`.

### Pooled cohort

| Model | AUROC | 95% CI | AUPRC | 95% CI | Brier | 95% CI |
|---|---|---|---|---|---|---|
| Clinical | 0.612 | 0.519–0.710 | 0.530 | 0.457–0.653 | 0.225 | 0.197–0.253 |
| Transcriptomic | 0.759 | 0.672–0.837 | 0.701 | 0.605–0.807 | 0.194 | 0.165–0.226 |
| Genomic | 0.615 | 0.538–0.692 | 0.528 | 0.470–0.621 | 0.236 | 0.222–0.250 |
| Proteomic | 0.744 | 0.656–0.831 | 0.637 | 0.549–0.760 | 0.202 | 0.170–0.236 |
| Whole-slide image | 0.590 | 0.486–0.697 | 0.566 | 0.469–0.669 | 0.237 | 0.220–0.255 |
| **Integrated (late fusion)** | **0.771** | 0.691–0.845 | 0.696 | 0.610–0.798 | 0.193 | 0.167–0.222 |

### DHP arm

| Model | AUROC | 95% CI | AUPRC | 95% CI | Brier | 95% CI |
|---|---|---|---|---|---|---|
| Clinical | 0.562 | 0.439–0.689 | 0.471 | 0.392–0.623 | 0.245 | 0.223–0.269 |
| Transcriptomic | 0.796 | 0.682–0.898 | 0.702 | 0.565–0.866 | 0.186 | 0.135–0.241 |
| Genomic | 0.685 | 0.564–0.804 | 0.570 | 0.479–0.714 | 0.221 | 0.190–0.254 |
| Proteomic | 0.816 | 0.706–0.908 | 0.699 | 0.586–0.867 | 0.165 | 0.120–0.217 |
| Whole-slide image | 0.605 | 0.475–0.727 | 0.509 | 0.421–0.666 | 0.238 | 0.209–0.269 |
| **Integrated (late fusion)** | **0.795** | 0.687–0.891 | 0.678 | 0.571–0.826 | 0.185 | 0.153–0.223 |

### T-DM1 arm

| Model | AUROC | 95% CI | AUPRC | 95% CI | Brier | 95% CI |
|---|---|---|---|---|---|---|
| Clinical | 0.583 | 0.477–0.690 | 0.509 | 0.436–0.657 | 0.256 | 0.222–0.293 |
| Transcriptomic | 0.737 | 0.594–0.869 | 0.690 | 0.565–0.854 | 0.206 | 0.162–0.261 |
| Genomic | 0.573 | 0.494–0.647 | 0.538 | 0.484–0.636 | 0.253 | 0.232–0.277 |
| Proteomic | 0.681 | 0.541–0.809 | 0.675 | 0.574–0.799 | 0.224 | 0.187–0.265 |
| Whole-slide image | 0.541 | 0.431–0.649 | 0.505 | 0.435–0.642 | 0.255 | 0.232–0.281 |
| **Integrated (late fusion)** | **0.694** | 0.572–0.801 | 0.647 | 0.551–0.779 | 0.221 | 0.191–0.254 |

### Discovery phase (fully nested)

The signature and classifier are re-selected independently inside every fold, so these estimates carry no consensus selection optimism. They are the conservative reading of the same data.

| Cohort | Model | Discovery AUROC | 95% CI | Consensus − discovery |
|---|---|---|---|---|
| Pooled cohort | Clinical | 0.604 | 0.506–0.699 | +0.007 |
| Pooled cohort | Transcriptomic | 0.744 | 0.655–0.828 | +0.015 |
| Pooled cohort | Genomic | 0.558 | 0.487–0.629 | +0.057 |
| Pooled cohort | Proteomic | 0.691 | 0.601–0.777 | +0.053 |
| Pooled cohort | Whole-slide image | 0.571 | 0.483–0.657 | +0.019 |
| Pooled cohort | Integrated (late fusion) | 0.737 | 0.657–0.812 | +0.034 |
| DHP arm | Clinical | 0.541 | 0.423–0.655 | +0.021 |
| DHP arm | Transcriptomic | 0.768 | 0.646–0.873 | +0.028 |
| DHP arm | Genomic | 0.638 | 0.529–0.744 | +0.047 |
| DHP arm | Proteomic | 0.790 | 0.689–0.884 | +0.026 |
| DHP arm | Whole-slide image | 0.592 | 0.470–0.717 | +0.013 |
| DHP arm | Integrated (late fusion) | 0.765 | 0.648–0.863 | +0.030 |
| T-DM1 arm | Clinical | 0.621 | 0.501–0.734 | -0.038 |
| T-DM1 arm | Transcriptomic | 0.687 | 0.556–0.815 | +0.051 |
| T-DM1 arm | Genomic | 0.614 | 0.499–0.721 | -0.041 |
| T-DM1 arm | Proteomic | 0.644 | 0.503–0.778 | +0.037 |
| T-DM1 arm | Whole-slide image | 0.518 | 0.406–0.635 | +0.023 |
| T-DM1 arm | Integrated (late fusion) | 0.660 | 0.549–0.762 | +0.034 |

## 3. Is integration better than the best single modality?

**No.** Paired patient-level cluster bootstrap, the same patient resample applied to both models and all repeats.

| Cohort | Integrated AUROC | Best single modality | Δ AUROC | 95% CI | P | Verdict |
|---|---|---|---|---|---|---|
| Pooled cohort | 0.771 | Transcriptomic 0.759 | +0.012 | -0.029 to 0.054 | 0.622 | not distinguishable |
| DHP arm | 0.795 | Proteomic 0.816 | -0.022 | -0.069 to 0.029 | 0.405 | not distinguishable |
| T-DM1 arm | 0.694 | Transcriptomic 0.737 | -0.044 | -0.107 to 0.018 | 0.176 | not distinguishable |

Against every comparator:

| Cohort | Integrated vs | Δ AUROC | 95% CI | P | BH q | Verdict |
|---|---|---|---|---|---|---|
| Pooled cohort | Clinical | +0.159 | 0.061–0.253 | 0.002 | 0.015 | integrated higher |
| Pooled cohort | Transcriptomic | +0.012 | -0.029 to 0.054 | 0.622 | 0.718 | not distinguishable |
| Pooled cohort | Genomic | +0.156 | 0.065–0.243 | 0.002 | 0.015 | integrated higher |
| Pooled cohort | Proteomic | +0.027 | -0.035 to 0.093 | 0.394 | 0.506 | not distinguishable |
| Pooled cohort | Whole-slide image | +0.181 | 0.063–0.299 | 0.004 | 0.015 | integrated higher |
| DHP arm | Clinical | +0.232 | 0.091–0.379 | 0.004 | 0.015 | integrated higher |
| DHP arm | Transcriptomic | -0.002 | -0.060 to 0.064 | 0.925 | 0.925 | not distinguishable |
| DHP arm | Genomic | +0.110 | 0.019–0.200 | 0.021 | 0.063 | integrated higher |
| DHP arm | Proteomic | -0.022 | -0.069 to 0.029 | 0.405 | 0.506 | not distinguishable |
| DHP arm | Whole-slide image | +0.190 | 0.029–0.344 | 0.026 | 0.065 | integrated higher |
| T-DM1 arm | Clinical | +0.111 | -0.018 to 0.233 | 0.093 | 0.155 | not distinguishable |
| T-DM1 arm | Transcriptomic | -0.044 | -0.107 to 0.018 | 0.176 | 0.264 | not distinguishable |
| T-DM1 arm | Genomic | +0.121 | -0.012 to 0.244 | 0.073 | 0.137 | not distinguishable |
| T-DM1 arm | Proteomic | +0.013 | -0.107 to 0.134 | 0.829 | 0.888 | not distinguishable |
| T-DM1 arm | Whole-slide image | +0.153 | 0.013–0.300 | 0.031 | 0.066 | integrated higher |

DeLong's test computed per repeat and summarised is reported in the workbook as a descriptive secondary analysis; the bootstrap is the primary comparison.

## 4. Calibration

Slope and intercept of `logit(pCR) = a + b · logit(p̂)`, fitted on each repeat's out-of-fold vector and averaged. Slope 1 and intercept 0 are perfect; slope below 1 means the predictions are too extreme (the classic overfitting signature), above 1 that they are compressed toward the base rate.

| Cohort | Slope | 95% CI | Intercept | 95% CI | Brier | ECE | Observed vs mean predicted |
|---|---|---|---|---|---|---|---|
| Pooled cohort | 1.15 | 0.77–1.72 | 0.08 | -0.11 to 0.36 | 0.193 | 0.105 | 0.418 vs 0.411 |
| DHP arm | 1.45 | 0.83–2.62 | 0.14 | -0.15 to 0.64 | 0.185 | 0.125 | 0.407 vs 0.404 |
| T-DM1 arm | 0.92 | 0.38–1.91 | 0.02 | -0.17 to 0.39 | 0.221 | 0.121 | 0.431 vs 0.422 |

Every slope interval covers 1 and every intercept interval covers 0.

<details><summary>Reliability bins (equal-count bins over all (patient, repeat) out-of-fold predictions)</summary>

| Cohort | Bin | Predictions | Distinct patients | Mean predicted | Observed | 95% CI |
|---|---|---|---|---|---|---|
| Pooled cohort | 1 | 2,200 | 47 | 0.093 | 0.047 | 0.012–0.113 |
| Pooled cohort | 2 | 2,200 | 62 | 0.178 | 0.165 | 0.071–0.274 |
| Pooled cohort | 3 | 2,200 | 73 | 0.246 | 0.222 | 0.127–0.312 |
| Pooled cohort | 4 | 2,200 | 85 | 0.305 | 0.296 | 0.203–0.397 |
| Pooled cohort | 5 | 2,200 | 82 | 0.363 | 0.399 | 0.293–0.501 |
| Pooled cohort | 6 | 2,200 | 86 | 0.420 | 0.466 | 0.366–0.559 |
| Pooled cohort | 7 | 2,200 | 77 | 0.486 | 0.517 | 0.421–0.612 |
| Pooled cohort | 8 | 2,200 | 74 | 0.564 | 0.593 | 0.489–0.718 |
| Pooled cohort | 9 | 2,200 | 61 | 0.659 | 0.688 | 0.570–0.800 |
| Pooled cohort | 10 | 2,200 | 45 | 0.800 | 0.789 | 0.625–0.930 |
| DHP arm | 1 | 843 | 25 | 0.107 | 0.065 | 0.000–0.180 |
| DHP arm | 2 | 843 | 32 | 0.221 | 0.087 | 0.007–0.203 |
| DHP arm | 3 | 843 | 40 | 0.298 | 0.197 | 0.052–0.508 |
| DHP arm | 4 | 843 | 34 | 0.422 | 0.571 | 0.351–0.711 |
| DHP arm | 5 | 843 | 35 | 0.497 | 0.575 | 0.457–0.721 |
| DHP arm | 6 | 843 | 35 | 0.572 | 0.681 | 0.541–0.795 |
| DHP arm | 7 | 842 | 31 | 0.709 | 0.672 | 0.489–0.852 |
| T-DM1 arm | 1 | 850 | 39 | 0.172 | 0.240 | 0.103–0.378 |
| T-DM1 arm | 2 | 850 | 46 | 0.305 | 0.306 | 0.196–0.411 |
| T-DM1 arm | 3 | 850 | 48 | 0.377 | 0.305 | 0.235–0.395 |
| T-DM1 arm | 4 | 850 | 49 | 0.437 | 0.459 | 0.345–0.575 |
| T-DM1 arm | 5 | 850 | 46 | 0.526 | 0.599 | 0.478–0.715 |
| T-DM1 arm | 6 | 850 | 37 | 0.715 | 0.680 | 0.527–0.822 |

</details>

## 5. Events per variable

The design caps signature size at five pCR events per selected variable. This table reports what was actually realised in each fold.

| Cohort | Model | Folds | Test-fold events (median, range) | Median signature size | Median EPV | Min EPV | Folds below EPV 5 |
|---|---|---|---|---|---|---|---|
| DHP arm | Clinical | 500 | 5 (4–5) | 4 | 10.00 | 10.00 | 0.0% |
| DHP arm | Genomic | 500 | 5 (4–5) | 5 | 7.80 | 5.00 | 0.0% |
| DHP arm | Integrated (late fusion) | 500 | 5 (4–5) | 3 | 6.33 | 3.80 | 39.2% |
| DHP arm | Proteomic | 500 | 5 (4–5) | 5 | 5.20 | 5.20 | 0.0% |
| DHP arm | Transcriptomic | 500 | 5 (4–5) | 5 | 7.80 | 6.50 | 0.0% |
| DHP arm | Whole-slide image | 500 | 5 (4–5) | 3 | 10.33 | 10.33 | 0.0% |
| Pooled cohort | Clinical | 1,000 | 9 (9–10) | 5 | 15.80 | 15.60 | 0.0% |
| Pooled cohort | Genomic | 1,000 | 9 (9–10) | 7 | 10.57 | 6.82 | 0.0% |
| Pooled cohort | Integrated (late fusion) | 1,000 | 9 (9–10) | 4 | 9.00 | 7.20 | 0.0% |
| Pooled cohort | Proteomic | 1,000 | 9 (9–10) | 7 | 7.43 | 6.50 | 0.0% |
| Pooled cohort | Transcriptomic | 1,000 | 9 (9–10) | 9 | 8.33 | 6.25 | 0.0% |
| Pooled cohort | Whole-slide image | 1,000 | 9 (9–10) | 3 | 20.67 | 20.33 | 0.0% |
| T-DM1 arm | Clinical | 500 | 4 (4–5) | 4 | 9.75 | 9.50 | 0.0% |
| T-DM1 arm | Genomic | 500 | 4 (4–5) | 4 | 8.75 | 5.00 | 0.0% |
| T-DM1 arm | Integrated (late fusion) | 500 | 4 (4–5) | 4 | 4.50 | 3.40 | 56.0% |
| T-DM1 arm | Proteomic | 500 | 4 (4–5) | 5 | 5.20 | 5.00 | 0.0% |
| T-DM1 arm | Transcriptomic | 500 | 4 (4–5) | 5 | 7.20 | 5.00 | 0.0% |
| T-DM1 arm | Whole-slide image | 500 | 4 (4–5) | 3 | 10.33 | 10.00 | 0.0% |

The arm-level fusion layer is the most exposed component: it takes five modality inputs by design and cannot be capped, so 39% of DHP folds and 56% of T-DM1 folds run below five events per variable. Every single-modality model, in every scenario, stays at or above the cap in every fold.

## 6. Feature-selection stability

How often each candidate feature was selected across the outer folds. Features above the pre-specified threshold (0.6 pooled, 0.5 per arm) are the consensus signature; 88 of 172 candidate rows clear it.

**The threshold is applied to the *eligible-fold* frequency** — the fraction of the folds in which the feature survived preprocessing and the in-fold screen at all. A feature can therefore be stable on that denominator while its all-fold frequency is low: it was rarely eligible, but was chosen almost whenever it was. Both columns are given below, with the Wilson interval on the eligible-fold proportion.

<details><summary><b>Pooled cohort</b> — 40 stable features</summary>

| Modality | Feature | All-fold frequency | Eligible-fold frequency | 95% Wilson CI | Folds selected |
|---|---|---|---|---|---|
| Clinical | `Clin_ANYNODES` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_Arm` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_ER` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_TUMSIZE` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Clinical | `Clin_prolifvalu` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.13` | 0.999 | 1.000 | 0.996–1.000 | 999 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.6` | 0.990 | 1.000 | 0.996–1.000 | 990 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.7` | 0.919 | 0.979 | 0.967–0.986 | 919 / 1,000 |
| Genomic | `DNA_PIK3CA_CNA` | 0.568 | 0.955 | 0.935–0.969 | 568 / 1,000 |
| Genomic | `DNA_meanHED` | 0.145 | 0.954 | 0.908–0.978 | 145 / 1,000 |
| Genomic | `DNA_ERBB2_CNA` | 0.656 | 0.874 | 0.848–0.895 | 656 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.2` | 0.817 | 0.864 | 0.840–0.884 | 817 / 1,000 |
| Genomic | `DNA_TCRA.tcell.fraction.adj` | 0.249 | 0.819 | 0.772–0.858 | 249 / 1,000 |
| Genomic | `DNA_coding_mutation_TP53` | 0.637 | 0.756 | 0.726–0.783 | 637 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.3` | 0.328 | 0.642 | 0.599–0.682 | 328 / 1,000 |
| Genomic | `DNA_TMB_clone_oncogenic` | 0.362 | 0.633 | 0.593–0.671 | 362 / 1,000 |
| Genomic | `DNA_COSMIC.Signature.10` | 0.306 | 0.627 | 0.583–0.669 | 306 / 1,000 |
| Genomic | `DNA_MED1_CNA` | 0.193 | 0.613 | 0.558–0.665 | 193 / 1,000 |
| Proteomic | `Prot_HER2_amplicon` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Proteomic | `Prot_RPL19` | 0.999 | 0.999 | 0.994–1.000 | 999 / 1,000 |
| Proteomic | `Prot_CDK12` | 0.920 | 0.920 | 0.902–0.935 | 920 / 1,000 |
| Proteomic | `Prot_VAMP3` | 0.881 | 0.910 | 0.890–0.927 | 881 / 1,000 |
| Proteomic | `Prot_SLC12A2` | 0.780 | 0.781 | 0.754–0.805 | 780 / 1,000 |
| Proteomic | `Prot_ERBB2_PG` | 0.750 | 0.750 | 0.722–0.776 | 750 / 1,000 |
| Proteomic | `Prot_EEA1` | 0.643 | 0.649 | 0.619–0.678 | 643 / 1,000 |
| Proteomic | `Prot_MIEN1` | 0.603 | 0.603 | 0.572–0.633 | 603 / 1,000 |
| Transcriptomic | `RNA_Exosome` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Transcriptomic | `RNA_HER2DX_HER2_amplicon` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Transcriptomic | `RNA_mRNA-ESR1` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Transcriptomic | `RNA_HER2DX_luminal` | 0.992 | 0.992 | 0.984–0.996 | 992 / 1,000 |
| Transcriptomic | `RNA_Mast-cells` | 0.924 | 0.963 | 0.949–0.973 | 924 / 1,000 |
| Transcriptomic | `RNA_HER2DX_pCR_likelihood_score` | 0.942 | 0.942 | 0.926–0.955 | 942 / 1,000 |
| Transcriptomic | `RNA_B-cells` | 0.570 | 0.906 | 0.881–0.927 | 570 / 1,000 |
| Transcriptomic | `RNA_CAF` | 0.360 | 0.902 | 0.869–0.928 | 360 / 1,000 |
| Transcriptomic | `RNA_sspbc_LumB` | 0.890 | 0.890 | 0.869–0.908 | 890 / 1,000 |
| Transcriptomic | `RNA_Neutrophils` | 0.056 | 0.862 | 0.757–0.925 | 56 / 1,000 |
| Transcriptomic | `RNA_mRNA-PGR` | 0.688 | 0.688 | 0.659–0.716 | 688 / 1,000 |
| Whole-slide image | `WSI_Cell_Interaction` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Whole-slide image | `WSI_Distance_tumor_immune` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |
| Whole-slide image | `WSI_Immune_Cell_prop` | 1.000 | 1.000 | 0.996–1.000 | 1,000 / 1,000 |

</details>

<details><summary><b>DHP arm</b> — 25 stable features</summary>

| Modality | Feature | All-fold frequency | Eligible-fold frequency | 95% Wilson CI | Folds selected |
|---|---|---|---|---|---|
| Clinical | `Clin_ANYNODES` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_ER` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_TUMSIZE` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_prolifvalu` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Genomic | `DNA_COSMIC.Signature.13` | 0.932 | 0.989 | 0.975–0.995 | 466 / 500 |
| Genomic | `DNA_ERBB2_CNA` | 0.982 | 0.982 | 0.966–0.991 | 491 / 500 |
| Genomic | `DNA_COSMIC.Signature.6` | 0.742 | 0.900 | 0.868–0.926 | 371 / 500 |
| Genomic | `DNA_LOH_Del_burden` | 0.412 | 0.741 | 0.686–0.789 | 206 / 500 |
| Genomic | `DNA_COSMIC.Signature.2` | 0.520 | 0.670 | 0.622–0.715 | 260 / 500 |
| Genomic | `DNA_coding_mutation_TP53` | 0.600 | 0.661 | 0.616–0.703 | 300 / 500 |
| Proteomic | `Prot_ERBB2_PG` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Proteomic | `Prot_HER2_amplicon` | 0.976 | 0.976 | 0.959–0.986 | 488 / 500 |
| Proteomic | `Prot_MIEN1` | 0.972 | 0.972 | 0.954–0.983 | 486 / 500 |
| Proteomic | `Prot_RPL19` | 0.902 | 0.906 | 0.877–0.928 | 451 / 500 |
| Proteomic | `Prot_CDK12` | 0.690 | 0.700 | 0.658–0.739 | 345 / 500 |
| Proteomic | `Prot_FLOT1` | 0.024 | 0.522 | 0.330–0.708 | 12 / 500 |
| Transcriptomic | `RNA_HER2DX_HER2_amplicon` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Transcriptomic | `RNA_mRNA-ESR1` | 0.946 | 0.946 | 0.923–0.963 | 473 / 500 |
| Transcriptomic | `RNA_HER2DX_pCR_likelihood_score` | 0.888 | 0.888 | 0.857–0.913 | 444 / 500 |
| Transcriptomic | `RNA_pik3ca_sig` | 0.734 | 0.857 | 0.821–0.887 | 367 / 500 |
| Transcriptomic | `RNA_HER2DX_luminal` | 0.756 | 0.756 | 0.716–0.792 | 378 / 500 |
| Transcriptomic | `RNA_sspbc_LumB` | 0.308 | 0.602 | 0.541–0.660 | 154 / 500 |
| Whole-slide image | `WSI_Cell_Interaction` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Distance_tumor_immune` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Immune_Cell_prop` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |

</details>

<details><summary><b>T-DM1 arm</b> — 23 stable features</summary>

| Modality | Feature | All-fold frequency | Eligible-fold frequency | 95% Wilson CI | Folds selected |
|---|---|---|---|---|---|
| Clinical | `Clin_ANYNODES` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_ER` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_TUMSIZE` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Clinical | `Clin_prolifvalu` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Genomic | `DNA_PIK3CA_CNA` | 0.974 | 0.974 | 0.956–0.985 | 487 / 500 |
| Genomic | `DNA_BRCA2_CNA` | 0.824 | 0.834 | 0.799–0.864 | 412 / 500 |
| Genomic | `DNA_HLA_Supertype_A01` | 0.584 | 0.781 | 0.736–0.820 | 292 / 500 |
| Genomic | `DNA_LOH_Del_burden` | 0.254 | 0.774 | 0.705–0.832 | 127 / 500 |
| Genomic | `DNA_NCOR1_CNA` | 0.652 | 0.749 | 0.707–0.788 | 326 / 500 |
| Genomic | `DNA_COSMIC.Signature.7` | 0.210 | 0.603 | 0.529–0.673 | 105 / 500 |
| Genomic | `DNA_COSMIC.Signature.13` | 0.112 | 0.523 | 0.430–0.616 | 56 / 500 |
| Proteomic | `Prot_SLC12A2` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Proteomic | `Prot_CTTN` | 0.110 | 0.917 | 0.819–0.964 | 55 / 500 |
| Proteomic | `Prot_VAMP3` | 0.800 | 0.866 | 0.832–0.894 | 400 / 500 |
| Proteomic | `Prot_RPL19` | 0.556 | 0.813 | 0.768–0.851 | 278 / 500 |
| Proteomic | `Prot_FLOT1` | 0.606 | 0.703 | 0.658–0.744 | 303 / 500 |
| Transcriptomic | `RNA_Exosome` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Transcriptomic | `RNA_Mast-cells` | 0.994 | 0.994 | 0.983–0.998 | 497 / 500 |
| Transcriptomic | `RNA_mRNA-ESR1` | 0.986 | 0.986 | 0.971–0.993 | 493 / 500 |
| Transcriptomic | `RNA_B-cells` | 0.746 | 0.846 | 0.809–0.877 | 373 / 500 |
| Whole-slide image | `WSI_Cell_Interaction` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Distance_tumor_immune` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |
| Whole-slide image | `WSI_Immune_Cell_prop` | 1.000 | 1.000 | 0.992–1.000 | 500 / 500 |

</details>

### Stability of the fusion weights

| Cohort | Modality | Mean weight | Median weight | Selection rate | 95% CI | Sign consistency |
|---|---|---|---|---|---|---|
| Pooled cohort | Transcriptomic | 2.27 | 2.21 | 100.0% | 1.00–1.00 | 1.00 |
| Pooled cohort | Proteomic | 2.15 | 2.13 | 98.3% | 0.97–0.99 | 1.00 |
| Pooled cohort | Whole-slide image | 1.63 | 1.43 | 76.8% | 0.74–0.79 | 1.00 |
| Pooled cohort | Clinical | 0.76 | 0.54 | 76.5% | 0.74–0.79 | 0.98 |
| Pooled cohort | Genomic | 0.85 | 0.42 | 62.3% | 0.59–0.65 | 1.00 |
| DHP arm | Transcriptomic | 1.07 | 0.96 | 97.8% | 0.96–0.99 | 1.00 |
| DHP arm | Proteomic | 1.90 | 1.66 | 97.0% | 0.95–0.98 | 1.00 |
| DHP arm | Genomic | 0.53 | 0.26 | 65.8% | 0.62–0.70 | 0.96 |
| DHP arm | Whole-slide image | 0.74 | 0.02 | 50.8% | 0.46–0.55 | 1.00 |
| DHP arm | Clinical | 0.22 | 0.00 | 26.0% | 0.22–0.30 | 0.92 |
| T-DM1 arm | Transcriptomic | 1.86 | 1.85 | 94.0% | 0.92–0.96 | 1.00 |
| T-DM1 arm | Proteomic | 1.86 | 1.79 | 88.0% | 0.85–0.91 | 1.00 |
| T-DM1 arm | Genomic | 1.72 | 1.64 | 84.0% | 0.81–0.87 | 1.00 |
| T-DM1 arm | Whole-slide image | 0.66 | 0.00 | 48.0% | 0.44–0.52 | 1.00 |
| T-DM1 arm | Clinical | 0.47 | 0.00 | 41.4% | 0.37–0.46 | 0.99 |

## 7. Consensus signatures and fusion weights

### Pooled cohort

| Modality | K | Winning classifier | Fold support | Signature (in rank order) |
|---|---|---|---|---|
| Clinical | 5 | `SVM_Linear` | 54% | `Clin_ER`, `Clin_prolifvalu`, `Clin_TUMSIZE`, `Clin_ANYNODES`, `Clin_Arm` |
| Transcriptomic | 10 | `ExtraTrees` | 33% | `RNA_HER2DX_HER2_amplicon`, `RNA_mRNA-ESR1`, `RNA_HER2DX_pCR_likelihood_score`, `RNA_Exosome`, `RNA_HER2DX_luminal`, `RNA_mRNA-PGR`, `RNA_Mast-cells`, `RNA_sspbc_LumB`, `RNA_B-cells`, `RNA_CAF` |
| Genomic | 8 | `HistGradBoost` | 30% | `DNA_COSMIC.Signature.13`, `DNA_COSMIC.Signature.6`, `DNA_COSMIC.Signature.2`, `DNA_PIK3CA_CNA`, `DNA_ERBB2_CNA`, `DNA_COSMIC.Signature.7`, `DNA_COSMIC.Signature.3`, `DNA_coding_mutation_TP53` |
| Proteomic | 7 | `SVM_Linear` | 37% | `Prot_RPL19`, `Prot_ERBB2_PG`, `Prot_HER2_amplicon`, `Prot_CDK12`, `Prot_VAMP3`, `Prot_SLC12A2`, `Prot_MIEN1` |
| Whole-slide image | 3 | `ElasticNet_LR` | 27% | `WSI_Cell_Interaction`, `WSI_Immune_Cell_prop`, `WSI_Distance_tumor_immune` |

### DHP arm

| Modality | K | Winning classifier | Fold support | Signature (in rank order) |
|---|---|---|---|---|
| Clinical | 4 | `ElasticNet_LR` | 44% | `Clin_ER`, `Clin_ANYNODES`, `Clin_TUMSIZE`, `Clin_prolifvalu` |
| Transcriptomic | 5 | `ElasticNet_LR` | 55% | `RNA_HER2DX_HER2_amplicon`, `RNA_mRNA-ESR1`, `RNA_pik3ca_sig`, `RNA_HER2DX_pCR_likelihood_score`, `RNA_HER2DX_luminal` |
| Genomic | 5 | `SVM_Linear` | 25% | `DNA_ERBB2_CNA`, `DNA_coding_mutation_TP53`, `DNA_COSMIC.Signature.13`, `DNA_COSMIC.Signature.6`, `DNA_LOH_Del_burden` |
| Proteomic | 5 | `SVM_Linear` | 51% | `Prot_ERBB2_PG`, `Prot_HER2_amplicon`, `Prot_MIEN1`, `Prot_RPL19`, `Prot_CDK12` |
| Whole-slide image | 3 | `ExtraTrees` | 45% | `WSI_Immune_Cell_prop`, `WSI_Distance_tumor_immune`, `WSI_Cell_Interaction` |

### T-DM1 arm

| Modality | K | Winning classifier | Fold support | Signature (in rank order) |
|---|---|---|---|---|
| Clinical | 4 | `ElasticNet_LR` | 75% | `Clin_ER`, `Clin_ANYNODES`, `Clin_TUMSIZE`, `Clin_prolifvalu` |
| Transcriptomic | 5 | `SVM_Linear` | 47% | `RNA_Exosome`, `RNA_Mast-cells`, `RNA_mRNA-ESR1`, `RNA_B-cells`, `RNA_CAF` |
| Genomic | 4 | `SVM_Linear` | 35% | `DNA_PIK3CA_CNA`, `DNA_HLA_Supertype_A01`, `DNA_NCOR1_CNA`, `DNA_BRCA2_CNA` |
| Proteomic | 5 | `SVM_Linear` | 35% | `Prot_SLC12A2`, `Prot_VAMP3`, `Prot_FLOT1`, `Prot_RPL19`, `Prot_CDK12` |
| Whole-slide image | 3 | `ElasticNet_LR` | 75% | `WSI_Cell_Interaction`, `WSI_Distance_tumor_immune`, `WSI_Immune_Cell_prop` |

### Late-fusion modality weights

| Cohort | Modality | Mean coefficient | SD | Selection rate |
|---|---|---|---|---|
| Pooled cohort | Transcriptomic | 2.15 | 0.72 | 99.7% |
| Pooled cohort | Proteomic | 2.03 | 0.95 | 97.5% |
| Pooled cohort | Whole-slide image | 1.46 | 1.39 | 68.5% |
| Pooled cohort | Clinical | 1.06 | 0.88 | 82.2% |
| Pooled cohort | Genomic | 0.60 | 0.84 | 52.4% |
| DHP arm | Proteomic | 1.95 | 0.97 | 100.0% |
| DHP arm | Transcriptomic | 1.06 | 0.57 | 99.4% |
| DHP arm | Whole-slide image | 0.67 | 1.04 | 44.4% |
| DHP arm | Genomic | 0.20 | 0.48 | 43.2% |
| DHP arm | Clinical | 0.04 | 0.28 | 12.8% |
| T-DM1 arm | Transcriptomic | 1.80 | 1.02 | 96.0% |
| T-DM1 arm | Proteomic | 1.48 | 1.23 | 79.8% |
| T-DM1 arm | Genomic | 0.97 | 1.17 | 55.8% |
| T-DM1 arm | Whole-slide image | 0.65 | 0.98 | 41.0% |
| T-DM1 arm | Clinical | 0.49 | 0.81 | 42.6% |

## 8. External validation

The pipeline's own transcriptomic consensus model was **frozen** — signature, classifier and hyper-parameters — refit once on PREDIX with no grid search, and applied to the external cohort. Nothing was refitted on external data. Both harmonisation schemes are reported so that a result present under only one would be identified as an artefact of that scheme.

Two refit populations were pre-specified and both are reported below: **arm-matched** (the model is refit on the PREDIX arm whose regimen the external cohort resembles) and **pooled** (refit on every PREDIX patient carrying transcriptomics, irrespective of arm). They answer different questions — arm-specific transfer, and transfer of one general model — so neither substitutes for the other.

### Arm-matched models

Source: `report/tables/revision/external_validation.xlsx`.

| Cohort | Refit on | Harmonisation | n | pCR | Internal AUROC | External AUROC | AUPRC | Brier | Calibration slope | P vs chance |
|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY2 | PREDIX DHP arm only | zscore | 44 | 26 | 0.748 [0.652–0.839] | **0.774 [0.622–0.904]** | 0.824 [0.715–0.932] | 0.1964 [0.1448–0.2491] | 1.07 (0.54–1.89) | 0.001 |
| I-SPY2 | PREDIX DHP arm only | rank | 44 | 26 | 0.748 [0.652–0.839] | **0.793 [0.645–0.915]** | 0.837 [0.736–0.939] | 0.1958 [0.1467–0.2441] | 1.21 (0.64–2.14) | < 0.001 |
| NCT02326974 | PREDIX T-DM1 arm only | zscore | 129 | 64 | 0.724 [0.626–0.816] | **0.644 [0.547–0.737]** | 0.656 [0.566–0.766] | 0.2414 [0.2133–0.2703] | 0.70 (0.28–1.29) | 0.003 |
| NCT02326974 | PREDIX T-DM1 arm only | rank | 129 | 64 | 0.724 [0.626–0.816] | **0.617 [0.518–0.711]** | 0.650 [0.561–0.756] | 0.2475 [0.2157–0.2803] | 0.53 (0.17–0.94) | 0.011 |

Locked specifications:

| External cohort | Resembles PREDIX arm | Refit on | Frozen classifier | Features | Refit on (n / events) |
|---|---|---|---|---|---|
| I-SPY2 (GSE194040) — trastuzumab/pertuzumab + chemotherapy | DHP | PREDIX DHP arm only | `ElasticNet_LR` {'C': 0.1} | 5 | 95 / 44 |
| NCT02326974 (GSE243375) — T-DM1 + pertuzumab | T-DM1 | PREDIX T-DM1 arm only | `SVM_Linear` {'C': 0.1} | 5 | 90 / 40 |

### Pooled model

Source: `report_pooled_external/tables/revision/external_validation_POOLED.xlsx`.

| Cohort | Refit on | Harmonisation | n | pCR | Internal AUROC | External AUROC | AUPRC | Brier | Calibration slope | P vs chance |
|---|---|---|---|---|---|---|---|---|---|---|
| I-SPY2 | pooled: all PREDIX patients with this modality | zscore | 44 | 26 | 0.745 [0.674–0.811] | **0.801 [0.664–0.915]** | 0.863 [0.766–0.947] | 0.1990 [0.1531–0.2455] | 1.40 (0.78–2.70) | < 0.001 |
| I-SPY2 | pooled: all PREDIX patients with this modality | rank | 44 | 26 | 0.745 [0.674–0.811] | **0.774 [0.628–0.893]** | 0.859 [0.771–0.937] | 0.2003 [0.1488–0.2498] | 1.21 (0.69–2.09) | < 0.001 |
| NCT02326974 | pooled: all PREDIX patients with this modality | zscore | 129 | 64 | 0.744 [0.673–0.810] | **0.669 [0.576–0.753]** | 0.693 [0.605–0.788] | 0.2260 [0.2016–0.2505] | 0.91 (0.46–1.55) | < 0.001 |
| NCT02326974 | pooled: all PREDIX patients with this modality | rank | 129 | 64 | 0.744 [0.673–0.810] | **0.707 [0.614–0.794]** | 0.737 [0.654–0.821] | 0.2163 [0.1873–0.2443] | 0.93 (0.55–1.47) | < 0.001 |

Locked specifications:

| External cohort | Resembles PREDIX arm | Refit on | Frozen classifier | Features | Refit on (n / events) |
|---|---|---|---|---|---|
| I-SPY2 (GSE194040) — trastuzumab/pertuzumab + chemotherapy | DHP | pooled: all PREDIX patients with this modality | `ExtraTrees` {'max_depth': 10, 'min_samples_leaf': 1, 'n_estimators': 300} | 8 | 185 / 84 |
| NCT02326974 (GSE243375) — T-DM1 + pertuzumab | T-DM1 | pooled: all PREDIX patients with this modality | `ExtraTrees` {'max_depth': 10, 'min_samples_leaf': 1, 'n_estimators': 300} | 8 | 185 / 84 |

**Both external cohorts discriminate above chance, under both refit populations and both harmonisation schemes.** AUROC across the harmonisation schemes: I-SPY2 0.774–0.793 (arm-matched, worst-case P 0.001); NCT02326974 0.617–0.644 (arm-matched, worst-case P 0.011); I-SPY2 0.774–0.801 (pooled, worst-case P < 0.001); NCT02326974 0.669–0.707 (pooled, worst-case P < 0.001).

Calibration is the honest qualifier, and it is reported separately from discrimination for exactly that reason: a frozen model can rank patients usefully in a cohort whose base rate and spread it mis-states. Read the calibration-slope column above — below 1 means the probabilities are more extreme than the cohort warrants, above 1 that they are compressed toward the base rate — and note that the two refit populations do not calibrate the same way in the same cohort. No result is withheld on calibration grounds and none is presented as though calibration were settled.

## 9. Figures

PNG renderings at 170 dpi; the citable vector versions are the PDFs in [`report/figures/`](report/figures) and [`report_pooled_external/figures/`](report_pooled_external/figures).

### Main figures

#### fig01_consensus_performance

![fig01_consensus_performance](report/figures_png/fig01_consensus_performance.png)

*Cross-validated AUROC of every consensus model with its 95% patient-level cluster-bootstrap interval, in the pooled cohort and each arm.*

#### fig02_consensus_signatures

![fig02_consensus_signatures](report/figures_png/fig02_consensus_signatures.png)

*The frozen consensus signature of each modality and scenario: mean absolute SHAP importance per feature, coloured by the direction of the association, with the winning classifier family above each panel.*

#### fig03_consensus_roc

![fig03_consensus_roc](report/figures_png/fig03_consensus_roc.png)

*Out-of-fold ROC curves of the integrated model and of the best single modality, drawn on all pooled (patient, repeat) predictions.*

#### fig04_consensus_modality_weights

![fig04_consensus_modality_weights](report/figures_png/fig04_consensus_modality_weights.png)

*Late-fusion modality weights of the consensus models: mean elastic-net coefficient and the fraction of folds in which each modality received a non-zero weight.*

#### fig05_consensus_feature_shap_Global

![fig05_consensus_feature_shap_Global](report/figures_png/fig05_consensus_feature_shap_Global.png)

*Feature-level SHAP attribution for the pooled-cohort consensus models, restricted to the consensus signature.*

#### fig05_consensus_feature_shap_DHP

![fig05_consensus_feature_shap_DHP](report/figures_png/fig05_consensus_feature_shap_DHP.png)

*Feature-level SHAP attribution for the DHP-arm consensus models.*

#### fig05_consensus_feature_shap_T_DM1

![fig05_consensus_feature_shap_T_DM1](report/figures_png/fig05_consensus_feature_shap_T_DM1.png)

*Feature-level SHAP attribution for the T-DM1-arm consensus models.*

#### fig06_counterfactual_summary

![fig06_counterfactual_summary](report/figures_png/fig06_counterfactual_summary.png)

*Counterfactual summary: predicted response under each treatment assignment.*

### Revision figures

Calibration, stability, events per variable, external validation, paired comparisons and fusion weights — the diagnostics added in this revision.

#### revfig01_calibration

![revfig01_calibration](report/figures_png/revfig01_calibration.png)

*Calibration of the consensus integrated model: reliability curves over ten equal-count bins of all out-of-fold predictions, with patient-level cluster-bootstrap intervals, and the slope, intercept and Brier score of each scenario.*

#### revfig02_selection_stability

![revfig02_selection_stability](report/figures_png/revfig02_selection_stability.png)

*Feature-selection frequency across the outer folds, with Wilson intervals and the pre-specified stability threshold (0.60 pooled, 0.50 per arm).*

#### revfig03_epv_per_fold

![revfig03_epv_per_fold](report/figures_png/revfig03_epv_per_fold.png)

*Per-fold pCR event counts and realised events-per-variable for every model.*

#### revfig06_external_validation

![revfig06_external_validation](report/figures_png/revfig06_external_validation.png)

*Locked-model external validation, arm-matched design: ROC and precision–recall curves and reliability of the frozen DHP and T-DM1 transcriptomic models in I-SPY2 and NCT02326974.*

#### revfig06_external_validation_POOLED

![revfig06_external_validation_POOLED](report/figures_png/revfig06_external_validation_POOLED.png)

*Locked-model external validation, pooled design: the same two cohorts scored by a single transcriptomic model refit on all PREDIX patients carrying transcriptomics, irrespective of treatment arm. Pre-specified alongside the arm-matched analysis above; both are reported, neither replaces the other.*

#### revfig07_model_comparisons

![revfig07_model_comparisons](report/figures_png/revfig07_model_comparisons.png)

*AUROC forest and paired ΔAUROC of the integrated model against every single-modality comparator, with 95% paired cluster-bootstrap intervals.*

#### revfig08_fusion_weights

![revfig08_fusion_weights](report/figures_png/revfig08_fusion_weights.png)

*Fold-wise distribution of the late-fusion modality weights and each modality's selection rate.*

### Supplementary figures — discovery phase

Diagnostics of the fully nested discovery phase, before consensus finalisation.

#### supp_fig01_roc_curves

![supp_fig01_roc_curves](report/figures_png/supp_fig01_roc_curves.png)

*Discovery-phase ROC curves.*

#### supp_fig02_performance_distributions

![supp_fig02_performance_distributions](report/figures_png/supp_fig02_performance_distributions.png)

*Discovery-phase distribution of per-fold performance for every model.*

#### supp_fig03_fusion_benefit

![supp_fig03_fusion_benefit](report/figures_png/supp_fig03_fusion_benefit.png)

*Discovery-phase fusion benefit against the best single modality.*

#### supp_fig04_forest_plot

![supp_fig04_forest_plot](report/figures_png/supp_fig04_forest_plot.png)

*Discovery-phase forest plot of per-fold AUROC.*

#### supp_fig05_feature_shap_Global

![supp_fig05_feature_shap_Global](report/figures_png/supp_fig05_feature_shap_Global.png)

*Discovery-phase SHAP attribution, pooled cohort.*

#### supp_fig05_feature_shap_DHP

![supp_fig05_feature_shap_DHP](report/figures_png/supp_fig05_feature_shap_DHP.png)

*Discovery-phase SHAP attribution, DHP arm.*

#### supp_fig05_feature_shap_T_DM1

![supp_fig05_feature_shap_T_DM1](report/figures_png/supp_fig05_feature_shap_T_DM1.png)

*Discovery-phase SHAP attribution, T-DM1 arm.*

#### supp_fig06_feature_selection_frequency

![supp_fig06_feature_selection_frequency](report/figures_png/supp_fig06_feature_selection_frequency.png)

*Discovery-phase selection frequency of every candidate feature.*

#### supp_fig07_cross_scenario_features

![supp_fig07_cross_scenario_features](report/figures_png/supp_fig07_cross_scenario_features.png)

*Features shared between the pooled and arm-specific signatures.*

#### supp_fig08_fusion_shap

![supp_fig08_fusion_shap](report/figures_png/supp_fig08_fusion_shap.png)

*SHAP attribution of the five modality streams inside the fusion layer.*

#### supp_fig09_modality_weights

![supp_fig09_modality_weights](report/figures_png/supp_fig09_modality_weights.png)

*Discovery-phase modality weights.*

#### supp_fig10_winner_classifier_heatmap

![supp_fig10_winner_classifier_heatmap](report/figures_png/supp_fig10_winner_classifier_heatmap.png)

*Which classifier family won each fold, by modality and scenario.*

#### supp_fig11_inner_auroc_comparison

![supp_fig11_inner_auroc_comparison](report/figures_png/supp_fig11_inner_auroc_comparison.png)

*Inner-cross-validation AUROC of each classifier family, the basis of the Stage A choice.*

#### supp_fig12_calibration_profile

![supp_fig12_calibration_profile](report/figures_png/supp_fig12_calibration_profile.png)

*Discovery-phase calibration profile.*

#### supp_fig13_signature_sizes

![supp_fig13_signature_sizes](report/figures_png/supp_fig13_signature_sizes.png)

*Distribution of discovered signature sizes across folds.*

#### supp_fig14_performance_CI

![supp_fig14_performance_CI](report/figures_png/supp_fig14_performance_CI.png)

*Discovery-phase AUROC with patient-level cluster-bootstrap intervals — the fully nested estimates, free of consensus selection optimism.*

---

Regenerate this page with `python docs/build_RESULTS_md.py`.
