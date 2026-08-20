#!/usr/bin/env python3
"""
MULTIMODAL pCR PREDICTION PIPELINE — PREDIX HER2
=================================================
Primary analysis mode (elasticnet) implements multi-classifier signature
discovery with leakage-safe stacking and Platt calibration.

CANDIDATE-POOL CONSTRUCTION AND THE UNIVARIATE SCREEN  (see --univariate_screen)
-------------------------------------------------------------------------------
The candidate feature panel used in the original submission was assembled in
two stages: (a) a-priori biological curation (established transcriptomic
signatures, known HER2/ADC-trafficking and immune features, recurrent CNAs,
mutational signatures), and (b) retention of features showing a univariate
association with pCR *in the full cohort*. Step (b) used the outcome across
all patients, so the candidate universe was partly outcome-informed and the
resulting internal cross-validated performance is optimistic.

This pipeline now supports both analyses explicitly:

  --univariate_screen in_fold   (DEFAULT — primary, leakage-free)
      The univariate association step is performed INSIDE each training fold
      (outer and inner), on training patients only. No test patient influences
      which features enter the model. Screening statistic is the tie-corrected
      Mann-Whitney U / univariate AUROC, Benjamini-Hochberg FDR controlled
      within modality within fold. See `univariate_screen_indices`.

  --univariate_screen none      (legacy / sensitivity analysis)
      Reproduces the original submission's behaviour, in which the univariate
      step had already been applied to the whole cohort before the data file
      was written. Retained so the optimism attributable to that step can be
      quantified by direct comparison.

  --feature_pool {curated, full}
      `curated` applies the fixed TIER1_REMOVE biological deduplication list.
      `full` disables it, so the run starts from the complete pre-curation
      feature set present in the input file. Combine `--feature_pool full
      --univariate_screen in_fold` for the fully leakage-free analysis
      starting from all measured features.

PRIMARY ANALYSIS — elasticnet mode
  Stage A Pass 1: All classifiers compared with fixed STAGE_A_PARAMS using inner CV.
           Feature importance converted to cross-classifier percentile ranks and
           averaged across inner folds. EPV=5 cap + 25th-percentile filter + floor=5
           derive signature per classifier. Clin and WSI keep all features.
           Calibration slope estimated from inner-loop OOF predictions.
  Stage A Pass 2: Pruned signature evaluated on each cached inner val fold.
           Winner selected by mean pruned inner AUROC (not all-feature AUROC).
  Stage B: Winner tuned with GridSearchCV on expanded training.
           Outer refit on winner signature features + Platt calibration if needed.
           OOF scores via make_oof_signature (expanded inner training, same config).
  Fusion:  Single Fused_ElasticNet (L1+L2, l1_ratio=0.5) trained on calibrated
           5-column OOF matrix. L1 zeroes non-contributing modalities for
           interpretable sparse modality weighting.

SUPPLEMENTARY MODES (best_per_fold, ensemble_weighted)
  Standard CC-only training, no signature discovery.
  Used for robustness comparison only.

EXPANDED TRAINING (elasticnet mode only)
  Each unimodal model trains on ALL patients with that modality minus test patients.
  Outer test sets always drawn from complete-case (n=110) for paired comparisons.

USAGE
-----
  # Primary analysis:
  python3 multimodal_pcr_pipeline.py --data_path /data/predix.txt \\
      --classifiers ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_Linear \\
      --repeats_global 200 --repeats_arm 100

OUTPUT PKL FORMAT  ({results_dir}/{exp}/{exp}_elasticnet_results.pkl)
  {
    "Clin" / "RNA" / "DNA" / "Prot" / "WSI":  [fold_dict, ...]
    "Fused_ElasticNet":                        [fold_dict, ...]
  }

  Unimodal fold_dict keys (elasticnet / primary mode):
    fold_idx, metrics (AUROC/AUPRC/Brier/Sensitivity/Specificity/Threshold)
    y_test, y_pred
    winner_clf, winner_signature, signature_size, n_events_inner
    inner_cv_aurocs_A  {clf: mean_auroc}   — Stage A fixed-params AUROC
    inner_cv_auroc_B   float               — Stage B tuned AUROC
    inner_cv_params    dict                — winner best hyperparams
    inner_importance   {clf: {feat: val}}  — normalised importance per clf
    signatures_all     {clf: [feats]}      — EPV-capped signature per clf
    calibration        {clf: {slope, needs_platt}}
    platt_applied      bool
    features           [feature names in winner signature]
    oof_shap           {feature_names, shap_values, X_test_scaled}

  Fusion fold_dict keys:
    fold_idx, metrics, y_test, y_pred
    tuned_C, modality_weights, selected_modalities
    oof_shap  {feature_names: [Clin,RNA,DNA,Prot,WSI], shap_values, X_test_scaled}
"""

import argparse, pickle, warnings, os
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# THREAD-POOL DISCIPLINE (must be set BEFORE numpy / sklearn are imported so
# BLAS/OpenMP libraries pick them up at init). Setting os.environ[...] later
# from inside joblib workers does NOT work because BLAS has already started
# its thread pool by then. Workers additionally use threadpool_limits() as a
# runtime belt-and-braces guard (see _process_single_fold).
# ---------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")
os.environ.setdefault("BLIS_NUM_THREADS",     "1")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from threadpoolctl import threadpool_limits
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV, ParameterGrid,
                                     StratifiedKFold,
                                     RepeatedStratifiedKFold)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              brier_score_loss, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shap

# ==============================================================================
# SECTION 1 — MODULE-LEVEL PLACEHOLDERS  (all assigned in main)
# ==============================================================================
DATA_PATH = RESULTS_DIR = RANDOM_SEED = None
GLOBAL_N_OUTER_FOLDS = GLOBAL_N_REPEATS = GLOBAL_N_INNER_FOLDS = None
ARM_N_OUTER_FOLDS    = ARM_N_REPEATS    = ARM_N_INNER_FOLDS    = None
CORR_THRESHOLD = NZV_RATIO_THRESHOLD = None
NZV_FREQ_GLOBAL = NZV_FREQ_ARM = NZV_FREQ_THRESHOLD = None
STABILITY_THRESHOLD_GLOBAL = STABILITY_THRESHOLD_ARM           = None
N_JOBS                                                         = None

# ── Tier 2.5 in-fold univariate screen (see SECTION 4b) ──────────────────────
# UNIVARIATE_SCREEN is the master switch. It is True for the primary,
# leakage-free analysis and False for the legacy analysis that reproduces the
# original submission (where the univariate step had already been applied to
# the whole cohort before the input file was written).
UNIVARIATE_SCREEN       = True
UNIV_SCREEN_FDR_Q       = 0.25   # BH q-value ceiling, per modality per fold
UNIV_SCREEN_MAX_K       = 40     # hard cap on candidates entering Stage A
UNIV_SCREEN_MIN_K       = 5      # floor so a modality can never collapse
UNIV_SCREEN_MIN_FEATURES = 6     # modalities at or below this keep all features

# Applied to the whole run and written into the provenance record.
PIPELINE_VERSION = "2.0.0-revision1"
FEATURE_POOL     = "curated"


def _resolve_parallel_budget(n_folds, n_jobs):
    """
    Allocate the available CPU budget between outer-fold parallelism and
    inner-fit parallelism.

    Three regimes:
      1. n_jobs == 1                 → sequential (debugging)
      2. n_jobs >= n_folds           → outer=n_folds, inner = n_jobs // n_folds
                                       (nested parallelism — the "CPU-rich" case
                                       the user reports when folds < CPUs)
      3. n_jobs <  n_folds           → outer=n_jobs, inner=1
                                       (outer-only — the classic case)

    Returns (n_outer_workers, n_inner_jobs).
    """
    if n_jobs is None or n_jobs == 1:
        return 1, 1
    total_cpus = n_jobs if n_jobs > 0 else joblib.cpu_count()
    n_outer    = min(total_cpus, n_folds)
    n_inner    = max(1, total_cpus // max(n_outer, 1))
    return n_outer, n_inner


# Fixed algorithmic constants
L1_RATIO            = 0.5
ELASTICNET_C_GRID   = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
FUSION_C_GRID       = ELASTICNET_C_GRID
# ---------------------------------------------------------------------------
# CORRELATION HANDLING (run 4 change — read this before altering either set)
# ---------------------------------------------------------------------------
# Up to run 3 a single constant, CORR_FILTER_MODS, governed TWO different
# things: the per-fold Tier 3 correlation filter, and the correlation-cluster
# pooling performed once at the consensus stage in _aggregate_signature. They
# are now separate, because run 4 removes the first and keeps the second.
#
# WHY TIER 3 IS REMOVED. Its job is taken over entirely by the fixed,
# outcome-blind TIER1_REMOVE list, which run 4 extends so that no pair of
# candidate features exceeds |r| = 0.90 on this dataset (verified by
# preflight.py, which FAILS the run if any pair does). Tier 3 was applied per
# fold and kept whichever cluster member won that fold's univariate contest, so
# the surviving representative rotated between folds. That rotation was visible
# in the run-3 output: DNA_coding_mutation_TP53 and its identical twin
# ..._TP53_oncokb each accumulated partial selection frequency (0.653 and 0.427,
# summing to ~1) and BOTH appeared in the pooled consensus signature. Deciding
# redundancy once, a priori and on biology, removes that artefact and removes a
# stage from the Methods.
#
# WHY THE CONSENSUS DEDUP IS KEPT. It is a cheap safety net against any
# correlated pair that survives Tier 1 in a future dataset, and it operates on
# the full cohort rather than fold by fold, so it does not rotate. On this
# dataset it should now be a no-op.
# RUN 5 — how the consensus signature is aggregated relative to the locked
# classifier. See finalize_consensus() for the full rationale.
#   "winner_folds"     : RUN-5 DEFAULT. Restrict the aggregation to the outer
#                        folds the modal classifier actually won, and use that
#                        fold's `winner_signature`. The reported classifier and
#                        the reported signature are then the same model.
#   "winner_all_folds" : the modal classifier's own Stage-A signature from
#                        EVERY fold, won or not — a larger sample (all
#                        500-1000 folds rather than the 26-55% it won), but see
#                        the warning below.
#   "all_folds"        : run-4 behaviour — every fold contributed its own
#                        winner's signature, mixing families. Retained so the
#                        change can be isolated without editing code.
#
# WHY winner_folds IS THE DEFAULT AND NOT winner_all_folds.
# The two draw from different objects. `winner_signature` (set at the end of
# the fold) is the Stage-A signature INTERSECTED with the features that
# survived that fold's outer preprocessing; `signatures_all[clf]` is the RAW
# Stage-A signature, before that intersection. Aggregating the raw list
# therefore ranks and sizes K on a strictly larger set: measured on the run-4
# PKLs, K moves 7 -> 9 for global DNA, 10 -> 11 for global RNA and 4 -> 5 for
# T-DM1 DNA, and the locked signature can acquire features that outer
# preprocessing routinely drops (which then surface as "dropped by fold
# preprocessing" at refit and inflate n_model_features in the external table).
# That growth is an artefact of the source swap, not of the coherence fix we
# actually wanted. winner_folds has no such mismatch. winner_all_folds is kept
# for sensitivity analysis and now intersects with `candidate_features` to
# remove the artefact, but the default stays on the object the pipeline has
# always used.
SIGNATURE_SOURCE      = "winner_folds"

CORR_FILTER_MODS      = set()             # Tier 3, per fold — REMOVED in run 4
CONSENSUS_DEDUP_MODS  = {"RNA", "DNA"}    # consensus-stage cluster pooling

# Populated in main() after seed is set
CLASSIFIERS = {}

# Fixed hyperparameters for Stage A (classifier comparison + feature ranking).
# These are applied uniformly to all classifiers so that the AUROC
# comparison is fair: no classifier benefits from extra tuning time.
# The winner classifier is then fully tuned in Stage B (GridSearchCV).
STAGE_A_PARAMS = {
    "ElasticNet_LR": {"C": 0.1},
    "RandomForest":  {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "ExtraTrees":    {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "HistGradBoost": {"learning_rate": 0.1, "max_depth": 3, "max_iter": 200},
    "SVM_Linear":    {"C": 0.1},
}



# ==============================================================================
# SECTION 2 — CLASSIFIER FACTORY
# ==============================================================================

def build_classifiers(seed):
    """
    Return full classifier config dict. Called once in main() after seed set.

    RANDOMNESS DESIGN
    -----------------
    Classifier internal randomness is DECOUPLED from the reproducibility seed.
    RepeatedStratifiedKFold uses `seed` so the train/test partitions are
    reproducible across runs — that is what anchors reproducibility. The
    classifiers themselves use `random_state=None`, i.e. fresh NumPy
    randomness on every fit. Without this, every RandomForest in every
    repeat sees the same bootstrap sample for a given fold and every
    saga-solver uses the same shuffle order — this makes the 200 repeats
    artificially correlated and produces overly narrow CIs on fold-averaged
    metrics. With random_state=None the repeats are genuinely independent
    for fixed fold assignments, so reported variance is honest.

    The `seed` argument is retained in the signature for backwards
    compatibility but is no longer propagated into classifier constructors.
    """
    return {
        "ElasticNet_LR": {
            "build":     lambda: LogisticRegression(
                             penalty="elasticnet", solver="saga",
                             l1_ratio=L1_RATIO, max_iter=2000,
                             random_state=None),
            "grid":      {"C": ELASTICNET_C_GRID},
            "shap_type": "linear",
        },
        "RandomForest": {
            "build":     lambda: RandomForestClassifier(
                             random_state=None, n_jobs=1),
            "grid":      {"n_estimators": [100, 300],
                          "max_depth": [None, 5, 10],
                          "min_samples_leaf": [1, 5]},
            "shap_type": "tree",
        },
        "ExtraTrees": {
            "build":     lambda: ExtraTreesClassifier(
                             random_state=None, n_jobs=1),
            "grid":      {"n_estimators": [100, 300],
                          "max_depth": [None, 5, 10],
                          "min_samples_leaf": [1, 5]},
            "shap_type": "tree",
        },
        "HistGradBoost": {
            "build":     lambda: HistGradientBoostingClassifier(
                             random_state=None),
            "grid":      {"learning_rate": [0.05, 0.1, 0.2],
                          "max_depth": [3, 5, None],
                          "max_iter": [100, 300]},
            "shap_type": "tree",
        },
        "SVM_RBF": {
            "build":     lambda: SVC(kernel="rbf", probability=True,
                                     random_state=None, cache_size=500),
            "grid":      {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]},
            "shap_type": "none",
        },
        "SVM_Linear": {
            "build":     lambda: SVC(kernel="linear", probability=True,
                                     random_state=None, cache_size=500),
            "grid":      {"C": [0.01, 0.1, 1.0, 10.0]},
            "shap_type": "linear_svm",
        },
    }


# SECTION 2b — TIER 1 BIOLOGICAL DEDUPLICATION (FIXED, PRE-SPECIFIED)
# ==============================================================================
# These features are removed BEFORE any analysis based purely on domain knowledge
# of the PREDIX HER2 genomic architecture. This is NOT a data-driven decision —
# it is data quality management. Each removal is individually justified below.
#
# Empirical verification: all removed features had r >= 0.90 with their retained
# counterpart in the complete-case cohort (r = 1.000 for exact duplicates).

TIER1_REMOVE = [
    # ------------------------------------------------------------------
    # DNA: 17q12 chromosomal amplicon co-amplification
    # ERBB2, GRB7, PPP1R1B, MIEN1 and CDK12 all reside on 17q12 and
    # co-amplify as a single genomic segment. Their CNA values are
    # therefore identical by construction (r = 1.000 with ERBB2_CNA).
    # Decision: keep DNA_ERBB2_CNA (the oncogenic driver of HER2+ BC).
    # ------------------------------------------------------------------
    "DNA_PPP1R1B_CNA",   # r=1.000 with DNA_ERBB2_CNA  (17q12 amplicon)
    "DNA_MIEN1_CNA",     # r=1.000 with DNA_ERBB2_CNA  (17q12 amplicon)
    "DNA_GRB7_CNA",      # r=1.000 with DNA_ERBB2_CNA  (17q12 amplicon)
    "DNA_CDK12_CNA",     # r=0.904 with DNA_ERBB2_CNA  (17q12 amplicon)

    # ------------------------------------------------------------------
    # DNA: 11q13 amplicon co-amplification
    # PPFIA1 and CTTN co-amplify on 11q13 (r = 1.000).
    # Decision: keep DNA_PPFIA1_CNA.
    # ------------------------------------------------------------------
    "DNA_CTTN_CNA",      # r=1.000 with DNA_PPFIA1_CNA (11q13 amplicon)

    # ------------------------------------------------------------------
    # DNA: TMB metric cluster
    # totalTMB, TMB_uniform, TMB_clone and pTMB measure the same underlying
    # mutational burden at different granularities. The list was written
    # against an earlier data release that carried DNA_totalTMB and
    # DNA_TMB_subclone; NEITHER IS PRESENT in the canonical file
    # (clin_multiomics_curated_metrics_PREDIX_HER2_new.txt, 197 x 114), and
    # nor are TMB_uniform or pTMB, so only DNA_TMB_clone is actually removed
    # by this block. That removal is still correct on this file, but for a
    # different reason than the one originally written down:
    # DNA_TMB_clone is an EXACT duplicate of DNA_TMB_clone_oncogenic
    # (r = 1.000 on the complete case, n = 109), which is retained.
    # Consequence to state in the Methods: on this data file the panel
    # carries clonal oncogenic TMB only — there is no total-TMB feature.
    # ------------------------------------------------------------------
    "DNA_TMB_uniform",   # not present in the canonical file: no effect
    "DNA_TMB_clone",     # r=1.000 with the retained DNA_TMB_clone_oncogenic
    "DNA_pTMB",          # not present in the canonical file: no effect

    # ------------------------------------------------------------------
    # RNA: immune infiltration cluster
    # CD8-T-cells, T-cells, CD45, and Cytotoxic-cells are near-identical
    # to RNA_TILs and RNA_mRNA-CD8A (r = 0.926–0.984). Retaining all
    # causes the 'rotating basis' instability in elastic net.
    # Decision: keep RNA_TILs  (global immune composite, widely used)
    #                RNA_mRNA-CD8A (cytotoxic-specific signal)
    #                RNA_NK-cells  (innate immunity, distinct biology)
    # ------------------------------------------------------------------
    "RNA_CD8-T-cells",    # r=0.984 with RNA_mRNA-CD8A
    "RNA_T-cells",        # r=0.972 with RNA_TILs
    "RNA_CD45",           # r=0.948 with RNA_TILs
    "RNA_Cytotoxic-cells",# r=0.940 with RNA_mRNA-CD8A

    # ------------------------------------------------------------------
    # RNA: HER2 expression redundancy
    # RNA_HER2DX_HER2_amplicon is a validated composite score that
    # subsumes raw RNA_mRNA-ERBB2 (r = 0.959). Keep the composite score
    # (HER2DX_HER2_amplicon) as it is the curated, clinically validated
    # representation.
    # ------------------------------------------------------------------
    "RNA_mRNA-ERBB2",    # r=0.959 with RNA_HER2DX_HER2_amplicon

    # ==================================================================
    # ADDED IN RUN 4. Everything below closes the gap that the per-fold
    # Tier 3 filter used to cover. Tier 3 is removed in run 4, so this
    # list must now leave NO pair of candidates above |r| = 0.90.
    # preflight.py verifies exactly that and fails the run otherwise.
    # ==================================================================

    # ------------------------------------------------------------------
    # DNA: OncoKB-annotated duplicates of their own base column.
    # These pairs are IDENTICAL on every one of the 190 rows where both
    # are observed — the OncoKB annotation step evidently reclassified
    # nothing in this cohort. Retaining both gave each half the selection
    # frequency of one real signal and put both into the run-3 consensus
    # signature (TP53 0.653 + TP53_oncokb 0.427). Keep the plain column:
    # the "_oncokb" name asserts a pathogenicity filter that demonstrably
    # made no difference here, so keeping it would be misleading.
    # ------------------------------------------------------------------
    "DNA_coding_mutation_TP53_oncokb",    # identical to DNA_coding_mutation_TP53
    "DNA_coding_mutation_GATA3_oncokb",   # identical to DNA_coding_mutation_GATA3
    "DNA_coding_mutation_PIK3CA_oncokb",  # r=0.954, differs in 2 of 190 rows

    # ------------------------------------------------------------------
    # DNA: FADD and PPFIA1 both sit on the 11q13 amplicon and are
    # identical here (r = 1.000), as CTTN already was. Keep PPFIA1, the
    # representative already chosen for that amplicon above.
    # ------------------------------------------------------------------
    "DNA_FADD_CNA",      # r=1.000 with DNA_PPFIA1_CNA (11q13 amplicon)

    # ------------------------------------------------------------------
    # RNA: the remaining immune-infiltration cluster.
    # The original Tier-1 block deliberately KEPT TILs, mRNA-CD8A and
    # NK-cells "for distinct biology", but they still cluster at
    # r = 0.913-0.942, and the per-fold Tier 3 filter then pruned two of
    # the three anyway — a direct contradiction between the two stages,
    # resolved silently and differently in each fold. With Tier 3 gone the
    # decision must be made here, once.
    #
    # KEEPING mRNA-CD8A rather than TILs. Both are defensible biologically;
    # the tie-break is external validity, decided on measurement
    # availability alone and never on outcome. RNA_TILs is 100% missing in
    # NCT02326974 and is already excluded from the transferable feature
    # universe of that cohort, so keeping it would leave the T-DM1 external
    # model with no immune-infiltration term at all. mRNA-CD8A is present
    # in both external cohorts.
    #
    # >>> TO REVERSE THIS CHOICE: swap "RNA_TILs" for "RNA_mRNA-CD8A" on
    # >>> the next line. Nothing else in the pipeline needs changing;
    # >>> preflight.py will confirm the cluster is still resolved.
    #
    # RNA_NK-cells is RETAINED. It correlated with TILs at 0.913, but once
    # TILs is removed its strongest remaining correlation is 0.845 (with
    # mRNA-CD8A), below the 0.90 threshold. Removing it as well would be
    # over-pruning and would discard the innate-immunity signal that the
    # original Tier-1 comment wanted to keep.
    # ------------------------------------------------------------------
    "RNA_TILs",          # r=0.942 with the retained RNA_mRNA-CD8A

    # ------------------------------------------------------------------
    # Prot: 17q12 amplicon proteins.  *** NEW CLASS OF DEFECT — PLEASE
    # CONFIRM WITH THE PROTEOMICS LEAD BEFORE PUBLICATION ***
    #
    # These pairs were never deduplicated by ANY stage of the pipeline up
    # to run 3. Tier 3 was restricted to RNA and DNA on the stated ground
    # that "Clin, Prot and WSI have <= 5 features" — but Prot has 19. The
    # consequence is visible in run 3, where Prot_ERBB2 and
    # Prot_HER2_amplicon both appear in the pooled consensus signature, and
    # again in the DHP signature, although they correlate at 0.923.
    #
    # Keeping the curated composite (Prot_HER2_amplicon) and dropping its
    # two individual constituents, by consistency with the RNA decision
    # above, where the curated HER2DX composite was kept over raw
    # mRNA-ERBB2. Prot_ERBB2_PG (proteogenomic status) is a DIFFERENT
    # quantity, correlates below threshold, and is retained.
    #
    # >>> TO REVERSE: remove the two lines below and add
    # >>> "Prot_HER2_amplicon" instead, keeping ERBB2 and GRB7 as the
    # >>> individually interpretable proteins.
    # ------------------------------------------------------------------
    "Prot_ERBB2",        # r=0.923 with the retained Prot_HER2_amplicon
    "Prot_GRB7",         # r=0.914 with the retained Prot_HER2_amplicon

    # ------------------------------------------------------------------
    # RNA: FCGR3B — excluded on measurement-validity grounds (run 5).
    #
    # JUSTIFICATION (outcome-blind, and the only basis for the exclusion).
    # FCGR3B (FcgammaRIIIb, CD16b) is expressed almost exclusively by
    # neutrophils. Neutrophils are not retained in fresh-frozen tumour
    # biopsies, so FCGR3B signal in bulk tumour RNA-seq is attributable to
    # peripheral-blood contamination rather than to the tumour immune
    # microenvironment, and its magnitude is not expected to be comparable
    # across cohorts processed under different protocols. That property is a
    # fact about the assay and is verifiable without reference to pCR, which
    # is what places this alongside the other Tier-1 entries rather than
    # outside them.
    #
    # NOTE that the FORM of the argument differs from the entries above.
    # Those are redundancy removals: an exact duplicate, or |r| above the
    # 0.90 gate. FCGR3B is not redundant with anything retained — its
    # strongest correlation with any other retained RNA feature is well below
    # the gate — so it is excluded for non-interpretability of the
    # measurement, not for duplication. State it that way in the Methods.
    #
    # IMPACT DISCLOSURE (not a justification; recorded so the change is not
    # silent). In run 4, with FCGR3B retained, it appeared in the T-DM1 arm
    # signature and in the pooled transcriptomic model, so run-5 numbers for
    # the T-DM1 arm and for both external validations are not comparable to
    # run 4's. The exclusion was decided on the grounds above; the resulting
    # numbers are reported as they fall, in either direction.
    #
    # >>> CONFIRM WITH THE BIOINFORMATICS LEAD that the paragraph above is
    # >>> their actual reasoning, and record the date of that decision in
    # >>> run_provenance.json. If the reasoning differs, replace this block —
    # >>> do NOT let the justification become "it predicted too well" or "it
    # >>> did not transfer", either of which is outcome-informed selection,
    # >>> the exact defect this revision exists to fix.
    # >>> TO REVERSE: delete the line below.
    # ------------------------------------------------------------------
    "RNA_FCGR3B",        # neutrophil-restricted; see the block above
]

# ==============================================================================
# SECTION 3 — DATA LOADING AND BASE ENCODING
# ==============================================================================

def load_and_encode_data(path: Path) -> pd.DataFrame:
    """
    Load the dataset and apply all fixed categorical/ordinal encodings.

    This function performs ONLY transformations that are non-data-dependent
    (i.e., they map known fixed categories to numbers). No imputation, scaling,
    feature selection, or any fold-dependent operation is performed here.

    Encoding decisions:
      - Boolean strings ('True'/'False') → 0/1
      - Clin_ER: 'positive'=1, 'negative'=0
      - Clin_ANYNODES: 'N+'=1, 'N0'=0
      - Clin_TUMSIZE: ordinal (<=20=1, 21-50=2, >50=3); NaN preserved → imputed in CV
      - Clin_Arm: 'DHP'=0, 'T-DM1'=1
      - Prot_ERBB2_PG: 'Positive'=1, 'Negative'=0
      - RNA_sspbc.subtype: one-hot with Her2 as reference category

    Parameters
    ----------
    path : Path
        Location of the tab-separated dataset file.

    Returns
    -------
    pd.DataFrame
        Encoded dataset with Tier 1 redundant features removed.
    """
    df = pd.read_csv(path, sep="\t")
    # Assign a stable integer patient ID before any filtering or reindexing.
    # This ID is used to match patients across modality-specific subsets and
    # to exclude test-set patients from training sets without index confusion.
    df["patient_id"] = range(len(df))
    print(f"[LOAD] Raw data: {df.shape[0]} patients, {df.shape[1]} columns")

    # --- 1. Convert boolean columns to 0/1 -----------------------------------
    # DNA mutation and genomic flag columns are stored as Python bool objects
    # (True/False) inside object-dtype arrays — NOT as strings 'True'/'False'.
    # We identify them by checking that all non-null unique values are a subset
    # of {True, False} (Python booleans), then cast with astype(int).
    bool_cols = [
        c for c in df.columns
        if df[c].dtype == object
        and set(df[c].dropna().unique()).issubset({True, False})
    ]
    for col in bool_cols:
        # Cast to float (not int) to preserve NaN — these are imputed inside each CV fold
        df[col] = df[col].astype(float)
    print(f"[ENCODE] {len(bool_cols)} boolean columns → 0/1: {bool_cols}")

    # --- 2. Clinical categorical encodings ------------------------------------
    df["Clin_ER"]       = df["Clin_ER"].map({"positive": 1, "negative": 0})
    df["Clin_ANYNODES"] = df["Clin_ANYNODES"].map({"N+": 1, "N0": 0})
    # Ordinal encoding: tumour size categories map to 1, 2, 3.
    # Missing TUMSIZE (n=6) kept as NaN and imputed within each CV fold.
    df["Clin_TUMSIZE"]  = df["Clin_TUMSIZE"].map({"<=20": 1, "21-50": 2, ">50": 3})
    df["Clin_Arm"]      = df["Clin_Arm"].map({"DHP": 0, "T-DM1": 1})
    print("[ENCODE] Clin_ER, Clin_ANYNODES, Clin_TUMSIZE, Clin_Arm encoded")

    # --- 3. Proteomics categorical encoding -----------------------------------
    df["Prot_ERBB2_PG"] = df["Prot_ERBB2_PG"].map({"Positive": 1, "Negative": 0})
    print("[ENCODE] Prot_ERBB2_PG: Positive=1, Negative=0")

    # --- 4. RNA_sspbc.subtype: one-hot encoding (Her2 as reference) ----------
    # This is a multiclass variable (Her2=104, LumA=36, LumB=35, Basal=10).
    # Her2 is used as reference because it is the most common category in this
    # HER2-enriched cohort. Rare categories (Basal, n=5 per arm) may be removed
    # by the NZV filter in arm-specific folds — this is correct and expected.
    subtype_dummies = pd.get_dummies(df["RNA_sspbc.subtype"], prefix="RNA_sspbc")
    # Drop Her2 dummy to avoid perfect multicollinearity (Her2 is the reference)
    subtype_dummies = subtype_dummies.drop(
        columns=["RNA_sspbc_Her2"], errors="ignore"
    )
    df = df.drop(columns=["RNA_sspbc.subtype"])
    df = pd.concat([df, subtype_dummies.astype(float)], axis=1)
    print(f"[ENCODE] RNA_sspbc.subtype → dummies: {subtype_dummies.columns.tolist()}")

    # --- 4b. Coerce any remaining non-numeric feature column ------------------
    # Every modelling column must be numeric by the time it reaches the
    # imputer. The explicit encodings above cover the categorical variables we
    # know about, but data files acquire columns over time — Clin_prolifvalu
    # (Ki67 percentage) arrives as strings, with an 'Unknown' token mixed in
    # among the numbers, and any such column crashes SimpleImputer with
    # "Cannot use median strategy with non-numeric data" only after all the
    # loading work has completed.
    #
    # Coercing here converts numeric strings to numbers and every
    # unrecognised token ('Unknown', 'NA', 'ND', ...) to NaN, which the
    # within-fold median imputer then handles like any other missing value.
    # The conversion is reported rather than silent, because a column that
    # turns out to be entirely non-numeric would otherwise become a silently
    # all-NaN feature.
    FEATURE_PREFIXES = ("Clin_", "RNA_", "DNA_", "Prot_", "WSI_")
    coerced = []
    for col in df.columns:
        if not col.startswith(FEATURE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        before_missing = int(df[col].isna().sum())
        converted = pd.to_numeric(df[col], errors="coerce")
        after_missing = int(converted.isna().sum())
        n_new_nan = after_missing - before_missing
        n_valid = len(converted) - after_missing
        if n_valid == 0:
            print(f"[ENCODE] WARNING: {col} has no numeric values after "
                  f"coercion and will be dropped by the variance filter.")
        df[col] = converted
        coerced.append((col, n_new_nan))

    if coerced:
        print(f"[ENCODE] Coerced {len(coerced)} non-numeric feature column(s) "
              f"to numeric:")
        for col, n_new_nan in coerced:
            print(f"         - {col}: {n_new_nan} unparseable value(s) -> NaN "
                  f"(imputed within each CV fold)")

    # --- 5. Apply Tier 1 biological deduplication ----------------------------
    # Features removed here are BIOLOGICALLY redundant (co-amplicons or
    # near-identical composite scores). This is a domain decision, not a
    # statistical one, and is applied before any train/test splitting. It is
    # outcome-blind: no pCR label is consulted, so it introduces no leakage.
    # --feature_pool full disables it so the run starts from the complete
    # pre-curation feature set.
    if FEATURE_POOL == "full":
        print("[TIER1] SKIPPED (--feature_pool full): starting from the "
              "complete pre-curation feature set")
    else:
        present = [c for c in TIER1_REMOVE if c in df.columns]
        df = df.drop(columns=present)
        print(f"[TIER1] Removed {len(present)} biologically redundant features")
        for feat in present:
            print(f"         - {feat}")

    return df


def define_modality_features(df: pd.DataFrame) -> dict:
    """
    Define the column sets for each modality after Tier 1 deduplication.

    Returns a dict with keys:
      'Clin_global' : all Clin_ columns including Clin_Arm
      'Clin_arm'    : Clin_ columns excluding Clin_Arm (for arm-specific models
                      where treatment arm is constant and non-informative)
      'RNA'         : all RNA_ columns
      'DNA'         : all DNA_ columns
      'Prot'        : all Prot_ columns
      'WSI'         : all WSI_ columns
    """
    features = {}
    features["Clin_global"] = [c for c in df.columns if c.startswith("Clin_")]
    features["Clin_arm"]    = [c for c in features["Clin_global"]
                                if c != "Clin_Arm"]
    features["RNA"]         = [c for c in df.columns if c.startswith("RNA_")]
    features["DNA"]         = [c for c in df.columns if c.startswith("DNA_")]
    features["Prot"]        = [c for c in df.columns if c.startswith("Prot_")]
    features["WSI"]         = [c for c in df.columns if c.startswith("WSI_")]

    print("\n[FEATURE SETS after Tier 1]")
    for name, cols in features.items():
        print(f"  {name:15s}: {len(cols):3d} features")

    return features


def get_complete_case(df: pd.DataFrame, features: dict,
                      active_mods=("Clin", "RNA", "DNA", "Prot", "WSI")
                      ) -> pd.DataFrame:
    """
    Restrict to patients who have ALL ACTIVE modalities measured.

    Rationale: using the same patients across all unimodal AND fused models
    ensures every performance comparison is perfectly paired on the same test
    patients. This avoids the confound of 'model A was evaluated on more or
    different patients than model B'.

    With all five modalities active this is the manuscript's n=110 cohort
    (87 excluded, primarily missing Proteomics — modality-level missingness,
    not random feature dropout; a design decision, not a limitation). With
    --modalities RNA the definition relaxes to every patient with complete
    RNA (n=185), which is the correct cohort for the transcriptomic-only
    model that is validated externally.

    Clin never enters the completeness definition (clinical covariates are
    recorded for everyone).

    Parameters
    ----------
    df          : encoded DataFrame
    features    : dict from define_modality_features()
    active_mods : modalities being modelled in this run (--modalities)

    Returns
    -------
    pd.DataFrame (index reset to 0..n-1 for clean positional indexing)
    """
    all_modality_cols = [c for m in ("RNA", "DNA", "Prot", "WSI")
                         if m in active_mods for c in features[m]]
    df_complete = df.dropna(subset=all_modality_cols).reset_index(drop=True)

    arm0 = (df_complete["Clin_Arm"] == 0).sum()
    arm1 = (df_complete["Clin_Arm"] == 1).sum()
    pcr_global = df_complete["pCR"].mean()
    pcr_arm0   = df_complete.loc[df_complete["Clin_Arm"] == 0, "pCR"].mean()
    pcr_arm1   = df_complete.loc[df_complete["Clin_Arm"] == 1, "pCR"].mean()

    print(f"\n[COMPLETE CASE] n={len(df_complete)} "
          f"(DHP={arm0}, T-DM1={arm1})")
    print(f"[COMPLETE CASE] pCR rate: "
          f"overall={pcr_global:.3f}, DHP={pcr_arm0:.3f}, T-DM1={pcr_arm1:.3f}")

    return df_complete



# ==============================================================================
# SECTION 3b — PER-MODALITY PATIENT DATASETS
# ==============================================================================

def get_modality_datasets(df_enc: pd.DataFrame, features: dict) -> dict:
    """
    For each modality, return ALL patients who have complete data for that
    modality (not just the 110 complete-case patients).

    These expanded datasets are used as training sets: each unimodal model
    trains on ALL patients with that modality minus the current outer test
    patients, rather than being restricted to the 110 complete-case patients.
    The outer TEST sets remain fixed to complete-case patients so that:
      (a) All five modality predictions are always available at test time,
          allowing the fusion model to always run.
      (b) All pairwise performance comparisons remain fully paired on
          identical test patients across all models.

    For arm-specific experiments, modality datasets are filtered to the
    appropriate treatment arm (Clin_Arm == 0 for DHP, == 1 for T-DM1) so
    that T-DM1 patients never enter DHP unimodal training and vice versa.

    Returns
    -------
    dict mapping modality key → pd.DataFrame with patient_id, pCR, and
    all modality feature columns. Only patients with non-null data for that
    modality are included.
    """
    datasets = {}
    for mod_key in ["Clin_global", "Clin_arm", "RNA", "DNA", "Prot", "WSI"]:
        cols = [c for c in features.get(mod_key, []) if c in df_enc.columns]
        feat_cols = [c for c in cols if c not in ["patient_id", "pCR"]]
        if mod_key.startswith("Clin"):
            # Clinical covariates are recorded for everyone; their sporadic
            # item-level missingness (Clin_TUMSIZE NaN, Clin_prolifvalu
            # 'Unknown'→NaN) is imputed WITHIN each CV fold by design.
            # Requiring completeness here silently dropped those patients
            # from the Clin expanded training pool — the expanded Clin model
            # could train on FEWER patients than the complete-case fold.
            # Molecular/imaging modalities keep the completeness rule:
            # their missingness is modality-level (not measured), not
            # item-level.
            mask = df_enc["pCR"].notna()
        elif feat_cols:
            mask = df_enc[feat_cols].notna().all(axis=1) & df_enc["pCR"].notna()
        else:
            mask = df_enc["pCR"].notna()
        # Keep patient_id and pCR alongside features (needed for set operations)
        keep_cols = ["patient_id", "pCR"] + cols
        keep_cols = [c for c in keep_cols if c in df_enc.columns]
        datasets[mod_key] = df_enc.loc[mask, keep_cols].copy().reset_index(drop=True)
    return datasets


# ==============================================================================
# SECTION 4 — WITHIN-FOLD PREPROCESSING UTILITIES
# ==============================================================================

def remove_near_zero_variance(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    freq_threshold:  float = None,
    ratio_threshold: float = None,
) -> tuple:
    """
    Remove near-zero variance (NZV) features — FITTED ON TRAINING SET ONLY.

    A feature is flagged as NZV if either condition holds on the training set:
      (a) The most common value occupies > freq_threshold of training samples.
      (b) The ratio of most-common-value frequency to second-most-common > ratio_threshold.

    The same features are then dropped from the test set.

    Why this matters: binary genomic features (e.g., rare mutation indicators)
    may have near-zero variance in small training folds, making them unstable
    predictors. For arm-specific models (n≈44 training), even features with 5%
    global prevalence may appear in 0–2 cases per inner training fold.

    IMPORTANT — default argument design:
    freq_threshold and ratio_threshold default to None and are resolved inside
    the function body by reading the module-level globals. This avoids the
    Python default-argument capture issue where default values are evaluated at
    function definition time rather than call time. CLI overrides of
    --nzv_freq and --nzv_ratio therefore take effect correctly.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame — must share the same column set.
    freq_threshold  : fraction of training samples occupied by most common value.
                      Defaults to NZV_FREQ_THRESHOLD (read at call time).
    ratio_threshold : freq(top1) / freq(top2) ratio above which a feature is NZV.
                      Defaults to NZV_RATIO_THRESHOLD (read at call time).

    Returns
    -------
    X_train_filtered, X_test_filtered : pd.DataFrame
    removed_features : list of column names removed
    """
    # Resolve defaults at call time so CLI overrides propagate correctly
    if freq_threshold  is None: freq_threshold  = NZV_FREQ_THRESHOLD
    if ratio_threshold is None: ratio_threshold = NZV_RATIO_THRESHOLD

    to_remove = []
    n_train   = len(X_train)

    for col in X_train.columns:
        counts = X_train[col].value_counts(dropna=True)

        # If all values are NaN, remove the feature
        if len(counts) == 0:
            to_remove.append(col)
            continue

        # Condition (a): dominant value frequency
        if counts.iloc[0] / n_train >= freq_threshold:
            to_remove.append(col)
            continue

        # Condition (b): top-1 to top-2 ratio, gated on dominance among the
        # OBSERVED values. Ungated, the ratio rule alone removed any binary
        # feature with carrier prevalence <= 1/(ratio+1) ≈ 4.8% regardless of
        # freq_threshold, silently overriding the --nzv_freq_arm loosening
        # whose whole purpose is to keep low-prevalence mutation indicators
        # (a 2-carrier feature in an arm fold: ratio 42/2 = 21 → removed,
        # contradicting "0.98 keeps features present in >= ~2%"). The gate
        # uses the observed-value denominator (counts.sum(), not n_train) so
        # the rule still catches what it exists for: features whose top value
        # dominates the observed distribution but escapes condition (a)
        # because missingness dilutes the n_train fraction.
        if (len(counts) >= 2
                and (counts.iloc[0] / counts.iloc[1]) >= ratio_threshold
                and (counts.iloc[0] / counts.sum()) >= freq_threshold):
            to_remove.append(col)

    X_train_f = X_train.drop(columns=to_remove)
    X_test_f  = X_test.drop(columns=to_remove)
    return X_train_f, X_test_f, to_remove


def remove_high_correlation(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train=None,
    threshold: float = None,
) -> tuple:
    """
    Remove highly correlated features — FITTED ON TRAINING SET ONLY.

    Algorithm:
      1. Compute the absolute Pearson correlation matrix on X_train for
         features with >2 unique values (continuous/ordinal only).
         Binary features are excluded because their correlations are usually
         lower and they carry distinct biological meaning per category.
      2. For each feature not yet assigned to a cluster, find all features
         correlated with it at |r| >= threshold. This defines a cluster.
      3. Within the cluster, KEEP the feature with the highest univariate
         signal and drop the rest (see "Cluster keeper criterion" below).

    What this function IS: a redundancy-removal step. The correlation-based
    clustering is the selector; it decides which features are drops.
    What this function is NOT: a global feature filter. Features not in any
    correlation cluster pass through untouched. Downstream elastic net + the
    signature-discovery logic in Stage A still handle multivariate selection
    across the full surviving feature set.

    Rationale for redundancy removal: correlated features cause the
    'rotating basis' problem in elastic net — the model selects feature A
    in fold 1 and feature B in fold 2 (both with ~r=0.95), making each
    appear only 50% stable, even though the biological signal is stable
    at 100%. Keeping one representative per cluster resolves this.

    Cluster keeper criterion
    -------------------------
    When y_train is provided AND has both classes, the keeper is the
    cluster member with the highest univariate discrimination measured
    as |AUROC - 0.5| + 0.5 (so an inverse-oriented predictor, AUROC=0.2,
    scores the same as a correctly-oriented AUROC=0.8; we only care
    about magnitude of signal, not direction). This replaces an older
    variance-based keeper which had three problems:
      1. Variance is scale-dependent and the correlation filter runs
         BEFORE standardization, so features on wider raw scales won
         regardless of informativeness.
      2. Variance is outlier-dominated — a single extreme value in a
         gene's training-fold expression can flip the keeper, and a
         different fold picks a different keeper, reintroducing the
         same instability the filter was meant to fix.
      3. Variance is y-blind, so the "representative" feature kept for
         downstream SHAP and interpretation was not selected for
         predictive content.

    Univariate AUROC fixes all three: it is rank-based (scale-invariant
    and outlier-robust) and y-aware. It is computed leakage-free on
    training data only.

    Tiebreaker: when y_train is missing, has one class, or AUROC cannot
    be computed, the function falls back to variance (the original
    behaviour). When both members of a cluster have AUROC exactly 0.5
    (both uninformative), variance is used as the secondary tiebreaker.

    The Tier 3 correlation filter is only applied to RNA and DNA.
    Clin, Prot, and WSI (3–5 features each) are too low-dimensional for
    aggressive correlation pruning — removing any feature from a 3-feature
    space (WSI) risks a degenerate model. Elastic net's L2 component handles
    residual correlation in small modalities adequately.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame — must share the same column set.
    y_train         : np.array or None — pCR labels for training patients,
                      same row order as X_train. Used only to pick the
                      keeper within correlation clusters.
    threshold       : |r| above which two features are considered redundant.

    Returns
    -------
    X_train_filtered, X_test_filtered : pd.DataFrame
    removed_features : list of column names removed
    """
    # Apply only to continuous/ordinal features (>2 unique values)
    candidate_cols = [c for c in X_train.columns if X_train[c].nunique() > 2]

    # Resolve default at call time so --corr_threshold CLI override takes effect
    if threshold is None:
        threshold = CORR_THRESHOLD

    if len(candidate_cols) < 2:
        return X_train, X_test, []

    corr_matrix = X_train[candidate_cols].corr().abs()

    # Validate y_train for use in the keeper criterion
    use_auroc = False
    if y_train is not None:
        y_arr = np.asarray(y_train)
        # Strip any NaN labels defensively (should not occur on CC training)
        y_mask = ~pd.isna(y_arr)
        if y_mask.sum() >= 3 and len(np.unique(y_arr[y_mask])) >= 2:
            y_arr     = y_arr[y_mask].astype(float)
            y_mask_np = y_mask
            use_auroc = True

    def _univariate_auroc_score(feat_col):
        """|AUROC - 0.5| + 0.5, with median imputation for any NaN feature values."""
        x = X_train[feat_col].values.astype(float)
        if use_auroc:
            x = x[y_mask_np]
        if np.isnan(x).any():
            med = np.nanmedian(x)
            x = np.where(np.isnan(x), med if not np.isnan(med) else 0.0, x)
        if len(np.unique(x)) < 2:
            return 0.5   # constant feature: no signal
        try:
            return abs(float(roc_auc_score(y_arr, x)) - 0.5) + 0.5
        except Exception:
            return 0.5

    # ── Connected components via BFS ──────────────────────────────────────
    # A greedy "star" approach (for each feature, find all features directly
    # correlated with it) misses transitive chains:
    #   A-B >= threshold, B-C >= threshold, A-C < threshold
    # → star clustering creates {A,B} and {C}, losing the B-C relationship.
    # BFS connected components correctly groups {A, B, C} in this case.
    adj = defaultdict(set)
    for i, fa in enumerate(candidate_cols):
        for fb in candidate_cols[i+1:]:
            if corr_matrix.loc[fa, fb] >= threshold:
                adj[fa].add(fb)
                adj[fb].add(fa)

    visited  = set()
    to_remove = []
    decided   = set()

    for start in candidate_cols:
        if start in visited:
            continue
        # BFS to find connected component
        cluster = []
        queue   = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend(adj[node] - visited)

        decided.update(cluster)

        if len(cluster) <= 1:
            continue

        # Pick the keeper within the cluster
        if use_auroc:
            scores    = {c: _univariate_auroc_score(c) for c in cluster}
            max_score = max(scores.values())
            top       = [c for c, s in scores.items() if s == max_score]
            if len(top) == 1:
                keeper = top[0]
            else:
                variances = X_train[top].var()
                keeper    = variances.idxmax()
        else:
            variances = X_train[cluster].var()
            keeper    = variances.idxmax()

        removals = [c for c in cluster if c != keeper]
        to_remove.extend(removals)

    to_remove = list(set(to_remove))

    X_train_f = X_train.drop(columns=to_remove)
    X_test_f  = X_test.drop(columns=to_remove)
    return X_train_f, X_test_f, to_remove


# ==============================================================================
# SECTION 4b — TIER 2.5: IN-FOLD UNIVARIATE OUTCOME SCREEN
# ==============================================================================
# This is the step that resolves the candidate-pool leakage identified in
# peer review. In the original submission the univariate association with pCR
# was evaluated ONCE on the whole cohort, before cross-validation, and used to
# decide which features entered the candidate panel. Every test patient
# therefore contributed to the choice of candidates, and the cross-validated
# performance was optimistic.
#
# The functions below perform exactly the same screening operation, but fitted
# on training patients only, separately in every outer fold and every inner
# fold. No test patient influences which features enter the model.
# ==============================================================================

def _mannwhitney_auroc_and_p(X: np.ndarray, y: np.ndarray):
    """
    Vectorised tie-corrected Mann-Whitney U test for every column of X against
    the binary outcome y. Equivalent to a univariate AUROC per feature plus its
    two-sided p-value under the normal approximation.

    This statistic was chosen over a t-test or univariate logistic regression
    because it is:
      - rank based, hence scale invariant and insensitive to the heavy-tailed
        distributions typical of expression and CNA data;
      - valid for the binary mutation indicators (heavy ties) once the tie
        correction below is applied;
      - computable for all features of a modality in a single pass, which
        matters because this now runs inside every inner fold of every outer
        fold rather than once on the whole cohort.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features) — no NaNs (already imputed).
    y : np.ndarray (n_samples,) — binary outcome (0/1).

    Returns
    -------
    auroc : np.ndarray (n_features,) — univariate AUROC, P(x_pos > x_neg).
    pval  : np.ndarray (n_features,) — two-sided p-value. 1.0 for degenerate
            (constant) features.
    """
    from scipy.stats import rankdata, norm

    y = np.asarray(y, dtype=float)
    pos = (y == 1)
    n1  = int(pos.sum())
    n0  = int((~pos).sum())
    n   = n1 + n0
    n_feat = X.shape[1]

    if n1 == 0 or n0 == 0 or n < 3:
        return np.full(n_feat, 0.5), np.ones(n_feat)

    # Column-wise ranks with average ranks for ties.
    ranks = rankdata(X, axis=0)

    R1 = ranks[pos].sum(axis=0)
    U1 = R1 - n1 * (n1 + 1) / 2.0
    auroc = U1 / (n1 * n0)

    # Tie correction: sum over tie groups of (t^3 - t), computed per column.
    tie_term = np.zeros(n_feat, dtype=float)
    for j in range(n_feat):
        _, counts = np.unique(ranks[:, j], return_counts=True)
        t = counts[counts > 1].astype(float)
        if t.size:
            tie_term[j] = float(np.sum(t ** 3 - t))

    mu    = n1 * n0 / 2.0
    var   = (n1 * n0 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    var   = np.maximum(var, 1e-12)
    # Continuity correction, matching scipy.stats.mannwhitneyu's default
    # (use_continuity=True). It shrinks |U - mu| by 0.5, giving slightly
    # larger p-values — the conservative direction, which is what we want for
    # a screening step, and it makes these p-values reproduce scipy's exactly
    # so the implementation can be checked against a standard reference.
    num   = np.abs(U1 - mu) - 0.5
    z     = np.where(num > 0, num, 0.0) / np.sqrt(var)
    pval  = 2.0 * norm.sf(z)

    # Constant features (single tie group spanning all samples) carry no signal.
    constant = np.array(
        [len(np.unique(X[:, j])) < 2 for j in range(n_feat)], dtype=bool)
    auroc[constant] = 0.5
    pval[constant]  = 1.0

    return auroc, np.clip(pval, 0.0, 1.0)


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg step-up FDR adjustment. Returns q-values in the same
    order as the input. Monotonicity is enforced from the largest p downwards,
    and q-values are capped at 1.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order   = np.argsort(p)
    ranked  = p[order]
    q_sorted = ranked * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty(m, dtype=float)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def univariate_screen_indices(X_train_scaled: np.ndarray,
                              y_train,
                              columns,
                              fdr_q: float = 0.25,
                              max_k: int = None,
                              min_k: int = 5):
    """
    Select candidate features by univariate association with the outcome,
    FITTED ON TRAINING DATA ONLY.

    Pre-specified rule (applied identically in every fold):
      1. Compute the tie-corrected Mann-Whitney univariate AUROC and p-value
         for every feature against pCR on the training patients.
      2. Adjust p-values within the modality using Benjamini-Hochberg and
         retain features with q <= fdr_q.
      3. Cap the retained set at max_k features, keeping the strongest by
         |AUROC - 0.5|.
      4. Floor: if fewer than min_k features survive, restore the top min_k by
         |AUROC - 0.5| so the modality can never collapse to an empty or
         degenerate design matrix.

    fdr_q defaults to 0.25 rather than the conventional 0.05 because this is a
    SCREENING step, not an inference step: its job is to reduce a
    high-dimensional pool to a tractable candidate set before the multivariable
    signature discovery in Stage A does the actual selection. A stringent
    threshold at n≈50-110 training patients would discard features whose
    contribution is only visible multivariably. The threshold is pre-specified
    and identical across folds, arms and modalities, and — critically — it is
    computed without ever seeing the test patients.

    Returns
    -------
    keep_idx : list[int] — column indices to retain, in original column order.
    stats    : dict — per-feature auroc/p/q plus the realised counts, recorded
               so the screen can be audited fold by fold.
    """
    cols = list(columns)
    n_feat = len(cols)
    if n_feat == 0:
        return [], {}

    y = np.asarray(y_train, dtype=float)
    if len(np.unique(y[~np.isnan(y)])) < 2:
        # Degenerate outcome in this fold — screening is undefined; keep all.
        return list(range(n_feat)), {"skipped": "single_class_outcome"}

    auroc, pval = _mannwhitney_auroc_and_p(np.asarray(X_train_scaled, float), y)
    qval = benjamini_hochberg(pval)
    strength = np.abs(auroc - 0.5)

    if max_k is None:
        max_k = n_feat

    passing = np.where(qval <= fdr_q)[0]
    if len(passing) > max_k:
        passing = passing[np.argsort(-strength[passing])[:max_k]]

    if len(passing) < min(min_k, n_feat):
        passing = np.argsort(-strength)[:min(min_k, n_feat)]

    keep_idx = sorted(int(i) for i in passing)

    stats = {
        "n_input":     n_feat,
        "n_retained":  len(keep_idx),
        "fdr_q":       float(fdr_q),
        "max_k":       int(max_k),
        "min_k":       int(min_k),
        "floor_used":  bool(len(np.where(qval <= fdr_q)[0]) < min(min_k, n_feat)),
        "auroc":       {cols[i]: float(auroc[i]) for i in keep_idx},
        "qval":        {cols[i]: float(qval[i])  for i in keep_idx},
    }
    return keep_idx, stats


def fit_imputer_scaler(X_train: pd.DataFrame,
                       y_train=None,
                       screen_cfg: dict = None) -> dict:
    """
    Fit median imputer and StandardScaler on the training set, and — when
    screen_cfg is supplied — the Tier 2.5 in-fold univariate outcome screen.

    Median imputation is used because:
      - Clinical ordinal variables (Clin_TUMSIZE) have a few missing values.
      - Median is robust to outliers common in biomedical data.

    StandardScaler ensures all features are on the same scale before
    logistic regression, regardless of original units (mRNA counts, CNA ratios,
    cell proportions, etc.).

    CRITICAL: every transformer here — imputer, scaler, AND the univariate
    screen — is fitted on training data only. Applying test-set statistics to
    imputation, scaling, or (most importantly) to the choice of which features
    enter the candidate pool would constitute data leakage.

    The screen is applied AFTER imputation and scaling so it operates on a
    complete matrix. Because the statistic is rank based, the affine scaling
    does not change which features it selects; running it here simply avoids
    having to handle missing values twice.

    Returns
    -------
    preprocessor : dict with keys
        'imputer', 'scaler'
        'pre_screen_columns' : columns the imputer/scaler were fitted on
        'columns'            : final columns AFTER screening (what callers see)
        'screen_idx'         : indices into pre_screen_columns, or None
        'screen_stats'       : audit dict from univariate_screen_indices
    """
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()

    X_arr = imputer.fit_transform(X_train)
    scaler.fit(X_arr)

    cols = X_train.columns.tolist()
    prep = {
        "imputer":            imputer,
        "scaler":             scaler,
        "pre_screen_columns": cols,
        "columns":            cols,
        "screen_idx":         None,
        "screen_stats":       None,
    }

    if screen_cfg and y_train is not None and len(cols) > 0:
        X_scaled = scaler.transform(X_arr)
        keep_idx, stats = univariate_screen_indices(
            X_scaled, y_train, cols, **screen_cfg)
        if keep_idx and len(keep_idx) < len(cols):
            prep["screen_idx"] = keep_idx
            prep["columns"]    = [cols[i] for i in keep_idx]
        prep["screen_stats"] = stats

    return prep


def apply_imputer_scaler(X: pd.DataFrame, preprocessor: dict) -> np.ndarray:
    """
    Apply a fitted imputer + scaler (+ univariate screen) to a new dataset.

    Columns not present in the fitted preprocessor are silently set to NaN
    (imputer handles them). This guards against edge cases where test set
    columns differ after preprocessing-step filtering.

    The returned array is restricted to preprocessor["columns"], i.e. to the
    features that survived the in-fold univariate screen when one was fitted.
    """
    # Backwards compatibility with preprocessor dicts written before the
    # Tier 2.5 screen existed (they carry only "columns").
    fit_cols = preprocessor.get("pre_screen_columns") or preprocessor["columns"]
    X_aligned = X.reindex(columns=fit_cols, fill_value=np.nan)
    X_imputed = preprocessor["imputer"].transform(X_aligned)
    X_scaled  = preprocessor["scaler"].transform(X_imputed)
    screen_idx = preprocessor.get("screen_idx")
    if screen_idx is not None:
        X_scaled = X_scaled[:, screen_idx]
    return X_scaled


def _resolve_screen_cfg(n_features: int, apply_screen: bool):
    """
    Build the screen_cfg dict handed to fit_imputer_scaler, or None when the
    Tier 2.5 screen is disabled.

    The screen is skipped entirely for modalities that are already small
    enough that no screening is meaningful (<= UNIV_SCREEN_MIN_FEATURES
    columns). This mirrors the SMALL_MODS carve-out in _derive_signature:
    Clin (5 features) and WSI (3 features) keep all features, so applying a
    univariate filter to them would only add variance without reducing
    dimensionality.
    """
    if not apply_screen or not UNIVARIATE_SCREEN:
        return None
    if n_features <= UNIV_SCREEN_MIN_FEATURES:
        return None
    return {
        "fdr_q": UNIV_SCREEN_FDR_Q,
        "max_k": UNIV_SCREEN_MAX_K,
        "min_k": UNIV_SCREEN_MIN_K,
    }


def preprocess_fold(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    apply_corr_filter: bool = True,
    y_train=None,
    apply_univariate_screen: bool = True,
) -> tuple:
    """
    Full preprocessing pipeline for a single (train, test) fold pair.

    Order of operations (ALL fitted on training, applied to test):
      1. Tier 2: Near-zero variance removal
      2. Tier 3: High-correlation filter (only if apply_corr_filter=True).
         When y_train is supplied, the per-cluster keeper is chosen by
         univariate AUROC (rank-based, y-aware, outlier-robust) instead
         of variance. See remove_high_correlation docstring.
      3. Median imputation + StandardScaling
      4. Tier 2.5: in-fold univariate outcome screen (only when the global
         UNIVARIATE_SCREEN flag is on and y_train is supplied). Numbered 2.5
         because it belongs conceptually with the other filters, but it is
         executed after imputation so it sees a complete matrix.

    Step 4 is the leakage fix. Every caller of this function — outer folds,
    inner folds, the OOF generator, the consensus refit — therefore performs
    the univariate association step on its own training patients only.

    The apply_corr_filter flag is False for Clin, Prot, and WSI because:
      - They have ≤5 features; elastic net's L2 component handles residual
        correlation adequately without explicit pruning.
      - Removing a feature from a 3-feature space (WSI) risks a degenerate model.

    Returns
    -------
    X_train_proc : np.ndarray  (processed training features)
    X_test_proc  : np.ndarray  (processed test features)
    preprocessor : dict        (fitted imputer + scaler + screen)
    removed_nzv  : list        (Tier 2 removed features)
    removed_corr : list        (Tier 3 removed features)
    final_cols   : list        (feature names after all preprocessing)
    """
    # Step 1: NZV removal (Tier 2)
    X_tr2, X_te2, removed_nzv = remove_near_zero_variance(X_train, X_test)

    # Step 2: Correlation filter (Tier 3) — only for high-dimensional modalities
    if apply_corr_filter and X_tr2.shape[1] > 3:
        X_tr3, X_te3, removed_corr = remove_high_correlation(
            X_tr2, X_te2, y_train=y_train)
    else:
        X_tr3, X_te3, removed_corr = X_tr2, X_te2, []

    # Degenerate fold: NZV (plus the correlation filter) removed every
    # column. SimpleImputer raises on a 0-column matrix, and an unguarded
    # raise here kills the entire multi-hour Parallel run from one bad fold.
    # Return an empty-but-well-formed result; downstream code treats an
    # empty final-column list as a neutral fold.
    if X_tr3.shape[1] == 0:
        empty_prep = {"imputer": None, "scaler": None,
                      "pre_screen_columns": [], "columns": [],
                      "screen_idx": None, "screen_stats": None}
        return (np.empty((len(X_tr3), 0)), np.empty((len(X_te3), 0)),
                empty_prep, removed_nzv, removed_corr, [])

    # Steps 3 + 4: imputation, scaling, and the in-fold univariate screen —
    # all fitted on training only.
    screen_cfg   = _resolve_screen_cfg(X_tr3.shape[1], apply_univariate_screen)
    preprocessor = fit_imputer_scaler(X_tr3, y_train=y_train,
                                      screen_cfg=screen_cfg)
    X_train_proc = apply_imputer_scaler(X_tr3, preprocessor)
    X_test_proc  = apply_imputer_scaler(X_te3, preprocessor)

    return (
        X_train_proc,
        X_test_proc,
        preprocessor,
        removed_nzv,
        removed_corr,
        preprocessor["columns"],  # final feature names (post-screen)
    )





# ==============================================================================
# SECTION 3d — FOLD-LEVEL METRICS
# ==============================================================================

def compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute discrimination and calibration metrics for a single outer fold.

    Metrics
    -------
    AUROC       : Area under the ROC curve. Primary discrimination metric.
    AUPRC       : Area under the precision-recall curve. Complementary to
                  AUROC; more sensitive to performance on the positive class.
    Brier       : Proper scoring rule (calibration + discrimination jointly).
    Sensitivity : True positive rate at the Youden-optimal threshold.
                  Youden index = Sensitivity + Specificity - 1 (maximised).
                  Using the fold-specific optimal threshold avoids committing
                  to a fixed operating point and gives the best-achievable
                  sensitivity/specificity pair for the model in that fold.
    Specificity : True negative rate at the same Youden-optimal threshold.

    Note: Sensitivity and Specificity are threshold-dependent. The Youden-
    optimal threshold maximises their sum and is the standard reporting choice
    for binary classifiers when no clinical cost asymmetry is specified.
    """
    if len(np.unique(y_true)) < 2:
        print("  [WARN] Degenerate fold: only one class in y_true")
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan,
                "Sensitivity": np.nan, "Specificity": np.nan,
                "Threshold": np.nan}

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Youden index: maximise TPR + TNR - 1  ≡  maximise TPR - FPR
    youden_idx  = np.argmax(tpr - fpr)
    best_thresh = float(thresholds[youden_idx])
    sensitivity = float(tpr[youden_idx])
    specificity = float(1.0 - fpr[youden_idx])

    return {
        "AUROC":       float(roc_auc_score(y_true, y_pred)),
        "AUPRC":       float(average_precision_score(y_true, y_pred)),
        "Brier":       float(brier_score_loss(y_true, y_pred)),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Threshold":   best_thresh,
    }


def compute_pooled_metrics(fold_list) -> dict:
    """
    Compute AUROC / Sensitivity / Specificity / Threshold on predictions
    POOLED across all folds (concatenated y_test, y_pred) rather than as
    the mean of per-fold values.

    Why this matters for Sens/Spec specifically
    -------------------------------------------
    Per-fold Sens/Spec are computed at the Youden-optimal threshold of
    that fold. With only ~30 test patients per fold, the Youden threshold
    is high-variance and optimistically chosen: it's the pair of
    (TPR, TNR) that happens to maximise TPR+TNR-1 on that specific fold's
    ROC curve. Averaging per-fold Sens/Spec therefore reports the
    *upper envelope* of achievable operating points, not the operating
    point you'd get if you deployed the model.

    Pooling y_test and y_pred across folds first, then picking ONE Youden
    threshold on the aggregated data, gives:
      - a single, reproducible threshold ("deploy this one")
      - a single Sens/Spec pair with honest variance (~N_total patients
        worth of statistics rather than mean of 1000 × 30-patient estimates)
      - a meaningful "Threshold" to report in the paper's operating-point
        table

    AUROC is threshold-free and is essentially unchanged between per-fold
    mean and pooled computation. AUPRC and Brier shift slightly; the
    pooled values are closer to what a real deployment would achieve.

    Returns same keys as compute_fold_metrics. Returns NaN dict if pooled
    data is degenerate.
    """
    if not fold_list:
        return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan,
                "Sensitivity": np.nan, "Specificity": np.nan,
                "Threshold": np.nan, "N_pooled": 0}

    y_true = np.concatenate([np.asarray(f["y_test"], dtype=float)
                             for f in fold_list])
    y_pred = np.concatenate([np.asarray(f["y_pred"], dtype=float)
                             for f in fold_list])

    pooled = compute_fold_metrics(y_true, y_pred)
    pooled["N_pooled"] = int(len(y_true))
    return pooled

# ==============================================================================
# SECTION 3c — SIGNATURE DISCOVERY: HELPER FUNCTIONS
# ==============================================================================

# (_map_cc_splits_to_expanded was removed 2026-08-14: it existed solely to
# feed Stage B's GridSearchCV over the outer-preprocessed matrix, which was
# replaced by the in-fold, signature-restricted manual grid in
# _fit_signature_model.)

def _compute_inner_importance(clf_name, model, X_train_p, fcols_inner_list):
    """
    Compute per-feature importance for one fitted classifier on inner
    TRAINING data (NOT validation — this avoids signature-selection optimism
    where the signature is tuned to the val-fold feature distribution and
    then scored on that same val fold in Stage A Pass 2).

    Linear classifiers (ElasticNet_LR, SVM_Linear):
      importance[f] = |coef[f]| / max(|coef|)   — normalised coefficient magnitude.

      Rationale for normalised magnitude rather than binary selection (0/1):
        Binary selection discards coefficient magnitude completely — every
        selected feature receives importance=1.0 regardless of whether its
        coefficient is 0.80 or 0.005. When converted to percentile ranks in
        Stage A Pass 1, every selected feature gets an identical rank, making
        the 25th-percentile filter in _derive_signature entirely ineffective
        for linear classifiers: it cannot distinguish a strongly weighted
        feature from a weakly weighted one.

        Normalised |coef| (divided by this fold's maximum |coef|) preserves
        relative magnitude in a scale-invariant way. Features with larger
        regularised weights consistently receive higher ranks across folds.
        Zero-coefficient features map to importance=0.0 and rank at the
        bottom, preserving the implicit selection behaviour of L1
        regularisation. The normalisation is fold-local, consistent with
        tree classifiers where SHAP values are also fold-local and
        immediately converted to percentile ranks.

    Tree classifiers (RandomForest, ExtraTrees, HistGradBoost):
      importance[f] = mean |SHAP| over inner TRAINING patients.
      Uses shap.TreeExplainer with model_output='probability'.
      Computed on inner-training rather than inner-val to remove the
      structural dependence of the derived signature on the val-fold feature
      distribution. Pass 2's scoring on inner-val is therefore a clean
      hold-out for the signature.

    SVM_RBF (shap_type='none'):
      Returns {} — classifier excluded from signature discovery.

    Returns
    -------
    dict {feature_name: importance_value}  (empty on failure)
    """
    stype = CLASSIFIERS.get(clf_name, {}).get("shap_type", "none")
    imp   = {}
    try:
        if stype in ("linear", "linear_svm"):
            abs_coefs = np.abs(model.coef_[0].astype(float))
            max_coef  = float(abs_coefs.max())
            # Normalise by fold-local maximum so scale is invariant to
            # regularisation strength. Degenerate all-zero case → uniform 0.
            denom = max_coef if max_coef > 1e-12 else 1.0
            for feat, ac in zip(fcols_inner_list, abs_coefs):
                imp[feat] = float(ac) / denom

        elif stype == "tree":
            exp = shap.TreeExplainer(model, data=X_train_p,
                                     feature_names=fcols_inner_list,
                                     model_output="probability")
            sv = exp.shap_values(X_train_p)
            if isinstance(sv, list): sv = sv[1]
            elif sv.ndim == 3:       sv = sv[:, :, 1]
            mean_abs = np.abs(sv).mean(axis=0)
            for feat, val in zip(fcols_inner_list, mean_abs):
                imp[feat] = float(val)
        # stype == "none": return {} → feature never selected → clf never wins
    except Exception as e:
        print(f"  [WARN] _compute_inner_importance failed for {clf_name}: "
              f"{type(e).__name__}: {e}")
    return imp


def _derive_signature(importance_dict, mod, n_events_expanded):
    """
    Select the feature signature from cross-classifier percentile-rank importance.

    Rules by modality:

    ── Small modalities: Clin, WSI ─────────────────────────────────────────────
    Keep ALL features. These modalities have ≤5 features; the elastic net's
    L1/L2 regularisation handles non-informative features by shrinking their
    coefficients toward zero. Removing any feature from a 3-feature space (WSI)
    risks a degenerate model and provides no methodological benefit.

    ── High-dimensional modalities: RNA, DNA, Prot ─────────────────────────────
    Three constraints applied in sequence:

    1. EPV ceiling: max_k = max(floor(n_pCR_events / EPV=5), FLOOR=5).
       Hard upper bound grounded in the events-per-variable literature, adjusted
       for regularised models (EPV=5 vs the classical EPV=10 for OLS/unregularised
       logistic regression). FLOOR=5 ensures a minimum of 5 features regardless
       of the EPV cap, including the T-DM1/Prot case where EPV=5 gives 4.

    2. 25th-percentile filter within cap: among the top max_k features by
       mean cross-classifier percentile-rank importance, drop those whose score
       falls below the 25th percentile of the retained set. This removes the
       bottom quartile — features the classifiers consistently ranked as least
       informative — while retaining all features with meaningful cross-classifier
       consensus.

    3. Floor protection: if the percentile filter would reduce the set below
       FLOOR=5, restore the top FLOOR features by importance rank regardless
       of the percentile threshold. This prevents over-pruning in arm scenarios
       where the importance distribution is compressed.

    Parameters
    ----------
    importance_dict   : {feature_name: mean_cross_classifier_percentile_rank}
                        Keys are features surviving outer NZV/corr preprocessing.
    mod               : str — modality name (Clin, RNA, DNA, Prot, WSI).
    n_events_expanded : int — pCR=1 count in the expanded outer training set.

    Returns
    -------
    list of feature names in descending importance order.
    """
    if not importance_dict:
        return []

    SMALL_MODS = {"Clin", "WSI"}
    EPV   = 5
    FLOOR = 5

    ranked = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)

    if mod in SMALL_MODS:
        return [f for f, _ in ranked]

    # ── EPV cap (with floor) ──────────────────────────────────────────────────
    max_k   = max(int(n_events_expanded // EPV), FLOOR)
    capped  = ranked[:max_k]

    if len(capped) <= FLOOR:
        return [f for f, _ in capped]

    # ── 25th-percentile filter within cap ────────────────────────────────────
    vals    = np.array([v for _, v in capped])
    p25     = float(np.percentile(vals, 25))
    filtered = [(f, v) for f, v in capped if v >= p25]

    # ── Floor protection ─────────────────────────────────────────────────────
    if len(filtered) < FLOOR:
        filtered = capped[:FLOOR]

    return [f for f, _ in filtered]


def _check_calibration(y_true_list, y_pred_list,
                       clf_name, mod, exp_name, fold_idx):
    """
    Estimate Platt calibration slope from inner-loop OOF predictions —
    DIAGNOSTIC ONLY. Since the pipeline now always applies calibration
    (see `_apply_global_calibration` below), this function is retained
    for its slope/Brier diagnostics in the terminal log; its return
    values are recorded in `fold_dict["calibration"]` for transparency
    but are no longer used as a gate.

    Fits LogisticRegression(C=1e6) on predicted_prob → true_label to obtain
    the Platt slope:
      slope ≈ 1.0 : well calibrated
      slope < 0.80: probabilities compressed  (RF/ET typical)
      slope > 1.20: probabilities overconfident

    Returns
    -------
    slope       : float  — Platt calibration slope
    needs_platt : bool   — True iff slope ∉ [0.80, 1.20]  (recorded but unused)
    diag        : str    — formatted diagnostic line for terminal output
    """
    y_true = np.array(y_true_list, dtype=float)
    y_pred = np.array(y_pred_list, dtype=float)

    tag = f"[CAL] {exp_name}/{mod} fold={fold_idx+1:04d} clf={clf_name:<15}"

    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return 1.0, False, f"{tag} → insufficient data (n={len(y_true)})"

    try:
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        cal.fit(y_pred.reshape(-1, 1), y_true)
        slope = float(cal.coef_[0][0])

        brier_raw = float(brier_score_loss(y_true, y_pred))
        cal_prob  = cal.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        brier_cal = float(brier_score_loss(y_true, cal_prob))
        delta_b   = brier_raw - brier_cal

        needs_platt = not (0.80 <= slope <= 1.20)
        status      = ("COMPRESSED"    if slope < 0.80
                       else "OVERCONF" if slope > 1.20
                       else "OK")
        action      = " ★ Platt APPLIED" if needs_platt else ""
        diag = (f"{tag} slope={slope:.3f} [{status}]  "
                f"Brier {brier_raw:.4f}→{brier_cal:.4f} "
                f"(Δ={delta_b:+.4f}){action}")
        return slope, needs_platt, diag

    except Exception as e:
        return 1.0, False, f"{tag} → failed: {e}"


def _fit_global_platt(y_true, y_pred_raw):
    """
    Fit a 2-parameter Platt sigmoid on (raw predicted probability → label)
    using all available inner-OOF data for the winner classifier. Returns
    a fitted LogisticRegression on a single feature (raw score).

    Why global OOF calibration instead of nested CalibratedClassifierCV:
    - The calibrator sees ALL cc-training-fold OOF predictions (~80-120
      patients) instead of being refit inside cv=3 splits of ~30 patients.
    - Sigmoid has 2 parameters (slope + intercept) — 30 patients is
      noticeably underpowered for this; 100+ is stable.
    - Applied uniformly to every modality: all OOF columns entering fusion
      are on the same (calibrated) probability scale, removing the
      heteroscedasticity that arises when some modalities are Platt-wrapped
      and others are not.

    Assumption: the calibration curve of the outer-refit model is close to
    that of the inner-fold models. For a 2-parameter sigmoid this is
    usually a safe assumption — slope/intercept calibration captures
    systematic, estimator-family miscalibration that is fairly stable
    across training set sizes.

    Returns None if there is insufficient data or only one class present.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred_raw, dtype=float)
    # Guard rails
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return None
    try:
        cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        cal.fit(y_pred.reshape(-1, 1), y_true)
        return cal
    except Exception:
        return None


def _apply_global_platt(calibrator, y_pred_raw):
    """Apply a fitted Platt calibrator to an array of raw probabilities."""
    if calibrator is None:
        return np.asarray(y_pred_raw, dtype=float)
    y = np.asarray(y_pred_raw, dtype=float).reshape(-1, 1)
    return calibrator.predict_proba(y)[:, 1]


def make_oof_signature(clf_name, best_params, signature_feats,
                       cc_train_raw_df, y_cc_train, cc_train_pids,
                       mod_full_df, feat_cols_raw, test_pids_set,
                       inner_splits, ac,
                       fold_cache=None, inner_jobs=1):
    """
    Generate RAW (uncalibrated) OOF probability scores for the complete-case
    training patients using the winner classifier on the winner signature
    features.

    CALIBRATION IS APPLIED SEPARATELY.
    The caller (_fit_signature_model) fits a global Platt sigmoid on
    (raw_OOF, y_cc_train) after this function returns, then applies it to
    both OOF and outer-test predictions. This centralises calibration,
    gives the sigmoid maximum training data (~100 patients vs ~30 in the
    former nested cv=3 approach), and ensures all modality OOF columns
    entering fusion are on the same calibrated scale.

    PERFORMANCE NOTE
    ----------------
    Stage A Pass 1 already preprocessed each inner fold and cached the result
    in `fold_cache`. We accept that cache here and reuse the preprocessed
    arrays instead of re-running NZV → correlation filter → impute → scale,
    which is the dominant per-fold cost for RNA/DNA modalities. Only the
    model fit itself needs to be repeated (different classifier / params /
    feature subset). When `fold_cache=None` the function falls back to the
    original from-raw behaviour for backwards compatibility.

    For each inner fold:
      Inner training : all modality patients minus (outer_test ∪ inner_val_cc),
                       preprocessed from raw (or reused from fold_cache).
      Inner validation: cc training patients at inner split relative indices.
      Features used   : intersection of signature_feats with inner fold's
                        post-preprocessing column set (some features may be
                        removed by inner-fold NZV/correlation filters).

    Returns
    -------
    np.array of shape (len(y_cc_train),) — RAW OOF probabilities.
    Unfitted positions (failed inner folds) default to 0.5 (neutral).
    """
    oof    = np.full(len(y_cc_train), np.nan)
    failed = 0
    cfg    = CLASSIFIERS[clf_name]
    sig_set = set(signature_feats)
    use_cache = fold_cache is not None and len(fold_cache) == len(inner_splits)

    for fold_pos, (i_tr_rel, i_va_rel) in enumerate(inner_splits):
        if use_cache and fold_cache[fold_pos] is not None:
            # Reuse preprocessed arrays from Stage A Pass 1
            X_itr_p, y_itr, X_iva_p, _y_iva_cached, fcols_inner_list = \
                fold_cache[fold_pos]
        else:
            # Fallback: preprocess from raw
            X_iva_raw = cc_train_raw_df.iloc[i_va_rel]
            val_pids  = set(int(p) for p in cc_train_pids[i_va_rel])
            excluded  = test_pids_set | val_pids
            itr_mask  = ~mod_full_df["patient_id"].isin(excluded)
            X_itr_raw = mod_full_df.loc[itr_mask, feat_cols_raw]
            y_itr     = mod_full_df.loc[itr_mask, "pCR"].values

            if len(X_itr_raw) == 0 or len(np.unique(y_itr)) < 2:
                failed += 1; continue

            try:
                X_itr_p, X_iva_p, fcols_inner = preprocess_fold_3(
                    X_itr_raw, X_iva_raw, ac, y_train=y_itr)
                fcols_inner_list = list(fcols_inner)
            except Exception:
                failed += 1; continue

        # Intersect signature with inner-fold surviving features
        sig_idx = [i for i, f in enumerate(fcols_inner_list) if f in sig_set]
        if len(sig_idx) == 0:
            failed += 1; continue

        try:
            X_itr_sig = X_itr_p[:, sig_idx]
            X_iva_sig = X_iva_p[:, sig_idx]

            m = cfg["build"]()
            m.set_params(**_params_with_inner_jobs(
                clf_name, best_params, inner_jobs))
            # No CalibratedClassifierCV wrap here — produces RAW OOF.
            # Calibration is applied globally by the caller.

            m.fit(X_itr_sig, y_itr)
            oof[i_va_rel] = m.predict_proba(X_iva_sig)[:, 1]

        except Exception:
            failed += 1

    if failed == len(inner_splits):
        print(f"  [WARN] make_oof_signature: all inner folds failed "
              f"for {clf_name} — using neutral 0.5")
    return np.where(np.isnan(oof), 0.5, oof)


def _neutral_fold_result(y_te, y_cc_train, fcols_list, reason="all classifiers failed"):
    """Return neutral (0.5) predictions with complete fold_dict structure."""
    print(f"  [WARN] _neutral_fold_result: {reason}")
    # Neutral cross-arm predictor: returns NaN for any input → filtered out
    # downstream. Keeps the dict shape consistent.
    def _neutral_predict(X_raw_df):
        return np.full(len(X_raw_df), np.nan)
    return {
        "metrics":          compute_fold_metrics(y_te, np.full(len(y_te), 0.5)),
        "y_test":           y_te,
        "y_pred":           np.full(len(y_te),       0.5),
        "winner_clf":       "none",
        "winner_signature": [],
        "signature_size":   0,
        "n_events_inner":   0,
        "inner_cv_aurocs_A": {},   # match _fit_signature_model key name
        "inner_cv_auroc_B":  0.0,
        "stage_b_status":    "fallback_stage_a",
        "inner_cv_params":   {},
        "inner_importance": {},
        "calibration":      {},
        "platt_applied":    False,
        "features":         fcols_list,
        "candidate_features": fcols_list,
        "oof_shap":         None,
        "_oof":             np.full(len(y_cc_train), 0.5),
        "_cross_arm_predict": _neutral_predict,
    }


def _params_with_inner_jobs(clf_name, params, inner_jobs):
    """
    Inject n_jobs=inner_jobs only for tree-ensemble classifiers that actually
    support it (RandomForest, ExtraTrees). Returned dict is safe to pass to
    estimator.set_params(**...).

    HistGradientBoosting does not expose n_jobs (single-threaded by design);
    SVM and LogisticRegression(saga) also do not benefit from this flag.
    """
    if clf_name in ("RandomForest", "ExtraTrees") and inner_jobs > 1:
        p = dict(params)
        p["n_jobs"] = inner_jobs
        return p
    return params


def _fit_signature_model(
    X_tr_p, y_tr_mod,
    X_te_p, y_te,
    fcols,
    df_mod_train,
    inner_splits,
    cc_train_pids,
    y_cc_train,
    cc_train_raw_df,
    mod_full_df,
    feat_cols_raw,
    test_pids_set,
    ac,
    active_clfs,
    n_events_expanded,
    mod, exp_name, fold_idx,
    inner_jobs=1,
    outer_prep=None,
    outer_feat_cols_raw=None,
):
    """
    Multi-classifier signature discovery pipeline (primary analysis).

    STAGE A — Classifier comparison with fixed parameters
    -------------------------------------------------------
    For each inner fold:
      1. Build expanded inner training set and cc inner validation set.
      2. Preprocess (Tier 2 + Tier 3 + imputer + scaler) on expanded inner
         training; apply to cc inner validation.
      3. For each C_i ∈ {ElasticNet, RF, ET, HGB, SVM_Lin}:
           Fit with STAGE_A_PARAMS → AUROC on cc inner val.
           Compute feature importance (selection freq / mean |SHAP|).
           Accumulate calibration OOF predictions.

    After K inner folds per C_i:
      - mean_auroc_A_i    (AUROC on cc inner val, Stage A params)
      - importance_A_i    (normalised across K folds)
      - signature_A_i     (EPV-capped, min 5)
      - calibration_i     (Platt slope + needs_platt flag)

    STAGE B — Winner tuning
    -----------------------
    Winner = argmax_i mean_auroc_A_i  (among classifiers with valid signature).
    Manual ParameterGrid over the Stage A fold_cache (per-inner-fold
    re-preprocessed matrices), restricted to winner_signature →
    best_params_winner. In-fold and signature-true: the tuned configuration
    is exactly the deployed configuration.
    Outer refit on winner_signature features; Platt-calibrate if slope ∉ [0.80,1.20].
    OOF via make_oof_signature (expanded inner train, same model+signature).
    SHAP on signature features.

    NO LEAKAGE:
    - Outer test set (X_te_p, y_te) is only used for final prediction.
    - All model selection, feature ranking, and tuning use inner splits.
    - Calibration assessed from inner-loop OOF predictions only.
    - Preprocessing in EVERY inner-fold evaluation — Stage A Pass 1/2,
      Stage B tuning, and make_oof_signature — is re-fitted on inner
      training patients only (Stage A/B share the per-inner-fold
      fold_cache).
    - Stage B tunes the winner on its SIGNATURE subset, so the tuned
      configuration is the configuration deployed. (An earlier version
      tuned via GridSearchCV on the full outer-preprocessed matrix, which
      scored inner-validation rows through outer-fitted preprocessing —
      internally optimistic though never outer-test-leaking; replaced
      2026-08-14.)

    Parameters
    ----------
    X_tr_p          : np.ndarray — outer-preprocessed expanded training features
    y_tr_mod        : np.array  — pCR labels for expanded training patients
    X_te_p          : np.ndarray — outer-preprocessed test features
    y_te            : np.array  — pCR labels for test patients
    fcols           : sequence  — feature names after outer preprocessing
    df_mod_train    : pd.DataFrame — expanded training patients (patient_id, pCR, feats)
    inner_splits    : list of (i_tr_rel, i_va_rel) — cc-relative indices
    cc_train_pids   : np.array  — patient IDs for cc training patients (ordered)
    y_cc_train      : np.array  — pCR labels for cc training patients (same order)
    cc_train_raw_df : pd.DataFrame — raw modality features for cc training patients
    mod_full_df     : pd.DataFrame — all modality patients (patient_id, pCR, feats)
    feat_cols_raw   : list — raw feature column names (without patient_id/pCR)
    test_pids_set   : set  — outer test patient IDs (always excluded from training)
    ac              : bool — apply Tier 3 correlation filter
    active_clfs     : list — classifiers to evaluate (SVM_RBF excluded externally)
    n_events_expanded : int — pCR=1 count in expanded outer training (for EPV)
    mod, exp_name, fold_idx : str/int — for terminal diagnostic output

    Returns
    -------
    dict with complete fold result including signature metadata and _oof key.
    """
    fcols_list = list(fcols)

    # Degenerate fold: outer preprocessing removed every feature (possible
    # when NZV + correlation filtering empty a small modality in an arm
    # fold). preprocess_fold returns an empty-but-well-formed result for
    # this case; here it becomes a neutral fold instead of a classifier
    # crash on a 0-column matrix.
    if len(fcols_list) == 0:
        return _neutral_fold_result(
            y_te, y_cc_train, fcols_list,
            reason=f"{exp_name}/{mod} fold {fold_idx}: no features survived "
                   f"outer preprocessing")

    # Audit trail for the Tier 2.5 in-fold univariate screen at the OUTER
    # level. Inner folds run their own independent screen; this records the
    # outer one so the supplementary table can report how many features the
    # screen admitted per fold and whether the min_k floor was hit.
    screen_audit = None
    if outer_prep is not None:
        st = outer_prep.get("screen_stats")
        if st:
            screen_audit = {k: v for k, v in st.items()
                            if k not in ("auroc", "qval")}

    # SVM_RBF excluded from signature: no SHAP capability
    sig_clfs = [c for c in active_clfs if c != "SVM_RBF"]

    if not sig_clfs:
        return _neutral_fold_result(y_te, y_cc_train, fcols_list,
                                    "no eligible classifiers")

    # ── STAGE A — PASS 1: Cross-classifier percentile-rank importance ─────────
    # For each inner fold: fit each classifier, compute importance,
    # convert to percentile ranks within that fold/classifier, accumulate.
    # This makes importance scale-invariant across classifier types:
    # selection frequencies (linear) and mean|SHAP| (tree) are both
    # mapped to [0,1] rank space before averaging.
    #
    # Also cache preprocessed inner fold data for Pass 2 (pruned evaluation).
    # Calibration OOF predictions accumulated here for Platt check.

    # Per-classifier accumulators.
    #   rank_acc — percentile-rank importance. Used for signature DERIVATION,
    #              because ranks are the only representation in which a linear
    #              model's |coef| and a tree model's mean |SHAP| are on a
    #              comparable scale and can be averaged across classifiers.
    #   mag_acc  — the RAW importance magnitude (normalised |coef| for linear
    #              models, mean |SHAP| for tree models). Kept separately and
    #              purely for REPORTING. Earlier versions stored only ranks
    #              while the Methods and the consensus summary described the
    #              values as mean |SHAP| importance; the two are different
    #              quantities and are now recorded and labelled separately.
    rank_acc   = {c: defaultdict(float) for c in sig_clfs}
    mag_acc    = {c: defaultdict(float) for c in sig_clfs}
    cal_acc    = {c: {"y_true": [], "y_pred": []} for c in sig_clfs}
    n_success  = {c: 0 for c in sig_clfs}

    # Cache for Pass 2 — list of (X_itr_p, y_itr, X_iva_p, y_iva, fcols_inner)
    fold_cache = []

    for i_tr_rel, i_va_rel in inner_splits:

        # ── Inner validation: cc training patients ────────────────────────────
        X_iva_raw = cc_train_raw_df.iloc[i_va_rel]
        y_iva     = y_cc_train[i_va_rel]
        val_pids  = set(int(p) for p in cc_train_pids[i_va_rel])

        # ── Inner training: expanded modality patients ────────────────────────
        excluded  = test_pids_set | val_pids
        itr_mask  = ~mod_full_df["patient_id"].isin(excluded)
        X_itr_raw = mod_full_df.loc[itr_mask, feat_cols_raw]
        y_itr     = mod_full_df.loc[itr_mask, "pCR"].values

        if (len(X_itr_raw) < 5 or len(np.unique(y_itr)) < 2
                or len(np.unique(y_iva)) < 2):
            fold_cache.append(None)
            continue

        # ── Preprocess once per inner fold ───────────────────────────────────
        try:
            X_itr_p_i, X_iva_p_i, fcols_i = preprocess_fold_3(
                X_itr_raw, X_iva_raw, ac, y_train=y_itr)
        except Exception:
            fold_cache.append(None)
            continue

        fcols_i_list = list(fcols_i)
        n_feats_i    = len(fcols_i_list)
        fold_cache.append((X_itr_p_i, y_itr, X_iva_p_i, y_iva, fcols_i_list))

        # ── Fit each classifier, compute percentile-rank importance ───────────
        for clf_name in sig_clfs:
            if clf_name not in STAGE_A_PARAMS:
                continue
            cfg = CLASSIFIERS[clf_name]
            try:
                m = cfg["build"]()
                m.set_params(**_params_with_inner_jobs(
                    clf_name, STAGE_A_PARAMS[clf_name], inner_jobs))
                m.fit(X_itr_p_i, y_itr)

                y_val_pred = m.predict_proba(X_iva_p_i)[:, 1]
                cal_acc[clf_name]["y_true"].extend(y_iva.tolist())
                cal_acc[clf_name]["y_pred"].extend(y_val_pred.tolist())
                n_success[clf_name] += 1

                # Raw importance for this fold — computed on INNER TRAINING
                # features (not inner val) to avoid signature-selection
                # optimism in Stage A Pass 2. SHAP and coef-based importance
                # remain well-defined on training data.
                imp_raw = _compute_inner_importance(
                    clf_name, m, X_itr_p_i, fcols_i_list)

                # Keep the raw magnitudes for reporting ...
                for feat, val in imp_raw.items():
                    mag_acc[clf_name][feat] += abs(float(val))

                # ... and convert to percentile ranks for signature derivation
                # (scale-invariant across classifiers).
                if imp_raw and n_feats_i > 1:
                    vals_arr = np.array(list(imp_raw.values()), dtype=float)
                    # Percentile rank: 0 = least important, 1 = most important
                    ranks    = (np.argsort(np.argsort(vals_arr)) + 1) / n_feats_i
                    for feat, rank in zip(imp_raw.keys(), ranks):
                        rank_acc[clf_name][feat] += float(rank)
                elif imp_raw:
                    for feat in imp_raw:
                        rank_acc[clf_name][feat] += 1.0

            except Exception:
                pass

    # Average percentile ranks and raw magnitudes across successful inner folds
    mean_rank = {}
    mean_mag  = {}
    for clf_name in sig_clfs:
        ns = n_success[clf_name]
        if ns > 0:
            mean_rank[clf_name] = {
                f: rank_acc[clf_name][f] / ns
                for f in rank_acc[clf_name]
            }
            mean_mag[clf_name] = {
                f: mag_acc[clf_name][f] / ns
                for f in mag_acc[clf_name]
            }
        else:
            mean_rank[clf_name] = {}
            mean_mag[clf_name]  = {}

    # ── Derive signature per classifier using new pruning rules ───────────────
    signatures = {
        clf: _derive_signature(mean_rank[clf], mod, n_events_expanded)
        for clf in sig_clfs
    }

    # ── Calibration check per classifier ──────────────────────────────────────
    calibration = {}
    for clf_name in sig_clfs:
        slope, needs_platt, diag = _check_calibration(
            cal_acc[clf_name]["y_true"],
            cal_acc[clf_name]["y_pred"],
            clf_name, mod, exp_name, fold_idx)
        calibration[clf_name] = {"slope": slope, "needs_platt": needs_platt}
        print(diag, flush=True)

    # ── STAGE A — PASS 2: Evaluate PRUNED signatures on inner val folds ───────
    # Re-use cached preprocessed fold data. For each inner fold, fit the
    # pruned signature model and score on the CC inner val set.
    # Mean pruned val AUROC → winner selection criterion.
    # This validates the signature itself on held-out inner data rather
    # than selecting by the all-feature model performance.

    pruned_auroc_acc = {c: [] for c in sig_clfs}

    for fold_data in fold_cache:
        if fold_data is None:
            continue
        X_itr_p_i, y_itr, X_iva_p_i, y_iva, fcols_i_list = fold_data

        for clf_name in sig_clfs:
            sig = signatures[clf_name]
            if not sig:
                continue
            cfg = CLASSIFIERS[clf_name]
            try:
                # Select signature columns that survived inner preprocessing
                sig_set_i = set(sig) & set(fcols_i_list)
                if not sig_set_i:
                    continue
                sig_idx_i = [fcols_i_list.index(f) for f in fcols_i_list
                             if f in sig_set_i]
                X_itr_sig = X_itr_p_i[:, sig_idx_i]
                X_iva_sig = X_iva_p_i[:, sig_idx_i]

                m = cfg["build"]()
                m.set_params(**_params_with_inner_jobs(
                    clf_name, STAGE_A_PARAMS[clf_name], inner_jobs))
                m.fit(X_itr_sig, y_itr)

                if len(np.unique(y_iva)) < 2:
                    continue
                y_pred_sig = m.predict_proba(X_iva_sig)[:, 1]
                pruned_auroc_acc[clf_name].append(
                    float(roc_auc_score(y_iva, y_pred_sig)))
            except Exception:
                pass

    # Mean pruned val AUROC per classifier (winner criterion)
    mean_pruned_auroc = {
        clf: (float(np.mean(pruned_auroc_acc[clf]))
              if pruned_auroc_acc[clf] else 0.0)
        for clf in sig_clfs
    }

    # Winner = highest mean pruned inner val AUROC with a non-empty signature
    valid_clfs = [c for c in sig_clfs
                  if mean_pruned_auroc[c] > 0 and len(signatures[c]) > 0]

    if not valid_clfs:
        return _neutral_fold_result(y_te, y_cc_train, fcols_list,
                                    "no classifier produced a valid pruned signature")

    winner_clf  = max(valid_clfs, key=lambda c: mean_pruned_auroc[c])
    winner_sig  = signatures[winner_clf]

    # Store both all-feature and pruned AUROCs for reporting
    mean_aurocs_A = mean_pruned_auroc   # now reflects pruned performance

    # ── STAGE B: winner hyperparameter tuning — in-fold and signature-true ───
    # This REPLACES the previous GridSearchCV over the outer-preprocessed
    # matrix (X_tr_p). That design had two inconsistencies:
    #   (1) inner-validation rows were scored through preprocessing —
    #       including the Tier 2.5 univariate outcome screen — fitted on ALL
    #       outer-training patients, inner-validation rows included, so
    #       inner_cv_auroc_B was optimistic and hyperparameter choice was
    #       mildly biased toward screen-favoured configurations (no
    #       outer-test leakage, but inconsistent with Stage A's protocol);
    #   (2) it tuned on the FULL feature matrix while the deployed model is
    #       refit on the winner signature — tuning a different model than
    #       the one shipped.
    # The manual grid below reuses the Stage A fold_cache, whose matrices
    # were re-preprocessed INSIDE each inner fold (identical protocol to
    # Stage A Pass 2), restricted to the winner signature: the tuned
    # configuration is exactly the configuration deployed, evaluated
    # leakage-consistently. Signature-restricted fits are small, so this is
    # also cheaper than the old full-matrix GridSearchCV.
    #
    # stage_b_status distinguishes "tuned" (stage_b_cv_auroc is from this
    # in-fold grid with best_params) from "fallback_stage_a" (tuning failed
    # on every grid point; STAGE_A_PARAMS retained, AUROC from Stage A
    # Pass 2). The two are not directly comparable across folds; always
    # inspect this flag before using inner_cv_auroc_B in aggregates.
    best_params = STAGE_A_PARAMS.get(winner_clf, {})   # fallback
    stage_b_cv_auroc = mean_aurocs_A[winner_clf]        # fallback
    stage_b_status = "fallback_stage_a"

    cfg = CLASSIFIERS[winner_clf]
    win_sig_set = set(winner_sig)
    grid_scores = []
    for params in ParameterGrid(cfg["grid"]):
        fold_aurocs = []
        for fold_data in fold_cache:
            if fold_data is None:
                continue
            X_itr_p_i, y_itr, X_iva_p_i, y_iva, fcols_i_list = fold_data
            sig_idx_i = [i for i, f in enumerate(fcols_i_list)
                         if f in win_sig_set]
            if not sig_idx_i or len(np.unique(y_iva)) < 2:
                continue
            try:
                m = cfg["build"]()
                m.set_params(**_params_with_inner_jobs(
                    winner_clf, dict(params), inner_jobs))
                m.fit(X_itr_p_i[:, sig_idx_i], y_itr)
                fold_aurocs.append(float(roc_auc_score(
                    y_iva, m.predict_proba(X_iva_p_i[:, sig_idx_i])[:, 1])))
            except Exception:
                continue
        if fold_aurocs:
            grid_scores.append((float(np.mean(fold_aurocs)), dict(params)))
    if grid_scores:
        stage_b_cv_auroc, best_params = max(grid_scores, key=lambda t: t[0])
        stage_b_status = "tuned"
    else:
        print(f"  [WARN] Stage B in-fold tuning produced no valid score for "
              f"{winner_clf}; falling back to Stage A params.")

    # ── Outer refit on winner signature features (RAW, no calibration wrap) ──
    sig_set   = set(winner_sig)
    sig_idx   = [i for i, f in enumerate(fcols_list) if f in sig_set]

    if len(sig_idx) == 0:
        return _neutral_fold_result(y_te, y_cc_train, fcols_list,
                                    "winner signature empty after outer preprocessing")

    X_tr_sig   = X_tr_p[:, sig_idx]
    X_te_sig   = X_te_p[:, sig_idx]
    sig_feats  = [fcols_list[i] for i in sig_idx]

    # Build the outer model WITHOUT a CalibratedClassifierCV wrap.
    # Calibration is applied globally below using a Platt sigmoid fit on
    # raw inner-OOF predictions — see _fit_global_platt for rationale.
    outer_m = CLASSIFIERS[winner_clf]["build"]()
    outer_m.set_params(**_params_with_inner_jobs(
        winner_clf, best_params, inner_jobs))
    outer_m.fit(X_tr_sig, y_tr_mod)
    y_pred_raw = outer_m.predict_proba(X_te_sig)[:, 1]

    # ── Raw OOF scores (uncalibrated) ─────────────────────────────────────────
    # Pass fold_cache so preprocessing (NZV + correlation + impute + scale)
    # is reused from Stage A Pass 1 rather than recomputed per inner fold.
    # For RNA/DNA this is typically the largest single speedup in the pipeline.
    oof_raw = make_oof_signature(
        clf_name       = winner_clf,
        best_params    = best_params,
        signature_feats= winner_sig,
        cc_train_raw_df= cc_train_raw_df,
        y_cc_train     = y_cc_train,
        cc_train_pids  = cc_train_pids,
        mod_full_df    = mod_full_df,
        feat_cols_raw  = feat_cols_raw,
        test_pids_set  = test_pids_set,
        inner_splits   = inner_splits,
        ac             = ac,
        fold_cache     = fold_cache,
        inner_jobs     = inner_jobs,
    )

    # ── Global Platt calibration (ALWAYS applied, per design) ─────────────────
    # Fit a 2-parameter sigmoid on (raw_OOF, y_cc_train) and apply it to both
    # OOF (for fusion training) and outer-test predictions. This uniformly
    # calibrates all modality OOF columns so the fusion layer sees
    # homogeneous probability inputs, regardless of whether the individual
    # modality's winner classifier naturally produces compressed (RF/ET) or
    # overconfident outputs.
    platt_cal = _fit_global_platt(y_cc_train, oof_raw)
    if platt_cal is not None:
        y_pred = _apply_global_platt(platt_cal, y_pred_raw)
        oof    = _apply_global_platt(platt_cal, oof_raw)
        platt_applied = True
    else:
        # Insufficient data to fit a calibrator (very small cc training fold
        # or single-class OOF). Fall back to raw scores.
        y_pred = y_pred_raw
        oof    = oof_raw
        platt_applied = False

    # ── SHAP on signature features ────────────────────────────────────────────
    # SHAP is computed on the uncalibrated model (outer_m); the Platt
    # sigmoid is monotonic, so SHAP feature rankings and signs are
    # unchanged by calibration — only the probability scale shifts.
    feat_shap = compute_shap(winner_clf, outer_m, X_tr_sig, X_te_sig, sig_feats)

    # ── Build a cross-arm predictor closure ───────────────────────────────────
    # Captures the fitted, calibrated unimodal model for this fold so it can
    # later be applied to raw features of patients NOT in the fold's original
    # training/test set (e.g. opposite-arm complete-case patients for the
    # counterfactual analysis). The closure is NOT pickled into the PKL —
    # it's consumed downstream in _process_single_fold_inner to compute
    # cross-arm predictions, which are then stored as plain floats.
    #
    # Importantly this uses the EXACT same preprocessing pipeline (NZV +
    # correlation filter + imputer + scaler) and the EXACT same Platt
    # calibrator as the in-arm predictions, so in-arm vs cross-arm
    # probabilities are on identical scales.
    sig_set_closure = set(sig_feats)

    def _cross_arm_predict(X_raw_df):
        """Return calibrated P(pCR) for rows of X_raw_df using this fold's
        unimodal model. X_raw_df must have the modality's raw feature columns
        (missing ones are filled with NaN and imputed via outer_prep)."""
        if outer_prep is None or outer_feat_cols_raw is None:
            return np.full(len(X_raw_df), np.nan)
        # Align to the raw feature columns this fold was trained on
        X_aligned = X_raw_df.reindex(columns=outer_feat_cols_raw,
                                     fill_value=np.nan)
        # Apply the fitted imputer + scaler (NZV/corr pruning already baked
        # into outer_prep["columns"])
        X_p = apply_imputer_scaler(X_aligned, outer_prep)
        # Select the winner signature columns
        prep_cols = outer_prep["columns"]
        sig_idx_local = [i for i, f in enumerate(prep_cols)
                         if f in sig_set_closure]
        if not sig_idx_local:
            return np.full(len(X_raw_df), np.nan)
        X_sig = X_p[:, sig_idx_local]
        # Raw probability, then Platt-calibrate if available
        p_raw = outer_m.predict_proba(X_sig)[:, 1]
        if platt_cal is not None:
            return _apply_global_platt(platt_cal, p_raw)
        return p_raw

    # ── Realised events-per-variable ─────────────────────────────────────────
    # The EPV<=5 cap in _derive_signature is a design constraint on the
    # signature SIZE. What reviewers need in order to judge overfitting is the
    # EPV that was actually realised in each fold, which can differ from the
    # cap because of the FLOOR=5 protection and because features can be
    # dropped by the outer preprocessing after the cap was applied. We record
    # it per fold rather than quoting the nominal cap.
    n_sig = max(len(sig_feats), 1)
    epv_realized = float(n_events_expanded) / n_sig

    # ── Assemble fold result ──────────────────────────────────────────────────
    return {
        "metrics":            compute_fold_metrics(y_te, y_pred),
        "y_test":             y_te,
        "y_pred":             y_pred,
        # Signature
        "winner_clf":         winner_clf,
        "winner_signature":   sig_feats,
        "signature_size":     len(sig_feats),
        # EPV was computed from expanded-training event count (not CC).
        # Signature applies to CC test patients — therefore signature size
        # can exceed what EPV/5 would suggest from CC events alone.
        # This is intentional: expanded training provides the events.
        "n_events_inner":     n_events_expanded,
        # Explicit per-fold sample size / event accounting (reported in the
        # supplementary EPV table).
        "n_train_expanded":       int(len(y_tr_mod)),
        "n_events_train_expanded": int(np.nansum(np.asarray(y_tr_mod, float))),
        "epv_realized":           epv_realized,
        "n_candidates_outer":     len(fcols_list),
        "univariate_screen":      screen_audit,
        # Per-classifier results
        "inner_cv_aurocs_A":  mean_aurocs_A,
        "inner_cv_auroc_B":   stage_b_cv_auroc,
        "stage_b_status":     stage_b_status,     # "tuned" | "fallback_stage_a"
        "inner_cv_params":    best_params,
        # Mean cross-classifier PERCENTILE RANK per feature. This is what
        # _derive_signature ranks on. It is NOT a SHAP magnitude.
        "inner_importance":   {c: dict(mean_rank[c]) for c in sig_clfs},
        # Mean RAW importance magnitude per feature (mean |SHAP| for tree
        # classifiers, fold-normalised |coef| for linear ones). Reported
        # separately so figures and tables can state a magnitude without
        # mislabelling the rank score as one.
        "inner_importance_magnitude": {c: dict(mean_mag[c]) for c in sig_clfs},
        "signatures_all":     signatures,
        "calibration":        calibration,        # diagnostic only
        "platt_applied":      platt_applied,      # True unless calibrator fit failed
        # SHAP
        "features":           sig_feats,
        # The full candidate pool this fold could select from (post outer
        # NZV/correlation/univariate-screen). "features" above holds only the
        # WINNER SIGNATURE — using it as the eligibility denominator in
        # stability analyses makes every selected feature look perfectly
        # stable (selected/selected = 1.0). revision_analyses.py's
        # selection_frequency reads this key for the eligible-fold
        # denominator.
        "candidate_features": fcols_list,
        "oof_shap":           feat_shap,
        # OOF (popped in run_experiment before writing to PKL)
        "_oof":               oof,
        # Cross-arm predictor closure — consumed by _process_single_fold_inner
        # before PKL serialisation (also underscore-prefixed = transient).
        "_cross_arm_predict": _cross_arm_predict,
    }



# ==============================================================================
# SECTION 4b — LEGACY MODEL LOGIC (SHAP, INNER CV, OOF, FUSION)
# ==============================================================================
# Used by supplementary modes (best_per_fold, ensemble_weighted).


def preprocess_fold_3(X_tr_df, X_te_df, apply_corr=True, y_train=None):
    """Thin wrapper: returns (X_train, X_test, feature_names) from 6-value preprocess_fold.

    y_train is forwarded to the correlation filter so it can use y-aware
    (univariate AUROC) keeper selection. Without y_train, the filter falls
    back to variance-based keeper selection.
    """
    X_tr_p, X_te_p, _, _, _, fcols = preprocess_fold(
        X_tr_df, X_te_df, apply_corr_filter=apply_corr, y_train=y_train)
    return X_tr_p, X_te_p, list(fcols)


def preprocess_fold_3_with_prep(X_tr_df, X_te_df, apply_corr=True, y_train=None):
    """
    Same as preprocess_fold_3 but also returns the fitted preprocessor dict
    (imputer + scaler + final columns after NZV/correlation filtering).

    Used when the outer-fold model needs to predict on NEW raw data later —
    e.g. cross-arm counterfactual prediction, where a DHP-trained model must
    transform T-DM1 patients' raw features through the same preprocessing
    that its training set saw.

    Returns
    -------
    X_train_proc : np.ndarray
    X_test_proc  : np.ndarray
    feature_names : list[str]
    preprocessor  : dict with keys 'imputer', 'scaler', 'columns'
    """
    X_tr_p, X_te_p, prep, _, _, fcols = preprocess_fold(
        X_tr_df, X_te_df, apply_corr_filter=apply_corr, y_train=y_train)
    return X_tr_p, X_te_p, list(fcols), prep

def compute_shap(clf_name, model, X_train, X_test, feature_names):
    """
    Feature-level SHAP for a fitted unimodal model on outer test patients.

    Handles CalibratedClassifierCV wrappers (applied when Platt scaling is
    needed): SHAP is computed on the first base estimator inside the wrapper,
    using the full outer training set as the background distribution.

    Linear models (ElasticNet_LR): shap.LinearExplainer (exact).
    Tree models (RF, ET, HGB):     shap.TreeExplainer (exact, prob output).
    SVM_Linear:                    coefficient-based approximation.
    SVM_RBF:                       None (KernelSHAP too slow for production).

    Returns dict or None on failure.
    """
    stype = CLASSIFIERS.get(clf_name, {}).get("shap_type", "none")

    # Unwrap Platt calibration: use first base estimator for SHAP.
    # The base estimator captures feature importance faithfully;
    # the calibration layer only transforms the output probability.
    shap_model = model
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        shap_model = model.calibrated_classifiers_[0].estimator

    try:
        if stype == "linear":
            exp = shap.LinearExplainer(shap_model, X_train,
                                       feature_names=feature_names)
            sv  = exp.shap_values(X_test)
        elif stype == "tree":
            exp = shap.TreeExplainer(shap_model, data=X_train,
                                     feature_names=feature_names,
                                     model_output="probability")
            sv  = exp.shap_values(X_test)
            if isinstance(sv, list): sv = sv[1]
            elif sv.ndim == 3:       sv = sv[:, :, 1]
        elif stype == "linear_svm":
            coef = shap_model.coef_[0]
            sv   = (X_test - X_train.mean(axis=0)) * coef[np.newaxis, :]
        else:
            return None
        return {"feature_names": list(feature_names),
                "shap_values":   np.array(sv),
                "X_test_scaled": np.array(X_test)}
    except Exception:
        return None


def compute_fusion_shap(model, X_train, X_test, mod_order):
    """Modality-level SHAP for fusion LR model (5 inputs = 5 modalities)."""
    try:
        exp = shap.LinearExplainer(model, X_train, feature_names=mod_order)
        sv  = exp.shap_values(X_test)
        return {"feature_names": list(mod_order),
                "shap_values":   np.array(sv),
                "X_test_scaled": np.array(X_test)}
    except Exception:
        return None


def inner_cv_all(X_train, y_train, inner_splits, active_clfs, inner_jobs=1):
    """
    GridSearchCV for every active classifier. Returns {clf: result_dict}.

    LEGACY — used only by the non-primary modes (best_per_fold,
    ensemble_weighted), which are NOT the reported analysis. Note their
    known limitation: X_train is the outer-preprocessed matrix, so inner
    validation rows are scored through preprocessing fitted on all outer
    training patients (internally optimistic; no outer-test leakage). The
    primary elasticnet mode (_fit_signature_model) re-preprocesses inside
    every inner fold and does not share this limitation.
    """
    out = {}
    for name in active_clfs:
        cfg  = CLASSIFIERS[name]
        base = cfg["build"]()
        # n_jobs=inner_jobs: parameter-combination parallelism. Safe under
        # threadpool_limits(1) because sub-fits do not oversubscribe BLAS.
        gs   = GridSearchCV(base, cfg["grid"], cv=inner_splits,
                            scoring="roc_auc", refit=True, n_jobs=inner_jobs)
        try:
            gs.fit(X_train, y_train)
            out[name] = {"model":       gs.best_estimator_,
                         "params":      gs.best_params_,
                         "inner_auroc": float(gs.best_score_),
                         "cv_scores":   {str(p): float(s)
                                         for p, s in zip(
                                             gs.cv_results_["params"],
                                             gs.cv_results_["mean_test_score"])}}
        except Exception as e:
            print(f"  [WARN] {name} failed for this fold: {type(e).__name__}: {e}")
            out[name] = {"model": None, "params": {},
                         "inner_auroc": 0.0, "cv_scores": {}}
    return out


def make_oof(clf_name, params, X_raw_df, y_train, inner_splits, apply_corr,
             inner_jobs=1):
    """
    OOF scores for one (modality, classifier) using fixed hyperparams.
    Falls back to 0.5 (neutral probability) for any failed inner fold.
    """
    oof         = np.full(len(y_train), np.nan)   # nan = not yet filled
    failed      = 0
    cfg         = CLASSIFIERS[clf_name]
    for i_tr, i_va in inner_splits:
        try:
            X_itr_p, X_iva_p, _ = preprocess_fold_3(
                X_raw_df.iloc[i_tr], X_raw_df.iloc[i_va], apply_corr,
                y_train=y_train[i_tr])
            m = cfg["build"]()
            m.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
            m.fit(X_itr_p, y_train[i_tr])
            oof[i_va] = m.predict_proba(X_iva_p)[:, 1]
        except Exception:
            failed += 1
    if failed == len(inner_splits):
        print(f"  [WARN] make_oof: all {failed} inner folds failed "
              f"for {clf_name} — using neutral 0.5 OOF scores")
    return np.where(np.isnan(oof), 0.5, oof)   # fill unfitted indices with 0.5


def fit_fusion(oof_dict, y_train, inner_splits, mod_order, inner_jobs=1):
    """
    Fit a single Fused_ElasticNet meta-learner on the 5-column OOF matrix.

    ElasticNet (L1+L2, l1_ratio=0.5) is used rather than Ridge because:
    - L1 can zero out non-contributing modalities, producing an interpretable
      sparse weighting — a publishable finding in itself.
    - L2 handles the inherent collinearity between modality OOF predictions
      without the instability of pure L1.
    - With 5 inputs and ~46 pCR events (Global), EPV≈9 — well within the
      range where ElasticNet is stable.
    C is tuned by inner CV over FUSION_C_GRID.
    """
    X   = np.column_stack([oof_dict[m] for m in mod_order])
    base = LogisticRegression(
        penalty="elasticnet", solver="saga",
        l1_ratio=L1_RATIO, max_iter=2000,
        random_state=None)
    gs = GridSearchCV(base, {"C": FUSION_C_GRID}, cv=inner_splits,
                      scoring="roc_auc", refit=True, n_jobs=inner_jobs)
    gs.fit(X, y_train)
    m     = gs.best_estimator_
    coefs = m.coef_[0]
    return {
        "Fused_ElasticNet": {
            "model":               m,
            "tuned_C":             float(gs.best_params_["C"]),
            "modality_weights":    {mod: float(c)
                                    for mod, c in zip(mod_order, coefs)},
            "selected_modalities": [mod for mod, c
                                    in zip(mod_order, coefs) if abs(c) > 1e-6],
        }
    }


def make_oof_expanded(clf_name, params,
                      cc_train_raw_df,  # pd.DataFrame: raw feat cols for cc training patients
                      y_cc_train,       # np.array: pCR for cc training patients (same order)
                      cc_train_pids,    # np.array: patient IDs for cc training patients
                      mod_full_df,      # pd.DataFrame: all modality patients (patient_id, pCR, feats)
                      feat_cols,        # list: feature column names
                      test_pids_set,    # set: patient IDs in the outer test set (always excluded)
                      inner_splits,     # list of (i_tr_rel, i_va_rel) into cc_train
                      apply_corr,       # bool: apply Tier 3 correlation filter
                      inner_jobs=1):
    """
    Generate OOF probability scores for complete-case training patients,
    using the EXPANDED modality training set in each inner fold.

    For each inner fold:
      - Inner VALIDATION: the subset of complete-case training patients at
        inner split indices (i_va_rel). pCR always available (complete-case).
      - Inner TRAINING: ALL modality-m patients EXCEPT test patients and the
        inner validation complete-case patients.

    This ensures:
      1. OOF scores are always generated for every complete-case training
         patient (the fusion model's training targets are complete-case, so
         it needs scores for all of them).
      2. Inner training uses the maximum available data per modality,
         matching the outer-fold training strategy.
      3. No leakage: test patients are always excluded from all inner fits.

    Falls back to neutral probability 0.5 for any inner fold that fails
    (degenerate folds: single-class inner validation, fit error, etc.).
    """
    oof    = np.full(len(y_cc_train), np.nan)
    failed = 0
    cfg    = CLASSIFIERS[clf_name]

    for i_tr_rel, i_va_rel in inner_splits:
        # ── Inner validation: complete-case patients ──────────────────────────
        X_iva_raw = cc_train_raw_df.iloc[i_va_rel]
        y_iva     = y_cc_train[i_va_rel]
        val_pids  = set(cc_train_pids[i_va_rel])

        # ── Inner training: all modality patients except test + inner val ─────
        excluded  = test_pids_set | val_pids
        itr_mask  = ~mod_full_df["patient_id"].isin(excluded)
        X_itr_raw = mod_full_df.loc[itr_mask, feat_cols]
        y_itr     = mod_full_df.loc[itr_mask, "pCR"].values

        # Skip if degenerate (only one class in inner training)
        if len(X_itr_raw) == 0 or len(np.unique(y_itr)) < 2:
            failed += 1
            continue

        try:
            X_itr_p, X_iva_p, _ = preprocess_fold_3(
                X_itr_raw, X_iva_raw, apply_corr, y_train=y_itr)
            m = cfg["build"]()
            m.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
            m.fit(X_itr_p, y_itr)
            oof[i_va_rel] = m.predict_proba(X_iva_p)[:, 1]
        except Exception:
            failed += 1

    if failed == len(inner_splits):
        print(f"  [WARN] make_oof_expanded: all {failed} inner folds failed "
              f"for {clf_name} — using neutral 0.5 OOF scores")
    return np.where(np.isnan(oof), 0.5, oof)

# ==============================================================================
# SECTION 5 — EXPERIMENT RUNNER
# ==============================================================================

ALL_MODS = ["Clin", "RNA", "DNA", "Prot", "WSI"]


# =============================================================================
# SECTION 4c — PARALLEL OUTER FOLD WORKER
# =============================================================================

def _worker_config_keys():
    """
    Module-level settings that must be re-applied inside every joblib worker.

    Worker processes do not inherit the values assigned in main(); they either
    re-import the module (getting definition-time defaults) or receive a
    cloudpickle snapshot whose freshness is not guaranteed. Every setting that
    a preprocessing function reads from module scope at call time is therefore
    passed explicitly per fold and re-applied here. Getting this wrong is
    silent — the run completes, but with default thresholds — so the list is
    kept in one place and asserted against the globals in main().
    """
    return ("NZV_FREQ_THRESHOLD", "NZV_RATIO_THRESHOLD", "CORR_THRESHOLD",
            "UNIVARIATE_SCREEN", "UNIV_SCREEN_FDR_Q", "UNIV_SCREEN_MAX_K",
            "UNIV_SCREEN_MIN_K", "UNIV_SCREEN_MIN_FEATURES",
            # --modalities rebinds this in main(); _process_single_fold reads
            # it at call time, so workers must receive the restricted list or
            # they would loop over all five modalities and crash on the ones
            # whose feature columns were never loaded.
            "ALL_MODS",
            # Needed by the CLASSIFIERS rebuild guard in
            # _apply_worker_config below.
            "RANDOM_SEED")


def _apply_worker_config(worker_cfg):
    """Re-apply module-scope settings inside a joblib worker process."""
    if not worker_cfg:
        return
    g = globals()
    for key in _worker_config_keys():
        if key in worker_cfg and worker_cfg[key] is not None:
            g[key] = worker_cfg[key]
    # CLASSIFIERS is populated in main() and read at call time throughout the
    # worker path (_fit_signature_model, inner_cv_all, compute_shap, ...).
    # When the module is __main__ cloudpickle snapshots it by value, but when
    # the pipeline is driven via `import multimodal_pcr_pipeline; main()`
    # workers re-import the module and see the empty definition-time dict —
    # every fold then dies with KeyError. Rebuild it here; build_classifiers
    # is deterministic given the seed.
    if not g.get("CLASSIFIERS"):
        g["CLASSIFIERS"] = build_classifiers(
            worker_cfg.get("RANDOM_SEED", 42))


def _process_single_fold(fi, tr_idx, te_idx, inner_splits,
                          df_cc_exp, y_cc, features, clin_key,
                          mod_datasets, mode, active_clfs,
                          use_expanded, ac_map, exp_name,
                          inner_jobs=1, nzv_freq=None,
                          cross_arm_df=None, cross_arm_label=None,
                          worker_cfg=None):
    """
    Process one complete outer fold: all 5 modalities + fusion.

    Designed to run as an independent joblib worker — all inputs are
    read-only and passed by value (cloudpickle serialisation via loky).
    Returns a dict keyed by modality/fusion name, each holding a tuple
    (fold_result_dict, oof_score_array) so the caller can assemble the
    full results structure.

    THREAD DISCIPLINE
    -----------------
    BLAS/OpenMP thread caps are enforced THREE ways:
      1. Parent-process env vars set at module import time (pre-numpy).
      2. parallel_backend(..., inner_max_num_threads=1) on the outer Parallel.
      3. threadpool_limits(1) runtime context below — catches anything the
         first two miss (e.g. threadpools lazily initialised inside sklearn
         C extensions). This is the only method that is guaranteed effective
         once numpy is already imported.

    When inner_jobs > 1 (nested-parallelism regime, more CPUs than folds),
    tree ensembles and GridSearchCV inside this worker are allowed to use
    `inner_jobs` threads — BUT threadpool_limits=1 means those threads
    don't spawn additional BLAS threads inside.

    nzv_freq: per-experiment NZV dominant-frequency threshold override
    (0.95 for global, 0.98 for arm). Mutates this worker's copy of the
    module global before any preprocessing runs. Because loky workers
    have their own process memory space, this does not bleed between
    experiments.
    """
    _apply_worker_config(worker_cfg)
    if nzv_freq is not None:
        globals()["NZV_FREQ_THRESHOLD"] = nzv_freq

    # Runtime guard: limit BLAS threads to 1 inside this worker regardless of
    # what the parent did. We still allow sklearn tree ensembles and
    # GridSearchCV to use `inner_jobs` parallel workers, because their
    # parallelism is over trees / parameter combinations, NOT over BLAS.
    with threadpool_limits(limits=1):
        return _process_single_fold_inner(
            fi, tr_idx, te_idx, inner_splits,
            df_cc_exp, y_cc, features, clin_key,
            mod_datasets, mode, active_clfs,
            use_expanded, ac_map, exp_name, inner_jobs,
            cross_arm_df, cross_arm_label)


def _process_single_fold_inner(fi, tr_idx, te_idx, inner_splits,
                                df_cc_exp, y_cc, features, clin_key,
                                mod_datasets, mode, active_clfs,
                                use_expanded, ac_map, exp_name,
                                inner_jobs,
                                cross_arm_df=None, cross_arm_label=None):
    import warnings
    warnings.filterwarnings("ignore")

    y_te          = y_cc[te_idx]
    test_pids     = set(df_cc_exp.iloc[te_idx]["patient_id"].values)
    cc_train_pids = df_cc_exp.iloc[tr_idx]["patient_id"].values
    y_cc_train    = y_cc[tr_idx]

    oof_scores = {}
    test_preds = {}
    fold_results = {}   # {mod: fold_res_dict}

    for mod in ALL_MODS:
        ac      = ac_map[mod]
        mod_key = clin_key if mod == "Clin" else mod
        cols    = [c for c in features.get(mod_key, [])
                   if c in df_cc_exp.columns]
        if not cols:
            raise RuntimeError(
                f"[{exp_name}/{mode}] No columns for modality '{mod}'.")

        X_te_df = df_cc_exp[cols].iloc[te_idx]

        if use_expanded:
            df_mod       = mod_datasets[mod_key]
            feat_cols_m  = [c for c in cols if c in df_mod.columns
                             and c not in ("patient_id", "pCR")]
            df_mod_train = df_mod[~df_mod["patient_id"].isin(test_pids)]
            X_tr_df      = df_mod_train[feat_cols_m]
            y_tr_mod     = df_mod_train["pCR"].values
            X_tr_p, X_te_p, fcols, outer_prep = preprocess_fold_3_with_prep(
                X_tr_df, X_te_df, ac, y_train=y_tr_mod)
            cc_train_raw_df = (df_cc_exp[feat_cols_m].iloc[tr_idx]
                               .reset_index(drop=True))
            n_events_exp = int(y_tr_mod.sum())

            if mode == "elasticnet":
                fold_res = _fit_signature_model(
                    X_tr_p=X_tr_p, y_tr_mod=y_tr_mod,
                    X_te_p=X_te_p, y_te=y_te, fcols=fcols,
                    df_mod_train=df_mod_train, inner_splits=inner_splits,
                    cc_train_pids=cc_train_pids, y_cc_train=y_cc_train,
                    cc_train_raw_df=cc_train_raw_df, mod_full_df=df_mod,
                    feat_cols_raw=feat_cols_m, test_pids_set=test_pids,
                    ac=ac, active_clfs=active_clfs,
                    n_events_expanded=n_events_exp,
                    mod=mod, exp_name=exp_name, fold_idx=fi,
                    inner_jobs=inner_jobs,
                    outer_prep=outer_prep,
                    outer_feat_cols_raw=feat_cols_m,
                )
            else:
                raise NotImplementedError(
                    f"Expanded training for mode={mode!r} not implemented.")

        else:
            X_tr_df = df_cc_exp[cols].iloc[tr_idx]
            y_tr    = y_cc[tr_idx]
            X_tr_p, X_te_p, fcols, outer_prep = preprocess_fold_3_with_prep(
                X_tr_df, X_te_df, ac, y_train=y_tr)

            if mode == "elasticnet":
                feat_cols_cc    = [c for c in cols
                                   if c not in ("patient_id", "pCR")]
                df_cc_mod       = df_cc_exp[["patient_id", "pCR"]
                                             + feat_cols_cc].copy()
                df_mod_train_cc = (df_cc_mod
                                   [~df_cc_mod["patient_id"].isin(test_pids)]
                                   .reset_index(drop=True))
                cc_train_raw_df_cc = (df_cc_exp[feat_cols_cc]
                                      .iloc[tr_idx].reset_index(drop=True))
                n_events_cc = int(y_tr.sum())
                fold_res = _fit_signature_model(
                    X_tr_p=X_tr_p, y_tr_mod=y_tr,
                    X_te_p=X_te_p, y_te=y_te, fcols=fcols,
                    df_mod_train=df_mod_train_cc, inner_splits=inner_splits,
                    cc_train_pids=cc_train_pids, y_cc_train=y_cc_train,
                    cc_train_raw_df=cc_train_raw_df_cc,
                    mod_full_df=df_cc_mod, feat_cols_raw=feat_cols_cc,
                    test_pids_set=test_pids, ac=ac, active_clfs=active_clfs,
                    n_events_expanded=n_events_cc,
                    mod=mod, exp_name=exp_name, fold_idx=fi,
                    inner_jobs=inner_jobs,
                    outer_prep=outer_prep,
                    outer_feat_cols_raw=feat_cols_cc,
                )
            elif mode == "best_per_fold":
                fold_res = _fit_best_per_fold(
                    X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                    X_tr_df.reset_index(drop=True), ac, active_clfs,
                    inner_jobs=inner_jobs)
            elif mode == "ensemble_weighted":
                fold_res = _fit_ensemble(
                    X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                    X_tr_df.reset_index(drop=True), ac, active_clfs,
                    inner_jobs=inner_jobs)

        fold_res["fold_idx"] = fi
        # ── Per-fold sample size / event accounting ──────────────────────────
        # Recorded on EVERY fold dict (unimodal and fusion) so the
        # supplementary table can report the realised pCR event count and EPV
        # for each outer fold rather than only the cohort-level totals.
        # test_pids is the key that makes the patient-level bootstrap
        # possible: without it, pooled out-of-fold predictions cannot be
        # collapsed back to one value per patient, and resampling the pooled
        # rows would treat the same patient's repeated predictions as
        # independent observations.
        fold_res["test_pids"]        = np.asarray(
            df_cc_exp.iloc[te_idx]["patient_id"].values, dtype=np.int64)
        fold_res["test_idx"]         = np.asarray(te_idx, dtype=np.int64)
        fold_res["n_test"]           = int(len(te_idx))
        fold_res["n_events_test"]    = int(np.nansum(np.asarray(y_te, float)))
        fold_res["n_train_cc"]       = int(len(tr_idx))
        fold_res["n_events_train_cc"] = int(
            np.nansum(np.asarray(y_cc_train, float)))
        oof_scores[mod]  = fold_res.pop("_oof")
        test_preds[mod]  = fold_res["y_pred"]
        fold_results[mod] = fold_res

    # ── Cross-arm unimodal predictions (before fusion, uses modality closures)
    # For arm experiments only (dhp → predicts on T-DM1 patients, and
    # vice-versa). Each fold's per-modality winner + calibrator is applied to
    # every patient in the opposite arm to produce a per-patient, per-fold
    # cross-arm probability. These are stacked into a 5-column matrix and
    # pushed through this fold's fusion model to produce a fused cross-arm
    # P(pCR) per patient, on the SAME scale as in-arm P(pCR) predictions.
    cross_arm_unimodal = {}   # {mod: np.array of shape (n_cross,) }
    cross_arm_pids     = None
    do_cross_arm       = (exp_name in ("dhp", "tdm1")
                          and cross_arm_df is not None
                          and len(cross_arm_df) > 0)

    if do_cross_arm:
        cross_arm_pids = np.asarray(cross_arm_df["patient_id"].values,
                                    dtype=np.int64)
        n_cross        = len(cross_arm_pids)
        for mod in ALL_MODS:
            mod_key = clin_key if mod == "Clin" else mod
            cols    = [c for c in features.get(mod_key, [])
                       if c in cross_arm_df.columns
                       and c not in ("patient_id", "pCR")]
            predictor = fold_results[mod].get("_cross_arm_predict")
            if predictor is None or not cols:
                cross_arm_unimodal[mod] = np.full(n_cross, np.nan)
                continue
            try:
                X_cross = cross_arm_df[cols]
                p_cross = predictor(X_cross)
                cross_arm_unimodal[mod] = np.asarray(p_cross, dtype=float)
            except Exception as e:
                print(f"  [WARN] Cross-arm predict failed fold={fi} mod={mod}: "
                      f"{type(e).__name__}: {e}")
                cross_arm_unimodal[mod] = np.full(n_cross, np.nan)

    # Pop the transient closures before any PKL serialisation can touch them
    for mod in ALL_MODS:
        fold_results[mod].pop("_cross_arm_predict", None)

    # ── Fusion ────────────────────────────────────────────────────────────
    fus_fit  = fit_fusion(oof_scores, y_cc_train, inner_splits, ALL_MODS,
                          inner_jobs=inner_jobs)
    X_fus_tr = np.column_stack([oof_scores[m] for m in ALL_MODS])
    X_fus_te = np.column_stack([test_preds[m] for m in ALL_MODS])
    for fkey, fd in fus_fit.items():
        y_pf     = fd["model"].predict_proba(X_fus_te)[:, 1]
        fus_shap = compute_fusion_shap(fd["model"], X_fus_tr, X_fus_te, ALL_MODS)
        n_sel_mod = max(len(fd["selected_modalities"]), 1)
        fusion_fold = {
            "fold_idx":            fi,
            "metrics":             compute_fold_metrics(y_te, y_pf),
            "y_test":              y_te,
            "y_pred":              y_pf,
            "tuned_C":             fd["tuned_C"],
            "modality_weights":    fd["modality_weights"],
            "selected_modalities": fd["selected_modalities"],
            "oof_shap":            fus_shap,
            # Same accounting as the unimodal folds. The fusion layer's EPV
            # is computed against the number of modality streams that
            # actually received a non-zero weight, which is what determines
            # its effective complexity — not the nominal 5 inputs.
            "test_pids":           np.asarray(
                df_cc_exp.iloc[te_idx]["patient_id"].values, dtype=np.int64),
            "test_idx":            np.asarray(te_idx, dtype=np.int64),
            "n_test":              int(len(te_idx)),
            "n_events_test":       int(np.nansum(np.asarray(y_te, float))),
            "n_train_cc":          int(len(tr_idx)),
            "n_events_train_cc":   int(np.nansum(np.asarray(y_cc_train, float))),
            "signature_size":      len(fd["selected_modalities"]),
            "epv_realized":        float(
                np.nansum(np.asarray(y_cc_train, float))) / n_sel_mod,
        }

        # ── Cross-arm fused prediction ────────────────────────────────────
        # Push per-modality cross-arm columns through the fitted fusion
        # model. Patients missing ANY modality column (all-NaN from a failed
        # predictor) are skipped. Stored as {patient_id: P_alt}.
        if do_cross_arm and cross_arm_unimodal:
            X_fus_cross = np.column_stack(
                [cross_arm_unimodal.get(m, np.full(n_cross, np.nan))
                 for m in ALL_MODS])
            valid_rows = ~np.any(np.isnan(X_fus_cross), axis=1)
            cross_preds = {}
            if valid_rows.any():
                try:
                    p_alt = fd["model"].predict_proba(
                        X_fus_cross[valid_rows])[:, 1]
                    valid_pids = cross_arm_pids[valid_rows]
                    cross_preds = {int(pid): float(p)
                                   for pid, p in zip(valid_pids, p_alt)}
                except Exception as e:
                    print(f"  [WARN] Cross-arm fusion predict failed "
                          f"fold={fi}: {type(e).__name__}: {e}")
            fusion_fold["cross_arm_preds"] = cross_preds
            fusion_fold["cross_arm_label"] = cross_arm_label

        fold_results[fkey] = fusion_fold

    return fold_results


def run_experiment(df_cc_exp, features, clin_key, splits, exp_name,
                   output_dir, mode, active_clfs,
                   mod_datasets=None,
                   cross_arm_df=None, cross_arm_label=None):
    """
    Run one experiment (global/dhp/tdm1) in the specified mode.

    PARALLELISATION STRATEGY:
    Outer folds are embarrassingly parallel — each fold is fully independent
    (same read-only data, independent random draws). joblib.Parallel dispatches
    N_JOBS workers, each running _process_single_fold for one outer fold.
    N_JOBS is set via --n_jobs (default: all available CPUs).

    All n_jobs=1 inside each worker (GridSearchCV, tree classifiers) to prevent
    CPU oversubscription. OMP/BLAS thread counts are set to 1 per worker.
    The loky backend (joblib default) uses process-based workers, avoiding
    Python GIL and BLAS thread-pool issues.

    EXPANDED TRAINING STRATEGY (mod_datasets provided, --training_data expanded):
    Each unimodal model trains on ALL patients who have data for that modality,
    minus the current outer test patients.

    COMPLETE-CASE-ONLY STRATEGY (mod_datasets=None, --training_data cc_only):
    All modalities train exclusively on the complete-case patients.

    Saves {exp_name}_{mode}_results.pkl and returns the results dict.
    """
    from joblib import Parallel, delayed

    y_cc       = df_cc_exp["pCR"].values
    n_folds    = len(splits)
    ac_map     = {m: (m in CORR_FILTER_MODS) for m in ALL_MODS}
    use_expanded = (mod_datasets is not None) and (mode == "elasticnet")

    print(f"\n[{exp_name.upper()} | {mode}] {n_folds} outer folds"
          + (" | expanded training" if use_expanded else "")
          + f" | n_jobs={N_JOBS}")

    if use_expanded and splits:
        _, te_idx_0, _ = splits[0]
        test_pids_0 = set(df_cc_exp.iloc[te_idx_0]["patient_id"].values)
        print(f"  Per-modality training sizes (approx., fold 1):")
        for mod in ALL_MODS:
            mod_key = clin_key if mod == "Clin" else mod
            df_mod  = mod_datasets[mod_key]
            n_train = (~df_mod["patient_id"].isin(test_pids_0)).sum()
            n_cc    = len(df_cc_exp) - len(te_idx_0)
            print(f"    {mod:<4}: {n_train:3d} patients  (was {n_cc} cc-only, +{n_train-n_cc})")

    # ── Parallel outer fold execution ──────────────────────────────────────
    # Each worker returns {mod: fold_res_dict} for one complete fold.
    #
    # CPU budget allocation:
    #   When n_jobs >= n_folds the outer loop CANNOT saturate all CPUs on its
    #   own, so we split the budget: n_outer_workers processes, each allowed
    #   inner_jobs threads for tree ensembles / GridSearchCV. This is the
    #   "CPU-rich" regime (e.g. 16 CPUs, 5 folds).
    #   When n_jobs < n_folds we use outer-only parallelism with inner=1.
    #
    # parallel_backend(..., inner_max_num_threads=1) is the correct way to
    # prevent BLAS oversubscription. The old os.environ[...] assignment
    # inside each worker ran too late (BLAS pools are initialised at
    # numpy import time).
    #
    # batch_size=1 prevents a worker from hoarding a batch of slow folds
    # (e.g. HGB-winner folds) while other workers sit idle — i.e. it
    # short-circuits the long tail of the fold distribution.
    from joblib import parallel_backend
    n_outer_workers, inner_jobs = _resolve_parallel_budget(n_folds, N_JOBS)
    # Per-experiment NZV threshold: arm cohorts (n≈50-60) need a looser
    # threshold (0.98) so low-prevalence binary mutation features are not
    # culled just for appearing in <5% of an already-small training fold.
    nzv_freq = NZV_FREQ_GLOBAL if exp_name == "global" else NZV_FREQ_ARM
    worker_cfg = {k: globals().get(k) for k in _worker_config_keys()}
    worker_cfg["NZV_FREQ_THRESHOLD"] = nzv_freq
    print(f"  CPU budget      : {n_outer_workers} outer worker(s) × "
          f"{inner_jobs} inner job(s) = {n_outer_workers * inner_jobs} threads")
    print(f"  NZV freq thresh : {nzv_freq} ({'global' if exp_name == 'global' else 'arm'})")
    print(f"  Univariate screen: "
          + (f"IN-FOLD (BH q<={UNIV_SCREEN_FDR_Q}, max_k={UNIV_SCREEN_MAX_K}, "
             f"min_k={UNIV_SCREEN_MIN_K}) — leakage-free"
             if UNIVARIATE_SCREEN else
             "DISABLED (legacy: candidate pool was outcome-informed upstream)"))
    if cross_arm_df is not None and exp_name in ("dhp", "tdm1"):
        print(f"  Cross-arm       : {cross_arm_label or 'opposite arm'} "
              f"(n={len(cross_arm_df)}) — per-fold calibrated predictions "
              f"will be saved to Fused_ElasticNet['cross_arm_preds']")

    with parallel_backend("loky", n_jobs=n_outer_workers,
                          inner_max_num_threads=1):
        fold_result_list = Parallel(
            n_jobs=n_outer_workers, verbose=5,
            batch_size=1, pre_dispatch="2*n_jobs",
        )(
            delayed(_process_single_fold)(
                fi, tr_idx, te_idx, inner_splits,
                df_cc_exp, y_cc, features, clin_key,
                mod_datasets, mode, active_clfs,
                use_expanded, ac_map, exp_name,
                inner_jobs, nzv_freq,
                cross_arm_df, cross_arm_label,
                worker_cfg,
            )
            for fi, (tr_idx, te_idx, inner_splits) in enumerate(splits)
        )

    # ── Assemble results dict (preserves fold_idx order) ─────────────────
    results = {m: [] for m in ALL_MODS + ["Fused_ElasticNet"]}
    for fold_results in fold_result_list:
        for key, res in fold_results.items():
            results[key].append(res)

    # ── Pooled metrics: concat y_test / y_pred across folds, pick one ─────
    # Youden threshold on pooled predictions. This gives an honest
    # operating-point (Sens/Spec) pair; the per-fold Sens/Spec are
    # upper-envelope optimistic because each fold picks its own
    # Youden-best threshold on ~30 test patients.
    results["_pooled_metrics"] = {
        key: compute_pooled_metrics(results[key])
        for key in ALL_MODS + ["Fused_ElasticNet"]
        if results.get(key)
    }

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = output_dir / f"{exp_name}_{mode}_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"  [SAVE] {out_path.name}")

    _print_summary(mode, results)
    return results


# ── Per-mode fitting helpers ─────────────────────────────────────────────────

def _fit_elasticnet(X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                    # Expanded OOF parameters (primary pipeline):
                    cc_train_raw_df=None, y_cc_train=None, cc_train_pids=None,
                    mod_full_df=None, feat_cols=None, test_pids_set=None, ac=None,
                    # Legacy fallback (not used in primary pipeline):
                    X_raw_df=None,
                    inner_jobs=1):
    """
    Elastic-net LR with C tuned by inner CV and expanded-training OOF generation.

    DEAD CODE — no caller anywhere in the pipeline (the elasticnet mode goes
    through _fit_signature_model). Retained for reference only. If ever
    revived, note it shares inner_cv_all's limitation: inner validation is
    scored through outer-fitted preprocessing.

    The model is fitted on X_tr_p / y_tr (all modality patients minus test).
    OOF scores are generated for the complete-case training patients using
    make_oof_expanded, which augments inner training with all modality patients.
    This makes OOF scores consistent with the outer-fold training strategy.
    """
    base = LogisticRegression(penalty="elasticnet", solver="saga",
                              l1_ratio=L1_RATIO, max_iter=2000,
                              random_state=None)
    gs = GridSearchCV(base, {"C": ELASTICNET_C_GRID}, cv=inner_splits,
                      scoring="roc_auc", refit=True, n_jobs=inner_jobs)
    gs.fit(X_tr_p, y_tr)
    model  = gs.best_estimator_
    best_C = float(gs.best_params_["C"])

    coefs         = model.coef_[0]
    selected_mask = np.abs(coefs) > 1e-6
    y_pred        = model.predict_proba(X_te_p)[:, 1]

    # OOF generation — expanded (primary) or legacy fallback
    if mod_full_df is not None:
        oof = make_oof_expanded(
            "ElasticNet_LR", {"C": best_C},
            cc_train_raw_df, y_cc_train, cc_train_pids,
            mod_full_df, feat_cols, test_pids_set,
            inner_splits, ac, inner_jobs=inner_jobs)
    else:
        # Legacy fallback: cc-only OOF (used if expanded data not provided)
        oof = make_oof("ElasticNet_LR", {"C": best_C}, X_raw_df,
                       y_tr, inner_splits, ac, inner_jobs=inner_jobs)

    feat_shap = compute_shap("ElasticNet_LR", model, X_tr_p, X_te_p, fcols)

    return {
        "metrics":           compute_fold_metrics(y_te, y_pred),
        "y_test":            y_te,
        "y_pred":            y_pred,
        "tuned_C":           best_C,
        "cv_C_scores":       {float(C): float(s) for C, s in
                              zip(ELASTICNET_C_GRID,
                                  gs.cv_results_["mean_test_score"])},
        "features":          list(fcols),
        "coefs":             coefs.tolist(),
        "selected":          selected_mask.tolist(),
        "selected_features": [f for f, s in zip(fcols, selected_mask) if s],
        "oof_shap":          feat_shap,
        "_oof":              oof,
    }


def _fit_best_per_fold(X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                       X_raw_df, ac, active_clfs, inner_jobs=1):
    """Inner CV selects best classifier; OOF and SHAP from winner."""
    clf_res  = inner_cv_all(X_tr_p, y_tr, inner_splits, active_clfs,
                            inner_jobs=inner_jobs)
    # Filter out classifiers where fitting failed (model is None)
    valid    = {c: r for c, r in clf_res.items() if r["model"] is not None}
    if not valid:
        # All classifiers failed — return neutral predictions
        neutral = np.full(len(y_te), 0.5)
        return {"metrics": compute_fold_metrics(y_te, neutral),
                "y_test": y_te, "y_pred": neutral,
                "selected_clf": "none", "inner_aurocs": {},
                "best_params": {}, "features": list(fcols),
                "oof_shap": None, "_oof": np.full(len(y_tr), 0.5)}
    best_clf = max(valid, key=lambda c: valid[c]["inner_auroc"])
    best_est = valid[best_clf]["model"]
    best_par = valid[best_clf]["params"]

    y_pred    = best_est.predict_proba(X_te_p)[:, 1]
    oof       = make_oof(best_clf, best_par, X_raw_df, y_tr, inner_splits, ac,
                         inner_jobs=inner_jobs)
    feat_shap = compute_shap(best_clf, best_est, X_tr_p, X_te_p, fcols)

    return {
        "metrics":      compute_fold_metrics(y_te, y_pred),
        "y_test":       y_te,
        "y_pred":       y_pred,
        "selected_clf": best_clf,
        "inner_aurocs": {c: clf_res[c]["inner_auroc"] for c in clf_res},
        "best_params":  best_par,
        "features":     list(fcols),
        "oof_shap":     feat_shap,
        "_oof":         oof,
    }


def _fit_ensemble(X_tr_p, y_tr, X_te_p, y_te, inner_splits, fcols,
                  X_raw_df, ac, active_clfs, inner_jobs=1):
    """AUROC-proportional ensemble of all classifiers."""
    clf_res = inner_cv_all(X_tr_p, y_tr, inner_splits, active_clfs,
                           inner_jobs=inner_jobs)
    valid   = {c: r for c, r in clf_res.items() if r["model"] is not None}
    if not valid:
        # All classifiers failed — return neutral predictions
        neutral = np.full(len(y_te), 0.5)
        return {"metrics": compute_fold_metrics(y_te, neutral),
                "y_test": y_te, "y_pred": neutral,
                "clf_weights": {}, "inner_aurocs": {},
                "features": list(fcols), "oof_shap": None,
                "_oof": np.full(len(y_tr), 0.5)}
    raw_au  = {c: max(r["inner_auroc"], 0.0) for c, r in valid.items()}
    total   = sum(raw_au.values())
    w       = ({c: raw_au[c] / total for c in raw_au} if total > 0
               else {c: 1.0 / len(valid) for c in valid})

    oof_ens  = np.zeros(len(y_tr))
    test_ens = np.zeros(len(y_te))
    shap_acc = None; X_acc = None

    for clf_name, wt in w.items():
        est = valid[clf_name]["model"]
        par = valid[clf_name]["params"]
        test_ens += wt * est.predict_proba(X_te_p)[:, 1]
        oof_ens  += wt * make_oof(clf_name, par, X_raw_df,
                                   y_tr, inner_splits, ac,
                                   inner_jobs=inner_jobs)
        sh = compute_shap(clf_name, est, X_tr_p, X_te_p, fcols)
        if sh is not None:
            sv = sh["shap_values"]
            shap_acc = wt * sv if shap_acc is None else shap_acc + wt * sv
            X_acc    = sh["X_test_scaled"]

    feat_shap = ({"feature_names": list(fcols),
                  "shap_values":   shap_acc,
                  "X_test_scaled": X_acc}
                 if shap_acc is not None else None)

    return {
        "metrics":      compute_fold_metrics(y_te, test_ens),
        "y_test":       y_te,
        "y_pred":       test_ens,
        "clf_weights":  w,
        "inner_aurocs": {c: clf_res[c]["inner_auroc"] for c in clf_res},
        "features":     list(fcols),
        "oof_shap":     feat_shap,
        "_oof":         oof_ens,
    }


def _print_summary(mode, results):
    print(f"\n  {'Model':<22}  {'AUROC':>7}  {'Sens':>7}  {'Spec':>7}  Notes")
    for mod in ALL_MODS + ["Fused_ElasticNet"]:
        folds = results.get(mod, [])
        if not folds: continue
        au = np.mean([f["metrics"]["AUROC"] for f in folds])
        sn_vals = [f["metrics"].get("Sensitivity", np.nan) for f in folds]
        sp_vals = [f["metrics"].get("Specificity", np.nan) for f in folds]
        sn = np.nanmean(sn_vals) if any(not np.isnan(v) for v in sn_vals) else float("nan")
        sp = np.nanmean(sp_vals) if any(not np.isnan(v) for v in sp_vals) else float("nan")
        note = ""
        if mod in ALL_MODS:
            if "winner_clf" in folds[0]:
                # Signature discovery mode (primary analysis)
                top = Counter(f["winner_clf"] for f in folds).most_common(1)[0]
                sig_sizes = [f.get("signature_size", 0) for f in folds]
                n_platt = sum(1 for f in folds if f.get("platt_applied", False))
                note = (f"winner={top[0]} ({top[1]/len(folds)*100:.0f}%)  "
                        f"~{int(np.mean(sig_sizes))} feats  "
                        f"Platt={n_platt/len(folds)*100:.0f}%")
            elif "selected_clf" in folds[0]:
                top = Counter(f["selected_clf"] for f in folds).most_common(1)[0]
                note = f"best={top[0]} ({top[1]/len(folds)*100:.0f}%)"
            elif "tuned_C" in folds[0]:
                cs   = [f["tuned_C"] for f in folds if f.get("tuned_C")]
                top  = Counter(cs).most_common(1)[0] if cs else (None, 0)
                nsel = int(np.mean([len(f.get("selected_features", [])) for f in folds]))
                note = f"C={top[0]} ({top[1]/len(folds)*100:.0f}%)  ~{nsel} feats"
            elif "clf_weights" in folds[0]:
                wts = {c: np.mean([f["clf_weights"].get(c, 0) for f in folds])
                       for c in folds[0]["clf_weights"]}
                top_clf = max(wts, key=wts.get) if wts else "?"
                note = f"top={top_clf} (w={wts.get(top_clf,0):.2f})"
        elif "selected_modalities" in folds[0]:
            sel = Counter(tuple(sorted(f.get("selected_modalities",[])))
                          for f in folds).most_common(1)[0][0]
            note = "sel=" + ",".join(sel) if sel else "all"
        sn_s = f"{sn:>7.3f}" if not np.isnan(sn) else "    ---"
        sp_s = f"{sp:>7.3f}" if not np.isnan(sp) else "    ---"
        print(f"  {mod:<22}  {au:>7.3f}  {sn_s}  {sp_s}  {note}")

    # ── Pooled operating-point table ─────────────────────────────────────────
    # Sens/Spec here are computed at a SINGLE Youden threshold on the
    # concatenated-across-folds (y_test, y_pred). They are the honest
    # deployment numbers; the per-fold table above is the upper envelope.
    pooled = results.get("_pooled_metrics", {})
    if pooled:
        print(f"\n  {'Model':<22}  {'AUROC':>7}  {'Sens':>7}  {'Spec':>7}  "
              f"{'Thresh':>7}  (pooled across folds)")
        for mod in ALL_MODS + ["Fused_ElasticNet"]:
            p = pooled.get(mod)
            if not p or np.isnan(p.get("AUROC", np.nan)):
                continue
            print(f"  {mod:<22}  {p['AUROC']:>7.3f}  "
                  f"{p['Sensitivity']:>7.3f}  {p['Specificity']:>7.3f}  "
                  f"{p['Threshold']:>7.3f}  N={p['N_pooled']}")

# ==============================================================================
# SECTION 6 — SPLITS MANAGEMENT
# ==============================================================================

def load_or_generate_splits(splits_dir, exp_name, y,
                             n_outer, n_repeats, n_inner, seed, pids=None):
    """
    Load primary-pipeline splits PKL or generate fresh. Returns 3-tuple list.

    The PKL is keyed only by experiment name, but the COHORT the indices
    point into varies with --modalities / --include_features / the data file
    (e.g. dhp is n=59 in the 5-modality complete case but n≈95 RNA-only).
    Loading splits generated for a different cohort is silent when the stale
    cohort was smaller — patients simply never appear in any fold — so the
    file now carries a metadata block that is validated on load. Stale or
    foreign files fail loudly instead of corrupting a multi-hour run.
    """
    import hashlib as _hl
    pkl = Path(splits_dir) / f"{exp_name}_cv_splits.pkl" if splits_dir else None
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    meta = {"n": int(len(y)),
            "y_sha1": _hl.sha1(np.asarray(y, np.int8).tobytes()).hexdigest(),
            "n_outer": int(n_outer), "n_repeats": int(n_repeats),
            "n_inner": int(n_inner), "seed": int(seed)}
    if pids is not None:
        # Patient identity, not just label sequence: two cohorts of equal
        # size with an identical 0/1 label vector (possible after a
        # different --include_features restriction shifts the complete-case
        # set) must NOT cross-validate. patient_id values are original-file
        # row positions, so they identify the cohort membership exactly.
        meta["pid_sha1"] = _hl.sha1(
            np.asarray(pids, np.int64).tobytes()).hexdigest()

    if pkl and pkl.exists():
        with open(pkl, "rb") as f:
            raw = pickle.load(f)
        if isinstance(raw, dict) and raw.get("meta") is not None:
            if raw["meta"] != meta:
                raise SystemExit(
                    f"[SPLITS] {pkl.name} is incompatible with this run:\n"
                    f"  stored : {raw['meta']}\n"
                    f"  current: {meta}\n"
                    f"  The cohort or CV design differs (different "
                    f"--modalities / --include_features / data file / "
                    f"repeats?). Point --splits_dir elsewhere or delete the "
                    f"stale file.")
        else:
            print(f"  [SPLITS] WARNING: {pkl.name} predates cohort-metadata "
                  f"validation — index bounds are checked, cohort identity "
                  f"is NOT.")
        if isinstance(raw, dict) and "outer" in raw:
            splits = []
            for fi, (tr, te) in enumerate(raw["outer"]):
                inn = raw["inner"].get(fi, raw["inner"].get(str(fi), []))
                if not inn: inn = list(skf.split(np.zeros(len(tr)), y[tr]))
                splits.append((tr, te, inn))
        elif isinstance(raw, list) and len(raw[0]) == 3:
            splits = raw
        else:
            splits = [(tr, te, list(skf.split(np.zeros(len(tr)), y[tr])))
                      for tr, te in raw]
        # Guards for any loaded format: indices must lie inside the cohort,
        # cover it exactly, and be sorted (the cc-only path assumes df-order
        # equals tr_idx-order, which holds only for sorted index arrays).
        max_idx = max(int(np.max(te)) for _, te, _ in splits)
        if max_idx >= len(y):
            raise SystemExit(
                f"[SPLITS] {pkl.name} indices exceed cohort size {len(y)} — "
                f"generated for a different cohort.")
        covered = set()
        for tr, te, _ in splits:
            covered.update(int(v) for v in te)
            if not (np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0)):
                raise SystemExit(
                    f"[SPLITS] {pkl.name} holds unsorted index arrays; the "
                    f"pipeline requires sorted splits (sklearn-generated "
                    f"splits are sorted).")
        if covered != set(range(len(y))):
            raise SystemExit(
                f"[SPLITS] {pkl.name} test folds cover {len(covered)} of "
                f"{len(y)} patients — stale splits from a smaller cohort.")
        print(f"  [SPLITS] Loaded {len(splits)} folds from {pkl.name} "
              f"(cohort validated, n={len(y)})")
    else:
        rskf   = RepeatedStratifiedKFold(n_splits=n_outer, n_repeats=n_repeats,
                                         random_state=seed)
        splits = [(tr, te, list(skf.split(np.zeros(len(tr)), y[tr])))
                  for tr, te in rskf.split(np.zeros(len(y)), y)]
        # Save splits for reproducibility and sharing across modes
        if splits_dir:
            out = Path(splits_dir) / f"{exp_name}_cv_splits.pkl"
            with open(out, "wb") as f:
                pickle.dump({"outer": [(tr, te) for tr, te, _ in splits],
                             "inner": {fi: inn for fi, (_, _, inn) in enumerate(splits)},
                             "meta": meta},
                            f)
            print(f"  [SPLITS] Generated {len(splits)} folds → saved to {out.name}")
        else:
            print(f"  [SPLITS] Generated {len(splits)} folds (not saved — no splits_dir)")
    return splits


# ==============================================================================
# SECTION 6b — CONSENSUS MODEL FINALIZATION (R2 protocol)
# ==============================================================================
#
# The discovery CV loop (Section 5) produces a distribution of per-fold
# models — each fold has its own winner classifier, hyperparameters, and
# signature. For the Nature Cancer paper, the scientific deliverable is a
# SINGLE consensus signature per modality and a SINGLE fusion model. The
# functions here produce those deliverables in two stages:
#
#   (1) finalize_consensus()    Aggregate per-fold objects into a fixed
#                                consensus: per-modality signature (top-K by
#                                mean SHAP importance), per-modality winner
#                                classifier (modal) with modal hyperparameters.
#
#   (2) evaluate_consensus()    Honest OOF re-evaluation of the FROZEN
#                                consensus under the SAME outer-CV splits.
#                                Signature is frozen. The classifier and the
#                                fusion elastic-net are re-fit WITHIN EACH
#                                FOLD using only that fold's training data —
#                                never using test-fold outcomes. This is the
#                                R2 protocol: consensus choices carry some
#                                selection-optimism from discovery (they were
#                                chosen with knowledge of all 110 CC outcomes)
#                                but the weights and fusion coefficients are
#                                re-estimated honestly per fold, so the OOF
#                                AUROC reported is not optimistic for those
#                                estimation steps.
#
# The PRIMARY HEADLINE AUROC for the paper is the pooled-OOF AUROC of the
# fused consensus model produced by evaluate_consensus().
# ==============================================================================

def _aggregate_signature(folds, size_strategy="median",
                          df_cc=None, feat_cols=None, clf_key=None):
    """
    Return (consensus_sig, K, mean_importance_dict).

    Consensus signature = top-K features by cluster-pooled global importance,
    with one representative selected per correlated cluster.

    PROBLEM BEING SOLVED
    ---------------------
    The per-fold Tier 3 correlation filter always keeps exactly one member
    from each correlated cluster {A, B} per fold — but different folds can
    choose different representatives because the AUROC-based keeper flips
    with the training set.  This has two consequences:

    (1) Both A and B can accumulate selection frequency and appear together
        in the raw top-K consensus list.

    (2) Even after deduplication picks A as the keeper, A's recorded
        frequency underestimates the true stability of the signal, because
        the 80 folds where B was kept (and B's SHAP was recorded instead of
        A's) contribute nothing to A's imp_sum.  The biological signal
        {A or B} was present in every fold, but A's frequency only reflects
        the subset of folds where A happened to win the per-fold AUROC
        competition.

    SOLUTION: CLUSTER-LEVEL IMPORTANCE POOLING
    -------------------------------------------
    Before ranking, build correlation clusters from the FULL complete-case
    dataset (df_cc) using the same |r| >= CORR_THRESHOLD threshold.  For
    each cluster, pool the imp_sums of ALL members:

        cluster_imp_sum = sum(imp_sum[m] for m in cluster)
        cluster_imp_cnt = sum(imp_cnt[m] for m in cluster)

    Assign the pooled signal to the representative (the member with the
    highest personal imp_sum — i.e. the one the per-fold filter chose more
    often and/or with higher SHAP):

        pooled_global_imp[rep] = cluster_imp_sum / n_winner_folds

    This correctly credits the representative with the full frequency of
    the biological signal, not just the subset of folds where it personally
    survived the per-fold filter.  Non-clustered features are unaffected.

    The returned mean_imp dict uses per-feature denominators (for reporting),
    while pooled_global_imp is used only for ranking and keeper selection.

    K = median per-fold signature size (rounded up).
    """
    # RUN 5 — clf_key selects WHOSE per-fold signature is aggregated.
    #   None  : each fold contributes its own winner's signature (run-4 rule,
    #           and still the rule when SIGNATURE_SOURCE == "all_folds").
    #   "<clf>": every fold contributes THAT classifier's Stage-A signature
    #           from `signatures_all`, whether or not it won the fold. Used by
    #           SIGNATURE_SOURCE == "winner_all_folds", which keeps the full
    #           1,000-fold sample instead of the 260-550 folds the classifier
    #           happened to win, while still applying the dedup and the K rule.
    if clf_key:
        winner_folds = [f for f in folds
                        if (f.get("signatures_all", {}) or {}).get(clf_key)]
    else:
        winner_folds = [f for f in folds
                        if f.get("winner_clf", "") not in ("", "none")
                        and f.get("winner_signature")]
    if not winner_folds:
        return [], 0, {}, {}

    n_winner_folds = len(winner_folds)

    # ── Step 1: accumulate per-feature importance sums and counts ─────────
    # NOTE ON WHAT `inner_importance` ACTUALLY CONTAINS.
    # It holds the mean cross-classifier PERCENTILE RANK of each feature, not
    # a SHAP magnitude. Ranks are what makes importances from linear models
    # (|coef|) and tree models (mean |SHAP|) comparable, so ranks are the
    # correct basis for the consensus ordering. The raw magnitudes are
    # accumulated in parallel from `inner_importance_magnitude` and reported
    # alongside, so the summary can quote a magnitude without relabelling the
    # rank score as one.
    imp_sum   = defaultdict(float)   # rank-based (drives the ranking)
    mag_sum   = defaultdict(float)   # raw magnitude (reporting only)
    imp_cnt   = defaultdict(int)
    feat_freq = Counter()

    def _sig_of(fold):
        """The per-fold signature this aggregation is keyed on.

        For clf_key, intersect the RAW Stage-A signature with the fold's
        surviving candidates, so the object matches `winner_signature`
        (which is already intersected — see :2409). Without this the raw
        list inflates K and can admit features outer preprocessing drops.
        """
        if not clf_key:
            return set(fold.get("winner_signature", []))
        raw  = set((fold.get("signatures_all", {}) or {}).get(clf_key) or [])
        cand = fold.get("candidate_features")
        return (raw & set(cand)) if cand else raw

    for fold in winner_folds:
        w   = clf_key or fold["winner_clf"]
        ii  = fold.get("inner_importance", {}).get(w, {})
        mm  = fold.get("inner_importance_magnitude", {}).get(w, {})
        sig = _sig_of(fold)
        for feat in sig:
            imp_sum[feat]   += abs(float(ii.get(feat, 0.0)))
            mag_sum[feat]   += abs(float(mm.get(feat, 0.0)))
            imp_cnt[feat]   += 1
            feat_freq[feat] += 1

    # Per-feature means over the folds where the feature was selected.
    mean_imp = {f: imp_sum[f] / imp_cnt[f] for f in imp_sum}
    mean_mag = {f: mag_sum[f] / imp_cnt[f] for f in imp_sum}

    # ── Step 2: build correlation clusters and pool importance ────────────
    # cluster_of[f] → index of cluster containing f
    # clusters      → list of sets of feature names
    # pooled_global_imp[f] → imp_sum of f's entire cluster / n_winner_folds
    # representative[cluster_idx] → the member with the highest personal
    #                                imp_sum (most consistently selected
    #                                and/or most strongly scored)
    all_feats = list(imp_sum.keys())

    # Default: each feature is its own cluster (no pooling)
    pooled_global_imp = {f: imp_sum[f] / n_winner_folds for f in all_feats}
    cluster_freq      = dict(feat_freq)   # for reporting cluster-level freq
    to_remove         = set()

    if df_cc is not None and feat_cols is not None and len(all_feats) > 1:
        present  = [f for f in all_feats if f in df_cc.columns]
        # RUN 4 FIX. This list previously required nunique() > 2, which silently
        # excluded every BINARY feature from consensus deduplication. The
        # duplicated pairs in this panel are precisely binary mutation
        # indicators (DNA_coding_mutation_TP53 and ..._TP53_oncokb are identical
        # on all 190 shared rows, likewise the GATA3 pair), so the one class of
        # feature that most needed deduplicating was the one class exempt from
        # it. In run 3 both TP53 columns reached the pooled AND the DHP consensus
        # signature as if they were two independent findings.
        #
        # A binary column has a perfectly well-defined Pearson correlation (it
        # is the phi coefficient), so there is no statistical reason to exclude
        # it. We require only that a column is not constant, since a constant
        # column has undefined correlation and would inject NaN into corr_mat.
        cont     = [f for f in present if df_cc[f].dropna().nunique() > 1]

        if len(cont) > 1:
            threshold = CORR_THRESHOLD if CORR_THRESHOLD else 0.90
            corr_mat  = df_cc[cont].corr().abs()

            # ── Connected components (transitive closure) ─────────────────
            # A greedy "star" approach (for each feature, find all features
            # correlated with IT) misses transitive chains:
            #   A-B >= threshold, B-C >= threshold, A-C < threshold
            # → star clustering creates {A,B} and {C} separately, losing the
            #   B-C competition that exists in the per-fold filter.
            # Connected components via BFS correctly groups {A, B, C} because
            # B mediates the relationship between A and C.
            adj = defaultdict(set)
            for i, fa in enumerate(cont):
                for fb in cont[i+1:]:
                    if corr_mat.loc[fa, fb] >= threshold:
                        adj[fa].add(fb)
                        adj[fb].add(fa)

            visited  = set()
            clusters = []
            for start in sorted(cont, key=lambda f: -imp_sum.get(f, 0)):
                if start in visited:
                    continue
                component = set()
                queue = [start]
                while queue:
                    node = queue.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    component.add(node)
                    queue.extend(adj[node] - visited)
                clusters.append(component)

            for cluster in clusters:
                if len(cluster) <= 1:
                    continue   # no pooling needed for singletons

                # Representative = highest personal imp_sum member
                rep    = max(cluster, key=lambda f: imp_sum.get(f, 0))
                others = cluster - {rep}

                # Pool at the FOLD level, not by summing member counts.
                # Summing feat_freq over members counts (fold, member) pairs:
                # the consensus clusters come from the FULL complete-case
                # correlation matrix, while the per-fold Tier-3 filter uses
                # training correlations, so a pair at full-data r=0.92 can
                # fall below the threshold in many folds and BOTH members
                # enter that fold's signature. Each such fold contributed 2
                # to the old pool_freq — selection_frequency could exceed 1.0
                # in the consensus summary — and both members' ranks inflated
                # pool_imp_sum, the ranking key, exactly for co-selecting
                # clusters. Count each fold once; credit the fold's best
                # member rank.
                # RUN 5 FIX. This block used to read winner_signature and
                # winner_clf unconditionally. Under SIGNATURE_SOURCE=
                # "winner_all_folds" the accumulation above keys on
                # signatures_all[clf_key] and inner_importance[clf_key], so
                # reading the FOLD WINNER here would have mixed two different
                # classifiers inside one ranking: the representative's pooled
                # score would come from whichever family won each fold while
                # every other feature's score came from the locked classifier.
                # _fold_sig keeps both quantities on the same source.
                #
                # Latent on this dataset — the block only executes when a pair
                # at |r| >= CORR_THRESHOLD survives Tier 1 in RNA or DNA, and
                # preflight reports the maxima as RNA 0.880 / DNA 0.834, so no
                # cluster fires. Fixed anyway: it is a correctness bug that
                # would surface silently on the next dataset, and the only
                # evidence it had fired would be a [CONSENSUS-POOL] log line.
                member_set = set(cluster)
                _fold_sig = _sig_of          # same source as the accumulation
                folds_hit = [f for f in winner_folds if member_set & _fold_sig(f)]
                pool_freq = len(folds_hit)
                pool_imp_sum = 0.0
                for f in folds_hit:
                    w_f  = clf_key or f["winner_clf"]
                    ii_f = f.get("inner_importance", {}).get(w_f, {})
                    sel  = member_set & _fold_sig(f)
                    pool_imp_sum += max(
                        abs(float(ii_f.get(m, 0.0))) for m in sel)

                # Assign pooled signal to representative
                pooled_global_imp[rep] = pool_imp_sum / n_winner_folds
                cluster_freq[rep]      = pool_freq   # total folds any member seen
                # mean_imp for rep: keep personal denominator for interpretability
                # but note in log that it reflects personal folds only

                # Mark others for removal from ranking pool
                to_remove |= others

                print(
                    f"  [CONSENSUS-POOL] cluster {sorted(cluster)} → "
                    f"rep={rep} "
                    f"(personal freq={feat_freq.get(rep,0)/n_winner_folds:.2f} "
                    f"pool freq={pool_freq/n_winner_folds:.2f} "
                    f"personal global_imp={imp_sum.get(rep,0)/n_winner_folds:.3f} "
                    f"pooled_global_imp={pool_imp_sum/n_winner_folds:.3f})"
                )
                for other in sorted(others):
                    print(
                        f"    dropped: {other} "
                        f"(personal freq={feat_freq.get(other,0)/n_winner_folds:.2f} "
                        f"personal global_imp={imp_sum.get(other,0)/n_winner_folds:.3f})"
                    )

    # ── Step 3: rank by pooled_global_imp, remove non-representatives ─────
    ranked_all = sorted(
        [f for f in all_feats if f not in to_remove],
        key=lambda f: (-pooled_global_imp[f], -cluster_freq.get(f, 0), f)
    )

    # ── Step 4: determine K ───────────────────────────────────────────────
    # Must be measured on the SAME per-fold signatures that were aggregated
    # above, or K describes a different object than the ranking does.
    sig_sizes = [len(_sig_of(f)) for f in winner_folds]
    if size_strategy == "median":
        K = int(np.ceil(float(np.median(sig_sizes))))
    elif size_strategy == "mean":
        K = int(np.ceil(float(np.mean(sig_sizes))))
    elif size_strategy == "mode":
        K = Counter(sig_sizes).most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown size_strategy: {size_strategy}")

    consensus_sig = ranked_all[:K]
    # Return the ranking key alongside the per-feature means so callers can
    # print the value the ordering was actually made on. Printing mean_imp
    # next to an ordering produced by pooled_global_imp made the summary look
    # unsorted, because the two are different quantities.
    detail = {
        "mean_rank_when_selected":   mean_imp,
        "mean_magnitude_when_selected": mean_mag,
        "pooled_global_importance":  {f: pooled_global_imp.get(f, 0.0)
                                      for f in ranked_all},
        "selection_frequency":       {f: cluster_freq.get(f, 0) / n_winner_folds
                                      for f in ranked_all},
        "n_winner_folds":            n_winner_folds,
    }
    return consensus_sig, K, mean_imp, detail


def _aggregate_classifier(folds):
    """
    Return (modal_clf, modal_params, support_fraction).

    Modal winner_clf across folds. Ties broken by mean inner-CV Stage B
    AUROC (or Stage A when Stage B unavailable) among the tied classifiers.
    Modal hyperparameters = the most common parameter dict among folds
    whose winner_clf == the modal classifier.
    """
    winner_folds = [f for f in folds
                    if f.get("winner_clf", "") not in ("", "none")]
    if not winner_folds:
        return "none", {}, 0.0

    clf_counts = Counter(f["winner_clf"] for f in winner_folds)
    top_count  = max(clf_counts.values())
    tied       = [c for c, n in clf_counts.items() if n == top_count]

    if len(tied) > 1:
        # Tie-break by mean inner AUROC among tied classifiers.
        #
        # Only Stage A pruned AUROCs are used. Stage B AUROCs come from the
        # in-fold tuning grid with tuned hyperparameters and are not
        # comparable with Stage A values from fixed parameters — the
        # pipeline's own stage_b_status flag exists to mark that distinction
        # — so mixing them across folds would decide the tie on which folds
        # happened to have a successful Stage B rather than on classifier
        # quality. An `or` was also treating a legitimate AUROC of exactly
        # 0.0 as missing.
        def _mean_auroc(c):
            vals = [f.get("inner_cv_aurocs_A", {}).get(c)
                    for f in winner_folds if f["winner_clf"] == c]
            vals = [float(v) for v in vals if v is not None]
            return float(np.mean(vals)) if vals else 0.0
        modal_clf = max(tied, key=_mean_auroc)
    else:
        modal_clf = tied[0]

    # Modal parameter dict among folds whose winner is modal_clf
    modal_folds = [f for f in winner_folds if f["winner_clf"] == modal_clf]
    param_strs  = [str(sorted((f.get("inner_cv_params") or {}).items()))
                   for f in modal_folds]
    if param_strs:
        top_param_str = Counter(param_strs).most_common(1)[0][0]
        modal_params  = next(
            (f["inner_cv_params"] for f, s in zip(modal_folds, param_strs)
             if s == top_param_str),
            {}
        )
    else:
        modal_params = {}

    support_fraction = top_count / len(winner_folds)
    return modal_clf, modal_params, support_fraction


def _aggregate_per_classifier_signatures(folds):
    """
    Final feature-selected signature PER CLASSIFIER FAMILY, aggregated over
    every outer fold in which that family produced a signature — not only
    the folds it won.

    The main consensus (via _aggregate_signature) reports ONE signature per
    modality, taken from the per-fold winner. This function answers the
    complementary deliverable: after the iterated inner/outer CV, what
    signature does EACH classifier family converge on? Stage A already runs
    every family in every fold and records its per-fold signature
    (`signatures_all`), its per-feature percentile ranks
    (`inner_importance`) and raw magnitudes (`inner_importance_magnitude`)
    — all computed leakage-free inside the inner folds — so this is pure
    aggregation, no refitting.

    Per family:
      K            = ceil(median per-fold signature size)
      signature    = top-K features by (selection frequency, then mean
                     percentile rank) across the family's folds
      won_folds    = folds where the family was the Stage A winner
      mean_stage_a_auroc = mean inner-CV AUROC of the family's pruned
                     signature model (descriptive, NOT a performance claim —
                     the honest performance estimate remains the outer OOF)

    Returns {clf: {...}} (empty dict if no fold recorded signatures_all).
    """
    valid = [f for f in folds if f.get("signatures_all")]
    if not valid:
        return {}
    out = {}
    clfs = sorted({c for f in valid for c in f["signatures_all"]})
    for clf in clfs:
        sizes, aurocs = [], []
        sel = Counter()
        rank_sum = defaultdict(float)
        rank_cnt = defaultdict(int)
        mag_sum  = defaultdict(float)
        n_folds = 0
        for f in valid:
            sig = f["signatures_all"].get(clf) or []
            if not sig:
                continue
            n_folds += 1
            sizes.append(len(sig))
            rr = f.get("inner_importance", {}).get(clf, {})
            mm = f.get("inner_importance_magnitude", {}).get(clf, {})
            a  = f.get("inner_cv_aurocs_A", {}).get(clf)
            if a is not None:
                aurocs.append(float(a))
            for feat in sig:
                sel[feat] += 1
                if feat in rr:
                    rank_sum[feat] += float(rr[feat])
                    rank_cnt[feat] += 1
                mag_sum[feat] += abs(float(mm.get(feat, 0.0)))
        if n_folds == 0:
            continue
        freq      = {ft: c / n_folds for ft, c in sel.items()}
        mean_rank = {ft: rank_sum[ft] / rank_cnt[ft]
                     for ft in rank_sum if rank_cnt[ft]}
        mean_mag  = {ft: mag_sum[ft] / sel[ft] for ft in mag_sum}
        K = int(np.ceil(np.median(sizes)))
        ranked = sorted(sel, key=lambda ft: (freq[ft],
                                             mean_rank.get(ft, 0.0)),
                        reverse=True)
        out[clf] = {
            "signature":           ranked[:K],
            "K":                   K,
            "n_folds":             n_folds,
            "won_folds":           sum(1 for f in valid
                                       if f.get("winner_clf") == clf),
            "selection_frequency": freq,
            "mean_rank":           mean_rank,
            "mean_magnitude":      mean_mag,
            "mean_stage_a_auroc":  (float(np.mean(aurocs))
                                    if aurocs else np.nan),
        }
    return out


def finalize_consensus(results, ALL_MODS=("Clin", "RNA", "DNA", "Prot", "WSI"),
                        df_cc=None, features=None):
    """
    Aggregate per-fold discovery results into a fixed consensus.

    df_cc     : the complete-case DataFrame (all patients, all columns) —
                used by _aggregate_signature to deduplicate correlated
                features in the consensus pool.  If None, the
                post-consensus correlation filter is skipped.
    features  : dict from define_modality_features() — used to resolve
                modality-specific column lists for the correlation check.

    Returns a dict:
      {mod: {signature: [feat], K: int, winner_clf: str, params: dict,
             support_fraction: float, mean_importance: {feat: float}},
       ...}

    This is the SCIENTIFIC DELIVERABLE — one signature per modality, one
    classifier choice per modality, one set of hyperparameters per modality.
    These are reported in the paper's Results section and in Supplementary
    Table S? as the final PREDIX HER2 multimodal signature.
    """
    # Modality → column list mapping (used for correlation check)
    mod_feat_cols = {}
    if features is not None:
        mod_feat_cols = {
            "Clin": features.get("Clin_global", []),
            "RNA":  features.get("RNA",  []),
            "DNA":  features.get("DNA",  []),
            "Prot": features.get("Prot", []),
            "WSI":  features.get("WSI",  []),
        }

    consensus = {}
    for mod in ALL_MODS:
        folds = results.get(mod, [])
        # Pass df_cc and modality columns only for high-dimensional modalities
        # where correlated clusters are expected (RNA, DNA). Governed by
        # CONSENSUS_DEDUP_MODS, NOT by CORR_FILTER_MODS: run 4 removes the
        # per-fold Tier 3 filter but retains this consensus-stage safety net.
        df_for_corr   = df_cc if mod in CONSENSUS_DEDUP_MODS else None
        cols_for_corr = mod_feat_cols.get(mod) if mod in CONSENSUS_DEDUP_MODS else None

        # RUN 5 — CLASSIFIER FIRST, THEN THE SIGNATURE THAT CLASSIFIER CHOSE.
        #
        # Up to run 4 these two were aggregated over DIFFERENT fold sets. The
        # classifier was the modal winner, but the signature pooled every fold
        # regardless of which family won it — so a fold won by SVM_Linear
        # contributed features to a signature that was then reported alongside
        # ExtraTrees. The deliverable was "the modal classifier" plus "the
        # features the fold winners collectively chose", which is not the same
        # object and takes a paragraph to explain.
        #
        # Run 5 restricts the signature aggregation to the folds the locked
        # classifier actually won, so the pair is coherent by construction and
        # the Methods sentence is one line: the modal winning classifier, and
        # the features that classifier selected.
        #
        # Restricting the FOLD SET is deliberate, rather than substituting
        # consensus[mod]["per_classifier"][clf]["signature"]. That per-family
        # aggregate applies NO correlation de-duplication and uses its own K,
        # so swapping it in wholesale would discard the run-4 safety net that
        # keeps duplicated features out of the signature. Restricting the folds
        # keeps the dedup, the K rule and the importance weighting untouched.
        #
        # Support is typically 26-55% of folds, so the signature is aggregated
        # over 130-550 folds instead of 500-1000. That is ample, but it is a
        # smaller sample and per-feature selection frequencies will shift.
        clf, prm, sup = _aggregate_classifier(folds)
        sig_folds, clf_key = folds, None
        if clf not in ("", "none"):
            if SIGNATURE_SOURCE == "winner_folds":
                # Only the folds this classifier WON. Conservative: a fold it
                # lost is weaker evidence about it. Costs sample size.
                sig_folds = [f for f in folds if f.get("winner_clf") == clf] or folds
            elif SIGNATURE_SOURCE == "winner_all_folds":
                # This classifier's OWN Stage-A signature in EVERY fold, won or
                # not. Keeps the full fold sample; the Stage-A signature is
                # computed for every family in every fold regardless of who won,
                # so it is well defined throughout.
                clf_key = clf
        sig, K, imp, detail = _aggregate_signature(
            sig_folds, df_cc=df_for_corr, feat_cols=cols_for_corr,
            clf_key=clf_key)
        consensus[mod] = {
            "signature":         sig,
            "K":                 K,
            "winner_clf":        clf,
            "params":            prm,
            "support_fraction":  sup,
            # Mean cross-classifier percentile rank over the folds where the
            # feature was selected. Retained under this key for backwards
            # compatibility with generate_report.py; see `importance_detail`
            # for the quantity the consensus ordering was actually made on and
            # for the true importance magnitudes.
            "mean_importance":   imp,
            "importance_detail": detail,
            "n_folds":           len(folds),
            # Final feature-selected signature for EVERY classifier family
            # (not only the winner) — aggregated from the leakage-free
            # per-fold Stage A signatures.
            "per_classifier":    _aggregate_per_classifier_signatures(folds),
        }
        print(f"  [CONSENSUS] {mod:<5} K={K:2d}  clf={clf:<16} "
              f"(support={sup*100:4.0f}%) "
              f"sig=[{', '.join(sig[:4])}{', ...' if len(sig) > 4 else ''}]")
        for pc_clf, pc in consensus[mod]["per_classifier"].items():
            print(f"      [PER-CLF] {pc_clf:<16} K={pc['K']:2d} "
                  f"won {pc['won_folds']}/{pc['n_folds']} folds  "
                  f"sig=[{', '.join(pc['signature'][:3])}"
                  f"{', ...' if len(pc['signature']) > 3 else ''}]")
    return consensus


def _refit_consensus_unimodal_fold(
    mod, consensus_mod, X_tr_raw_df, y_tr, X_te_raw_df, inner_splits,
    ac, inner_jobs, y_cc_train, df_cc_train=None, expanded_mode=None):
    """
    Refit a single modality's consensus classifier within ONE outer fold.

    Two training regimes:

    cc-only mode: X_tr_raw_df IS the CC training fold. df_cc_train is None
        (or equal to X_tr_raw_df). Inner CV runs directly on X_tr_raw_df,
        and inner_splits index positions into X_tr_raw_df. The resulting
        oof array has len == len(y_tr) == len(CC training fold).

    expanded mode: X_tr_raw_df is the EXPANDED pool for this modality
        (all patients with this modality available, minus CC test
        patients). df_cc_train is the CC training fold. Inner CV runs by
        validating on CC training positions (inner_splits index into
        df_cc_train) and training on the expanded pool minus the
        inner-val CC patients. The resulting oof array has len ==
        len(df_cc_train) == len(CC training fold), so it aligns with
        other modalities' oof for the fusion layer.
    """
    consensus_sig = set(consensus_mod["signature"])
    clf_name      = consensus_mod["winner_clf"]
    params        = consensus_mod["params"] or {}

    # Strip non-feature columns (patient_id, pCR) before preprocessing.
    # Keep references to the originals so we can align CC indices by pid.
    def _feats_only(df_):
        return df_[[c for c in df_.columns
                    if c not in ("patient_id", "pCR")]]

    X_tr_feats   = _feats_only(X_tr_raw_df)
    X_te_feats   = _feats_only(X_te_raw_df)
    cc_df_full   = df_cc_train if df_cc_train is not None else X_tr_raw_df
    cc_df_feats  = _feats_only(cc_df_full)

    if clf_name == "none" or not consensus_sig:
        # Degenerate — return neutral
        n_cc_tr = len(cc_df_full)
        n_te    = len(X_te_raw_df)
        return (np.full(n_cc_tr, 0.5), np.full(n_cc_tr, 0.5),
                np.full(n_te, 0.5), None, [])

    # (1) Preprocess using the FULL training source for the outer refit.
    # In expanded mode this is the expanded pool; in cc-only it's CC train.
    X_tr_p, X_te_p, fcols, _prep = preprocess_fold_3_with_prep(
        X_tr_feats, X_te_feats, ac, y_train=y_tr)

    # (2) Intersect consensus signature with surviving features.
    surviving = [f for f in fcols if f in consensus_sig]
    dropped   = consensus_sig - set(surviving)
    if dropped:
        print(f"  [CONSENSUS-EVAL] {mod}: {len(dropped)} consensus features "
              f"dropped by fold preprocessing: {sorted(dropped)}")
    if not surviving:
        n_cc_tr = len(cc_df_full)
        n_te    = len(X_te_raw_df)
        return (np.full(n_cc_tr, 0.5), np.full(n_cc_tr, 0.5),
                np.full(n_te, 0.5), None, [])

    sig_idx = [fcols.index(f) for f in surviving]
    X_tr_sig = X_tr_p[:, sig_idx]
    X_te_sig = X_te_p[:, sig_idx]

    # (3) Fit outer model on the full training source
    cfg   = CLASSIFIERS[clf_name]
    model = cfg["build"]()
    try:
        model.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
    except ValueError:
        pass
    model.fit(X_tr_sig, y_tr)
    y_pred_raw_test = model.predict_proba(X_te_sig)[:, 1]

    # (4) Inner OOF — indexed to CC training positions so the fusion layer
    # can stack all 5 modality OOFs into one matrix. In expanded mode,
    # inner training pulls from the expanded pool (minus inner-val CC
    # patients); in cc-only mode, inner training is the CC inner training.
    n_cc_tr = len(cc_df_full)
    oof_raw = np.full(n_cc_tr, 0.5)

    # Which regime are we in? The caller knows, and now says so explicitly.
    #
    # This used to be inferred as `len(X_tr_raw_df) != len(cc_df_full)`. That
    # inference is wrong whenever the expanded pool happens to have the same
    # number of rows as the complete-case training fold without being the same
    # patients in the same order: the cc-only branch would then apply
    # inner-split indices — which are positions in the CC frame — to the
    # expanded frame, scattering out-of-fold predictions onto the wrong
    # patients and fitting the Platt calibrator on misaligned labels. Nothing
    # raises; the AUROC just quietly collapses toward chance. Clin can lose CC
    # patients to missing Clin_TUMSIZE, so the row counts really can coincide.
    if expanded_mode is None:
        exp_pids = (X_tr_raw_df["patient_id"].values
                    if "patient_id" in X_tr_raw_df.columns else None)
        cc_pids  = (cc_df_full["patient_id"].values
                    if "patient_id" in cc_df_full.columns else None)
        if exp_pids is not None and cc_pids is not None:
            expanded_mode = not (len(exp_pids) == len(cc_pids)
                                 and np.array_equal(exp_pids, cc_pids))
        else:
            expanded_mode = (df_cc_train is not None
                             and len(X_tr_raw_df) != len(cc_df_full))

    if expanded_mode:
        # X_tr_raw_df is the expanded pool. Align by patient_id: drop CC
        # inner-val patients from the expanded training pool for each
        # inner fold. Both frames carry patient_id (we passed them in
        # with that column).
        exp_pids = (X_tr_raw_df["patient_id"].values
                    if "patient_id" in X_tr_raw_df.columns else None)
        cc_pids  = (cc_df_full["patient_id"].values
                    if "patient_id" in cc_df_full.columns else None)

        for i_tr, i_va in inner_splits:
            if cc_pids is not None and exp_pids is not None:
                val_pid_set = set(cc_pids[i_va])
                mask_tr   = ~np.isin(exp_pids, list(val_pid_set))
                Xi_tr_raw = X_tr_feats[mask_tr]
                y_i_tr    = y_tr[mask_tr.nonzero()[0]]
                Xi_va_raw = cc_df_feats.iloc[i_va]
            else:
                # Safety fallback if pids missing: use CC training for inner CV
                Xi_tr_raw = cc_df_feats.iloc[i_tr]
                Xi_va_raw = cc_df_feats.iloc[i_va]
                y_i_tr    = y_cc_train[i_tr]

            if len(np.unique(y_i_tr)) < 2:
                continue
            try:
                Xi_tr_p, Xi_va_p, fcols_i = preprocess_fold_3(
                    Xi_tr_raw, Xi_va_raw, ac, y_train=y_i_tr)
                surv_i = [f for f in fcols_i if f in consensus_sig]
                if not surv_i:
                    continue
                sig_idx_i = [fcols_i.index(f) for f in surv_i]
                m_i = cfg["build"]()
                try:
                    m_i.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
                except ValueError:
                    pass
                m_i.fit(Xi_tr_p[:, sig_idx_i], y_i_tr)
                oof_raw[i_va] = m_i.predict_proba(Xi_va_p[:, sig_idx_i])[:, 1]
            except Exception as e:
                print(f"  [CONSENSUS-EVAL] {mod} inner fold failed: {e}")
    else:
        # cc-only mode — inner CV directly on CC training fold
        for i_tr, i_va in inner_splits:
            Xi_tr_raw = X_tr_feats.iloc[i_tr]
            Xi_va_raw = X_tr_feats.iloc[i_va]
            y_i_tr    = y_tr[i_tr]
            if len(np.unique(y_i_tr)) < 2:
                continue
            try:
                Xi_tr_p, Xi_va_p, fcols_i = preprocess_fold_3(
                    Xi_tr_raw, Xi_va_raw, ac, y_train=y_i_tr)
                surv_i = [f for f in fcols_i if f in consensus_sig]
                if not surv_i:
                    continue
                sig_idx_i = [fcols_i.index(f) for f in surv_i]
                m_i = cfg["build"]()
                try:
                    m_i.set_params(**_params_with_inner_jobs(clf_name, params, inner_jobs))
                except ValueError:
                    pass
                m_i.fit(Xi_tr_p[:, sig_idx_i], y_i_tr)
                oof_raw[i_va] = m_i.predict_proba(Xi_va_p[:, sig_idx_i])[:, 1]
            except Exception as e:
                print(f"  [CONSENSUS-EVAL] {mod} inner fold failed: {e}")

    # (5) Global Platt on inner OOF (CC-indexed), applied to outer-test preds
    platt_cal = _fit_global_platt(y_cc_train, oof_raw)
    if platt_cal is not None:
        y_pred_test = _apply_global_platt(platt_cal, y_pred_raw_test)
        oof_cal     = _apply_global_platt(platt_cal, oof_raw)
    else:
        y_pred_test = y_pred_raw_test
        oof_cal     = oof_raw

    return oof_raw, oof_cal, y_pred_test, platt_cal, surviving


def _evaluate_consensus_single_fold(
    fi, tr_idx, te_idx, inner_splits,
    df_cc_exp, y_cc, features, clin_key,
    consensus, ac_map, ALL_MODS, inner_jobs, nzv_freq,
    mod_datasets=None, worker_cfg=None):
    """One outer fold of the consensus re-evaluation."""
    # Re-apply module-scope settings for THIS worker process before any
    # preprocessing runs, mirroring what _process_single_fold does in the
    # discovery path. Required because remove_near_zero_variance and the
    # Tier 2.5 screen read their thresholds from module scope at call time
    # rather than as arguments. We use globals() rather than
    # sys.modules[__name__] so this works correctly regardless of how the
    # module was imported (as script, importlib, or loky worker process).
    _apply_worker_config(worker_cfg)
    globals()["NZV_FREQ_THRESHOLD"] = nzv_freq

    y_te          = y_cc[te_idx]
    y_cc_train    = y_cc[tr_idx]

    oof_by_mod   = {}
    test_by_mod  = {}
    metrics_by_mod = {}
    surviving_by_mod = {}

    # Complete-case test patients (identified by patient_id). Used to
    # exclude them from the expanded training pool when mod_datasets is
    # provided — matches the discovery phase's expanded-training protocol.
    test_pids = set(df_cc_exp.iloc[te_idx]["patient_id"].values)

    for mod in ALL_MODS:
        cons_mod = consensus.get(mod, {})
        if not cons_mod.get("signature"):
            n_tr, n_te = len(tr_idx), len(te_idx)
            oof_by_mod[mod]  = np.full(n_tr, 0.5)
            test_by_mod[mod] = np.full(n_te, 0.5)
            metrics_by_mod[mod] = compute_fold_metrics(y_te, np.full(n_te, 0.5))
            surviving_by_mod[mod] = []
            continue

        fc_key    = clin_key if mod == "Clin" else mod
        feat_cols = [c for c in features.get(fc_key, [])
                     if c in df_cc_exp.columns
                     and c not in ("patient_id", "pCR")]
        # Keep patient_id alongside feature columns so the consensus refit
        # function can align CC indices during inner CV in expanded mode.
        feat_cols_with_pid = feat_cols + ["patient_id"]
        X_te_raw  = df_cc_exp.iloc[te_idx][feat_cols]

        # Choose training source: expanded pool (all modality patients
        # minus this fold's CC test patients) OR cc-only (this fold's CC
        # training patients). Must mirror the discovery training strategy.
        # `is_expanded` is passed down explicitly rather than being inferred
        # from row counts inside the refit function — see the note there.
        if mod_datasets is not None:
            mod_key = clin_key if mod == "Clin" else mod
            df_mod  = mod_datasets.get(mod_key)
            if df_mod is None:
                # Modality dataset missing → fall back to cc training
                X_tr_raw = df_cc_exp.iloc[tr_idx][feat_cols_with_pid]
                y_tr     = y_cc[tr_idx]
                is_expanded = False
            else:
                feat_cols_m = [c for c in feat_cols if c in df_mod.columns]
                df_mod_tr   = df_mod[~df_mod["patient_id"].isin(test_pids)]
                X_tr_raw    = df_mod_tr[feat_cols_m + ["patient_id"]]
                y_tr        = df_mod_tr["pCR"].values
                is_expanded = True
        else:
            X_tr_raw = df_cc_exp.iloc[tr_idx][feat_cols_with_pid]
            y_tr     = y_cc[tr_idx]
            is_expanded = False

        _oof_raw, oof_cal, y_pred_test, _platt, surviving = \
            _refit_consensus_unimodal_fold(
                mod, cons_mod, X_tr_raw, y_tr, X_te_raw, inner_splits,
                ac_map[mod], inner_jobs, y_cc_train,
                df_cc_train=df_cc_exp.iloc[tr_idx][feat_cols_with_pid],
                expanded_mode=is_expanded)

        oof_by_mod[mod]     = oof_cal
        test_by_mod[mod]    = y_pred_test
        metrics_by_mod[mod] = compute_fold_metrics(y_te, y_pred_test)
        surviving_by_mod[mod] = surviving

    # Fusion: refit elastic-net LR on stacked OOF → predict on stacked test
    X_fus_tr = np.column_stack([oof_by_mod[m] for m in ALL_MODS])
    X_fus_te = np.column_stack([test_by_mod[m] for m in ALL_MODS])
    fus_fit  = fit_fusion({m: oof_by_mod[m] for m in ALL_MODS},
                           y_cc_train, inner_splits, list(ALL_MODS),
                           inner_jobs=inner_jobs)
    fusion_dict = fus_fit.get("Fused_ElasticNet", {})
    if fusion_dict and fusion_dict.get("model") is not None:
        y_pred_fus = fusion_dict["model"].predict_proba(X_fus_te)[:, 1]
        fused_metrics = compute_fold_metrics(y_te, y_pred_fus)
        modality_weights = fusion_dict.get("modality_weights", {})
        tuned_C          = fusion_dict.get("tuned_C", None)
    else:
        y_pred_fus = np.full(len(y_te), 0.5)
        fused_metrics = compute_fold_metrics(y_te, y_pred_fus)
        modality_weights = {m: 0.0 for m in ALL_MODS}
        tuned_C = None

    # Per-modality realised EPV under the FROZEN consensus signature. The
    # denominator is the number of consensus features that actually survived
    # this fold's preprocessing, which is the model's real complexity here.
    epv_by_mod = {
        m: (float(np.nansum(np.asarray(y_cc_train, float)))
            / max(len(surviving_by_mod.get(m, [])), 1))
        for m in ALL_MODS
    }
    n_sel_mod = max(
        sum(1 for w in modality_weights.values() if abs(w) > 1e-6), 1)

    return {
        "fold_idx":          fi,
        "y_test":            y_te,
        "test_idx":          np.asarray(te_idx, dtype=np.int64),
        # patient_id for every test patient — required to collapse pooled
        # out-of-fold predictions to one value per patient before bootstrapping.
        "test_pids":         np.asarray(
            df_cc_exp.iloc[te_idx]["patient_id"].values, dtype=np.int64),
        "n_test":            int(len(te_idx)),
        "n_events_test":     int(np.nansum(np.asarray(y_te, float))),
        "n_train_cc":        int(len(tr_idx)),
        "n_events_train_cc": int(np.nansum(np.asarray(y_cc_train, float))),
        "unimodal_y_pred":   test_by_mod,
        "unimodal_metrics":  metrics_by_mod,
        "unimodal_surviving": surviving_by_mod,
        "unimodal_epv":      epv_by_mod,
        "fused_y_pred":      y_pred_fus,
        "fused_metrics":     fused_metrics,
        "fused_epv":         float(
            np.nansum(np.asarray(y_cc_train, float))) / n_sel_mod,
        "modality_weights":  modality_weights,
        "tuned_C":           tuned_C,
    }


def evaluate_consensus(df_cc_exp, features, clin_key, splits, exp_name,
                       output_dir, consensus, active_clfs_unused=None,
                       ALL_MODS=("Clin", "RNA", "DNA", "Prot", "WSI"),
                       mod_datasets=None):
    """
    Run the frozen-consensus OOF re-evaluation on the same CV splits used
    for discovery. Saves {exp_name}_consensus_eval.pkl with per-fold results
    and pooled metrics.

    mod_datasets (optional)
        Per-modality expanded training datasets (same dict shape used by the
        discovery phase). If provided, per-modality classifier REFITS happen
        on the expanded pool (all patients with that modality available,
        minus this fold's CC test patients) to mirror the discovery training
        strategy. OOF and test predictions are still evaluated on CC
        patients only (the only cohort with all 5 modalities). If None, the
        cc-only discovery strategy is mirrored: classifier refits happen on
        the fold's CC training patients.

    Returns a dict with:
      folds:           per-fold results (list)
      pooled:          {mod: {AUROC, AUPRC, Brier, Sens, Spec, Threshold,
                              N_pooled}, "Fused": {...}}
      consensus:       the consensus dict used for evaluation (for audit)
    """
    from joblib import Parallel, delayed, parallel_backend

    y_cc   = df_cc_exp["pCR"].values
    n_folds = len(splits)
    ac_map  = {m: (m in CORR_FILTER_MODS) for m in ALL_MODS}
    use_expanded = mod_datasets is not None
    print(f"\n[{exp_name.upper()} | CONSENSUS-EVAL] "
          f"Frozen-signature OOF over {n_folds} outer folds"
          + (" | expanded training" if use_expanded else " | cc-only training"))

    n_outer_workers, inner_jobs = _resolve_parallel_budget(n_folds, N_JOBS)
    print(f"  CPU budget      : {n_outer_workers} outer × {inner_jobs} inner")

    # Per-experiment NZV threshold (global uses 0.95; dhp/tdm1 use 0.98 by default)
    nzv_freq = NZV_FREQ_GLOBAL if exp_name == "global" else NZV_FREQ_ARM
    worker_cfg = {k: globals().get(k) for k in _worker_config_keys()}
    worker_cfg["NZV_FREQ_THRESHOLD"] = nzv_freq
    print(f"  NZV freq thresh : {nzv_freq}")

    with parallel_backend("loky", n_jobs=n_outer_workers,
                          inner_max_num_threads=1):
        fold_results = Parallel(
            n_jobs=n_outer_workers, verbose=5,
            batch_size=1, pre_dispatch="2*n_jobs",
        )(
            delayed(_evaluate_consensus_single_fold)(
                fi, tr_idx, te_idx, inner_splits,
                df_cc_exp, y_cc, features, clin_key,
                consensus, ac_map, tuple(ALL_MODS), inner_jobs, nzv_freq,
                mod_datasets, worker_cfg,
            )
            for fi, (tr_idx, te_idx, inner_splits) in enumerate(splits)
        )

    fold_results.sort(key=lambda f: f["fold_idx"])

    # Pooled metrics: concatenate (y_test, y_pred) across all folds and
    # compute AUROC / AUPRC / Brier / Youden Sens/Spec on the pool.
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  brier_score_loss, roc_curve)
    pooled = {}

    def _pool_metrics(y_t_all, y_p_all):
        if len(np.unique(y_t_all)) < 2:
            return {"AUROC": np.nan, "AUPRC": np.nan, "Brier": np.nan,
                    "Sensitivity": np.nan, "Specificity": np.nan,
                    "Threshold": np.nan, "N_pooled": len(y_t_all)}
        fpr, tpr, thr = roc_curve(y_t_all, y_p_all)
        yi = int(np.argmax(tpr - fpr))
        return {
            "AUROC":       float(roc_auc_score(y_t_all, y_p_all)),
            "AUPRC":       float(average_precision_score(y_t_all, y_p_all)),
            "Brier":       float(brier_score_loss(y_t_all, y_p_all)),
            "Sensitivity": float(tpr[yi]),
            "Specificity": float(1.0 - fpr[yi]),
            "Threshold":   float(thr[yi]),
            "N_pooled":    int(len(y_t_all)),
        }

    for mod in ALL_MODS:
        y_t_all = np.concatenate([f["y_test"] for f in fold_results])
        y_p_all = np.concatenate([f["unimodal_y_pred"][mod] for f in fold_results])
        pooled[mod] = _pool_metrics(y_t_all, y_p_all)
        # Fold-averaged as a secondary measurement
        fold_aurocs = [f["unimodal_metrics"][mod]["AUROC"]
                        for f in fold_results]
        pooled[mod]["mean_fold_AUROC"] = float(np.nanmean(fold_aurocs))
        pooled[mod]["std_fold_AUROC"]  = float(np.nanstd(fold_aurocs))

    # Fusion pooled
    y_t_all = np.concatenate([f["y_test"] for f in fold_results])
    y_p_all = np.concatenate([f["fused_y_pred"] for f in fold_results])
    pooled["Fused_ElasticNet"] = _pool_metrics(y_t_all, y_p_all)
    fold_aurocs_fus = [f["fused_metrics"]["AUROC"] for f in fold_results]
    pooled["Fused_ElasticNet"]["mean_fold_AUROC"] = float(np.nanmean(fold_aurocs_fus))
    pooled["Fused_ElasticNet"]["std_fold_AUROC"]  = float(np.nanstd(fold_aurocs_fus))

    out_pkl = Path(output_dir) / f"{exp_name}_consensus_eval.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump({
            "folds":     fold_results,
            "pooled":    pooled,
            "consensus": consensus,
            "exp_name":  exp_name,
        }, f)
    print(f"  [SAVE] {out_pkl.name}")

    # Print summary table
    print(f"\n  CONSENSUS OOF PERFORMANCE — {exp_name}")
    print(f"  {'Model':<20} {'Pooled AUROC':>14} {'Mean fold AUROC':>18} "
          f"{'Pooled AUPRC':>14} {'Pooled Sens':>13} {'Pooled Spec':>13}")
    for mod in list(ALL_MODS) + ["Fused_ElasticNet"]:
        p = pooled[mod]
        print(f"  {mod:<20} "
              f"{p['AUROC']:>14.4f} "
              f"{p['mean_fold_AUROC']:>14.4f} ± {p['std_fold_AUROC']:.3f} "
              f"{p['AUPRC']:>14.4f} "
              f"{p['Sensitivity']:>13.4f} "
              f"{p['Specificity']:>13.4f}")

    return {
        "folds":     fold_results,
        "pooled":    pooled,
        "consensus": consensus,
    }


def write_consensus_summary(consensus, eval_result, exp_name, output_dir):
    """Write a human-readable consensus_summary.txt for the paper."""
    path = Path(output_dir) / f"{exp_name}_consensus_summary.txt"
    lines = []
    lines.append("=" * 70)
    lines.append(f"PREDIX HER2 — CONSENSUS MODEL SUMMARY ({exp_name})")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Per-modality consensus signatures (R2 protocol)")
    lines.append("-" * 70)
    for mod, c in consensus.items():
        detail = c.get("importance_detail") or {}
        pooled = detail.get("pooled_global_importance", {})
        freq   = detail.get("selection_frequency", {})
        mag    = detail.get("mean_magnitude_when_selected", {})
        lines.append(f"\n  {mod}")
        lines.append(f"    Winner classifier:     {c['winner_clf']}")
        lines.append(f"    Classifier support:    {c['support_fraction']*100:.0f}% of folds")
        lines.append(f"    Hyperparameters:       {c['params']}")
        lines.append(f"    Signature size (K):    {c['K']}")
        lines.append("    Signature features, ordered by the ranking key")
        lines.append("    (score = cluster-pooled mean percentile rank x selection")
        lines.append("     frequency; freq = fraction of folds selecting the feature")
        lines.append("     or a correlated cluster member; |imp| = mean raw importance")
        lines.append("     magnitude over the folds where it was selected):")
        for rank, feat in enumerate(c["signature"], 1):
            lines.append(
                f"      {rank:2d}. {feat:<40} "
                f"score = {pooled.get(feat, 0.0):.4f}  "
                f"freq = {freq.get(feat, 0.0):.2f}  "
                f"|imp| = {mag.get(feat, 0.0):.4f}")

    # ── Per-classifier final signatures ──────────────────────────────────────
    # Every classifier family's converged signature after the iterated
    # inner/outer CV — the deliverable complementing the single winner-based
    # consensus above. Also written machine-readable as
    # {exp_name}_per_classifier_signatures.csv.
    lines.append("")
    lines.append("Per-classifier final signatures (all families, all folds)")
    lines.append("-" * 70)
    lines.append("  freq = fraction of that family's folds selecting the feature;")
    lines.append("  rank = mean within-fold importance percentile rank;")
    lines.append("  |imp| = mean raw importance magnitude when selected.")
    lines.append("  Stage A AUROC is an inner-CV diagnostic, NOT a performance")
    lines.append("  estimate — quote only the outer OOF numbers.")
    csv_rows = []
    for mod, c in consensus.items():
        pc_all = c.get("per_classifier") or {}
        if not pc_all:
            continue
        lines.append(f"\n  {mod}")
        for clf_name in sorted(pc_all,
                               key=lambda k: -pc_all[k]["won_folds"]):
            pc = pc_all[clf_name]
            lines.append(
                f"    {clf_name:<16} K={pc['K']:2d}  "
                f"won {pc['won_folds']}/{pc['n_folds']} folds  "
                f"inner Stage A AUROC={pc['mean_stage_a_auroc']:.3f}")
            for rank_i, feat in enumerate(pc["signature"], 1):
                lines.append(
                    f"      {rank_i:2d}. {feat:<40} "
                    f"freq = {pc['selection_frequency'].get(feat, 0.0):.2f}  "
                    f"rank = {pc['mean_rank'].get(feat, 0.0):.3f}  "
                    f"|imp| = {pc['mean_magnitude'].get(feat, 0.0):.4f}")
                csv_rows.append({
                    "modality": mod, "classifier": clf_name,
                    "n_folds": pc["n_folds"], "won_folds": pc["won_folds"],
                    "mean_stage_a_auroc": pc["mean_stage_a_auroc"],
                    "K": pc["K"], "rank_in_signature": rank_i,
                    "feature": feat,
                    "selection_frequency":
                        pc["selection_frequency"].get(feat, 0.0),
                    "mean_percentile_rank":
                        pc["mean_rank"].get(feat, 0.0),
                    "mean_importance_magnitude":
                        pc["mean_magnitude"].get(feat, 0.0),
                })
    if csv_rows:
        csv_path = Path(output_dir) / f"{exp_name}_per_classifier_signatures.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"  [SAVE] {csv_path.name}")

    lines.append("")
    lines.append("Frozen-consensus OOF performance")
    lines.append("-" * 70)
    pooled = eval_result["pooled"]
    lines.append(f"\n  {'Model':<22} {'Pooled AUROC':>14} "
                 f"{'Mean fold AUROC':>20} {'Pooled AUPRC':>14}")
    for mod in list(consensus.keys()) + ["Fused_ElasticNet"]:
        p = pooled[mod]
        lines.append(f"  {mod:<22} {p['AUROC']:>14.4f} "
                     f"{p['mean_fold_AUROC']:>14.4f} ± {p['std_fold_AUROC']:.3f} "
                     f"{p['AUPRC']:>14.4f}")

    lines.append("")
    lines.append("Notes")
    lines.append("-" * 70)
    lines.append("  - Signatures are the top-K features by cluster-pooled mean cross-classifier")
    lines.append("    PERCENTILE RANK importance, weighted by selection frequency across the")
    lines.append("    discovery folds. Ranks — not raw SHAP magnitudes — are what make a linear")
    lines.append("    model's |coef| and a tree model's mean |SHAP| comparable enough to average")
    lines.append("    across classifiers. Raw magnitudes are reported in the |imp| column above")
    lines.append("    but are not the ranking key.")
    lines.append("  - K = median per-fold signature size from the winner-classifier folds.")
    lines.append("  - Consensus choices (signature + classifier identity + hyperparameters) are FROZEN.")
    lines.append("  - Classifier weights and fusion coefficients ARE REFIT within each CV fold,")
    lines.append("    using only that fold's training data — no test-fold leakage.")
    lines.append("  - The consensus signature was chosen with knowledge of all complete-case")
    lines.append("    outcomes, so it carries selection optimism that this re-evaluation does not")
    lines.append("    remove. Only the weight-estimation and fusion steps are honest here.")
    lines.append("  - Report the pooled out-of-fold AUROC with a PATIENT-LEVEL bootstrap interval")
    lines.append("    (revision_analyses.py), not the standard deviation of per-fold AUROC.")
    lines.append("")

    # encoding='utf-8' is required: these lines contain em dashes, and the
    # Windows locale codec cannot always encode them.
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [SAVE] {path.name}")


# ==============================================================================
# SECTION 7 — CLI + MAIN
# ==============================================================================

def parse_args():
    """Single source of truth for all parameters."""
    p = argparse.ArgumentParser(
        description="PREDIX HER2 multimodal pCR prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Paths
    p.add_argument("--data_path",   type=Path, required=True,
        help="Input dataset (.txt, tab-separated).")
    p.add_argument("--results_dir", type=Path, default=Path("./results"),
        help="Output directory for PKL files.")
    p.add_argument("--splits_dir",  type=Path, default=None,
        help="Directory for CV split PKLs (read or write). Defaults to "
             "--results_dir if not set. IMPORTANT: when running both "
             "--training_data strategies for comparison, point both runs "
             "at the same --splits_dir so that outer test folds are "
             "identical across strategies. The first run generates and "
             "saves the splits; the second run loads them.")

    # Mode
    p.add_argument("--mode",
        choices=["elasticnet", "best_per_fold", "ensemble_weighted", "all"],
        default="elasticnet",
        help="elasticnet: primary analysis (elastic-net LR, tuned C). "
             "best_per_fold: best classifier per modality per fold. "
             "ensemble_weighted: AUROC-weighted ensemble. "
             "all: run all three modes.")

    # Training data strategy
    p.add_argument("--training_data",
        choices=["expanded", "cc_only"],
        default="expanded",
        help="expanded (default): each unimodal model trains on ALL patients "
             "with that modality available, minus the current outer test fold. "
             "Test sets remain complete-case for paired comparisons. "
             "This maximises training data and is the recommended strategy. "
             "cc_only: restrict training to the complete-case patients only "
             "(those with all five modalities). Produces a more conservative "
             "estimate; useful to isolate the effect of expanded training or "
             "when modality-specific patients are not available.")

    # Classifiers (used for best_per_fold and ensemble_weighted only)
    p.add_argument("--classifiers", nargs="+",
        default=["ElasticNet_LR", "RandomForest", "ExtraTrees",
                 "HistGradBoost", "SVM_RBF", "SVM_Linear"],
        help="Classifiers evaluated in Stage A (signature discovery). "
             "SVM_RBF is automatically excluded from signature ranking "
             "(no SHAP capability) but can be listed without error. "
             "Winner selected per modality per fold by inner CV AUROC.")

    # Reproducibility
    p.add_argument("--seed", type=int, default=42)

    # Parallelism
    p.add_argument("--n_jobs", type=int, default=-1,
        help="Number of parallel workers for the outer fold loop. "
             "-1 = use all available CPUs (default). "
             "1 = sequential (useful for debugging). "
             "Set to the number of CPUs allocated in your SLURM job "
             "(e.g. --cpus-per-task=32 → --n_jobs=32).")

    # Outer CV
    p.add_argument("--outer_folds_global", type=int, default=5)
    p.add_argument("--outer_folds_arm",    type=int, default=5)
    p.add_argument("--repeats_global",     type=int, default=20,
        help="Production: 200.")
    p.add_argument("--repeats_arm",        type=int, default=10,
        help="Production: 100.")

    # Inner CV
    p.add_argument("--inner_folds_global", type=int, default=5)
    p.add_argument("--inner_folds_arm",    type=int, default=3,
        help="3 ensures ≥25 inner training patients.")

    # Preprocessing
    p.add_argument("--corr_threshold", type=float, default=0.90)
    p.add_argument("--nzv_freq_global", type=float, default=0.95,
        help="NZV dominant-value-frequency threshold for GLOBAL experiment. "
             "A feature whose most common value occupies ≥ this fraction of "
             "training samples is removed.")
    p.add_argument("--nzv_freq_arm",    type=float, default=0.98,
        # NOTE: argparse %-formats help strings under
        # ArgumentDefaultsHelpFormatter, so every literal percent sign here
        # MUST be written as '%%'. An unescaped '%' raises
        # "ValueError: badly formed help string" at add_argument() time on
        # Python 3.13+, which prevented the deposited version of this script
        # from starting at all.
        help="NZV threshold for ARM experiments (DHP, TDM1). Higher than "
             "--nzv_freq_global because arm cohorts are small (n≈50-60) and "
             "the 0.95 cutoff silently culls low-prevalence binary mutation "
             "features (e.g. features present in ~5%% of patients) which are "
             "precisely the biologically meaningful DNA features the study "
             "is interested in. 0.98 keeps features present in ≥ ~2%% of "
             "arm training patients.")
    p.add_argument("--nzv_ratio",      type=float, default=20.0)

    # ── Tier 2.5: in-fold univariate outcome screen ──────────────────────────
    p.add_argument("--univariate_screen",
        choices=["in_fold", "none"], default="in_fold",
        help="in_fold (DEFAULT, primary analysis): the univariate association "
             "between each feature and pCR is evaluated INSIDE every training "
             "fold, on training patients only, so no test patient influences "
             "which features enter the model. This is the leakage-free "
             "protocol. none: skip the screen entirely, reproducing the "
             "original submission in which the univariate step had already "
             "been applied to the whole cohort before the input file was "
             "written. Run both and compare to quantify the optimism "
             "attributable to that step.")
    p.add_argument("--univ_fdr_q", type=float, default=0.25,
        help="Benjamini-Hochberg q-value ceiling for the in-fold univariate "
             "screen, applied within modality within fold. Deliberately "
             "permissive (0.25, not 0.05) because this is a screening step "
             "feeding multivariable signature discovery, not an inference "
             "step. Pre-specified and identical across all folds and arms.")
    p.add_argument("--univ_max_k", type=int, default=40,
        help="Hard cap on the number of features surviving the in-fold "
             "univariate screen per modality per fold.")
    p.add_argument("--univ_min_k", type=int, default=5,
        help="Floor on the number of features surviving the screen, so a "
             "modality can never collapse to a degenerate design matrix.")

    # ── Candidate feature pool ───────────────────────────────────────────────
    p.add_argument("--feature_pool", choices=["curated", "full"],
        default="curated",
        help="curated (default): apply the fixed TIER1_REMOVE biological "
             "deduplication list (co-amplicons and near-identical composite "
             "scores). full: disable TIER1_REMOVE and start from every "
             "measured feature present in the input file. Combine "
             "'--feature_pool full --univariate_screen in_fold' for the fully "
             "leakage-free analysis starting from the complete pre-curation "
             "feature set.")

    # ── Run 5: coherence of the locked (classifier, signature) pair ──────────
    p.add_argument("--signature_source",
        choices=["winner_folds", "winner_all_folds", "all_folds"],
        default="winner_folds",
        help="Whose per-fold signature is aggregated into the locked "
             "signature. winner_folds (DEFAULT, run 5): restrict to the outer "
             "folds the modal classifier won, so the reported classifier and "
             "the reported signature are the same model. winner_all_folds: "
             "that classifier's own Stage-A signature from every fold, won or "
             "not — a larger sample, intersected with each fold's surviving "
             "candidates so it stays comparable; sensitivity analysis only. "
             "all_folds: run-4 behaviour, where every fold contributed its own "
             "winner's signature, so a fold won by SVM_Linear fed a signature "
             "reported alongside ExtraTrees. All three keep the correlation "
             "dedup and the K rule.")

    # ── Run 5: skip discovery and re-finalise from an existing PKL ───────────
    p.add_argument("--consensus_only", action="store_true",
        help="Skip the discovery CV loop and run ONLY the consensus "
             "finalisation and frozen re-evaluation, reading the existing "
             "{exp}_elasticnet_results.pkl. Use to re-derive the consensus "
             "under a different --signature_source without repeating the "
             "expensive per-fold model search. Fails if the discovery PKL is "
             "absent. NOTE the discovery PKL must have been produced with the "
             "SAME feature pool: changing TIER1_REMOVE invalidates it.")

    # Stability thresholds (elasticnet mode reporting)
    p.add_argument("--stability_thresh_global", type=float, default=0.60)
    p.add_argument("--stability_thresh_arm",    type=float, default=0.50)

    # Experiments
    p.add_argument("--experiments", nargs="+",
        choices=["global", "dhp", "tdm1"],
        default=["global", "dhp", "tdm1"])

    # Modality restriction (external validation support)
    p.add_argument("--modalities", nargs="+",
        choices=["Clin", "RNA", "DNA", "Prot", "WSI"],
        default=["Clin", "RNA", "DNA", "Prot", "WSI"],
        help="Modalities to model. Default: all five (the primary multimodal "
             "analysis). Restricting (e.g. '--modalities RNA') runs the SAME "
             "pipeline protocol — in-fold filters, univariate screen, "
             "signature discovery, classifier tournament, consensus — on the "
             "listed modalities only, and defines the complete-case cohort "
             "over those modalities alone (e.g. RNA-only: every patient with "
             "complete RNA, n=185, instead of the 5-modality n=110). Used to "
             "build the transcriptomic-only model that is validated "
             "externally. With a single modality the fusion layer degenerates "
             "to a recalibration of that modality's predictions; the "
             "modality-specific consensus model is the deliverable.")
    p.add_argument("--include_features", type=Path, default=None,
        help="Optional text file with one feature column name per line "
             "(# comments allowed). Feature columns NOT listed are dropped "
             "before modelling; patientID, pCR and Clin_Arm are always kept. "
             "Used to restrict an RNA-only run to the features measured in an "
             "external cohort (written by external_validation.py as "
             "shared_features_<cohort>.txt), so the internal estimate and the "
             "locked external model use the identical feature universe.")

    # Consensus finalization — frozen-signature re-evaluation (R2)
    p.add_argument("--consensus", action="store_true", default=True,
        help="After discovery CV completes, aggregate per-fold signatures "
             "into a per-modality consensus, then re-evaluate that frozen "
             "consensus under the SAME CV splits (classifier + fusion re-fit "
             "within each fold, signature frozen). This is the R2 protocol "
             "from the Nature Cancer methods: signatures and classifier "
             "choices are the scientific deliverable; performance reported "
             "is the consensus OOF AUROC from this re-evaluation.")
    p.add_argument("--no-consensus", dest="consensus", action="store_false",
        help="Skip the consensus finalization phase (speeds up smoke tests).")

    return p.parse_args()


def write_provenance(args, results_dir, active_clfs, splits_dir):
    """
    Record exactly what produced the results in this directory.

    Written before any modelling starts, so it exists even if the run is
    interrupted. It captures the pipeline version, every command-line
    parameter, the random seed, the resolved package versions, and a hash of
    the input file — enough to reproduce the run or to detect that the inputs
    have changed since it was made.
    """
    import json, platform, hashlib, datetime

    def _pkg_version(name):
        try:
            import importlib
            return getattr(importlib.import_module(name), "__version__",
                           "unknown")
        except Exception:
            return "not installed"

    data_hash = "unavailable"
    try:
        h = hashlib.sha256()
        with open(args.data_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        data_hash = h.hexdigest()
    except Exception:
        pass

    prov = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(timespec="seconds"),
        "random_seed": args.seed,
        "input_data": {
            "path": str(args.data_path),
            "sha256": data_hash,
        },
        "resolved_splits_dir": str(splits_dir),
        "active_classifiers": list(active_clfs),
        "parameters": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in sorted(vars(args).items())},
        "leakage_control": {
            "univariate_screen": args.univariate_screen,
            "note": ("in_fold: the univariate association step is performed "
                     "inside each training fold, so no test patient "
                     "influences which features enter the model. "
                     "none: reproduces the original submission, in which "
                     "that step had been applied to the whole cohort."),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scikit-learn": _pkg_version("sklearn"),
            "scipy": _pkg_version("scipy"),
            "shap": _pkg_version("shap"),
            "joblib": _pkg_version("joblib"),
        },
        "reproducibility_note": (
            "Cross-validation partitions are fully determined by random_seed. "
            "Classifier internal randomness is deliberately NOT seeded "
            "(random_state=None): seeding it would make every repeat see the "
            "same bootstrap sample for a given fold, correlating the repeats "
            "and producing artificially narrow variance on fold-averaged "
            "metrics. Point estimates are stable across runs; the last digit "
            "of a per-fold metric may vary."),
    }

    out = Path(results_dir) / "run_provenance.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"[PROVENANCE] {out}")
    return prov


def write_cv_design_statement(args, results_dir):
    """
    Emit the canonical description of the cross-validation design, generated
    from the ACTUAL run parameters, as methods_cv_statement.txt.

    Why this exists: the submitted manuscript described the CV design
    differently in three places — the Results and the Extended Data Fig. 11a
    legend said "100 stratified shuffle-split iterations (80/20)", while the
    Methods said "repeated stratified nested 5-fold cross-validation (1,000
    outer evaluations)". The code has only ever implemented the latter
    (RepeatedStratifiedKFold outer, StratifiedKFold inner); the shuffle-split
    wording was a leftover from an earlier analysis version. Every manuscript
    passage that describes the CV design must be copied from this file, so the
    text can never drift from the code again. The same content is embedded in
    run_provenance.json under "cv_design".
    """
    og, rg, ig = args.outer_folds_global, args.repeats_global, args.inner_folds_global
    oa, ra, ia = args.outer_folds_arm,    args.repeats_arm,    args.inner_folds_arm
    if og < 2 or oa < 2:
        raise SystemExit("[CV-DESIGN] outer folds must be >= 2 "
                         f"(got global={og}, arm={oa}).")
    n_eval_g, n_eval_a = og * rg, oa * ra
    train_pct, test_pct = round(100 * (og - 1) / og), round(100 / og)

    methods = (
        f"Prediction performance was evaluated by repeated stratified nested "
        f"cross-validation. The outer loop was stratified {og}-fold "
        f"cross-validation repeated {rg} times with different random "
        f"partitions ({n_eval_g} outer test-fold evaluations; each outer "
        f"training set comprised ~{train_pct}% of patients and each held-out "
        f"test fold ~{test_pct}%). Within every outer training fold, "
        f"stratified {ig}-fold inner cross-validation was used for classifier "
        f"selection, hyperparameter tuning and feature-signature discovery; "
        f"all preprocessing (imputation, near-zero-variance filtering, "
        f"correlation pruning, scaling"
        + (", the univariate outcome screen" if args.univariate_screen == "in_fold" else "")
        + f") was fitted on outer training patients only. Arm-specific models "
        f"used {oa}-fold outer cross-validation repeated {ra} times "
        f"({n_eval_a} evaluations per arm) with {ia}-fold inner "
        f"cross-validation. No shuffle-split resampling was used at any "
        f"stage."
    )
    legend = (
        f"Schematic of the repeated stratified nested cross-validation "
        f"framework. The outer loop is stratified {og}-fold cross-validation "
        f"repeated {rg} times ({n_eval_g} outer evaluations; ~{train_pct}% "
        f"training / ~{test_pct}% testing per fold). Within each outer "
        f"training fold, stratified {ig}-fold inner cross-validation performs "
        f"model selection and feature ranking. The selected model is "
        f"evaluated on the corresponding held-out outer test fold, and "
        f"performance is pooled across all {n_eval_g} outer evaluations."
    )

    out = Path(results_dir) / "methods_cv_statement.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(
            "CANONICAL CROSS-VALIDATION DESCRIPTION\n"
            f"Generated by pipeline {PIPELINE_VERSION} from the actual run "
            "parameters.\n"
            "Copy the manuscript text from HERE; do not hand-write it.\n"
            "If the run parameters change, this file changes with them.\n\n"
            "--- METHODS PARAGRAPH ---\n\n" + methods + "\n\n"
            "--- FIGURE LEGEND (Extended Data Fig. 11a) ---\n\n" + legend + "\n\n"
            "--- NUMBERS FOR THE RESULTS TEXT ---\n\n"
            f"outer design            : {og}-fold x {rg} repeats "
            f"= {n_eval_g} outer evaluations (global)\n"
            f"arm design              : {oa}-fold x {ra} repeats "
            f"= {n_eval_a} outer evaluations per arm\n"
            f"inner folds             : {ig} (global), {ia} (arm)\n"
            f"train/test per fold     : ~{train_pct}% / ~{test_pct}%\n"
            f"resampling scheme       : RepeatedStratifiedKFold "
            f"(NOT shuffle-split)\n"
            f"univariate screen       : {args.univariate_screen}\n"
            f"random seed             : {args.seed}\n"
        )
    print(f"[CV-DESIGN] {out}")
    return {"methods_paragraph": methods, "figure_legend": legend,
            "outer_evaluations_global": n_eval_g,
            "outer_evaluations_per_arm": n_eval_a,
            "scheme": "RepeatedStratifiedKFold"}


def main():
    global DATA_PATH, RESULTS_DIR, RANDOM_SEED, CLASSIFIERS
    global GLOBAL_N_OUTER_FOLDS, GLOBAL_N_REPEATS, GLOBAL_N_INNER_FOLDS
    global ARM_N_OUTER_FOLDS, ARM_N_REPEATS, ARM_N_INNER_FOLDS
    global CORR_THRESHOLD, NZV_RATIO_THRESHOLD
    global NZV_FREQ_GLOBAL, NZV_FREQ_ARM, NZV_FREQ_THRESHOLD
    global STABILITY_THRESHOLD_GLOBAL, STABILITY_THRESHOLD_ARM
    global N_JOBS
    global UNIVARIATE_SCREEN, UNIV_SCREEN_FDR_Q, UNIV_SCREEN_MAX_K
    global UNIV_SCREEN_MIN_K, FEATURE_POOL, ALL_MODS
    global SIGNATURE_SOURCE

    args = parse_args()

    SIGNATURE_SOURCE = args.signature_source
    # --consensus_only is meaningless without the consensus phase itself.
    if args.consensus_only:
        args.consensus = True

    DATA_PATH                = args.data_path
    RESULTS_DIR              = args.results_dir
    RANDOM_SEED              = args.seed
    GLOBAL_N_OUTER_FOLDS     = args.outer_folds_global
    GLOBAL_N_REPEATS         = args.repeats_global
    GLOBAL_N_INNER_FOLDS     = args.inner_folds_global
    ARM_N_OUTER_FOLDS        = args.outer_folds_arm
    ARM_N_REPEATS            = args.repeats_arm
    ARM_N_INNER_FOLDS        = args.inner_folds_arm
    CORR_THRESHOLD           = args.corr_threshold
    NZV_FREQ_GLOBAL          = args.nzv_freq_global
    NZV_FREQ_ARM             = args.nzv_freq_arm
    NZV_FREQ_THRESHOLD       = args.nzv_freq_global  # default; overridden per-experiment
    NZV_RATIO_THRESHOLD      = args.nzv_ratio
    STABILITY_THRESHOLD_GLOBAL = args.stability_thresh_global
    STABILITY_THRESHOLD_ARM    = args.stability_thresh_arm
    N_JOBS                     = args.n_jobs
    UNIVARIATE_SCREEN          = (args.univariate_screen == "in_fold")
    UNIV_SCREEN_FDR_Q          = args.univ_fdr_q
    UNIV_SCREEN_MAX_K          = args.univ_max_k
    UNIV_SCREEN_MIN_K          = args.univ_min_k
    FEATURE_POOL               = args.feature_pool

    CLASSIFIERS  = build_classifiers(RANDOM_SEED)
    active_clfs  = [c for c in args.classifiers if c in CLASSIFIERS]
    splits_dir   = args.splits_dir or RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # load_or_generate_splits writes into splits_dir but never created it, so
    # any run passing a --splits_dir that did not already exist died with
    # FileNotFoundError after completing all the data loading.
    Path(splits_dir).mkdir(parents=True, exist_ok=True)

    # Warn when splits_dir is not explicitly set in cc_only mode.
    # In that case splits default to RESULTS_DIR, which differs between
    # the two strategy runs — test fold identity is not guaranteed.
    splits_dir_explicit = args.splits_dir is not None
    if args.training_data == "cc_only" and not splits_dir_explicit:
        print(
            "\n[WARNING] --splits_dir not set.\n"
            "  Outer test folds will be saved to --results_dir, which is "
            "separate from your expanded-training run.\n"
            "  If you intend to compare expanded vs cc_only strategies, "
            "re-run BOTH with the same --splits_dir so test folds are "
            "guaranteed identical:\n"
            "    --splits_dir ./shared_splits\n"
        )

    # Determine which modes to run
    modes = (["elasticnet", "best_per_fold", "ensemble_weighted"]
             if args.mode == "all" else [args.mode])

    print("=" * 70)
    print("PREDIX HER2 — MULTIMODAL pCR PREDICTION PIPELINE")
    print(f"  Mode(s)        : {modes}")
    if len(modes) > 1 or modes[0] != "elasticnet":
        print(f"  Classifiers    : {active_clfs}")
    print(f"  Experiments    : {args.experiments}")
    print(f"  Training data  : {args.training_data}  "
          f"({'per-modality expanded sets' if args.training_data == 'expanded' else 'complete-case only'})")
    print(f"  Splits dir     : {splits_dir}"
          f"{'  ← shared, test folds guaranteed identical' if splits_dir_explicit else '  ← defaults to results_dir (not shared)'}")
    print(f"  Global CV      : {GLOBAL_N_OUTER_FOLDS}-fold × {GLOBAL_N_REPEATS} "
          f"= {GLOBAL_N_OUTER_FOLDS*GLOBAL_N_REPEATS} outer folds")
    print(f"  Arm CV         : {ARM_N_OUTER_FOLDS}-fold × {ARM_N_REPEATS} "
          f"= {ARM_N_OUTER_FOLDS*ARM_N_REPEATS} outer folds per arm")
    print(f"  Results dir    : {RESULTS_DIR}")
    print(f"  Parallel jobs  : {N_JOBS} ({'all CPUs' if N_JOBS == -1 else 'sequential' if N_JOBS == 1 else f'{N_JOBS} workers'})")
    print(f"  Feature pool   : {FEATURE_POOL}")
    print(f"  Univar. screen : {args.univariate_screen}"
          + ("  ← leakage-free primary analysis"
             if UNIVARIATE_SCREEN else
             "  ← LEGACY: candidate pool was outcome-informed upstream"))
    print(f"  Version        : {PIPELINE_VERSION}")
    print("=" * 70)

    prov = write_provenance(args, RESULTS_DIR, active_clfs, splits_dir)
    # Canonical CV-design text: the manuscript's Methods, Results and figure
    # legends must all be copied from this generated file so the three
    # descriptions can never disagree again (as they did in the submission).
    cv_design = write_cv_design_statement(args, RESULTS_DIR)
    try:
        import json as _json
        prov["cv_design"] = cv_design
        with open(Path(RESULTS_DIR) / "run_provenance.json", "w",
                  encoding="utf-8") as f:
            _json.dump(prov, f, indent=2)
    except Exception as e:
        print(f"[PROVENANCE] cv_design embed skipped: {e}")

    df_enc   = load_and_encode_data(DATA_PATH)

    # ── Optional feature restriction (--include_features) ────────────────────
    # Restrict the feature universe to an explicit list (e.g. the features
    # shared with an external cohort) BEFORE modality definition, so every
    # downstream step — screen, signature discovery, consensus — sees only
    # the transferable features.
    if args.include_features:
        with open(args.include_features, encoding="utf-8") as f:
            keep = [ln.strip() for ln in f
                    if ln.strip() and not ln.strip().startswith("#")]
        prefixes = ("Clin_", "RNA_", "DNA_", "Prot_", "WSI_")
        feat_cols = [c for c in df_enc.columns if c.startswith(prefixes)]
        keep_set  = set(keep) | {"Clin_Arm"}   # arm split always needs it
        missing   = [c for c in keep if c not in df_enc.columns]
        dropped   = [c for c in feat_cols if c not in keep_set]
        df_enc    = df_enc.drop(columns=dropped)
        print(f"\n[INCLUDE] --include_features {args.include_features.name}: "
              f"kept {len(feat_cols) - len(dropped)} of {len(feat_cols)} "
              f"feature columns, dropped {len(dropped)}")
        if missing:
            # Distinguish Tier-1 curation removals from genuine mismatches:
            # shared-feature lists are exported from the RAW file, so with
            # --feature_pool curated the Tier-1 co-amplicon/composite
            # features are gone before this filter runs. Reporting them as
            # spelling errors would be actively misleading.
            tier1_hits = [c for c in missing if c in TIER1_REMOVE]
            truly_missing = [c for c in missing if c not in TIER1_REMOVE]
            if tier1_hits:
                print(f"[INCLUDE] NOTE — {len(tier1_hits)} listed feature(s) "
                      f"were removed by Tier 1 curation before this filter "
                      f"ran (candidate universe is smaller than the exported "
                      f"shared list; use --feature_pool full if the run must "
                      f"see them):")
                for c in tier1_hits:
                    print(f"          - {c}")
            if truly_missing:
                print(f"[INCLUDE] WARNING — {len(truly_missing)} listed "
                      f"feature(s) not in the data file (check "
                      f"spelling/aliases):")
                for c in truly_missing:
                    print(f"          - {c}")

    features = define_modality_features(df_enc)

    # ── Modality restriction (--modalities) ──────────────────────────────────
    # Rebind the module-global ALL_MODS to the requested modalities that
    # actually have feature columns. Workers receive it via worker_cfg.
    requested = list(dict.fromkeys(args.modalities))
    active = []
    for m in requested:
        if m == "Clin":
            # Clin_Arm is force-kept by --include_features for the arm split,
            # so Clin_global is never empty. Testing it would (a) let a
            # "clinical" model whose only feature is the randomisation arm
            # into the global experiment, and (b) pass Clin into arm
            # experiments where Clin_arm (which excludes Clin_Arm) is empty —
            # crashing every worker mid-run. Test the real clinical features.
            ok = bool([c for c in features.get("Clin_global", [])
                       if c != "Clin_Arm"])
        else:
            ok = bool(features.get(m))
        if ok:
            active.append(m)
        else:
            print(f"[MODALITIES] {m} requested but no usable {m} feature "
                  f"columns present after restriction — skipped")
    if not active:
        raise SystemExit("[MODALITIES] No requested modality has any feature "
                         "columns — nothing to model.")
    ALL_MODS = active
    if len(ALL_MODS) < 5:
        print(f"[MODALITIES] Active modalities: {ALL_MODS}")
        if len(ALL_MODS) == 1:
            print(f"[MODALITIES] Single modality — the fusion layer reduces "
                  f"to a recalibration of the {ALL_MODS[0]} predictions; the "
                  f"scientific deliverable of this run is the "
                  f"{ALL_MODS[0]} consensus model.")

    df_cc    = get_complete_case(df_enc, features, active_mods=ALL_MODS)
    df_dhp   = df_cc[df_cc["Clin_Arm"] == 0].reset_index(drop=True)
    df_tdm1  = df_cc[df_cc["Clin_Arm"] == 1].reset_index(drop=True)
    print(f"\n[COHORT] complete-case n={len(df_cc)} "
          f"(DHP={len(df_dhp)}, T-DM1={len(df_tdm1)}), "
          f"pCR={df_cc['pCR'].mean():.3f}")

    # ── Per-modality datasets ──────────────────────────────────────────────────
    # Only computed when expanded training is requested.
    # In cc_only mode, mod_datasets is None → run_experiment uses the
    # complete-case training set only (use_expanded=False branch).
    if args.training_data == "expanded":
        mod_datasets_global = get_modality_datasets(df_enc, features)
        mod_datasets_dhp    = get_modality_datasets(
            df_enc[df_enc["Clin_Arm"] == 0].copy(), features)
        mod_datasets_tdm1   = get_modality_datasets(
            df_enc[df_enc["Clin_Arm"] == 1].copy(), features)

        print("\n[EXPANDED TRAINING] Available patients per modality:")
        print(f"  {'Modality':<6}  {'Global':>8}  {'DHP':>6}  {'T-DM1':>7}  "
              f"{'CC baseline':>12}")
        for mod_key, label in [("Clin_global","Clin"),("RNA","RNA"),
                                ("DNA","DNA"),("Prot","Prot"),("WSI","WSI")]:
            n_g = len(mod_datasets_global[mod_key])
            n_d = len(mod_datasets_dhp.get(mod_key, pd.DataFrame()))
            n_t = len(mod_datasets_tdm1.get(mod_key, pd.DataFrame()))
            print(f"  {label:<6}  {n_g:>8}  {n_d:>6}  {n_t:>7}  "
                  f"{'110 / 59 / 51':>12}")
    else:
        mod_datasets_global = None
        mod_datasets_dhp    = None
        mod_datasets_tdm1   = None
        print("\n[CC-ONLY TRAINING] Training restricted to complete-case "
              f"patients (n={len(df_cc)}) for all modalities.")

    exp_map = {
        "global": (df_cc,   "Clin_global", mod_datasets_global,
                   GLOBAL_N_OUTER_FOLDS, GLOBAL_N_REPEATS, GLOBAL_N_INNER_FOLDS),
        "dhp":    (df_dhp,  "Clin_arm",    mod_datasets_dhp,
                   ARM_N_OUTER_FOLDS,    ARM_N_REPEATS,    ARM_N_INNER_FOLDS),
        "tdm1":   (df_tdm1, "Clin_arm",    mod_datasets_tdm1,
                   ARM_N_OUTER_FOLDS,    ARM_N_REPEATS,    ARM_N_INNER_FOLDS),
    }

    # Cross-arm CC dataframes for counterfactual analysis. Only arm experiments
    # consume these — global uses cross_arm_df=None and produces no cross_arm_preds.
    # Pre-treatment features make cross-arm prediction meaningful: applying a
    # DHP-trained model to T-DM1 patients' pre-treatment features estimates
    # P(pCR | they had received DHP). Saved per-fold under Fused_ElasticNet
    # fold dict key "cross_arm_preds" = {patient_id: float}.
    cross_arm_map = {
        "global": (None, None),
        "dhp":    (df_tdm1.reset_index(drop=True), "T-DM1"),
        "tdm1":   (df_dhp.reset_index(drop=True),  "DHP"),
    }

    for exp_name in args.experiments:
        df_cc_exp, clin_key, mod_ds, n_outer, n_rep, n_inner = exp_map[exp_name]
        exp_dir = RESULTS_DIR / exp_name
        exp_dir.mkdir(exist_ok=True)

        # CV splits defined on complete-case patients — shared across all modes
        splits = load_or_generate_splits(
            splits_dir, exp_name, df_cc_exp["pCR"].values,
            n_outer, n_rep, n_inner, RANDOM_SEED,
            pids=df_cc_exp["patient_id"].values)
        # methods_cv_statement.txt and the provenance 'cv_design' block are
        # generated from args — if a LEGACY splits file (no metadata block)
        # delivered a different design, they would certify a CV design the
        # run did not execute. New-format files can't reach here mismatched.
        if len(splits) != n_outer * n_rep:
            print(f"[CV-DESIGN] WARNING: {exp_name} runs {len(splits)} outer "
                  f"folds from a legacy splits file, but args imply "
                  f"{n_outer * n_rep}. methods_cv_statement.txt and "
                  f"run_provenance.json 'cv_design' do NOT describe the "
                  f"executed design — regenerate the splits before quoting "
                  f"either.")

        cross_arm_df, cross_arm_label = cross_arm_map[exp_name]

        if args.consensus_only:
            # Run 5: re-derive the consensus from an existing discovery PKL
            # without repeating the per-fold model search. Only valid when the
            # PKL was produced with the same candidate pool — changing
            # TIER1_REMOVE (as run 5 does for RNA_FCGR3B) invalidates it, so
            # this path is for isolating --signature_source, not for shipping.
            print(f"\n[{exp_name.upper()}] --consensus_only: skipping the "
                  f"discovery CV loop, re-using the existing discovery PKL.")
        else:
            for mode in modes:
                # All active classifiers are passed for all modes.
                # _fit_signature_model (elasticnet + expanded) uses all of them
                # for signature discovery and selects the winner by inner AUROC.
                # _fit_best_per_fold and _fit_ensemble also use all active_clfs.
                run_experiment(
                    df_cc_exp=df_cc_exp, features=features, clin_key=clin_key,
                    splits=splits, exp_name=exp_name, output_dir=exp_dir,
                    mode=mode,
                    active_clfs=active_clfs,
                    mod_datasets=mod_ds,
                    cross_arm_df=cross_arm_df,
                    cross_arm_label=cross_arm_label)

        # ── R2 CONSENSUS FINALIZATION ─────────────────────────────────────
        # After the discovery CV loop completes, (1) aggregate per-fold
        # signatures and classifier choices into a single consensus, and
        # (2) re-evaluate that FROZEN consensus under the SAME CV splits
        # with classifier + fusion re-fit within each fold. The pooled
        # OOF AUROC from (2) is the PRIMARY HEADLINE for the paper.
        if args.consensus and "elasticnet" in modes:
            disc_pkl = exp_dir / f"{exp_name}_elasticnet_results.pkl"
            if not disc_pkl.exists():
                if args.consensus_only:
                    # --consensus_only exists ONLY to re-derive from an
                    # existing PKL. Skipping would do nothing and exit 0, which
                    # run_step logs as DONE — a silent no-op run.
                    raise SystemExit(
                        f"[{exp_name.upper()}] --consensus_only was requested "
                        f"but {disc_pkl} does not exist. Nothing to "
                        f"re-finalise; refusing to exit 0 on a no-op.")
                print(f"\n[{exp_name.upper()} | CONSENSUS] "
                      f"Discovery PKL not found — skipping consensus phase.")
                continue
            print("\n" + "=" * 70)
            print(f"[{exp_name.upper()}] R2 CONSENSUS FINALIZATION")
            print("=" * 70)
            with open(disc_pkl, "rb") as f:
                disc_results = pickle.load(f)
            print(f"\n  [1/2] Aggregating per-fold discovery into consensus ...")
            consensus = finalize_consensus(
                disc_results, ALL_MODS=ALL_MODS,
                df_cc=df_cc_exp, features=features)

            print(f"\n  [2/2] Re-evaluating frozen consensus under same CV splits ...")
            eval_result = evaluate_consensus(
                df_cc_exp=df_cc_exp, features=features, clin_key=clin_key,
                splits=splits, exp_name=exp_name, output_dir=exp_dir,
                consensus=consensus, ALL_MODS=ALL_MODS,
                mod_datasets=mod_ds)  # None in cc_only; expanded dict otherwise

            write_consensus_summary(consensus, eval_result, exp_name, exp_dir)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Results → {RESULTS_DIR}")
    print("  Report :")
    print(f"    python3 generate_report.py \\")
    print(f"        --results_dir {RESULTS_DIR} --out_dir ./report")
    print("=" * 70)


if __name__ == "__main__":
    main()
