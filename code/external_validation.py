#!/usr/bin/env python3
"""
EXTERNAL VALIDATION OF THE TRANSCRIPTOMIC pCR MODEL — PREDIX HER2
=================================================================
Independent validation of the RNA-only model in two external cohorts.

WHY THIS EXISTS
---------------
Peer review made the point that because the candidate panel was assembled with
the outcome in view across the whole PREDIX cohort, internal cross-validation
cannot settle the question of generalisation, and that independent validation
was "feasible at minimum for the genomic/transcriptomic-only version". This
script does exactly that. The RNA model is locked on PREDIX and then applied,
unchanged, to two cohorts that contributed nothing to its development:

  I-SPY2 (GSE194040)          n=44   pCR 26 (59.1%)   trastuzumab/pertuzumab
                                                       + chemotherapy  -> DHP-like
  NCT02326974 (GSE243375)     n=129  pCR 64 (49.6%)   T-DM1 + pertuzumab
                                                       -> T-DM1-like

Each external cohort validates the arm-matched PREDIX model. No external
patient is used for feature selection, hyper-parameter choice, or thresholds.

THE THREE THINGS THAT MAKE THIS HONEST
--------------------------------------
1. LOCK, THEN APPLY. The model is fitted once on PREDIX and frozen. No external
   outcome and no external patient contributes to feature selection,
   hyper-parameters, coefficients or thresholds. The per-cohort standardisation
   IS estimated on the external cohort (unsupervised, outcome-blind), which
   makes the procedure transductive — see the note in the workbook.
   Recalibration, where reported, is a separate row so its effect is visible
   rather than absorbed.

2. THE COMPARISON IS LIKE FOR LIKE. The internal estimate quoted alongside each
   external result is computed on the SAME restricted feature set the external
   cohort supports. On the current delivery PREDIX carries 42 RNA columns and
   the transferable sets are 39 (I-SPY2) and 38 (NCT02326974); those counts
   move with the input file, so read them from the run log rather than from
   this docstring.

3. SCALE HARMONISATION IS DECLARED, NOT HIDDEN. The three cohorts are on
   incompatible scales — PREDIX signatures are z-scored while both external
   files carry raw values, and several signatures have standard deviations an
   order of magnitude apart (RNA_Th2 cells: SD 0.84 in PREDIX, 0.019 in
   NCT02326974). A model trained on PREDIX coefficients therefore cannot be
   applied to raw external values. Every feature is standardised WITHIN its own
   cohort before the model is applied. Because that choice could itself drive
   the result, the script runs the whole validation under two independent
   harmonisation schemes and reports both:

     zscore : per-cohort mean 0, SD 1.
     rank   : per-cohort rank transform to a common uniform scale, which
              additionally removes any difference in distributional shape and
              is insensitive to outliers.

   A result that holds under both is a property of the biology. One that
   appears under only one is an artefact of the harmonisation.

A KNOWN NAME MISMATCH
---------------------
The external files spell the ADC-trafficking signature `RNA_ADC_traficking`
(one f); PREDIX spelled it `RNA_ADC_trafficking` (two f). FEATURE_ALIASES below
repairs that and any comparable mismatch, and the script prints what it renamed.

NOTE, as of the current delivery, that this particular alias is INERT: the
authors withdrew RNA_ADC_trafficking upstream, so PREDIX carries it under
neither spelling. `harmonise_columns` therefore applies an alias only when the
target exists in PREDIX, and reports any it declined to apply. Earlier versions
renamed unconditionally, producing a dead column and a "renamed / rescued" row
in Feature_provenance for a feature present in no model — and describing it as
defining the S3 group of a treatment-selection scheme that has itself been
withdrawn. Both claims are gone.

`RNA_TILs` is entirely missing in NCT02326974 and is dropped for that cohort
only, with the drop reported.

USAGE
-----
  python3 external_validation.py \\
      --predix     clin_multiomics_curated_metrics_PREDIX_HER2_new.txt \\
      --ispy2      RNA_curated_metrics_ISPY2.txt \\
      --nct        RNA_curated_metrics_NCT02326974.txt \\
      --out_dir    ./report

OUTPUTS
-------
  {out_dir}/tables/revision/external_validation{suffix}.xlsx
  {out_dir}/figures/revision/revfig06_external_validation{suffix}.pdf

  where {suffix} comes from --output_suffix (default: empty).

  RUN 5 — WHY THE SUFFIX EXISTS. Run 4 wrote the pooled-model results into a
  separate directory (report_pooled_external/) but kept identical BASENAMES, so
  two different analyses produced two files called
  revfig06_external_validation.pdf. Opening the obvious path showed the
  arm-matched model, which is the one that FAILS on NCT02326974 (0.572), while
  the pooled result (0.679) sat under a name that looked like a duplicate. That
  cost real confusion. Pass --output_suffix _POOLED for the pooled run so the
  two are distinguishable by filename alone.
"""

import argparse
import hashlib
import pickle
import warnings
from pathlib import Path

# RUN 5 FIX: was a bare filterwarnings("ignore"). It silenced cv_estimands'
# two guard warnings — that a PKL carries only positional test_idx rather than
# test_pids, and that N of R repeats do not predict every patient. The second
# matters here: the locked-mode internal comparator is read straight from the
# consensus PKL, so an aborted pipeline run would yield per-repeat AUROCs
# computed on subsets of patients, averaged into internal_AUROC, with no trace
# in the log or the workbook — and internal_AUROC is the number the external
# result is judged against. Keep RuntimeWarning audible.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openpyxl

from revision_analyses import (
    bootstrap_metric_ci, calibration_metrics, format_ci, wilson_ci,
    delong_test, _write_sheet, _savefig, _asym_err, benjamini_hochberg,
    N_BOOT, BOOT_SEED,
)
import cv_estimands as CE


# =============================================================================
# CONSTANTS
# =============================================================================

# Name mismatches between PREDIX and the external files. Keys are the external
# spelling, values the PREDIX spelling. Applied to the external frames.
FEATURE_ALIASES = {
    "RNA_ADC_traficking": "RNA_ADC_trafficking",   # single f externally
}

# PREDIX RNA columns that cannot be harmonised and are excluded up front.
# RNA_sspbc.subtype is categorical and is handled separately via its dummies.
NON_TRANSFERABLE = {"RNA_TCR_clonality", "RNA_BCR_clonality"}

# Appended to output basenames; set from --output_suffix in main(). Run 5.
OUTPUT_SUFFIX = ""

COHORTS = {
    "I-SPY2": {
        "arm": "DHP", "arm_code": 0,
        "label": "I-SPY2 (GSE194040) — trastuzumab/pertuzumab + chemotherapy",
    },
    "NCT02326974": {
        "arm": "T-DM1", "arm_code": 1,
        "label": "NCT02326974 (GSE243375) — T-DM1 + pertuzumab",
    },
}

COHORT_COLOR = {"I-SPY2": "#2166ac", "NCT02326974": "#d6604d",
                "PREDIX (internal)": "#6a1f6a"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 300,
})


# =============================================================================
# SECTION 1 — HARMONISATION
# =============================================================================

def harmonise_columns(df_ext, cohort_name, predix_cols=None):
    """Apply the alias map and report every rename, so nothing is silent.

    RUN 5 FIX — only rename when the TARGET EXISTS IN PREDIX. The alias map
    still carries RNA_ADC_traficking -> RNA_ADC_trafficking, but the authors
    withdrew that feature upstream and the production PREDIX file contains it
    under NEITHER spelling. The rename therefore produced a dead column in the
    external frame, and because `shared_features` iterates PREDIX columns the
    feature was invisible to `excluded` — yet a "renamed / rescued" row was
    still written into Feature_provenance, for a feature in no model, citing a
    treatment-selection scheme (S1/S2/S3) that has itself been withdrawn.
    That was a false claim in a supplementary table.
    """
    renames, skipped = {}, {}
    for c in df_ext.columns:
        if c not in FEATURE_ALIASES:
            continue
        target = FEATURE_ALIASES[c]
        if predix_cols is not None and target not in set(predix_cols):
            skipped[c] = target
            continue
        renames[c] = target
    if renames:
        df_ext = df_ext.rename(columns=renames)
        for old, new in renames.items():
            print(f"  [{cohort_name}] renamed {old!r} -> {new!r}")
    for old, new in skipped.items():
        print(f"  [{cohort_name}] alias {old!r} -> {new!r} NOT applied: "
              f"{new!r} is absent from PREDIX (withdrawn upstream). The "
              f"external column is left untouched and unused.")
    return df_ext, renames


def shared_features(df_predix, df_ext, cohort_name):
    """
    Determine the RNA features usable in both cohorts.

    A feature is usable when it is present in both frames, numeric in both, has
    at least one observed value in the external cohort, and is not on the
    non-transferable list. Every exclusion is recorded with its reason so the
    supplementary table can state exactly why the external model uses fewer
    features than the internal one.
    """
    predix_rna = [c for c in df_predix.columns
                  if c.startswith("RNA_") and c != "RNA_sspbc.subtype"]
    excluded = []
    usable = []

    for col in predix_rna:
        if col in NON_TRANSFERABLE:
            excluded.append((col, "not measured in external cohorts"))
            continue
        if col not in df_ext.columns:
            excluded.append((col, "absent from external file"))
            continue
        ext_vals = pd.to_numeric(df_ext[col], errors="coerce")
        if ext_vals.notna().sum() == 0:
            excluded.append((col, "entirely missing in external cohort"))
            continue
        # RUN 5: qualifying on "at least one observed value" let a feature with
        # 1 of 129 values through. Missing cells standardise to the cohort mean
        # (0), so such a feature contributes nothing while still being counted
        # in n_model_features and printed as "used". Same for a feature that is
        # constant externally: SD 0 -> all zeros. Neither was reported.
        # Empirically clean on the current two cohorts (zero missing, zero
        # near-constant among shared features) — this guards the next one.
        _obs_frac = float(ext_vals.notna().mean())
        if _obs_frac < 0.8:
            excluded.append((col, f"observed in only {_obs_frac:.0%} of the "
                                  f"external cohort (<80%)"))
            continue
        if ext_vals.dropna().nunique() < 2:
            excluded.append((col, "constant in the external cohort — "
                                  "standardises to all zeros"))
            continue
        # to_numeric(errors="coerce") ALWAYS returns a numeric dtype, so an
        # is_numeric_dtype test on its result can never fire. The real
        # question is whether anything survives coercion: a text-valued
        # PREDIX column would coerce to all-NaN, silently standardise to
        # all-zeros internally while staying live externally.
        if pd.to_numeric(df_predix[col], errors="coerce").notna().sum() == 0:
            excluded.append((col, "non-numeric or empty in PREDIX"))
            continue
        usable.append(col)

    print(f"  [{cohort_name}] {len(usable)} shared RNA features, "
          f"{len(excluded)} excluded")
    for col, reason in excluded:
        print(f"      excluded {col}: {reason}")
    return usable, excluded


def standardise_within_cohort(X, method="zscore"):
    """
    Put a cohort's features on a scale comparable with the other cohorts'.

    zscore : subtract the cohort mean, divide by the cohort SD. Preserves the
             shape of each feature's distribution and only removes location and
             scale differences.
    rank   : replace each value by its within-cohort quantile, then map through
             the standard normal. Removes distributional shape differences too,
             so it is the more aggressive harmonisation, and it is insensitive
             to the outliers and heavy tails present in raw expression scores.

    Both are computed WITHIN each cohort independently and use no outcome
    information, so neither leaks the external labels into the model.

    Constant features (SD 0) are returned as zeros rather than NaN, so a
    feature that happens to be constant in one cohort contributes nothing
    instead of destroying the whole design matrix.
    """
    X = np.asarray(X, dtype=float)
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        obs = np.isfinite(col)
        if obs.sum() < 2:
            continue
        if method == "zscore":
            # RUN 5: nanmean/nanstd ignore NaN but NOT +/-inf, while `obs`
            # (np.isfinite) excludes both. One infinite value therefore made mu
            # and sd infinite, every z-score NaN, and nan_to_num later zeroed
            # the whole feature — silently dropping it from the model while it
            # still counted in n_model_features. Compute on the finite subset.
            mu = np.mean(col[obs])
            sd = np.std(col[obs])
            if sd > 1e-12:
                out[obs, j] = (col[obs] - mu) / sd
        elif method == "rank":
            r = stats.rankdata(col[obs])
            # Map ranks to (0,1) then to the normal quantile scale.
            u = (r - 0.5) / len(r)
            out[obs, j] = stats.norm.ppf(u)
        else:
            raise ValueError(f"unknown standardisation method: {method}")
        # Unobserved entries stay at 0, i.e. the cohort mean after
        # standardisation — the natural neutral value on this scale.
    return out


# =============================================================================
# SECTION 2 — MODEL
# =============================================================================

def build_model(name, params=None):
    """Instantiate one of the pipeline's classifier families."""
    params = params or {}
    if name == "ElasticNet_LR":
        m = LogisticRegression(penalty="elasticnet", solver="saga",
                               l1_ratio=0.5, max_iter=4000, random_state=0)
    elif name == "RandomForest":
        m = RandomForestClassifier(random_state=0, n_jobs=1)
    elif name == "ExtraTrees":
        m = ExtraTreesClassifier(random_state=0, n_jobs=1)
    elif name == "HistGradBoost":
        m = HistGradientBoostingClassifier(random_state=0)
    elif name == "SVM_Linear":
        m = SVC(kernel="linear", probability=True, random_state=0)
    elif name == "SVM_RBF":
        # RUN 5: the pipeline's registry has six families, this had five. If an
        # RNA consensus winner ever came back as SVM_RBF the locked run died
        # with "unknown classifier", and the only way forward would have been
        # substituting a different classifier — i.e. re-tuning the locked model.
        # Production passes five families so it cannot win today; the two lists
        # were coupled by convention only.
        m = SVC(kernel="rbf", probability=True, random_state=0)
    else:
        raise ValueError(
            f"unknown classifier: {name}. This registry must stay in step with "
            f"CLASSIFIERS in multimodal_pcr_pipeline.py — a locked consensus "
            f"can name any family the pipeline was run with.")
    if params:
        m.set_params(**params)
    return m


def internal_cv_estimate(X, y, clf_name, grid, n_splits=5, n_repeats=10,
                         seed=42):
    """
    Honest internal estimate on the restricted feature set, by repeated
    stratified cross-validation with the hyper-parameter search inside each
    training fold.

    This is the internal comparator of the GRID (sensitivity) design only.
    Quoting the manuscript's headline AUROC instead would compare a 48-feature
    model against a 44-feature one and attribute the difference to cohort
    transfer.

    Returns one out-of-fold prediction per patient obtained by AVERAGING that
    patient's predictions over the CV repeats (plus the per-patient count),
    and validate_cohort's grid branch bootstraps patients on those averaged
    values. NOTE this is a repeat-ensemble estimand and differs from the
    locked design, whose internal comparator is the mean over CV repeats of
    the pooled out-of-fold metric with a patient-level cluster-bootstrap CI
    (cv_estimands). It is retained unchanged as the sensitivity path.
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)
    sums = np.zeros(len(y))
    counts = np.zeros(len(y))

    for tr, te in rskf.split(X, y):
        gs = GridSearchCV(build_model(clf_name), grid, cv=3,
                          scoring="roc_auc", refit=True, n_jobs=1)
        try:
            gs.fit(X[tr], y[tr])
            p = gs.best_estimator_.predict_proba(X[te])[:, 1]
        except Exception:
            continue
        sums[te] += p
        counts[te] += 1

    ok = counts > 0
    oof = np.full(len(y), np.nan)
    oof[ok] = sums[ok] / counts[ok]
    return oof, counts


def lock_model(X, y, clf_name, grid, seed=42):
    """
    Fit the final locked model on ALL PREDIX patients.

    Hyper-parameters are chosen by internal cross-validation on PREDIX only.
    After this function returns, the model is frozen: the external cohorts see
    it exactly as it is here.
    """
    gs = GridSearchCV(build_model(clf_name), grid, cv=5, scoring="roc_auc",
                      refit=True, n_jobs=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


CANDIDATE_MODELS = {
    "ElasticNet_LR": {"C": [0.01, 0.05, 0.1, 0.5, 1.0]},
    "RandomForest":  {"n_estimators": [300], "max_depth": [None, 5],
                      "min_samples_leaf": [1, 5]},
    "ExtraTrees":    {"n_estimators": [300], "max_depth": [None, 5],
                      "min_samples_leaf": [1, 5]},
    "SVM_Linear":    {"C": [0.01, 0.1, 1.0]},
}


# =============================================================================
# SECTION 2b — LOCKED PIPELINE-CONSENSUS MODEL
# =============================================================================
#
# --locked_ispy2 / --locked_nct switch the validation from this script's own
# grid-searched model to the MAIN PIPELINE's consensus RNA model: the
# feature-selected signature, the winner classifier, and the modal
# hyperparameters produced by an RNA-only run of multimodal_pcr_pipeline.py
# (--modalities RNA --include_features shared_features_<cohort>.txt). In that
# mode:
#   - the locked model = consensus signature + winner_clf + params, refit once
#     on all PREDIX arm patients (no grid search — the model is frozen);
#   - the internal comparator = the pipeline's own frozen-consensus OOF
#     performance: the metric on each CV repeat's complete out-of-fold
#     vector, averaged over repeats, with a patient-level CLUSTER-bootstrap
#     CI (all repeats of a resampled patient move together) — the same
#     estimand revision_performance_CI.xlsx reports, on the same restricted
#     feature universe. Predictions are never averaged across repeats into
#     one number per patient (see cv_estimands.py for why that is biased);
#   - external patients still play no role in feature selection,
#     hyperparameters, or coefficients.

def _sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_locked_consensus(results_dir, exp_name, predix_path):
    """
    Load the pipeline's consensus RNA model spec from an RNA-only run.

    Reads {results_dir}/{exp_name}/{exp_name}_consensus_eval.pkl and returns
    {"signature", "winner_clf", "params", "blob", "exp", "results_dir"}.
    Verifies via run_provenance.json that the pipeline was trained on the
    same input file passed here as --predix; a mismatch means the locked
    model and the locking cohort come from different data.
    """
    results_dir = Path(results_dir)
    pkl = results_dir / exp_name / f"{exp_name}_consensus_eval.pkl"
    if not pkl.exists():
        raise FileNotFoundError(
            f"{pkl} not found — run the RNA-only pipeline first:\n"
            f"  python multimodal_pcr_pipeline.py --modalities RNA "
            f"--include_features shared_features_<cohort>.txt "
            f"--experiments {exp_name} --consensus ...")
    with open(pkl, "rb") as f:
        blob = pickle.load(f)
    cons = blob.get("consensus", {}).get("RNA")
    if not cons or not cons.get("signature") or cons.get("winner_clf") in (
            None, "none"):
        raise ValueError(f"{pkl} has no usable RNA consensus "
                         f"(signature/classifier missing).")

    prov = results_dir / "run_provenance.json"
    if prov.exists():
        import json
        with open(prov, encoding="utf-8") as f:
            p = json.load(f)
        trained_sha = p.get("input_data", {}).get("sha256")
        here_sha = _sha256_of(predix_path)
        if not trained_sha:
            # RUN 5 FIX: `if trained_sha and ...` silently skipped the whole
            # check when the key was absent. The identity check downstream
            # compares SETS OF ROW POSITIONS, so its entire guarantee that the
            # refit cohort is the pipeline's cohort rests on this hash. A
            # different file with the same row count and the same complete-case
            # positions would pass unnoticed.
            raise ValueError(
                f"{prov} has no input_data.sha256, so the locked model cannot "
                f"be tied to --predix. The downstream identity check compares "
                f"row positions only and would pass on a different file with "
                f"the same shape. Re-run the RNA-only pipeline to regenerate "
                f"provenance.")
        if trained_sha != here_sha:
            raise ValueError(
                f"Pipeline at {results_dir} was trained on a DIFFERENT input "
                f"file than --predix "
                f"({p.get('input_data', {}).get('path')}). The locked model "
                f"and the locking cohort must come from the same data. "
                f"Re-run the RNA-only pipeline on this file, or pass the "
                f"file it was trained on.")
    else:
        print(f"  [LOCKED] note: no run_provenance.json in {results_dir}; "
              f"cannot verify the training file matches --predix.")

    print(f"  [LOCKED] {exp_name}: K={len(cons['signature'])} features, "
          f"clf={cons['winner_clf']}, params={cons['params']}")
    return {"signature": list(cons["signature"]),
            "winner_clf": cons["winner_clf"],
            "params": cons["params"] or {},
            "blob": blob, "exp": exp_name, "results_dir": str(results_dir)}


# =============================================================================
# SECTION 3 — VALIDATION DRIVER
# =============================================================================

def validate_cohort(df_predix, df_ext, cohort_name, method, clf_name,
                    arm_code=None, seed=42, n_boot=N_BOOT,
                    locked_spec=None):
    """
    Run the full lock-and-apply validation for one cohort under one
    harmonisation scheme.

    locked_spec (optional): dict from load_locked_consensus(). When given,
    the validated model is the MAIN PIPELINE's consensus RNA model — its
    feature-selected signature, winner classifier and modal hyperparameters —
    refit once (frozen, no grid search) on all PREDIX arm patients; the
    internal comparator is the pipeline's own frozen-consensus OOF
    performance — mean over CV repeats of the pooled out-of-fold metric,
    95% patient-level cluster-bootstrap CI (cv_estimands). When None, this
    script's original grid-searched all-shared-features model is used (kept
    as a sensitivity analysis; its internal estimate is computed on one
    repeat-averaged OOF probability per patient, see internal_cv_estimate).

    Returns a dict with the internal and external results plus everything
    needed to audit the run.
    """
    info = COHORTS[cohort_name]
    mode_tag = "pipeline-locked" if locked_spec else "grid"
    print(f"\n  --- {cohort_name} | {method} | {clf_name} | {mode_tag} ---")

    # df_predix.columns, not dp.columns: dp is not bound until the arm-filter
    # block below, and the alias target's presence is a property of the file.
    df_ext, renames = harmonise_columns(df_ext, cohort_name, df_predix.columns)
    feats, excluded = shared_features(df_predix, df_ext, cohort_name)
    if len(feats) < 5:
        print(f"  [{cohort_name}] too few shared features — skipping")
        return None

    if locked_spec is not None:
        # The externally-applied features are the pipeline's consensus
        # signature. Every signature feature must be measurable in the
        # external cohort; the pipeline run was restricted to the shared
        # feature list precisely so that this holds.
        clf_name = locked_spec["winner_clf"]
        sig = locked_spec["signature"]
        missing_sig = [f for f in sig if f not in feats]
        if missing_sig:
            raise ValueError(
                f"[{cohort_name}] consensus signature feature(s) not "
                f"available externally: {missing_sig}. The RNA-only pipeline "
                f"run must use --include_features "
                f"shared_features_{cohort_name}.txt.")
        # Keep the FULL shared list for the completeness filter below: the
        # pipeline's complete case is defined over all shared RNA features,
        # not just the signature. Filtering on signature columns alone could
        # admit a patient the pipeline never trained on (complete on the
        # signature, missing elsewhere).
        shared_all = list(feats)
        feats = sig
        print(f"  [{cohort_name}] locked signature: {len(feats)} features "
              f"({', '.join(feats[:5])}{', ...' if len(feats) > 5 else ''})")

    # ── PREDIX side: restrict to the arm this cohort corresponds to ──────────
    # An external T-DM1 cohort must be predicted by the PREDIX T-DM1 model, not
    # by the pooled model, because the treatments differ.
    dp = df_predix.copy()
    if arm_code is not None:
        # RUN 5 FIX: the missing-column case used to fall through the compound
        # condition silently, refitting on all 197 patients while every output
        # still claimed arm matching. Locked mode caught it via the identity
        # check; grid mode did not. Fail loudly instead.
        if "Clin_Arm" not in dp.columns:
            raise ValueError(
                f"Arm matching was requested for {cohort_name} (arm_code="
                f"{arm_code}) but PREDIX has no 'Clin_Arm' column. Refusing to "
                f"silently refit on the whole cohort while reporting an "
                f"arm-matched model.")
        # is_numeric_dtype, not `== object`: pandas 3 gives string columns a
        # dedicated `str` dtype, so an `== object` test silently skips the
        # mapping, leaves 'DHP'/'T-DM1' as text, and the arm filter then
        # selects zero patients.
        arm_num = (dp["Clin_Arm"] if pd.api.types.is_numeric_dtype(dp["Clin_Arm"])
                   else dp["Clin_Arm"].map({"DHP": 0, "T-DM1": 1}))
        # Guard the ENCODING CONVENTION, not just the match count. A file using
        # 1/2 coding, or DHP=1, would map to a valid-looking but wrong subset.
        _codes = set(int(v) for v in arm_num.dropna().unique())
        if not _codes <= {0, 1}:
            raise ValueError(
                f"Clin_Arm encodes to {sorted(_codes)}; this script assumes the "
                f"pipeline's DHP=0 / T-DM1=1 convention. Refusing to guess.")
        dp = dp[arm_num == arm_code]
        if len(dp) == 0:
            raise ValueError(
                f"Arm filter for {cohort_name} (Clin_Arm == {arm_code}) matched "
                f"no PREDIX patients. Observed Clin_Arm values: "
                f"{sorted(set(df_predix['Clin_Arm'].dropna().astype(str)))}")
        _n_other = int((arm_num == (1 - arm_code)).sum())
        if not (0.3 <= len(dp) / max(len(dp) + _n_other, 1) <= 0.7):
            print(f"  [{cohort_name}] WARNING: arm split is {len(dp)} vs "
                  f"{_n_other}; PREDIX HER2 randomised ~99/98. Check the "
                  f"Clin_Arm coding.")
        print(f"  [{cohort_name}] PREDIX {info['arm']} arm: {len(dp)} patients, "
              f"{int(dp['pCR'].sum())} pCR events")
    if locked_spec is not None:
        # Match the pipeline's cohort exactly: its RNA-only complete case is
        # defined over ALL shared RNA features, not just the signature.
        # Filtering on signature columns alone could admit a patient the
        # pipeline never trained on (complete on the signature, missing
        # elsewhere), which would also shift the within-cohort
        # standardisation of every feature.
        n_before = len(dp)
        dp = dp.dropna(subset=[c for c in shared_all if c in dp.columns])
        if len(dp) < n_before:
            print(f"  [{cohort_name}] locked mode: dropped "
                  f"{n_before - len(dp)} patient(s) without complete RNA "
                  f"(refit cohort n={len(dp)}, matching the pipeline's "
                  f"complete case)")
    y_int = dp["pCR"].astype(float).values
    X_int_raw = dp[feats].apply(pd.to_numeric, errors="coerce").values

    y_ext = df_ext["pCR"].astype(float).values
    X_ext_raw = df_ext[feats].apply(pd.to_numeric, errors="coerce").values

    # ── Harmonise, each cohort independently ─────────────────────────────────
    X_int = standardise_within_cohort(X_int_raw, method)
    X_ext = standardise_within_cohort(X_ext_raw, method)

    # Any residual non-finite value (a feature constant in one cohort) becomes
    # the cohort mean, which is 0 on the standardised scale.
    X_int = np.nan_to_num(X_int, nan=0.0, posinf=0.0, neginf=0.0)
    X_ext = np.nan_to_num(X_ext, nan=0.0, posinf=0.0, neginf=0.0)

    if locked_spec is not None:
        # ── Internal comparator: the pipeline's frozen-consensus OOF ─────────
        # performance, taken from the consensus-eval PKL: the metric on each
        # CV repeat's complete out-of-fold vector, averaged over repeats, with
        # a patient-level CLUSTER-bootstrap CI in which all R predictions of a
        # resampled patient travel together. This is the identical estimand
        # to revision_performance_CI.xlsx, on the same restricted feature
        # universe. The R predictions of a patient are NOT averaged into one
        # number first — that collapse scores a 200-model ensemble and carries
        # a systematic held-out-outcome shift (see cv_estimands.py).
        rm = CE.consensus_repeat_matrix(locked_spec["blob"], "RNA")
        if rm.n_patients == 0:
            raise ValueError(f"[{cohort_name}] consensus-eval PKL holds no "
                             f"RNA OOF predictions.")
        # HARD identity check: the refit cohort must be exactly the cohort
        # the pipeline trained on — equal counts with different patients
        # would silently compare different populations. The pipeline's
        # patient_id is the ROW POSITION in the input file
        # (`df["patient_id"] = range(len(df))`), NOT the raw patientID
        # column — and dp preserves the original RangeIndex through
        # filtering, so dp.index is the same identifier space.
        refit_pids = set(int(v) for v in dp.index)
        oof_pids = set(int(v) for v in rm.pids)
        if oof_pids != refit_pids:
            raise ValueError(
                f"[{cohort_name}] refit cohort != pipeline OOF cohort "
                f"({len(refit_pids)} vs {len(oof_pids)} patients; "
                f"symmetric difference "
                f"{sorted(oof_pids ^ refit_pids)[:10]}...). "
                f"--predix must be the same file the pipeline trained on, "
                f"with rows in the same order.")
        # BOOT_SEED (not the CLI seed) so this CI shares revision_analyses'
        # seed family. Note: revision_analyses adds a per-cell crc32 offset,
        # so POINT ESTIMATES match exactly while CI endpoints can differ in
        # the last digit between the two workbooks.
        int_auroc = CE.bootstrap_repeat_metric_ci(rm.P, rm.y, "AUROC",
                                                  n_boot=n_boot, seed=BOOT_SEED)
        int_auprc = CE.bootstrap_repeat_metric_ci(rm.P, rm.y, "AUPRC",
                                                  n_boot=n_boot,
                                                  seed=BOOT_SEED + 1)
        print(f"  [{cohort_name}] internal comparator: {rm.n_repeats} CV "
              f"repeats x {rm.n_patients} patients "
              f"({rm.n_events} events); mean over repeats of the pooled OOF "
              f"metric, patient-level cluster-bootstrap CI")

        # ── Frozen refit on all PREDIX arm patients, apply to external ───────
        # No grid search: signature, classifier and hyperparameters all come
        # from the pipeline consensus. This single fit is the locked model.
        model = build_model(clf_name, locked_spec["params"])
        model.fit(X_int, y_int)
        best_params, cv_score = locked_spec["params"], np.nan

        # ── Calibration parity ───────────────────────────────────────────────
        # The pipeline's internal OOF probabilities are always Platt-
        # calibrated; a raw predict_proba here would make the external
        # Brier/slope/intercept measure a probability pipeline the internal
        # model never uses (AUROC is unaffected — Platt is monotone). Fit the
        # same sigmoid on internal CV raw OOF of the frozen spec and apply it
        # to the external predictions used for calibration metrics.
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        try:
            cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            p_cv = cross_val_predict(build_model(clf_name,
                                                 locked_spec["params"]),
                                     X_int, y_int, cv=cv5,
                                     method="predict_proba")[:, 1]
            platt = LogisticRegression(C=1e6, max_iter=1000)
            platt.fit(p_cv.reshape(-1, 1), y_int)
        except Exception as e:
            print(f"  [{cohort_name}] Platt layer skipped ({e}); external "
                  f"calibration metrics use raw probabilities.")
            platt = None
    else:
        grid = CANDIDATE_MODELS[clf_name]

        # ── Internal estimate on the SAME restricted feature set ─────────────
        oof, counts = internal_cv_estimate(X_int, y_int, clf_name, grid,
                                           seed=seed)
        ok = np.isfinite(oof)
        int_auroc = bootstrap_metric_ci(y_int[ok], oof[ok], "AUROC",
                                        n_boot=n_boot, seed=seed)
        int_auprc = bootstrap_metric_ci(y_int[ok], oof[ok], "AUPRC",
                                        n_boot=n_boot, seed=seed + 1)

        # ── Lock on all PREDIX patients of this arm, apply to external ───────
        model, best_params, cv_score = lock_model(X_int, y_int, clf_name,
                                                  grid, seed=seed)
        platt = None
    p_ext = model.predict_proba(X_ext)[:, 1]
    if platt is not None:
        # Same monotone Platt layer the pipeline applies to every internal
        # probability. AUROC/AUPRC are unchanged; Brier, slope, intercept and
        # the reliability sheet become comparable with the internal model.
        p_ext = platt.predict_proba(p_ext.reshape(-1, 1))[:, 1]

    ext_auroc = bootstrap_metric_ci(y_ext, p_ext, "AUROC",
                                    n_boot=n_boot, seed=seed + 2)
    ext_auprc = bootstrap_metric_ci(y_ext, p_ext, "AUPRC",
                                    n_boot=n_boot, seed=seed + 3)
    ext_brier = bootstrap_metric_ci(y_ext, p_ext, "Brier",
                                    n_boot=n_boot, seed=seed + 4)
    # RUN 5 FIX: was min(n_boot, 500), while the workbook note asserted every CI
    # used 2000 resamples. The calibration SLOPE CI is precisely what the
    # arm-matched-vs-pooled conclusion rests on (NCT 0.35 [0.08-0.67] vs 0.97
    # [0.56-1.51]), so it should not be the one interval computed at a quarter
    # of the stated resolution. Use the full count; the cost is seconds.
    ext_cal = calibration_metrics(y_ext, p_ext, n_boot=n_boot)

    # ── Is the external AUROC better than chance? ────────────────────────────
    # A one-sided bootstrap tail probability against 0.5, which is the question
    # that actually matters for a validation: does the locked model carry any
    # signal into a cohort it has never seen?
    rng = np.random.default_rng(seed + 5)
    idx_pos = np.where(y_ext == 1)[0]
    idx_neg = np.where(y_ext == 0)[0]
    boots = []
    from sklearn.metrics import roc_auc_score
    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True)])
        if len(np.unique(y_ext[take])) < 2:
            continue
        boots.append(roc_auc_score(y_ext[take], p_ext[take]))
    boots = np.asarray(boots)
    p_vs_chance = (float((np.sum(boots <= 0.5) + 1) / (len(boots) + 1))
                   if len(boots) else np.nan)

    return {
        "cohort": cohort_name,
        "label": info["label"],
        "arm": info["arm"],
        # RUN 5 FIX — record what the model was ACTUALLY refit on.
        # `arm` is a property of the COHORT (which PREDIX arm it resembles) and
        # is constant. Reporting it as `matched_PREDIX_arm` was true for the
        # arm-matched runs and FALSE for the pooled run, where arm_code is None
        # and the refit uses all 185 PREDIX patients carrying RNA. The run-4
        # pooled workbook and figure both claimed arm matching that did not
        # happen — for the very rows carrying the NCT 0.679 headline.
        "refit_population": ("pooled: all PREDIX patients with this modality"
                             if arm_code is None
                             else f"PREDIX {info['arm']} arm only"),
        "arm_matched": arm_code is not None,
        "method": method,
        "model_source": ("pipeline consensus (locked)" if locked_spec
                         else "grid-searched all-shared-features"),
        "locked_from": (locked_spec["results_dir"] if locked_spec else ""),
        "classifier": clf_name,
        "best_params": str(best_params),
        "n_features": len(feats),
        "features": feats,
        "excluded": excluded,
        "renames": renames,
        "n_internal": int(len(y_int)),
        "n_events_internal": int(y_int.sum()),
        "internal_AUROC": int_auroc,
        "internal_AUPRC": int_auprc,
        # RUN 5 FIX — in LOCKED mode the internal comparator is read from the
        # consensus PKL and does NOT depend on `method`. Labelling it with the
        # external harmonisation implied it had been recomputed under each
        # scheme: the workbook printed two identical internal_AUROC values for
        # zscore and rank, the figure drew two identically-valued bars, and
        # AUROC_drop_internal_to_external in the rank rows subtracted a
        # rank-harmonised external from a z-scored internal.
        "internal_harmonisation": ("pipeline within-fold standardisation "
                                   "(z-score); not recomputed per scheme"
                                   if locked_spec is not None else method),
        "n_external": int(len(y_ext)),
        "n_events_external": int(y_ext.sum()),
        "external_AUROC": ext_auroc,
        "external_AUPRC": ext_auprc,
        "external_Brier": ext_brier,
        "external_calibration": ext_cal,
        "p_vs_chance": p_vs_chance,
        "y_ext": y_ext,
        "p_ext": p_ext,
        "internal_cv_auroc_locked": cv_score,
        "n_boot_used": int(n_boot),
    }


# =============================================================================
# SECTION 4 — REPORTING
# =============================================================================

def _fmt_ci_pair(ci, fmt="{:.2f}"):
    """Render a (low, high) pair, or say so when it is not estimable."""
    if ci is None:
        return "not reported"
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError, IndexError):
        return "not estimable"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "not estimable (recalibration fit did not converge)"
    return f"{fmt.format(lo)}–{fmt.format(hi)}"


def _resolved_n_boot(results):
    """The bootstrap count actually used, for the workbook note."""
    for r in results:
        if r is not None and r.get("n_boot_used"):
            return r["n_boot_used"]
    return N_BOOT


def build_report(results, td, fd):
    """Write the external-validation workbook and figure."""
    rows = []
    for r in results:
        if r is None:
            continue
        cal = r["external_calibration"]
        rows.append({
            "cohort": r["cohort"],
            "cohort_description": r["label"],
            "cohort_resembles_PREDIX_arm": r["arm"],
            "model_refit_population": r.get("refit_population", ""),
            "arm_matched": bool(r.get("arm_matched", True)),
            # Kept as `harmonisation` (the EXTERNAL scheme, which is what it has
            # always meant) so downstream consumers do not break; the new
            # internal_harmonisation column below removes the ambiguity.
            "harmonisation": r["method"],
            "model_source": r.get("model_source", ""),
            "locked_from": r.get("locked_from", ""),
            "classifier": r["classifier"],
            "hyperparameters": r["best_params"],
            "n_model_features": r["n_features"],
            "n_PREDIX_train": r["n_internal"],
            "events_PREDIX_train": r["n_events_internal"],
            "internal_harmonisation": r.get("internal_harmonisation", ""),
            "internal_AUROC": r["internal_AUROC"]["estimate"],
            "internal_AUROC_CI": format_ci(r["internal_AUROC"]),
            "n_external": r["n_external"],
            "events_external": r["n_events_external"],
            "external_AUROC": r["external_AUROC"]["estimate"],
            "external_AUROC_CI": format_ci(r["external_AUROC"]),
            "external_AUPRC": r["external_AUPRC"]["estimate"],
            "external_AUPRC_CI": format_ci(r["external_AUPRC"]),
            "external_event_rate": (r["n_events_external"] / r["n_external"]
                                    if r["n_external"] else np.nan),
            "external_Brier": r["external_Brier"]["estimate"],
            "external_Brier_CI": format_ci(r["external_Brier"], "{:.4f}"),
            "calibration_slope": cal["slope"],
            # RUN 5: render "not estimable" rather than "nan–nan" when the
            # recalibration fit did not converge, and carry the intercept's
            # interval, which was computed and then dropped.
            "calibration_slope_CI": _fmt_ci_pair(cal.get("slope_ci")),
            "calibration_intercept": cal["intercept"],
            "calibration_intercept_CI": _fmt_ci_pair(cal.get("intercept_ci")),
            "AUROC_drop_internal_to_external": (
                r["internal_AUROC"]["estimate"] - r["external_AUROC"]["estimate"]),
            "p_vs_chance_one_sided": r["p_vs_chance"],
        })
    df = pd.DataFrame(rows)

    # Feature provenance sheet — exactly which features transferred and why the
    # rest did not.
    # Feature provenance is identical across harmonisation schemes, so take it
    # from one scheme only. RUN 5 FIX: this was hardcoded to "zscore", so a run
    # with `--methods rank` alone produced an EMPTY provenance sheet. Use
    # whichever scheme actually ran first.
    prov = []
    _prov_method = next((r["method"] for r in results if r is not None), None)
    for r in results:
        if r is None or r["method"] != _prov_method:
            continue
        for f in r["features"]:
            prov.append({"cohort": r["cohort"], "feature": f,
                         "status": "used", "reason": ""})
        for f, reason in r["excluded"]:
            prov.append({"cohort": r["cohort"], "feature": f,
                         "status": "excluded", "reason": reason})
        for old, new in r["renames"].items():
            prov.append({"cohort": r["cohort"], "feature": new,
                         "status": "renamed",
                         "reason": f"external file spells it {old!r}"})
    df_prov = pd.DataFrame(prov)

    # Reliability data for the external cohorts.
    rel = []
    for r in results:
        if r is None:
            continue
        rr = r["external_calibration"]["reliability"].copy()
        if rr.empty:
            continue
        rr.insert(0, "harmonisation", r["method"])
        rr.insert(0, "cohort", r["cohort"])
        rel.append(rr)
    df_rel = pd.concat(rel, ignore_index=True) if rel else pd.DataFrame()

    wb = openpyxl.Workbook()
    _write_sheet(
        wb, "External_validation", df, first=True,
        note=("INDEPENDENT EXTERNAL VALIDATION OF THE TRANSCRIPTOMIC pCR MODEL.\n"
              "The model is fitted on PREDIX and then FROZEN. No external "
              "outcome and no external patient contributes to feature "
              "selection, hyper-parameter choice, coefficients or any "
              "threshold.\n"
              "One qualification, stated rather than glossed: the per-cohort "
              "standardisation (z-score or rank) IS estimated on the external "
              "cohort itself, including the patients being scored. That step "
              "is unsupervised — it never sees an outcome — but it makes the "
              "procedure transductive, so these results describe validation on "
              "an assembled cohort and not prospective scoring of a single "
              "new patient.\n"
              "model_source says which validation design produced each row: "
              "'pipeline consensus (locked)' = the main pipeline's feature-"
              "selected consensus RNA model (signature + winner classifier + "
              "modal hyperparameters from the results dir in locked_from), "
              "refit once with no grid search, with internal_AUROC = the "
              "pipeline's own frozen-consensus out-of-fold performance: the "
              "mean over cross-validation repeats of the pooled out-of-fold "
              "AUROC, with a 95% patient-level cluster-bootstrap CI (all "
              "repeats of a resampled patient move together; identical "
              "estimand to revision_performance_CI.xlsx), and external "
              "probabilities passed through the same Platt layer the "
              "pipeline applies internally. 'grid-searched all-shared-"
              "features' = this script's own model (sensitivity design), "
              "with internal_AUROC recomputed by repeated cross-validation on "
              "the same restricted feature set using one repeat-averaged "
              "out-of-fold probability per patient and a patient-level "
              "bootstrap. In both designs the internal and external "
              "estimates share one feature universe, so the internal-to-"
              "external drop reflects cohort transfer rather than differing "
              "feature availability. n_model_features is the number of "
              "features the validated model actually uses (the signature size "
              "K in locked mode; all shared features in grid mode).\n"
              + ("Each external cohort is predicted by the ARM-MATCHED PREDIX "
                 "model: I-SPY2 (trastuzumab/pertuzumab + chemotherapy) by the "
                 "DHP model, NCT02326974 (T-DM1 + pertuzumab) by the T-DM1 "
                 "model.\n"
                 if all(r.get("arm_matched", True)
                        for r in results if r is not None) else
                 "POOLED MODEL. Each external cohort is predicted by a single "
                 "model refit on ALL PREDIX patients carrying the modality, "
                 "irrespective of treatment arm — NOT by an arm-matched model. "
                 "cohort_resembles_PREDIX_arm records only which PREDIX arm "
                 "each cohort's regimen resembles; model_refit_population "
                 "records what the model was actually trained on.\n")
              +
              "Every result is reported under TWO independent harmonisation schemes "
              "(zscore and rank). The cohorts are on incompatible measurement scales, "
              "so some harmonisation is unavoidable; running both shows whether the "
              "result depends on which was chosen. A finding present under only one "
              "scheme is an artefact of that scheme.\n"
              # Interpolate the RESOLVED count, not the module constant: a run
              # with --n_boot 200 used to write "2000 resamples" into a
              # supplementary table.
              "All confidence intervals are patient-level stratified bootstrap "
              f"intervals ({_resolved_n_boot(results)} resamples); the "
              "locked-mode internal CI is the patient-level CLUSTER bootstrap "
              "described above.\n"
              "internal_harmonisation vs harmonisation: in locked mode the "
              "internal comparator is the pipeline's own cross-validated "
              "performance, standardised within training folds; it is NOT "
              "recomputed under the external harmonisation scheme, so the two "
              "internal rows for a cohort are identical by construction and "
              "AUROC_drop_internal_to_external compares an externally "
              "harmonised number against that fixed internal one.\n"
              "p_vs_chance_one_sided is the bootstrap tail probability that the "
              "external AUROC is no better than 0.5."))
    _write_sheet(
        wb, "Feature_provenance", df_prov,
        note=("Exactly which RNA features transferred to each external cohort, "
              "and the reason for every exclusion.\n"
              "Name mismatches between PREDIX and the external files are "
              "repaired explicitly and recorded here as 'renamed', but ONLY "
              "when the PREDIX target actually exists; an alias whose target "
              "was withdrawn upstream is reported in the run log and produces "
              "no row, because renaming into a column no model uses would put "
              "a feature in this table that is in no signature.\n"
              "A feature is 'used' only if it is observed in at least 80% of "
              "the external cohort and is not constant there. Missing and "
              "constant values standardise to the cohort mean (zero), so a "
              "sparsely measured feature would otherwise be counted in "
              "n_model_features while contributing nothing."))
    _write_sheet(
        wb, "External_reliability", df_rel,
        note=("Reliability-curve data for the external cohorts: equal-count bins with "
              "mean predicted risk, observed rate, and a Wilson interval on the "
              "observed rate.\n"
              "A calibration slope below 1 in external data is expected and means the "
              "locked model's probabilities are too extreme for the new cohort. "
              "Discrimination (AUROC) is unaffected by this, since it is invariant to "
              "any monotone recalibration."))
    td.mkdir(parents=True, exist_ok=True)
    path = td / f"external_validation{OUTPUT_SUFFIX}.xlsx"
    wb.save(path)
    print(f"\n  -> {path.name}")

    # ── Figure ───────────────────────────────────────────────────────────────
    valid = [r for r in results if r is not None]
    if not valid:
        return df

    methods = sorted({r["method"] for r in valid})
    cohorts = [c for c in COHORTS if any(r["cohort"] == c for r in valid)]

    fig, axes = plt.subplots(2, len(cohorts),
                             figsize=(5.6 * len(cohorts), 8.6), squeeze=False)

    for ci, cohort in enumerate(cohorts):
        # Top row: internal vs external AUROC under both harmonisations.
        ax = axes[0][ci]
        labels, vals, los, his, cols = [], [], [], [], []
        for m in methods:
            r = next((x for x in valid
                      if x["cohort"] == cohort and x["method"] == m), None)
            if r is None:
                continue
            labels.append(f"PREDIX internal\n({m})")
            vals.append(r["internal_AUROC"]["estimate"])
            los.append(r["internal_AUROC"]["ci_low"])
            his.append(r["internal_AUROC"]["ci_high"])
            cols.append(COHORT_COLOR["PREDIX (internal)"])
            labels.append(f"{cohort} external\n({m})")
            vals.append(r["external_AUROC"]["estimate"])
            los.append(r["external_AUROC"]["ci_low"])
            his.append(r["external_AUROC"]["ci_high"])
            cols.append(COHORT_COLOR.get(cohort, "#555"))

        xs = np.arange(len(vals))
        ax.bar(xs, vals, 0.62, color=cols, alpha=0.85, edgecolor="white")
        ax.errorbar(xs, vals, yerr=_asym_err(vals, los, his), fmt="none",
                    ecolor="#333", elinewidth=1.2, capsize=4)
        for x, v in zip(xs, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold")
        ax.axhline(0.5, color="#aaa", ls=":", lw=1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUROC (95% patient-level bootstrap CI)")
        r0 = next(x for x in valid if x["cohort"] == cohort)
        # RUN 5 FIX: do not assert arm matching on a pooled run.
        _scope = (f"matched to PREDIX {r0['arm']} arm"
                  if r0.get("arm_matched", True)
                  else "POOLED model (all PREDIX patients, both arms)")
        ax.set_title(f"{cohort} — {_scope}\n"
                     f"n={r0['n_external']}, {r0['n_events_external']} pCR events",
                     fontsize=9.5, color=COHORT_COLOR.get(cohort, "#333"))
        ax.grid(axis="y", alpha=0.3)

        # Bottom row: external calibration.
        ax = axes[1][ci]
        ax.plot([0, 1], [0, 1], ls=":", c="#999", lw=1.1,
                label="perfect calibration")
        for m, marker in zip(methods, ["o-", "s--"]):
            r = next((x for x in valid
                      if x["cohort"] == cohort and x["method"] == m), None)
            if r is None:
                continue
            rr = r["external_calibration"]["reliability"]
            if rr.empty:
                continue
            ax.errorbar(rr["mean_predicted"], rr["observed"],
                        yerr=_asym_err(rr["observed"], rr["obs_ci_low"],
                                       rr["obs_ci_high"]),
                        fmt=marker, ms=4.5, lw=1.3, capsize=3,
                        label=f"{m} (slope "
                              f"{r['external_calibration']['slope']:.2f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability of pCR")
        ax.set_ylabel("Observed pCR fraction")
        ax.set_title(f"{cohort} — external calibration", fontsize=9.5)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Independent external validation of the transcriptomic pCR model\n"
        "Model locked on PREDIX and applied unchanged; features standardised within "
        "each cohort under two independent schemes",
        fontsize=11, fontweight="bold", y=1.005)
    plt.tight_layout()
    _savefig(fig, fd / f"revfig06_external_validation{OUTPUT_SUFFIX}.pdf")
    return df


# =============================================================================
# SECTION 5 — CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="External validation of the PREDIX HER2 transcriptomic pCR "
                    "model in the I-SPY2 and NCT02326974 cohorts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--predix", type=Path, required=True,
                   help="PREDIX dataset (tab-separated).")
    p.add_argument("--ispy2", type=Path, default=None,
                   help="I-SPY2 RNA metrics — validates the DHP model.")
    p.add_argument("--nct", type=Path, default=None,
                   help="NCT02326974 RNA metrics — validates the T-DM1 model.")
    p.add_argument("--out_dir", type=Path, default=Path("./report"))
    p.add_argument("--output_suffix", default="",
                   help="Appended to the output basenames, e.g. '_POOLED'. Use "
                        "it whenever a second external-validation run writes "
                        "alongside the primary one, so the two are "
                        "distinguishable by filename and not only by "
                        "directory. See the OUTPUTS note at the top of this "
                        "file for why run 5 added this.")
    p.add_argument("--classifier", default="ElasticNet_LR",
                   choices=list(CANDIDATE_MODELS.keys()),
                   help="Classifier family for the locked model. ElasticNet_LR is "
                        "the default because a sparse linear model transfers across "
                        "cohorts more reliably than a tree ensemble, which can encode "
                        "cohort-specific split points.")
    p.add_argument("--methods", nargs="+", default=["zscore", "rank"],
                   choices=["zscore", "rank"],
                   help="Harmonisation schemes to run. Both by default, so the "
                        "result's dependence on the choice is visible.")
    p.add_argument("--pooled_arms", action="store_true",
                   help="Train on all PREDIX patients rather than the arm matched to "
                        "each external cohort. Off by default: the arms received "
                        "different treatments, so the arm-matched model is the correct "
                        "comparison.")
    p.add_argument("--export_shared_features_only", action="store_true",
                   help="Write shared_features_<cohort>.txt (the transferable "
                        "feature lists, one column name per line) into "
                        "out_dir/tables/revision and exit. These files are the "
                        "--include_features input for the RNA-only pipeline runs "
                        "that produce the locked consensus models.")
    p.add_argument("--locked_ispy2", type=Path, default=None,
                   help="Results dir of an RNA-only pipeline run restricted to the "
                        "I-SPY2 shared features (--modalities RNA --include_features "
                        "shared_features_I-SPY2.txt --experiments dhp). When given, "
                        "I-SPY2 is validated with the pipeline's locked consensus RNA "
                        "model (feature-selected signature + winner classifier + "
                        "modal hyperparameters) instead of this script's grid model.")
    p.add_argument("--locked_nct", type=Path, default=None,
                   help="Same for NCT02326974: results dir of the RNA-only pipeline "
                        "run restricted to its shared features (--experiments tdm1).")
    p.add_argument("--locked_experiment", choices=("arm", "global"), default="arm",
                   help="RUN 4. Which experiment inside the locked results dir to "
                        "freeze. 'arm' (default, and the behaviour of runs 1-3) "
                        "takes the arm-matched model: dhp for I-SPY2, tdm1 for "
                        "NCT02326974. 'global' takes the POOLED model instead, and "
                        "refits it on all PREDIX patients rather than on the matched "
                        "arm. The pooled model is trained on roughly twice the "
                        "patients and, unlike the T-DM1 arm model, its transcriptomic "
                        "signature retains the HER2DX HER2-amplicon score; run 3 "
                        "showed the arm-matched T-DM1 model failing to transfer to "
                        "NCT02326974 (0.572, p=0.075) while a previously published "
                        "pooled transcriptomic model reached 0.71 on the same cohort. "
                        "Report BOTH, declared in advance — never whichever transfers "
                        "better.")
    p.add_argument("--n_boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    global OUTPUT_SUFFIX
    args = parse_args()
    OUTPUT_SUFFIX = args.output_suffix
    td = args.out_dir / "tables" / "revision"
    fd = args.out_dir / "figures" / "revision"
    td.mkdir(parents=True, exist_ok=True)
    fd.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PREDIX HER2 — EXTERNAL VALIDATION (transcriptomic model)")
    print(f"  PREDIX      : {args.predix}")
    print(f"  Classifier  : {args.classifier}")
    print(f"  Harmonising : {', '.join(args.methods)}")
    print(f"  Arm matching: {'OFF (pooled)' if args.pooled_arms else 'ON'}")
    print("=" * 72)

    df_predix = pd.read_csv(args.predix, sep="\t")
    print(f"\n[LOAD] PREDIX: {df_predix.shape[0]} patients, "
          f"{df_predix.shape[1]} columns")

    sources = []
    if args.ispy2 and args.ispy2.exists():
        sources.append(("I-SPY2", pd.read_csv(args.ispy2, sep="\t")))
    if args.nct and args.nct.exists():
        sources.append(("NCT02326974", pd.read_csv(args.nct, sep="\t")))
    if not sources:
        raise FileNotFoundError(
            "No external cohort supplied. Pass --ispy2 and/or --nct.")

    for name, d in sources:
        print(f"[LOAD] {name}: {d.shape[0]} patients, {d.shape[1]} columns, "
              f"{int(d['pCR'].sum())} pCR events "
              f"({d['pCR'].mean() * 100:.1f}%)")

    # ── Always export the transferable feature lists ─────────────────────────
    # One file per cohort; consumed by the RNA-only pipeline runs via
    # --include_features so the pipeline's candidate pool is exactly the
    # externally measurable universe.
    for name, df_ext in sources:
        dfx, _ = harmonise_columns(df_ext, name, df_predix.columns)
        feats, excl = shared_features(df_predix, dfx, name)
        out = td / f"shared_features_{name}.txt"
        # newline="\n": byte-identical output on Windows and Linux, so a
        # feature list re-exported on one machine hashes the same as the one
        # the RNA-only pipeline runs consumed on the other.
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# Transferable RNA features PREDIX <-> {name} "
                    f"({len(feats)} kept, {len(excl)} excluded)\n")
            for c in feats:
                f.write(c + "\n")
        print(f"[EXPORT] {out} ({len(feats)} features)")
    if args.export_shared_features_only:
        print("[EXPORT] --export_shared_features_only: done, exiting before "
              "any modelling.")
        return None

    # ── Locked pipeline-consensus models, where supplied ─────────────────────
    locked_dirs = {"I-SPY2": args.locked_ispy2, "NCT02326974": args.locked_nct}
    # RUN 4: --locked_experiment global freezes the POOLED model instead of the
    # arm-matched one, and necessarily refits it on all PREDIX patients — that
    # is what makes it the pooled model. The old incompatibility below therefore
    # applies only to the arm-matched case.
    pooled_locked = (args.locked_experiment == "global")
    if args.pooled_arms and any(locked_dirs.values()) and not pooled_locked:
        raise SystemExit("--pooled_arms is incompatible with --locked_* unless "
                         "--locked_experiment global is given: the arm-matched "
                         "locked models are arm-specific by construction.")
    locked = {}
    for name, _ in sources:
        if locked_dirs.get(name):
            exp = ("global" if pooled_locked
                   else ("dhp" if COHORTS[name]["arm_code"] == 0 else "tdm1"))
            locked[name] = load_locked_consensus(
                locked_dirs[name], exp, args.predix)

    results = []
    for name, df_ext in sources:
        arm_code = (None if (args.pooled_arms or pooled_locked)
                    else COHORTS[name]["arm_code"])
        for method in args.methods:
            results.append(validate_cohort(
                df_predix, df_ext, name, method, args.classifier,
                arm_code=arm_code, seed=args.seed, n_boot=args.n_boot,
                locked_spec=locked.get(name)))

    df = build_report(results, td, fd)

    print("\n" + "=" * 72)
    print("EXTERNAL VALIDATION SUMMARY  (AUROC [95% CI])")
    if locked:
        print("  internal (locked rows) = mean over CV repeats of the pooled "
              "out-of-fold AUROC;\n"
              "                           95% patient-level cluster-bootstrap CI")
    if len(locked) < len(sources):
        print("  internal (grid rows)   = AUROC of one repeat-averaged OOF "
              "probability per patient;\n"
              "                           95% patient-level bootstrap CI "
              "(sensitivity design)")
    print("=" * 72)
    for r in results:
        if r is None:
            continue
        print(f"  {r['cohort']:<14} {r['method']:<7} "
              f"internal {format_ci(r['internal_AUROC'])}  ->  "
              f"external {format_ci(r['external_AUROC'])}   "
              f"(n={r['n_external']}, {r['n_events_external']} events, "
              f"{r['n_features']} features)")
    print("=" * 72)
    return df


if __name__ == "__main__":
    main()
