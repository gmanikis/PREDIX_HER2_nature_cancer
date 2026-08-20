"""
cv_estimands.py — repeat-aware, patient-clustered performance estimands for
repeated K-fold cross-validation.

WHY THIS MODULE EXISTS
======================
The revision originally summarised each patient's R out-of-fold predictions
(one per repeat) by their MEAN and then computed AUROC / AUPRC / Brier /
calibration on those n averaged values, bootstrapping patients. That collapse
looked innocent but is not a clean estimand of a single model's performance:

  * It is a 200-model ENSEMBLE score, so it is mildly optimistic for models
    with signal (bagging over repeats removes fit noise).
  * It is severely PESSIMISTIC for models with little within-stratum signal.
    Every fold's intercept (and Platt map) is fitted on the outer training
    patients; when a pCR patient is held out the training set holds one fewer
    event, so that patient's own prediction is shifted down by a hair, and a
    non-pCR held-out patient's prediction is shifted up. Averaged over 200
    repeats the noise vanishes and only that systematic shift remains. For a
    clinical model that is essentially an ER-status indicator, this produced a
    patient-level AUROC of 0.41 (within each ER stratum every pCR patient
    ranked BELOW every non-pCR patient) while every fold-level estimate sat
    at 0.61-0.64. The point estimate fell outside its own bootstrap interval.

THE ESTIMAND USED HERE
======================
In one repeat of stratified K-fold CV every patient receives exactly ONE
out-of-fold prediction. The performance of that repeat is the metric computed
on that complete out-of-fold vector — the classical cross-validated estimate.
With R repeats we report

    theta_hat = (1/R) * sum_r  metric( y , p_r )

i.e. the cross-validated metric averaged over R random partitions. This is a
single-model estimand (no averaging of probabilities across models), it is
what the Methods text describes ("performance pooled across the outer
evaluations"), and stratified folds keep the fold-intercept effect from
biasing it.

UNCERTAINTY: patient-level CLUSTER bootstrap. Patients are resampled with
replacement (stratified by outcome so every resample keeps the observed event
count) and ALL R predictions of a resampled patient travel together. For each
resample the estimand is recomputed on the (R x n) matrix of the resampled
patients. Paired comparisons apply the SAME patient resample to both models.
This makes the interval reflect n independent patients, exactly as the
reviewers asked, without collapsing predictions into an ensemble.

DeLong's test needs one prediction per patient; it is therefore run once per
repeat and summarised across repeats (median p, fraction < 0.05). The verdict
on every comparison is taken from the paired cluster bootstrap, never from
DeLong.

Everything is vectorised over repeats: AUROC via midranks (identical to
sklearn.metrics.roc_auc_score, ties included), AUPRC via the step-wise
precision-recall sum over distinct thresholds (identical to
sklearn.metrics.average_precision_score), calibration slope/intercept via a
vectorised Newton fit of logit(P) = a + b*logit(p). Tests in
tests/test_statistics.py assert equality with sklearn and reproduce the
held-out-outcome artefact synthetically.

Public API
----------
build_repeat_matrix(folds, get_pred)          -> RepeatMatrix
consensus_repeat_matrix(cons_blob, model)     -> RepeatMatrix
discovery_repeat_matrix(folds)                -> RepeatMatrix
metric_rows(P, y, metric)                     -> (R,) per-repeat metric
repeat_mean_metric(P, y, metric)              -> float
bootstrap_repeat_metric_ci(P, y, metric, ...) -> dict
paired_bootstrap_repeat_delta(P1, P2, y, ...) -> dict
per_repeat_test_summary(P1, P2, y, test_fn)   -> dict
mean_fold_metric(folds, get_pred, metric)     -> float   (legacy estimand)
calibration_repeat_summary(P, y, ...)         -> dict    (slope/intercept/Brier/ECE + CIs)
reliability_pooled(P, y, pids, ...)           -> DataFrame
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata

FUSED_DEFAULT = "Fused_ElasticNet"


# =============================================================================
# 1. Building the (repeats x patients) matrix from fold dicts
# =============================================================================

@dataclass
class RepeatMatrix:
    """Out-of-fold predictions arranged as repeats x patients."""
    pids: np.ndarray            # (n,) patient identifiers, sorted ascending
    y: np.ndarray               # (n,) outcome per patient
    P: np.ndarray               # (R, n) prediction; NaN where none available
    n_folds: int
    folds_per_repeat: list = field(default_factory=list)
    incomplete_repeats: int = 0
    pid_key: str = "test_pids"

    @property
    def n_repeats(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_patients(self) -> int:
        return int(self.P.shape[1])

    @property
    def n_events(self) -> int:
        return int(np.nansum(self.y))

    def predictions_per_patient(self) -> np.ndarray:
        return np.isfinite(self.P).sum(axis=0)


def _resolve_pid_key(folds, pid_key=None):
    for candidate in (pid_key, "test_pids", "test_idx"):
        if candidate and candidate in folds[0]:
            if candidate == "test_idx":
                warnings.warn(
                    "Fold dicts carry only positional 'test_idx'; using it as "
                    "the patient identifier. Re-run the pipeline to record "
                    "test_pids.", RuntimeWarning)
            return candidate
    raise KeyError(
        "Fold dicts carry no patient identifier ('test_pids' or 'test_idx'). "
        "Repeat-level pooling is impossible without one.")


def build_repeat_matrix(folds, get_pred, pid_key=None, label_key="y_test",
                        strict=True) -> RepeatMatrix:
    """
    Arrange out-of-fold predictions as an (R x n) matrix.

    Repeat membership is inferred by walking the folds in fold_idx order: a
    fold opens a new repeat as soon as it would predict a patient already
    predicted in the current repeat. For RepeatedStratifiedKFold (folds
    enumerated repeat by repeat) this recovers the repeats exactly. Every
    repeat is then checked for completeness (each patient predicted exactly
    once); an incomplete trailing repeat (aborted run) is kept with NaN and
    counted in `incomplete_repeats`, but if `strict` any repeat that predicts
    a patient twice raises — that would mean the folds are not from a
    repeated K-fold design and the estimand does not apply.
    """
    if not folds:
        return RepeatMatrix(np.array([], dtype=np.int64), np.array([]),
                            np.empty((0, 0)), 0)
    key = _resolve_pid_key(folds, pid_key)

    order = sorted(range(len(folds)),
                   key=lambda i: (folds[i].get("fold_idx", i), i))

    # Pass 1: patient universe and labels
    labels = {}
    for i in order:
        fd = folds[i]
        pids = np.asarray(fd[key]).ravel()
        ys = np.asarray(fd[label_key], dtype=float).ravel()
        if len(pids) != len(ys):
            raise ValueError(f"Fold {fd.get('fold_idx', i)}: {len(pids)} ids "
                             f"vs {len(ys)} labels.")
        for pid, yv in zip(pids, ys):
            pid = int(pid)
            if pid in labels and labels[pid] != float(yv):
                raise ValueError(f"Patient {pid} carries different labels in "
                                 f"different folds ({labels[pid]} vs {yv}).")
            labels[pid] = float(yv)
    pids_sorted = np.array(sorted(labels), dtype=np.int64)
    col = {int(p): j for j, p in enumerate(pids_sorted)}
    n = len(pids_sorted)

    # Pass 2: greedy repeat assignment
    rep_of_fold = {}
    rep, seen = 0, set()
    for i in order:
        pids = set(int(v) for v in np.asarray(folds[i][key]).ravel())
        if seen & pids:
            rep += 1
            seen = set()
        seen |= pids
        rep_of_fold[i] = rep
    R = rep + 1

    P = np.full((R, n), np.nan, dtype=float)
    folds_per_repeat = [0] * R
    for i in order:
        fd = folds[i]
        r = rep_of_fold[i]
        folds_per_repeat[r] += 1
        pids = np.asarray(fd[key]).ravel()
        preds = np.asarray(get_pred(fd), dtype=float).ravel()
        if len(preds) != len(pids):
            raise ValueError(f"Fold {fd.get('fold_idx', i)}: {len(pids)} ids "
                             f"vs {len(preds)} predictions.")
        for pid, pr in zip(pids, preds):
            j = col[int(pid)]
            if strict and np.isfinite(P[r, j]):
                raise ValueError(
                    f"Patient {int(pid)} predicted twice within repeat {r} "
                    f"(fold {fd.get('fold_idx', i)}). The folds are not a "
                    f"repeated K-fold design.")
            P[r, j] = pr

    complete = np.isfinite(P).all(axis=1)
    incomplete = int((~complete).sum())
    if incomplete:
        warnings.warn(f"{incomplete} of {R} repeats do not predict every "
                      f"patient (NaN kept; metrics use available entries).",
                      RuntimeWarning)
    return RepeatMatrix(pids=pids_sorted,
                        y=np.array([labels[int(p)] for p in pids_sorted]),
                        P=P, n_folds=len(folds), folds_per_repeat=folds_per_repeat,
                        incomplete_repeats=incomplete, pid_key=key)


def consensus_repeat_matrix(cons_blob, model, fused_name=FUSED_DEFAULT,
                            pid_key=None) -> RepeatMatrix:
    """RepeatMatrix for one model of a consensus-evaluation PKL."""
    folds = cons_blob.get("folds", []) if isinstance(cons_blob, dict) else []
    if not folds:
        return build_repeat_matrix([], lambda f: f)
    if model == fused_name:
        return build_repeat_matrix(folds, lambda f: f["fused_y_pred"], pid_key)
    return build_repeat_matrix(folds, lambda f: f["unimodal_y_pred"][model],
                               pid_key)


def discovery_repeat_matrix(folds, pred_key="y_pred", pid_key=None) -> RepeatMatrix:
    """RepeatMatrix for a list of discovery fold dicts (y_pred per fold)."""
    return build_repeat_matrix(folds, lambda f: f[pred_key], pid_key)


# =============================================================================
# 2. Vectorised per-repeat metrics
# =============================================================================

def _as_matrix(P):
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[None, :]
    return P


def auroc_rows(P, y):
    """
    AUROC of every row of P against y (midrank Mann-Whitney form; equals
    sklearn.metrics.roc_auc_score for every row, ties included). NaN entries
    are ignored per row; a row without both classes returns NaN.
    """
    P = _as_matrix(P)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(P)
    pos = (y == 1)[None, :] & valid
    neg = (y == 0)[None, :] & valid
    n1 = pos.sum(axis=1).astype(float)
    n0 = neg.sum(axis=1).astype(float)
    if valid.all():
        ranks = rankdata(P, axis=1)
    else:
        ranks = rankdata(P, axis=1, nan_policy="omit")
        ranks = np.where(valid, ranks, 0.0)
    S = (ranks * pos).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (S - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    auc[(n1 < 1) | (n0 < 1)] = np.nan
    return auc


def auprc_rows(P, y):
    """
    Average precision of every row of P against y, identical to
    sklearn.metrics.average_precision_score (step-wise sum over DISTINCT
    thresholds, so tied scores enter together). NaN entries ignored.
    """
    P = _as_matrix(P)
    y = np.asarray(y, dtype=float)
    R, n = P.shape
    valid = np.isfinite(P)
    # NaN sorts last with argsort on -P (−NaN is NaN → last).
    order = np.argsort(-P, axis=1, kind="stable")
    ps = np.take_along_axis(P, order, axis=1)
    vs = np.take_along_axis(valid, order, axis=1)
    ys = np.take_along_axis(np.broadcast_to(y, (R, n)), order, axis=1)
    ys = np.where(vs, ys, 0.0)
    npos = ys.sum(axis=1)
    tp = np.cumsum(ys, axis=1)
    fp = np.cumsum(np.where(vs, 1.0 - ys, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = tp / (tp + fp)
    prec = np.where(np.isfinite(prec), prec, 0.0)
    # group ends: last position of each run of tied scores (NaN != NaN → True)
    ends = np.ones((R, n), dtype=bool)
    ends[:, :-1] = ps[:, :-1] != ps[:, 1:]
    idx = np.broadcast_to(np.arange(n), (R, n))
    end_idx = np.where(ends, idx, n)
    next_end = np.minimum.accumulate(end_idx[:, ::-1], axis=1)[:, ::-1]
    next_end = np.minimum(next_end, n - 1)
    prec_at_end = np.take_along_axis(prec, next_end, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (ys * prec_at_end).sum(axis=1) / npos
    ap[npos < 1] = np.nan
    ap[(vs.sum(axis=1) - npos) < 1] = np.nan
    return ap


def brier_rows(P, y):
    P = _as_matrix(P)
    y = np.asarray(y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean((np.clip(P, 0.0, 1.0) - y[None, :]) ** 2, axis=1)


METRIC_ROW_FNS = {"AUROC": auroc_rows, "AUPRC": auprc_rows, "Brier": brier_rows}


def metric_rows(P, y, metric="AUROC"):
    return METRIC_ROW_FNS[metric](P, y)


def repeat_mean_metric(P, y, metric="AUROC"):
    """Point estimate: metric per repeat, averaged over repeats."""
    vals = metric_rows(P, y, metric)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


# =============================================================================
# 3. Patient-level cluster bootstrap
# =============================================================================

def _stratified_take(rng, y, stratified=True):
    y = np.asarray(y)
    n = len(y)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if stratified and len(idx_pos) > 0 and len(idx_neg) > 0:
        return np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True)])
    return rng.integers(0, n, size=n)


def bootstrap_repeat_metric_ci(P, y, metric="AUROC", n_boot=2000, seed=0,
                               ci=0.95, stratified=True):
    """
    Point estimate = mean over repeats of the per-repeat metric.
    CI = patient-level cluster bootstrap (all repeats of a patient move
    together; stratified by outcome).
    """
    P = _as_matrix(P)
    y = np.asarray(y, dtype=float)
    R, n = P.shape
    out = {"metric": metric, "estimate": np.nan, "ci_low": np.nan,
           "ci_high": np.nan, "se": np.nan, "n": int(n),
           "n_events": int(np.nansum(y)), "n_repeats": int(R),
           "n_boot_valid": 0}
    if n == 0 or R == 0:
        return out
    out["estimate"] = repeat_mean_metric(P, y, metric)
    if not np.isfinite(out["estimate"]):
        return out
    rng = np.random.default_rng(seed)
    fn = METRIC_ROW_FNS[metric]
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        take = _stratified_take(rng, y, stratified)
        vals = fn(P[:, take], y[take])
        vals = vals[np.isfinite(vals)]
        boots[b] = vals.mean() if len(vals) else np.nan
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return out
    alpha = (1.0 - ci) / 2.0
    out["ci_low"] = float(np.percentile(boots, alpha * 100))
    out["ci_high"] = float(np.percentile(boots, (1 - alpha) * 100))
    out["se"] = float(np.std(boots, ddof=1)) if len(boots) > 1 else np.nan
    out["n_boot_valid"] = int(len(boots))
    return out


def paired_bootstrap_repeat_delta(P1, P2, y, metric="AUROC", n_boot=2000,
                                  seed=0, stratified=True):
    """
    Paired patient-level cluster bootstrap of  theta(P1) - theta(P2).
    The SAME patient resample is applied to both models and to every repeat.
    Two-sided p = twice the smaller tail proportion of resampled deltas on
    the wrong side of zero (floored at 1/n_boot).
    """
    P1 = _as_matrix(P1); P2 = _as_matrix(P2)
    y = np.asarray(y, dtype=float)
    if P1.shape != P2.shape:
        raise ValueError(f"Shape mismatch {P1.shape} vs {P2.shape}: paired "
                         f"comparison needs identical repeats x patients.")
    R, n = P1.shape
    out = {"metric": metric, "delta": np.nan, "ci_low": np.nan,
           "ci_high": np.nan, "p_value": np.nan, "n": int(n),
           "n_events": int(np.nansum(y)), "n_repeats": int(R),
           "estimate_1": np.nan, "estimate_2": np.nan}
    if n == 0 or R == 0 or len(np.unique(y)) < 2:
        return out
    fn = METRIC_ROW_FNS[metric]

    def _theta(A, yy):
        v = fn(A, yy)
        v = v[np.isfinite(v)]
        return v.mean() if len(v) else np.nan

    e1, e2 = _theta(P1, y), _theta(P2, y)
    out["estimate_1"], out["estimate_2"] = float(e1), float(e2)
    out["delta"] = float(e1 - e2)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        take = _stratified_take(rng, y, stratified)
        deltas[b] = _theta(P1[:, take], y[take]) - _theta(P2[:, take], y[take])
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return out
    out["ci_low"] = float(np.percentile(deltas, 2.5))
    out["ci_high"] = float(np.percentile(deltas, 97.5))
    prop_le = float(np.mean(deltas <= 0))
    prop_ge = float(np.mean(deltas >= 0))
    p = 2 * min(prop_le, prop_ge)
    out["p_value"] = float(min(1.0, max(p, 1.0 / len(deltas))))
    return out


def per_repeat_test_summary(P1, P2, y, test_fn):
    """
    Run a paired two-model test (signature test_fn(y, p1, p2) -> dict with
    'p_value') once per repeat and summarise across repeats.
    """
    P1 = _as_matrix(P1); P2 = _as_matrix(P2)
    y = np.asarray(y, dtype=float)
    ps = []
    for r in range(P1.shape[0]):
        m = np.isfinite(P1[r]) & np.isfinite(P2[r])
        if m.sum() < 5 or len(np.unique(y[m])) < 2:
            continue
        res = test_fn(y[m], P1[r, m], P2[r, m])
        pv = res.get("p_value", np.nan) if isinstance(res, dict) else res
        if pv is not None and np.isfinite(pv):
            ps.append(float(pv))
    ps = np.asarray(ps, dtype=float)
    out = {"n_repeats_tested": int(len(ps)), "p_median": np.nan,
           "p_q25": np.nan, "p_q75": np.nan, "frac_p_below_0.05": np.nan}
    if len(ps):
        out["p_median"] = float(np.median(ps))
        out["p_q25"] = float(np.percentile(ps, 25))
        out["p_q75"] = float(np.percentile(ps, 75))
        out["frac_p_below_0.05"] = float(np.mean(ps < 0.05))
    return out


# =============================================================================
# 4. Legacy estimand (mean of per-fold metrics), for continuity columns
# =============================================================================

def mean_fold_metric(folds, get_pred, metric="AUROC", label_key="y_test"):
    """Mean over outer folds of the metric computed inside each test fold."""
    fn = METRIC_ROW_FNS[metric]
    vals = []
    for fd in folds:
        p = np.asarray(get_pred(fd), dtype=float).ravel()
        yy = np.asarray(fd[label_key], dtype=float).ravel()
        v = fn(p[None, :], yy)[0]
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else np.nan


# =============================================================================
# 5. Calibration, per repeat
# =============================================================================

def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# =============================================================================
# SHARED BOOTSTRAP SETTINGS — one definition, imported by every consumer.
# =============================================================================
# generate_report.py and revision_analyses.py both derive their bootstrap
# settings from here. Before run 5 each carried its own copy: generate_report
# hard-coded 2000/20240517 while revision_analyses rebound both from the CLI,
# so a single --n_boot or --seed made two workbooks disagree about the same
# quantity with nothing to explain the difference.
DEFAULT_N_BOOT = 2000
DEFAULT_BOOT_SEED = 20240517


def shared_seed(tag, base=DEFAULT_BOOT_SEED):
    """Deterministic per-cell bootstrap seed.

    crc32, not hash(): Python salts string hashing per process, so a
    hash()-derived seed changes between runs and every regeneration would shift
    the CI endpoints.

    The offset uses the FULL crc32 range. It was `% 10000`, which compressed
    the ~126 tags these scripts generate into 10,000 slots — a birthday
    expectation of ~0.8 collisions, and in practice exactly 3 pairs of analyses
    silently shared a resample stream. Each CI was still individually valid;
    only their Monte-Carlo errors were correlated, undocumented. Widening the
    range costs nothing and meets the stated design intent.
    """
    import zlib
    return int(base) + zlib.crc32(str(tag).encode()) % (2 ** 31)


def logistic_recalibration_rows(LP, y, max_iter=50, tol=1e-8):
    """
    Vectorised Newton fit of  logit P(y=1) = a + b * lp  for every row of LP.
    Returns (slope b, intercept a) arrays of shape (R,). NaN entries of LP are
    ignored. Rows that fail to converge (or separate) return NaN.
    """
    LP = _as_matrix(LP)
    y = np.asarray(y, dtype=float)
    R, n = LP.shape
    valid = np.isfinite(LP)
    X = np.where(valid, LP, 0.0)
    W_valid = valid.astype(float)
    a = np.zeros(R)
    b = np.ones(R)
    ok = np.ones(R, dtype=bool)
    ok &= (valid & (y == 1)[None, :]).sum(axis=1) >= 1
    ok &= (valid & (y == 0)[None, :]).sum(axis=1) >= 1
    for _ in range(max_iter):
        eta = a[:, None] + b[:, None] * X
        eta = np.clip(eta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu) * W_valid
        resid = (y[None, :] - mu) * W_valid
        g_a = resid.sum(axis=1)
        g_b = (resid * X).sum(axis=1)
        h_aa = w.sum(axis=1)
        h_ab = (w * X).sum(axis=1)
        h_bb = (w * X * X).sum(axis=1)
        det = h_aa * h_bb - h_ab * h_ab
        with np.errstate(invalid="ignore", divide="ignore"):
            da = (h_bb * g_a - h_ab * g_b) / det
            db = (-h_ab * g_a + h_aa * g_b) / det
        bad = ~np.isfinite(da) | ~np.isfinite(db) | (det <= 1e-12)
        da = np.where(bad, 0.0, da)
        db = np.where(bad, 0.0, db)
        # step-halving guard against overshoot
        step = np.maximum(1.0, np.maximum(np.abs(da), np.abs(db)) / 5.0)
        a = a + da / step
        b = b + db / step
        ok &= ~bad
        if np.all(np.abs(da) < tol) and np.all(np.abs(db) < tol):
            break
    # RUN 5: record whether the Newton loop actually CONVERGED, rather than
    # only whether it stayed finite. A quasi-separated row can drift for all 50
    # damped iterations, end at b = 30, pass the |b| < 50 magnitude cap and be
    # accepted as a valid slope — inflating both the mean slope and its CI. The
    # convergence flag is returned so callers can report how many repeats and
    # bootstrap replicates were actually usable, which was previously computed
    # and then discarded.
    converged = (np.abs(da) < np.sqrt(tol)) & (np.abs(db) < np.sqrt(tol))
    ok &= converged
    # separation / divergence guard
    ok &= np.isfinite(a) & np.isfinite(b) & (np.abs(b) < 50) & (np.abs(a) < 50)
    slope = np.where(ok, b, np.nan)
    intercept = np.where(ok, a, np.nan)
    return slope, intercept


def calibration_in_the_large_rows(LP, y, max_iter=50, tol=1e-8):
    """
    Calibration-in-the-large: the intercept of  logit P(y=1) = a + 1*lp,
    i.e. the SLOPE IS FIXED AT 1 and lp enters as an offset. Returns an array
    of shape (R,).

    This is the quantity that answers "does the model's average predicted risk
    match the observed event rate", and it is NOT the intercept of the
    two-parameter recalibration fit above. Those two coincide only when the
    fitted slope happens to be 1; away from that they diverge by up to ~1.3
    log-odds on realistic data, and the two-parameter intercept was being
    labelled as calibration-in-the-large throughout the calibration section.
    One free parameter, so plain Newton with no step-halving is stable.
    """
    LP = _as_matrix(LP)
    y = np.asarray(y, dtype=float)
    R, _ = LP.shape
    valid = np.isfinite(LP)
    X = np.where(valid, LP, 0.0)
    W = valid.astype(float)
    a = np.zeros(R)
    ok = ((valid & (y == 1)[None, :]).sum(axis=1) >= 1) & \
         ((valid & (y == 0)[None, :]).sum(axis=1) >= 1)
    for _ in range(max_iter):
        eta = np.clip(a[:, None] + X, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        g = ((y[None, :] - mu) * W).sum(axis=1)
        h = (mu * (1 - mu) * W).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = g / h
        step = np.where(np.isfinite(step) & (h > 1e-12), step, 0.0)
        a = a + step
        if np.all(np.abs(step) < tol):
            break
    ok &= np.isfinite(a) & (np.abs(a) < 50)
    return np.where(ok, a, np.nan)


def ece_rows(P, y, n_bins=10):
    """Expected calibration error per row on equal-count bins (loop over rows)."""
    P = _as_matrix(P)
    y = np.asarray(y, dtype=float)
    out = np.full(P.shape[0], np.nan)
    for r in range(P.shape[0]):
        m = np.isfinite(P[r])
        p, yy = P[r, m], y[m]
        if len(p) < 4:
            continue
        nb = int(min(n_bins, max(2, len(p) // 8)))
        order = np.argsort(p, kind="stable")
        bin_of = np.empty(len(p), dtype=int)
        bin_of[order] = (np.arange(len(p)) * nb) // len(p)
        gap = 0.0
        for bb in range(nb):
            mm = bin_of == bb
            if mm.any():
                gap += mm.sum() * abs(p[mm].mean() - yy[mm].mean())
        out[r] = gap / len(p)
    return out


def calibration_repeat_summary(P, y, n_boot=2000, seed=0, n_bins=10,
                               stratified=True):
    """
    Calibration of a model under repeated CV: slope, intercept, Brier and
    ECE computed per repeat and averaged; slope/intercept/Brier CIs from the
    patient-level cluster bootstrap. Point estimates and CIs are therefore
    single-model quantities, not properties of an averaged ensemble.
    """
    P = _as_matrix(P)
    y = np.asarray(y, dtype=float)
    R, n = P.shape
    out = {"n": int(n), "n_events": int(np.nansum(y)), "n_repeats": int(R),
           "slope": np.nan, "intercept": np.nan,
           "slope_ci": (np.nan, np.nan), "intercept_ci": (np.nan, np.nan),
           "brier": np.nan, "brier_ci": (np.nan, np.nan),
           "ece": np.nan, "observed_rate": np.nan, "mean_predicted": np.nan,
           "n_repeats_slope_valid": 0,
           # RUN 5 additions: the quantity that actually is
           # calibration-in-the-large, and the convergence bookkeeping that was
           # previously computed and then discarded.
           "citl": np.nan, "citl_ci": (np.nan, np.nan),
           "n_boot_slope_valid": 0, "n_boot_requested": int(n_boot)}
    if n < 10 or len(np.unique(y)) < 2 or R == 0:
        return out
    LP = _logit(P)
    LP = np.where(np.isfinite(P), LP, np.nan)
    s, a = logistic_recalibration_rows(LP, y)
    ok = np.isfinite(s)
    out["n_repeats_slope_valid"] = int(ok.sum())
    if ok.any():
        out["slope"] = float(np.mean(s[ok]))
        out["intercept"] = float(np.mean(a[ok]))
    citl = calibration_in_the_large_rows(LP, y)
    if np.isfinite(citl).any():
        out["citl"] = float(np.nanmean(citl))
    br = brier_rows(P, y)
    out["brier"] = float(np.nanmean(br))
    ec = ece_rows(P, y, n_bins)
    out["ece"] = float(np.nanmean(ec)) if np.isfinite(ec).any() else np.nan
    out["observed_rate"] = float(np.mean(y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["mean_predicted"] = float(np.nanmean(P))

    rng = np.random.default_rng(seed)
    sl_b, ic_b, br_b, citl_b = [], [], [], []
    for _ in range(n_boot):
        take = _stratified_take(rng, y, stratified)
        s_b, a_b = logistic_recalibration_rows(LP[:, take], y[take])
        okb = np.isfinite(s_b)
        # NOTE (run 5): a replicate contributes only if at least one repeat
        # converged, and within a replicate the mean is over the converged
        # repeats only. Replicates where many repeats separate — the ones with
        # the most extreme slopes — are therefore thinned or dropped, so this
        # interval is CONDITIONAL ON CONVERGENCE and is narrower than the
        # unconditional one. That was true before and is unchanged; what is new
        # is that we now count and report how much conditioning occurred, so a
        # slope resting on a small surviving fraction is visible rather than
        # indistinguishable from one resting on all 2000.
        if okb.any():
            sl_b.append(float(np.mean(s_b[okb])))
            ic_b.append(float(np.mean(a_b[okb])))
        c_b = calibration_in_the_large_rows(LP[:, take], y[take])
        if np.isfinite(c_b).any():
            citl_b.append(float(np.nanmean(c_b)))
        br_b.append(float(np.nanmean(brier_rows(P[:, take], y[take]))))
    out["n_boot_slope_valid"] = int(len(sl_b))

    def _pct(arr):
        arr = np.asarray([v for v in arr if np.isfinite(v)], dtype=float)
        if len(arr) == 0:
            return (np.nan, np.nan)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    out["slope_ci"] = _pct(sl_b)
    out["intercept_ci"] = _pct(ic_b)
    out["citl_ci"] = _pct(citl_b)
    out["brier_ci"] = _pct(br_b)
    return out


def reliability_pooled(P, y, n_bins=10, n_boot=2000, seed=0, stratified=True):
    """
    Reliability-curve data from ALL (repeat, patient) out-of-fold predictions
    pooled: equal-count bins on the pooled predictions, mean predicted risk and
    observed pCR fraction per bin. Bin uncertainty is a patient-level cluster
    bootstrap of the observed fraction (bins re-derived per resample), which
    respects that each patient contributes R correlated rows.
    """
    P = _as_matrix(P)
    y = np.asarray(y, dtype=float)
    R, n = P.shape
    if n == 0 or R == 0:
        return pd.DataFrame()

    def _bins(A, yy, nb):
        Y = np.broadcast_to(yy, A.shape)
        m = np.isfinite(A)
        p = A[m]; yv = Y[m]
        pat = np.broadcast_to(np.arange(A.shape[1]), A.shape)[m]
        order = np.argsort(p, kind="stable")
        bin_of = np.empty(len(p), dtype=int)
        bin_of[order] = (np.arange(len(p)) * nb) // len(p)
        rows = []
        for bb in range(nb):
            mm = bin_of == bb
            if not mm.any():
                rows.append((np.nan, np.nan, 0, 0, 0))
                continue
            rows.append((float(p[mm].mean()), float(yv[mm].mean()), int(mm.sum()),
                         int(yv[mm].sum()), int(len(np.unique(pat[mm])))))
        return rows

    nb = int(min(n_bins, max(2, n // 8)))
    base = _bins(P, y, nb)
    rng = np.random.default_rng(seed)
    obs_b = np.full((n_boot, nb), np.nan)
    for b in range(n_boot):
        take = _stratified_take(rng, y, stratified)
        rows = _bins(P[:, take], y[take], nb)
        obs_b[b] = [r[1] for r in rows]
    recs = []
    for bb, (mp, ob, cnt, ev, npat) in enumerate(base):
        col = obs_b[:, bb]
        col = col[np.isfinite(col)]
        lo, hi = ((float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5)))
                  if len(col) else (np.nan, np.nan))
        recs.append({"bin": bb + 1, "n_rows": cnt, "n_patients_distinct": npat,
                     "n_events_rows": ev, "mean_predicted": mp, "observed": ob,
                     "obs_ci_low": lo, "obs_ci_high": hi})
    return pd.DataFrame(recs)


# =============================================================================
# 6. Threshold-dependent operating points (mean over folds, cluster bootstrap)
# =============================================================================

def bootstrap_fold_operating_point_ci(folds, get_pred, get_threshold,
                                      pid_key=None, label_key="y_test",
                                      n_boot=2000, seed=0, stratified=True):
    """
    Sensitivity and specificity averaged over outer folds, each fold at ITS
    OWN stored decision threshold, with a patient-level cluster bootstrap CI.

    Per resample the multiplicity of every patient is applied as a weight to
    that patient's rows in every fold's test set; each fold's weighted
    sensitivity/specificity is recomputed and averaged over folds. Fully
    vectorised over folds (one bincount per resample).

    Returns {"Sensitivity": {...}, "Specificity": {...}} with estimate,
    ci_low, ci_high, n, n_events, n_folds.
    """
    empty = {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
             "n": 0, "n_events": 0, "n_folds": len(folds)}
    if not folds:
        return {"Sensitivity": dict(empty), "Specificity": dict(empty)}
    key = _resolve_pid_key(folds, pid_key)
    labels = {}
    fold_id, col_pid, yy, ind = [], [], [], []
    for i, fd in enumerate(folds):
        pids = np.asarray(fd[key]).ravel()
        ys = np.asarray(fd[label_key], dtype=float).ravel()
        p = np.asarray(get_pred(fd), dtype=float).ravel()
        thr = get_threshold(fd)
        if thr is None or not np.isfinite(thr):
            continue
        m = np.isfinite(p)
        for pid, yv in zip(pids, ys):
            labels[int(pid)] = float(yv)
        fold_id.append(np.full(int(m.sum()), i))
        col_pid.append(pids[m].astype(np.int64))
        yy.append(ys[m])
        ind.append((p[m] >= thr).astype(float))
    if not fold_id:
        return {"Sensitivity": dict(empty), "Specificity": dict(empty)}
    fold_id = np.concatenate(fold_id)
    col_pid = np.concatenate(col_pid)
    yy = np.concatenate(yy)
    ind = np.concatenate(ind)
    pids_sorted = np.array(sorted(labels), dtype=np.int64)
    col = {int(p): j for j, p in enumerate(pids_sorted)}
    cidx = np.array([col[int(p)] for p in col_pid])
    y_pat = np.array([labels[int(p)] for p in pids_sorted])
    n = len(pids_sorted)
    nf = len(folds)

    def _stats(w):
        tp = np.bincount(fold_id, weights=w * ind * yy, minlength=nf)
        pos = np.bincount(fold_id, weights=w * yy, minlength=nf)
        tn = np.bincount(fold_id, weights=w * (1 - ind) * (1 - yy), minlength=nf)
        neg = np.bincount(fold_id, weights=w * (1 - yy), minlength=nf)
        with np.errstate(invalid="ignore", divide="ignore"):
            sens = tp / pos
            spec = tn / neg
        return (float(np.nanmean(sens)) if np.isfinite(sens).any() else np.nan,
                float(np.nanmean(spec)) if np.isfinite(spec).any() else np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        se0, sp0 = _stats(np.ones(len(fold_id)))
        rng = np.random.default_rng(seed)
        se_b, sp_b = np.empty(n_boot), np.empty(n_boot)
        for b in range(n_boot):
            take = _stratified_take(rng, y_pat, stratified)
            counts = np.bincount(take, minlength=n).astype(float)
            se_b[b], sp_b[b] = _stats(counts[cidx])

    def _pack(est, arr):
        arr = arr[np.isfinite(arr)]
        return {"estimate": est,
                "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else np.nan,
                "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else np.nan,
                "n": int(n), "n_events": int(y_pat.sum()), "n_folds": int(nf)}

    return {"Sensitivity": _pack(se0, se_b), "Specificity": _pack(sp0, sp_b)}


# =============================================================================
# 7. Convenience: one call → the standard trio with CIs
# =============================================================================

def summarize_repeat_matrix(rm: RepeatMatrix, n_boot=2000, seed=0,
                            metrics=("AUROC", "AUPRC", "Brier"), seed_offsets=None):
    """Return {metric: bootstrap dict} for a RepeatMatrix."""
    res = {}
    for k, metric in enumerate(metrics):
        s = seed + (seed_offsets[k] if seed_offsets else k)
        res[metric] = bootstrap_repeat_metric_ci(rm.P, rm.y, metric,
                                                 n_boot=n_boot, seed=s)
    return res
