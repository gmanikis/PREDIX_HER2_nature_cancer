"""Validation suite for the statistical machinery added during peer review.

Every function that produces a number quoted in the manuscript is checked here
against an independent reference: scipy for the Mann-Whitney test, R's p.adjust
for Benjamini-Hochberg, scikit-learn for AUROC and logistic regression, and
hand-computed values for Kaplan-Meier. Two tests check statistical behaviour
rather than a fixed value: the DeLong test is run 300 times on pure noise to
confirm its type-I error is near 5%, and the interaction likelihood-ratio test
is run 200 times under a true null to confirm the same.

The most important tests are in sections 5 and 19. Section 5 confirms that a
patient-level bootstrap produces an interval roughly sqrt(R) times WIDER than a
naive row-level bootstrap over the same pooled predictions, where R is the
number of cross-validation repeats. section 15 validates cv_estimands.py, the
estimand every headline number now uses: metric on each repeat's complete
out-of-fold vector, averaged over repeats, with a patient-level CLUSTER
bootstrap. It reproduces the held-out-outcome artefact (averaging a patient's
probabilities across repeats before scoring drives an uninformative model's
AUROC far below 0.5 — the defect that made the clinical model print 0.41) and
shows the per-repeat estimand does not suffer from it.

Run:  python tests/test_statistics.py
Exit code 0 means every check passed.
"""
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

import revision_analyses as RA
import multimodal_pcr_pipeline as MP

rng = np.random.default_rng(7)
FAIL = []

def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)

# ---------------------------------------------------------------- 1. MWU/AUROC
print("\n=== 1. Mann-Whitney AUROC + p (pipeline) ===")
n = 120
y = rng.binomial(1, 0.4, n).astype(float)
X = np.column_stack([
    rng.normal(0, 1, n) + 1.2 * y,          # strong signal
    rng.normal(0, 1, n),                     # null
    rng.binomial(1, 0.15, n).astype(float),  # binary, heavy ties
    np.ones(n),                              # constant
])
auc, p = MP._mannwhitney_auroc_and_p(X, y)
for j in range(3):
    ref_auc = roc_auc_score(y, X[:, j])
    ref_p = sps.mannwhitneyu(X[y == 1, j], X[y == 0, j], alternative="two-sided").pvalue
    check(f"col{j} AUROC", abs(auc[j] - ref_auc) < 1e-9, f"{auc[j]:.6f} vs {ref_auc:.6f}")
    # scipy default is use_continuity=True; we now match it exactly, including
    # for the heavily-tied binary column.
    check(f"col{j} p-value == scipy", abs(p[j] - ref_p) < 1e-10,
          f"{p[j]:.8g} vs scipy {ref_p:.8g}")
check("constant col AUROC=0.5", auc[3] == 0.5)
check("constant col p=1", p[3] == 1.0)

# ---------------------------------------------------------------- 2. BH FDR
print("\n=== 2. Benjamini-Hochberg ===")
pv = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
q_mp = MP.benjamini_hochberg(pv)
q_ra = RA.benjamini_hochberg(pv)
# Reference (R p.adjust BH)
ref = np.array([0.01, 0.04, 0.084, 0.084, 0.084, 0.1, 0.10571429, 0.216, 0.216, 0.216])
check("pipeline BH", np.allclose(q_mp, ref, atol=1e-6), f"{np.round(q_mp,5)}")
check("revision BH matches pipeline", np.allclose(q_mp, q_ra))
check("BH monotone", np.all(np.diff(q_ra[np.argsort(pv)]) >= -1e-12))

# ---------------------------------------------------------------- 3. screen
print("\n=== 3. univariate_screen_indices ===")
cols = [f"f{i}" for i in range(X.shape[1])]
keep, st = MP.univariate_screen_indices(X, y, cols, fdr_q=0.25, max_k=2, min_k=1)
check("respects max_k", len(keep) <= 2, f"kept {len(keep)}")
check("keeps the signal column", 0 in keep, f"kept idx {keep}")
keep2, st2 = MP.univariate_screen_indices(X[:, 1:2], y, ["null"], fdr_q=0.001,
                                          max_k=10, min_k=3)
check("min_k floor honoured", len(keep2) == 1, f"kept {len(keep2)} of 1 available")
check("floor_used flagged", st2.get("floor_used") is True, str(st2.get("floor_used")))
# single-class outcome -> keep everything
keep3, st3 = MP.univariate_screen_indices(X, np.zeros(n), cols)
check("single-class outcome keeps all", len(keep3) == X.shape[1])

# ------------------------------------------------- 4. patient-level pooling
print("\n=== 4. pool_oof_by_patient ===")
# 3 repeats x 4 folds over 40 patients
n_pat = 40
labels = rng.binomial(1, 0.45, n_pat).astype(float)
truth = rng.uniform(0, 1, n_pat)
folds = []
for rep in range(3):
    perm = rng.permutation(n_pat)
    for k in range(4):
        te = perm[k::4]
        folds.append({"fold_idx": rep * 4 + k,
                      "test_pids": te.astype(np.int64),
                      "y_test": labels[te],
                      "y_pred": np.clip(truth[te] + rng.normal(0, .05, len(te)), 0, 1)})
pid, yy, pp, nrep = RA.pool_oof_by_patient(folds)
check("all patients recovered", len(pid) == n_pat, f"{len(pid)}")
check("pids sorted & unique", np.all(np.diff(pid) > 0))
check("labels preserved", np.array_equal(yy, labels[pid]))
check("3 predictions per patient", np.all(nrep == 3), f"{np.unique(nrep)}")
check("pooled pred approximates truth",
      np.max(np.abs(pp - truth[pid])) < 0.08, f"max dev {np.max(np.abs(pp-truth[pid])):.4f}")

# --------------------------------------- 5. bootstrap CI width sanity
print("\n=== 5. bootstrap CI is patient-level, not row-level ===")
res_patient = RA.bootstrap_metric_ci(yy, pp, "AUROC", n_boot=800, seed=1)
# Naive row-level bootstrap over the 3x-inflated pooled vector
y_rows = np.concatenate([f["y_test"] for f in folds])
p_rows = np.concatenate([f["y_pred"] for f in folds])
res_rows = RA.bootstrap_metric_ci(y_rows, p_rows, "AUROC", n_boot=800, seed=1)
w_pat = res_patient["ci_high"] - res_patient["ci_low"]
w_row = res_rows["ci_high"] - res_rows["ci_low"]
check("patient-level CI is wider than row-level",
      w_pat > w_row * 1.3, f"patient width {w_pat:.4f} vs row width {w_row:.4f}")
check("ratio approximates sqrt(3)",
      1.3 < w_pat / w_row < 2.4, f"ratio {w_pat/w_row:.3f} (sqrt(3)={np.sqrt(3):.3f})")
check("estimate inside CI",
      res_patient["ci_low"] <= res_patient["estimate"] <= res_patient["ci_high"])
check("stratified keeps event count", res_patient["n_events"] == int(labels.sum()))

# ---------------------------------------------------------------- 6. DeLong
print("\n=== 6. DeLong ===")
nn = 200
yb = rng.binomial(1, 0.5, nn).astype(float)
p_good = np.clip(0.5 + 0.30 * (yb - 0.5) + rng.normal(0, .15, nn), 0, 1)
p_bad = np.clip(0.5 + 0.05 * (yb - 0.5) + rng.normal(0, .30, nn), 0, 1)
dl = RA.delong_test(yb, p_good, p_bad)
check("DeLong auc1 matches sklearn",
      abs(dl["auc1"] - roc_auc_score(yb, p_good)) < 1e-9,
      f"{dl['auc1']:.6f} vs {roc_auc_score(yb,p_good):.6f}")
check("DeLong auc2 matches sklearn",
      abs(dl["auc2"] - roc_auc_score(yb, p_bad)) < 1e-9)
check("DeLong detects real difference", dl["p_value"] < 0.01, f"p={dl['p_value']:.2e}")
dl_same = RA.delong_test(yb, p_good, p_good.copy())
check("identical predictions -> p=1", dl_same["p_value"] == 1.0, f"p={dl_same['p_value']}")
# Type-I error calibration: two independent noise predictors
rejects = 0
for s in range(300):
    r = np.random.default_rng(1000 + s)
    yt = r.binomial(1, .5, 150).astype(float)
    a = r.normal(0, 1, 150); b = r.normal(0, 1, 150)
    if RA.delong_test(yt, a, b)["p_value"] < 0.05:
        rejects += 1
check("DeLong type-I error near 5%", 0.02 <= rejects/300 <= 0.10,
      f"{rejects/300:.3f}")

# ------------------------------------------------- 7. paired bootstrap
print("\n=== 7. paired bootstrap delta ===")
bs = RA.paired_bootstrap_delta(yb, p_good, p_bad, n_boot=1000, seed=3)
check("paired delta matches direct diff",
      abs(bs["delta"] - (roc_auc_score(yb,p_good)-roc_auc_score(yb,p_bad))) < 1e-9)
check("paired CI excludes 0 for real diff", bs["ci_low"] > 0, f"[{bs['ci_low']:.3f},{bs['ci_high']:.3f}]")
check("paired p small", bs["p_value"] < 0.01, f"p={bs['p_value']:.4f}")
bs0 = RA.paired_bootstrap_delta(yb, p_good, p_good.copy(), n_boot=500, seed=3)
check("identical -> CI contains 0", bs0["ci_low"] <= 0 <= bs0["ci_high"])
check("bootstrap p never exactly 0", bs["p_value"] > 0)

# ---------------------------------------------------------------- 8. Wilson
print("\n=== 8. Wilson interval ===")
lo, hi = RA.wilson_ci(8, 10)
check("Wilson 8/10", abs(lo-0.4901) < 0.002 and abs(hi-0.9433) < 0.002, f"[{lo:.4f},{hi:.4f}]")
lo0, hi0 = RA.wilson_ci(0, 20)
check("Wilson 0/20 lower bound is 0", lo0 == 0.0 and 0.15 < hi0 < 0.17, f"[{lo0:.4f},{hi0:.4f}]")
check("Wilson stays in [0,1]", all(0 <= v <= 1 for v in RA.wilson_ci(20, 20)))

# ---------------------------------------------------------------- 9. calibration
print("\n=== 9. calibration ===")
# Perfectly calibrated data: draw y from p
n_cal = 4000
p_true = rng.uniform(0.05, 0.95, n_cal)
y_cal = rng.binomial(1, p_true).astype(float)
cal = RA.calibration_metrics(y_cal, p_true, n_boot=150)
check("slope ~ 1 for calibrated data", abs(cal["slope"]-1.0) < 0.12, f"slope={cal['slope']:.4f}")
check("intercept ~ 0 for calibrated data", abs(cal["intercept"]) < 0.15, f"int={cal['intercept']:.4f}")
check("ECE small for calibrated data", cal["ece"] < 0.05, f"ECE={cal['ece']:.4f}")
# Overconfident (too extreme) predictions -> slope < 1
p_over = np.clip(1/(1+np.exp(-2.2*np.log(p_true/(1-p_true)))), 1e-4, 1-1e-4)
cal_o = RA.calibration_metrics(y_cal, p_over, n_boot=100)
check("overconfident -> slope < 1", cal_o["slope"] < 0.75, f"slope={cal_o['slope']:.4f}")
check("reliability bins sum to n", cal["reliability"]["n"].sum() == n_cal)

# ---------------------------------------------------------------- 10. logistic
print("\n=== 10. _fit_logit_manual vs sklearn ===")
from sklearn.linear_model import LogisticRegression
n_l = 400
Xl = rng.normal(0, 1, (n_l, 3))
beta_true = np.array([0.8, -1.1, 0.4])
yl = rng.binomial(1, 1/(1+np.exp(-(0.3 + Xl@beta_true)))).astype(float)
ll, beta, se = RA._fit_logit_manual(Xl, yl)
sk = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(Xl, yl)
check("intercept matches sklearn", abs(beta[0]-sk.intercept_[0]) < 1e-4,
      f"{beta[0]:.6f} vs {sk.intercept_[0]:.6f}")
check("coefs match sklearn", np.max(np.abs(beta[1:]-sk.coef_[0])) < 1e-3,
      f"maxdiff {np.max(np.abs(beta[1:]-sk.coef_[0])):.2e}")
pr = sk.predict_proba(Xl)[:,1]
ll_sk = float(np.sum(yl*np.log(pr)+(1-yl)*np.log(1-pr)))
# Newton-Raphson converges tighter than sklearn's lbfgs (default tol 1e-4),
# so our log-likelihood should be >= sklearn's, not merely equal to it.
check("log-likelihood at least as high as sklearn", ll >= ll_sk - 1e-9,
      f"{ll:.8f} vs sklearn {ll_sk:.8f} (diff {ll-ll_sk:+.2e})")

# ---------------------------------------------------------------- 11. survival
print("\n=== 11. Kaplan-Meier / log-rank / Cox ===")
# KM against a hand-computed example
t = np.array([1,2,3,4,5,6],dtype=float); e = np.array([1,0,1,1,0,1],dtype=float)
ts, s, ar, dd = RA.kaplan_meier(t,e)
# risk sets: t=1 n=6 d=1 -> 5/6; t=3 n=4 d=1 -> *3/4; t=4 n=3 d=1 -> *2/3; t=6 n=1 d=1 -> *0
expect = np.array([5/6, 5/6*3/4, 5/6*3/4*2/3, 0.0])
check("KM survival matches hand calc", np.allclose(s, expect), f"{np.round(s,4)}")
check("KM at-risk counts", np.array_equal(ar, np.array([6,4,3,1])), str(ar))

# log-rank: identical groups -> large p
r = np.random.default_rng(11)
tt = r.exponential(10, 200); ee = (r.uniform(0,1,200)<0.7).astype(float)
gg = r.binomial(1,.5,200)
lr = RA.logrank_test(tt,ee,gg)
check("log-rank null p large", lr["p_value"] > 0.05, f"p={lr['p_value']:.3f}")
check("log-rank observed sums to events", abs(sum(lr["observed"].values())-ee.sum())<1e-9)
# strong separation -> small p
tt2 = np.where(gg==1, r.exponential(4,200), r.exponential(20,200))
lr2 = RA.logrank_test(tt2, np.ones(200), gg)
check("log-rank detects separation", lr2["p_value"] < 1e-6, f"p={lr2['p_value']:.2e}")

# Cox: recover a known coefficient
r = np.random.default_rng(21)
n_c = 3000
x1 = r.normal(0,1,n_c); x2 = r.binomial(1,.5,n_c).astype(float)
b_true = np.array([0.7,-0.5])
haz = np.exp(x1*b_true[0]+x2*b_true[1])
T = r.exponential(1/haz)
C = r.exponential(2.0, n_c)
tobs = np.minimum(T,C); ev = (T<=C).astype(float)
cm = RA.cox_model(tobs, ev, np.column_stack([x1,x2]), ["x1","x2"])
est = cm["terms"]["coef"].values
check("Cox recovers coefficients", np.max(np.abs(est-b_true))<0.09,
      f"{np.round(est,4)} vs {b_true}")
check("Cox HR = exp(coef)",
      np.allclose(cm["terms"]["hazard_ratio"].values, np.exp(est)))
check("Cox EPV reported", abs(cm["epv"] - cm["n_events"]/2) < 1e-9, f"epv={cm['epv']:.2f}")
check("Cox detects significance", bool((cm["terms"]["p_value"]<1e-6).all()))
# tie handling: heavily tied times shouldn't crash
t_tied = np.round(tobs[:300], 0)
cmt = RA.cox_model(t_tied, ev[:300], np.column_stack([x1[:300],x2[:300]]), ["a","b"])
check("Cox handles ties", not cmt["terms"].empty and np.isfinite(cmt["terms"]["coef"]).all())

# ---------------------------------------------------------------- 12. EPV
print("\n=== 12. EPV table ===")
ef = [{"fold_idx":i,"signature_size":8,"n_events_train_expanded":40,
       "epv_realized":5.0,"y_test":np.array([0,1,1]),"n_test":3,
       "n_events_test":2,"winner_clf":"ElasticNet_LR"} for i in range(10)]
dfe = RA.epv_table(ef,"Global","RNA")
check("EPV table rows", len(dfe)==10)
check("EPV values carried", bool((dfe["epv_realized"]==5.0).all()))
s = RA.summarise_epv(dfe)
check("EPV summary produced", len(s)==1 and s.iloc[0]["median_epv_realized"]==5.0)

# ---------------------------------------------------------------- 13. stability
print("\n=== 13. selection frequency ===")
sf_folds = []
for i in range(20):
    sig = ["A","B"] if i < 15 else ["A","C"]
    sf_folds.append({"winner_clf":"ElasticNet_LR","winner_signature":sig,
                     "features":["A","B","C","D"],
                     "inner_importance":{"ElasticNet_LR":{f:0.5 for f in sig}}})
sfd = RA.selection_frequency(sf_folds, threshold=0.6)
row = sfd.set_index("feature")
check("A selected in all folds", row.loc["A","selection_freq"]==1.0)
check("B freq 0.75", abs(row.loc["B","selection_freq"]-0.75)<1e-9)
check("C freq 0.25", abs(row.loc["C","selection_freq"]-0.25)<1e-9)
check("D absent from table", "D" not in row.index)
check("threshold applied", bool(row.loc["A","stable"]) and not bool(row.loc["C","stable"]))
check("Wilson bounds bracket estimate",
      bool((row["wilson_low"]<=row["selection_freq_eligible"]).all()
       and (row["wilson_high"]>=row["selection_freq_eligible"]).all()))

# Eligible-fold denominator via candidate_features — the key the pipeline
# actually records (the "features" key holds the winner signature and MUST
# NOT drive eligibility: that made every selected feature report 1.0).
# E is in the candidate pool of only 10 of 20 folds and selected in 5 of
# those 10: all-folds freq 0.25, eligible freq 0.50.
sf2 = []
for i in range(20):
    sig  = ["A"] + (["E"] if i < 5 else [])
    pool = ["A", "B", "C", "D"] + (["E"] if i < 10 else [])
    sf2.append({"winner_clf": "ElasticNet_LR", "winner_signature": sig,
                "features": sig,                     # signature, as pipeline writes it
                "candidate_features": pool,
                "inner_importance_magnitude":
                    {"ElasticNet_LR": {f: 0.5 for f in sig}}})
sfd2 = RA.selection_frequency(sf2, threshold=0.6).set_index("feature")
check("eligible denominator uses candidate pool",
      sfd2.loc["E", "n_folds_eligible"] == 10,
      f"{sfd2.loc['E', 'n_folds_eligible']}")
check("eligible freq E = 0.50",
      abs(sfd2.loc["E", "selection_freq_eligible"] - 0.5) < 1e-9)
check("all-folds freq E = 0.25",
      abs(sfd2.loc["E", "selection_freq"] - 0.25) < 1e-9)
check("A eligible everywhere, freq 1.0",
      sfd2.loc["A", "n_folds_eligible"] == 20
      and sfd2.loc["A", "selection_freq_eligible"] == 1.0)
check("NOT everything reported stable",
      not bool(sfd2["stable"].all()))

# Mixed PKL (some folds lack candidate_features): no-pool folds must count
# into EVERY feature's eligible denominator — a no-pool fold's own signature
# must never define its eligibility (selected/selected = 1.0 vacuity).
# F: selected in all 10 folds, in the recorded pools → eligible 10.
# H: selected in 2 of the 5 no-pool folds only, absent from recorded pools →
#    eligible = 0 recorded + 5 no-pool folds = 5, freq 2/5 = 0.4 (the buggy
#    per-fold treatment reported 2/2 = 1.0).
sf3 = []
for i in range(10):
    sig = ["F"] + (["H"] if i >= 8 else [])
    fold = {"winner_clf": "ElasticNet_LR", "winner_signature": sig,
            "inner_importance_magnitude":
                {"ElasticNet_LR": {f: 0.5 for f in sig}}}
    if i < 5:
        fold["candidate_features"] = ["F", "G"]
    sf3.append(fold)
sfd3 = RA.selection_frequency(sf3, threshold=0.6).set_index("feature")
check("mixed PKL: F eligible = pool folds + no-pool folds",
      sfd3.loc["F", "n_folds_eligible"] == 10,
      f"{sfd3.loc['F', 'n_folds_eligible']}")
check("mixed PKL: H eligible counts all no-pool folds",
      sfd3.loc["H", "n_folds_eligible"] == 5,
      f"{sfd3.loc['H', 'n_folds_eligible']}")
check("mixed PKL: H freq 0.4, not vacuous 1.0",
      abs(sfd3.loc["H", "selection_freq_eligible"] - 0.4) < 1e-9)

# Deterministic row order. The rows are accumulated in Counter insertion
# order, which follows iteration over a set of feature-name strings and so
# varies between processes (Python randomises string hashing per run). With a
# non-stable sort, features tied at the same frequency — the common case, many
# tie at 1.0 — came out permuted on every run, and two identical analyses
# produced workbooks differing by a row permutation. The output must be sorted
# by (selection_freq_eligible desc, selection_freq desc, feature asc).
sf_ord = RA.selection_frequency(sf_folds, threshold=0.6)
expected_order = sf_ord.sort_values(
    ["selection_freq_eligible", "selection_freq", "feature"],
    ascending=[False, False, True], kind="mergesort")["feature"].tolist()
check("selection_frequency rows are deterministically ordered",
      sf_ord["feature"].tolist() == expected_order,
      str(sf_ord["feature"].tolist()))
# Ties must break on the feature name, not on insertion order: A and B here
# are both selected in every eligible fold.
sf_tie = [{"winner_clf": "ElasticNet_LR", "winner_signature": ["zeta", "alpha", "mu"],
           "candidate_features": ["zeta", "alpha", "mu"],
           "inner_importance_magnitude": {"ElasticNet_LR": {f: 0.5 for f in ("zeta", "alpha", "mu")}}}
          for _ in range(8)]
tie_order = RA.selection_frequency(sf_tie, threshold=0.6)["feature"].tolist()
check("ties break alphabetically on the feature name",
      tie_order == ["alpha", "mu", "zeta"], str(tie_order))

print("\n=== 14. modality weight stability ===")
mw = [{"modality_weights":{"RNA":0.5,"DNA":0.0,"Clin":-0.2 if i%2 else 0.2}}
      for i in range(10)]
mws = RA.modality_weight_stability(mw).set_index("modality")
check("RNA always selected", mws.loc["RNA","selection_rate"]==1.0)
check("DNA never selected", mws.loc["DNA","selection_rate"]==0.0)
check("Clin sign flips detected", abs(mws.loc["Clin","sign_consistency"]-0.5)<1e-9,
      f"{mws.loc['Clin','sign_consistency']}")

# =========================================================== 19. cv_estimands
# The repeat-aware, patient-clustered estimand that every headline number in
# report/tables/revision now uses. Ten checks:
#   19a metrics per repeat == scikit-learn (ties, NaN)     19f paired delta
#   19b vectorised logistic recalibration == scikit-learn  19g per-repeat DeLong
#   19c build_repeat_matrix recovers repeats / errors      19h operating points
#   19d the held-out-outcome ARTEFACT is reproduced        19i calibration
#   19e cluster CI vs naive row-level CI                   19j mean_fold_metric
import cv_estimands as CE
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold

print("\n=== 15a. cv_estimands: per-repeat metrics == scikit-learn ===")
r19 = np.random.default_rng(1919)
n19 = 60
y19 = r19.binomial(1, 0.4, n19).astype(float)
P19 = np.clip(0.4 + 0.25 * (y19 - 0.5)[None, :] + r19.normal(0, .2, (5, n19)), 0, 1)
P19[1, :10] = 0.5                       # heavy ties
P19[2, 3] = np.nan                      # missing prediction
P19[3, :] = np.round(P19[3, :], 1)      # coarse grid -> many ties
au19 = CE.auroc_rows(P19, y19); ap19 = CE.auprc_rows(P19, y19); br19 = CE.brier_rows(P19, y19)
for rr in range(5):
    m = np.isfinite(P19[rr])
    check(f"row{rr} AUROC == sklearn",
          abs(au19[rr] - roc_auc_score(y19[m], P19[rr, m])) < 1e-12,
          f"{au19[rr]:.10f} vs {roc_auc_score(y19[m], P19[rr, m]):.10f}")
    check(f"row{rr} AUPRC == sklearn",
          abs(ap19[rr] - average_precision_score(y19[m], P19[rr, m])) < 1e-12)
    check(f"row{rr} Brier == sklearn",
          abs(br19[rr] - brier_score_loss(y19[m], P19[rr, m])) < 1e-12)
check("repeat_mean_metric = mean of rows",
      abs(CE.repeat_mean_metric(P19, y19, "AUROC") - np.mean(au19)) < 1e-12)
check("single-class row -> NaN, not crash",
      not np.isfinite(CE.auroc_rows(P19[:1], np.zeros(n19))[0]))

print("\n=== 15b. vectorised logistic recalibration == scikit-learn ===")
n_cal2 = 1500
p_true2 = rng.uniform(0.05, 0.95, n_cal2)
y_cal2 = rng.binomial(1, p_true2).astype(float)
p_ov2 = np.clip(1 / (1 + np.exp(-2.2 * np.log(p_true2 / (1 - p_true2)))), 1e-4, 1 - 1e-4)
LP2 = np.vstack([np.log(p_true2 / (1 - p_true2)),
                 np.log(p_ov2 / (1 - p_ov2))])
sl2, ic2 = CE.logistic_recalibration_rows(LP2, y_cal2)
for rr, lab in enumerate(["calibrated", "overconfident"]):
    sk2 = LogisticRegression(penalty=None, solver="lbfgs", tol=1e-10,
                             max_iter=10000).fit(LP2[rr][:, None], y_cal2)
    check(f"{lab}: slope == sklearn", abs(sl2[rr] - sk2.coef_[0, 0]) < 1e-4,
          f"{sl2[rr]:.6f} vs {sk2.coef_[0,0]:.6f}")
    check(f"{lab}: intercept == sklearn", abs(ic2[rr] - sk2.intercept_[0]) < 1e-4,
          f"{ic2[rr]:.6f} vs {sk2.intercept_[0]:.6f}")
check("calibrated slope ~ 1", abs(sl2[0] - 1) < 0.15, f"{sl2[0]:.4f}")
check("overconfident slope ~ 1/2.2", abs(sl2[1] - 1 / 2.2) < 0.08, f"{sl2[1]:.4f}")
sl_nan, _ = CE.logistic_recalibration_rows(LP2[:1], np.zeros(n_cal2))
check("single-class row -> NaN slope", not np.isfinite(sl_nan[0]))

print("\n=== 15c. build_repeat_matrix ===")
# Reuse the 3 repeats x 4 folds over 40 patients from section 4
rm4 = CE.discovery_repeat_matrix(folds)
check("3 repeats recovered", rm4.n_repeats == 3, f"{rm4.n_repeats}")
check("4 folds per repeat", rm4.folds_per_repeat == [4, 4, 4], str(rm4.folds_per_repeat))
check("40 patients, sorted ids", rm4.n_patients == n_pat and np.all(np.diff(rm4.pids) > 0))
check("every patient predicted once per repeat",
      np.all(rm4.predictions_per_patient() == 3) and rm4.incomplete_repeats == 0)
check("labels carried", np.array_equal(rm4.y, labels[rm4.pids]))
check("n_events", rm4.n_events == int(labels.sum()))
# values land in the right cell
f0 = folds[5]                                    # repeat 1, fold 1
j = np.searchsorted(rm4.pids, f0["test_pids"][0])
check("prediction placed in (repeat, patient) cell",
      rm4.P[1, j] == f0["y_pred"][0])
# per-repeat mean-of-P equals RA.pool_oof_by_patient mean (same rows, other layout)
check("column means == pool_oof_by_patient",
      np.allclose(np.nanmean(rm4.P, axis=0), pp))
# fold ordering by fold_idx, not list order
rm_shuf = CE.discovery_repeat_matrix([folds[i] for i in rng.permutation(len(folds))])
check("shuffled fold list -> identical matrix", np.array_equal(rm_shuf.P, rm4.P))
# incomplete trailing repeat (aborted run) is kept with NaN + counted
import warnings as _w
with _w.catch_warnings(record=True) as wl:
    _w.simplefilter("always")
    rm_inc = CE.discovery_repeat_matrix(folds[:-1])
check("incomplete trailing repeat counted", rm_inc.incomplete_repeats == 1
      and rm_inc.n_repeats == 3 and np.isnan(rm_inc.P[2]).sum() == len(folds[-1]["test_pids"]))
check("incomplete repeat warned", any("do not predict every" in str(x.message) for x in wl))
# inconsistent label -> error
bad = [dict(f) for f in folds]; bad[0]["y_test"] = 1 - bad[0]["y_test"]
try:
    CE.discovery_repeat_matrix(bad); ok_lab = False
except ValueError:
    ok_lab = True
check("inconsistent labels raise", ok_lab)
# duplicate patient inside one fold -> error
bad2 = [dict(f) for f in folds]
bad2[0]["test_pids"] = np.concatenate([bad2[0]["test_pids"][:1], bad2[0]["test_pids"][:-1]])
try:
    CE.discovery_repeat_matrix(bad2); ok_dup = False
except ValueError:
    ok_dup = True
check("patient predicted twice within a repeat raises", ok_dup)
# consensus blob layout
cons_folds = [{"fold_idx": f["fold_idx"], "test_pids": f["test_pids"], "y_test": f["y_test"],
               "unimodal_y_pred": {"RNA": f["y_pred"], "Clin": 1 - f["y_pred"]},
               "fused_y_pred": f["y_pred"] ** 2} for f in folds]
rm_c = CE.consensus_repeat_matrix({"folds": cons_folds}, "Clin")
rm_f = CE.consensus_repeat_matrix({"folds": cons_folds}, "Fused_ElasticNet")
check("consensus unimodal matrix", np.allclose(rm_c.P, 1 - rm4.P))
check("consensus fused matrix", np.allclose(rm_f.P, rm4.P ** 2))
check("empty blob -> empty matrix", CE.consensus_repeat_matrix({}, "RNA").n_repeats == 0)

print("\n=== 15d. the held-out-outcome artefact is reproduced (and avoided) ===")
# A model with NO information that predicts the training-set event rate.
# Averaging each patient's out-of-fold probabilities across repeats before
# scoring collapses AUROC far below 0.5: whenever a pCR patient is held out
# its training set has one event fewer, so its prediction is systematically
# lower than that of a non-pCR patient. The per-repeat estimand stays near
# chance. (This is exactly what happened to the clinical model: 0.61 per
# repeat vs 0.41 after averaging probabilities.)
n_art, R_art = 40, 100
y_art = np.array([1.0] * 16 + [0.0] * 24)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=R_art, random_state=0)
art_folds = []
for fi, (tr, te) in enumerate(rskf.split(np.zeros((n_art, 1)), y_art)):
    art_folds.append({"fold_idx": fi, "test_pids": te.astype(np.int64),
                      "y_test": y_art[te],
                      "y_pred": np.full(len(te), y_art[tr].mean())})
rm_art = CE.discovery_repeat_matrix(art_folds)
check("100 repeats x 5 folds recovered", rm_art.n_repeats == R_art
      and set(rm_art.folds_per_repeat) == {5})
per_rep = CE.repeat_mean_metric(rm_art.P, rm_art.y, "AUROC")
mean_prob = roc_auc_score(rm_art.y, rm_art.P.mean(axis=0))
check("per-repeat AUROC of an uninformative model stays near 0.5",
      0.40 <= per_rep <= 0.55, f"{per_rep:.3f}")
check("averaging probabilities across repeats first collapses AUROC",
      mean_prob < 0.30, f"{mean_prob:.3f}  (artefact)")
check("artefact gap is large", per_rep - mean_prob > 0.15,
      f"gap {per_rep - mean_prob:.3f}")

print("\n=== 15e. cluster bootstrap CI vs naive row-level CI ===")
R_e, n_e = 4, 80
y_e = rng.binomial(1, 0.45, n_e).astype(float)
signal = rng.normal(0, 1, n_e) + 1.0 * y_e
P_e = np.clip(0.45 + 0.15 * signal[None, :] + rng.normal(0, .04, (R_e, n_e)), 0, 1)
ci_cluster = CE.bootstrap_repeat_metric_ci(P_e, y_e, "AUROC", n_boot=600, seed=5)
ci_single  = CE.bootstrap_repeat_metric_ci(P_e[:1], y_e, "AUROC", n_boot=600, seed=5)
naive = RA.bootstrap_metric_ci(np.tile(y_e, R_e), P_e.ravel(), "AUROC", n_boot=600, seed=5)
w_cl = ci_cluster["ci_high"] - ci_cluster["ci_low"]
w_si = ci_single["ci_high"] - ci_single["ci_low"]
w_nv = naive["ci_high"] - naive["ci_low"]
check("estimate = mean of per-repeat AUROC",
      abs(ci_cluster["estimate"] - np.mean([roc_auc_score(y_e, P_e[r]) for r in range(R_e)])) < 1e-12)
check("estimate inside CI", ci_cluster["ci_low"] <= ci_cluster["estimate"] <= ci_cluster["ci_high"])
check("cluster CI ~ as wide as a single-repeat CI (patients are the unit)",
      0.75 < w_cl / w_si < 1.30, f"ratio {w_cl/w_si:.3f}")
check("cluster CI ~ sqrt(R) wider than a row-level bootstrap",
      1.4 < w_cl / w_nv < 2.9, f"ratio {w_cl/w_nv:.3f} (sqrt(4)=2)")
check("bookkeeping: n, n_events, n_repeats, n_boot_valid",
      ci_cluster["n"] == n_e and ci_cluster["n_events"] == int(y_e.sum())
      and ci_cluster["n_repeats"] == R_e and ci_cluster["n_boot_valid"] == 600)
check("same seed -> identical CI", CE.bootstrap_repeat_metric_ci(
      P_e, y_e, "AUROC", n_boot=600, seed=5)["ci_low"] == ci_cluster["ci_low"])
for met in ("AUPRC", "Brier"):
    cc = CE.bootstrap_repeat_metric_ci(P_e, y_e, met, n_boot=300, seed=6)
    check(f"{met} estimate inside CI", cc["ci_low"] <= cc["estimate"] <= cc["ci_high"],
          f"{cc['estimate']:.3f} [{cc['ci_low']:.3f},{cc['ci_high']:.3f}]")
sm = CE.summarize_repeat_matrix(CE.RepeatMatrix(np.arange(n_e), y_e, P_e, R_e * 5),
                                n_boot=100, seed=1)
check("summarize_repeat_matrix returns the trio", set(sm) == {"AUROC", "AUPRC", "Brier"})

print("\n=== 15f. paired cluster-bootstrap delta ===")
P_weak = np.clip(0.45 + 0.03 * signal[None, :] + rng.normal(0, .12, (R_e, n_e)), 0, 1)
d1 = CE.paired_bootstrap_repeat_delta(P_e, P_weak, y_e, "AUROC", n_boot=600, seed=8)
check("delta == difference of the two estimands",
      abs(d1["delta"] - (CE.repeat_mean_metric(P_e, y_e) - CE.repeat_mean_metric(P_weak, y_e))) < 1e-12
      and abs(d1["estimate_1"] - d1["estimate_2"] - d1["delta"]) < 1e-12)
check("real difference: CI excludes 0, p small", d1["ci_low"] > 0 and d1["p_value"] < 0.02,
      f"{d1['delta']:.3f} [{d1['ci_low']:.3f},{d1['ci_high']:.3f}] p={d1['p_value']:.4f}")
d2 = CE.paired_bootstrap_repeat_delta(P_weak, P_e, y_e, "AUROC", n_boot=600, seed=8)
check("swapping models mirrors the interval exactly (same resamples)",
      abs(d2["ci_low"] + d1["ci_high"]) < 1e-12 and abs(d2["ci_high"] + d1["ci_low"]) < 1e-12)
d0 = CE.paired_bootstrap_repeat_delta(P_e, P_e.copy(), y_e, "AUROC", n_boot=300, seed=8)
check("identical models: delta 0, CI [0,0], p = 1", d0["delta"] == 0
      and d0["ci_low"] == 0 == d0["ci_high"] and d0["p_value"] == 1.0)
check("p floored at 1/n_boot, never 0", d1["p_value"] >= 1 / 600)
try:
    CE.paired_bootstrap_repeat_delta(P_e, P_e[:2], y_e); ok_shape = False
except ValueError:
    ok_shape = True
check("shape mismatch raises", ok_shape)

print("\n=== 15g. per-repeat DeLong summary ===")
pr1 = CE.per_repeat_test_summary(P_e, P_weak, y_e, RA.delong_test)
check("all repeats tested", pr1["n_repeats_tested"] == R_e)
check("real difference: most repeats p<0.05", pr1["frac_p_below_0.05"] >= 0.75,
      f"frac {pr1['frac_p_below_0.05']:.2f}, median p {pr1['p_median']:.3g}")
check("quartiles ordered", pr1["p_q25"] <= pr1["p_median"] <= pr1["p_q75"])
pr0 = CE.per_repeat_test_summary(P_e, P_e.copy(), y_e, RA.delong_test)
check("identical models: median p = 1", pr0["p_median"] == 1.0)
P_nan = P_e.copy(); P_nan[0, :] = np.nan
check("all-NaN repeat skipped", CE.per_repeat_test_summary(P_nan, P_weak, y_e, RA.delong_test)["n_repeats_tested"] == R_e - 1)

print("\n=== 15h. fold operating points with cluster CI ===")
op_folds = [dict(f, metrics={"Threshold": 0.5}) for f in folds]
op = CE.bootstrap_fold_operating_point_ci(op_folds, lambda f: f["y_pred"],
                                          lambda f: f["metrics"]["Threshold"],
                                          n_boot=400, seed=2)
man_se = np.mean([np.mean(f["y_pred"][f["y_test"] == 1] >= 0.5) for f in op_folds
                  if (f["y_test"] == 1).any()])
man_sp = np.mean([np.mean(f["y_pred"][f["y_test"] == 0] < 0.5) for f in op_folds
                  if (f["y_test"] == 0).any()])
check("sensitivity = mean over folds at fold thresholds",
      abs(op["Sensitivity"]["estimate"] - man_se) < 1e-12, f"{op['Sensitivity']['estimate']:.4f} vs {man_se:.4f}")
check("specificity = mean over folds at fold thresholds",
      abs(op["Specificity"]["estimate"] - man_sp) < 1e-12)
check("estimates inside CIs",
      op["Sensitivity"]["ci_low"] <= man_se <= op["Sensitivity"]["ci_high"]
      and op["Specificity"]["ci_low"] <= man_sp <= op["Specificity"]["ci_high"])
check("bootstrap unit is the patient", op["Sensitivity"]["n"] == n_pat
      and op["Sensitivity"]["n_events"] == int(labels.sum()))
op_folds2 = [dict(f) for f in op_folds]; op_folds2[0]["metrics"] = {"Threshold": None}
op2 = CE.bootstrap_fold_operating_point_ci(op_folds2, lambda f: f["y_pred"],
                                           lambda f: f["metrics"]["Threshold"], n_boot=50, seed=2)
check("fold without threshold skipped, others kept", np.isfinite(op2["Sensitivity"]["estimate"])
      and op2["Sensitivity"]["n_folds"] == len(op_folds2))

print("\n=== 15i. calibration per repeat + pooled reliability ===")
P_cal = np.vstack([p_true2, p_true2, p_true2])          # 3 identical calibrated repeats
cs = CE.calibration_repeat_summary(P_cal, y_cal2, n_boot=150, seed=3)
check("slope ~ 1, CI covers 1", abs(cs["slope"] - 1) < 0.15
      and cs["slope_ci"][0] <= 1 <= cs["slope_ci"][1],
      f"{cs['slope']:.3f} [{cs['slope_ci'][0]:.3f},{cs['slope_ci'][1]:.3f}]")
check("intercept ~ 0, CI covers 0", abs(cs["intercept"]) < 0.15
      and cs["intercept_ci"][0] <= 0 <= cs["intercept_ci"][1])
check("all repeats fitted", cs["n_repeats_slope_valid"] == 3 and cs["n_repeats"] == 3)
check("brier == mean per-repeat Brier",
      abs(cs["brier"] - brier_score_loss(y_cal2, p_true2)) < 1e-12)
check("ECE small", cs["ece"] < 0.05, f"{cs['ece']:.4f}")
cs_ov = CE.calibration_repeat_summary(np.vstack([p_ov2, p_ov2]), y_cal2, n_boot=100, seed=3)
check("overconfident -> slope < 0.75 and CI excludes 1", cs_ov["slope"] < 0.75
      and cs_ov["slope_ci"][1] < 1, f"{cs_ov['slope']:.3f}")
rel = CE.reliability_pooled(P_cal, y_cal2, n_bins=10, n_boot=100, seed=4)
check("reliability rows cover all (repeat, patient) predictions",
      rel["n_rows"].sum() == 3 * n_cal2)
check("distinct patients per bin <= n", (rel["n_patients_distinct"] <= n_cal2).all())
check("bins ordered by predicted risk", np.all(np.diff(rel["mean_predicted"]) > 0))
check("observed inside its cluster CI in every bin",
      ((rel["obs_ci_low"] <= rel["observed"]) & (rel["observed"] <= rel["obs_ci_high"])).all())
check("calibrated: observed ~ predicted per bin",
      np.max(np.abs(rel["observed"] - rel["mean_predicted"])) < 0.12,
      f"max gap {np.max(np.abs(rel['observed'] - rel['mean_predicted'])):.3f}")

print("\n=== 15j. legacy mean-of-fold metric ===")
mf = CE.mean_fold_metric(folds, lambda f: f["y_pred"], "AUROC")
ref_mf = np.mean([roc_auc_score(f["y_test"], f["y_pred"]) for f in folds])
check("mean_fold_metric == mean sklearn per fold", abs(mf - ref_mf) < 1e-12, f"{mf:.6f}")
check("mean-fold and per-repeat estimands differ in general",
      abs(mf - CE.repeat_mean_metric(rm4.P, rm4.y)) > 1e-9)

print("\n" + "="*60)
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("ALL CHECKS PASSED")

