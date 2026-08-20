#!/usr/bin/env python3
"""
Build RESULTS.md — every result of the analysis, rendered so that it can be read
on GitHub without downloading or running anything.

Tables are generated from the deposited workbooks, never typed, so the page
cannot drift away from the analysis. Figures are embedded as PNG (GitHub does
not render PDF inline); the PDFs remain the citable artefacts.

    python revision_deliverables/build_RESULTS_md.py

Writes  predix-her2-multimodal/RESULTS.md
        predix-her2-multimodal/report/figures_png/*.png
"""
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "predix-her2-multimodal"
TAB = REPO / "report" / "tables"
# Run 5: the pooled-model external validation lives in its own report tree, so
# that the two analyses can never again be shipped under identical basenames.
POOL_TAB = REPO / "report_pooled_external" / "tables"
PNG_SRC = ROOT / "revision_deliverables" / "figures_png"
PNG_DST = REPO / "report" / "figures_png"
OUT = REPO / "RESULTS.md"

if not REPO.exists():
    sys.exit("run build_github_repo.py first")


# --------------------------------------------------------------- reading ----
def sheet(path, name, first_header):
    """Read one styled worksheet; the header row is the row whose first cell
    equals `first_header` (explanatory note rows precede it)."""
    ws = openpyxl.load_workbook(path, data_only=True)[name]
    rows = [[c.value for c in r] for r in ws.iter_rows()]
    i = next(k for k, r in enumerate(rows) if r and r[0] == first_header)
    hdr = [h for h in rows[i] if h is not None]
    body = [list(r[:len(hdr)]) + [None] * max(0, len(hdr) - len(r))
            for r in rows[i + 1:] if any(v is not None for v in r)]
    return pd.DataFrame(body, columns=hdr)


def col(row, *names):
    """The first of `names` that exists in the row.

    The revision workbooks renamed columns between run 3 and run 5 —
    `calibration_intercept` → `recalibration_intercept`, `matched_PREDIX_arm` →
    `cohort_resembles_PREDIX_arm`, `delta_vs_best_p_bootstrap` →
    `delta_vs_best_p_marginal_selected_comparator`. Every renamed cell is read
    through here (run-5 name first) so this page builds against either vintage
    instead of dying on a KeyError halfway through.
    """
    for n in names:
        if n in row.index:
            return row[n]
    raise KeyError(f"none of {names} present; row has {list(row.index)}")


def _f(v):
    """Coerce a worksheet cell to float, or None. numpy integer types are not
    instances of `int` on Windows, so isinstance checks are not usable here."""
    if v is None or isinstance(v, str):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def num(v, nd=3):
    """A measurement: always nd decimals."""
    if isinstance(v, str):
        return v
    f = _f(v)
    return "—" if f is None else f"{f:.{nd}f}"


def cnt(v):
    """A count: integer, with thousands separators."""
    if isinstance(v, str):
        return v
    f = _f(v)
    return "—" if f is None else f"{int(round(f)):,}"


def pct(v, nd=1):
    f = _f(v)
    return "—" if f is None else f"{f:.{nd}%}"


def pval(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    v = float(v)
    return "< 0.001" if v < 0.001 else f"{v:.3f}"


def ci(lo, hi, nd=3):
    """An en-dash reads as a minus sign when a bound is negative, so switch to
    'a to b' whenever either bound is below zero."""
    a, b = _f(lo), _f(hi)
    sep = " to " if (a is not None and a < 0) or (b is not None and b < 0) else "–"
    return f"{num(lo, nd)}{sep}{num(hi, nd)}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


MOD = {"Clin": "Clinical", "RNA": "Transcriptomic", "DNA": "Genomic",
       "Prot": "Proteomic", "WSI": "Whole-slide image",
       "Fused_ElasticNet": "**Integrated (late fusion)**"}
ORDER = ["Clin", "RNA", "DNA", "Prot", "WSI", "Fused_ElasticNet"]
SC = ["Global", "DHP", "T-DM1"]
SCL = {"Global": "Pooled cohort", "DHP": "DHP arm", "T-DM1": "T-DM1 arm"}

FIG_CAPTION = {
    "fig01_consensus_performance": "Cross-validated AUROC of every consensus model with its 95% patient-level cluster-bootstrap interval, in the pooled cohort and each arm.",
    "fig02_consensus_signatures": "The frozen consensus signature of each modality and scenario: mean absolute SHAP importance per feature, coloured by the direction of the association, with the winning classifier family above each panel.",
    "fig03_consensus_roc": "Out-of-fold ROC curves of the integrated model and of the best single modality, drawn on all pooled (patient, repeat) predictions.",
    "fig04_consensus_modality_weights": "Late-fusion modality weights of the consensus models: mean elastic-net coefficient and the fraction of folds in which each modality received a non-zero weight.",
    "fig05_consensus_feature_shap_Global": "Feature-level SHAP attribution for the pooled-cohort consensus models, restricted to the consensus signature.",
    "fig05_consensus_feature_shap_DHP": "Feature-level SHAP attribution for the DHP-arm consensus models.",
    "fig05_consensus_feature_shap_T_DM1": "Feature-level SHAP attribution for the T-DM1-arm consensus models.",
    "fig06_counterfactual_summary": "Counterfactual summary: predicted response under each treatment assignment.",
    "revfig01_calibration": "Calibration of the consensus integrated model: reliability curves over ten equal-count bins of all out-of-fold predictions, with patient-level cluster-bootstrap intervals, and the slope, intercept and Brier score of each scenario.",
    "revfig02_selection_stability": "Feature-selection frequency across the outer folds, with Wilson intervals and the pre-specified stability threshold (0.60 pooled, 0.50 per arm).",
    "revfig03_epv_per_fold": "Per-fold pCR event counts and realised events-per-variable for every model.",
    "revfig06_external_validation": "Locked-model external validation, arm-matched design: ROC and precision–recall curves and reliability of the frozen DHP and T-DM1 transcriptomic models in I-SPY2 and NCT02326974.",
    # Run 5 renders a second external-validation figure for the pooled model.
    # Without a caption entry the PNG is copied into the repository but never
    # appears on this page, because the rendering loop iterates FIG_CAPTION.
    "revfig06_external_validation_POOLED": "Locked-model external validation, pooled design: the same two cohorts scored by a single transcriptomic model refit on all PREDIX patients carrying transcriptomics, irrespective of treatment arm. Pre-specified alongside the arm-matched analysis above; both are reported, neither replaces the other.",
    "revfig07_model_comparisons": "AUROC forest and paired ΔAUROC of the integrated model against every single-modality comparator, with 95% paired cluster-bootstrap intervals.",
    "revfig08_fusion_weights": "Fold-wise distribution of the late-fusion modality weights and each modality's selection rate.",
    "supp_fig01_roc_curves": "Discovery-phase ROC curves.",
    "supp_fig02_performance_distributions": "Discovery-phase distribution of per-fold performance for every model.",
    "supp_fig03_fusion_benefit": "Discovery-phase fusion benefit against the best single modality.",
    "supp_fig04_forest_plot": "Discovery-phase forest plot of per-fold AUROC.",
    "supp_fig05_feature_shap_Global": "Discovery-phase SHAP attribution, pooled cohort.",
    "supp_fig05_feature_shap_DHP": "Discovery-phase SHAP attribution, DHP arm.",
    "supp_fig05_feature_shap_T_DM1": "Discovery-phase SHAP attribution, T-DM1 arm.",
    "supp_fig06_feature_selection_frequency": "Discovery-phase selection frequency of every candidate feature.",
    "supp_fig07_cross_scenario_features": "Features shared between the pooled and arm-specific signatures.",
    "supp_fig08_fusion_shap": "SHAP attribution of the five modality streams inside the fusion layer.",
    "supp_fig09_modality_weights": "Discovery-phase modality weights.",
    "supp_fig10_winner_classifier_heatmap": "Which classifier family won each fold, by modality and scenario.",
    "supp_fig11_inner_auroc_comparison": "Inner-cross-validation AUROC of each classifier family, the basis of the Stage A choice.",
    "supp_fig12_calibration_profile": "Discovery-phase calibration profile.",
    "supp_fig13_signature_sizes": "Distribution of discovered signature sizes across folds.",
    "supp_fig14_performance_CI": "Discovery-phase AUROC with patient-level cluster-bootstrap intervals — the fully nested estimates, free of consensus selection optimism.",
}

M = []          # markdown accumulator
def w(s=""):
    M.append(s)


# =============================================================== header =====
prov = json.loads((REPO / "results" / "run_provenance.json").read_text(encoding="utf-8-sig"))
P = prov["parameters"]

w("# Results")
w()
w("Every table on this page is generated directly from the deposited workbooks "
  "under [`report/tables/`](report/tables) by "
  "[`docs/build_RESULTS_md.py`](docs/build_RESULTS_md.py), and every figure is the "
  "PNG rendering of the corresponding PDF in [`report/figures/`](report/figures). "
  "Nothing here is typed by hand.")
w()
w(f"Pipeline `{prov['pipeline_version']}`, seed {prov['random_seed']}. "
  f"Regenerated {date.today().isoformat()}.")
w()
w("> **How to read every number below.** In each cross-validation repeat every "
  "patient has exactly one out-of-fold prediction; the metric is computed on "
  "that complete out-of-fold vector and averaged over the repeats (200 pooled, "
  "100 per arm). The 95% interval is a patient-level **cluster** bootstrap — "
  "2,000 stratified resamples of patients, a resampled patient carrying all of "
  "its repeat predictions. Predictions are never averaged across repeats or "
  "across models. A comparison whose interval for ΔAUROC includes zero is "
  "reported as *not distinguishable*, however large the point difference.")
w()
w("## Contents")
w()
for i, t in enumerate([
        "Design and cohort", "Cross-validated performance",
        "Is integration better than the best single modality?",
        "Calibration", "Events per variable",
        "Feature-selection stability", "Consensus signatures and fusion weights",
        "External validation", "Figures"], 1):
    w(f"{i}. [{t}](#{i}-{t.lower().replace(' ', '-').replace('?', '').replace(',', '')})")
w()

# ===================================================== 1. design & cohort ====
perf = sheet(TAB / "revision" / "revision_performance_CI.xlsx",
             "Performance_patient_CI", "scenario")
cons = perf[perf["source"] == "consensus"]
w("## 1. Design and cohort")
w()
rows = []
for sc in SC:
    r = cons[(cons["scenario"] == sc) & (cons["model"] == "Fused_ElasticNet")].iloc[0]
    rows.append([SCL[sc], cnt(r["n_patients"]), cnt(r["n_events"]),
                 f"{float(r['n_events']) / float(r['n_patients']):.1%}",
                 cnt(r["n_cv_repeats"]), cnt(r["n_outer_folds"])])
w(table(rows, ["Cohort", "Patients", "pCR events", "pCR rate",
               "CV repeats", "Outer evaluations"]))
w()
w(table([
    ["Outer resampling", f"stratified {P['outer_folds_global']}-fold "
                         f"`RepeatedStratifiedKFold` (no shuffle-split)"],
    ["Inner resampling", f"{P['inner_folds_global']}-fold (pooled), "
                         f"{P['inner_folds_arm']}-fold (per arm)"],
    # run 5: 110 metrics, TIER1_REMOVE lists 21 / 18 present (was "112 → 101")
    ["Candidate panel", "110 pre-defined metrics → 92 after the outcome-blind "
                        "biological deduplication"],
    ["Feature screen", f"in-fold Mann–Whitney AUROC, BH q ≤ {P['univ_fdr_q']}, "
                       f"keep {P['univ_min_k']}–{P['univ_max_k']}"],
    ["Classifier families", ", ".join(f"`{c}`" for c in P["classifiers"])],
    ["Signature size cap", "at least 5 pCR events per selected variable"],
    ["Fusion", "elastic-net logistic regression (l1_ratio 0.5) over the five "
               "Platt-calibrated modality probability streams"],
    ["Consensus finalisation", f"features above the stability threshold "
                               f"({P['stability_thresh_global']} pooled, "
                               f"{P['stability_thresh_arm']} per arm); modal classifier"],
    # run 5 introduced signature_source; read from provenance rather than typed
    ["Signature aggregation", f"`{P.get('signature_source', 'all_folds')}`"
                              + (" — aggregated only over the outer folds the "
                                 "modal classifier won, so the reported "
                                 "classifier and signature are one model"
                                 if P.get("signature_source") == "winner_folds"
                                 else "")],
    ["Training cohort", f"`{P.get('training_data', 'cc_only')}`"
                        + (" — each modality trains on every patient carrying "
                           "it; evaluation is on the complete cases only"
                           if P.get("training_data") == "expanded" else "")],
    ["Random seed", str(prov["random_seed"])],
], ["Design element", "Value"]))
w()
w("The full design is drawn in "
  "[`docs/ED_Fig11a_CV_schematic.pdf`](docs/ED_Fig11a_CV_schematic.pdf) and stated "
  "in [`docs/methods_cv_statement.txt`](docs/methods_cv_statement.txt), both "
  "generated from the run's own parameters.")
w()

# ===================================================== 2. performance ========
w("## 2. Cross-validated performance")
w()
w("Consensus models — the frozen signature and classifier re-evaluated on the "
  "same outer splits. Source: `report/tables/revision/revision_performance_CI.xlsx`.")
w()
for sc in SC:
    d = cons[cons["scenario"] == sc]
    w(f"### {SCL[sc]}")
    w()
    rows = []
    for m in ORDER:
        r = d[d["model"] == m]
        if r.empty:
            continue
        r = r.iloc[0]
        rows.append([MOD[m],
                     f"**{num(r['AUROC'])}**" if m == "Fused_ElasticNet" else num(r["AUROC"]),
                     ci(r["AUROC_CI_low"], r["AUROC_CI_high"]),
                     num(r["AUPRC"]), ci(r["AUPRC_CI_low"], r["AUPRC_CI_high"]),
                     num(r["Brier"]), ci(r["Brier_CI_low"], r["Brier_CI_high"])])
    w(table(rows, ["Model", "AUROC", "95% CI", "AUPRC", "95% CI", "Brier", "95% CI"]))
    w()

disc = perf[perf["source"] == "discovery"]
w("### Discovery phase (fully nested)")
w()
w("The signature and classifier are re-selected independently inside every fold, "
  "so these estimates carry no consensus selection optimism. They are the "
  "conservative reading of the same data.")
w()
rows = []
for sc in SC:
    d = disc[disc["scenario"] == sc]
    for m in ORDER:
        r = d[d["model"] == m]
        if r.empty:
            continue
        r = r.iloc[0]
        c = cons[(cons["scenario"] == sc) & (cons["model"] == m)]
        gap = (float(c.iloc[0]["AUROC"]) - float(r["AUROC"])) if not c.empty else None
        rows.append([SCL[sc], MOD[m].replace("**", ""), num(r["AUROC"]),
                     ci(r["AUROC_CI_low"], r["AUROC_CI_high"]),
                     f"{gap:+.3f}" if gap is not None else "—"])
w(table(rows, ["Cohort", "Model", "Discovery AUROC", "95% CI",
               "Consensus − discovery"]))
w()

# ===================================================== 3. comparisons =======
cmp_ = sheet(TAB / "revision" / "revision_model_comparisons.xlsx",
             "Model_comparisons", "scenario")
fb = sheet(TAB / "revision" / "revision_model_comparisons.xlsx",
           "Fusion_benefit", "scenario")
cc = cmp_[cmp_["source"] == "consensus"]
w("## 3. Is integration better than the best single modality?")
w()
# The headline verdict was typed as a literal "No." until run 5. It is now read
# off the workbook's own verdict column, so it can never contradict the table
# printed directly beneath it.
_fbc = fb[fb["source"] == "consensus"]
_verdicts = {str(v).strip().lower() for v in _fbc["verdict_vs_best_unimodal"]}
_headline = ("**No.**" if _verdicts <= {"not distinguishable"}
             else "**Not in every scenario** — see the verdict column.")
w(f"{_headline} Paired patient-level cluster bootstrap, the same patient "
  "resample applied to both models and all repeats.")
w()
rows = []
for sc in SC:
    r = fb[(fb["scenario"] == sc) & (fb["source"] == "consensus")]
    if r.empty:
        continue
    r = r.iloc[0]
    rows.append([SCL[sc], num(r["fused_AUROC"]),
                 f"{MOD.get(r['best_unimodal'], r['best_unimodal'])} "
                 f"{num(r['best_unimodal_AUROC'])}".replace("**", ""),
                 f"{float(r['delta_vs_best_unimodal']):+.3f}",
                 str(r["delta_vs_best_CI"]),
                 # run 5 renamed this column: the P value is marginal on the
                 # comparator that was *selected* as best, which is what the
                 # longer name records
                 pval(col(r, "delta_vs_best_p_marginal_selected_comparator",
                          "delta_vs_best_p_bootstrap")),
                 str(r["verdict_vs_best_unimodal"])])
w(table(rows, ["Cohort", "Integrated AUROC", "Best single modality", "Δ AUROC",
               "95% CI", "P", "Verdict"]))
w()
w("Against every comparator:")
w()
rows = []
for sc in SC:
    d = cc[cc["scenario"] == sc]
    for _, r in d.iterrows():
        rows.append([SCL[sc], MOD.get(r["comparator"], r["comparator"]).replace("**", ""),
                     f"{float(r['delta_AUROC']):+.3f}",
                     ci(r["delta_CI_low"], r["delta_CI_high"]),
                     pval(r["p_bootstrap"]), pval(r["q_bootstrap_BH"]),
                     str(r["verdict"]).replace("Fused_ElasticNet", "integrated")])
w(table(rows, ["Cohort", "Integrated vs", "Δ AUROC", "95% CI", "P", "BH q", "Verdict"]))
w()
w("DeLong's test computed per repeat and summarised is reported in the workbook "
  "as a descriptive secondary analysis; the bootstrap is the primary comparison.")
w()

# ======================================================= 4. calibration =====
cal = sheet(TAB / "revision" / "revision_calibration.xlsx",
            "Calibration_summary", "scenario")
rel = sheet(TAB / "revision" / "revision_calibration.xlsx",
            "Reliability_bins", "scenario")
w("## 4. Calibration")
w()
w("Slope and intercept of `logit(pCR) = a + b · logit(p̂)`, fitted on each "
  "repeat's out-of-fold vector and averaged. Slope 1 and intercept 0 are perfect; "
  "slope below 1 means the predictions are too extreme (the classic overfitting "
  "signature), above 1 that they are compressed toward the base rate.")
w()
rows = []
_covers = True
for sc in SC:
    r = cal[cal["scenario"] == sc].iloc[0]
    # run 5 renamed the intercept columns to `recalibration_intercept*`
    _b = col(r, "recalibration_intercept", "calibration_intercept")
    _blo = col(r, "recalibration_intercept_CI_low", "intercept_CI_low")
    _bhi = col(r, "recalibration_intercept_CI_high", "intercept_CI_high")
    if not (_f(r["slope_CI_low"]) <= 1 <= _f(r["slope_CI_high"])
            and _f(_blo) <= 0 <= _f(_bhi)):
        _covers = False
    rows.append([SCL[sc], num(r["calibration_slope"], 2),
                 ci(r["slope_CI_low"], r["slope_CI_high"], 2),
                 num(_b, 2), ci(_blo, _bhi, 2),
                 num(r["brier"]), num(r["ECE"]),
                 f"{float(r['observed_pCR_rate']):.3f} vs {float(r['mean_predicted']):.3f}"])
w(table(rows, ["Cohort", "Slope", "95% CI", "Intercept", "95% CI", "Brier", "ECE",
               "Observed vs mean predicted"]))
w()
# derived, not typed: an interval that stopped covering its null would otherwise
# leave a sentence asserting the opposite of the table above it
w("Every slope interval covers 1 and every intercept interval covers 0."
  if _covers else
  "Not every slope interval covers 1 or every intercept interval covers 0 — "
  "read the two CI columns above.")
w()
w("<details><summary>Reliability bins (equal-count bins over all "
  "(patient, repeat) out-of-fold predictions)</summary>")
w()
rows = [[SCL[r["scenario"]], cnt(r["bin"]), cnt(r["n_rows"]),
         cnt(r["n_patients_distinct"]), num(r["mean_predicted"]),
         num(r["observed"]), ci(r["obs_ci_low"], r["obs_ci_high"])]
        for _, r in rel.iterrows()]
w(table(rows, ["Cohort", "Bin", "Predictions", "Distinct patients",
               "Mean predicted", "Observed", "95% CI"]))
w()
w("</details>")
w()

# ============================================================== 5. EPV ======
epv = sheet(TAB / "revision" / "revision_epv_per_fold.xlsx", "EPV_summary", "scenario")
w("## 5. Events per variable")
w()
w("The design caps signature size at five pCR events per selected variable. "
  "This table reports what was actually realised in each fold.")
w()
rows = []
for _, r in epv.iterrows():
    rows.append([SCL.get(r["scenario"], r["scenario"]),
                 MOD.get(r["model"], r["model"]).replace("**", ""),
                 cnt(r["n_folds"]),
                 f"{cnt(r['median_n_events_test'])} ({cnt(r['min_n_events_test'])}–{cnt(r['max_n_events_test'])})",
                 cnt(r["median_signature_size"]), num(r["median_epv_realized"], 2),
                 num(r["min_epv_realized"], 2),
                 f"{float(r['pct_folds_epv_below_5']):.1f}%"])
w(table(rows, ["Cohort", "Model", "Folds", "Test-fold events (median, range)",
               "Median signature size", "Median EPV", "Min EPV", "Folds below EPV 5"]))
w()
# run 5: DHP 39.2% and T-DM1 56.0% of fusion folds below EPV 5 (was "a third"
# and "almost half"). Read from the table so the sentence tracks the numbers.
_fus = epv[epv["model"] == "Fused_ElasticNet"].set_index("scenario")
_pdhp = _f(_fus.loc["DHP", "pct_folds_epv_below_5"])
_ptdm = _f(_fus.loc["T-DM1", "pct_folds_epv_below_5"])
_others = epv[(epv["model"] != "Fused_ElasticNet")
              & (epv["pct_folds_epv_below_5"].map(_f) > 0)]
w("The arm-level fusion layer is the most exposed component: it takes five "
  "modality inputs by design and cannot be capped, so "
  f"{_pdhp:.0f}% of DHP folds and {_ptdm:.0f}% of T-DM1 folds run below five "
  "events per variable."
  + (" Every single-modality model, in every scenario, stays at or above the "
     "cap in every fold." if _others.empty else
     f" {len(_others)} single-modality rows also fall below it; see the table."))
w()

# ======================================================== 6. stability ======
stab = sheet(TAB / "revision" / "revision_stability.xlsx",
             "Feature_selection_stability", "scenario")
mws = sheet(TAB / "revision" / "revision_stability.xlsx",
            "Modality_weight_stability", "scenario")
stable = stab[stab["stable"].astype(str).str.lower().isin(["true", "yes", "1"])]
w("## 6. Feature-selection stability")
w()
w(f"How often each candidate feature was selected across the outer folds. "
  f"Features above the pre-specified threshold ({P['stability_thresh_global']} "
  f"pooled, {P['stability_thresh_arm']} per arm) are the consensus signature; "
  f"{len(stable)} of {len(stab)} candidate rows clear it.")
w()
w("**The threshold is applied to the *eligible-fold* frequency** — the fraction "
  "of the folds in which the feature survived preprocessing and the in-fold "
  "screen at all. A feature can therefore be stable on that denominator while "
  "its all-fold frequency is low: it was rarely eligible, but was chosen almost "
  "whenever it was. Both columns are given below, with the Wilson interval on "
  "the eligible-fold proportion.")
w()
for sc in SC:
    d = stable[stable["scenario"] == sc]
    if d.empty:
        continue
    w(f"<details><summary><b>{SCL[sc]}</b> — {len(d)} stable features</summary>")
    w()
    rows = [[MOD.get(r["modality"], r["modality"]).replace("**", ""),
             f"`{r['feature']}`", num(r["selection_freq"]),
             num(r["selection_freq_eligible"]),
             ci(r["wilson_low"], r["wilson_high"]),
             cnt(r["n_selected"]) + " / " + cnt(r["n_folds_total"])]
            for _, r in d.sort_values(
                ["modality", "selection_freq_eligible", "selection_freq"],
                ascending=[True, False, False]).iterrows()]
    w(table(rows, ["Modality", "Feature", "All-fold frequency",
                   "Eligible-fold frequency", "95% Wilson CI", "Folds selected"]))
    w()
    w("</details>")
    w()
w("### Stability of the fusion weights")
w()
rows = []
for _, r in mws.iterrows():
    rows.append([SCL.get(r["scenario"], r["scenario"]),
                 MOD.get(r["modality"], r["modality"]).replace("**", ""),
                 num(r["mean_weight"], 2), num(r["median_weight"], 2),
                 pct(r["selection_rate"]),
                 ci(r["selection_rate_ci_low"], r["selection_rate_ci_high"], 2),
                 num(r["sign_consistency"], 2)])
w(table(rows, ["Cohort", "Modality", "Mean weight", "Median weight",
               "Selection rate", "95% CI", "Sign consistency"]))
w()

# ===================================================== 7. signatures ========
sig = sheet(TAB / "PREDIX_HER2_results.xlsx", "Signatures", "Scenario")
fus = sheet(TAB / "PREDIX_HER2_results.xlsx", "Fusion", "Scenario")
w("## 7. Consensus signatures and fusion weights")
w()
for sc in SC:
    d = sig[sig["Scenario"] == sc]
    if d.empty:
        continue
    w(f"### {SCL[sc]}")
    w()
    rows = []
    for m in ORDER[:-1]:
        dm = d[d["Modality"] == m].sort_values("Rank")
        if dm.empty:
            continue
        feats = ", ".join(f"`{f}`" for f in dm["Feature"])
        r0 = dm.iloc[0]
        rows.append([MOD[m], cnt(r0["Signature size (K)"]),
                     f"`{r0['Winner classifier']}`",
                     f"{float(r0['Classifier support (%)']):.0f}%", feats])
    w(table(rows, ["Modality", "K", "Winning classifier", "Fold support",
                   "Signature (in rank order)"]))
    w()
w("### Late-fusion modality weights")
w()
rows = []
for sc in SC:
    d = fus[fus["Scenario"] == sc].sort_values("Mean fusion coefficient",
                                               ascending=False)
    for _, r in d.iterrows():
        rows.append([SCL[sc], MOD.get(r["Modality"], r["Modality"]).replace("**", ""),
                     num(r["Mean fusion coefficient"], 2),
                     num(r["SD fusion coefficient"], 2),
                     pct(r["Selection rate (|coef| > 1e-6)"])])
w(table(rows, ["Cohort", "Modality", "Mean coefficient", "SD", "Selection rate"]))
w()

# ================================================== 8. external validation ==
# Run 5 ships TWO pre-specified external analyses. The arm-matched one refits the
# frozen model on the PREDIX arm whose regimen the external cohort resembles; the
# pooled one refits it on every PREDIX patient carrying transcriptomics. Run 4
# wrote both under identical basenames and only one was ever read, so run 5 gives
# the pooled analysis its own tree and the `_POOLED` suffix. Both are rendered
# here, in that order, and neither replaces the other.
ext = sheet(TAB / "revision" / "external_validation.xlsx",
            "External_validation", "cohort")
ext_pool_path = POOL_TAB / "revision" / "external_validation_POOLED.xlsx"
ext_pool = (sheet(ext_pool_path, "External_validation", "cohort")
            if ext_pool_path.exists() else None)
w("## 8. External validation")
w()
w("The pipeline's own transcriptomic consensus model was **frozen** — signature, "
  "classifier and hyper-parameters — refit once on PREDIX with no grid search, "
  "and applied to the external cohort. Nothing was refitted on external data. "
  "Both harmonisation schemes are reported so that a result present under only "
  "one would be identified as an artefact of that scheme.")
w()
w("Two refit populations were pre-specified and both are reported below: "
  "**arm-matched** (the model is refit on the PREDIX arm whose regimen the "
  "external cohort resembles) and **pooled** (refit on every PREDIX patient "
  "carrying transcriptomics, irrespective of arm). They answer different "
  "questions — arm-specific transfer, and transfer of one general model — so "
  "neither substitutes for the other.")
w()


def ext_table(df, heading, source):
    w(f"### {heading}")
    w()
    w(f"Source: `{source}`.")
    w()
    rows = []
    for _, r in df.iterrows():
        rows.append([str(r["cohort"]),
                     # run 5 replaced `matched_PREDIX_arm` with a pair of
                     # columns, because in the pooled design the arm the cohort
                     # resembles is NOT the population the model was refit on
                     str(col(r, "model_refit_population", "matched_PREDIX_arm")),
                     str(r["harmonisation"]),
                     cnt(r["n_external"]), cnt(r["events_external"]),
                     str(r["internal_AUROC_CI"]),
                     f"**{str(r['external_AUROC_CI'])}**",
                     str(r["external_AUPRC_CI"]), str(r["external_Brier_CI"]),
                     f"{num(r['calibration_slope'], 2)} ({r['calibration_slope_CI']})",
                     pval(r["p_vs_chance_one_sided"])])
    w(table(rows, ["Cohort", "Refit on", "Harmonisation", "n", "pCR",
                   "Internal AUROC", "External AUROC", "AUPRC", "Brier",
                   "Calibration slope", "P vs chance"]))
    w()
    w("Locked specifications:")
    w()
    seen = set()
    rows = []
    for _, r in df.iterrows():
        k = str(r["cohort"])
        if k in seen:
            continue
        seen.add(k)
        rows.append([str(r["cohort_description"]),
                     str(col(r, "cohort_resembles_PREDIX_arm",
                             "matched_PREDIX_arm")),
                     str(col(r, "model_refit_population", "matched_PREDIX_arm")),
                     f"`{r['classifier']}` {r['hyperparameters']}",
                     cnt(r["n_model_features"]),
                     f"{cnt(r['n_PREDIX_train'])} / {cnt(r['events_PREDIX_train'])}"])
    w(table(rows, ["External cohort", "Resembles PREDIX arm", "Refit on",
                   "Frozen classifier", "Features", "Refit on (n / events)"]))
    w()


ext_table(ext, "Arm-matched models",
          "report/tables/revision/external_validation.xlsx")
if ext_pool is not None:
    ext_table(ext_pool, "Pooled model",
              "report_pooled_external/tables/revision/"
              "external_validation_POOLED.xlsx")

# The verdict sentence is assembled from the workbooks. Up to run 4 it was typed,
# and it said the T-DM1 model "does not transfer" — a claim run 5 contradicts
# (arm-matched NCT02326974 AUROC 0.644, P = 0.003; pooled 0.669, P < 0.001).
_bits = []
for _lbl, _df in [("arm-matched", ext)] + ([("pooled", ext_pool)]
                                           if ext_pool is not None else []):
    for _c in _df["cohort"].unique():
        _d = _df[_df["cohort"] == _c]
        _lo = min(_f(v) for v in _d["external_AUROC"])
        _hi = max(_f(v) for v in _d["external_AUROC"])
        _p = max(_f(v) for v in _d["p_vs_chance_one_sided"])
        _rng = f"{_lo:.3f}" if abs(_hi - _lo) < 5e-4 else f"{_lo:.3f}–{_hi:.3f}"
        _bits.append(f"{_c} {_rng} ({_lbl}, worst-case P {pval(_p)})")
_all = [_f(v) for _df in ([ext] + ([ext_pool] if ext_pool is not None else []))
        for v in _df["p_vs_chance_one_sided"]]
_scope = ("under both refit populations and both harmonisation schemes"
          if ext_pool is not None else "under both harmonisation schemes")
w((f"**Both external cohorts discriminate above chance, {_scope}.**"
   if max(_all) < 0.05 else
   "**Not every external estimate is distinguishable from chance.**")
  + " AUROC across the harmonisation schemes: " + "; ".join(_bits) + ".")
w()
w("Calibration is the honest qualifier, and it is reported separately from "
  "discrimination for exactly that reason: a frozen model can rank patients "
  "usefully in a cohort whose base rate and spread it mis-states. Read the "
  "calibration-slope column above — below 1 means the probabilities are more "
  "extreme than the cohort warrants, above 1 that they are compressed toward "
  "the base rate — and note that the two refit populations do not calibrate the "
  "same way in the same cohort. No result is withheld on calibration grounds "
  "and none is presented as though calibration were settled.")
w()

# ============================================================ 10. figures ===
PNG_DST.mkdir(parents=True, exist_ok=True)
copied = 0
for p in sorted(PNG_SRC.glob("*.png")):
    shutil.copy2(p, PNG_DST / p.name)
    copied += 1

w("## 9. Figures")
w()
w("PNG renderings at 170 dpi; the citable vector versions are the PDFs in "
  "[`report/figures/`](report/figures) and "
  "[`report_pooled_external/figures/`](report_pooled_external/figures).")
w()


def fig_block(title, stems, note=""):
    w(f"### {title}")
    w()
    if note:
        w(note)
        w()
    for s in stems:
        f = PNG_DST / f"{s}.png"
        if not f.exists():
            continue
        w(f"#### {s}")
        w()
        w(f"![{s}](report/figures_png/{s}.png)")
        w()
        w(f"*{FIG_CAPTION.get(s, '')}*")
        w()


main = [s for s in FIG_CAPTION if s.startswith("fig")]
rev = [s for s in FIG_CAPTION if s.startswith("revfig")]
supp = [s for s in FIG_CAPTION if s.startswith("supp_fig")]
fig_block("Main figures", main)
fig_block("Revision figures", rev,
          "Calibration, stability, events per variable, external validation, "
          "paired comparisons and fusion weights — the diagnostics added in "
          "this revision.")
fig_block("Supplementary figures — discovery phase", supp,
          "Diagnostics of the fully nested discovery phase, before consensus "
          "finalisation.")

# The PNGs are copied by a blind glob but rendered from FIG_CAPTION, so the two
# can drift apart in either direction and both failures are silent:
#   * a PNG with no caption ships but never appears on the page — how the pooled
#     external figure was nearly lost in run 5;
#   * a caption with no PNG is a withdrawn or renamed analysis lying in wait for
#     a stray file to resurrect it (the withdrawn S1/S2/S3 biomarker-group
#     figure was exactly this).
# Neither is fatal to the build, so report both loudly instead.
orphan_png = sorted(p.stem for p in PNG_SRC.glob("*.png") if p.stem not in FIG_CAPTION)
dead_caption = sorted(s for s in FIG_CAPTION if not (PNG_DST / f"{s}.png").exists())

w("---")
w()
w("Regenerate this page with `python docs/build_RESULTS_md.py`.")

OUT.write_text("\n".join(M) + "\n", encoding="utf-8", newline="\n")
shutil.copy2(Path(__file__), REPO / "docs" / "build_RESULTS_md.py")

# The manifest must cover the files just added. Same rule as Section 14 of the
# notebook (the notebook and the manifest itself excluded, no header lines), so
# that re-running the notebook regenerates it byte-identically.
import hashlib
man = sorted((p for p in REPO.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and "_regenerated" not in p.parts and p.suffix != ".pyc"
              and not p.name.startswith((".~lock", "~$"))
              and p.name not in ("MANIFEST_SHA256.txt",
                                 "PREDIX_HER2_reproducibility.ipynb")),
             key=lambda p: p.relative_to(REPO).as_posix())


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


(REPO / "MANIFEST_SHA256.txt").write_text(
    "\n".join(f"{_sha(p)}  {p.relative_to(REPO).as_posix()}" for p in man) + "\n",
    encoding="utf-8", newline="\n")
print(f"  MANIFEST_SHA256.txt regenerated: {len(man)} files")
text = "\n".join(M)
n_tables = sum(1 for line in text.splitlines() if line.startswith("|---"))
n_figs = sum(1 for line in text.splitlines() if line.startswith("!["))
n_rows = sum(1 for line in text.splitlines()
             if line.startswith("|") and not line.startswith("|---"))
print(f"wrote {OUT.relative_to(ROOT)}  "
      f"{OUT.stat().st_size / 1024:.0f} KB, {len(M)} lines")
print(f"  {n_tables} tables ({n_rows - n_tables} data rows), "
      f"{n_figs} figures embedded, {copied} PNGs copied")
if orphan_png:
    print(f"  WARNING  {len(orphan_png)} PNG(s) shipped with no FIG_CAPTION entry, "
          f"so they never render: {orphan_png}")
if dead_caption:
    print(f"  WARNING  {len(dead_caption)} FIG_CAPTION entr(ies) with no PNG: "
          f"{dead_caption}")
if ext_pool is None:
    print("  WARNING  report_pooled_external/ is absent, so the pooled external "
          "validation is NOT on this page — rebuild with build_github_repo.py")
