#!/usr/bin/env python3
"""
Extended Data Fig. 11a — schematic of the analysis actually implemented.

The submitted panel a depicts 100 shuffle-split iterations and names AdaBoost
and CatBoost; the pipeline has never used either. This script redraws the panel
from the run's own provenance record, so every count in the figure is read from
the analysis rather than typed by hand.

    python revision_deliverables/build_ED_Fig11a_schematic.py

Writes ED_Fig11a_CV_schematic.{pdf,png} next to the revised manuscript.

WHICH STAGES ARE DRAWN IS READ FROM THE PIPELINE, NOT FROM THE CLI ARGUMENTS.
Until 2026-08-20 the in-fold "correlation pruning |r| > 0.9 (RNA, DNA)" box was
drawn whenever `parameters.corr_threshold` appeared in run_provenance.json. That
argument still has a value in every run, but the stage it configures was DELETED
in run 4: the pipeline sets CORR_FILTER_MODS = set(), and the per-fold call site
`ac_map = {m: (m in CORR_FILTER_MODS) for m in ALL_MODS}` therefore passes
apply_corr=False for every modality, so remove_high_correlation() never executes.
A CLI default is not evidence that a stage ran. Every optional stage in this
figure is now decided by the module constant that actually gates it, parsed out
of the pipeline source with `ast` — importing the pipeline would drag in
sklearn/shap and set BLAS environment variables, which a figure script must not
do. Feature redundancy is instead removed ONCE, before any fold, by the fixed
outcome-blind TIER1_REMOVE list drawn in the candidate-panel box.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
# Production run. Every parameter drawn in the figure comes from this run's own
# provenance record, so this constant decides which analysis the figure depicts.
# It pointed at run 3 until 2026-08-20, four runs after production moved on.
RUN = ROOT / "ubuntu_results_run5"
PROV = RUN / "results" / "run_provenance.json"
DATA = ROOT / "clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
PIPE = ROOT / "multimodal_pcr_pipeline.py"
OUT = Path(__file__).resolve().parent / "2_revised_manuscript" / "ED_Fig11a_CV_schematic"

for _p in (PROV, DATA, PIPE):
    if not _p.exists():
        raise SystemExit(f"missing input: {_p}\nThe figure is drawn entirely "
                         f"from the run's own records; it is not redrawn from "
                         f"remembered numbers. Point RUN at the production run.")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.4, "savefig.bbox": "tight", "savefig.dpi": 400,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# --- everything quantitative is read from the run, not written here ----------
prov = json.loads(PROV.read_text(encoding="utf-8"))
P = prov["parameters"]
CV = prov["cv_design"]

import pandas as pd
_cols = list(pd.read_csv(DATA, sep="\t", nrows=1).columns)
_feat = [c for c in _cols if c not in ("patientID", "pCR")]
MOD_N = {m: sum(1 for c in _feat if c.startswith(m + "_")) for m in P["modalities"]}

# Pipeline constants are read from the source, never retyped here. `ast` reads
# the literals without importing the modelling stack; utf-8-sig tolerates a BOM.
import ast
_src = PIPE.read_text(encoding="utf-8-sig")
_tree = ast.parse(_src)


def _empty_container(node):
    """ast.literal_eval cannot evaluate `set()` — it is a Call, not a literal,
    and CORR_FILTER_MODS is written exactly that way. Resolve the no-argument
    builtin containers by hand and defer everything else to literal_eval."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and not node.args and not node.keywords
            and node.func.id in ("set", "frozenset", "list", "tuple", "dict")):
        return {"set": set(), "frozenset": frozenset(),
                "list": [], "tuple": (), "dict": {}}[node.func.id]
    return ast.literal_eval(node)


def _pipeline_const(name):
    """Value of a module-level constant of the pipeline, with the two ways this
    parse can silently lie guarded: more than one binding at module level, and
    in-place mutation after the binding (run 6 appends to TIER1_REMOVE under
    KEEP_RNA_FCGR3B, and a future run could just as easily .add() a modality
    back into CORR_FILTER_MODS)."""
    assigns = [n for n in _tree.body
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name
                       for t in n.targets)]
    if len(assigns) != 1:
        raise SystemExit(f"expected exactly one module-level {name} binding in "
                         f"{PIPE}, found {len(assigns)}. The figure would "
                         f"depict a stage configuration the run did not use.")
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and isinstance(n.func.value, ast.Name)
           and n.func.value.id == name for n in ast.walk(_tree)):
        raise SystemExit(f"{PIPE} mutates {name} after defining it (run 6 "
                         f"appends RNA_FCGR3B to TIER1_REMOVE under "
                         f"KEEP_RNA_FCGR3B). This script reads the literal "
                         f"only, so the figure would be wrong. Resolve the "
                         f"flag and update this parse.")
    try:
        return _empty_container(assigns[0].value)
    except ValueError as exc:
        raise SystemExit(f"{name} in {PIPE} is not a literal this script can "
                         f"read without importing the pipeline: {exc}")


_tier1 = list(_pipeline_const("TIER1_REMOVE"))
if not _tier1 or not all(isinstance(f, str) for f in _tier1):
    raise SystemExit(f"TIER1_REMOVE parsed from {PIPE} is not a list of names")
N_DEDUP = sum(1 for f in _tier1 if f in _cols)
N_PANEL, N_CAND = len(_feat), len(_feat) - N_DEDUP

# --- which OPTIONAL in-fold stages actually ran ------------------------------
# CORR_FILTER_MODS gates Tier 3 (see the module docstring): empty set = the
# per-fold correlation filter is dead code for every modality, whatever
# --corr_threshold was passed. CONSENSUS_DEDUP_MODS is a different stage that
# survived run 4 and runs once, at the consensus step, not per fold.
CORR_MODS = set(_pipeline_const("CORR_FILTER_MODS"))
DEDUP_MODS = set(_pipeline_const("CONSENSUS_DEDUP_MODS"))
SIG_SOURCE = _pipeline_const("SIGNATURE_SOURCE")
UNIV_MIN_FEATURES = _pipeline_const("UNIV_SCREEN_MIN_FEATURES")
UNIV_ON = _pipeline_const("UNIVARIATE_SCREEN")

# CORR_FILTER_MODS has NO CLI flag, so run_provenance.json cannot corroborate
# it (run 6 adds an `analysis_constants` block for exactly this reason; run 5
# has none). The only evidence that ROOT's pipeline is the code that produced
# RUN is that it is byte-identical to the hand-off copy shipped with that run —
# check it, rather than assume it, because a stage with no flag is precisely the
# kind of thing that can change under a figure without the figure noticing.
_handoff = ROOT / f"predix{RUN.name[len('ubuntu_results'):]}_ubuntu" / PIPE.name
if RUN.name != "ubuntu_results" and _handoff.exists():
    if _handoff.read_bytes() != PIPE.read_bytes():
        raise SystemExit(
            f"{PIPE} differs from {_handoff}, the pipeline actually shipped "
            f"for {RUN.name}. Constants with no CLI flag — CORR_FILTER_MODS, "
            f"SIGNATURE_SOURCE, UNIVARIATE_SCREEN — are read from the former "
            f"and are not recorded in run_provenance.json, so the figure could "
            f"depict stages this run never executed. Reconcile the two files "
            f"or point PIPE at the hand-off copy.")
    _pipe_prov = f"identical to {_handoff.parent.name}/{PIPE.name}"
else:
    _pipe_prov = f"NOT CHECKED — no hand-off copy found for {RUN.name}"

# The provenance record and the source must agree, or the figure is drawn from
# code that is not the code that produced RUN.
if prov["parameters"].get("signature_source") != SIG_SOURCE:
    raise SystemExit(
        f"{RUN.name} ran with signature_source="
        f"{prov['parameters'].get('signature_source')!r} but {PIPE.name} now "
        f"says {SIG_SOURCE!r}. The pipeline has moved on since this run; point "
        f"RUN at the matching results directory before redrawing the figure.")
if prov["leakage_control"]["univariate_screen"] != ("in_fold" if UNIV_ON
                                                    else "none"):
    raise SystemExit(
        f"{RUN.name} ran with univariate_screen="
        f"{prov['leakage_control']['univariate_screen']!r} but "
        f"{PIPE.name} sets UNIVARIATE_SCREEN={UNIV_ON!r}.")

# Complete case by the PIPELINE's rule (get_complete_case): Clin never enters
# the completeness definition. A dropna() over ALL features is the trap here —
# Clin_TUMSIZE and Clin_prolifvalu carry the string "Unknown" rather than NaN,
# so the naive version agrees today and would silently draw 104 the moment
# those tokens are encoded properly.
_full = pd.read_csv(DATA, sep="\t")
_mol = [c for c in _feat if c.split("_", 1)[0] in ("RNA", "DNA", "Prot", "WSI")]
_cc = _full.dropna(subset=_mol)
N_CC, N_EV = len(_cc), int(_cc["pCR"].sum())
_arm = "Clin_Arm"
_g = _cc.groupby(_arm)["pCR"].agg(["size", "sum"]).sort_index()
ARMS = [(str(v), int(r["size"]), int(r["sum"])) for v, r in _g.iterrows()]

# Candidates per modality after the one-off TIER1_REMOVE deduplication. Used to
# work out which modalities the Tier 2.5 screen skips: _resolve_screen_cfg()
# returns None when a modality carries <= UNIV_SCREEN_MIN_FEATURES columns, so
# the exemption is a threshold on the pool, not a hard-coded pair of names.
MOD_CAND = {m: MOD_N[m] - sum(1 for f in _tier1
                              if f in _cols and f.startswith(m + "_"))
            for m in P["modalities"]}
SCREEN_EXEMPT = [m for m in P["modalities"]
                 if MOD_CAND[m] <= UNIV_MIN_FEATURES]

OF, RG = P["outer_folds_global"], P["repeats_global"]
OA, RA = P["outer_folds_arm"], P["repeats_arm"]
IG, IA = P["inner_folds_global"], P["inner_folds_arm"]
EVAL_G, EVAL_A = CV["outer_evaluations_global"], CV["outer_evaluations_per_arm"]
CLFS = P["classifiers"]

# ---------------------------------------------------------------- palette ---
INK = "#1c2833"
MUTED = "#5d6d7e"
BOX_BG, BOX_EC = "#f4f6f7", "#aab7b8"
OUTER_BG, OUTER_EC = "#eaf2f8", "#5499c7"
INNER_BG, INNER_EC = "#fef5e7", "#d68910"
FUSE_BG, FUSE_EC = "#eafaf1", "#28b463"
SIG_BG, SIG_EC = "#fdedec", "#cd6155"
EST_BG, EST_EC = "#f5eef8", "#8e44ad"
TRAIN, TEST = "#aed6f1", "#e59866"
MOD_COL = {"Clin": "#95a5a6", "RNA": "#5499c7", "DNA": "#af7ac5",
           "Prot": "#48c9b0", "WSI": "#f0b27a"}
MOD_LAB = {"Clin": "Clinical", "RNA": "Transcriptomic", "DNA": "Genomic",
           "Prot": "Proteomic", "WSI": "WSI"}

fig, ax = plt.subplots(figsize=(7.28, 11.02))
ax.set_xlim(0, 100)
ax.set_ylim(0, 157)
ax.axis("off")
ax.invert_yaxis()


def box(x, y, w, h, fc, ec, lw=0.9, r=1.6, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw,
                                zorder=z, linestyle=ls))


def txt(x, y, s, size=7.4, weight="normal", color=INK, ha="center", va="center",
        style="normal", z=4, ls=1.45):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            style=style, zorder=z, linespacing=ls)


def arrow(x1, y1, x2, y2, color=MUTED, lw=1.0, z=3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, color=color, linewidth=lw,
                                 shrinkA=0, shrinkB=0, zorder=z))


def label(x, y, s):
    txt(x, y, s, size=7.0, weight="bold", color=MUTED, ha="left")


# ============================================================ 1. cohort =====
box(2, 2, 96, 12.0, BOX_BG, BOX_EC)
txt(50, 4.8, "PREDIX HER2 — patients with complete data across all five modalities", weight="bold")
_armtxt = "        ".join(f"{v}  n = {n} ({e} pCR)" for v, n, e in ARMS)
txt(50, 7.6, f"n = {N_CC}  ({N_EV} pCR)              {_armtxt}")
txt(50, 11.2, "each modality-specific model is trained on all patients in whom that modality was measured,\n"
              "minus the patients of the current test fold, and is evaluated only on the complete-case cohort",
    size=6.3, color=MUTED, style="italic")
arrow(50, 14.2, 50, 16.4)

# ==================================================== 2. candidate panel =====
box(2, 16.6, 96, 14.0, BOX_BG, BOX_EC)
txt(50, 19.2, "Candidate features — a-priori biological curation, no outcome used", weight="bold")
cw, gap = 17.2, 1.6
x0 = 50 - (5 * cw + 4 * gap) / 2
for i, m in enumerate(["Clin", "RNA", "DNA", "Prot", "WSI"]):
    x = x0 + i * (cw + gap)
    box(x, 21.2, cw, 5.2, MOD_COL[m], MOD_COL[m], lw=0, r=1.1)
    txt(x + cw / 2, 22.9, MOD_LAB[m], size=7.0, weight="bold", color="white")
    txt(x + cw / 2, 25.0, f"{MOD_N[m]} metrics", size=6.8, color="white")
txt(50, 28.8,
    f"{N_PANEL} pre-defined metrics   →   fixed outcome-blind deduplication of co-amplified and\n"
    f"near-identical features, applied once before any fold (−{N_DEDUP})   →   {N_CAND} candidates",
    size=6.9)
arrow(50, 30.8, 50, 33.0)

# ======================================================= 3. outer loop ======
box(2, 33.2, 96, 24.4, OUTER_BG, OUTER_EC, lw=1.1)
label(4.4, 36.0, f"OUTER LOOP — stratified {OF}-fold cross-validation, "
                 f"{CV['scheme']}, seed {prov['random_seed']}")
fw, fg = 15.6, 1.5
fx0 = 50 - (5 * fw + 4 * fg) / 2
for k in range(5):
    x = fx0 + k * (fw + fg)
    is_test = (k == 3)
    box(x, 38.6, fw, 4.6, TEST if is_test else TRAIN,
        "#ca6f1e" if is_test else "#5499c7", lw=0.7, r=1.0)
    txt(x + fw / 2, 40.9, "held-out\ntest fold" if is_test else "training",
        size=6.5, weight="bold" if is_test else "normal",
        color="#7e5109" if is_test else "#1a5276")
txt(50, 45.2, "each fold is held out in turn (~80 % train / ~20 % test), stratified on pCR",
    size=6.5, color=MUTED, style="italic")

box(4.4, 47.4, 91.2, 8.0, "white", OUTER_EC, lw=0.7, r=1.2)
txt(50, 49.8, f"the whole partition is redrawn  ×  {RG} repeats   "
              f"=   {EVAL_G:,} outer evaluations   (pooled cohort)", size=7.2, weight="bold")
txt(50, 53.3, f"arm-specific models: {OA}-fold × {RA} repeats = {EVAL_A} evaluations per arm,\n"
              f"with {IA}-fold inner cross-validation.   No shuffle-split resampling at any stage.",
    size=6.5, color=MUTED)
arrow(50, 57.8, 50, 60.0)

# =============================== 4. inside one outer training fold ==========
box(2, 60.2, 96, 49.6, INNER_BG, INNER_EC, lw=1.1)
label(4.4, 63.0, "WITHIN ONE OUTER TRAINING FOLD — fitted on training patients only")

box(4.4, 65.2, 43.4, 22.4, "white", INNER_EC, lw=0.7, r=1.2)
txt(26.1, 67.4, "Preprocessing, in this order", size=7.2, weight="bold")

# The stage list is assembled, not typed, so a stage that no longer runs cannot
# survive in the figure. Tier 3 is drawn only if CORR_FILTER_MODS is non-empty;
# in run 5 it is empty and the line is replaced by the explicit statement below,
# because a reader of the submitted panel will look for the stage it used to
# show and needs to be told it is gone rather than left to notice.
_exempt = ", ".join(MOD_LAB[m] if m == "WSI" else MOD_LAB[m].lower()
                    for m in SCREEN_EXEMPT)
_stages = ["near-zero-variance filter"]
if CORR_MODS:
    _stages.append(f"correlation pruning |r| > {P['corr_threshold']} "
                   f"({', '.join(sorted(CORR_MODS))})")
_stages += ["median imputation, then standardisation"]
if UNIV_ON:
    _stages += [
        "univariate screen: Mann–Whitney AUROC,",
        f"BH q ≤ {P['univ_fdr_q']}, keep {P['univ_min_k']}–{P['univ_max_k']}"
        f" candidates",
        f"({_exempt}: ≤ {UNIV_MIN_FEATURES} candidates, unscreened)",
    ]
txt(26.1, 69.8, "\n".join(_stages), size=6.3, va="top", ls=1.75)
if not CORR_MODS:
    txt(26.1, 85.4,
        "no in-fold correlation filter: redundancy is\n"
        "removed once, above, before any fold is drawn",
        size=6.1, color=MUTED, style="italic", ls=1.55)

box(52.2, 65.2, 43.4, 22.4, "white", INNER_EC, lw=0.7, r=1.2)
txt(73.9, 67.4, f"Inner {IG}-fold cross-validation", size=7.2, weight="bold")
txt(73.9, 69.8,
    "Stage A   the five classifier families below are\n"
    "compared under fixed grids; importances become\n"
    "percentile ranks and define the feature signature,\n"
    "capped at ≥ 5 pCR events per selected variable\n"
    "Stage B   the winning family is tuned by grid\n"
    "search, restricted to the discovered signature",
    size=6.3, va="top", ls=1.75)

cw2, g2 = 17.0, 1.4
cx0 = 50 - (len(CLFS) * cw2 + (len(CLFS) - 1) * g2) / 2
for i, c in enumerate(CLFS):
    x = cx0 + i * (cw2 + g2)
    box(x, 89.6, cw2, 4.0, "white", "#b9770e", lw=0.7, r=1.0)
    txt(x + cw2 / 2, 91.6, c.replace("_LR", " logistic").replace("_", " "), size=6.4)

txt(50, 96.5, "probabilities recalibrated by Platt scaling fitted inside the cross-validation\n"
              "→  a leakage-safe out-of-fold probability for each of the five modalities",
    size=6.5, ls=1.7)

box(4.4, 99.4, 91.2, 8.4, FUSE_BG, FUSE_EC, lw=0.9, r=1.2)
txt(50, 101.8, "Late-fusion stacking", size=7.2, weight="bold")
txt(50, 105.0, "elastic-net logistic regression (L1 + L2, l1_ratio = 0.5) over the five calibrated "
               "probability streams;\nthe L1 term sets non-contributing modalities to exactly zero, "
               "giving interpretable modality weights", size=6.4, ls=1.7)
arrow(50, 110.0, 50, 112.2)

# ================================================= 5. held-out evaluation ===
box(2, 112.4, 96, 9.6, BOX_BG, BOX_EC)
txt(50, 115.0, "Applied unchanged to the held-out outer test fold", weight="bold")
txt(50, 118.6, "in every repeat each patient receives exactly one out-of-fold predicted probability;\n"
               "no patient is ever scored by a model that saw them in training", size=6.6, ls=1.7)
arrow(50, 122.2, 50, 124.4)

# ================================== 5b. locked consensus signature ==========
# Run 5 changed which folds the reported signature is aggregated from, so the
# figure has to say which rule was used; the sentence is selected by the
# pipeline constant rather than written for the run that happens to be current.
_SIG_SENTENCE = {
    "winner_folds":
        "the signature is aggregated only over the folds that family won — one "
        "classifier, one matching signature",
    "winner_all_folds":
        "the signature is that family's own per-fold signature aggregated over "
        "every fold, won or not",
    "all_folds":
        "every fold contributes its own winner's signature, so the reported "
        "signature mixes classifier families",
}
if SIG_SOURCE not in _SIG_SENTENCE:
    raise SystemExit(f"SIGNATURE_SOURCE={SIG_SOURCE!r} in {PIPE} is not one of "
                     f"{sorted(_SIG_SENTENCE)}; the figure has no sentence for "
                     f"it and must not guess.")
box(2, 124.6, 96, 13.2, SIG_BG, SIG_EC, lw=1.1)
txt(50, 127.2, f"Locked consensus model — aggregated over the {EVAL_G:,} outer folds",
    weight="bold")
txt(50, 132.6,
    f"for each modality the locked classifier is the family that won the most outer folds;\n"
    f"{_SIG_SENTENCE[SIG_SOURCE]}\n"
    f"K = the median per-fold signature size; correlated "
    f"{' and '.join(sorted(DEDUP_MODS))} candidates are pooled to one representative",
    size=6.4, ls=1.75)
arrow(50, 138.0, 50, 140.2)

# ============================================================ 6. estimand ===
box(2, 140.4, 96, 14.0, EST_BG, EST_EC, lw=1.1)
txt(50, 143.0, "Reported performance", weight="bold")
txt(50, 148.8,
    f"the metric (AUROC, AUPRC, Brier) is computed on each repeat's complete out-of-fold vector\n"
    f"and averaged over the {RG} repeats ({RA} per arm);  95 % confidence intervals are patient-level\n"
    "cluster bootstraps — 2,000 stratified resamples of patients, a resampled patient carrying all of its\n"
    "repeat predictions.  Predictions are never averaged across repeats or across models.",
    size=6.5, ls=1.75)

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"))
plt.close(fig)
print(f"wrote {OUT.with_suffix('.pdf').name} and .png in {OUT.parent}")
print(f"  drawn from {RUN.name}  (seed {prov['random_seed']})")
print(f"  stage constants read from {PIPE.name}: {_pipe_prov}")
print(f"  cohort n={N_CC} ({N_EV} pCR); arms {ARMS}")
print(f"  panel {N_PANEL} metrics -> TIER1_REMOVE {len(_tier1)} listed, "
      f"{N_DEDUP} present -> {N_CAND} candidates")
print(f"  outer {OF}x{RG}={EVAL_G} global, {OA}x{RA}={EVAL_A} per arm; inner {IG}/{IA}")
print(f"  classifiers: {', '.join(CLFS)}")
print(f"  in-fold Tier 3 correlation filter: "
      f"{'DRAWN for ' + ', '.join(sorted(CORR_MODS)) if CORR_MODS else 'NOT DRAWN'}"
      f"  (CORR_FILTER_MODS={CORR_MODS or 'set()'}; --corr_threshold="
      f"{P['corr_threshold']} is inert while that set is empty)")
print(f"  in-fold univariate screen: {'DRAWN' if UNIV_ON else 'NOT DRAWN'}"
      f"  (q<={P['univ_fdr_q']}, keep {P['univ_min_k']}-{P['univ_max_k']}, "
      f"exempt {SCREEN_EXEMPT} at <={UNIV_MIN_FEATURES} candidates)")
print(f"  signature aggregation: SIGNATURE_SOURCE={SIG_SOURCE!r}; "
      f"consensus dedup {sorted(DEDUP_MODS)}")
print(f"  candidates per modality: {MOD_CAND}")
