#!/usr/bin/env python3
"""Pre-flight check on the run-5 input file. Runs in seconds; stops the pipeline
before the multi-hour step if the data is not what this run assumes.

RUN 5 EXPECTS THE **FULL** DELIVERY, not a complete-case extract.

Carried over from run 3: expanded training only works if the file still contains
the modality-incomplete patients, because a complete-case file silently turns
`--training_data expanded` into a no-op (that is what happened in run 2). The
decisive check is that the file is LARGER than its own complete case.

WHICH CHECKS BLOCK AND WHICH ONLY REPORT. `check()` appends to `fail` and the
script exits 1; `note()` appends to `warn`, prints "PRE-FLIGHT OK WITH NOTES"
and exits 0, so a note does NOT stop the run. The one gate that neither routes
through is the outcome-derived-feature gate below: it calls sys.exit(1) on the
spot, because if that column is present nothing measured afterwards means
anything.

THE COLLINEARITY GATE (added in run 4). Run 4 deletes the per-fold Tier 3
correlation filter and relies entirely on the fixed, outcome-blind TIER1_REMOVE
list. That is only sound if Tier 1 leaves no correlated pair the pipeline is
still blind to, so this script recomputes every within-modality pairwise
correlation on the complete case AFTER applying TIER1_REMOVE — pooled, DHP and
T-DM1 — and FAILS the run if a pair exceeds |r| = 0.90 in a modality the
consensus-stage dedup does NOT cover. A pair above the gate inside
CONSENSUS_DEDUP_MODS (RNA, DNA) is printed as a NOTE and does not block, because
that stage removes it on the full cohort without the fold-to-fold rotation that
made Tier 3 an artefact; see the block above the `_covered` / `_uncovered` split
for the full reasoning. If the check fires, either add the offending feature to
TIER1_REMOVE or do not remove Tier 3.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

DATA = Path("clin_multiomics_curated_metrics_PREDIX_HER2_new.txt")

# The complete case is what every model is evaluated on; it must reproduce the
# cohort the manuscript reports.
EXPECT_CC = 110
EXPECT_CC_EVENTS = 46
EXPECT_CC_ARMS = {"DHP": 59, "T-DM1": 51}

# ---------------------------------------------------------------------------
# FEATURES WITHDRAWN BY THE AUTHORS BEFORE ROUND 2
# ---------------------------------------------------------------------------
# Two of them, and they are NOT the same kind of problem, so they are not
# enforced the same way. Read this before merging the two tuples back together.
#
# *** RNA_ADC_trafficking MUST NEVER BE RESTORED. THIS IS A HARD CHECK. ***
# It is not a predictor. Per the bioinformatics lead (2026-08-20), the signature
# is constructed from a mixture of pCR and residual disease — i.e. it is derived
# from the OUTCOME. Including it would regress pCR partly on a transformed copy
# of pCR, which is circular by construction and would inflate every metric that
# touches it. It is NOT in TIER1_REMOVE, so if the column ever reappears in the
# input file it enters the candidate pool and the whole run is contaminated:
# hence an immediate sys.exit(1) rather than a note.
#
# This retroactively explains three things, and they are worth recording so the
# question is not reopened:
#   * why it was withdrawn upstream before round 2;
#   * why it scored AUROC 0.740 (q = 0.025, rank 6/40) in the T-DM1 arm and
#     0.513 (q = 0.97) in DHP — a "predictor" that tracks the outcome will look
#     strong wherever the outcome is being tracked, not where biology says it
#     should be;
#   * why the previously published T-DM1 signature, which contained it, looked
#     better than anything the current pipeline produces.
#
# It was investigated for restoration on 2026-08-20 (it is present in
# old_codes/clin_multiomics_curated_metrics_PREDIX_HER2.txt and in BOTH external
# cohorts as 'RNA_ADC_traficking' — one 'f') and REJECTED on the grounds above.
# The candidate v3 input file built for that purpose was deleted. Any
# external-cohort column of that name is equally unusable; external_validation.py
# only applies the alias when the PREDIX target exists, so with the feature
# absent from PREDIX the alias is inert and reports itself as not applied. BOTH
# spellings are gated here, because FEATURE_ALIASES would map the external one
# onto the PREDIX name the moment a merged file was built.
OUTCOME_DERIVED = ("RNA_ADC_trafficking", "RNA_ADC_traficking")

# DNA_TMB_clone is a SOFT check — reported, does not block. It was withdrawn on
# measurement grounds, not because it is outcome-derived, and unlike
# RNA_ADC_trafficking it IS in TIER1_REMOVE, so even a stale file that carries
# the column cannot get it into the candidate pool. Presence is a signal that
# the wrong delivery was dropped in, which is what the note says.
WITHDRAWN = ("RNA_ADC_trafficking", "DNA_TMB_clone")

MODALITIES = ("Clin", "RNA", "DNA", "Prot", "WSI")
COMPLETENESS = ("RNA", "DNA", "Prot", "WSI")   # Clin is imputed in-fold

fail, warn = [], []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fail.append(name)


def note(name, detail=""):
    print(f"NOTE  {name}  {detail}")
    warn.append(name)


if not DATA.exists():
    sys.exit(f"MISSING: {DATA}\nDrop the full-cohort delivery into this folder "
             f"under exactly that name, then re-run.")

sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
print(f"input SHA-256: {sha}\n")

df = pd.read_csv(DATA, sep="\t")
check("patientID and pCR present", {"patientID", "pCR"} <= set(df.columns))

feat = [c for c in df.columns if c not in ("patientID", "pCR")]
by = {m: sum(1 for c in feat if c.startswith(m + "_")) for m in MODALITIES}
check("all five modalities present", all(v > 0 for v in by.values()), str(by))
print(f"      {len(df)} rows x {len(df.columns)} columns; {len(feat)} features\n")

# ---- the complete case must reproduce the reported cohort -------------------
molecular = [c for c in feat if c.split("_")[0] in COMPLETENESS]
cc = df.dropna(subset=molecular)
check(f"complete case = {EXPECT_CC} patients", len(cc) == EXPECT_CC, str(len(cc)))
check(f"complete case = {EXPECT_CC_EVENTS} pCR events",
      int(cc["pCR"].sum()) == EXPECT_CC_EVENTS, str(int(cc["pCR"].sum())))

arm = cc["Clin_Arm"]
if pd.api.types.is_numeric_dtype(arm):
    got = {("DHP" if k == 0 else "T-DM1"): int(v)
           for k, v in arm.value_counts().to_dict().items()}
else:
    got = {str(k): int(v) for k, v in arm.value_counts().to_dict().items()}
check("complete-case arm sizes", got == EXPECT_CC_ARMS, str(got))

# ---- THE decisive check: expanded training must have something to expand ----
print()
check("the file is LARGER than its complete case, so expanded training is real",
      len(df) > len(cc),
      f"{len(df)} rows vs {len(cc)} complete cases "
      f"({len(df) - len(cc)} modality-incomplete patients available for training)")

print("\nper-modality training cohorts (patients with that modality observed):")
for m in MODALITIES:
    cols = [c for c in feat if c.startswith(m + "_")]
    n = int(df.dropna(subset=cols).shape[0])
    ev = int(df.dropna(subset=cols)["pCR"].sum())
    flag = "" if n > len(cc) else "   <-- no wider than the complete case"
    print(f"      {m:<5} n={n:4d}  events={ev:3d}{flag}")

# ---- features withdrawn upstream -------------------------------------------
print()

# HARD GATE. Not a note: an outcome-derived column in the input file invalidates
# every number the run would go on to produce, so stop here and stop loudly.
_contaminated = [c for c in OUTCOME_DERIVED if c in df.columns]
if _contaminated:
    print("!" * 78)
    print("!!  PRE-FLIGHT FAILED — OUTCOME-DERIVED FEATURE PRESENT IN THE INPUT")
    print("!" * 78)
    for c in _contaminated:
        _n = int(df[c].notna().sum())
        print(f"!!  {c} is a column of {DATA} ({_n}/{len(df)} observed).")
    print("!!")
    print("!!  The ADC-trafficking signature is CONSTRUCTED FROM THE OUTCOME: it")
    print("!!  is built from a mixture of pCR and residual disease. Regressing")
    print("!!  pCR on it is circular by construction, and every AUROC, every")
    print("!!  confidence interval and every external-validation number produced")
    print("!!  from a file containing it is inflated and unpublishable.")
    print("!!")
    print("!!  It is NOT in TIER1_REMOVE, so nothing downstream would remove it.")
    print("!!  It was withdrawn upstream before round 2; restoring it was")
    print("!!  investigated on 2026-08-20 and REJECTED by the bioinformatics")
    print("!!  lead. 'RNA_ADC_traficking' (one 'f') is the external-cohort")
    print("!!  spelling of the same signature and is equally unusable.")
    print("!!")
    print("!!  Do NOT 'fix' this by editing preflight.py. Fix it by supplying the")
    print("!!  round-2 delivery, which does not contain the column:")
    print("!!      clin_multiomics_curated_metrics_PREDIX_HER2_new.txt")
    print("!!      sha256 64dd2f3ff1c99170c70a27685c7d9d5633c5ae2edb23b45dbabc1b88a575cef0")
    print("!" * 78)
    print("\nThe pipeline was NOT started.")
    sys.exit(1)
print(f"PASS  no outcome-derived ADC-trafficking column "
      f"({', '.join(OUTCOME_DERIVED)}) is present  [HARD CHECK]")

for c in WITHDRAWN:
    if c in df.columns:
        note(f"{c} is PRESENT in this file",
             "it was withdrawn before round 2 — confirm this is intended")
    else:
        print(f"PASS  {c} absent (withdrawn upstream)")

# ---- RUN 4: Tier 1 must leave nothing for the deleted Tier 3 to do ----------
print()
import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from multimodal_pcr_pipeline import (TIER1_REMOVE, CORR_FILTER_MODS,  # noqa: E402
                                     CONSENSUS_DEDUP_MODS, SIGNATURE_SOURCE)

CORR_GATE = 0.90
t1_present = [c for c in TIER1_REMOVE if c in df.columns]
candidates = [c for c in feat if c not in TIER1_REMOVE]
print(f"TIER1_REMOVE lists {len(TIER1_REMOVE)}, {len(t1_present)} present "
      f"-> {len(candidates)} candidates enter the fold loop")

check("Tier 3 (per-fold correlation filter) is disabled in run 4",
      len(CORR_FILTER_MODS) == 0, f"CORR_FILTER_MODS={CORR_FILTER_MODS or '{}'}")
check("the consensus-stage dedup is still enabled as a safety net",
      CONSENSUS_DEDUP_MODS == {"RNA", "DNA"}, str(CONSENSUS_DEDUP_MODS))

# ---- RUN 5 settings -------------------------------------------------------
check("the locked signature belongs to the locked classifier",
      SIGNATURE_SOURCE in ("winner_all_folds", "winner_folds"),
      f"SIGNATURE_SOURCE={SIGNATURE_SOURCE!r} — 'winner_folds' (run-5 default) "
      f"aggregates the folds the modal classifier won; 'winner_all_folds' uses "
      f"its own signature from every fold; 'all_folds' is run-4 behaviour and "
      f"mixes classifier families")

check("RNA_FCGR3B is removed (run-5 author decision)",
      "RNA_FCGR3B" in TIER1_REMOVE,
      "the Methods must carry the OUTCOME-BLIND justification for this one; "
      "it is not a collinearity removal like the others")
if "RNA_FCGR3B" in df.columns:
    _obs = int(df["RNA_FCGR3B"].notna().sum())
    print(f"      RNA_FCGR3B was observed in {_obs}/{len(df)} patients and is "
          f"excluded from the candidate pool before any outcome is examined")

# RUN 5 FIX — THE GATE MUST SEE THE CATEGORICAL FEATURES.
#
# Run 4 computed this matrix with `.apply(pd.to_numeric, errors="coerce")`,
# which turns every non-numeric column into NaN. The `np.isfinite(v)` test
# below then skipped those pairs SILENTLY, so the gate never examined 21 of the
# candidates: every DNA_coding_mutation_* (stored as the strings True/False),
# Clin_Arm / Clin_ER / Clin_ANYNODES / Clin_TUMSIZE, RNA_sspbc.subtype, and
# Prot_ERBB2_PG (Positive/Negative), which coerces to a 100%-NaN column.
# Prot_ERBB2_PG is not a minor case: it is the top-ranked feature of the run-4
# DHP proteomic signature, and the gate had never tested it against anything.
#
# Encoding first and re-running the gate on run-4's pool gives the SAME verdict
# (zero pairs above 0.90; the maxima are RNA 0.880 ESR1~HER2DX_luminal, Prot
# 0.857 MIEN1~HER2_amplicon, DNA 0.835, WSI 0.615, Clin 0.294), so run 4's
# conclusion stands. The check was simply weaker than it claimed to be.
# Use the PIPELINE'S OWN encoder, not a local re-implementation. A local
# version got two things wrong that a hand-rolled encoder always will:
# Clin_TUMSIZE was alphabetised ('21-50' < '<=20' < '>50') instead of ordered
# (<=20 < 21-50 < >50) — Pearson between the two codings is only 0.464, so for
# a >2-level ordinal the gate's |r| is not a monotone function of the modelled
# |r| and a collinear pair could slip through — and RNA_sspbc.subtype was
# integer-coded 0-3 as ONE column while the pipeline one-hots it into three
# dummies, so the gate tested a column the model never sees and never tested
# the three it does (RNA_sspbc_LumB is in the global RNA signature).
# load_and_encode_data also applies TIER1_REMOVE, so its columns ARE the
# candidate pool the fold loop will see.
from multimodal_pcr_pipeline import load_and_encode_data  # noqa: E402

_enc = load_and_encode_data(DATA)
_enc_feats = [c for c in _enc.columns
              if c.split("_")[0] in MODALITIES and c not in ("patient_id", "pCR")]

# The gate must cover every cohort the consensus dedup will actually run on.
# finalize_consensus receives df_cc_exp — the ARM frame for dhp/tdm1, not the
# pooled complete case — and Tier 3 is deleted for all three. A pair can be far
# more correlated inside one arm than pooled: DNA_coding_mutation_HER_pathway ~
# DNA_coding_mutation_ERBB2_oncokb is 0.834 pooled but 1.000 on the 59 DHP
# complete cases. Checking pooled only proved inertness for one of three runs.
# Restrict to the ALREADY-VERIFIED complete case (cc, n=110 above) rather than
# re-deriving it. Encoding maps the 'Unknown' tokens in Clin_TUMSIZE and
# Clin_prolifvalu to NaN, so a dropna() over the encoded columns would silently
# shrink the cohort to 104 and the gate would then describe a population the
# pipeline never evaluates. .corr() uses pairwise-complete deletion, which is
# the right treatment for those few missing cells.
_enc_cc = _enc.loc[cc.index]
_arm_col = "Clin_Arm"
_cohorts = [("pooled", _enc_cc)]
if _arm_col in _enc_cc.columns:
    _cohorts += [("DHP", _enc_cc[_enc_cc[_arm_col] == 0]),
                 ("T-DM1", _enc_cc[_enc_cc[_arm_col] == 1])]

residual = []
for _label, _cc in _cohorts:
    print(f"      {_label:6s} complete case n={len(_cc):3d} — strongest "
          f"retained pair per modality:")
    if len(_cc) < 3:
        print(f"        too few complete cases to correlate — INVESTIGATE")
        continue
    for m in MODALITIES:
        cols = [c for c in _enc_feats if c.startswith(m + "_")]
        if len(cols) < 2:
            continue
        R = _cc[cols].corr().abs()
        pr = [(R.loc[a, b], a, b) for i, a in enumerate(cols)
              for b in cols[i + 1:] if np.isfinite(R.loc[a, b])]
        if not pr:
            print(f"        {m:5s} no finite pair — INVESTIGATE")
            continue
        r, a, b = max(pr)
        flag = "  <-- ABOVE GATE" if r > CORR_GATE else ""
        print(f"        {m:5s} {r:.3f}  {a} ~ {b}{flag}")
        for rr, aa, bb in pr:
            if rr > CORR_GATE:
                residual.append((f"{_label}/{m}", aa, bb, float(rr)))

# Deleting Tier 3 is safe for a modality if EITHER no pair exceeds the gate,
# OR the consensus-stage dedup covers that modality (it operates on the full
# cohort, so unlike Tier 3 it does not rotate representatives between folds).
# Only an uncovered modality is a genuine blocker.
_covered = [t for t in residual if t[0].split("/")[1] in CONSENSUS_DEDUP_MODS]
_uncovered = [t for t in residual if t[0].split("/")[1] not in CONSENSUS_DEDUP_MODS]

if _covered:
    print(f"      NOTE {len(_covered)} correlated pair(s) in "
          f"{sorted(CONSENSUS_DEDUP_MODS)} — these are handled by the "
          f"consensus-stage dedup, which is why it was kept when Tier 3 was "
          f"deleted. Expect [CONSENSUS-POOL] lines in the run log:")
    for m, a, b, v in sorted(_covered, key=lambda t: -t[3]):
        print(f"        {m}  {v:.4f}  {a} ~ {b}")

check(f"no UNPROTECTED candidate pair exceeds |r| = {CORR_GATE} after Tier 1",
      not _uncovered,
      "every modality is either free of correlated pairs or covered by the "
      "consensus dedup — Tier 3's removal is safe" if not _uncovered
      else f"{len(_uncovered)} pair(s) exceed the gate in a modality the "
           f"consensus dedup does NOT cover ({sorted(set(t[0] for t in _uncovered))}) "
           f"— add them to TIER1_REMOVE or restore Tier 3")
for m, a, b, v in sorted(_uncovered, key=lambda t: -t[3]):
    print(f"        {m}  {v:.4f}  {a} ~ {b}")

# Report the identical-column pairs explicitly: these are the ones that produced
# duplicated entries in the run-3 consensus signatures.
print("\nexact-duplicate check on the retained candidates:")
dupes = []
for m in MODALITIES:
    cols = [c for c in candidates if c.startswith(m + "_")]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            both = df[a].notna() & df[b].notna()
            if int(both.sum()) > 0 and bool((df.loc[both, a] == df.loc[both, b]).all()):
                dupes.append((a, b, int(both.sum())))
check("no two retained candidates are byte-identical", not dupes,
      "none" if not dupes else str(dupes))

# ---- missingness profile, reported not asserted -----------------------------
n_missing = int(df[feat].isna().sum().sum())
print(f"\nmissing feature cells in the full file: {n_missing:,} "
      f"({n_missing / (len(df) * len(feat)):.1%}) — expected and fine; the "
      f"complete case above is what is evaluated")

print()
if fail:
    print("PRE-FLIGHT FAILED:", fail)
    print("The pipeline was NOT started. Fix the input or update preflight.py.")
    sys.exit(1)
if warn:
    print("PRE-FLIGHT OK WITH NOTES:", warn)
else:
    print("PRE-FLIGHT OK", end=" ")
print("— starting the run.")
