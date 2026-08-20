"""Build Supplementary Table S-ML8: the complete candidate feature panel and the
fixed biological deduplication, with the |r| values recomputed from the data
rather than copied from the code comments.

NOTE ON THIS SCRIPT'S OWN FILENAME
----------------------------------
The table is **S-ML8** and this file is still called `build_supp_table_S-ML9.py`.
That is deliberate, not an oversight. The item was renumbered on 2026-08-21 when
a three-way collision was reconciled (see §9 of FIGURE_AND_TABLE_MAP.md): the
citations in the manuscript and the response letter were already consistent at
S-ML8, so the artefact moved to match them rather than the other way round.
The script keeps its old name because `build_github_repo.py` copies it by that
name into the deposit as `supplementary/build_S-ML9_candidate_panel.py`, and a
build-tool filename is not a citation anyone reads. What it WRITES carries the
correct number: `supp_table_S-ML8_candidate_panel.xlsx`, sheets headed S-ML8a
and S-ML8b.

WHY THE REMOVAL LIST IS NOT TYPED INTO THIS FILE
------------------------------------------------
It used to be. A hand-copied copy of TIER1_REMOVE sat here and went stale: it
still held the run-3 list (13 entries, 10 present, 100 candidates) after the
pipeline had moved to the run-5 list (21 entries, 18 present, 92 candidates),
and nothing failed — the script simply emitted an out-of-date supplementary
table. The list is now PARSED from the pipeline source, which is the one
authoritative definition, and every count in the table is derived from that
parse plus the data file.

WHY IT IS PARSED RATHER THAN IMPORTED
-------------------------------------
`import multimodal_pcr_pipeline` would be cleaner, but importing it pulls in
sklearn, joblib, threadpoolctl and shap and sets the BLAS thread-pool
environment variables at import time. This script only needs one list of
strings, and it must not depend on the modelling stack being installed or on
those side effects. `ast` reads the literal without executing anything.

The parser understands the two forms the pipeline has used: a plain list
literal (runs 1-5), and a literal followed by a guarded mutation such as
run 6's `if not KEEP_RNA_FCGR3B: TIER1_REMOVE.append("RNA_FCGR3B")`. Anything
it does not recognise is a hard failure, never a silent partial answer.

Every number this script depends on is checked against a tripwire below and the
script exits non-zero on a mismatch, so a future change to the pipeline or to
the data file stops the table being built instead of ageing quietly inside it.
"""
import ast
import hashlib
import re
import sys
import tokenize
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\georg\Documents\claude_kang_multimodal_natcancer")
DATA = ROOT / "clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
OUT = ROOT / "revision_deliverables" / "supp_table_S-ML8_candidate_panel.xlsx"

# Where the authoritative TIER1_REMOVE lives. First existing path wins; the
# later entries cover the deposited GitHub layout, where this script ships as
# supplementary/build_S-ML9_candidate_panel.py next to code/.
_HERE = Path(__file__).resolve().parent
PIPELINE_CANDIDATES = [
    ROOT / "multimodal_pcr_pipeline.py",
    _HERE.parent / "code" / "multimodal_pcr_pipeline.py",
    _HERE.parent / "multimodal_pcr_pipeline.py",
    _HERE / "multimodal_pcr_pipeline.py",
]

# ---------------------------------------------------------------------------
# TRIPWIRES. These are NOT the source of any number in the table — everything
# below is computed from the pipeline source and from the data file. They exist
# so that a change in either one FAILS HERE instead of silently producing an
# out-of-date supplementary table. Update them only alongside a documented
# production run, and re-read the deduplication annotations when you do.
# Current values describe run 5 (production, 2026-08-19).
# ---------------------------------------------------------------------------
EXPECT_RUN            = "run 5"
EXPECT_TIER1_LISTED   = 21   # entries in the TIER1_REMOVE literal, FCGR3B included
EXPECT_TIER1_PRESENT  = 18   # of those, present as columns in this data file
EXPECT_PANEL          = 110  # metrics in the panel (112 columns - patientID - pCR)
EXPECT_CANDIDATES     = 92   # EXPECT_PANEL - EXPECT_TIER1_PRESENT
EXPECT_COMPLETE_CASE  = 110  # patients with all of RNA/DNA/Prot/WSI observed
EXPECT_CC_EVENTS      = 46
EXPECT_CC_ARMS        = {"DHP": 59, "T-DM1": 51}
EXPECT_DATA_SHA256    = ("64dd2f3ff1c99170c70a27685c7d9d5633c5ae2edb23b45"
                         "dbabc1b88a575cef0")

# Modality prefixes that define completeness. This is the pipeline's rule
# (get_complete_case): Clin never enters it, because clinical covariates are
# recorded for everyone and are median-imputed within each fold. A naive
# dropna() over ALL features is the trap here — Clin_TUMSIZE and
# Clin_prolifvalu carry the string "Unknown" rather than NaN, so the naive
# version happens to agree today and would silently return 104 the moment
# those tokens are encoded properly.
COMPLETENESS = ("RNA", "DNA", "Prot", "WSI")


def fail(msg):
    """Stop with a message that says what to do. Never emit a partial table."""
    sys.exit("\n".join(["", "=" * 78, "BUILD ABORTED — Supplementary Table S-ML8 was NOT written.",
                        "=" * 78, msg, "=" * 78]))


# ---------------------------------------------------------------------------
# Parse TIER1_REMOVE out of the pipeline source
# ---------------------------------------------------------------------------
_MUTATORS = {"append", "extend", "insert", "remove", "pop", "clear"}


def _pipeline_path():
    for p in PIPELINE_CANDIDATES:
        if p.is_file():
            return p
    fail("multimodal_pcr_pipeline.py was not found. Looked in:\n  "
         + "\n  ".join(str(p) for p in PIPELINE_CANDIDATES)
         + "\nThe deduplication list is read from that file and is not "
           "duplicated here on purpose. Point PIPELINE_CANDIDATES at the "
           "pipeline that produced the results being reported.")


def _module_constants(tree):
    """Module-level `NAME = <constant>` assignments, for resolving if-guards."""
    out = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)):
            out[node.targets[0].id] = node.value.value
    return out


def _truthy(test, consts, where):
    """Resolve a guard such as `if not KEEP_RNA_FCGR3B:` at parse time."""
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.Name):
        if test.id in consts:
            return bool(consts[test.id])
        fail(f"{where}: the guard on a TIER1_REMOVE mutation depends on "
             f"{test.id!r}, which is not a module-level constant in the "
             f"pipeline. This script cannot resolve it without executing the "
             f"pipeline. Resolve it by hand and update the parser.")
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return not _truthy(test.operand, consts, where)
    fail(f"{where}: unrecognised guard on a TIER1_REMOVE mutation "
         f"({ast.dump(test)[:120]}...). Update the parser rather than "
         f"guessing which branch runs.")


def _mutation_nodes(tree):
    """Every syntactic mutation of TIER1_REMOVE anywhere in the module."""
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "TIER1_REMOVE"
                and node.func.attr in _MUTATORS):
            found.append(node)
        elif (isinstance(node, ast.AugAssign)
              and isinstance(node.target, ast.Name)
              and node.target.id == "TIER1_REMOVE"):
            found.append(node)
        elif (isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Subscript)
                      and isinstance(t.value, ast.Name)
                      and t.value.id == "TIER1_REMOVE" for t in node.targets)):
            found.append(node)
    return found


def _trailing_comments(src):
    """{line number: comment text} for the whole source."""
    out = {}
    try:
        for tok in tokenize.generate_tokens(StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string.lstrip("#").strip()
    except (tokenize.TokenError, IndentationError):
        pass          # comments are decoration; the names are what matter
    return out


def parse_tier1_remove(path):
    """Return (list of feature names, {feature: source comment}, n_listed).

    Reads the authoritative definition out of the pipeline source with `ast`.
    Fails loudly on anything it does not fully understand.
    """
    if not path.is_file():
        fail(f"pipeline source not found: {path}")
    # utf-8-sig, not utf-8: a BOM would otherwise abort ast.parse with
    # "invalid non-printable character U+FEFF" (predix_run6_ubuntu's copy of
    # the pipeline carries one).
    src = path.read_text(encoding="utf-8-sig")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        fail(f"{path} does not parse as Python: {exc}")

    assigns = [n for n in tree.body
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "TIER1_REMOVE"
                       for t in n.targets)]
    if len(assigns) != 1:
        fail(f"expected exactly one module-level `TIER1_REMOVE = [...]` in "
             f"{path}, found {len(assigns)}.")
    node = assigns[0]
    if not isinstance(node.value, (ast.List, ast.Tuple)):
        fail(f"TIER1_REMOVE at {path}:{node.lineno} is not a list literal. "
             f"This script reads the literal without executing the pipeline.")
    try:
        names = list(ast.literal_eval(node.value))
    except ValueError:
        fail(f"TIER1_REMOVE at {path}:{node.lineno} is not a literal list of "
             f"constants; it cannot be read without executing the pipeline.")
    if not names or not all(isinstance(x, str) for x in names):
        fail(f"TIER1_REMOVE at {path}:{node.lineno} is not a non-empty list "
             f"of strings.")

    comments = _trailing_comments(src)
    ann = {}
    for elt, name in zip(node.value.elts, names):
        ann[name] = comments.get(elt.lineno, "")

    # Mutations after the literal (run 6 appends RNA_FCGR3B under a flag).
    consts = _module_constants(tree)
    handled = set()
    for stmt in tree.body:
        if not _mutation_nodes(ast.Module(body=[stmt], type_ignores=[])):
            continue
        if isinstance(stmt, ast.If):
            if stmt.orelse:
                fail(f"{path}:{stmt.lineno}: a TIER1_REMOVE mutation sits in an "
                     f"if/else. The parser only understands a plain guard.")
            take = _truthy(stmt.test, consts, f"{path}:{stmt.lineno}")
            body = stmt.body
        elif isinstance(stmt, ast.Expr):
            take, body = True, [stmt]
        else:
            fail(f"{path}:{stmt.lineno}: TIER1_REMOVE is modified by a "
                 f"statement this parser does not understand "
                 f"({type(stmt).__name__}).")
        for sub in body:
            subs = _mutation_nodes(ast.Module(body=[sub], type_ignores=[]))
            if not subs:
                continue          # a print() beside the mutation is harmless
            if not (isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call)
                    and sub.value in subs):
                fail(f"{path}:{sub.lineno}: TIER1_REMOVE is modified by a "
                     f"construct this parser does not read. Fix the parser "
                     f"before building the table.")
            call = sub.value
            handled.add(id(call))
            if not take:
                continue
            if call.func.attr == "append" and len(call.args) == 1:
                names.append(ast.literal_eval(call.args[0]))
            elif call.func.attr == "extend" and len(call.args) == 1:
                names.extend(ast.literal_eval(call.args[0]))
            else:
                fail(f"{path}:{call.lineno}: TIER1_REMOVE.{call.func.attr}() is "
                     f"not supported by this parser. Only append/extend of a "
                     f"literal are, because anything else can change the list "
                     f"in ways a static read cannot follow.")

    stray = [m for m in _mutation_nodes(tree) if id(m) not in handled]
    if stray:
        lines = sorted({getattr(m, "lineno", -1) for m in stray})
        fail(f"{path}: TIER1_REMOVE is modified at line(s) {lines} in a place "
             f"this parser does not read (inside a function or class, or by "
             f"assignment to an index). The parsed list would be wrong. Fix "
             f"the parser before building the table.")

    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        fail(f"{path}: TIER1_REMOVE contains duplicate entries {dup}. A "
             f"duplicate would double-count in every total in this table.")
    return names, ann


# ---------------------------------------------------------------------------
# Presentation-only annotation of each removal.
#
# The NAMES come from the pipeline (above); only the prose and the retained
# representative live here, and the coverage check below fails if the two ever
# disagree, so a feature can never be added to TIER1_REMOVE and silently
# inherit no explanation — or be removed from it and keep one.
#
# `retained` is the feature the pipeline keeps in place of the removed one and
# it is verified to be present in the data AND absent from TIER1_REMOVE. That
# check matters: run 5 removed RNA_TILs, which had itself been the stated
# counterpart of RNA_T-cells and RNA_CD45 when the list was written. The
# pipeline's own comment block settles the substitution — "KEEPING mRNA-CD8A
# rather than TILs" — so the retained representative of the whole
# immune-infiltration cluster is RNA_mRNA-CD8A. The counterpart named in the
# code comment is reported verbatim alongside it, so the reader sees both.
# `retained = None` means the removal is not a redundancy removal at all.
# ---------------------------------------------------------------------------
ANNOTATIONS = {
    "DNA_PPP1R1B_CNA": ("DNA_ERBB2_CNA",
        "17q12 amplicon: co-amplifies with ERBB2 as one genomic segment"),
    "DNA_MIEN1_CNA": ("DNA_ERBB2_CNA",
        "17q12 amplicon: co-amplifies with ERBB2 as one genomic segment"),
    "DNA_GRB7_CNA": ("DNA_ERBB2_CNA",
        "17q12 amplicon: co-amplifies with ERBB2 as one genomic segment"),
    "DNA_CDK12_CNA": ("DNA_ERBB2_CNA", "17q12 amplicon"),
    "DNA_CTTN_CNA": ("DNA_PPFIA1_CNA",
        "11q13 amplicon: co-amplifies with PPFIA1"),
    "DNA_FADD_CNA": ("DNA_PPFIA1_CNA",
        "11q13 amplicon: co-amplifies with PPFIA1"),
    "DNA_TMB_uniform": ("DNA_TMB_clone_oncogenic",
        "alternative parameterisation of the same mutational burden"),
    "DNA_TMB_clone": ("DNA_TMB_clone_oncogenic",
        "duplicate column: the same values are carried by the retained metric"),
    "DNA_pTMB": ("DNA_TMB_clone_oncogenic",
        "alternative parameterisation of the same mutational burden"),
    "DNA_coding_mutation_TP53_oncokb": ("DNA_coding_mutation_TP53",
        "OncoKB re-annotation reclassified nothing in this cohort: identical "
        "to the retained base column"),
    "DNA_coding_mutation_GATA3_oncokb": ("DNA_coding_mutation_GATA3",
        "OncoKB re-annotation reclassified nothing in this cohort: identical "
        "to the retained base column"),
    "DNA_coding_mutation_PIK3CA_oncokb": ("DNA_coding_mutation_PIK3CA",
        "OncoKB re-annotation differs from the retained base column in 2 of "
        "190 patients"),
    "RNA_CD8-T-cells": ("RNA_mRNA-CD8A",
        "immune deconvolution near-identical to the CD8A transcript"),
    "RNA_T-cells": ("RNA_mRNA-CD8A",
        "immune-infiltration cluster: near-identical to the TIL score, which "
        "is itself removed, so the retained representative is mRNA-CD8A"),
    "RNA_CD45": ("RNA_mRNA-CD8A",
        "immune-infiltration cluster: near-identical to the TIL score, which "
        "is itself removed, so the retained representative is mRNA-CD8A"),
    "RNA_Cytotoxic-cells": ("RNA_mRNA-CD8A",
        "near-identical to the CD8A transcript"),
    "RNA_TILs": ("RNA_mRNA-CD8A",
        "immune-infiltration cluster: mRNA-CD8A is kept instead because TILs "
        "is not measured in NCT02326974 and could not transfer externally; "
        "the choice used measurement availability only, never the outcome"),
    "RNA_mRNA-ERBB2": ("RNA_HER2DX_HER2_amplicon",
        "subsumed by the validated HER2DX HER2-amplicon composite score"),
    "Prot_ERBB2": ("Prot_HER2_amplicon",
        "constituent of the retained curated amplicon composite"),
    "Prot_GRB7": ("Prot_HER2_amplicon",
        "constituent of the retained curated amplicon composite"),
    "RNA_FCGR3B": (None,
        "not a redundancy removal: FCGR3B is neutrophil-restricted and "
        "neutrophils are not retained in fresh-frozen biopsies, so the signal "
        "reflects peripheral-blood contamination and is not comparable across "
        "protocols. Excluded on measurement validity, outcome-blind"),
}

# Presentation categories, assigned from the feature names. Prefix match, first
# hit wins; the catch-all per modality is last.
CATEGORY_RULES = [
    ("Clin_", None, "Clinical"),
    ("RNA_HER2DX", None, "Transcriptomics — validated composite signature"),
    ("RNA_sspbc", None, "Transcriptomics — validated composite signature"),
    ("RNA_pik3ca_sig", None, "Transcriptomics — validated composite signature"),
    ("RNA_Taxane", None, "Transcriptomics — validated composite signature"),
    ("RNA_ADC_traffick", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Endocytosis", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Lysosome", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Exosome", None, "Transcriptomics — ADC trafficking / vesicular programme"),
    ("RNA_Oxidative", None, "Transcriptomics — metabolic programme"),
    ("RNA_Glycolysis", None, "Transcriptomics — metabolic programme"),
    ("RNA_Fatty_acid", None, "Transcriptomics — metabolic programme"),
    ("RNA_Glutathione", None, "Transcriptomics — metabolic programme"),
    ("RNA_mRNA-", None, "Transcriptomics — single transcript"),
    ("RNA_FCGR3", None, "Transcriptomics — single transcript"),
    ("RNA_", None, "Transcriptomics — immune microenvironment"),
    ("DNA_coding_mutation", None, "Genomics — coding mutation (gene or pathway)"),
    ("DNA_COSMIC", None, "Genomics — COSMIC mutational signature"),
    (None, "_CNA", "Genomics — copy number at a recurrently altered locus"),
    ("DNA_", None, "Genomics — burden / immunogenomic metric"),
    ("Prot_", None, "Proteomics"),
    ("WSI_", None, "Whole-slide image — spatial metric"),
]

PROT_TRAFFICKING = {"Prot_RAB11FIP1", "Prot_RAB11B", "Prot_RAB5C", "Prot_EEA1",
                    "Prot_ARL1", "Prot_FLOT1", "Prot_VAMP3", "Prot_SLC12A2"}


def category(col):
    if col in PROT_TRAFFICKING:
        return "Proteomics — endosomal / vesicular trafficking machinery"
    if col.startswith("Prot_"):
        return "Proteomics — 17q12 / 11q13 amplicon protein"
    for pre, suf, cat in CATEGORY_RULES:
        if pre is not None and col.startswith(pre):
            return cat
        if suf is not None and col.endswith(suf) and col.startswith("DNA_"):
            return cat
    return "unclassified"


def numeric(series):
    """Numeric view of a feature column, preserving boolean columns.

    pd.to_numeric alone is not safe on this file: several DNA columns hold
    Python booleans and Prot_ERBB2_PG holds Positive/Negative, which coerces to
    an all-NaN column. Nothing in the deduplication list is a text categorical
    today, but this keeps that from becoming a silent zero-row correlation.
    """
    s = series.dropna()
    if len(s) and set(s.unique()) <= {True, False}:
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce")


def abs_corr(frame, a, b):
    """|Pearson r| on the rows where both are observed; NaN if undefined."""
    x, y = numeric(frame[a]), numeric(frame[b])
    ok = x.notna() & y.notna()
    if int(ok.sum()) < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return np.nan, int(ok.sum())
    return abs(float(np.corrcoef(x[ok], y[ok])[0, 1])), int(ok.sum())


# ---------------------------------------------------------------------------
# 1. Authoritative inputs
# ---------------------------------------------------------------------------
PIPE = _pipeline_path()
TIER1, TIER1_COMMENT = parse_tier1_remove(PIPE)
print(f"TIER1_REMOVE read from {PIPE}: {len(TIER1)} entries")

if len(TIER1) != EXPECT_TIER1_LISTED:
    fail(f"TIER1_REMOVE in {PIPE} now lists {len(TIER1)} features; this script "
         f"expects {EXPECT_TIER1_LISTED} ({EXPECT_RUN}).\n"
         f"parsed: {TIER1}\n"
         f"The pipeline is the authority, so the list itself is not the "
         f"problem — the tripwires and the ANNOTATIONS block in this file are "
         f"out of date. Update EXPECT_TIER1_* and ANNOTATIONS together, "
         f"confirm which production run the results folder holds, and re-run.")

missing_ann = [f for f in TIER1 if f not in ANNOTATIONS]
extra_ann = [f for f in ANNOTATIONS if f not in TIER1]
if missing_ann or extra_ann:
    fail(f"the deduplication annotations in this file no longer match "
         f"TIER1_REMOVE in {PIPE}.\n"
         f"  in the pipeline but not annotated here: {missing_ann}\n"
         f"  annotated here but no longer removed:  {extra_ann}\n"
         f"Add or delete the ANNOTATIONS entries (prose only — the names are "
         f"read from the pipeline).")

if not DATA.is_file():
    fail(f"data file not found: {DATA}")
sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
if sha != EXPECT_DATA_SHA256:
    fail(f"the input file is not the one this table was verified against.\n"
         f"  expected SHA-256 {EXPECT_DATA_SHA256}\n"
         f"  found            {sha}\n"
         f"Every count and every |r| below is computed from this file. "
         f"Confirm which delivery the production run used, update "
         f"EXPECT_DATA_SHA256 and the other tripwires, and re-run.")

df = pd.read_csv(DATA, sep="\t")
cols = [c for c in df.columns if c not in ("patientID", "pCR")]

# Complete case by the PIPELINE's rule (Clin is excluded from the definition).
molecular = [c for c in cols if c.split("_", 1)[0] in COMPLETENESS]
cc = df.dropna(subset=molecular)
naive = df.dropna(subset=cols + ["pCR"])
if len(naive) != len(cc):
    print(f"NOTE: a naive dropna() over all {len(cols)} features returns "
          f"{len(naive)} patients, not {len(cc)} — the pipeline's completeness "
          f"rule (RNA/DNA/Prot/WSI only) is the one used here.")

print(f"file: {df.shape[0]} patients x {df.shape[1]} columns; "
      f"{len(cols)} features; complete case n = {len(cc)}; SHA-256 {sha[:8]}…")

removed_present = [f for f in TIER1 if f in df.columns]
n_candidates = len(cols) - len(removed_present)

checks = [
    ("panel size", len(cols), EXPECT_PANEL),
    ("TIER1_REMOVE entries present in the data", len(removed_present),
     EXPECT_TIER1_PRESENT),
    ("candidates entering the fold loop", n_candidates, EXPECT_CANDIDATES),
    ("complete-case cohort", len(cc), EXPECT_COMPLETE_CASE),
    ("complete-case pCR events", int(cc["pCR"].sum()), EXPECT_CC_EVENTS),
]
bad = [(what, got, want) for what, got, want in checks if got != want]
arms = {str(k): int(v) for k, v in cc["Clin_Arm"].value_counts().items()}
if arms != EXPECT_CC_ARMS:
    bad.append(("complete-case arm sizes", arms, EXPECT_CC_ARMS))
if bad:
    fail("the table would not describe the run it is meant to describe:\n"
         + "\n".join(f"  {what}: computed {got}, expected {want}"
                     for what, got, want in bad)
         + f"\nThese are tripwires, not inputs — the computed value is what the "
           f"data and the pipeline actually say. Work out which production run "
           f"the manuscript is quoting, update the EXPECT_* block, and re-run.")

# The retained counterparts must be real, retained features.
broken = []
for feat, (keep, _reason) in ANNOTATIONS.items():
    if keep is None:
        continue
    if keep not in df.columns:
        broken.append(f"  {feat}: retained counterpart {keep} is not a column "
                      f"of the data file")
    elif keep in TIER1:
        broken.append(f"  {feat}: retained counterpart {keep} is itself in "
                      f"TIER1_REMOVE, so the table would claim a feature was "
                      f"kept when it was removed")
if broken:
    fail("the deduplication annotations name counterparts that do not hold:\n"
         + "\n".join(broken)
         + "\nPick the representative the pipeline actually retains for that "
           "cluster and say so in ANNOTATIONS.")

removed = set(TIER1)

# ---- sheet 1: the panel ------------------------------------------------------
rows = []
for c in cols:
    mod = c.split("_", 1)[0]
    rows.append({
        "modality": {"Clin": "Clinical", "RNA": "Transcriptomics",
                     "DNA": "Genomics", "Prot": "Proteomics",
                     "WSI": "Whole-slide image"}.get(mod, mod),
        "feature": c,
        "category (assigned for presentation)": category(c),
        "removed by the fixed biological deduplication": "yes" if c in removed else "no",
        "enters the cross-validation fold loop": "no" if c in removed else "yes",
    })
panel = pd.DataFrame(rows)
if (panel["category (assigned for presentation)"] == "unclassified").any():
    fail("these features match no presentation category — extend "
         "CATEGORY_RULES:\n  "
         + "\n  ".join(panel.loc[panel["category (assigned for presentation)"]
                                 == "unclassified", "feature"]))

# ---- sheet 2: the deduplication, with r recomputed ---------------------------
def show(v):
    return round(v, 3) if isinstance(v, float) and np.isfinite(v) else "not computable"


ded = []
for feat in TIER1:
    keep, reason = ANNOTATIONS[feat]
    comment = TIER1_COMMENT.get(feat, "")
    m_named = re.search(r"(?:with|identical to) (?:the retained )?"
                        r"([A-Za-z][\w.\-]*)", comment)
    named = m_named.group(1) if m_named else ""
    m_r = re.search(r"r\s*=\s*([01](?:\.\d+)?)", comment)
    stated_r = float(m_r.group(1)) if m_r else np.nan

    present = feat in df.columns
    r_cc = r_all = np.nan
    if present and keep and keep in df.columns:
        r_cc, _ = abs_corr(cc, feat, keep)
        r_all, _ = abs_corr(df, feat, keep)
    if not named:
        r_named = "none stated"
    elif named == keep:
        r_named = "same as the retained feature"
    elif present and named in df.columns:
        r_named = show(abs_corr(cc, feat, named)[0])
    else:
        r_named = "not computable"

    ded.append({
        "removed feature": feat,
        "reason": reason,
        "retained instead": keep if keep else "not applicable — see reason",
        "counterpart named in the pipeline comment": named or "none stated",
        "the named counterpart is itself removed":
            "yes" if named and named in removed else "no",
        "|r| stated in the pipeline comment": stated_r if np.isfinite(stated_r) else "none stated",
        "|r| recomputed vs the retained feature, complete case (n=%d)" % len(cc): show(r_cc),
        "|r| recomputed vs the retained feature, all patients with both measured": show(r_all),
        "|r| recomputed vs the named counterpart, complete case": r_named,
        "present in the analysis file": "yes" if present else "no (already absent)",
    })
ded = pd.DataFrame(ded)

print("\nDEDUPLICATION — stated vs recomputed |r|:")
print(ded.drop(columns=["reason"]).to_string(index=False))

print(f"\npanel: {len(panel)} metrics; deduplication removes "
      f"{len(removed_present)} present of {len(TIER1)} listed -> "
      f"{n_candidates} candidates")

by_mod = panel.groupby("modality").size()
print("\nby modality:\n", by_mod.to_string())

hdr1 = ("SUPPLEMENTARY TABLE S-ML8a. The complete candidate feature panel "
        "(a-priori biological curation). No outcome information was used to "
        "assemble this panel. The 'category' column is assigned for "
        "presentation from the feature naming and should be checked by the "
        "authors against the assay documentation. "
        f"{len(panel)} metrics; {len(removed_present)} of the "
        f"{len(TIER1)} listed deduplication entries are present in this data "
        f"file, leaving {n_candidates} candidates.")
hdr2 = ("SUPPLEMENTARY TABLE S-ML8b. The fixed, pre-specified biological "
        "deduplication applied before any train/test split (TIER1_REMOVE in "
        "multimodal_pcr_pipeline.py; disabled by --feature_pool full). "
        "Correlations are between features only; the pCR label is never used. "
        "Where the counterpart named when an entry was first written has since "
        "been removed as well, the retained representative of that cluster is "
        "given and both correlations are reported.")

with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    pd.DataFrame({hdr1: []}).to_excel(xw, sheet_name="Candidate_panel", index=False)
    panel.to_excel(xw, sheet_name="Candidate_panel", index=False, startrow=2)
    pd.DataFrame({hdr2: []}).to_excel(xw, sheet_name="Deduplication", index=False)
    ded.to_excel(xw, sheet_name="Deduplication", index=False, startrow=2)
print("\nwrote", OUT)
