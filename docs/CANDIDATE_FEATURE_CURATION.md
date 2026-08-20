# The candidate feature panel: what "a-priori biological curation" means

The candidate panel was assembled in two stages. **Only the second used the
outcome**, and it is the stage that was removed in this revision.

## Stage 1 — a-priori biological curation (no outcome is used)

No model was ever fitted to raw high-dimensional data. Each assay pipeline first
reduced its measurements to a fixed panel of pre-defined, biologically
interpretable metrics, chosen from prior knowledge of HER2-positive disease and
of the mechanisms of the two treatments compared in the trial, and computed for
every patient before any association with pCR was examined.

| Modality | Metrics | Composition |
|---|---:|---|
| Transcriptomics | 42 | validated composite signatures (five HER2DX modules — HER2 amplicon, pCR likelihood, luminal, proliferation, IGG; the sspbc intrinsic-subtype call; a PIK3CA-mutation signature; taxane response); ADC-relevant trafficking programmes (endocytosis, lysosome, exosome); metabolic programmes (oxidative phosphorylation, glycolysis, fatty-acid and glutathione metabolism); single transcripts of established clinical relevance (ESR1, ERBB2, PGR, MKI67, CD8A, FCGR3A, FCGR3B); immune-microenvironment deconvolutions (TILs, CD45, B/T/CD8-T/NK/Treg/mast/dendritic cells, macrophages, TAM-M2, CAF, cytotoxic cells, Th2 cells, neutrophils, MHC-I, T-cell dysfunction and exclusion, TCR and BCR clonality) |
| Genomics | 41 | coding mutations in genes and pathways of established relevance (ERBB2, PIK3CA, TP53, GATA3; HER, PI3K–AKT, MAPK–ERK, CDK–RB pathways; OncoKB-annotated counterparts); copy number at recurrently altered loci (ERBB2, PIK3CA, BRCA2, NCOR1, RAB11FIP1, FADD, PPFIA1, CTTN, RPL19, MED1, CDK12, PPP1R1B, MIEN1, GRB7); COSMIC mutational signatures 2, 3, 6, 7, 10, 13; burden and immunogenomic metrics (clonal oncogenic mutational burden, HRD, LOH-deletion burden, CNV burden, neoantigen load, HLA-A01 supertype, mean HED, LOHHLA, TCRA T-cell fraction) |
| Proteomics | 19 | the 17q12 and 11q13 amplicon proteins (ERBB2 and its protein group, GRB7, MIEN1, PPP1R1B, CDK12, MED1, RPL19, PPFIA1, CTTN, HER2-amplicon composite) together with the endosomal and vesicular trafficking machinery governing antibody–drug-conjugate internalisation and payload release (RAB11FIP1, RAB11B, RAB5C, EEA1, ARL1, FLOT1, VAMP3, SLC12A2) |
| Whole-slide image | 3 | cell–cell interaction, immune-cell proportion, tumour–immune distance |
| Clinical | 5 | ER status, nodal status, tumour size, proliferation, treatment arm |

**110 metrics in total.** (Run-5 figures, counted from the analysis file itself;
earlier drafts said 43 transcriptomic, 42 genomic and 112 in total. The authors
withdrew `RNA_ADC_trafficking` and `DNA_TMB_clone` from the panel before this
round, which is the whole of the difference.) The complete list is in
`supplementary/S-ML8_candidate_panel.xlsx` (sheet `Candidate_panel`).

This stage consults no pCR label at any point. It is a dimensionality-reduction
and interpretability decision — the same reduction that underlies every
signature-level analysis in the article — and it cannot inflate cross-validated
discrimination, because the outcome plays no part in it.

Note that the panel carries **clonal oncogenic mutational burden only**; a
total-TMB feature is not present in it.

## Stage 2 — univariate retention (uses the outcome; this was the leakage)

In the original submission, 54 features were retained from that panel because
they showed univariate associations with response *in the entire cohort or
within either treatment arm*. The pCR labels of all patients — including those
later held out in every cross-validation fold — therefore determined which
features were eligible to enter any model, so the reported internal performance
was optimistic. **This step has been removed.**

## What the current analysis does instead

The primary analysis starts from the complete 110-metric panel, not from the 54.
Two reductions are applied, and neither uses the outcome across the cohort.

### 1. A fixed biological deduplication, before any train/test split

`TIER1_REMOVE` in `code/multimodal_pcr_pipeline.py` names twenty-one features.
Eighteen of them are present in the analysis file, leaving **92 candidates**.
(Run-5 counts; earlier drafts said thirteen listed, eleven present, 101
candidates — the list was extended in runs 4 and 5, see below.) Correlations are
recomputed on the complete-case cohort (n = 110), with categorical columns
encoded 0/1 first; the pCR label is not used anywhere in this step.

| Removed | Reason | \|r\| with the retained counterpart | Retained instead |
|---|---|---:|---|
| PPP1R1B, MIEN1, GRB7 (copy number) | 17q12: co-amplify with *ERBB2* as a single genomic segment, so the values are identical by construction | 1.000 | *ERBB2* copy number |
| CDK12 (copy number) | 17q12 amplicon | 0.904 | *ERBB2* copy number |
| CTTN, FADD (copy number) | 11q13: co-amplify with *PPFIA1* | 1.000 | *PPFIA1* copy number |
| *TP53*, *GATA3* OncoKB calls | identical to the plain coding-mutation column on every row where both are observed: the OncoKB annotation reclassified nothing in this cohort | 1.000 | the plain coding-mutation call |
| *PIK3CA* OncoKB call | differs from the plain column in 2 of 190 rows | 0.954 | the plain coding-mutation call |
| CD8 T cells (RNA) | immune deconvolution near-identical to the *CD8A* transcript | 0.984 | *CD8A* mRNA |
| Cytotoxic cells (RNA) | near-identical to the *CD8A* transcript | 0.940 | *CD8A* mRNA and NK cells |
| TILs (RNA) | the immune-infiltration cluster collapses onto one representative; *CD8A* is the tie-break because `RNA_TILs` is 100 % missing in NCT02326974 and could not transfer | 0.942 | *CD8A* mRNA and NK cells |
| T cells, CD45 (RNA) | members of the same immune-infiltration cluster (0.972 and 0.930 against TILs) | 0.928, 0.877 | *CD8A* mRNA and NK cells |
| *ERBB2* mRNA | subsumed by the validated HER2DX HER2-amplicon score | 0.959 | HER2DX HER2 amplicon |
| ERBB2, GRB7 (protein) | 17q12 amplicon proteins, subsumed by the curated composite — by consistency with the RNA decision above | 0.923, 0.914 | HER2-amplicon protein composite |
| *FCGR3B* (RNA) | **not a redundancy removal.** FCGR3B is neutrophil-restricted; neutrophils are not retained in fresh-frozen biopsies, so its signal in bulk tumour RNA-seq reflects peripheral-blood contamination and is not comparable across protocols. It is excluded for non-interpretability of the measurement, which is a fact about the assay and is checkable without reference to pCR | 0.683 (its highest, vs neutrophils) | nothing; the feature has no retained counterpart |

Three further listed entries (uniform TMB, clonal TMB, pTMB) are
parameterisations of mutational burden that are not present in the analysis file
and therefore have no effect — the authors removed the duplicate clonal-TMB
column upstream, so the panel carries clonal **oncogenic** TMB only. The whole
step can be disabled with `--feature_pool full`.

This is not a second route by which information could enter. Up to run 3 a
per-fold correlation filter (Tier 3) did the same job for RNA and DNA, but it
kept whichever member of a redundant pair won *that fold's* univariate contest
against pCR, so the surviving representative rotated from fold to fold — the
rotating-basis instability that makes penalised-regression signatures
uninterpretable, and visible in the run-3 output as *TP53* and its identical
OncoKB twin each accumulating part of one selection frequency. Tier 3 is
therefore **removed**, and the fixed list is extended until it leaves no
within-modality pair of candidates above |r| = 0.90 on the complete case
(largest remaining: 0.880, *ESR1* mRNA against the HER2DX luminal score).
`preflight.py` recomputes every within-modality pair after `TIER1_REMOVE` and
fails the run if any exceeds the gate. The a-priori list replaces a
fold-dependent, outcome-driven choice with a fixed biological one, and so
reduces rather than increases the analysis's dependence on the outcome.

### 2. The univariate association step, moved inside every training fold

Features are ranked by the tie-corrected Mann–Whitney U statistic (equivalently,
the univariate AUROC) against pCR **on the training patients of that fold only**,
adjusted within modality by the Benjamini–Hochberg procedure and retained at
q ≤ 0.25, keeping between 5 and 40 features. Modalities with six or fewer
features (clinical, WSI) are not screened at all. No held-out patient influences
which features enter a model.

## The resulting funnel

Of the 92 candidates entering each outer fold of the pooled-cohort analysis, the
median number surviving preprocessing and the in-fold screen is 11
transcriptomic, 9 genomic and 11 proteomic features (clinical 5 and WSI 3,
unscreened), from which the events-per-variable cap admits a median signature of
9 transcriptomic, 7 genomic, 7 proteomic, 5 clinical and 3 WSI features.
(Run-5 medians over the 1,000 pooled outer folds, read from
`results/global/global_elasticnet_results.pkl`; the run-3 text said 14/6/13 and
11/5/7/5/3.)

Every count on this page is re-derived by `verify_quoted_numbers.py` from the
data file and from `TIER1_REMOVE` itself, so the documentation cannot drift away
from the code.
