#!/usr/bin/env bash
# =============================================================================
# PREDIX HER2 - PRODUCTION RUN 5 (Ubuntu, flat directory layout)
# =============================================================================
# WHAT CHANGED SINCE RUN 4 (all five are in this folder already):
#
#  1. TRANSPARENCY OF THE LOCKED MODEL. The consensus signature is now
#     aggregated ONLY over the outer folds won by the modal classifier
#     (--signature_source winner_folds, the new DEFAULT — the script relies on
#     that default and does not pass the flag). Up to run 4 the
#     classifier was the modal winner but the signature pooled every fold
#     regardless of which family won it, so the deliverable was "the modal
#     classifier" plus "the features the fold winners collectively chose" —
#     two different objects. Now it is one sentence: the modal winning
#     classifier, and the features that classifier selected. Fusion, the
#     internal metrics and the external validation all read the same locked
#     object, so they stay consistent with each other automatically.
#     EXPECT DNA and Prot signatures to move most: those are where the modal
#     support was weakest (Global DNA 30%, Global Prot 38%).
#
#  2. RNA_FCGR3B IS REMOVED (author decision, on bioinformatics advice).
#     TIER1_REMOVE goes 20 -> 21 entries, 18 present -> 92 candidates.
#     THIS ONE IS NOT A COLLINEARITY REMOVAL. FCGR3B is not redundant with
#     anything retained, and it was the top-ranked feature of the run-4 T-DM1
#     signature (100% of folds) and one of the 9 features in the pooled model
#     that validated on NCT02326974 at 0.679. Removing it WILL change the
#     T-DM1 arm result and the external validation, in an unknown direction.
#     The Methods must carry an outcome-blind justification — see the comment
#     block at RNA_FCGR3B in multimodal_pcr_pipeline.py. Whatever the external
#     numbers do, they are the result; do not revisit the decision afterwards.
#
#  3. THE PRE-FLIGHT COLLINEARITY GATE NOW SEES CATEGORICAL FEATURES. Run 4
#     coerced with pd.to_numeric before correlating, which silently skipped 21
#     candidates (every DNA_coding_mutation_*, the Clin_* text columns,
#     RNA_sspbc.subtype, and Prot_ERBB2_PG, which became a 100%-NaN column
#     despite being the top DHP proteomic feature). Encoding first gives the
#     same verdict — zero pairs above 0.90 — so run 4's conclusion stands, but
#     the gate now actually tests what it claims, and prints the strongest
#     surviving pair per modality so an all-NaN failure cannot look like a pass.
#
#  4. THE POOLED EXTERNAL OUTPUTS ARE RENAMED (--output_suffix _POOLED).
#     Run 4 wrote two different analyses to files with identical basenames in
#     different directories, so opening the obvious path showed the model that
#     FAILS on NCT02326974. Now: external_validation_POOLED.xlsx and
#     revfig06_external_validation_POOLED.pdf.
#
#  5. NEW step 5 draws the internal-vs-external comparison figure, and the logs
#     are tarred at the end (run 4 lost four log files in the copy back).
#
# Unchanged from run 4: the input file (SHA-256 64dd2f3ff1c99170...), expanded
# training, Tier 3 deleted, the binary-feature fix in _aggregate_signature, and
# pooled-model external validation.
# =============================================================================
# CARRIED FORWARD FROM RUN 4 (unchanged, listed so the design is in one place):
#   * Same input file, full 197-row delivery, SHA-256 64dd2f3ff1c99170...
#   * Expanded training: each unimodal model trains on every patient carrying
#     that modality; all models are SCORED on the identical 110 complete cases.
#   * Tier 3, the per-fold correlation filter, stays DELETED. preflight.py
#     proves it inert; the consensus-stage dedup remains as the safety net.
#   * The consensus dedup still covers binary features (nunique() > 1).
#   * Steps 4d/4e still validate the POOLED transcriptomic model against both
#     external cohorts alongside the arm-matched model. Both are reported.
#
# WHAT TO EXPECT IN THE NUMBERS
#   * DNA and Prot signatures should move most, from change 1: those are the
#     modalities where the modal classifier's support was weakest, so
#     restricting to its folds changes the aggregation pool the most.
#   * The T-DM1 RNA signature MUST change, from change 2: RNA_FCGR3B was its
#     top-ranked feature and is now gone.
#   * The external validation of the pooled model WILL move, because FCGR3B was
#     one of its nine features. Direction unknown. Report what comes out.
#   * The pooled fused AUROC has no reason to shift far from run 4's 0.781; if
#     it does, something other than these five changes is in play.
#
# Launch (survives logout, blocks system sleep for the duration):
#   mkdir -p logs
#   nohup systemd-inhibit --what=sleep:idle --why="PREDIX run 5" \
#       bash production_run_ubuntu.sh > logs/nohup.out 2>&1 &
# Progress:
#   cat logs/production_status.txt
#   tail -n 20 logs/step1_models.log
# =============================================================================

set -u
cd "$(dirname "$0")"

DATA="clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
SEED=42
RESULTS="results"
REPORT="report"
LOGS="logs"

NPROC=$(nproc)
NJOBS=$(( NPROC > 2 ? NPROC - 2 : NPROC ))

mkdir -p "${LOGS}"
STATUS="${LOGS}/production_status.txt"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >> "${STATUS}"
}

run_step() {
    local name="$1" logfile="$2"
    shift 2
    log_status "START ${name}"
    if "$@" > "${LOGS}/${logfile}" 2>&1; then
        log_status "DONE  ${name}"
    else
        local rc=$?
        log_status "FAILED ${name} (exit ${rc}) - see ${LOGS}/${logfile}. STOPPING."
        exit 1
    fi
}

PY=python3
command -v "${PY}" >/dev/null || { echo "python3 not found"; exit 1; }

log_status "RUN 5 LAUNCHED (n_jobs=${NJOBS} of ${NPROC} cores, seed=${SEED}, data=${DATA})"

# Step -1: pre-flight on the input file (seconds). Stops before the long job if
# the data is not what the run assumes.
run_step "preflight" "step_preflight.log" \
    "${PY}" preflight.py

# Step 0: statistics test suite (fast gate)
run_step "step0 tests" "step0_tests.log" \
    "${PY}" tests/test_statistics.py

# Step 1: models - the main computational step (hours)
run_step "step1 models (5x200 global, 5x100 arms)" "step1_models.log" \
    "${PY}" multimodal_pcr_pipeline.py \
        --data_path "${DATA}" \
        --results_dir "${RESULTS}" --splits_dir "${RESULTS}/shared_splits" \
        --mode elasticnet --training_data expanded \
        --experiments global dhp tdm1 \
        --classifiers ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_Linear \
        --repeats_global 200 --repeats_arm 100 \
        --outer_folds_global 5 --outer_folds_arm 5 \
        --inner_folds_global 5 --inner_folds_arm 3 \
        --univariate_screen in_fold --feature_pool curated \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

# Step 2: figures and tables
run_step "step2 report" "step2_report.log" \
    "${PY}" generate_report.py --results_dir "${RESULTS}" --out_dir "${REPORT}"

# Step 3: revision analyses (CIs, calibration, stability, EPV, comparisons).
# The S-group section and its --s_group_spec / --groups_json / --cutpoint_q /
# --n_perm flags no longer exist; do not pass them.
run_step "step3 revision analyses" "step3_revision.log" \
    "${PY}" revision_analyses.py --results_dir "${RESULTS}" --out_dir "${REPORT}" \
        --data_path "${DATA}" --n_boot 2000

# Step 4a: transferable feature lists
run_step "step4a shared features" "step4a_shared.log" \
    "${PY}" external_validation.py --predix "${DATA}" \
        --ispy2 RNA_curated_metrics_ISPY2.txt \
        --nct RNA_curated_metrics_NCT02326974.txt \
        --out_dir "${REPORT}" --export_shared_features_only

# Step 4b: RNA-only pipeline runs (locked models), one per cohort
run_step "step4b RNA-only dhp (I-SPY2 features)" "step4b_rna_ispy2.log" \
    "${PY}" multimodal_pcr_pipeline.py --data_path "${DATA}" \
        --results_dir results_rna_ispy2 --splits_dir results_rna_ispy2/splits \
        --mode elasticnet --training_data cc_only --experiments dhp \
        --modalities RNA \
        --include_features "${REPORT}/tables/revision/shared_features_I-SPY2.txt" \
        --repeats_arm 100 --univariate_screen in_fold \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

run_step "step4b RNA-only tdm1 (NCT features)" "step4b_rna_nct.log" \
    "${PY}" multimodal_pcr_pipeline.py --data_path "${DATA}" \
        --results_dir results_rna_nct --splits_dir results_rna_nct/splits \
        --mode elasticnet --training_data cc_only --experiments tdm1 \
        --modalities RNA \
        --include_features "${REPORT}/tables/revision/shared_features_NCT02326974.txt" \
        --repeats_arm 100 --univariate_screen in_fold \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

# Step 4c: locked external validation
run_step "step4c locked external validation" "step4c_external.log" \
    "${PY}" external_validation.py --predix "${DATA}" \
        --ispy2 RNA_curated_metrics_ISPY2.txt \
        --nct RNA_curated_metrics_NCT02326974.txt \
        --locked_ispy2 results_rna_ispy2 --locked_nct results_rna_nct \
        --out_dir "${REPORT}" --n_boot 2000

# -----------------------------------------------------------------------------
# Step 4d/4e: the POOLED transcriptomic model, validated
# against the same two cohorts. Pre-specified: both the arm-matched result
# (above) and the pooled result (here) are reported, whichever way they fall.
#
# The pooled RNA-only run is trained on every PREDIX patient carrying RNA
# (n = 185, 84 events) rather than on one arm, and its signature retains the
# HER2DX HER2-amplicon score that the T-DM1 arm signature drops. Output goes to
# a SEPARATE report directory so the primary table above is not overwritten.
#
# NOTE --training_data cc_only, matching steps 4b exactly. With a single
# modality the "complete case" is defined over RNA alone, so cc_only and
# expanded select the SAME 185 patients; but external_validation.py enforces a
# hard identity check that the refit cohort equals the pipeline's OOF cohort
# exactly, and there is no reason to risk that check on a configuration
# difference that buys nothing.
# -----------------------------------------------------------------------------
REPORT_POOLED="report_pooled_external"
mkdir -p "${REPORT_POOLED}"

run_step "step4d RNA-only pooled (I-SPY2 features)" "step4d_rna_pooled_ispy2.log" \
    "${PY}" multimodal_pcr_pipeline.py --data_path "${DATA}" \
        --results_dir results_rna_pooled_ispy2 \
        --splits_dir results_rna_pooled_ispy2/splits \
        --mode elasticnet --training_data cc_only --experiments global \
        --modalities RNA \
        --include_features "${REPORT}/tables/revision/shared_features_I-SPY2.txt" \
        --repeats_global 200 --univariate_screen in_fold \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

run_step "step4d RNA-only pooled (NCT features)" "step4d_rna_pooled_nct.log" \
    "${PY}" multimodal_pcr_pipeline.py --data_path "${DATA}" \
        --results_dir results_rna_pooled_nct \
        --splits_dir results_rna_pooled_nct/splits \
        --mode elasticnet --training_data cc_only --experiments global \
        --modalities RNA \
        --include_features "${REPORT}/tables/revision/shared_features_NCT02326974.txt" \
        --repeats_global 200 --univariate_screen in_fold \
        --n_jobs "${NJOBS}" --seed "${SEED}" --consensus

run_step "step4e pooled-model external validation" "step4e_external_pooled.log" \
    "${PY}" external_validation.py --predix "${DATA}" \
        --ispy2 RNA_curated_metrics_ISPY2.txt \
        --nct RNA_curated_metrics_NCT02326974.txt \
        --locked_ispy2 results_rna_pooled_ispy2 \
        --locked_nct results_rna_pooled_nct \
        --locked_experiment global \
        --output_suffix _POOLED \
        --out_dir "${REPORT_POOLED}" --n_boot 2000

# -----------------------------------------------------------------------------
# Package the logs. Run 4 lost production_status.txt, step2_report.log and both
# step4d logs somewhere between the Ubuntu box and Windows, which made the run
# harder to audit than it should have been. Tar them so the copy back is one
# file that either arrives intact or obviously does not.
#
# THIS RUNS BEFORE STEP 5, DELIBERATELY. Every run_step is fatal, and step 5 is
# a cosmetic figure that can abort on a missing or non-finite CI cell. If the
# tarball came afterwards, a failed figure would throw away the logs for all
# eight hours of modelling that had already succeeded — losing the audit trail
# to protect nothing. nohup.out is excluded because it is the live stdout of
# this very shell: tar would read a file being written and exit 1.
# -----------------------------------------------------------------------------
tar -czf logs_run5.tar.gz --exclude=nohup.out "${LOGS}" \
    && log_status "logs archived to logs_run5.tar.gz" \
    || log_status "WARNING: could not archive logs (see stderr above)"

# Step 5: the internal-vs-external comparison figure. Reads both external
# tables, draws z-score only (the training phase standardises within folds, so
# z-scoring the external cohort is its analogue; rank has no counterpart).
# Cosmetic: every number it draws already exists in the two external workbooks,
# so a failure here costs a figure, not a result.
run_step "step5 internal-vs-external figure" "step5_scope_figure.log" \
    "${PY}" make_fig_internal_vs_external.py "${REPORT}/figures/revision"

log_status "RUN 5 COMPLETE. Results: ${RESULTS} | Report: ${REPORT} | Pooled external: ${REPORT_POOLED}"
log_status "Copy BACK to Windows: results/ results_rna_ispy2/ results_rna_nct/ results_rna_pooled_ispy2/ results_rna_pooled_nct/ report/ report_pooled_external/ logs/ logs_run5.tar.gz"
echo
echo "================================================================"
echo "COPY BACK INTO A FOLDER NAMED ubuntu_results_run5:"
echo "  results/ results_rna_ispy2/ results_rna_nct/"
echo "  results_rna_pooled_ispy2/ results_rna_pooled_nct/"
echo "  report/ report_pooled_external/ logs/ logs_run5.tar.gz"
echo "Verify 14 log files arrived:  ls logs/ | wc -l"
echo "================================================================"
