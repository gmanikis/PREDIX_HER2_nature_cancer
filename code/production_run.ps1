# =============================================================================
# PREDIX HER2 - PRODUCTION RUN (Windows PowerShell mirror of run_predix_pipeline.sh)
# =============================================================================
# ASCII ONLY in this file: PowerShell 5.1 decodes BOM-less files as ANSI, and
# multi-byte punctuation (em-dashes etc.) decodes into stray smart quotes that
# corrupt the parser. Keep every character 7-bit.
#
# Data: clin_multiomics_curated_metrics_PREDIX_HER2_new.txt (canonical,
#       197 x 114; complete case n=109, RNA-complete n=185)
# Design: 5-fold x 200 repeats global (1,000 outer evaluations, the Methods
#         claim), 5 x 100 per arm. In-fold univariate screen. Consensus on.
# Requires next to this file: multimodal_pcr_pipeline.py, generate_report.py,
#   revision_analyses.py, external_validation.py, cv_estimands.py (shared
#   estimand module imported by the three post-processing scripts), tests/.
# Launch detached:
#   powershell -NoProfile -ExecutionPolicy Bypass -File production_run.ps1
# Progress:
#   Get-Content logs\production_status.txt
#   Get-Content logs\step1_models.log -Tail 20
# =============================================================================

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$DATA    = "clin_multiomics_curated_metrics_PREDIX_HER2_new.txt"
$NJOBS   = 20
$SEED    = 42
$RESULTS = "results"
$REPORT  = "report"
$LOGS    = "logs"

New-Item -ItemType Directory -Force $LOGS | Out-Null
$status = Join-Path $LOGS "production_status.txt"

function Log-Status($msg) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Add-Content $status
}

function Run-Step($name, $logfile, $cmd) {
    Log-Status "START $name"
    & cmd /c "$cmd > `"$LOGS\$logfile`" 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Log-Status "FAILED $name (exit $LASTEXITCODE) - see $LOGS\$logfile. STOPPING."
        exit 1
    }
    Log-Status "DONE  $name"
}

Log-Status "PRODUCTION RUN LAUNCHED (n_jobs=$NJOBS, seed=$SEED, data=$DATA)"

# Step 0: statistics test suite (fast gate)
Run-Step "step0 tests" "step0_tests.log" "python tests\test_statistics.py"

# Step 1: models - the main computational step (hours)
Run-Step "step1 models (5x200 global, 5x100 arms)" "step1_models.log" `
    ("python multimodal_pcr_pipeline.py " +
     "--data_path $DATA --results_dir $RESULTS --splits_dir $RESULTS\shared_splits " +
     "--mode elasticnet --training_data expanded --experiments global dhp tdm1 " +
     "--classifiers ElasticNet_LR RandomForest ExtraTrees HistGradBoost SVM_Linear " +
     "--repeats_global 200 --repeats_arm 100 " +
     "--outer_folds_global 5 --outer_folds_arm 5 " +
     "--inner_folds_global 5 --inner_folds_arm 3 " +
     "--univariate_screen in_fold --feature_pool curated " +
     "--n_jobs $NJOBS --seed $SEED --consensus")

# Step 2: figures and tables
Run-Step "step2 report" "step2_report.log" `
    "python generate_report.py --results_dir $RESULTS --out_dir $REPORT"

# Step 3: revision analyses.
# NOTE: runs with the DEFAULT S-group spec (still unconfirmed, handoff 1.1).
# When the confirmed spec exists, re-run JUST this step with --s_group_spec;
# it reads only the PKLs, so the re-run takes minutes, not hours.
Run-Step "step3 revision analyses" "step3_revision.log" `
    ("python revision_analyses.py --results_dir $RESULTS --out_dir $REPORT " +
     "--data_path $DATA --n_boot 2000 --n_perm 1000")

# Step 4a: transferable feature lists
Run-Step "step4a shared features" "step4a_shared.log" `
    ("python external_validation.py --predix $DATA " +
     "--ispy2 RNA_curated_metrics_ISPY2.txt --nct RNA_curated_metrics_NCT02326974.txt " +
     "--out_dir $REPORT --export_shared_features_only")

# Step 4b: RNA-only pipeline runs (locked models), one per cohort
Run-Step "step4b RNA-only dhp (I-SPY2 features)" "step4b_rna_ispy2.log" `
    ("python multimodal_pcr_pipeline.py --data_path $DATA " +
     "--results_dir results_rna_ispy2 --splits_dir results_rna_ispy2\splits " +
     "--mode elasticnet --training_data cc_only --experiments dhp --modalities RNA " +
     "--include_features $REPORT\tables\revision\shared_features_I-SPY2.txt " +
     "--repeats_arm 100 --univariate_screen in_fold " +
     "--n_jobs $NJOBS --seed $SEED --consensus")
Run-Step "step4b RNA-only tdm1 (NCT features)" "step4b_rna_nct.log" `
    ("python multimodal_pcr_pipeline.py --data_path $DATA " +
     "--results_dir results_rna_nct --splits_dir results_rna_nct\splits " +
     "--mode elasticnet --training_data cc_only --experiments tdm1 --modalities RNA " +
     "--include_features $REPORT\tables\revision\shared_features_NCT02326974.txt " +
     "--repeats_arm 100 --univariate_screen in_fold " +
     "--n_jobs $NJOBS --seed $SEED --consensus")

# Step 4c: locked external validation
Run-Step "step4c locked external validation" "step4c_external.log" `
    ("python external_validation.py --predix $DATA " +
     "--ispy2 RNA_curated_metrics_ISPY2.txt --nct RNA_curated_metrics_NCT02326974.txt " +
     "--locked_ispy2 results_rna_ispy2 --locked_nct results_rna_nct " +
     "--out_dir $REPORT --n_boot 2000")

Log-Status "PRODUCTION RUN COMPLETE. Results: $RESULTS | Report: $REPORT"
Log-Status "Quote numbers ONLY from report\tables\revision (patient-level bootstrap CIs)."
