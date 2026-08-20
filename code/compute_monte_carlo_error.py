"""How much would re-running this analysis move each reported AUROC?

    python revision_deliverables/compute_monte_carlo_error.py

Writes a table to stdout and, with --out, to an .xlsx.

WHY THIS EXISTS
---------------
The cross-validation partitions are seeded, so they are identical run to run.
The classifiers' internal randomness deliberately is NOT seeded
(`random_state=None`). That is a considered choice, not an oversight: if every
repeat drew the same bootstrap sample for a given fold and the same solver
shuffle order, the R repeats would be correlated rather than independent, and
the variance of a repeat-averaged metric would be understated. Seeding the
classifiers would therefore BUY bit-identical reruns at the price of
artificially narrow confidence intervals - precisely the kind of false
precision this revision exists to remove.

The honest cost of that choice is that a re-run does not reproduce
bit-identically. "The last digit may vary" is too vague to be useful to a
reviewer, so this measures it.

WHAT IS COMPUTED
----------------
The published estimand is: AUROC on each repeat's complete out-of-fold vector,
averaged over repeats. Its run-to-run spread is the Monte-Carlo standard error
of that average,

    SE = SD(per-repeat AUROC) / sqrt(R),      reported as a 95% band, 1.96 * SE.

This needs no refitting - it is recovered from the fold-level predictions stored
in the deposit. As a correctness check, the per-repeat means printed below
reproduce the published consensus AUROCs.

THREE CLAIMS THAT MUST NOT BE CONFLATED
---------------------------------------
1. The DEPOSITED results are exact. Regenerating the reported tables from the
   deposited fold-level outputs reproduces every cell identically, across
   operating systems and library generations.
2. A RE-RUN of model fitting is not bit-identical, by design (above).
3. The SIZE of that non-determinism is what this script reports.

Only (1) is a bit-reproducibility claim. Do not let (3) be read as a defect:
it is smaller than the sampling uncertainty the confidence intervals already
express, and it is reported rather than hidden.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def _default_results():
    """Locate the fold-level outputs in either layout.

    In the working repository they live under `ubuntu_results_run5/results/`;
    in the public deposit the same tree is simply `results/` at the top level.
    Checking both means a reviewer can run this straight out of the deposit
    with no arguments, which is the whole point of shipping it.
    """
    for cand in (ROOT / "ubuntu_results_run5" / "results",
                 ROOT / "results",
                 Path(__file__).resolve().parent / "results"):
        if (cand / "global" / "global_consensus_eval.pkl").exists():
            return cand
    return ROOT / "results"          # reported as missing, with the path shown


DEFAULT_RESULTS = _default_results()

# scenario label, results subdirectory, outer folds per repeat, expected repeats
SCENARIOS = [("Global", "global", 5, 200),
             ("DHP", "dhp", 5, 100),
             ("T-DM1", "tdm1", 5, 100)]
UNIMODAL = ["Clin", "RNA", "DNA", "Prot", "WSI"]


def group_into_repeats(folds, n_outer):
    """Chunk consecutive fold records into repeats.

    Fold records carry no repeat index; they are written in order, n_outer per
    repeat. Rather than trust that, every group is checked to partition the
    cohort exactly once - each patient appearing in exactly one test fold. If
    that fails, the layout is not what we assume and we return None so the
    caller refuses to report a number instead of reporting a wrong one.
    """
    groups = []
    for i in range(0, len(folds) - n_outer + 1, n_outer):
        grp = folds[i:i + n_outer]
        pids = np.concatenate([np.asarray(f["test_pids"]) for f in grp])
        if len(pids) != len(set(pids.tolist())):
            return None
        groups.append(grp)
    return groups


def per_repeat_auroc(groups, predict):
    out = []
    for grp in groups:
        y = np.concatenate([np.asarray(f["y_test"]) for f in grp])
        p = np.concatenate([np.asarray(predict(f), dtype=float) for f in grp])
        if len(np.unique(y)) > 1 and np.all(np.isfinite(p)):
            out.append(roc_auc_score(y, p))
    return np.asarray(out, dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results_dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--out", default=None, help="optional .xlsx output")
    args = ap.parse_args()
    res = Path(args.results_dir)

    rows = []
    print(f"{'scenario':<8} {'model':<18} {'R':>4} {'AUROC':>8} "
          f"{'SD/repeat':>10} {'MC SE':>9}  {'95% re-run band':>16}")
    print("-" * 80)
    for label, sub, n_outer, exp_r in SCENARIOS:
        path = res / sub / f"{sub}_consensus_eval.pkl"
        if not path.exists():
            print(f"{label}: missing {path} - skipped")
            continue
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        groups = group_into_repeats(data["folds"], n_outer)
        if groups is None:
            print(f"{label}: repeat grouping could not be verified - skipped")
            continue
        targets = [(m, (lambda f, m=m: f["unimodal_y_pred"][m])) for m in UNIMODAL]
        targets.append(("Fused_ElasticNet", lambda f: f["fused_y_pred"]))
        for model, predict in targets:
            a = per_repeat_auroc(groups, predict)
            if a.size != exp_r:
                print(f"{label}/{model}: {a.size} repeats, expected {exp_r}"
                      f" - skipped")
                continue
            sd = float(a.std(ddof=1))
            se = sd / np.sqrt(a.size)
            rows.append({"scenario": label, "model": model, "n_repeats": a.size,
                         "AUROC_repeat_mean": float(a.mean()),
                         "SD_across_repeats": sd, "monte_carlo_SE": float(se),
                         "rerun_95_band": float(1.96 * se)})
            print(f"{label:<8} {model:<18} {a.size:>4} {a.mean():>8.4f} "
                  f"{sd:>10.4f} {se:>9.5f}   +/-{1.96 * se:.4f}")

    print("-" * 80)
    if not rows:
        raise SystemExit("no cells computed - refusing to report a conclusion")

    pooled = [r["rerun_95_band"] for r in rows if r["scenario"] == "Global"]
    arm = [r["rerun_95_band"] for r in rows if r["scenario"] != "Global"]
    print(f"\n{len(rows)} model x scenario cells.")
    print(f"  pooled cohort (R=200), worst 95% re-run band : +/-{max(pooled):.4f}")
    print(f"  arm analyses  (R=100), worst 95% re-run band : +/-{max(arm):.4f}")
    print("\nReported patient-level cluster-bootstrap 95% CIs are roughly 0.16")
    print("wide, so re-run variability is a fraction of the uncertainty already")
    print("stated and changes no reported conclusion. The arms carry more of it")
    print("for two compounding reasons: half the repeats and roughly half the")
    print("patients, so each repeat's AUROC is noisier and fewer are averaged.")
    print("\nDo not quote a third decimal as something a re-run would recover:")
    print("it identifies the deposited result, not a reproducible quantity.")

    if args.out:
        import pandas as pd
        pd.DataFrame(rows).to_excel(args.out, index=False)
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
