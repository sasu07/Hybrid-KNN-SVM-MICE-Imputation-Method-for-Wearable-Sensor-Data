# Copyright (c) 2025 Gabriel-Vasilica Sasu
# All rights reserved.
#
# run_all.py
# ---------------------------------------------------------------------------
# Runs the full HAR70+ external-validation pipeline end to end, in order:
#   1. extract windowed features            -> data/har70_full.csv
#   2. imputation benchmark (6 methods)     -> results_*.csv
#   3. downstream activity classification   -> downstream_*.csv
#   4. ablation + runtime + per-variable    -> ablation_*.csv, runtime_*.csv, per_variable_*.csv
#
# Run from inside the package folder:   python run_all.py
# You can also run each step individually (see README).
# ---------------------------------------------------------------------------
import subprocess, sys, time, os

STEPS = [
    ("Extract HAR70+ features",       "01_extract_har70_features.py"),
    ("Imputation benchmark",          "run_benchmark.py"),
    ("Downstream classification",     "run_downstream.py"),
    ("Ablation + runtime + per-var",  "run_ablation_runtime.py"),
]

def main():
    if not os.path.isdir("har70plus"):
        sys.exit("ERROR: folder ./har70plus/ not found. Unzip the HAR70+ archive here first "
                 "so that ./har70plus/501.csv ... 518.csv exist.")
    for title, script in STEPS:
        print("\n" + "="*70)
        print(f">>> {title}  ({script})")
        print("="*70, flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, script])
        if r.returncode != 0:
            sys.exit(f"\nStep failed: {script} (exit {r.returncode}). Fix the error above and re-run.")
        print(f"--- {script} finished in {time.time()-t0:.0f}s ---", flush=True)
    print("\nALL DONE. Send back these CSVs:")
    print("  results_overall.csv, results_by_mechanism.csv, results_summary.csv,")
    print("  downstream_summary.csv, ablation_summary.csv, ablation_by_mechanism.csv,")
    print("  runtime_summary.csv, per_variable_summary.csv")

if __name__ == "__main__":
    main()
