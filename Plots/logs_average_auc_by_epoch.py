#!/usr/bin/env python3
"""Compute per-epoch average (and std) of test AUC across runs for each MedMNIST dataset.

Outputs a CSV per dataset with columns: epoch, mean_test_auc, std_test_auc, count
"""
import re
import os
import csv
import math
from collections import defaultdict

ROOT = os.path.dirname(__file__)
LOG_DIR = os.path.join(ROOT, "medmnist_swinbaseline_logs")
OUT_DIR = os.path.join(ROOT, "medmnist_swin_baseline_logs_avg")


def ensure_out_dir():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

try:
    from epoch_comparision import DATASETS
except Exception:
    # Fallback: simple autodetect by listing files
    DATASETS = []


def find_log_files(dataset):
    """Return list of absolute paths for log files belonging to dataset."""
    files = []
    if not os.path.isdir(LOG_DIR):
        return files
    for fn in os.listdir(LOG_DIR):
        if fn.lower().startswith(dataset.lower()) and fn.lower().endswith('.txt'):
            files.append(os.path.join(LOG_DIR, fn))
    return sorted(files)


epoch_re = re.compile(r"epoch:\s*(\d+).*?test auc:\s*([0-9.]+)", re.IGNORECASE)


def parse_log_for_test_auc(path):
    """Parse a single log file, return dict epoch->test_auc (float)."""
    results = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = epoch_re.search(line)
            if m:
                e = int(m.group(1))
                auc = float(m.group(2))
                results[e] = auc
    return results


def mean_std(values):
    n = len(values)
    if n == 0:
        return (float('nan'), float('nan'))
    mean = sum(values) / n
    if n == 1:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (mean, math.sqrt(var))


def process_dataset(dataset):
    files = find_log_files(dataset)
    if not files:
        print(f"No log files found for dataset: {dataset}")
        return None

    per_epoch_values = defaultdict(list)  # epoch -> list of test aucs from different runs

    for p in files:
        d = parse_log_for_test_auc(p)
        if not d:
            continue
        for e, auc in d.items():
            per_epoch_values[e].append(auc)

    if not per_epoch_values:
        print(f"No epoch/test auc data parsed for dataset: {dataset}")
        return None

    # Write CSV
    out_csv = os.path.join(OUT_DIR, f"medmnist_baseline_logs_{dataset}_avg.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['epoch', 'mean_test_auc', 'std_test_auc', 'count'])
        for e in sorted(per_epoch_values.keys()):
            vals = per_epoch_values[e]
            mean, std = mean_std(vals)
            writer.writerow([e, f"{mean:.6f}", f"{std:.6f}", len(vals)])

    print(f"Wrote: {out_csv}  (runs={len(files)})")
    return out_csv


def main():
    datasets = DATASETS if DATASETS else None
    if not datasets:
        # autodetect unique prefixes before first underscore or 'log'
        names = set()
        if os.path.isdir(LOG_DIR):
            for fn in os.listdir(LOG_DIR):
                if not fn.lower().endswith('.txt'):
                    continue
                base = fn.rsplit('_', 1)[0]
                names.add(base)
        datasets = sorted(names)

    if not datasets:
        print("No datasets found/defined.")
        return

    ensure_out_dir()
    outputs = []
    for ds in datasets:
        out = process_dataset(ds)
        if out:
            outputs.append(out)

    if outputs:
        print('\nSample of first output file:')
        with open(outputs[0], 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                print(line.strip())
                if i > 5:
                    break


if __name__ == '__main__':
    main()
