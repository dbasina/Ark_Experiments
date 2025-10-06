import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def parse_line(line):
    epoch_match = re.search(r"Epoch (\d+)", line)
    recent_match = re.search(r"Recent Dataset ([^,]+)", line)
    student_match = re.search(r"Student AUC: ([0-9.]+)", line)
    teacher_match = re.search(r"Teacher AUC: ([0-9.]+)", line)
    if epoch_match and recent_match and student_match and teacher_match:
        return {
            "epoch": int(epoch_match.group(1)),
            "recent": recent_match.group(1),
            "student_auc": float(student_match.group(1)),
            "teacher_auc": float(teacher_match.group(1)),
        }
    return None


def plot_cycle_for_test_dataset(results, test_dataset):
    # --- normalize tag we compare against (strip "MNIST")
    test_tag = test_dataset.replace('MNIST', '')

    # --- canonical order for datasets (lowercase, no MNIST)
    canonical_order = [
        'path', 'chest', 'derma', 'oct', 'pneumonia', 'retina',
        'breast', 'blood', 'tissue', 'organa', 'organc', 'organs'
    ]

    # --- group results by epoch
    from collections import defaultdict
    epoch_groups = defaultdict(list)
    for r in results:
        epoch_groups[r['epoch']].append(r)

    # --- order within each epoch by canonical_order
    ordered_results = []
    for epoch in sorted(epoch_groups.keys()):
        group = epoch_groups[epoch]
        # map: tag (lower, no MNIST) -> result
        tag2res = {r['recent'].replace('MNIST','').lower(): r for r in group}
        for tag in canonical_order:
            if tag in tag2res:
                ordered_results.append(tag2res[tag])

    # --- build plotting arrays
    x_labels = [f"E{r['epoch']}-{r['recent'].replace('MNIST','')}" for r in ordered_results]
    epochs    = [r['epoch'] for r in ordered_results]
    student_aucs = [r['student_auc'] for r in ordered_results]
    teacher_aucs = [r['teacher_auc'] for r in ordered_results]
    x = np.arange(len(x_labels))

    # --- indices where recent == test_dataset (ignoring "MNIST")
    solid_idx = [i for i, r in enumerate(results) if r['recent'].replace('MNIST','') == test_tag]
    solid_student = [student_aucs[i] for i in solid_idx]
    solid_teacher = [teacher_aucs[i] for i in solid_idx]

    # --- vertical separators when epoch changes
    epoch_change_indices = [i for i in range(1, len(epochs)) if epochs[i] != epochs[i-1]]

    # --- output dir
    out_dir = os.path.join('plots_cycle', test_tag)
    os.makedirs(out_dir, exist_ok=True)

    # =========================
    # Student figure
    # =========================
    plt.figure(figsize=(max(12, len(x_labels)//2), 6))
    plt.scatter(x, student_aucs, label='Student AUC')

    if len(x) > 1:
        plt.plot(x, student_aucs, alpha=0.7)

    # overlay solid line for points where recent == test_dataset
    if len(solid_idx) >= 2:
        plt.plot(solid_idx, solid_student, linewidth=2, marker='o', color='black',
                 label=f'{test_tag} only', zorder=5)
    elif len(solid_idx) == 1:
        plt.scatter([solid_idx[0]], [solid_student[0]], s=60, edgecolors='k', linewidths=1.5,
                    label=f'{test_tag} point', zorder=6)

    for idx in epoch_change_indices:
        plt.axvline(x=idx-0.5, linestyle='--', alpha=0.5)

    plt.ylim(0, 1)
    plt.xlabel('Epoch - Recent Dataset')
    plt.ylabel('Student AUC')
    plt.title(f'Student AUC for Test Dataset: {test_tag}')
    plt.xticks(x, x_labels, rotation=90, fontsize=8)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'student_auc_cycle.png'))
    plt.close()

    # =========================
    # Teacher figure
    # =========================
    plt.figure(figsize=(max(12, len(x_labels)//2), 6))
    plt.scatter(x, teacher_aucs, label='Teacher AUC')

    if len(x) > 1:
        plt.plot(x, teacher_aucs, alpha=0.7)

    if len(solid_idx) >= 2:
        plt.plot(solid_idx, solid_teacher, linewidth=2, marker='o', color='red',
                 label=f'{test_tag} only', zorder=5)
    elif len(solid_idx) == 1:
        plt.scatter([solid_idx[0]], [solid_teacher[0]], s=60, edgecolors='k', linewidths=1.5,
                    label=f'{test_tag} point', zorder=6)

    for idx in epoch_change_indices:
        plt.axvline(x=idx-0.5, linestyle='--', alpha=0.5)

    plt.ylim(0, 1)
    plt.xlabel('Epoch - Recent Dataset')
    plt.ylabel('Teacher AUC')
    plt.title(f'Teacher AUC for Test Dataset: {test_tag}')
    plt.xticks(x, x_labels, rotation=90, fontsize=8)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'teacher_auc_cycle.png'))
    plt.close()


def main():
    for fname in os.listdir('.'):
        if fname.endswith('_results.txt'):
            test_dataset = fname.replace('_results.txt', '')
            results = []
            with open(fname, 'r') as f:
                for line in f:
                    parsed = parse_line(line)
                    if parsed:
                        results.append(parsed)
            if results:
                plot_cycle_for_test_dataset(results, test_dataset)

if __name__ == "__main__":
    main()
    print("Cycle plots created for all test datasets.")
