import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def parse_line(line):
    # Example line:
    # Epoch 0, Recent Dataset PathMNIST, Test Dataset TissueMNIST (multi-class classification): Student AUC: 0.6269, Teacher AUC: 0.6385
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

def plot_aucs(results, test_dataset):
    # Group by epoch
    epochs = sorted(set(r["epoch"] for r in results))
    # Create plots directory and subdirectory for this test dataset
    plots_dir = os.path.join("plots", test_dataset)
    os.makedirs(plots_dir, exist_ok=True)
    for auc_type in ["student_auc", "teacher_auc"]:
        plt.figure(figsize=(10, 6))
        for epoch in epochs:
            epoch_results = [r for r in results if r["epoch"] == epoch]
            x_labels = [r["recent"].replace('MNIST', '') for r in epoch_results]
            y = [r[auc_type] for r in epoch_results]
            # Use integer positions for x for plotting
            x = np.arange(len(x_labels))
            plt.scatter(x_labels, y, label=f"Epoch {epoch}")
            if len(x) > 1:
                plt.plot(x, y, linestyle='-', alpha=0.7)
        plt.ylim(0, 1)
        plt.xlabel("Recent Dataset")
        plt.ylabel(f"{auc_type.replace('_', ' ').title()}")
        # Remove MNIST from test_dataset in title
        title_test_dataset = test_dataset.replace('MNIST', '')
        plt.title(f"{auc_type.replace('_', ' ').title()} for Test Dataset: {title_test_dataset}")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"{auc_type}_per_epoch.png")
        plt.savefig(plot_path)
        plt.close()

def main():
    # Find all *_results.txt files
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
                plot_aucs(results, test_dataset)

if __name__ == "__main__":
    main()
    print("Plots created for all test datasets.")
