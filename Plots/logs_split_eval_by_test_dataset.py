import os
from collections import defaultdict

# Path to the evaluation file
input_file = "Mid_Epoch_Eval.txt"
output_dir = os.path.dirname(input_file)

# Dictionary to hold lines for each test dataset
dataset_lines = defaultdict(list)

def extract_test_dataset(line):
    # Example: 'Test Dataset PathMNIST (multi-class classification):'
    try:
        start = line.index("Test Dataset ") + len("Test Dataset ")
        end = line.index(" (", start)
        return line[start:end]
    except ValueError:
        return None

# Read and group lines by test dataset
with open(input_file, "r") as f:
    for line in f:
        dataset = extract_test_dataset(line)
        if dataset:
            dataset_lines[dataset].append(line)

# Write each dataset's results to its own file
for dataset, lines in dataset_lines.items():
    # Clean up dataset name for filename
    safe_name = dataset.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(output_dir, f"{safe_name}_results.txt")
    with open(out_path, "w") as out_f:
        out_f.writelines(lines)

print("Done! Created one file per test dataset.")
