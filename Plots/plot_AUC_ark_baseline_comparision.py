

import re
import os
import csv
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_ark_epoch_log(path):
	"""Parse the single ARK results text file and return per-dataset lists of student and teacher mAUCs by epoch.

	Returns: dict dataset -> {'student': [mAUCs], 'teacher': [mAUCs]}
	"""
	data = defaultdict(lambda: {'student': [], 'teacher': []})
	epoch_re = re.compile(r'^Epoch\s*(\d+):', re.IGNORECASE)
	mauc_re = re.compile(r'([A-Za-z0-9_]+): Student mAUC = ([0-9.]+), Teacher mAUC = ([0-9.]+)')

	with open(path, 'r', encoding='utf-8', errors='ignore') as f:
		for line in f:
			line = line.strip()
			m = mauc_re.search(line)
			if m:
				name = m.group(1)
				student = float(m.group(2))
				teacher = float(m.group(3))
				data[name]['student'].append(student)
				data[name]['teacher'].append(teacher)

	return data


def load_baseline_csvs(folder):
	"""Load baseline CSVs (epoch,mean_test_auc,...) and return dict dataset->(epochs,means)"""
	result = {}
	for csv_path in glob(os.path.join(folder, 'medmnist_baseline_logs_*_avg.csv')):
		base = os.path.basename(csv_path)
		# extract dataset name part
		m = re.match(r'medmnist_baseline_logs_(.+)_avg.csv', base)
		if not m:
			continue
		dataset = m.group(1)
		epochs = []
		means = []
		with open(csv_path, 'r') as f:
			reader = csv.DictReader([l for l in f if l.strip()])
			for row in reader:
				try:
					epochs.append(int(row['epoch']))
					means.append(float(row['mean_test_auc']))
				except Exception:
					continue
		result[dataset.upper()] = (epochs, means)
	return result


def ensure_outdir(path):
	if not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def dataset_plot_name(name):
	# map dataset keys from ARK file to baseline filenames
	mapping = {
		'PATHMNIST': 'pathmnist',
		'CHESTMNIST': 'chestmnist',
		'DERMAMNIST': 'dermamnist',
		'OCTMNIST': 'octmnist',
		'PNEUMONIAMNIST': 'pneumoniamnist',
		'RETINAMNIST': 'retinamnist',
		'BREASTMNIST': 'breastmnist',
		'BLOODMNIST': 'bloodmnist',
		'TISSUEMNIST': 'tissuemnist',
		'ORGANAMNIST': 'organamnist',
		'ORGANCMNIST': 'organcmnist',
		'ORGANS MNIST': 'organsmnist',
		'ORG ANSMNIST': 'organsmnist',
		'ORGANSMNIST': 'organsmnist'
	}
	return mapping.get(name.upper(), name.lower())


def plot_all(ark_data, baseline_data, outdir):
	ensure_outdir(outdir)
	max_epochs = 10
	for dataset, vals in ark_data.items():
		student = vals['student'][:max_epochs]
		teacher = vals['teacher'][:max_epochs]
		epochs = list(range(len(student)))

		base_key = dataset.upper()
		baseline_key = dataset.upper()
		# try normalized key
		baseline_key = dataset.upper()
		baseline_name = dataset_plot_name(dataset)
		baseline = baseline_data.get(baseline_name.upper()) or baseline_data.get(baseline_name)

		plt.figure(figsize=(8,5))
		plt.plot(epochs, student, label='ARK Student mAUC', marker='o')
		plt.plot(epochs, teacher, label='ARK Teacher mAUC', marker='s')
		if baseline:
			bepochs, bmeans = baseline
			# truncate baseline to max_epochs (baseline epochs may start at 0)
			bepochs_trunc = []
			bmeans_trunc = []
			for e, m in zip(bepochs, bmeans):
				if e < max_epochs:
					bepochs_trunc.append(e)
					bmeans_trunc.append(m)
			if bepochs_trunc:
				plt.plot(bepochs_trunc, bmeans_trunc, label='Baseline mean AUC', linestyle='--', marker='^')

		plt.xlabel('Epoch')
		plt.ylabel('AUC (mAUC)')
		plt.title(f'{dataset} AUC by epoch')
		plt.ylim(0.0, 1.0)
		plt.grid(True, linestyle='--', alpha=0.4)
		plt.legend()

		out_path = os.path.join(outdir, f'{dataset}_ark_vs_baseline.png')
		plt.tight_layout()
		plt.savefig(out_path)
		plt.close()
		print('Saved', out_path)

	# Also create a combined multi-panel figure suitable for embedding in a PDF
	create_combined_pdf(ark_data, baseline_data, outdir, max_epochs=max_epochs)


def create_combined_pdf(ark_data, baseline_data, outdir, max_epochs=10):
	"""Create a grid of subplots (one per dataset) and save as a PDF sized for paper."""
	# enforce user-requested order, append any remaining datasets after
	wanted_order = ['PATHMNIST','CHESTMNIST','DERMAMNIST','OCTMNIST','PNEUMONIAMNIST',
					'RETINAMNIST','BREASTMNIST','BLOODMNIST','TISSUEMNIST',
					'ORGANAMNIST','ORGANCMNIST','ORGANSMNIST']
	all_keys = list(ark_data.keys())
	datasets = []
	# pick keys matching wanted order (case-insensitive exact match)
	for name in wanted_order:
		for k in all_keys:
			if k.upper() == name:
				datasets.append(k)
				break
	# append any remaining datasets that weren't specified
	for k in all_keys:
		if k not in datasets:
			datasets.append(k)
	n = len(datasets)
	if n == 0:
		return

	# choose grid: aim for 3 columns
	cols = 3
	rows = (n + cols - 1) // cols

	# A4 landscape in inches ~ 11.7 x 8.3; use larger for clarity
	fig_w, fig_h = 16, 11
	fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
	for idx, ds in enumerate(datasets):
		r = idx // cols
		c = idx % cols
		ax = axes[r][c]
		student = ark_data[ds]['student'][:max_epochs]
		teacher = ark_data[ds]['teacher'][:max_epochs]
		epochs = list(range(len(student)))

		ax.plot(epochs, student, label='ARK Student', marker='o', markersize=4)
		ax.plot(epochs, teacher, label='ARK Teacher', marker='s', markersize=4)

		# lookup baseline using the same naming mapping as single plots
		baseline_name = dataset_plot_name(ds)
		baseline = (baseline_data.get(baseline_name.upper())
				or baseline_data.get(baseline_name)
				or baseline_data.get(ds.upper())
				or baseline_data.get(ds))
		if baseline:
			bepochs, bmeans = baseline
			bepochs_trunc = [e for e in bepochs if e < max_epochs]
			bmeans_trunc = bmeans[:len(bepochs_trunc)]
			if bepochs_trunc:
				ax.plot(bepochs_trunc, bmeans_trunc, label='Baseline', linestyle='--', marker='^', markersize=4)

		ax.set_title(ds)
		ax.set_ylim(0.0, 1.0)
		ax.grid(True, linestyle='--', alpha=0.4)
		if r == rows - 1:
			ax.set_xlabel('Epoch')
		if c == 0:
			ax.set_ylabel('mAUC')
		ax.set_xlim(0, max_epochs - 1)
		ax.set_xticks(list(range(max_epochs)))

	# turn off empty axes
	total_axes = rows * cols
	for idx in range(n, total_axes):
		r = idx // cols
		c = idx % cols
		axes[r][c].axis('off')

	# one legend for the whole figure
	handles, labels = axes[0][0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=3)
	plt.tight_layout(rect=[0, 0.05, 1, 1])

	out_png = os.path.join(outdir, 'all_datasets_ark_vs_baseline.png')
	# save as high-resolution PNG suitable for embedding
	fig.savefig(out_png, dpi=300)
	plt.close(fig)
	# remove old PDF if present
	old_pdf = os.path.join(outdir, 'all_datasets_ark_vs_baseline.pdf')
	if os.path.exists(old_pdf):
		try:
			os.remove(old_pdf)
		except Exception:
			pass
	print('Saved combined PNG:', out_png)


def main():
	ark_folder = os.path.join(os.path.dirname(__file__), 'ark_medmnist_epoch_logs')
	baseline_folder = os.path.join(os.path.dirname(__file__), 'medmnist_swin_baseline_logs_avg')
	outdir = os.path.join(os.path.dirname(__file__), 'plots_ark_baseline_comparision')

	# find ARK combined log file(s)
	ark_files = glob(os.path.join(ark_folder, '*_results.txt'))
	if not ark_files:
		print('No ARK epoch log files found in', ark_folder)
		return

	# parse all ARK files (if multiple we'll merge by appending epochs)
	merged = defaultdict(lambda: {'student': [], 'teacher': []})
	for af in ark_files:
		d = parse_ark_epoch_log(af)
		for k, v in d.items():
			merged[k]['student'].extend(v['student'])
			merged[k]['teacher'].extend(v['teacher'])

	baseline = load_baseline_csvs(baseline_folder)

	plot_all(merged, baseline, outdir)


if __name__ == '__main__':
	main()


