import glob
import os
import sys

try:
    folder = sys.argv[1]
except IndexError:
    print('please run this script like `python avg_results.py <foldername>`')
    exit()

print('reading from folder', folder)

filenames = glob.glob(os.path.join(folder, 'out*.txt'))

all_samples = []
all_accs = []
all_pert_percs = []
all_queries = []
for filename in filenames:
    lines = open(filename).readlines()
    samples = None
    acc = None
    pertperc = None
    queries = None
    for line_raw in lines:
        line = line_raw.strip()
        if line.startswith('Number of successful attacks:'):
            samples = int(line.split()[-1])
        if line.startswith('Accuracy under attack:'):
            acc = float(line.split()[-1][:-1])
        if line.startswith('Average perturbed word %:'):
            pertperc = float(line.split()[-1][:-1])
        if line.startswith('Avg num queries:'):
            queries = float(line.split()[-1])
    if (samples is None) or (acc is None) or (pertperc is None) or (queries is None):
        raise ValueError('Bad file', filename)
    all_samples.append(samples)
    all_accs.append(acc)
    all_pert_percs.append(pertperc)
    all_queries.append(queries)


avg_acc = 0
avg_pert_perc = 0
avg_queries = 0
total_samples = 0
for num_samples, acc, pert_perc, queries in zip(all_samples, all_accs, all_pert_percs, all_queries):
    total_samples += num_samples
    avg_acc += (acc * num_samples)
    avg_pert_perc += (pert_perc * num_samples)
    avg_queries += (queries * num_samples)

avg_acc /= float(total_samples)
avg_pert_perc /= float(total_samples)
avg_queries /= float(total_samples)

print('Number of successful attacks:', total_samples, '(across', len(filenames), 'files)')
print('Accuracy under attack:', avg_acc)
print('Average perturbed word %:', avg_pert_perc)
print('Avg num queries:', avg_queries)
