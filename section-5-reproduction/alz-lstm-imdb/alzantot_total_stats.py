# jm8wx 11/22/19
# -- and edited 12/3/19

import glob
import os
import re
import sys

succ_thresh = 0.25

try:
  print('reading final.txt from folder', sys.argv[1])
except IndexError:
  print('please provide an argument with the folder containing final.txt')
  exit()

print('success threshold =', succ_thresh)

exp = "^\[[0-9]+\] Words.*$"

folder = sys.argv[1]
file_paths = glob.glob(os.path.join(folder, '*'))

p = [] # list of % words modified for each sample

total_words = 0; total_swaps = 0
successes = 0;  failures = 0
for file_path in file_paths:
  lines = open(file_path).readlines()
  for line in lines:
    if re.match(exp, line):
      tokens = line.strip().split()
      # print('line:',tokens)
      words = int(tokens[2])
      swaps = int(tokens[4])
      total_words += words ; total_swaps += swaps
      perc_mod = float(swaps) / float(words)
      if(perc_mod == 0): import pdb; pdb.set_trace()
      if perc_mod >= succ_thresh: 
        failures += 1
        continue
      else:
        successes += 1
      p.append(perc_mod)

print('read', len(p),'thingies')

import statistics
mean_p = statistics.mean(p)

print('average % words perturbed over', len(p),'samples:', mean_p * 100.0)

print('success rate: {}/{} or {}'.format(successes, failures+successes, successes/float(failures+successes)*100.0))
