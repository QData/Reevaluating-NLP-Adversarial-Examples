# jm8wx 11/22/19

import os
import sys

succ_thresh = 0.20

try:
  print('reading final.txt from folder', sys.argv[1])
except IndexError:
  print('please provide an argument with the folder containing final.txt')
  exit()

print('success threshold =', succ_thresh)

file_path = os.path.join(sys.argv[1], 'final.txt')
lines = open(file_path).readlines()

class1_color = '\x1b[92m'
class2_color = '\x1b[91m'

out = []
for i in range(2, len(lines), 4):
    print(lines[i])
    num_words = lines[i].count(' ')
    c1_count = lines[i].count(class1_color)
    c2_count = lines[i].count(class2_color)
    num_swaps = max(c1_count, c2_count)
    if num_swaps==0: import pdb; pdb.set_trace()
    out.append((num_words, num_swaps))

for i,(w,s) in enumerate(out):
    perc = s / float(w)
    succ_str = 'Succeeded' if perc <= succ_thresh else 'Failed'
    print(f'[{i}]', 'Words:',w, 'Swaps:', s, 'Perc:', perc, succ_str)

