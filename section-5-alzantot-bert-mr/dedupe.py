# jm8wx 12/9/19
import glob

args_files = glob.glob('*/args.txt')

offsets = set()
offset_map = {}
for filename in args_files:
  lines = open(filename,'r').readlines()
  first_line = lines[0].split()
  # --num_examples 25 --num_examples_offset 950
  i = first_line.index('--num_examples_offset')
  offset = first_line[i+1]
  if offset in offsets:  
    print(filename, 'has a same offset as', offset_map[offset])
  else:
    offsets.add(offset)
    offset_map[offset] = filename

good_files = offset_map.values()
print(sorted(list(offset_map.keys())))

good_folders = [f.split('/')[0] for f in good_files]

print()
print(good_folders)

print()
print('\n'.join(sorted(good_folders)))
