import numpy as np
import pandas as pd

# examples has 3907 datapoints
examples_file = 'examples.csv'

df = pd.read_csv(examples_file)

orig = df['original_text'].to_frame(name='text')
orig.insert(0, 'label', 0) # 0 means real
pert = df['perturbed_text'].to_frame(name='text')
pert.insert(0, 'label', 1) # 1 means fake

new_df = pd.concat((orig, pert))
print(new_df.sample(10).head())

# train test split thanks to stackoverflow.com/questions/24147278
train_perc = 0.90
msk = np.random.rand(len(new_df)) < train_perc
train = new_df[msk]
test = new_df[~msk]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)