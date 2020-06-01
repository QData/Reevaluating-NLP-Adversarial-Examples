import pandas as pd
import csv
from sklearn.utils import shuffle

CSV_PATH = 'alz_mr_examples.csv'
NUM_SAMPLES = 32

df = pd.read_csv(CSV_PATH)
out_df = df
out_df = out_df.sample(NUM_SAMPLES)
out_df = shuffle(out_df)
out_df.to_csv('alz_mr_examples_sampled.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
