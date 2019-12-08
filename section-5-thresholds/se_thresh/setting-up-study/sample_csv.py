import pandas as pd
import csv
from sklearn.utils import shuffle

CSV_PATH = 'use-thresh-examples.csv'
NUM_SAMPLES = 100
THRESHOLDS = [.96, .96, .97, .98, .99]

df = pd.read_csv(CSV_PATH)
out_df = pd.DataFrame()
for thresh in THRESHOLDS:
    run_df = df[df['SE_Thresh'] == thresh]
    if len(run_df) > NUM_SAMPLES:
        run_df = run_df.sample(NUM_SAMPLES)
    out_df = out_df.append(run_df)
out_df = shuffle(out_df)
out_df.to_csv('use_100_sampled_per_threshold.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
