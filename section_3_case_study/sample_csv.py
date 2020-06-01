import pandas as pd
import csv
from sklearn.utils import shuffle

CSV_PATH = 'case_study_examples.csv'
NUM_SAMPLES = 100
DATASETS = ['movie_review','yelp_polarity','imdb','imdb']
ATTACK_TYPE = ['jin','jin','jin','alzantot']

df = pd.read_csv(CSV_PATH)
out_df = pd.DataFrame()
for i in range(len(DATASETS)):
    run_df = df[(df['dataset'] == DATASETS[i]) & (df['run_type'] == ATTACK_TYPE[i])]
    if len(run_df) > NUM_SAMPLES:
        run_df = run_df.sample(NUM_SAMPLES)
    out_df = out_df.append(run_df)
out_df = shuffle(out_df)
out_df.to_csv('100_sampled_per_run.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
