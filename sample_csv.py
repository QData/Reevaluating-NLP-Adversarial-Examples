import pandas as pd

CSV_PATH = 'case_study_examples.csv'
NUM_SAMPLES = 100
DATASETS = ['movie_review','yelp_polarity','imdb','imdb']
ATTACK_TYPE = ['mit','mit','mit','ucla']

df = pd.read_csv(CSV_PATH)
out_df = pd.DataFrame()
for i in range(len(DATASETS)):
    run_df = df[(df['dataset'] == DATASETS[i]) & (df['run_type'] == ATTACK_TYPE[i])]
    if ATTACK_TYPE[i] == 'mit':
        run_df['run_type'] = 'jin'
    else:
        run_df['run_type'] = 'alzantot'
    print(run_df)
    # sample_df = run_df.sample(NUM_SAMPLES)
    out_df = out_df.append(run_df)
out_df = out_df.reset_index(drop=True)
out_df.to_csv('fixed_case_study_examples.csv', index=False)
