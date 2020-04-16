import os
import pandas as pd
import csv

input_path = '/p/qdata/jm8wx/research/text_attacks/acl_2020_results/s5-alz-bert-mr/'
output_path = 'alz_mr_examples.csv'

def pwp(t1,t2):
    diff = 0
    w1 = t1.split(' ')
    w2 = t2.split(' ')
    for i in range(len(w1)):
        if w1[i] != w2[i]:
            diff+=1
    return diff / len(w1)

def main():
    df = pd.DataFrame()
    num_qs = []
    # folders = os.listdir(input_path)
    # folders from dedupe
    folders = ['attack-2019-12-08-22:34:23-923356', 'attack-2019-12-08-17:18:03-220311', 'attack-2019-12-08-17:16:24-983551', 'attack-2019-12-08-21:52:17-766437', 'attack-2019-12-08-17:49:52-315811', 'attack-2019-12-08-17:57:33-805298', 'attack-2019-12-08-17:14:39-301279', 'attack-2019-12-08-17:33:25-663849', 'attack-2019-12-09-03:14:04-360069', 'attack-2019-12-08-17:35:25-738573', 'attack-2019-12-08-17:40:24-429223', 'attack-2019-12-08-17:29:56-552997', 'attack-2019-12-08-18:37:28-891468', 'attack-2019-12-08-17:29:51-892332', 'attack-2019-12-08-22:33:17-473491', 'attack-2019-12-08-22:34:29-972407', 'attack-2019-12-08-17:40:48-095518', 'attack-2019-12-09-03:12:00-462270', 'attack-2019-12-08-17:32:55-653956', 'attack-2019-12-08-17:08:42-307895']
    for run_folder in folders:
        run_path = os.path.join(input_path, run_folder)
        run_files = os.listdir(run_path)
        # files from dedupe
        if 'final.txt' not in run_files:
            continue
        lines = open(os.path.join(run_path,'final.txt'),'r').readlines()
        num_q = float(lines[-1].split()[-1])
        num_qs.append(num_q)
        args_path = os.path.join(run_path,'args.txt')
        args_lines = open(args_path, 'r').readlines()
        se_thresh_idx = args_lines[0].find('alz-adjusted') + 13
        se_thresh = args_lines[0][se_thresh_idx:se_thresh_idx+4]
        words = args_lines[0].split()
        dataset = ''
        for word in words:
            if word[:5] == 'bert-':
                dataset = word[5:]
        for f in os.listdir(run_path):
            if f.find('.csv') == -1:
                continue
            add_df = pd.read_csv(os.path.join(run_path,f), index_col=0)
            if not len(add_df):
                continue
            add_df['SE_Thresh'] = float(se_thresh)
            add_df['dataset'] = dataset
            add_df['SE_Model'] = 'BERT'
            df = df.append(add_df)
    
    df.to_csv('no_dedup_'+output_path, quoting=csv.QUOTE_NONNUMERIC)
    df = df.drop_duplicates(('passage_1'))
    pws = []
    for i, row in df.iterrows():
        pws.append(pwp(row['passage_1'],row['passage_2']))
    df.to_csv(output_path, quoting=csv.QUOTE_NONNUMERIC)
    print('avg pert word %:', sum(pws)/len(pws))
    print('query num:', sum(num_qs)/len(num_qs))
if __name__ == '__main__':
    main()
