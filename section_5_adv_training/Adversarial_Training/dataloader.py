import csv
import os

def read_examples_from_csv(csv_path, label_map, ignore_header=False):
    print(f'Loading examples from CSV at {csv_path}.')
    train_data_path = os.path.join(csv_path, 'train.csv')
    eval_data_path = os.path.join(csv_path, 'test.csv')
    
    train_text_list = []
    train_labels = []
    with open(train_data_path, "r", encoding="utf-8", errors='ignore') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        if ignore_header:
            next(reader)
        for line in reader:
            label, text = line
            if label not in label_map:
                logger.info('Label:',label)
                logger.info('Label map:', label_map)
                raise ValueError(f'No mapping for label {label} to an int')
            train_labels.append(label_map[label])
            train_text_list.append(text)
    
    eval_text_list = []
    eval_labels = []
    with open(eval_data_path, "r", encoding="utf-8", errors='ignore') as f:
        reader = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
        if ignore_header:
            next(reader)
        for line in reader:
            label, text = line
            if label not in label_map:
                logger.info('Label:',label)
                logger.info('Label map:', label_map)
                raise ValueError(f'No mapping for label {label} to an int')
            eval_labels.append(label_map[label])
            eval_text_list.append(text)
    return train_text_list, train_labels, eval_text_list, eval_labels

def get_examples_and_labels(dataset):
    if dataset == 'yelp':
        data_dir = '/net/bigtemp/jg6yd/treeattack/textdata/yelp_review_polarity_csv/'
        label_map = {
            '1': 0, # negative
            '2': 1  # positive
        }
        return read_examples_from_csv(data_dir, label_map)
    elif dataset == 'imdb':
        data_dir = '/net/bigtemp/jg6yd/treeattack/textdata/imdb_csv/'
        label_map = {
            '1': 1, # positive
            '2': 0  # negative
        }
        return read_examples_from_csv(data_dir, label_map)
    elif dataset == 'mr':
        data_dir = '/p/qdata/jm8wx/datasets/mr/sentence-level/processed/data/'
        label_map = {
            '0': 0, # negative
            '1': 1  # positive
        }
        return read_examples_from_csv(data_dir, label_map, ignore_header=True)
    elif dataset == 'mr_tf_full':
        data_dir = './mr-data/'
        label_map = {
            '0': 0, # negative
            '1': 1  # positive
        }
        return read_examples_from_csv(data_dir, label_map, ignore_header=True)
    elif dataset == 'ag_news':
        data_dir = '/bigtemp/jg6yd/treeattack/textdata/ag_news_csv'
        
        label_map = {
            '1': 0, # World
            '2': 1, # Sports
            '3': 2, # Business
            '4': 3 # Sci/Tech
        }
        
        return read_examples_from_csv(data_dir, label_map, ignore_header=True)
    raise ValueError(f'Invalid dataset {dataset}')

def get_sts_data(tokenizer, skip_unlabeled=True):
    """ load a dataset in the sts tsv format """
    s0 = []
    s1 = []
    labels = []
    with codecs.open(dsfile, encoding='utf8') as f:
        for line in f:
            line = line.rstrip()
            label, s0x, s1x = line.split('\t')
            if label == '':
                if skip_unlabeled:
                    continue
                else:
                    labels.append(-1.)
            else:
                labels.append(float(label))
            s0.append(tokenizer.encode(s0x))
            s1.append(tokenizer.encode(s1x))
    return s0, s1, np.array(labels)
