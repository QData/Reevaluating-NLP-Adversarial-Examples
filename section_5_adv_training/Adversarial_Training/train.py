import argparse
import datetime
import logging
import numpy as np
import os
import random
import time
import torch
import tqdm
import wandb

from dataloader import get_examples_and_labels
from multiprocessing import Pool, cpu_count
from tensorboardX import SummaryWriter
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from transformers import BertForSequenceClassification
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from textattack.augmentation import EmbeddingAugmenter

wandb.init(project='augmentation2', sync_tensorboard=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('root')

# Print a "Loading" message, since loading actually takes awhile.
logger.info('Loading...')

# Make the HuggingFace logger be quiet.
logging.getLogger("transformers.tokenization_bert").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_dir', type=str, default=None, 
        help='directory of model to train')
    
    parser.add_argument('--dataset', type=str,
        default='mr', help='dataset for training, like \'yelp\'')
    
    parser.add_argument('--augmentation_recipe', type=str, help='type of augmentation: {tf-adjusted, textfooler}')
    parser.add_argument('--augmentation_num', type=int, default=-1, help='num examples for augmentation (-1 for all)')
    
    parser.add_argument('--logging_steps', type=int, 
        default=500, help='log model after this many steps')
    
    parser.add_argument('--checkpoint_steps', type=int, 
        default=5000, help='save model after this many steps')
    
    parser.add_argument('--checkpoint_every_epoch', action='store_true',
        default=True, help='save model checkpoint after this many steps')
    
    parser.add_argument('--output_prefix', type=str,
        default='', help='prefix for model saved output')
    
    parser.add_argument('--cased', action='store_true', default=False,
         help='if true, bert is cased, if false, bert is uncased')
    
    parser.add_argument('--debug_cuda_memory', type=str,
        default=False, help='Print CUDA memory info periodically')
    
    parser.add_argument('--num_train_epochs', '--epochs', type=int, 
        default=100, help='Total number of epochs to train for')
        
    parser.add_argument('--early_stopping_epochs', type=int, 
        default=-1, help='Number of epochs validation must increase'
                           ' before stopping early')
        
    parser.add_argument('--batch_size', type=int, default=128, 
        help='Batch size for training')
        
    parser.add_argument('--max_seq_len', type=int, default=128, 
        help='Maximum length of a sequence (anything beyond this will '
             'be truncated) - '
             '# BERT\'s max seq length is 512 so can\'t go higher than that.')
        
    parser.add_argument('--learning_rate', '--lr', type=float, default=2e-5, 
        help='Learning rate for Adam Optimization')
        
    parser.add_argument('--tb_writer_step', type=int, default=1000, 
        help='Number of steps before writing to tensorboard')
        
    parser.add_argument('--grad_accum_steps', type=int, default=1, 
        help='Number of steps to accumulate gradients before optimizing, '
                'advancing scheduler, etc.')
        
    parser.add_argument('--warmup_proportion', type=int, default=0.1, 
        help='Warmup proportion for linear scheduling')
        
    parser.add_argument('--config_name', type=str, default='config.json', 
        help='Filename to save BERT config as')
        
    parser.add_argument('--weights_name', type=str, default='pytorch_model.bin', 
        help='Filename to save model weights as')

    parser.add_argument('--random_seed', type=int, default=42, help='random seed to set')
    
    args = parser.parse_args()
    
    if not args.model_dir:
        args.model_dir = 'bert-base-cased' if args.cased else 'bert-base-uncased'
    
    if args.output_prefix: args.output_prefix += '-'
    
    cased_str = '-' + ('cased' if args.cased else 'uncased')
    date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
    root_output_dir = 'outputs'
    args.output_dir = os.path.join(root_output_dir, 
        f'{args.output_prefix}{args.dataset}{cased_str}-{date_now}/')
    
    # Use multiple GPUs if we can!
    args.num_gpus = torch.cuda.device_count()
    
    # set random seed
    set_seed(args.random_seed)
    
    return args

def print_cuda_memory(args):
    if args.debug_cuda_memory:
        logger.info('*** CUDA STATS')
        logger.info('*** current memory allocated: {} MiB'.format(torch.cuda.memory_allocated() / 1024 ** 2))
        logger.info('*** max memory allocated: {} MiB'.format(torch.cuda.max_memory_allocated() / 1024 ** 2))
        logger.info('*** cached memory: {} MiB'.format(torch.cuda.memory_cached() / 1024 ** 2))

def main():
    start_time = time.time()
    args = parse_args()
    make_directories(args.output_dir)
    
    # Start Tensorboard and log hyperparams.
    tb_writer = SummaryWriter(args.output_dir)
    tb_writer.add_hparams(vars(args), {})
    
    file_log_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logger.addHandler(file_log_handler)
    
    # Get list of text and list of label (integers) from disk.
    train_text, train_label_id_list, eval_text, eval_label_id_list = \
        get_examples_and_labels(args.dataset)
    
    # Augment training data.
    if (args.augmentation_recipe is not None) and len(args.augmentation_recipe):
        import pandas as pd
        
        if args.augmentation_recipe =='textfooler':
            aug_csv = '/p/qdata/jm8wx/research/text_attacks/textattack/outputs/attack-1590551967800.csv'
        elif args.augmentation_recipe == 'tf-adjusted':
            aug_csv = '/p/qdata/jm8wx/research/text_attacks/textattack/outputs/attack-1590564015768.csv'
        else:
            raise ValueError(f'Unknown augmentation recipe {args.augmentation_recipe}')
        
        aug_df = pd.read_csv(aug_csv)
        
        # filter skipped outputs
        aug_df = aug_df[aug_df['original_text'] != aug_df['perturbed_text']]
        
        print(f'Augmentation recipe {args.augmentation_recipe} / augmentation num. examples {args.augmentation_num}/ len {len(aug_df)}')
        
        original_text  = aug_df['original_text']
        perturbed_text = aug_df['perturbed_text']
        
        # convert `train_text` and `train_label_id_list` to an np array so things are faster
        train_text = np.array(train_text)
        train_label_id_list = np.array(train_label_id_list)
        
        x_adv_list = []
        x_adv_id_list = []
        for (x, x_adv) in zip(original_text, perturbed_text):
            x = x.replace('[[', '').replace(']]', '')
            x_adv = x_adv.replace('[[', '').replace(']]', '')
            x_idx = (train_text == x).nonzero()[0][0]
            x_adv_label = train_label_id_list[x_idx]
            x_adv_id_list.append(x_adv_label)
            x_adv_list.append(x_adv)
            
        # truncate to `args.augmentation_num` examples
        if (args.augmentation_num >= 0):
            perm = list(range(len(x_adv_list)))
            random.shuffle(perm)
            perm = perm[:args.augmentation_num]
            x_adv_list = [x_adv_list[i] for i in perm]
            x_adv_id_list = [x_adv_id_list[i] for i in perm]
        
        train_text = train_text.tolist() + x_adv_list
        train_label_id_list = train_label_id_list.tolist() + x_adv_id_list
        
        print(f'Augmentation added {len(x_adv_list)} examples, for a total of {len(train_text)}')
        
    
        
    label_id_len = len(train_label_id_list)
    num_labels = len(set(train_label_id_list))
    logger.info('num_labels: %s', num_labels)
    
    train_examples_len = len(train_text)
    
    if len(train_label_id_list) != train_examples_len:
        raise ValueError(f'Number of train examples ({train_examples_len}) does not match number of labels ({len(train_label_id_list)})')
    if len(eval_label_id_list) != len(eval_text):
        raise ValueError(f'Number of teste xamples ({len(eval_text)}) does not match number of labels ({len(eval_label_id_list)})')
    
    print_cuda_memory(args)
     # old INFO:__main__:Loaded data and tokenized in 189.66675066947937s
    
        # @TODO support other vocabularies, or at least, support case
    tokenizer = BertWordPieceTokenizer('bert-base-uncased-vocab.txt', lowercase=True)
    tokenizer.enable_padding(max_length=args.max_seq_len)
    tokenizer.enable_truncation(max_length=args.max_seq_len)
    
    logger.info(f'Tokenizing training data. (len: {train_examples_len})')
    train_text_ids = [encoding.ids for encoding in tokenizer.encode_batch(train_text)]
    logger.info(f'Tokenizing test data (len: {len(eval_label_id_list)})')
    eval_text_ids = [encoding.ids for encoding in tokenizer.encode_batch(eval_text)]
    load_time = time.time()
    logger.info(f'Loaded data and tokenized in {load_time-start_time}s')
    
    print_cuda_memory(args)
    
    # Load pre-trained model tokenizer (vocabulary)
    logger.info('Loading model: %s', args.model_dir)
    # Load pre-trained model (weights)
    logger.info(f'Model class: (vanilla) BertForSequenceClassification.')
    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=num_labels)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.to(device)
    # print(model)
    
    # multi-gpu training
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
    logger.info(f'Training model across {args.num_gpus} GPUs')
    
    num_train_optimization_steps = int(
        train_examples_len / args.batch_size / args.grad_accum_steps) * args.num_train_epochs
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=args.learning_rate)
                         
    scheduler = get_linear_schedule_with_warmup(optimizer, 
        num_warmup_steps=args.warmup_proportion, 
        num_training_steps=num_train_optimization_steps)
    
    global_step = 0
    
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_examples_len)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Max sequence length = %d", args.max_seq_len)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    
    wandb.log({'train_examples_len': train_examples_len})
    
    train_input_ids = torch.tensor(train_text_ids, dtype=torch.long)
    train_label_ids = torch.tensor(train_label_id_list, dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    eval_input_ids = torch.tensor(eval_text_ids, dtype=torch.long)
    eval_label_ids = torch.tensor(eval_label_id_list, dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_label_ids)
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)
    
    def get_eval_acc():
        correct = 0
        total = 0
        for input_ids, label_ids in tqdm.tqdm(eval_dataloader, desc="Evaluating accuracy"):
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)
        
            with torch.no_grad():
                    logits = model(input_ids)[0]
                
            correct += (logits.argmax(dim=1)==label_ids).sum()
            total += len(label_ids)
        
        return float(correct) / total
    
    
    def save_model():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        
        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, args.weights_name)
        output_config_file = os.path.join(args.output_dir, args.config_name)
        
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        
        logger.info(f'Best acc found. Saved tokenizer, model config, and model to {args.output_dir}.')
    
    global_step = 0
    def save_model_checkpoint(checkpoint_name=None):
        # Save model checkpoint
        checkpoint_name = checkpoint_name or 'checkpoint-{}'.format(global_step)
        output_dir = os.path.join(args.output_dir, checkpoint_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model  
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info('Checkpoint saved to %s.', output_dir)
    
    print_cuda_memory(args)
    model.train()
    best_eval_acc = 0
    steps_since_best_eval_acc = 0
    
    def loss_backward(loss):
        if args.num_gpus > 1:
            loss = loss.mean() # mean() to average on multi-gpu parallel training
        if args.grad_accum_steps > 1:
            loss = loss / args.grad_accum_steps
        loss.backward()
    
    for epoch in tqdm.trange(int(args.num_train_epochs), desc="Epoch"):
        prog_bar = tqdm.tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(prog_bar):
            print_cuda_memory(args)
            batch = tuple(t.to(device) for t in batch)
            input_ids, labels = batch
            logits = model(input_ids)[0]
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = torch.nn.CrossEntropyLoss()(
                logits.view(-1, num_labels), labels.view(-1))
            if global_step % args.tb_writer_step == 0:
                tb_writer.add_scalar('loss', loss, global_step)
                tb_writer.add_scalar('lr', loss, global_step)
            loss_backward(loss)
            prog_bar.set_description(f"Loss {loss.item()}")
            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            # Save model checkpoint to file.
            if global_step % args.checkpoint_steps == 0:
                save_model_checkpoint()
            
            model.zero_grad()
            
            # Inc step counter.
            global_step += 1
        
        # Check accuracy after each epoch.
        eval_acc = get_eval_acc()
        tb_writer.add_scalar('epoch_eval_acc', eval_acc, global_step)
        wandb.log({'epoch_eval_acc': eval_acc, 'epoch': epoch})
        
        if args.checkpoint_every_epoch:
            save_model_checkpoint(f'epoch-{epoch}')
                    
        logger.info(f'Eval acc: {eval_acc*100}%')
        if eval_acc > best_eval_acc:
            best_eval_acc = eval_acc
            steps_since_best_eval_acc = 0
            save_model()
        else:
            steps_since_best_eval_acc += 1
            if (args.early_stopping_epochs > 0) and (steps_since_best_eval_acc > args.early_stopping_epochs):
                logger.info(f'Stopping early since it\'s been {args.early_stopping_epochs} steps since validation acc increased')
                break


if __name__ == '__main__': main()
