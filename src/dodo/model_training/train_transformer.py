import os
import argparse
import pandas as pd
import numpy as np
import json
import shutil
from sklearn.metrics import f1_score
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
from dodo.utils.helper_functions import return_relative_dirs, check_dir_exists, blob_model_saver

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_PROJECT"] = "dodo-experiments"

def load_and_sample_data(data_dir, dodos, ratios, data_order, label_col):
    data = {}
    for dataset in dodos:
        data[dataset] = {}
        # Load data from file
        for split in ['train', 'test', 'validation']:
            data[dataset][split] = pd.read_csv(f'{data_dir}/{dataset}/{split}.csv')
    
    sampled_data = {}
    for dataset, ratio in ratios.items():
        sampled_data[dataset] = {}
        for split in ['train', 'test', 'validation']:
            # Load first n rows sample for train and val
            n_rows = int(len(data[dataset][split]) * ratio)
            prop_df = data[dataset][split].iloc[:n_rows][['text', label_col, 'tweet_id', 'dodo']]
            prop_df['selected'] = True
            # When split = test, take full test sets but indicate in-selection examples
            if split == 'test':
                full_df = data[dataset][split][['text', label_col, 'tweet_id', 'dodo']]
                merge_df = full_df.merge(prop_df['selected'], how = 'left', left_index = True, right_index = True)
                merge_df['selected'].fillna(False, inplace=True)
                sampled_data[dataset][split] = merge_df
            else:
                sampled_data[dataset][split] = prop_df
    
    # Concatenate dodo frames
    test_data = pd.concat([sampled_data[k]['test'] for k in ['fb_m', 'fb_w', 'mp_m', 'mp_w']]) #.sample(5)
    train_data = pd.concat([sampled_data[k]['train'] for k in ['fb_m', 'fb_w', 'mp_m', 'mp_w']]) #.sample(5)
    val_data = pd.concat([sampled_data[k]['validation'] for k in ['fb_m', 'fb_w', 'mp_m', 'mp_w']]) #.sample(5)
    split_names = ['train', 'val', 'test']
    
    for data_split, split_name in zip([train_data, val_data, test_data], split_names):
        print(f'\nSPLIT = {split_name}')
        value_counts = data_split["dodo"].value_counts(normalize = True)
        label_counts = data_split[label_col].value_counts()
        n_entries = len(data_split)
        print(f'Entries = {n_entries}, Ratios = {dict(value_counts)}, Labels = {dict(label_counts)}')

    # Only shuffle train and val if specified
    if data_order == 'shuffled':
        train_data = train_data.sample(frac=1, random_state=123).reset_index(drop=True)
        val_data = val_data.sample(frac=1, random_state=123).reset_index(drop=True)
    
    return train_data, val_data, test_data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience):
        super().__init__(early_stopping_patience)
        self.stopping_epoch = None

    def on_train_end(self, args, state, control, **kwargs):
        # Call the original implementation
        super().on_train_end(args, state, control, **kwargs)

        # Save the stopping epoch as an attribute
        self.stopping_epoch = state.epoch

class BestModelEpochCallback(TrainerCallback):
    def __init__(self, best_model_metric, greater_is_better=True):
        self.best_model_metric = best_model_metric
        self.greater_is_better = greater_is_better
        self.best_epoch = None
        self.best_metric = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.best_metric is None or \
           (self.greater_is_better and metrics[self.best_model_metric] > self.best_metric) or \
           (not self.greater_is_better and metrics[self.best_model_metric] < self.best_metric):
            self.best_metric = metrics[self.best_model_metric]
            self.best_epoch = state.epoch
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {'macro_f1': macro_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subexperiment_name', type=str, required=True, help='Name of subexperiment for storing results')
    parser.add_argument('--batch_name', type=str, required=True, help='Name of experiment batch')
    parser.add_argument('--experiment_family', type=str, required=True, help='Type of experiment batch to run')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--ratios', type=str, required=True, help='Sampling ratios as a JSON string')
    parser.add_argument('--data_order', type=str, required=True, help='Whether to take shuffle or ordered data')
    parser.add_argument('--label_column', type=str, required=True, help='Name of column with class labels')
    parser.add_argument('--num_labels', type=int, required=True, help='Number of class labels')
    parser.add_argument('--num_epochs', type=int, default = 5, help='Number of epochs')
    parser.add_argument('--es_patience', type=int, default = 1, help='Early stopping patience')
    parser.add_argument('--best_model_metric', type=str, default = 'eval_macro_f1', help='Metric for best model')
    parser.add_argument('--eval_strategy', type = str, default = 'epoch', help = 'Eval and save strategy')
    parser.add_argument('--report_to', type = str, default = 'wandb', required = True, help = 'Monitoring reporting')

    args = parser.parse_args()

    print(f"{'-'*25}\nArguments:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    print(f"{'-'*25}\n")

    # Set directories
    repo_dir, working_dir = return_relative_dirs('dodo_learning')
    data_dir = f'{repo_dir}/data/labelled_data/splits'
    print(f"{'-'*25}\nDirectories:")
    print(f'repo_dir: {repo_dir}')
    print(f'working dir: {working_dir}')
    print(f'data_dir: {data_dir}')
    print(f"{'-'*25}\n")

    # Set logging inputs
    subexperiment_name = args.subexperiment_name
    batch_name = args.batch_name
    experiment_family = args.experiment_family

    # Set model inputs
    model_name = args.model
    seed = args.seed
    
    # Set data inputs
    label_col = args.label_column
    num_labels = args.num_labels
    ratios = json.loads(args.ratios)
    data_order = args.data_order
    dodos = list(ratios.keys())

    # Set training args
    num_epochs = args.num_epochs
    es_patience = args.es_patience
    best_model_metric = args.best_model_metric
    if best_model_metric == 'eval_macro_f1':
        greater_is_better = True
    elif best_model_metric == 'eval_loss':
        greater_is_better = False
    eval_strategy = args.eval_strategy
    report_to = args.report_to

    # GPU check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Set seed
    torch.manual_seed(seed)

    # Load and sample data
    print(f"{'-'*25}\nLoading Data...")
    train_data, val_data, test_data = load_and_sample_data(data_dir, dodos, ratios, data_order, label_col)

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add the special tokens
    special_tokens = {'additional_special_tokens': ['[PLAYER]', '[MP]', '[USER]', '[BODY]', '[CLUB]', '[URL]']}
    tokenizer.add_special_tokens(special_tokens)

    train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_data['text'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)

    # Prepare datasets
    train_dataset = Dataset(train_encodings, train_data[label_col].tolist())
    val_dataset = Dataset(val_encodings, val_data[label_col].tolist())
    test_dataset = Dataset(test_encodings, test_data[label_col].tolist())

    # Load PT model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Resize the token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Set Logging
    config = {}
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    if report_to == 'wandb':
        run = wandb.init(project=f"{batch_name}", 
           name=subexperiment_name,
           tags=batch_name.split('-'),
           config=config)
    
    
    # Set training args
    training_args = TrainingArguments(
        output_dir=f'{working_dir}/checkpoints',
        num_train_epochs=num_epochs,
        auto_find_batch_size= True,
        logging_dir=f'{repo_dir}/experiments/{experiment_family}/{batch_name}/logs',
        do_eval=True,
        load_best_model_at_end = True,
        evaluation_strategy=eval_strategy,
        metric_for_best_model = best_model_metric,
        greater_is_better = greater_is_better,
        save_strategy = eval_strategy,
        logging_strategy = eval_strategy,
        seed = seed,
        optim='adamw_torch',
        report_to=report_to,
        run_name=subexperiment_name,
        save_total_limit=1
    )


    # Set callbacks
    custom_es_callback = CustomEarlyStoppingCallback(early_stopping_patience=es_patience)
    best_model_epoch_callback = BestModelEpochCallback(best_model_metric=best_model_metric, greater_is_better=greater_is_better)

    # Set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks = [custom_es_callback, best_model_epoch_callback])

    # Train 
    print(f"{'-'*25}\nTraining...")
    train_output = trainer.train()
    metrics = train_output.metrics
    runtime = metrics['train_runtime']
    stopping_epoch = custom_es_callback.stopping_epoch
    best_model_epoch = best_model_epoch_callback.best_epoch
    print(f'Runtime: {runtime}')
    print(f'Stopping epoch: {stopping_epoch}')
    print(f'Best model epoch: {best_model_epoch}')


    # Save the fine-tuned model locally
    print(f"{'-'*25}\nSaving model...")
    model_dir = f'{working_dir}/tmp_model_files'
    check_dir_exists(model_dir)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    # Save to blob
    out_dir = f'fine_tuned_models/{experiment_family}/{batch_name}/{subexperiment_name}'
    blob_model_saver(model_dir, out_dir)


    # Evaluate and save test set predictions
    print(f"{'-'*25}\nEvaluating...")
    test_logits = trainer.predict(test_dataset).predictions
    print(f"Length of test_logits: {len(test_logits)}")
    test_preds = np.argmax(test_logits, axis=1)
    test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()

    # Save results
    print(f"{'-'*25}\nSaving preds...")
    out_dir = f'{repo_dir}/experiments/{experiment_family}/results/{batch_name}'
    check_dir_exists(out_dir)
    output_df = pd.DataFrame(test_probs, columns=[f'prob_class_{i}' for i in range(num_labels)])
    output_df['predicted_class'] = test_preds
    output_df['true_class'] = test_data[label_col].reset_index(drop=True)
    output_df['tweet_id'] = test_data['tweet_id'].reset_index(drop=True)  # Add tweet_id column
    output_df['selected'] = test_data['selected'].reset_index(drop=True)
    output_df['dodo'] = test_data['dodo'].reset_index(drop=True)
    output_df['total_runtime'] = runtime
    output_df['stopping_epoch'] = stopping_epoch
    output_df['best_model_epoch'] = best_model_epoch
    # Save args
    for k,v in config.items():
        if k!='ratios':
            output_df[k] = v
    output_df.to_csv(f'{out_dir}/{subexperiment_name}.csv', index=False)

    # Remove checkpoint files
    print(f"{'-'*25}\nRemoving temp files...")
    shutil.rmtree(f'{working_dir}/checkpoints')
    shutil.rmtree(f'{working_dir}/tmp_model_files')

    if report_to == 'wandb':
        run.finish()

if __name__ == '__main__':
    main()