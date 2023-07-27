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

def load_n_rows_data(data_dir, dodos, adapt_dodo, n_add_train):
    sampled_data = {}
    for dataset in dodos:
        sampled_data[dataset] = {}
        # Load data from file
        for split in ['train', 'test', 'validation']:
            if split == 'train':
                sampled_data[dataset][split] = pd.read_csv(f'{data_dir}/{dataset}/{split}.csv', nrows = n_add_train)
            else:
                sampled_data[dataset][split] = pd.read_csv(f'{data_dir}/{dataset}/{split}.csv')
    
    # Concatenate dodo frames
    # Take all test
    test_data = pd.concat([sampled_data[k]['test'] for k in dodos])
    # Take all val data for adapt dodo
    val_data = pd.concat([sampled_data[adapt_dodo]['validation']]) #.sample(5)
    # Take n_rows of train data
    train_data = pd.concat([sampled_data[adapt_dodo]['train']])

    split_names = ['train', 'val', 'test']
    
    for data_split, split_name in zip([train_data, val_data, test_data], split_names):
        print(f'\nSPLIT = {split_name}')
        value_counts = data_split["dodo"].value_counts(normalize = True)
        n_entries = len(data_split)
        print(f'Entries = {n_entries}, Ratios = {dict(value_counts)}')
        print(data_split[['tweet_id', 'dodo']].head(2))
        print(data_split[['tweet_id', 'dodo']].tail(2))
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

    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {'macro_f1': macro_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subexperiment_name', type=str, required=True, help='Name of subexperiment for storing results')
    parser.add_argument('--batch_name', type=str, required=True, help='Name of experiment batch')
    parser.add_argument('--subbatch_name', type=str, required=True, help='Name of experiment batch')
    parser.add_argument('--experiment_family', type=str, required=True, help='Type of experiment batch to run')
    parser.add_argument('--model_path', type=str, required=True, help='Path to fine-tuned model')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--n_add_train', type = int, required = True, help = 'Number of added training rows')
    parser.add_argument('--adapt_dodo', type = str, required = True, help = 'Dataset dodo to adapt to')
    parser.add_argument('--label_column', type=str, required=True, help='Name of column with class labels')
    parser.add_argument('--num_labels', type=int, required=True, help='Number of class labels')
    parser.add_argument('--num_epochs', type=int, default = 5, help='Number of epochs')
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
    subbatch_name = args.subbatch_name
    experiment_family = args.experiment_family

    # Set model inputs
    seed = args.seed
    model_path = args.model_path
    
    # Set data inputs
    label_col = args.label_column
    num_labels = args.num_labels
    n_add_train = args.n_add_train
    adapt_dodo = args.adapt_dodo
    dodos = ['fb_m', 'fb_w', 'mp_m', 'mp_w']

    # Set training args
    num_epochs = args.num_epochs
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
    train_data, val_data, test_data = load_n_rows_data(data_dir, dodos, adapt_dodo, n_add_train)

    # Load tokenizer from path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True)
    val_encodings = tokenizer(val_data['text'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True)

    # Prepare datasets
    train_dataset = Dataset(train_encodings, train_data[label_col].tolist())
    val_dataset = Dataset(val_encodings, val_data[label_col].tolist())
    test_dataset = Dataset(test_encodings, test_data[label_col].tolist())

    # Load PT model
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    # Set Logging
    config = {}
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    if report_to == 'wandb':
        run = wandb.init(project=f"{subbatch_name}", 
           name=subexperiment_name,
           tags=batch_name.split('-'),
           config=config)
    
    
    # Set training args
    training_args = TrainingArguments(
        output_dir=f'{working_dir}/checkpoints',
        num_train_epochs=num_epochs,
        auto_find_batch_size= True,
        logging_dir=f'{repo_dir}/experiments/{experiment_family}/{batch_name}/{subbatch_name}/logs',
        do_eval=True,
        evaluation_strategy=eval_strategy,
        save_strategy = eval_strategy,
        logging_strategy = eval_strategy,
        seed = seed,
        optim='adamw_torch',
        report_to=report_to,
        run_name=subexperiment_name,
        save_total_limit=1
    )

    # Set trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics)

    # Train 
    print(f"{'-'*25}\nTraining...")
    train_output = trainer.train()
    metrics = train_output.metrics
    runtime = metrics['train_runtime']
    print(f'Runtime: {runtime}')

    # Save the fine-tuned model locally
    print(f"{'-'*25}\nSaving model...")
    model_dir = f'{working_dir}/tmp_model_files'
    check_dir_exists(model_dir)
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    # Save to blob
    out_dir = f'fine_tuned_models/{experiment_family}/{batch_name}/{subbatch_name}/{subexperiment_name}'
    blob_model_saver(model_dir, out_dir)


    # Evaluate and save test set predictions
    print(f"{'-'*25}\nEvaluating...")
    test_logits = trainer.predict(test_dataset).predictions
    print(f"Length of test_logits: {len(test_logits)}")
    test_preds = np.argmax(test_logits, axis=1)
    test_probs = torch.softmax(torch.tensor(test_logits), dim=1).numpy()

    # Save results
    print(f"{'-'*25}\nSaving preds...")
    out_dir = f'{repo_dir}/experiments/{experiment_family}/results/{batch_name}/{subbatch_name}'
    check_dir_exists(out_dir)
    output_df = pd.DataFrame(test_probs, columns=[f'prob_class_{i}' for i in range(num_labels)])
    output_df['predicted_class'] = test_preds
    output_df['true_class'] = test_data[label_col].reset_index(drop=True)
    output_df['tweet_id'] = test_data['tweet_id'].reset_index(drop=True)  # Add tweet_id column
    output_df['dodo'] = test_data['dodo'].reset_index(drop=True)
    output_df['total_runtime'] = runtime
    # Save args
    for k,v in config.items():
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


