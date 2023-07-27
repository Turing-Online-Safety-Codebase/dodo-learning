import argparse
import os
import sys
import pandas as pd
import ast
import json
import subprocess
from dodo.utils.helper_functions import return_relative_dirs, check_dir_exists

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_family', type = str, default = 'baselines', help='Type of experiment batch to run')
    parser.add_argument('--batch_name', type=str, required=True, help='Batch name to run')
    parser.add_argument('--num_epochs', type=int, default = 5, help='Number of epochs')
    parser.add_argument('--es_patience', type=int, default = 2, help='Early stopping patience')
    parser.add_argument('--best_model_metric', type=str, default = 'eval_macro_f1', help='Metric for best model')
    parser.add_argument('--eval_strategy', type = str, default = 'epoch', help = 'Eval and save strategy')
    parser.add_argument('--report_to', type = str, default = 'none', help = 'Monitoring reporting')
    parser.add_argument('--label_col', type = str, default = 'stance', help = 'Name of label col to use (stance or abuse)')
    parser.add_argument('--overwrite_results', required=False, default=False, action='store_true', help='Overwrite existing experiment results if present, instead of skipping them')


    args = parser.parse_args()

    print(f"{'-'*25}\nArguments:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    print(f"{'-'*25}\n")

    # Set directory
    repo_dir, working_dir = return_relative_dirs('dodo_learning')
    print(f'Main directiory: {repo_dir}')
    data_dir = f'{repo_dir}/data/labelled_data/splits'
    print(f'Data directiory: {data_dir}')


    # Set inputs
    batch_name = args.batch_name
    experiment_family = args.experiment_family
    dodos = ['fb_m', 'fb_w', 'mp_m', 'mp_w']
    label_column = args.label_col
    num_labels = 4 if label_column=='stance_label' else 2
    num_epochs = args.num_epochs
    es_patience = args.es_patience
    best_model_metric = args.best_model_metric
    eval_strategy = args.eval_strategy
    report_to = args.report_to


    # Load experiment data
    csv_file_path = f'{repo_dir}/experiments/{experiment_family}/batches/{batch_name}.csv'
    # For testing take one row
    df = pd.read_csv(csv_file_path, nrows = 15, converters={'dodo_order':ast.literal_eval})

    # Define outdir
    outdir = f'{repo_dir}/experiments/{experiment_family}/results/{batch_name}'
    check_dir_exists(outdir)

    # Print string
    print(f'\nRunning {len(df)} experiments for batch {batch_name}')

    for index, row in df.iterrows():
        subexperiment_name = row['subexperiment_string']
        # Check if experiment has already run
        output_file = f"{subexperiment_name}.csv"
        
        if os.path.exists(f'{outdir}/{output_file}'):
            print(f"Output file for {subexperiment_name} already exists ...")
            if args.overwrite_results: 
                print("Overwriting previous results.")
            else: 
                print("Skipping experiment.")
                continue

        else:
            print(f'Running for {subexperiment_name}')

        # Set up sampling ratio:
        dodo_order = row['dodo_order']
        sampling_ratios = {}
        for i in dodo_order:
            sampling_ratios[i] = row[f'{i}_ratio']
        for i in dodos:
            if i not in sampling_ratios.keys():
                sampling_ratios[i] = 0
        
        
        cmd = [
            sys.executable, f"{repo_dir}/src/dodo/model_training/train_transformer.py",
            '--subexperiment_name', subexperiment_name,
            '--batch_name', batch_name,
            '--experiment_family', experiment_family,
            '--model', row['model'],
            '--seed', str(row['seed']),
            '--ratios', json.dumps(sampling_ratios),
            '--data_order', row['data_order'],
            '--label_column', label_column,
            '--num_labels', str(num_labels),
            '--num_epochs', str(num_epochs),
            '--es_patience', str(es_patience),
            '--best_model_metric', best_model_metric,
            '--eval_strategy', eval_strategy,
            '--report_to', report_to
        ]

        subprocess.run(cmd)

if __name__ == '__main__':
    main()