import argparse
import os
import sys
import pandas as pd
import ast
import json
import subprocess
from dodo.utils.helper_functions import return_relative_dirs, check_dir_exists, blob_model_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_family', type = str, default = 'adaptations', help='Type of experiment batch to run')
    parser.add_argument('--batch_name', type=str, required=True, help='Batch name to run')
    parser.add_argument('--subbatch_name', type=str, required=True, help='Subbatch name to run')
    parser.add_argument('--num_epochs', type=int, default = 3, help='Number of epochs')
    parser.add_argument('--eval_strategy', type = str, default = 'epoch', help = 'Eval and save strategy')
    parser.add_argument('--report_to', type = str, default = 'none', help = 'Monitoring reporting')
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
    subbatch_name = args.subbatch_name
    experiment_family = args.experiment_family
    label_column = 'stance_label'
    num_labels = 4
    num_epochs = args.num_epochs
    eval_strategy = args.eval_strategy
    report_to = args.report_to


    # Load experiment data
    csv_file_path = f'{repo_dir}/experiments/{experiment_family}/batches/{batch_name}/{subbatch_name}.csv'

    # For testing take one row
    df = pd.read_csv(csv_file_path)

    # Download ft model
    print(f'\nLoading model from blob storage...')
    model_in_dir = df['ft_model'].iloc[0]
    model_name = df['model_name'].iloc[0]
    start_dodo = df['start_on'].iloc[0]
    seed = df['seed'].iloc[0]

    model_out_dir = f'{working_dir}/input_tmp_model_files/{start_dodo}-{model_name}-s{seed}'
    if os.path.exists(model_out_dir) is False:
        print('Model not loaded. Loading model')
        check_dir_exists(model_out_dir)
        blob_model_loader(model_in_dir, model_out_dir)
    else:
        print('Model already loaded.')

    # Define outdir
    results_out_dir = f'{repo_dir}/experiments/{experiment_family}/results/{batch_name}/{subbatch_name}/'
    check_dir_exists(results_out_dir)

    # Print string
    print(f'\nRunning {len(df)} experiments for batch {batch_name}, subbatch {subbatch_name}')

    for index, row in df.iterrows():
        subexperiment_name = row['subexperiment_string']
        # Check if experiment has already run
        output_file = f"{subexperiment_name}.csv"

        if os.path.exists(f'{results_out_dir}/{output_file}'):
            print(f"Output file for {subexperiment_name} already exists ...")
            if args.overwrite_results: 
                print("Overwriting previous results.")
            else: 
                print("Skipping experiment.")
                continue

        else:
            print(f'Running for {subexperiment_name}')
    
        cmd = [
            sys.executable, f"{repo_dir}/src/dodo/model_adaptation/adapt_transformer.py",
            '--subexperiment_name', subexperiment_name,
            '--batch_name', batch_name,
            '--subbatch_name', subbatch_name,
            '--experiment_family', experiment_family,
            '--model_path', model_out_dir,
            '--seed', str(row['seed']),
            '--n_add_train', str(row['n_add_train']),
            '--adapt_dodo', row['adapt_to'],
            '--label_column', label_column,
            '--num_labels', str(num_labels),
            '--num_epochs', str(num_epochs),
            '--eval_strategy', eval_strategy,
            '--report_to', report_to
        ]

        subprocess.run(cmd)

if __name__ == '__main__':
    main()