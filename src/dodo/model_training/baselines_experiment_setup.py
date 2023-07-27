import argparse
import os
import pandas as pd
import numpy as np
import itertools
from dodo.utils.helper_functions import check_dir_exists, return_relative_dirs


def generate_dodo_combinations(items, fixed_datasize, data_method, order_matters, order_method):
    # Create list of items and all possible combinations
    combinations = []
    orders = []
    experiment_types = []
    for i in range(1, 5):
        if order_matters == True:
            combo_set = itertools.permutations(items, i)
        else:
            combo_set = itertools.combinations(items, i)
        for subset in combo_set:
            if fixed_datasize == True:
                combination = [1/i if item in subset else 0 for item in items]
            else:
                combination = [1 if item in subset else 0 for item in items]
            combinations.append(combination)
            experiment_types.append(f'dodo{i}')
            orders.append(subset)
    
    # Create dataframe
    num_combos = len(combinations)
    experiment_ids = ['exp{}'.format(i) for i in range(num_combos)]
    items = [f'{item}_ratio' for item in items]
    df = pd.DataFrame(combinations, columns=items, index = experiment_ids)
    df['dodo_order'] = orders
    # All datasets have 3000 entries fixed
    df['single_dodo_train_size'] = 3000
    df['total_ratio'] = df[items].sum(axis=1)
    df['total_train_size'] = df['single_dodo_train_size'] * df['total_ratio']
    df['experiment_type'] = experiment_types

    # Add indicator column for data sampling meta info
    df['data_size'] = data_method
    df['data_order'] = order_method
    return df, num_combos


def create_batch_string(row):
    batch_string = f"{row['data_size']}-{row['data_order']}-{row['model_name']}-s{row['seed']}"
    return batch_string

def create_subexperiment_string(row):
    order = row['dodo_order']
    subexp_string = '-'.join([o for o in order])
    subexp_string = f"{row['experiment_type']}-{subexp_string}"
    return subexp_string


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_family', type = str, default = 'baselines', required = True, help='Type of experiment batch to generate')
    parser.add_argument('--fixed_datasize', action='store_true', help='If fix data size across experiments')
    parser.add_argument('--order_matters', action='store_true', help='If shuffle dodo order within experiment')
    args = parser.parse_args()

    print(f"{'-'*25}\nArguments:")
    for arg in vars(args):
        print(f"{arg} is {getattr(args, arg)}")
    print(f"{'-'*25}\n")


    # Set directories
    repo_dir, working_dir = return_relative_dirs('dodo_learning')
    data_dir = f'{repo_dir}/data/labelled_data'
    print(f"{'-'*25}\nDirectories:")
    print(f'repo_dir: {repo_dir}')
    print(f'working dir: {working_dir}')
    print(f'data_dir: {data_dir}')
    print(f"{'-'*25}\n")

    # Set args
    order_matters = args.order_matters
    if order_matters is True:
        order_method = 'ordered'
    else:
        order_method = 'shuffled'
    fixed_datasize = args.fixed_datasize
    if fixed_datasize is True:
        data_method = 'fixedsize'
    else:
        data_method = 'rawsize'
    experiment_family = args.experiment_family

    # Define items, seeds and models
    dodos = ['fb_m', 'fb_w', 'mp_m', 'mp_w']
    seeds = [1,2,3]
    model_dict = {'distilbert-base-cased':'diBERT', 'microsoft/deberta-v3-base': 'deBERT'}

    # Generate dataset combinations
    experiment_df, num_combos = generate_dodo_combinations(dodos, fixed_datasize, data_method, order_matters, order_method)
    experiment_df = experiment_df.reset_index().rename(columns = {'index':'subexperiment_id'})


    # Expand combinations with seed and model choices
    experiment_df['seed'] = np.tile(seeds, (len(experiment_df),1)).tolist()
    experiment_df['model'] = np.tile(list(model_dict.keys()), (len(experiment_df),1)).tolist()

    # Explode
    subexperiment_df = experiment_df.explode(['model']).explode(['seed']).sort_values(by = ['model', 'seed'])

    # Define model name
    subexperiment_df['model_name'] = subexperiment_df['model'].map(lambda x: model_dict[x])

    # Reset index
    subexperiment_df.index = pd.RangeIndex(start=0, step=1, stop=len(subexperiment_df))

    # Define identifiying strings for storing results
    subexperiment_df['batch_string'] = subexperiment_df.apply(lambda x: create_batch_string(x), axis = 1)
    subexperiment_df['subexperiment_string'] = subexperiment_df.apply(lambda x: create_subexperiment_string(x), axis = 1)
    # Add experiment_type
    subexperiment_df['experiment_family'] = experiment_family

    # Set order
    col_order = ['subexperiment_string', 'subexperiment_id', 
                 'batch_string', 'experiment_family',
                 'fb_m_ratio', 'fb_w_ratio','mp_m_ratio', 'mp_w_ratio', 
                 'dodo_order', 'single_dodo_train_size',
                 'total_ratio', 'total_train_size', 'experiment_type', 'data_size',
                 'data_order', 'seed', 'model', 'model_name',]


    # Save experiment batches
    save_dir = f'{repo_dir}/experiments/{experiment_family}/batches'
    check_dir_exists(save_dir)
    for batch in subexperiment_df['batch_string'].unique():
        run_df = subexperiment_df[subexperiment_df['batch_string']==batch]
        run_df = run_df[col_order]
        # exclude dodo1 with rawsize as already run
        if data_method == 'rawsize':
            run_df = run_df[run_df['experiment_type']!='dodo1']
        if experiment_family == 'baselines':
            run_df.to_csv(f'{save_dir}/{batch}.csv', index = False)

        # Option to save pilot experiments only for specialist and generalist models
        if experiment_family == 'pilots':
            pilot_df = run_df[(run_df['experiment_type']=='dodo1') | (run_df['experiment_type']=='dodo4')]#
            pilot_df.to_csv(f'{save_dir}/{batch}.csv', index = False)

if __name__ == '__main__':
    main()




