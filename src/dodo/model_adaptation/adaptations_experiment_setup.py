import argparse
import os
import pandas as pd
import numpy as np
import itertools
from dodo.model_training.baselines_experiment_setup import create_batch_string
from dodo.utils.helper_functions import check_dir_exists, return_relative_dirs



# Create subbatch string
def create_subexperiment_string(row):
    subbatch_string = row['subbatch_string']
    n_add_train = row['n_add_train']
    subexperiment_str = f'{subbatch_string}-{n_add_train}'
    return subexperiment_str

# Create subbatch string
def create_subbatch_string(row):
    start_dodo = row['start_on']
    adapt_dodo = row['adapt_to']
    subbatch_str = f'{start_dodo}-to-{adapt_dodo}'
    return subbatch_str

# Get fine-tuned model path from blob storage
def get_ft_model_path(row):
    batch_string = row['batch_string']
    start_dodo = row['start_on']
    path = f'fine_tuned_models/baselines/{batch_string}/dodo1-{start_dodo}'
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_family', type = str, default = 'adaptations', required = True, help='Type of experiment batch to generate')
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
    train_increments = [50, 125, 250, 500, 1000, 1500, 2000, 3000]
    model_dict = {'microsoft/deberta-v3-base': 'deBERT'}

    # Generate dataset combinations
    combinations = itertools.permutations(dodos, 2)
    start_on = []
    adapt_to = []
    for i in combinations:
        start_on.append(i[0])
        adapt_to.append(i[1])
    

    # Get main experiments
    experiment_df = pd.DataFrame({'start_on':start_on, 'adapt_to': adapt_to})
    experiment_df = experiment_df.reset_index().rename(columns = {'index':'pair_id'})

    # Expand combinations with seed and model choices
    experiment_df['seed'] = np.tile(seeds, (len(experiment_df),1)).tolist()
    experiment_df['model'] = np.tile(list(model_dict.keys()), (len(experiment_df),1)).tolist()
    experiment_df['n_add_train'] = np.tile(train_increments, (len(experiment_df),1)).tolist()

    # Explode
    subexperiment_df = experiment_df.explode(['model']).explode(['seed']).explode(['n_add_train']).sort_values(by = ['start_on', 'adapt_to', 'model', 'seed', 'n_add_train'])

    # Update experiment details
    subexperiment_df['model_name'] = subexperiment_df['model'].map(lambda x: model_dict[x])
    subexperiment_df['data_order'] = order_method
    subexperiment_df['data_size'] = data_method

    # Define identifiying strings for storing results
    subexperiment_df['batch_string'] = subexperiment_df.apply(lambda x: create_batch_string(x), axis = 1)
    subexperiment_df['subbatch_string'] = subexperiment_df.apply(lambda x: create_subbatch_string(x), axis = 1)
    subexperiment_df['subexperiment_string'] = subexperiment_df.apply(lambda x: create_subexperiment_string(x), axis = 1)
    # Add experiment_type
    subexperiment_df['experiment_family'] = experiment_family

    # Get FT model path for linking to blob storage
    subexperiment_df['ft_model'] = subexperiment_df.apply(lambda x: get_ft_model_path(x), axis = 1)

    # Set order
    col_order = ['subexperiment_string', 'pair_id', 
                    'batch_string',  'subbatch_string','experiment_family',
                    'start_on', 'adapt_to', 'n_add_train',
                    'data_size','data_order', 
                    'seed', 'model', 'model_name', 'ft_model']

    # Save experiment batches
    for batch in subexperiment_df['batch_string'].unique():
        batch_df = subexperiment_df[subexperiment_df['batch_string']==batch]
        save_dir = f'{repo_dir}/experiments/{experiment_family}/batches/{batch}'
        check_dir_exists(save_dir)
        batch_df = batch_df[col_order]
        for subbatch in batch_df['subbatch_string'].unique():
            subbatch_df = batch_df[batch_df['subbatch_string']==subbatch]
            subbatch_df.to_csv(f'{save_dir}/{subbatch}.csv', index = False)

if __name__ == '__main__':
    main()




