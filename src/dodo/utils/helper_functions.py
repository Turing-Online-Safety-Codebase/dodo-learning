#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Helper functions used across scripts.
"""

import os
import random
import glob
from io import StringIO
from datetime import date
import pandas as pd
from pathlib import Path
import numpy as np
import azure.storage.blob as azureblob
import yaml

SEED = 123
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def blob_setup():
    repo_dir, working_dir = return_relative_dirs('dodo_learning')
    # Set up blob storage
    BLOB_CONFIG_FILE = f'{repo_dir}/config.yaml'
    with open(BLOB_CONFIG_FILE, 'r') as config_file:
        config = yaml.safe_load(config_file)

    BLOB_CONNECT_STR = config["blob"]["blob_connect_str"]
    BLOB_CONTAINER = config["blob"]["blob_container"]
    blob_account = azureblob.BlobServiceClient.from_connection_string(BLOB_CONNECT_STR)
    blob_container = blob_account.get_container_client(BLOB_CONTAINER)
    return blob_account, blob_container

def data_loader(data_path, columns = None):
    """Loads data from csv, keeps selected columns.
    Args:
        data_path (str): filepath to csv.
        columns (list[str], optional): columns to keep. Defaults to ['id', 'text_replaced'].
    Returns:
        df (pandas.DataFrame)
    """
    print('--Loading data--')
    loaded_df = pd.read_csv(data_path)
    if columns is None:
        keep_columns = ['pool_id', 'orig_id', 'text']
        loaded_df = loaded_df[keep_columns]
    elif columns == 'all':
        loaded_df = loaded_df
    else:
        keep_columns = columns
        loaded_df = loaded_df[keep_columns]
    print(f'df size: {len(loaded_df)}')
    return loaded_df

def random_sample(pool, n_items, seed_value = SEED):
    """Takes a random sample of n_items.
    Args:
        pool (pandas.DataFrame): unlabelled pool of data.
        N (int): number of items to sample randomly from pool.
        SEED (int, optional): random state for sampling. Defaults to global SEED.
    Returns:
        sample (pandas.DataFrame): randomly-selected sample of n
    """
    print('--Random sampling--')
    sample = pool.sample(n_items, random_state = seed_value)
    assert len(sample) == n_items
    return sample

def update_pool(pool, sample):
    """Removes sample from pool by index.
    Args:
        pool (pandas.DataFrame): unlabelled pool of data.
        sample (pandas.DataFrame): sampled items of data.
    Returns:
        updated_pool (pandas.DataFrame): pandas dataframe of remaining items.
    """
    print('--Updating pool--')
    selected_indices = sample.index
    updated_pool = pool.loc[~pool.index.isin(selected_indices)]
    print(f'orig pool size: {len(pool)}')
    print(f'sample size: {len(sample)}')
    print(f'updated pool size: {len(updated_pool)}')
    assert len(updated_pool) == len(pool) - len(sample)
    return updated_pool


def check_dir_exists(path):
    """Checks if folder directory already exists, else makes directory.
    Args:
        path (str): folder path for saving.
    """
    is_exist = os.path.exists(path)
    if not is_exist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Creating {path} folder")
    else:
        print(f"Folder exists: {path}")


def data_saver(save_df, out_dir, file_name, columns = None):
    """Saves text, id columns of pandas.DataFrame to csv at out_dir. file_name,
    Args:
        save_df (pandas.DataFrame): dataframe to save.
        out_dir (str): folderpath to save location.
        file_name (str): name to save file under.
        columns (list[str], optional): columns to keep. Defaults to ['id', 'text_replaced'].
    Returns:
        None
    """
    print(f'--Saving data to {out_dir}/{file_name}--')
    blob_account, blob_container = blob_setup()
    if columns is None:
        keep_columns = ['pool_id', 'orig_id', 'text']#,'perspective_label'] #, 'batch']
    else:
        keep_columns = columns
    if keep_columns != 'all':
        save_df = save_df[keep_columns]
    check_dir_exists(out_dir)
    # save to csv
    save_df.to_csv(f'{out_dir}/{file_name}', encoding = 'utf-8', index = False)
    # save to blob
    blob_save_dir = out_dir.split('/data/')[-1]
    print(f'Saving blob data to {blob_save_dir}')
    blob_container.upload_blob(name=f'{blob_save_dir}/{file_name}',data=save_df.to_csv(index=False),encoding='utf-8',overwrite=True, )


def replace_special_tokens(input_df, input_text_col = 'text'):
    """Replaces special tokens seen by annotators to special tokens for transformer model.
    Args:
        input_df (pandas.DataFrame): pandas dataframe with text to be replaced.
        input_text_col (str, optional): column with text for replacing special tokens. Defaults to 'text'.
    Returns:
        pandas.DataFrame: dataframe with new column added with replacements.
    """
    print('\n--Replacing special tokens--\n')
    replace_map = {'@PLAYER': '[PLAYER]',
                    '@BODY': '[BODY]',
                    '@CLUB': '[CLUB]',
                    '@USER': '[USER]',
                    'URL': '[URL]'}

    input_df[input_text_col] = input_df[input_text_col].replace(replace_map, regex = True)
    return input_df

def return_relative_dirs(repo_name):
    # Get the absolute path of the current working directory
    current_working_directory = Path.cwd()

    # Find the home repo folder in the path
    home_folder = None
    for folder in current_working_directory.parents:
        if folder.name == repo_name:
            home_folder = folder
            return home_folder, current_working_directory

    if home_folder is None:
        raise ValueError(f"Cannot find the {repo_name} folder in the current working directory's path.")


def blob_data_saver(save_df, out_dir, file_name):
    blob_account, blob_container = blob_setup()
    # save to blob
    blob_container.upload_blob(name=f'{out_dir}/{file_name}',data=save_df.to_csv(index=False),encoding='utf-8',overwrite=True, )


def blob_model_saver(in_dir, out_dir):
    blob_account, blob_container = blob_setup()
    # load model from file
    files = os.listdir(in_dir)
    # save to blob
    for file in files:
        with open(f'{in_dir}/{file}','rb') as data:
            blob_container.upload_blob(
            name=f"{out_dir}/{file}",
            data=data,
            overwrite=True,
            )

def blob_model_loader(in_dir, out_dir):
    blob_account, blob_container = blob_setup()
    blobs = blob_container.list_blobs(name_starts_with=in_dir)
    # Download each blob in the folder
    for blob in blobs:
        # Get blob client for the specific blob
        blob_client = blob_container.get_blob_client(blob=blob.name)
        name = 'blah'
        # Download blob
        with open(f"{out_dir}/{blob.name.split('/')[-1]}", "wb") as my_blob:
                download_stream = blob_client.download_blob()
                my_blob.write(download_stream.readall())