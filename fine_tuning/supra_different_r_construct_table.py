import importlib.util

import argparse
import os
import sys
from platform import python_version

original_find_spec = importlib.util.find_spec
def custom_find_spec(name, *args, **kwargs):
    if name == 'peft':
        return None
    return original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = custom_find_spec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEFT_PATH = os.path.abspath(os.path.join(os.getcwd(), "peft/src/"))

sys.path.insert(0, PEFT_PATH)
sys.path.insert(1, BASE_DIR)

import pickle

import pandas as pd
import torch
import argparse

import random
import numpy as np

from importlib.metadata import version

from finetune import train
from evaluate import eval_model


def get_lists():
    models = ['meta-llama/Llama-3.2-1B']
    lrs = [5e-5, 1e-4, 2e-4, 5e-4]
    lora_params_ratios = np.arange(0, 1.1, 0.1).tolist()
    datasets = ['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']

    return models, lrs, lora_params_ratios, datasets


def create_initial_eval_table():
    models, lrs, lora_params_ratios, datasets = get_lists()

    datasets.append('Average')

    # Create a MultiIndex for rows with sparsity, methods, and tasks
    index = pd.MultiIndex.from_product([lrs, models, lora_params_ratios], names=['lr', 'Model', 'lora rate'])

    # Create an empty DataFrame with sparsity, methods, and tasks as rows and models as columns
    df = pd.DataFrame(index=index, columns=datasets)

    return df


def create_initial_eval_avg_table():
    models, lrs, lora_params_ratios, _ = get_lists()

    # Create a MultiIndex for rows with methods and sparsity
    index = pd.MultiIndex.from_product([lrs], names=['lr'])

    # Create an empty DataFrame with methods and sparsity as rows and models as columns
    df = pd.DataFrame(index=index, columns=lora_params_ratios)

    return df


def print_latex_table(df):
    model_short_names = {
        'meta-llama/Llama-3.2-1B': 'Llama-3-1B',
        'meta-llama/Llama-3.2-3B': 'Llama-3-3B',
        'meta-llama/Meta-Llama-3-8B': 'Llama-3-8B',
    }

    # Replace model names if they exist in the DataFrame columns or index
    df.columns = [model_short_names.get(col, col) for col in df.columns]
    if 'Model' in df.index.names:
        df.index = df.index.set_levels([
            [model_short_names.get(level, level) if name == 'Model' else level for level in
             df.index.levels[idx]]
            for idx, name in enumerate(df.index.names)
        ])

    latex_table = df.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else x).to_latex(escape=False)
    print(latex_table)


def save_table(df, filename, dir="out"):
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_filepath = os.path.join(dir, f"{filename}.pkl")
    with open(save_filepath, 'wb') as f:
        pickle.dump(df, f)


def load_table(filename, dir="out"):
    load_filepath = os.path.join(dir, f"{filename}.pkl")

    if os.path.exists(load_filepath):
        with open(load_filepath, 'rb') as f:
            df = pickle.load(f)
        return df
    return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def construct_table(seed):
    print("CUDA Available:", torch.cuda.is_available())
    for __i in range(torch.cuda.device_count()):
        print(f"GPU {__i}: {torch.cuda.get_device_name(__i)}")

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    models, lrs, lora_params_ratios, datasets = get_lists()

    target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    data_path = 'ft-training_set/math_10k.json'

    # TODO: replace it with the call of compute_sparse_rate(lora_adapter_model) function from finetune.py
    sparse_rates = {
        'meta-llama/Llama-3.2-1B': 0.005962171052631579,
        'meta-llama/Llama-3.2-3B': 0.004464285714285714,
        'meta-llama/Llama-3.1-8B': 0.0031020220588235292,
        'meta-llama/Meta-Llama-3-8B': 0.0031020220588235292,
    }

    eval_table = load_table("eval_table")
    eval_avg_table = load_table("eval_avg_table")
    if eval_table is None:
        eval_table = create_initial_eval_table()
    if eval_avg_table is None:
        eval_avg_table = create_initial_eval_avg_table()

    for model_name in models:
        sparse_rate = sparse_rates[model_name]

        for lr in lrs:
            for lora_params_ratio in lora_params_ratios:
                set_seed(seed)

                print("Model: " + model_name + " lr = " + str(lr) + " adapter: supra-wanda lora_params_ratio = " + str(lora_params_ratio))
                #if pd.notna(eval_avg_table.loc[lr, r_lora]):
                #    print("Already computed...")
                #    continue

                model, tokenizer = train(base_model=model_name, data_path=data_path, target_modules=target_modules,
                                         eval_step=50, batch_size=16, micro_batch_size=16,
                                         num_epochs=3, learning_rate=lr, cutoff_len=256,
                                         val_set_size=120, compile=0, seed=seed,
                                         lora_params_ratio=lora_params_ratio, adapter_name='supra',
                                         random_indices=False, sparse_rate=sparse_rate)

                name_to_acc = {task: 0 for task in datasets}

                for dataset in datasets:
                    accuracy = eval_model(dataset_name=dataset, model=model, tokenizer=tokenizer) * 100
                    print(dataset + " accuracy: " + str(accuracy) + "%")

                    name_to_acc[dataset] = accuracy
                    eval_table.loc[(lr, model_name, lora_params_ratio), dataset] = accuracy

                average_score = sum(name_to_acc.values()) / len(name_to_acc)
                eval_table.loc[(lr, model_name, lora_params_ratio), 'Average'] = average_score
                eval_avg_table.loc[lr, lora_params_ratio] = average_score

                save_table(eval_table, filename="eval_table")
                save_table(eval_avg_table, filename="eval_avg_table")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    #eval_table = load_table("eval_table")
    #eval_avg_table = load_table("eval_avg_table")
    #print_latex_table(eval_table)

    construct_table(args.seed)
