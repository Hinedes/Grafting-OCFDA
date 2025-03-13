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
    r_loras = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    datasets = ['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']

    return models, lrs, r_loras, datasets


def create_initial_eval_table():
    models, lrs, r_loras, datasets = get_lists()

    datasets.append('Average')

    # Create a MultiIndex for rows with sparsity, methods, and tasks
    index = pd.MultiIndex.from_product([lrs, models, r_loras], names=['lr', 'Model', 'lora r'])

    # Create an empty DataFrame with sparsity, methods, and tasks as rows and models as columns
    df = pd.DataFrame(index=index, columns=datasets)

    return df


def create_initial_eval_avg_table():
    models, lrs, r_loras, _ = get_lists()

    # Create a MultiIndex for rows with methods and sparsity
    index = pd.MultiIndex.from_product([lrs], names=['lr'])

    # Create an empty DataFrame with methods and sparsity as rows and models as columns
    df = pd.DataFrame(index=index, columns=r_loras)

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

    models, lrs, r_loras, datasets = get_lists()

    target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
    data_path = 'ft-training_set/math_10k.json'

    eval_table = load_table("eval_table")
    eval_avg_table = load_table("eval_avg_table")
    if eval_table is None:
        eval_table = create_initial_eval_table()
    if eval_avg_table is None:
        eval_avg_table = create_initial_eval_avg_table()

    for model_name in models:
        for lr in lrs:
            for r_lora in r_loras:
                set_seed(seed)

                print("Model: " + model_name + " lr = " + str(lr) + " adapter: supra-wanda r_lora = " + str(r_lora))
                #if pd.notna(eval_avg_table.loc[lr, r_lora]):
                #    print("Already computed...")
                #    continue

                model, tokenizer = train(base_model=model_name, data_path=data_path, target_modules=target_modules,
                                         eval_step=1000, batch_size=16, micro_batch_size=16,
                                         num_epochs=3, learning_rate=lr, cutoff_len=256,
                                         val_set_size=120, compile=0, seed=seed,
                                         supra_lora_r=r_lora, supra_super_r=8-r_lora,
                                         adapter_name='supra', random_indices=False)

                name_to_acc = {task: 0 for task in datasets}

                for dataset in datasets:
                    accuracy = eval_model(dataset_name=dataset, model=model, tokenizer=tokenizer) * 100
                    print(dataset + " accuracy: " + str(accuracy) + "%")

                    name_to_acc[dataset] = accuracy
                    eval_table.loc[(lr, model_name, r_lora), dataset] = accuracy

                average_score = sum(name_to_acc.values()) / len(name_to_acc)
                eval_table.loc[(lr, model_name, r_lora), 'Average'] = average_score
                eval_avg_table.loc[lr, r_lora] = average_score

                save_table(eval_table, filename="eval_table", dir="out/" + str(lr))
                save_table(eval_avg_table, filename="eval_avg_table", dir="out/" + str(lr))


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
