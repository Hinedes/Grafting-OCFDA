import importlib.util

import argparse
import os
import sys

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

import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
from importlib.metadata import version

import importlib.util

import sys
import os

from finetune import train, custom_find_spec
from evaluate import eval_model



def main():
    args = parse_args()

    model, tokenizer = train(base_model=args.base_model,
                             data_path=args.data_path,
                             eval_step=args.eval_step,
                             batch_size=args.batch_size,
                             micro_batch_size=args.micro_batch_size,
                             num_epochs=args.num_epochs,
                             learning_rate=args.learning_rate,
                             cutoff_len=args.cutoff_len,
                             val_set_size=args.val_set_size,
                             compile=args.compile,
                             seed=args.seed,
                             adapter_name=args.adapter_name,
                             random_indices=args.random_indices)

    average_score = 0.0
    for dataset in args.datasets:
        accuracy = eval_model(dataset_name=dataset, model=model, tokenizer=tokenizer)
        average_score += accuracy
        print(args.adapter_name + " accuracy on " + dataset + ":", accuracy)
    average_score /= len(args.datasets)

    print("Average accuracy on all datasets:", average_score)


def parse_args():
    parser = argparse.ArgumentParser()

    # args for finetuning
    parser.add_argument('--base_model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--adapter_name', choices=['lora', 'AdapterP', 'AdapterH', 'Parallel', 'no', 'orig', 'super', 'supra'], default='lora')
    parser.add_argument('--random_indices', default=False)
    parser.add_argument('--target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"])
    parser.add_argument('--data_path', default='ft-training_set/math_10k.json')
    parser.add_argument('--r', default=8)
    parser.add_argument('--eval_step', default=200)
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--micro_batch_size', default=16)
    parser.add_argument('--num_epochs', default=3)
    parser.add_argument('--learning_rate', default=2e-4)
    parser.add_argument('--cutoff_len', default=256)
    parser.add_argument('--val_set_size', default=120)
    parser.add_argument('--compile', default=0)
    parser.add_argument('--seed', default=0)

    # args for eval
    parser.add_argument('--datasets', default=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'])


    return parser.parse_args()


if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    for __i in range(torch.cuda.device_count()):
        print(f"GPU {__i}: {torch.cuda.get_device_name(__i)}")

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    main()