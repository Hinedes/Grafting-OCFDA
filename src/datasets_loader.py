# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
import os
import json
import hashlib

from datasets import load_dataset


LOADER_CACHE_VERSION = "v2"


# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=True)
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    trainloader = []

    seqlen = min(2048, seqlen)

    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
                             split='train', trust_remote_code=True)
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation', trust_remote_code=True)

    # Generate samples from training set
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    trainloader = []

    seqlen = min(2048, seqlen)

    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100

        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def render_instruction_record(record):
    instruction = str(record.get("instruction", ""))
    input_text = str(record.get("input", ""))
    output = str(record.get("output", ""))

    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def get_json_instruction_calibration(path, nsamples, seed, seqlen, tokenizer):
    with open(path, "r") as f:
        records = json.load(f)

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    seqlen = min(2048, seqlen)
    texts = [render_instruction_record(record) for record in records]
    text = "\n\n".join(texts)
    trainenc = tokenizer(text, return_tensors="pt")

    while trainenc.input_ids.shape[1] <= seqlen + 1:
        text = text + "\n\n" + text
        trainenc = tokenizer(text, return_tensors="pt")

    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, TokenizerWrapper(trainenc.input_ids)


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):

    cache_dir = "loaders_cache"
    seqlen = min(2048, seqlen)
    if os.path.exists(name):
        stat = os.stat(name)
        cache_source = f"{os.path.abspath(name)}:{stat.st_size}:{int(stat.st_mtime)}"
        cache_key = hashlib.md5(cache_source.encode()).hexdigest()[:10]
    else:
        cache_key = name
    tokenizer_name = tokenizer.name_or_path.split('/')[-1] if tokenizer is not None else "unknown-tokenizer"
    name_cache_dir = (
        f"{LOADER_CACHE_VERSION}_{cache_key}"
        f"_nsamples_{nsamples}_seed_{seed}_seqlen_{seqlen}_{tokenizer_name}"
    )

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    full_cache_dir = os.path.join(cache_dir, name_cache_dir)
    full_cache_dir_train = os.path.join(full_cache_dir, 'train.pt')
    full_cache_dir_test = os.path.join(full_cache_dir, 'test.pt')

    train_loader, test_loader = None, None

    if os.path.exists(full_cache_dir):
        print(f"Loading calibration cache: {full_cache_dir}")
        train_loader = torch.load(full_cache_dir_train, weights_only=True)
        test_loader = torch.load(full_cache_dir_test, weights_only=False)
    else:
        print(f"Building calibration cache: {full_cache_dir}")
        match name:
            case 'wikitext2':
                train_loader, test_loader = get_wikitext2(nsamples, seed, seqlen, tokenizer)
            case 'c4':
                train_loader, test_loader = get_c4(nsamples, seed, seqlen, tokenizer)
            case _ if os.path.exists(name) and name.endswith(".json"):
                train_loader, test_loader = get_json_instruction_calibration(name, nsamples, seed, seqlen, tokenizer)
            case _:
                raise ValueError(f"Unsupported calibration dataset for get_loaders: {name}")

        os.makedirs(full_cache_dir, exist_ok=True)

        torch.save(train_loader, full_cache_dir_train)
        torch.save(test_loader, full_cache_dir_test)

    return train_loader, test_loader
