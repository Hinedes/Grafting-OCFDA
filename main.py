import argparse
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from src.mask import prepare_super_mask
from src.finetune import fine_tune_model
from utils.eval import eval_ppl, eval_zero_shot

# In case you want to select particular GPUs
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

print("CUDA Available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    # Llama 1 family models:
    # baffo32/decapoda-research-llama-7b-hf
    # baffo32/decapoda-research-llama-13b-hf
    # baffo32/decapoda-research-llama-30b-hf
    # baffo32/decapoda-research-llama-65b-hf

    # Llama 2 family models:
    # meta-llama/Llama-2-7b-hf
    # meta-llama/Llama-2-13b-hf
    # meta-llama/Llama-2-70b-hf

    # Tiny Llama
    # TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Llama 3 family models:
    # meta-llama/Llama-3.2-1B
    # meta-llama/Llama-3.2-3B
    # meta-llama/Llama-3.2-11B-Vision
    # meta-llama/Meta-Llama-3-8B
    # meta-llama/Meta-Llama-3-70B

    # Mistral
    # mistralai/Mistral-7B-v0.1
    # mistralai/Mixtral-8x7B-Instruct-v0.1

    # OPT family models:
    # facebook/opt-125m
    # facebook/opt-350m
    # facebook/opt-1.3b
    # facebook/opt-2.7b
    # facebook/opt-6.7b
    # facebook/opt-13b
    # facebook/opt-30b
    # facebook/opt-66b
    # facebook/opt-175b

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model', default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--save_model', type=str, help='Path to save the pruned model.')

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')

    parser.add_argument('--outliers_ratio', type=float, default=0.01, help='The percentage of outliers to fine tune')

    parser.add_argument("--eval_ppl", action="store_true", default=False)
    parser.add_argument("--eval_zero_shot", action="store_true", default=False)


    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print(f"loading llm model {args.model}")

    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.outliers_ratio != 0:
        print('Starting preparing super masks...')

        tick = time.perf_counter()
        prepare_super_mask(args, model, tokenizer, device, outliers_ratio=args.outliers_ratio)

        print('Masks preparation time %.2f' % (time.perf_counter() - tick))

        print('Starting fine-tuning a model...')

        tick = time.perf_counter()
        fine_tune_model(model=model, tokenizer=tokenizer, device=device, dataset='c4', epocs=1)

        print('Fine-tuning time %.2f' % (time.perf_counter() - tick))

    if args.eval_ppl:
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"WikiText-2 perplexity {ppl_test}")

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

    if args.eval_zero_shot:
        # Evaluate using lm-evaluation-harness
        task_list = ['winogrande', 'openbookqa', 'boolq', 'piqa', 'hellaswag', 'arc_easy', 'arc_challenge']
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("zero_shot evaluation results")

        name_to_acc = {task: data['acc,none'] * 100 for task, data in results['results'].items()}
        average_score = sum(name_to_acc.values()) / len(name_to_acc)

        print(name_to_acc)
        print("Average:", average_score)


if __name__ == '__main__':
    main()
