import copy
import json
import os
import re
import sys
import argparse

import fire

import torch

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def eval_model(dataset_name, model_name, adapter, base_model, lora_weights, load_8bit, debug, target_modules, r) -> float:
    def evaluate(
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=256,
            **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
    save_file = f'experiment/{model_name}-{adapter}-{dataset_name}.json'
    create_dir('experiment/')

    dataset = load_data(dataset_name)
    tokenizer, model = load_model(base_model, model_name, lora_weights, load_8bit, adapter, target_modules, r)

    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total)

    accuracy = 0.0

    for idx, data in enumerate(dataset):
        instruction = data.get('instruction')

        outputs = evaluate(instruction)
        label = data.get('answer')
        flag = False
        if dataset_name.lower() in ['aqua']:
            predict = extract_answer_letter(outputs)
            if label == predict:
                correct += 1
                flag = True
        else:
            if isinstance(label, str):
                label = float(label)
            predict = extract_answer_number(dataset_name, outputs)
            if abs(label - predict) <= miss:
                correct += 1
                flag = True
        new_data = copy.deepcopy(data)
        new_data['output_pred'] = outputs
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)
        print(' ')
        print('---------------')
        print(outputs)
        print('prediction:', predict)
        print('label:', label)
        print('---------------')

        accuracy = correct / (idx + 1)

        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {accuracy}')
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)
        pbar.update(1)
    pbar.close()
    print('\n')
    print('test finished')

    return accuracy


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(dataset) -> list:
    """
    read data from dataset file
    Args:
        dataset:

    Returns: list

    """
    file_path = f'dataset/{dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True)
    parser.add_argument('--model_name', choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B', 'LLaMA-3-8B', 'LLaMA-3.1-8B', 'LLaMA-3.2-1B'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'no', 'orig', 'super', 'supra'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--load_8bit', action='store_true', default=False)

    parser.add_argument('--debug', action='store_true', default=False)

    # TODO rewrite it so that one don't have to pass these args
    parser.add_argument('--target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"])

    parser.add_argument('--r', default=8)


    return parser.parse_args()


def load_model(base_model, model_name, lora_weights, load_8bit, adapter, target_modules, r) -> tuple:
    """
    load tuned model
    Args:
        base_model, model_name, lora_weights, load_8bit, adapter, target_modules, sparse_rate

    Returns:
        tuple(tokenizer, model)
    """
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {model_name}')
    if not lora_weights:
        raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    # if "LLaMA" in args.model:
    #     tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # else:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    if device == "cuda":
        if adapter != "no":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )  # fix zwq
            if adapter not in ["orig", "super", "supra"]:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16,
                    device_map={"": 0}
                )
            elif adapter == "super":
                # TODO rewrite, for now it is just copypaste from `finetune.py`
                # (requires the same input arguments as training)
                from dense_plus_sparse_linear import get_dense_plus_sparse_model
                from safetensors.torch import load_file
                import glob
                print(model)
                model = get_dense_plus_sparse_model(
                    model,
                    target_modules_list=target_modules,
                    r=r,
                    indices_choice="random",
                )
                print(model)

                def load_safetensors_model(model_path):
                    if os.path.isfile(f"{model_path}"):
                        state_dict = load_file(f"{model_path}")
                        return state_dict

                    pattern = os.path.join(os.path.dirname(model_path), "model-*-of-*.safetensors")
                    print('\n' * 3)
                    print(pattern)
                    print('\n' * 3)
                    shard_files = sorted(glob.glob(pattern))
                    if not shard_files:
                        raise FileNotFoundError(f"No safetensors file or shards found for base path: {model_path}")

                    print(f"Found {len(shard_files)} shard files:")
                    for shard in shard_files:
                        print("  ", shard)

                    state_dict = {}
                    for shard in shard_files:
                        shard_state = load_file(shard)
                        state_dict.update(shard_state)

                    return state_dict

                state_dict = load_safetensors_model(f"{lora_weights}/model.safetensors")
                model.load_state_dict(state_dict, strict=False)
            elif adapter == "supra":
                from dense_plus_sparse_linear_plus_lora import get_dense_plus_sparse_plus_lora_model
                from safetensors.torch import load_file
                import glob
                print(model)
                model = get_dense_plus_sparse_plus_lora_model(
                    model,
                    target_modules_list=target_modules,
                    r_lora=r//2,
                    r_super=r//2,
                    indices_choice="random",
                )
                print(model)

                def load_safetensors_model(model_path):
                    if os.path.isfile(f"{model_path}"):
                        state_dict = load_file(f"{model_path}")
                        return state_dict

                    pattern = os.path.join(os.path.dirname(model_path), "model-*-of-*.safetensors")
                    print('\n' * 3)
                    print(pattern)
                    print('\n' * 3)
                    shard_files = sorted(glob.glob(pattern))
                    if not shard_files:
                        raise FileNotFoundError(f"No safetensors file or shards found for base path: {model_path}")

                    print(f"Found {len(shard_files)} shard files:")
                    for shard in shard_files:
                        print("  ", shard)

                    state_dict = {}
                    for shard in shard_files:
                        shard_state = load_file(shard)
                        state_dict.update(shard_state)

                    return state_dict

                state_dict = load_safetensors_model(f"{lora_weights}/model.safetensors")
                model.load_state_dict(state_dict, strict=False)

        else:
            model = AutoModelForCausalLM.from_pretrained(
                lora_weights,
                local_files_only=True,
                # load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer_number(dataset_name, sentence: str) -> float:
    if dataset_name.lower() in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset_name))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    args = parse_args()

    eval_model(args.dataset, args.model_name, args.adapter, args.base_model, args.lora_weights, args.load_8bit, args.debug, args.target_modules, args.r)
