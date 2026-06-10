# Monkey patching to force transformers ignore peft library 
# (for yandex infrastructure; in regular env one can just uninstall peft)

import importlib.util

original_find_spec = importlib.util.find_spec


def custom_find_spec(name, *args, **kwargs):
    if name == 'peft':
        return None
    return original_find_spec(name, *args, **kwargs)


importlib.util.find_spec = custom_find_spec

import os
import sys
from typing import List

import fire
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from typing import List, Optional, Union
from importlib.metadata import version

import sys
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel, Trainer  # noqa: F402
from transformers.trainer_utils import get_last_checkpoint

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PEFT_PATH = os.path.abspath(os.path.join(os.getcwd(), "peft/src/"))

sys.path.insert(0, PEFT_PATH)
sys.path.insert(1, BASE_DIR)

from peft import (
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from src.super import Super

from nirvana_utils import TrainerNirvana, copy_out_to_snapshot, copy_snapshot_to_out

from SIFT.sift import SIFT
from dense_plus_sparse_linear import get_dense_plus_sparse_model, get_sparse_dense_model_state_dict
from dense_plus_sparse_linear_plus_lora import get_dense_plus_sparse_plus_lora_model, \
    get_sparse_dense_lora_model_state_dict
from custom_lora import get_custom_lora_model, get_custom_lora_model_state_dict


def compute_sparse_rate(model, target_modules):
    def is_in_target_modules(_name, additional_weights="values"):
        if additional_weights in _name:
            return True

        for item in target_modules:
            if item in _name:
                return True
        return False

    num_trainable = 0
    num_non_trainable = 0
    for name, p in model.named_parameters():
        if is_in_target_modules(name):
            if p.requires_grad:
                num_trainable += p.data.numel()
            else:
                num_non_trainable += p.data.numel()

    return num_trainable, num_non_trainable, num_trainable / (num_non_trainable + 1e-9)


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca",
        overwrite_output_dir: bool = False,
        adapter_name: str = "lora",
        load_8bit: bool = False,
        sparse_rate=0.005962171052631579,
        calibration_data: str = "c4",
        calibration_nsamples: int = 128,
        calibration_seed: int = 228,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        load_from_checkpoints: bool = False,
        eval_step: int = 50,
        save_step: int = 50,
        val_split_seed: int = 42,
        seed=0,
        # lora hyperparams
        lora_r: int = 8,
        dynamic_r: bool = False,
        lora_params_ratio: float = 0.5,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        # torch_compile
        compile=False,
        attn_implementation="sdpa",

        # optimizer hyperparams
        optimizer_name: str = "adam",

        # debug params
        max_steps=-1,

        # SIFT params
        sparse_exception=[],
        random_indices=False,
):
    compile = bool(compile)
    sparse_module = target_modules
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"weight_decay: {weight_decay}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"val_split_seed: {val_split_seed}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"optimizer_name: {optimizer_name}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"compile: {compile}\n"
        f"attn_implementation: {attn_implementation}\n"
        f"optimizer_name: {optimizer_name}\n"
        f"max_steps: {max_steps}\n"
        f"sparse_exception: {sparse_exception}\n"
        f"random_indices: {random_indices}\n"
        f"calibration_data: {calibration_data}\n"
        f"calibration_nsamples: {calibration_nsamples}\n"
        f"calibration_seed: {calibration_seed}\n"
        f"seed: {seed}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16 if adapter_name != "sift" else torch.float32,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
            trust_remote_code=True,
            attn_implementation=attn_implementation
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    torch.manual_seed(seed)

    sift = None
    if adapter_name not in ["sift", "super", "no", "supra"]:
        model = get_peft_model(model, config)
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    elif adapter_name == "sift":
        sift = SIFT(
            model,
            sparse_rate=sparse_rate,
            sparse_module=sparse_module,
            exception=sparse_exception,
            grad_acc=gradient_accumulation_steps,
            random_indices=random_indices,
        )
    elif adapter_name == "super":
        model.seqlen = model.config.max_position_embeddings
        model = get_dense_plus_sparse_model(
            model,
            target_modules_list=target_modules,
            sparse_rate=sparse_rate,
            indices_choice="random" if random_indices else "super",
            tokenizer=tokenizer,
            exception=sparse_exception,
            calibration_data=calibration_data,
            calibration_nsamples=calibration_nsamples,
            calibration_seed=calibration_seed,
        )
        print('\n' * 3)
        print(model)
        print('\n' * 3)
    elif adapter_name == "supra":
        model.seqlen = model.config.max_position_embeddings
        model = get_dense_plus_sparse_plus_lora_model(
            model,
            lora_params_ratio=lora_params_ratio,
            sparse_rate=sparse_rate,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules_list=target_modules,
            indices_choice="random" if random_indices else "super",
            tokenizer=tokenizer,
            exception=sparse_exception,
            calibration_data=calibration_data,
            calibration_nsamples=calibration_nsamples,
            calibration_seed=calibration_seed,
        )
        print('\n' * 3)
        print(model)
        print('\n' * 3)
    elif adapter_name == "custom-lora":
        model.seqlen = model.config.max_position_embeddings
        model = get_custom_lora_model(
            model,
            r=lora_r,
            dynamic_r=dynamic_r,
            sparse_rate=sparse_rate,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules_list=target_modules,
            tokenizer=tokenizer,
            exception=sparse_exception,
        )
        print('\n' * 3)
        print(model)
        print('\n' * 3)
    elif adapter_name == "no":
        pass
    else:
        raise ValueError("Incorrect `adapter_name`")

    if adapter_name == "prefix-tuning":
        model.to('cuda')

    num_trainable, num_non_trainable, sp_rate = compute_sparse_rate(model=model, target_modules=target_modules)
    print("Initial number of parameters (non trainable):", num_non_trainable)
    print("Number of trainable params:", num_trainable)
    print("Sparse_rate =", sp_rate)

    if adapter_name == "sift" and sift is not None:
        trainable_params = list(sift.parameters_in_optimizer())
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_trainable_params = sum(p.numel() for p in trainable_params)
    requires_grad_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.optimizer_trainable_params = optimizer_trainable_params
    model.requires_grad_trainable_params = requires_grad_trainable_params
    print("Optimizer trainable params:", optimizer_trainable_params)
    print("Requires-grad params:", requires_grad_trainable_params)

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("wrong optimizer name.")

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    copy_snapshot_to_out(output_dir)
    last_checkpoint = None
    if (load_from_checkpoints):
        if os.path.isdir(output_dir) and not overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(output_dir)
            if last_checkpoint is None and len(os.listdir(output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and resume_from_checkpoint is None:
                print(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=val_split_seed
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    if not int(os.environ.get("LOCAL_RANK", 0)):
        ddp_find_unused_parameters = False if ddp and adapter_name not in ["sift"] else True
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        optimizers=(optimizer, None),
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            seed=seed,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp and adapter_name not in ["sift"] else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
            torch_compile=compile,

            max_steps=max_steps
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    if adapter_name in ["sift"]:
        sift.print_trainable_parameters()
        sift.set_trainer(trainer)
    model.config.use_cache = False

    # if adapter_name not in ["sift"]:
    # TODO load only adapter
    # if adapter_name not in ["sift", "super", "supra"]:

    get_state_dict_func = get_peft_model_state_dict
    if adapter_name == "super":
        get_state_dict_func = get_sparse_dense_model_state_dict
    elif adapter_name == "supra":
        get_state_dict_func = get_sparse_dense_lora_model_state_dict
    elif adapter_name == "custom-lora":
        get_state_dict_func = get_custom_lora_model_state_dict

    if adapter_name in ["super", "supra", "custom-lora"]:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_state_dict_func(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    '''
    checkpoint = None

    if load_from_checkpoints:
        if resume_from_checkpoint is not None:
            checkpoint = resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
            if (not int(os.environ.get("LOCAL_RANK", 0))
                    and use_wandb):
                import wandb
                import json
                with open(os.path.join(output_dir, "run_metadata.json"), 'r') as f:
                    run_metadata = json.load(f)
                wandb.init(
                    project=run_metadata["project"],
                    id=run_metadata["run_id"],
                    name=run_metadata.get("run_name"),
                    entity=run_metadata.get("entity"),
                    resume="must"
                )
                print(f"Resumed run: {wandb.run.name} (ID: {wandb.run.id})")
    
    trainer.train(resume_from_checkpoint=checkpoint)
    '''

    trainer.train()

    if not int(os.environ.get("LOCAL_RANK", 0)):
        # save pretrained
        model.save_pretrained(output_dir)

        # # some yandex infrastructure logic
        # os.system(f"rm -rf {output_dir}/checkpoint*")
        print('\n' * 10)
        print("copying result to snapshot")
        print('-' * 20)
        print("LS OUTPUT_DIR")
        os.system(f"ls {output_dir}")
        print('-' * 20)
        copy_out_to_snapshot(output_dir)

    return model, tokenizer


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501


if __name__ == "__main__":
    print("CUDA Available:", torch.cuda.is_available())
    for __i in range(torch.cuda.device_count()):
        print(f"GPU {__i}: {torch.cuda.get_device_name(__i)}")

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    fire.Fire(train)
