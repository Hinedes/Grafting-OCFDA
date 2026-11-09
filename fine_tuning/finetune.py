# Adapted from tloen/alpaca-lora's Apache-2.0-licensed training script.
# Modified for Super-Tuning's sparse adapters and Math17K experiment pipeline.

import json
import os
from hashlib import sha256
from importlib.metadata import version
from typing import List, Optional

import fire
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.trainer_utils import get_last_checkpoint

from dense_plus_sparse_linear import get_dense_plus_sparse_model, get_sparse_dense_model_state_dict
from dense_plus_sparse_linear_plus_lora import (
    get_dense_plus_sparse_plus_lora_model,
    get_sparse_dense_lora_model_state_dict,
)

try:
    from .artifacts import sha256_file
except ImportError:
    from artifacts import sha256_file

try:
    from .ocfda import (
        OCFDA_PROJECTIONS,
        get_ocfda_model,
        get_ocfda_model_state_dict,
        host_tensor_hashes,
        verify_host_tensor_hashes,
        verify_ocfda_detach,
        verify_optimizer_ownership,
        verify_zero_graft_noop,
    )
except ImportError:
    from ocfda import (
        OCFDA_PROJECTIONS,
        get_ocfda_model,
        get_ocfda_model_state_dict,
        host_tensor_hashes,
        verify_host_tensor_hashes,
        verify_ocfda_detach,
        verify_optimizer_ownership,
        verify_zero_graft_noop,
    )

try:
    from .baselines import SIFT
    from .training_curve_utils import TrainingCurveCallback
except ImportError:
    from baselines import SIFT
    from training_curve_utils import TrainingCurveCallback


SUPPORTED_ADAPTERS = {"lora", "sift", "super", "supra", "ocfda", "no"}


def resolve_artifact_path(path, output_dir):
    """Resolve a path returned by ``save_pretrained`` relative to ``output_dir``.

    Save implementations may return basenames, output-dir-relative paths, or
    absolute paths. Use the returned path directly when it already exists;
    otherwise treat it as relative to ``output_dir``. This prevents the
    historical doubling bug where an output-dir-prefixed relative path was
    joined onto ``output_dir`` a second time.
    """
    path = os.fspath(path)
    if os.path.exists(path):
        return path
    return os.path.join(output_dir, path)


def compute_sparse_rate(model, target_modules):
    def is_in_target_modules(_name, additional_weights="values"):
        if additional_weights in _name or "graft_delta" in _name:
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
        method_name: str = "",
        load_8bit: bool = False,
        sparse_rate=0.005962171052631579,
        calibration_data: str = "c4",
        calibration_nsamples: int = 128,
        calibration_seed: int = 228,
        full_ft_checkpoint: str = "",
        model_revision: str = "",
        tokenizer_revision: str = "",
        support_seed: int = 0,
        ocfda_geometry: str = "aligned",
        ocfda_k: int = 57,
        artifact_manifest_path: str = "",
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
        warmup_steps: int = 100,
        val_split_seed: int = 42,
        seed=0,
        # lora hyperparams
        lora_r: int = 8,
        lora_params_ratio: float = 0.5,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"],
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
        save_model: bool = True,
        logging_steps: int = 10,
        training_curve_path: str = "",
        training_curve_metadata: Optional[dict] = None,
        bf16: bool = False,

        # SIFT params
        sparse_exception=[],
        random_indices=False,
        mask_choice=None,
):
    if adapter_name not in SUPPORTED_ADAPTERS:
        supported = ", ".join(sorted(SUPPORTED_ADAPTERS))
        raise ValueError(f"Unsupported adapter_name={adapter_name!r}. Choose one of: {supported}.")
    if adapter_name == "ocfda":
        target_modules = list(OCFDA_PROJECTIONS)
        if not artifact_manifest_path:
            raise ValueError("B1 OCFDA requires an artifact manifest")
        if optimizer_name.lower() != "adamw" or weight_decay != 0.0:
            raise ValueError("B1 OCFDA fixes AdamW with zero weight decay")
        if ocfda_k != 57:
            raise ValueError("B1 OCFDA fixes k=57")
    if mask_choice is None:
        mask_choice = "random" if random_indices else "super"
    if not (
        mask_choice in {
            "random",
            "super",
            "super-bottom",
            "super-bottom-structured",
            "magnitude",
            "magnitude-bottom",
            "full-delta",
            "full-delta-naive",
        }
        or mask_choice.startswith("super-hybrid-")
    ):
        raise ValueError(
            "mask_choice must be 'random', 'super', 'super-bottom', 'super-bottom-structured', 'super-hybrid-<beta>', "
            "'magnitude', 'magnitude-bottom', 'full-delta', or 'full-delta-naive'."
        )
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
        f"optimizer_name: {optimizer_name}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"adapter_name: {adapter_name}\n"
        f"method_name: {method_name or adapter_name}\n"
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
        f"warmup_steps: {warmup_steps}\n"
        f"max_steps: {max_steps}\n"
        f"save_model: {save_model}\n"
        f"sparse_exception: {sparse_exception}\n"
        f"random_indices: {random_indices}\n"
        f"mask_choice: {mask_choice}\n"
        f"calibration_data: {calibration_data}\n"
        f"calibration_nsamples: {calibration_nsamples}\n"
        f"calibration_seed: {calibration_seed}\n"
        f"full_ft_checkpoint: {full_ft_checkpoint}\n"
        f"model_revision: {model_revision}\n"
        f"tokenizer_revision: {tokenizer_revision}\n"
        f"support_seed: {support_seed}\n"
        f"ocfda_geometry: {ocfda_geometry}\n"
        f"ocfda_k: {ocfda_k}\n"
        f"artifact_manifest_path: {artifact_manifest_path}\n"
        f"seed: {seed}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    artifact_manifest = None
    artifact_manifest_sha256 = None
    if artifact_manifest_path:
        with open(artifact_manifest_path, "rb") as manifest_file:
            manifest_bytes = manifest_file.read()
        artifact_manifest = json.loads(manifest_bytes)
        artifact_manifest_sha256 = sha256(manifest_bytes).hexdigest()
    if adapter_name == "ocfda" and (
        artifact_manifest is None or artifact_manifest.get("protocol") != "B1-OCFDA"
    ):
        raise ValueError("B1 OCFDA requires a B1 artifact manifest")
    if adapter_name == "ocfda" and (
        artifact_manifest.get("model") != base_model
        or artifact_manifest.get("model_revision") != model_revision
        or artifact_manifest.get("tokenizer_revision") != (tokenizer_revision or model_revision)
    ):
        raise ValueError("B1 OCFDA training arguments do not match the artifact manifest model pin")
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

    if not torch.cuda.is_available():
        model_dtype = torch.float32
        device_map = None
        effective_bf16 = False
    else:
        effective_bf16 = bool(bf16)
        model_dtype = torch.float32 if adapter_name == "sift" else (torch.bfloat16 if effective_bf16 else torch.float16)
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}

    revision_kwargs = {"revision": model_revision} if model_revision else {}
    tokenizer_revision = tokenizer_revision or model_revision
    tokenizer_revision_kwargs = {"revision": tokenizer_revision} if tokenizer_revision else {}

    if load_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=model_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **revision_kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=model_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            **revision_kwargs,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, **tokenizer_revision_kwargs)

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

    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    ocfda_host_hashes = None
    if adapter_name == "ocfda":
        ocfda_host_hashes = host_tensor_hashes(model)

    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    torch.manual_seed(seed)

    sift = None
    if adapter_name == "lora":
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
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
            indices_choice=mask_choice,
            tokenizer=tokenizer,
            exception=sparse_exception,
            calibration_data=calibration_data,
            calibration_nsamples=calibration_nsamples,
            calibration_seed=calibration_seed,
            full_ft_checkpoint=full_ft_checkpoint,
        )
    elif adapter_name == "supra":
        model.seqlen = model.config.max_position_embeddings
        model = get_dense_plus_sparse_plus_lora_model(
            model,
            lora_params_ratio=lora_params_ratio,
            sparse_rate=sparse_rate,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules_list=target_modules,
            indices_choice=mask_choice,
            tokenizer=tokenizer,
            exception=sparse_exception,
            calibration_data=calibration_data,
            calibration_nsamples=calibration_nsamples,
            calibration_seed=calibration_seed,
        )
    elif adapter_name == "ocfda":
        model = get_ocfda_model(
            model,
            geometry=ocfda_geometry,
            support_seed=support_seed,
            k=ocfda_k,
        )
        verify_zero_graft_noop(model)
    elif adapter_name == "no":
        pass

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
    if adapter_name == "ocfda":
        verify_optimizer_ownership(optimizer, model)

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    last_checkpoint = None
    if load_from_checkpoints:
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
            train_val["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer_callbacks = []
    if training_curve_path:
        trainer_callbacks.append(TrainingCurveCallback(training_curve_path, training_curve_metadata))

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        optimizers=(optimizer, None),
        callbacks=trainer_callbacks or None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            seed=seed,
            fp16=adapter_name != "sift" and not effective_bf16 and torch.cuda.is_available(),
            bf16=effective_bf16,
            logging_steps=logging_steps,
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps" if save_model else "no",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 and save_model else False,
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
    elif adapter_name == "ocfda":
        get_state_dict_func = get_ocfda_model_state_dict

    if adapter_name in ["super", "supra", "ocfda"]:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_state_dict_func(
                self, old_state_dict()
            )
        ).__get__(model, type(model))

    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)

    checkpoint = resume_from_checkpoint
    if load_from_checkpoints and checkpoint is None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ocfda_ownership = None
    if adapter_name == "ocfda":
        host_report = verify_host_tensor_hashes(model, ocfda_host_hashes)
        ocfda_ownership = {
            "host": host_report,
            "detach": verify_ocfda_detach(model, ocfda_host_hashes, host_report=host_report),
            "optimizer": verify_optimizer_ownership(optimizer, model),
        }
        model.ocfda_ownership_report = ocfda_ownership

    if save_model and not int(os.environ.get("LOCAL_RANK", 0)):
        if adapter_name == "sift":
            dense_state = {
                key: value
                for key, value in model.state_dict().items()
                if not key.startswith("_sift_")
            }
            model.save_pretrained(output_dir, state_dict=dense_state)
        else:
            model.save_pretrained(output_dir)
        tokenizer_files = tokenizer.save_pretrained(output_dir) or ()
        tokenizer_artifacts = {}
        for tokenizer_file in tokenizer_files:
            tokenizer_file = resolve_artifact_path(tokenizer_file, output_dir)
            relative_path = os.path.relpath(tokenizer_file, output_dir).replace(os.sep, "/")
            tokenizer_artifacts[relative_path] = sha256_file(tokenizer_file)
        if adapter_name == "ocfda" and artifact_manifest is not None:
            with open(os.path.join(output_dir, "artifact_manifest.json"), "wb") as manifest_file:
                manifest_file.write(manifest_bytes)
        metadata = {
            "format_version": 1,
            "method": method_name or adapter_name,
            "adapter_name": adapter_name,
            "base_model": base_model,
            "target_modules": list(target_modules),
            "sparse_rate": float(sparse_rate),
            "mask_choice": mask_choice,
            "lora_r": int(lora_r),
            "lora_params_ratio": float(lora_params_ratio),
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "calibration_data": calibration_data,
            "calibration_nsamples": int(calibration_nsamples),
            "calibration_seed": int(calibration_seed),
            "bf16": bool(effective_bf16),
            "model_revision": model_revision or None,
            "tokenizer_revision": tokenizer_revision or None,
            "training_seed": int(seed),
            "weight_decay": float(weight_decay),
            "optimizer_name": optimizer_name,
        }
        if adapter_name == "ocfda":
            metadata.update(
                {
                    "protocol": "B1-OCFDA",
                    "geometry": ocfda_geometry,
                    "support_seed": int(support_seed),
                    "ocfda_k": int(ocfda_k),
                    "ocfda_trainable_scalars": int(model.optimizer_trainable_params),
                    "ocfda_supports": model.ocfda_supports,
                    "ocfda_ownership": ocfda_ownership,
                    "artifact_manifest_path": os.path.abspath(artifact_manifest_path)
                    if artifact_manifest_path
                    else None,
                    "artifact_manifest_sha256": artifact_manifest_sha256,
                    "tokenizer_files": tokenizer_artifacts,
                    "dataset_artifacts": {
                        key: artifact_manifest[key]
                        for key in ("train_data", "heldout_dataset", "benchmark_source")
                        if artifact_manifest is not None and key in artifact_manifest
                    },
                }
            )
        with open(os.path.join(output_dir, "supertuning_config.json"), "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)

    if adapter_name == "ocfda" and not int(os.environ.get("LOCAL_RANK", 0)):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "ocfda_ownership.json"), "w", encoding="utf-8") as ownership_file:
            json.dump(ocfda_ownership, ownership_file, indent=2, sort_keys=True)

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
    print("GPU Available:", torch.cuda.is_available())
    print("PyTorch HIP:", getattr(torch.version, "hip", None))
    print("PyTorch CUDA compatibility version:", getattr(torch.version, "cuda", None))
    for __i in range(torch.cuda.device_count()):
        print(f"GPU {__i}: {torch.cuda.get_device_name(__i)}")

    print('torch', version('torch'))
    print('transformers', version('transformers'))
    print('accelerate', version('accelerate'))
    print('# of gpus: ', torch.cuda.device_count())

    fire.Fire(train)
