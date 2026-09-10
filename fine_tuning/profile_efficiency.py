import argparse
import csv
import gc
import json
import math
import os
import platform
import shutil
import time
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from dense_plus_sparse_linear import (
    get_dense_plus_sparse_model,
    get_sparse_dense_model_state_dict,
)
from dense_plus_sparse_linear_plus_lora import (
    get_dense_plus_sparse_plus_lora_model,
    get_sparse_dense_lora_model_state_dict,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from .baselines import SIFT
    from .math_experiment_tables import (
        FULL_LLAMA_TARGET_MODULES,
        LEGACY_TARGET_MODULES,
        RunSpec,
        build_budget_plan,
        parse_csv_list,
        parse_method,
    )
except ImportError:
    from baselines import SIFT
    from math_experiment_tables import (
        FULL_LLAMA_TARGET_MODULES,
        LEGACY_TARGET_MODULES,
        RunSpec,
        build_budget_plan,
        parse_csv_list,
        parse_method,
    )

DEFAULT_METHODS = (
    "full,rosa,sift-topk,lora,super-wanda-bottom,magnitude-bottomk,"
    "supra-0.8-bottom,supra-magnitude-0.3"
)
DEFAULT_MODELS = "meta-llama/Llama-3.2-1B,meta-llama/Meta-Llama-3-8B"

METHOD_LABELS = {
    "full": "Full FT",
    "rosa": "\\algname{RoSA}",
    "sift-topk": "\\algname{SIFT} (TopK)",
    "lora": "\\algname{LoRA}",
    "super-wanda-bottom": "\\algname{Super} (BottomK)",
    "magnitude-bottomk": "\\algname{Magnitude} (BottomK)",
    "supra-0.8-bottom": "\\algname{Supra} (BottomK, $\\lambda=0.8$)",
    "supra-magnitude-0.3": "\\algname{Supra-Mag} (BottomK, $\\lambda=0.3$)",
}


def import_rosa_components():
    try:
        from .rosa.rosa.scheduler import RosaScheduler
        from .rosa.rosa_adapter import get_rosa_model, get_rosa_model_state_dict
    except ImportError:
        from rosa.rosa.scheduler import RosaScheduler
        from rosa.rosa_adapter import get_rosa_model, get_rosa_model_state_dict
    return get_rosa_model, get_rosa_model_state_dict, RosaScheduler


class ProfilingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_tokens = 0

    def training_step(self, model, inputs, *args, **kwargs):
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            self.profile_tokens += int(attention_mask.detach().sum().item())
        return super().training_step(model, inputs, *args, **kwargs)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_prompt(data_point: dict) -> str:
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""


def model_label(model_name: str) -> str:
    short = model_name.split("/")[-1]
    if short == "Llama-3.2-1B":
        return "\\texttt{Llama-3.2-1B}"
    if short == "Meta-Llama-3-8B":
        return "\\texttt{Meta-Llama-3-8B}"
    return "\\texttt{" + short.replace("_", "\\_") + "}"


def plain_model_label(model_name: str) -> str:
    return model_name.split("/")[-1]


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", "\\_"))


def sparse_impl_label(method: str) -> str:
    adapter_name, mask_choice, _ = parse_method(method)
    if adapter_name == "full":
        return "dense weights"
    if adapter_name == "lora":
        return "--"
    if adapter_name == "rosa":
        return "sparse values + LoRA"
    if adapter_name == "sift":
        return "sparse values; dense grads"
    if adapter_name == "super":
        return "sparse values; dense kernels"
    if adapter_name == "supra":
        if mask_choice == "magnitude-bottom":
            return "sparse values + LoRA; dense kernels"
        return "sparse values + LoRA; dense kernels"
    return "--"


def sparse_impl_machine(method: str) -> str:
    adapter_name, mask_choice, _ = parse_method(method)
    if adapter_name == "full":
        return "dense full-model parameters"
    if adapter_name == "lora":
        return "dense low-rank LoRA tensors"
    if adapter_name == "rosa":
        return "low-rank tensors plus sparse trainable values/indices"
    if adapter_name == "sift":
        return "sparse trainable values/indices with dense gradients gathered by hooks"
    if adapter_name == "super":
        return f"sparse trainable values/indices ({mask_choice}) with dense linear kernels and dense gradient gather"
    if adapter_name == "supra":
        return f"sparse trainable values/indices ({mask_choice}) plus LoRA with dense linear kernels and dense gradient gather"
    return "unknown"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def optimizer_state_nbytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += tensor_nbytes(value)
            elif isinstance(value, dict):
                total += sum(tensor_nbytes(v) for v in value.values() if torch.is_tensor(v))
    return int(total)


def directory_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return int(total)


def get_adapter_state_dict(model, adapter_name: str, sift: Optional[SIFT]) -> Dict[str, torch.Tensor]:
    if adapter_name == "lora":
        return {key: value.detach() for key, value in get_peft_model_state_dict(model).items()}
    if adapter_name == "super":
        return {key: value.detach() for key, value in get_sparse_dense_model_state_dict(model).items()}
    if adapter_name == "supra":
        return {key: value.detach() for key, value in get_sparse_dense_lora_model_state_dict(model).items()}
    if adapter_name == "rosa":
        _, get_rosa_model_state_dict, _ = import_rosa_components()
        return {key: value.detach() for key, value in get_rosa_model_state_dict(model).items()}
    if adapter_name == "sift":
        if sift is None:
            raise ValueError("SIFT state requested before SIFT wrapper was constructed.")
        return sift.adapter_state_dict()
    raise ValueError(f"No adapter state for adapter type: {adapter_name}")


def save_profile_checkpoint(
    model,
    tokenizer,
    method: str,
    checkpoint_dir: str,
    sift: Optional[SIFT],
    metadata: dict,
) -> int:
    adapter_name, _, _ = parse_method(method)
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if adapter_name == "full":
        previous_use_cache = getattr(model.config, "use_cache", None)
        if previous_use_cache is not None:
            model.config.use_cache = True
        try:
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
        finally:
            if previous_use_cache is not None:
                model.config.use_cache = previous_use_cache
    else:
        state_dict = get_adapter_state_dict(model, adapter_name, sift)
        torch.save(state_dict, os.path.join(checkpoint_dir, "adapter_state.pt"))
        with open(os.path.join(checkpoint_dir, "profile_adapter_config.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    return directory_size_bytes(checkpoint_dir)


def make_tokenizer(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return tokenizer


def tokenize_dataset(args, tokenizer):
    def tokenize(prompt: str, add_eos_token: bool = True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        return tokenize(generate_prompt(data_point))

    if args.train_data.endswith(".json"):
        data = load_dataset("json", data_files=args.train_data)
    else:
        data = load_dataset(args.train_data)

    if args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size,
            shuffle=True,
            seed=args.val_split_seed,
        )
        train_data = train_val["train"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt)
    return train_data


def profile_model_dtype(adapter_name: str, args) -> torch.dtype:
    if adapter_name == "sift":
        return torch.float32
    if adapter_name == "full" or adapter_name == "rosa" or args.bf16:
        return torch.bfloat16
    return torch.float16


def build_model_and_optimizer(
    args,
    spec: RunSpec,
    budget_plan: dict,
    target_modules: List[str],
) -> Tuple[object, object, torch.optim.Optimizer, Optional[SIFT], List[object], dict]:
    adapter_name, mask_choice, lora_params_ratio = parse_method(spec.method)
    if adapter_name == "rosa":
        lora_params_ratio = args.rosa_lora_budget_ratio

    profile_stats: Dict[str, Optional[float]] = {"calibration_time_sec": None}
    tokenizer = make_tokenizer(spec.model)
    dtype = profile_model_dtype(adapter_name, args)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model,
        load_in_8bit=False,
        torch_dtype=dtype,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    torch.manual_seed(spec.seed)

    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)
    sift = None
    callbacks: List[object] = []

    if adapter_name == "full":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif adapter_name == "lora":
        config = LoraConfig(
            r=budget_plan["train_lora_r"],
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    elif adapter_name == "sift":
        sift = SIFT(
            model,
            sparse_rate=budget_plan["train_sparse_rate"],
            sparse_module=target_modules,
            exception=[],
            grad_acc=gradient_accumulation_steps,
            random_indices=(mask_choice == "random"),
        )
    elif adapter_name == "super":
        model.seqlen = model.config.max_position_embeddings
        model = get_dense_plus_sparse_model(
            model,
            target_modules_list=target_modules,
            sparse_rate=budget_plan["train_sparse_rate"],
            indices_choice=mask_choice,
            tokenizer=tokenizer,
            exception=[],
            calibration_data=args.calibration_data,
            calibration_nsamples=args.calibration_nsamples,
            calibration_seed=args.calibration_seed,
            profile_stats=profile_stats,
        )
    elif adapter_name == "supra":
        model.seqlen = model.config.max_position_embeddings
        model = get_dense_plus_sparse_plus_lora_model(
            model,
            lora_params_ratio=lora_params_ratio,
            sparse_rate=budget_plan["train_sparse_rate"],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules_list=target_modules,
            indices_choice=mask_choice,
            tokenizer=tokenizer,
            exception=[],
            calibration_data=args.calibration_data,
            calibration_nsamples=args.calibration_nsamples,
            calibration_seed=args.calibration_seed,
            profile_stats=profile_stats,
        )
    elif adapter_name == "rosa":
        get_rosa_model, _, RosaScheduler = import_rosa_components()
        model.seqlen = model.config.max_position_embeddings
        model = get_rosa_model(
            model,
            target_modules=target_modules,
            r=budget_plan["train_lora_r"],
            d=budget_plan["train_sparse_rate"],
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            impl=args.rosa_impl,
            schedule=args.rosa_schedule,
            spa_num_grads=args.rosa_spa_num_grads,
            rosa_dtype=args.rosa_dtype,
        )
        callbacks.append(RosaScheduler(model))
    else:
        raise ValueError(f"Unsupported profile method: {spec.method}")

    if adapter_name == "sift" and sift is not None:
        trainable_params = list(sift.parameters_in_optimizer())
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_trainable_params = sum(p.numel() for p in trainable_params)
    requires_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.optimizer_trainable_params = optimizer_trainable_params
    model.requires_grad_trainable_params = requires_grad_params

    if args.optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=spec.lr, weight_decay=args.weight_decay)
    elif args.optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=spec.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer_name}")

    return model, tokenizer, optimizer, sift, callbacks, profile_stats


def profile_one(args, spec: RunSpec, target_modules: List[str]) -> dict:
    set_seed(spec.seed)
    budget_plan = build_budget_plan(args, spec, target_modules)
    adapter_name, _, _ = parse_method(spec.method)
    model = tokenizer = optimizer = sift = None
    try:
        model, tokenizer, optimizer, sift, callbacks, profile_stats = build_model_and_optimizer(
            args=args,
            spec=spec,
            budget_plan=budget_plan,
            target_modules=target_modules,
        )
        train_data = tokenize_dataset(args, tokenizer)

        use_bf16_training = profile_model_dtype(adapter_name, args) == torch.bfloat16
        use_fp16_training = profile_model_dtype(adapter_name, args) == torch.float16
        trainer = ProfilingTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=None,
            optimizers=(optimizer, None),
            callbacks=callbacks or None,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=args.micro_batch_size,
                gradient_accumulation_steps=max(1, args.batch_size // args.micro_batch_size),
                warmup_steps=args.warmup_steps,
                num_train_epochs=1,
                learning_rate=spec.lr,
                seed=spec.seed,
                fp16=use_fp16_training,
                bf16=use_bf16_training,
                logging_steps=args.logging_steps,
                eval_strategy="no",
                save_strategy="no",
                output_dir=os.path.join(args.output_dir, "trainer_tmp", spec.run_id),
                group_by_length=False,
                report_to="none",
                max_steps=args.profile_steps,
                torch_compile=bool(args.compile),
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        if adapter_name == "sift" and sift is not None:
            sift.set_trainer(trainer)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        trainer.train()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_time_sec = time.perf_counter() - start

        completed_steps = int(trainer.state.global_step)
        tokens = int(trainer.profile_tokens)
        steps_per_sec = completed_steps / wall_time_sec if wall_time_sec > 0 else float("nan")
        tokens_per_sec = tokens / wall_time_sec if wall_time_sec > 0 else float("nan")
        peak_allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
        optimizer_bytes = optimizer_state_nbytes(optimizer)

        checkpoint_dir = os.path.join(args.output_dir, "checkpoints", spec.run_id)
        metadata = {
            "method": spec.method,
            "model": spec.model,
            "rank_equivalent_r0": spec.lora_r,
            "target_modules": target_modules,
            "train_sparse_rate": budget_plan["train_sparse_rate"],
            "train_lora_r": budget_plan["train_lora_r"],
            "component_lora_ratio": budget_plan["component_lora_ratio"],
            "sparse_impl": sparse_impl_machine(spec.method),
        }
        checkpoint_size = save_profile_checkpoint(
            model=model,
            tokenizer=tokenizer,
            method=spec.method,
            checkpoint_dir=checkpoint_dir,
            sift=sift,
            metadata=metadata,
        )

        return {
            "completed": True,
            "run_id": spec.run_id,
            "model": spec.model,
            "model_label": plain_model_label(spec.model),
            "method": spec.method,
            "method_label": method_label(spec.method),
            "rank_equivalent_r0": spec.lora_r,
            "learning_rate": spec.lr,
            "seed": spec.seed,
            "train_data": args.train_data,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "cutoff_len": args.cutoff_len,
            "profile_steps_requested": args.profile_steps,
            "profile_steps_completed": completed_steps,
            "tokens": tokens,
            "wall_time_sec": wall_time_sec,
            "steps_per_sec": steps_per_sec,
            "tokens_per_sec": tokens_per_sec,
            "trainable_params": int(getattr(model, "optimizer_trainable_params")),
            "requires_grad_params": int(getattr(model, "requires_grad_trainable_params")),
            "reference_lora_params": int(budget_plan["reference_lora_params"]),
            "target_dense_params": int(budget_plan["target_dense_params"]),
            "train_sparse_rate": budget_plan["train_sparse_rate"],
            "train_lora_r": int(budget_plan["train_lora_r"]),
            "component_lora_ratio": budget_plan["component_lora_ratio"],
            "adapter_trainable_params_estimate": budget_plan["adapter_trainable_params_estimate"],
            "adapter_budget_error_pct": budget_plan["adapter_budget_error_pct"],
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_size_bytes": checkpoint_size,
            "calibration_time_sec": profile_stats.get("calibration_time_sec"),
            "peak_memory_allocated_bytes": peak_allocated,
            "peak_memory_reserved_bytes": peak_reserved,
            "optimizer_state_bytes": optimizer_bytes,
            "sparse_impl": sparse_impl_machine(spec.method),
            "sparse_impl_table": sparse_impl_label(spec.method),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
    finally:
        del model
        del tokenizer
        del optimizer
        del sift
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def human_count(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    value = float(value)
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.1f}K"
    return str(int(value))


def human_bytes(value: Optional[float]) -> str:
    if value is None or value == "" or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    value = float(value)
    gib = 1024 ** 3
    mib = 1024 ** 2
    if value >= gib:
        return f"{value / gib:.2f} GiB"
    return f"{value / mib:.1f} MiB"


def human_seconds(value: Optional[float]) -> str:
    if value is None or value == "" or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    value = float(value)
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    if value >= 60:
        return f"{value / 60:.2f} min"
    return f"{value:.1f} s"


def human_rate(value: Optional[float], digits: int = 2) -> str:
    if value is None or value == "" or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def sort_rows(rows: Iterable[dict], model_order: List[str], method_order: List[str]) -> List[dict]:
    model_rank = {model: idx for idx, model in enumerate(model_order)}
    method_rank = {method: idx for idx, method in enumerate(method_order)}
    return sorted(
        rows,
        key=lambda row: (
            model_rank.get(row.get("model"), 999),
            method_rank.get(row.get("method"), 999),
            row.get("method", ""),
        ),
    )


def write_latex_table(path: str, rows: List[dict], caption: str, label: str) -> None:
    model_row_counts: Dict[str, int] = {}
    for row in rows:
        model_row_counts[row["model"]] = model_row_counts.get(row["model"], 0) + 1

    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{clrrrrrrrr}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Method} & \\textbf{Trainable Params} & "
        "\\textbf{Ckpt. Size} & \\textbf{Calib. Time} & \\textbf{Peak Mem.} & "
        "\\textbf{Adam State} & \\textbf{Steps/s} & \\textbf{Tokens/s} & "
        "\\textbf{Wall Time} \\\\",
        "\\midrule",
    ]
    previous_model = None
    seen_model_rows: Dict[str, int] = {}
    for row in rows:
        if previous_model is not None and row["model"] != previous_model:
            lines.append("\\midrule")
        previous_model = row["model"]
        seen_count = seen_model_rows.get(row["model"], 0)
        if seen_count == 0:
            model_cell = f"\\multirow{{{model_row_counts[row['model']]}}}{{*}}{{{model_label(row['model'])}}}"
        else:
            model_cell = ""
        seen_model_rows[row["model"]] = seen_count + 1
        lines.append(
            " & ".join(
                [
                    model_cell,
                    method_label(row["method"]),
                    human_count(row.get("trainable_params")),
                    human_bytes(row.get("checkpoint_size_bytes")),
                    human_seconds(row.get("calibration_time_sec")),
                    human_bytes(row.get("peak_memory_allocated_bytes")),
                    human_bytes(row.get("optimizer_state_bytes")),
                    human_rate(row.get("steps_per_sec"), digits=3),
                    human_rate(row.get("tokens_per_sec"), digits=0),
                    human_seconds(row.get("wall_time_sec")),
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table}",
            "",
        ]
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def materialize_outputs(args, rows: List[dict], model_order: List[str], method_order: List[str]) -> None:
    completed = [row for row in rows if row.get("completed")]
    sorted_rows = sort_rows(completed, model_order=model_order, method_order=method_order)
    write_csv(os.path.join(args.output_dir, "efficiency_profile.csv"), sorted_rows)
    write_latex_table(
        os.path.join(args.output_dir, "efficiency_profile_table.tex"),
        sorted_rows,
        caption=(
            "Implementation-level efficiency measurements for Math17K profiling runs. "
            "Both models use rank-equivalent budget $r_0=8$, batch size 16, micro-batch size 16, "
            f"and {args.profile_steps} optimizer steps on a single GPU. "
            "Checkpoint size is the saved full-model checkpoint for full fine-tuning and the saved adapter state otherwise. "
            "Peak memory is peak GPU allocated memory during the measured training region. "
            "Adam state is measured directly from tensor storage present in the instantiated optimizer after the profiling steps; "
            "it reflects the actual parameter dtypes and optimizer-state representation used by each method rather than a uniform "
            "two-FP32-state estimate. In particular, full fine-tuning stores two bf16 Adam moment tensors in these runs, so its "
            "state occupies approximately the same memory as one FP32 tensor per trainable parameter."
        ),
        label="tab:efficiency_measurements",
    )


def iter_specs(args) -> Iterable[RunSpec]:
    for model in parse_csv_list(args.models, str):
        for method in parse_csv_list(args.methods, str):
            yield RunSpec(
                seed=args.seed,
                model=model,
                lora_r=args.lora_r,
                lr=args.learning_rate,
                method=method,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile implementation-level PEFT efficiency.")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--output_dir", default="out_efficiency_profile")
    parser.add_argument(
        "--train_data",
        default=os.path.join(SCRIPT_DIR, "ft-training_set", "math_17k.json"),
    )
    parser.add_argument("--calibration_data", default="c4")
    parser.add_argument("--calibration_nsamples", type=int, default=128)
    parser.add_argument("--calibration_seed", type=int, default=228)
    parser.add_argument("--target_modules", default=",".join(FULL_LLAMA_TARGET_MODULES))
    parser.add_argument("--legacy_target_modules", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--rosa_lora_budget_ratio", type=float, default=0.5)
    parser.add_argument("--rosa_schedule", default="wl64")
    parser.add_argument("--rosa_spa_num_grads", type=int, default=1)
    parser.add_argument("--rosa_dtype", default="bf16")
    parser.add_argument("--rosa_impl", default="sp_add")
    parser.add_argument("--sparse_rate_override", type=float, default=None)
    parser.add_argument("--budget_tolerance_pct", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--val_set_size", type=int, default=120)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--profile_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer_name", default="adam")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--only_missing", action="store_true", default=True)
    parser.add_argument("--rerun_existing", dest="only_missing", action="store_false")
    parser.add_argument("--continue_on_error", action="store_true", default=True)
    parser.add_argument("--stop_on_error", dest="continue_on_error", action="store_false")
    parser.add_argument("--merge_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "efficiency_profile.jsonl")
    failures_path = os.path.join(args.output_dir, "efficiency_profile_failures.jsonl")
    model_order = parse_csv_list(args.models, str)
    method_order = parse_csv_list(args.methods, str)
    target_modules = LEGACY_TARGET_MODULES if args.legacy_target_modules else parse_csv_list(args.target_modules, str)

    existing = {row["run_id"]: row for row in read_jsonl(jsonl_path) if row.get("completed")}
    if not args.merge_only:
        for spec in iter_specs(args):
            if args.only_missing and spec.run_id in existing:
                print("Already profiled:", spec.run_id)
                continue
            print("=" * 100)
            print("Profiling", spec.model, spec.method)
            try:
                row = profile_one(args, spec, target_modules)
                append_jsonl(jsonl_path, row)
                existing[row["run_id"]] = row
                materialize_outputs(args, list(existing.values()), model_order, method_order)
                print("Profiled:", row["run_id"])
            except Exception as exc:  # noqa: BLE001
                failure = {
                    "completed": False,
                    "run_id": spec.run_id,
                    "model": spec.model,
                    "method": spec.method,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                append_jsonl(failures_path, failure)
                if not args.continue_on_error:
                    raise
                print("FAILED:", json.dumps(failure, indent=2))

    all_rows = read_jsonl(jsonl_path)
    materialize_outputs(args, all_rows, model_order, method_order)
    print("Wrote profiling results to:", args.output_dir)


if __name__ == "__main__":
    main()
