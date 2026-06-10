import argparse
import gc
import json
import math
import os
import pickle
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from importlib.metadata import version
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, REPO_DIR)

from evaluate import eval_model  # noqa: E402
from finetune import train  # noqa: E402


FULL_LLAMA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

LEGACY_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]
MATH_BENCHMARKS = ["AddSub", "MultiArith", "SingleEq", "gsm8k", "AQuA", "SVAMP"]
DEFAULT_METHODS = "base,lora,super-wanda,super-rand,supra-0.3,supra-0.5,supra-0.8,sift-topk,sift-rand,rosa"
DEFAULT_LRS = "5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1"


@dataclass(frozen=True)
class RunSpec:
    seed: int
    model: str
    lora_r: int
    lr: float
    method: str

    @property
    def run_id(self) -> str:
        model_name = self.model.split("/")[-1].replace(".", "_")
        lr = f"{self.lr:g}".replace("-", "m").replace(".", "p")
        return f"{model_name}_r{self.lora_r}_{self.method}_lr{lr}_seed{self.seed}"


def parse_csv_list(value: str, cast=str) -> List:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_environment() -> None:
    print("CUDA Available:", torch.cuda.is_available())
    for device_idx in range(torch.cuda.device_count()):
        print(f"GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
    for package in ["torch", "transformers", "accelerate", "datasets"]:
        try:
            print(package, version(package))
        except Exception as exc:  # noqa: BLE001
            print(package, f"unavailable ({exc})")


def first_present(row: dict, keys: Iterable[str]):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def finite_or_none(value):
    if value is None:
        return None
    try:
        is_na = pd.isna(value)
        if isinstance(is_na, bool) and is_na:
            return None
    except Exception:  # noqa: BLE001
        pass
    return value


def load_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def resolve_dataset_path(dataset_name: str) -> str:
    return os.path.join(SCRIPT_DIR, "dataset", dataset_name, "test.json")


def generate_prompt(instruction: str, input_text: Optional[str] = None) -> str:
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                """
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response:
                """


def get_gold_output(record: dict) -> str:
    output = record.get("output")
    if output:
        return str(output)
    answer = record.get("answer")
    return f"The answer is {answer}."


def get_model_input_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_perplexity_on_records(
    model,
    tokenizer,
    records: List[dict],
    max_length: int,
    max_examples: Optional[int] = None,
) -> Tuple[float, float, int]:
    model.eval()
    device = get_model_input_device(model)
    eos = tokenizer.eos_token or ""
    total_nll = 0.0
    total_tokens = 0
    used_examples = 0

    if max_examples is not None:
        records = records[:max_examples]

    for record in records:
        prompt = generate_prompt(record.get("instruction", ""), record.get("input"))
        target = get_gold_output(record)
        full_text = prompt + target + eos

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[:, :prompt_len] = -100
        token_count = int((labels != -100).sum().item())
        if token_count == 0:
            continue

        batch = {key: value.to(device) for key, value in encoded.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(**batch, labels=labels, use_cache=False)

        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count
        used_examples += 1

    if total_tokens == 0:
        return float("nan"), float("nan"), used_examples

    mean_nll = total_nll / total_tokens
    return float(math.exp(min(mean_nll, 50.0))), mean_nll, used_examples


def evaluate_perplexity(
    model,
    tokenizer,
    datasets: Iterable[str],
    max_length: int,
    max_examples: Optional[int],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    ppl_by_dataset: Dict[str, float] = {}
    nll_by_dataset: Dict[str, float] = {}
    count_by_dataset: Dict[str, int] = {}

    total_nll_weighted = 0.0
    total_examples = 0

    for dataset in datasets:
        records = load_json(resolve_dataset_path(dataset))
        ppl, nll, count = evaluate_perplexity_on_records(
            model=model,
            tokenizer=tokenizer,
            records=records,
            max_length=max_length,
            max_examples=max_examples,
        )
        ppl_by_dataset[dataset] = ppl
        nll_by_dataset[dataset] = nll
        count_by_dataset[dataset] = count
        if count and not math.isnan(nll):
            total_nll_weighted += nll * count
            total_examples += count
        print(f"{dataset} perplexity: {ppl:.4f} (nll={nll:.4f}, examples={count})")

    if total_examples:
        avg_nll = total_nll_weighted / total_examples
        ppl_by_dataset["Average"] = float(math.exp(min(avg_nll, 50.0)))
        nll_by_dataset["Average"] = avg_nll
    else:
        ppl_by_dataset["Average"] = float("nan")
        nll_by_dataset["Average"] = float("nan")
    return ppl_by_dataset, nll_by_dataset, count_by_dataset


def load_lr_tuning_records(train_data: str, val_set_size: int, split_seed: int) -> List[dict]:
    from datasets import load_dataset  # noqa: PLC0415

    if val_set_size <= 0:
        records = load_json(train_data)
        return records

    if train_data.endswith(".json"):
        data = load_dataset("json", data_files=train_data)
    else:
        data = load_dataset(train_data)
    train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=split_seed)
    return [dict(record) for record in train_val["test"]]


def llama_module_shapes(config, target_modules: Iterable[str]) -> List[Tuple[int, int]]:
    num_layers = int(config.num_hidden_layers)
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(getattr(config, "num_key_value_heads", num_heads))
    head_dim = int(getattr(config, "head_dim", hidden // num_heads))
    kv_out = num_kv_heads * head_dim

    per_layer_shapes = {
        "q_proj": (hidden, hidden),
        "k_proj": (kv_out, hidden),
        "v_proj": (kv_out, hidden),
        "o_proj": (hidden, hidden),
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    shapes = []
    for module in target_modules:
        if module not in per_layer_shapes:
            raise ValueError(f"Unknown Llama target module for budget computation: {module}")
        shapes.extend([per_layer_shapes[module]] * num_layers)
    return shapes


def rank_equivalent_sparse_rate(
    model_name: str,
    target_modules: Iterable[str],
    lora_r: int,
    override: Optional[float] = None,
) -> float:
    if override is not None:
        return override
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    shapes = llama_module_shapes(config, target_modules)
    total_lora_params = sum(lora_r * (out_features + in_features) for out_features, in_features in shapes)
    total_dense_params = sum(out_features * in_features for out_features, in_features in shapes)
    return total_lora_params / total_dense_params


def lora_param_count(shapes: Iterable[Tuple[int, int]], lora_r: int) -> int:
    return sum(lora_r * (out_features + in_features) for out_features, in_features in shapes)


def sparse_param_count(shapes: Iterable[Tuple[int, int]], sparse_rate: float, add_one: bool) -> int:
    total = 0
    for out_features, in_features in shapes:
        numel = out_features * in_features
        count = int(sparse_rate * numel)
        if add_one:
            count += 1
        total += min(count, numel)
    return total


def supra_param_count(
    shapes: Iterable[Tuple[int, int]],
    sparse_rate: float,
    lora_params_ratio: float,
) -> int:
    total = 0
    for out_features, in_features in shapes:
        numel = out_features * in_features
        target_params = int(sparse_rate * numel)
        lora_rank = math.ceil(lora_params_ratio * sparse_rate * numel / (out_features + in_features))
        lora_params = (out_features + in_features) * lora_rank
        sparse_params = target_params - lora_params
        if sparse_params < 0:
            raise ValueError(
                "Supra budget split produced a negative sparse budget. "
                f"shape=({out_features}, {in_features}), sparse_rate={sparse_rate}, "
                f"lora_params_ratio={lora_params_ratio}, lora_rank={lora_rank}"
            )
        total += lora_params + sparse_params
    return total


def parse_method(method: str) -> Tuple[str, bool, float]:
    if method == "base":
        return "base", False, 0.0
    if method == "lora":
        return "lora", False, 0.0
    if method == "rosa":
        return "rosa", False, 0.0
    if method.startswith("sift"):
        return "sift", "rand" in method, 0.0
    if method.startswith("super"):
        return "super", "rand" in method, 0.0
    if method.startswith("supra"):
        pieces = method.split("-", 1)
        if len(pieces) != 2:
            raise ValueError("Supra method names must look like 'supra-0.3'.")
        return "supra", "rand" in method, float(pieces[1].replace("rand-", ""))
    raise ValueError(f"Unknown method: {method}")


def build_budget_plan(args, spec: RunSpec, target_modules: List[str]) -> dict:
    config = AutoConfig.from_pretrained(spec.model, trust_remote_code=True)
    shapes = llama_module_shapes(config, target_modules)
    target_dense_params = sum(out_features * in_features for out_features, in_features in shapes)
    reference_lora_params = lora_param_count(shapes, spec.lora_r)
    total_sparse_rate = args.sparse_rate_override
    if total_sparse_rate is None:
        total_sparse_rate = reference_lora_params / target_dense_params

    adapter_name, _, lora_params_ratio = parse_method(spec.method)
    train_sparse_rate = total_sparse_rate
    train_lora_r = spec.lora_r
    component_lora_ratio = None

    if adapter_name == "base":
        train_sparse_rate = 0.0
        train_lora_r = 0
        adapter_params = 0
    elif adapter_name == "lora":
        train_sparse_rate = 0.0
        adapter_params = reference_lora_params
    elif adapter_name in {"super", "sift"}:
        adapter_params = sparse_param_count(shapes, total_sparse_rate, add_one=True)
    elif adapter_name == "supra":
        component_lora_ratio = lora_params_ratio
        adapter_params = supra_param_count(shapes, total_sparse_rate, lora_params_ratio)
    elif adapter_name == "rosa":
        component_lora_ratio = args.rosa_lora_budget_ratio
        train_lora_r = int(round(spec.lora_r * component_lora_ratio))
        train_sparse_rate = total_sparse_rate * (1.0 - component_lora_ratio)
        adapter_params = lora_param_count(shapes, train_lora_r) + sparse_param_count(
            shapes, train_sparse_rate, add_one=False
        )
    else:
        raise ValueError(f"Unsupported adapter for budget planning: {adapter_name}")

    if adapter_name == "base":
        budget_error_pct = 0.0
    elif reference_lora_params == 0:
        budget_error_pct = 0.0
    else:
        budget_error_pct = 100.0 * (adapter_params - reference_lora_params) / reference_lora_params
    if adapter_name != "base" and abs(budget_error_pct) > args.budget_tolerance_pct:
        raise ValueError(
            f"{spec.method} budget differs from the rank-{spec.lora_r} LoRA reference by "
            f"{budget_error_pct:.2f}% ({adapter_params} vs {reference_lora_params} params). "
            "Adjust the budget split or increase --budget_tolerance_pct if this is intentional."
        )

    return {
        "budget_lora_r": spec.lora_r,
        "total_sparse_rate": total_sparse_rate,
        "train_sparse_rate": train_sparse_rate,
        "train_lora_r": train_lora_r,
        "component_lora_ratio": component_lora_ratio,
        "target_dense_params": target_dense_params,
        "reference_lora_params": reference_lora_params,
        "adapter_trainable_params_estimate": adapter_params,
        "adapter_budget_error_pct": budget_error_pct,
        "is_baseline": adapter_name == "base",
    }


def import_rosa_train():
    rosa_dir = os.path.join(SCRIPT_DIR, "rosa")
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    if rosa_dir not in sys.path:
        sys.path.insert(1, rosa_dir)
    from rosa.finetune_rosa import train as train_rosa  # noqa: PLC0415

    return train_rosa


def collect_trainable_param_report(model, budget_plan: dict) -> dict:
    requires_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_params = getattr(model, "optimizer_trainable_params", None)
    if optimizer_params is None:
        optimizer_params = requires_grad_params
    reference_params = budget_plan["reference_lora_params"]
    if budget_plan.get("is_baseline"):
        trainable_budget_error_pct = 0.0
    elif reference_params:
        trainable_budget_error_pct = 100.0 * (optimizer_params - reference_params) / reference_params
    else:
        trainable_budget_error_pct = 0.0

    return {
        "trainable_params": int(optimizer_params),
        "requires_grad_params": int(requires_grad_params),
        "budget_trainable_params": int(budget_plan["adapter_trainable_params_estimate"]),
        "trainable_budget_error_pct": float(trainable_budget_error_pct),
    }


def train_one_run(args, spec: RunSpec, budget_plan: dict, target_modules: List[str]):
    adapter_name, random_indices, lora_params_ratio = parse_method(spec.method)
    output_dir = os.path.join(args.checkpoint_dir, spec.run_id)
    calibration_data = args.train_data if args.calibration_data == "same_as_train" else args.calibration_data
    if adapter_name == "rosa":
        lora_params_ratio = args.rosa_lora_budget_ratio

    if adapter_name == "base":
        tokenizer = AutoTokenizer.from_pretrained(spec.model, trust_remote_code=True)
        tokenizer.pad_token_id = 0
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            spec.model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map={"": int(os.environ.get("LOCAL_RANK", 0))},
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.optimizer_trainable_params = 0
        model.requires_grad_trainable_params = 0
        model.config.use_cache = False
        model.eval()
        return model, tokenizer, output_dir

    common_kwargs = dict(
        base_model=spec.model,
        data_path=args.train_data,
        target_modules=target_modules,
        eval_step=args.eval_step,
        save_step=args.save_step,
        val_split_seed=args.val_split_seed,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        sparse_rate=budget_plan["train_sparse_rate"],
        num_epochs=args.num_epochs,
        learning_rate=spec.lr,
        cutoff_len=args.cutoff_len,
        output_dir=output_dir,
        val_set_size=args.val_set_size,
        compile=args.compile,
        seed=spec.seed,
        lora_r=budget_plan["train_lora_r"],
        lora_params_ratio=lora_params_ratio,
        adapter_name=adapter_name,
        random_indices=random_indices,
        max_steps=args.max_steps,
        optimizer_name=args.optimizer_name,
    )

    if adapter_name != "rosa":
        common_kwargs.update(
            calibration_data=calibration_data,
            calibration_nsamples=args.calibration_nsamples,
            calibration_seed=args.calibration_seed,
        )

    if args.dry_run:
        print("DRY RUN:", json.dumps({**common_kwargs, "target_modules": target_modules}, indent=2, default=str))
        return None, None, output_dir

    if adapter_name == "rosa":
        train_rosa = import_rosa_train()
        model, tokenizer = train_rosa(**common_kwargs)
    else:
        model, tokenizer = train(**common_kwargs)
    return model, tokenizer, output_dir


def load_existing_results(path: str) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    if not os.path.exists(path):
        return results
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            results[row["run_id"]] = row
    return results


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def save_pickle(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def wandb_result_columns(datasets: List[str]) -> List[str]:
    columns = [
        "completed_run",
        "run_id",
        "model",
        "method",
        "budget_lora_r",
        "train_lora_r",
        "lr",
        "seed",
        "trainable_params",
        "requires_grad_params",
        "reference_lora_params",
        "trainable_budget_error_pct",
        "lr_tuning_ppl",
        "lr_tuning_nll",
        "average_accuracy",
        "average_ppl",
    ]
    columns.extend([f"accuracy_{dataset}" for dataset in datasets])
    columns.extend([f"ppl_{dataset}" for dataset in datasets])
    return columns


def make_wandb_run_name(args) -> str:
    if args.wandb_run_name:
        return args.wandb_run_name
    model_name = parse_csv_list(args.models, str)[0].split("/")[-1].replace(".", "_")
    methods = args.methods.replace(",", "_")
    return f"math_{model_name}_{methods}_r{args.lora_rs}"


def init_wandb(args, datasets: List[str], target_modules: List[str], total_specs: int, pending_specs: int):
    if not args.wandb_project:
        return None
    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging requested, but the `wandb` package is not installed. "
            "Install it or remove --wandb_project."
        ) from exc

    tags = parse_csv_list(args.wandb_tags, str) if args.wandb_tags else []
    config = vars(args).copy()
    config.update(
        total_specs=total_specs,
        pending_specs=pending_specs,
        datasets=datasets,
        target_modules=target_modules,
    )
    init_kwargs = dict(
        project=args.wandb_project,
        name=make_wandb_run_name(args),
        group=args.wandb_group or None,
        tags=tags,
        notes=args.wandb_notes or None,
        job_type="math_adapter_eval",
        config=config,
        mode=args.wandb_mode,
    )
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    run = wandb.init(**init_kwargs)
    define_metric = getattr(run, "define_metric", wandb.define_metric)
    define_metric("progress/completed_adapters")
    define_metric("progress/evaluated_fraction")
    define_metric("accuracy/*", step_metric="progress/completed_adapters")
    define_metric("ppl/*", step_metric="progress/completed_adapters")
    define_metric("lr_tuning/*", step_metric="progress/completed_adapters")
    define_metric("budget/*", step_metric="progress/completed_adapters")
    return {
        "run": run,
        "table": wandb.Table(columns=wandb_result_columns(datasets)),
        "datasets": datasets,
        "pending_specs": pending_specs,
        "started": 0,
    }


def log_wandb_run_started(wandb_state, spec: RunSpec, budget_plan: dict) -> None:
    if wandb_state is None:
        return
    wandb_state["started"] += 1
    pending_specs = wandb_state["pending_specs"]
    run = wandb_state["run"]
    run.log(
        {
            "progress/started_adapters": wandb_state["started"],
            "progress/pending_adapters": pending_specs,
            "progress/current_lr": spec.lr,
            "progress/current_budget_lora_r": spec.lora_r,
            "budget/reference_lora_params": budget_plan["reference_lora_params"],
            "budget/estimated_trainable_params": budget_plan["adapter_trainable_params_estimate"],
            "budget/estimated_error_pct": budget_plan["adapter_budget_error_pct"],
        }
    )


def log_wandb_result(wandb_state, row: dict, completed_runs: int) -> None:
    if wandb_state is None:
        return
    run = wandb_state["run"]
    datasets = wandb_state["datasets"]
    pending_specs = wandb_state["pending_specs"]
    accuracy = row.get("accuracy", {})
    ppl = row.get("ppl", {})
    lr_tuning = row.get("lr_tuning", {})

    table_values = [
        completed_runs,
        row.get("run_id"),
        row.get("model"),
        row.get("method"),
        row.get("budget_lora_r"),
        row.get("train_lora_r"),
        row.get("lr"),
        row.get("seed"),
        row.get("trainable_params"),
        row.get("requires_grad_params"),
        row.get("reference_lora_params"),
        row.get("trainable_budget_error_pct"),
        lr_tuning.get("ppl"),
        lr_tuning.get("nll"),
        accuracy.get("Average"),
        ppl.get("Average"),
    ]
    table_values.extend([accuracy.get(dataset) for dataset in datasets])
    table_values.extend([ppl.get(dataset) for dataset in datasets])
    wandb_state["table"].add_data(*table_values)

    metrics = {
        "progress/completed_adapters": completed_runs,
        "progress/pending_adapters": pending_specs,
        "progress/evaluated_fraction": completed_runs / pending_specs if pending_specs else 1.0,
        "hparams/lr": row.get("lr"),
        "hparams/budget_lora_r": row.get("budget_lora_r"),
        "budget/trainable_params": row.get("trainable_params"),
        "budget/requires_grad_params": row.get("requires_grad_params"),
        "budget/reference_lora_params": row.get("reference_lora_params"),
        "budget/trainable_error_pct": row.get("trainable_budget_error_pct"),
        "lr_tuning/ppl": lr_tuning.get("ppl"),
        "lr_tuning/nll": lr_tuning.get("nll"),
        "accuracy/Average": accuracy.get("Average"),
        "ppl/Average": ppl.get("Average"),
        "results/table": wandb_state["table"],
    }
    for dataset in datasets:
        metrics[f"accuracy/{dataset}"] = accuracy.get(dataset)
        metrics[f"ppl/{dataset}"] = ppl.get(dataset)
    metrics = {key: finite_or_none(value) for key, value in metrics.items()}
    run.log(metrics)


def log_wandb_failure(wandb_state, spec: RunSpec, exc: Exception, failure_count: int) -> None:
    if wandb_state is None:
        return
    wandb_state["run"].log(
        {
            "progress/failed_adapters": failure_count,
            "failure/lr": spec.lr,
            "failure/budget_lora_r": spec.lora_r,
            "failure/message": str(exc),
        }
    )


def finish_wandb(wandb_state) -> None:
    if wandb_state is not None:
        wandb_state["run"].finish()


def build_tables(results: Iterable[dict], datasets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index_cols = ["seed", "model", "lora_r", "lr", "method"]
    accuracy_rows = []
    ppl_rows = []
    summary_rows = []

    for row in results:
        index_values = {key: row[key] for key in index_cols}
        trainable_params = first_present(
            row,
            ["trainable_params", "adapter_trainable_params_estimate", "budget_trainable_params"],
        )

        accuracy_row = dict(index_values)
        accuracy_row["trainable_params"] = trainable_params
        for dataset in datasets:
            accuracy_row[dataset] = row.get("accuracy", {}).get(dataset)
        accuracy_row["Average"] = row.get("accuracy", {}).get("Average")
        accuracy_rows.append(accuracy_row)

        ppl_row = dict(index_values)
        ppl_row["trainable_params"] = trainable_params
        for dataset in datasets:
            ppl_row[dataset] = row.get("ppl", {}).get(dataset)
        ppl_row["Average"] = row.get("ppl", {}).get("Average")
        ppl_rows.append(ppl_row)

        summary_row = dict(index_values)
        summary_row.update(
            sparse_rate=row.get("sparse_rate"),
            total_sparse_rate=row.get("total_sparse_rate"),
            train_sparse_rate=row.get("train_sparse_rate"),
            train_lora_r=row.get("train_lora_r"),
            budget_lora_r=row.get("budget_lora_r"),
            component_lora_ratio=row.get("component_lora_ratio"),
            target_dense_params=row.get("target_dense_params"),
            reference_lora_params=row.get("reference_lora_params"),
            adapter_trainable_params_estimate=row.get("adapter_trainable_params_estimate"),
            adapter_budget_error_pct=row.get("adapter_budget_error_pct"),
            trainable_params=row.get("trainable_params"),
            requires_grad_params=row.get("requires_grad_params"),
            budget_trainable_params=row.get("budget_trainable_params"),
            trainable_budget_error_pct=row.get("trainable_budget_error_pct"),
            checkpoint_dir=row.get("checkpoint_dir"),
            target_modules=",".join(row.get("target_modules", [])),
            train_data=row.get("train_data"),
            calibration_data=row.get("calibration_data"),
            calibration_nsamples=row.get("calibration_nsamples"),
            calibration_seed=row.get("calibration_seed"),
            val_split_seed=row.get("val_split_seed"),
            lr_tuning_ppl=row.get("lr_tuning", {}).get("ppl"),
            lr_tuning_nll=row.get("lr_tuning", {}).get("nll"),
            lr_tuning_examples=row.get("lr_tuning", {}).get("examples"),
            average_accuracy=row.get("accuracy", {}).get("Average"),
            average_ppl=row.get("ppl", {}).get("Average"),
        )
        summary_rows.append(summary_row)

    accuracy_df = pd.DataFrame(accuracy_rows)
    ppl_df = pd.DataFrame(ppl_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not accuracy_df.empty:
        accuracy_df = accuracy_df.set_index(index_cols).sort_index()
    if not ppl_df.empty:
        ppl_df = ppl_df.set_index(index_cols).sort_index()
    if not summary_df.empty:
        summary_df = summary_df.set_index(index_cols).sort_index()
    return accuracy_df, ppl_df, summary_df


def format_mean_std(mean_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    formatted = mean_df.copy()
    for row_idx in mean_df.index:
        for col in mean_df.columns:
            mean_value = mean_df.loc[row_idx, col]
            std_value = std_df.loc[row_idx, col] if col in std_df.columns and row_idx in std_df.index else np.nan
            if pd.isna(mean_value):
                formatted.loc[row_idx, col] = ""
            elif col.endswith("params"):
                formatted.loc[row_idx, col] = f"{int(round(mean_value))}"
            elif pd.isna(std_value):
                formatted.loc[row_idx, col] = f"{mean_value:.2f}"
            else:
                formatted.loc[row_idx, col] = f"{mean_value:.2f} $\\pm$ {std_value:.2f}"
    return formatted


def save_selected_lr_tables(out_dir: str, results: Iterable[dict], datasets: List[str]) -> None:
    rows = [
        row
        for row in results
        if row.get("lr_tuning", {}).get("nll") is not None
        and np.isfinite(row["lr_tuning"]["nll"])
    ]
    if not rows:
        return

    df_rows = []
    for row in rows:
        trainable_params = first_present(
            row,
            ["trainable_params", "adapter_trainable_params_estimate", "budget_trainable_params"],
        )
        df_rows.append(
            {
                "model": row["model"],
                "lora_r": row["lora_r"],
                "method": row["method"],
                "lr": row["lr"],
                "seed": row["seed"],
                "lr_tuning_nll": row["lr_tuning"]["nll"],
                "lr_tuning_ppl": row["lr_tuning"]["ppl"],
                "train_lora_r": row.get("train_lora_r"),
                "train_sparse_rate": row.get("train_sparse_rate"),
                "trainable_params": trainable_params,
                "adapter_trainable_params_estimate": row.get("adapter_trainable_params_estimate"),
                "adapter_budget_error_pct": row.get("adapter_budget_error_pct"),
                "trainable_budget_error_pct": row.get("trainable_budget_error_pct"),
            }
        )
    tune_df = pd.DataFrame(df_rows)
    grouped = (
        tune_df.groupby(["model", "lora_r", "method", "lr"], dropna=False)
        .agg(
            selection_nll=("lr_tuning_nll", "mean"),
            selection_ppl=("lr_tuning_ppl", "mean"),
            available_seeds=("seed", "nunique"),
            train_lora_r=("train_lora_r", "first"),
            train_sparse_rate=("train_sparse_rate", "first"),
            trainable_params=("trainable_params", "first"),
            adapter_trainable_params_estimate=("adapter_trainable_params_estimate", "first"),
            adapter_budget_error_pct=("adapter_budget_error_pct", "first"),
            trainable_budget_error_pct=("trainable_budget_error_pct", "first"),
        )
        .reset_index()
    )
    idx = grouped.groupby(["model", "lora_r", "method"])["selection_nll"].idxmin()
    selected_lr_df = grouped.loc[idx].sort_values(["model", "lora_r", "method"]).rename(columns={"lr": "selected_lr"})
    selected_lr_df.to_csv(os.path.join(out_dir, "selected_lr_by_method.csv"), index=False)
    with open(os.path.join(out_dir, "selected_lr_by_method.tex"), "w") as f:
        f.write(selected_lr_df.to_latex(index=False, float_format=lambda value: f"{value:.4g}"))

    selected_lookup = {
        (row.model, row.lora_r, row.method): row.selected_lr
        for row in selected_lr_df.itertuples(index=False)
    }

    selected_results = []
    for row in rows:
        key = (row["model"], row["lora_r"], row["method"])
        if row["lr"] == selected_lookup.get(key):
            selected_results.append(row)

    selected_by_id = {
        f'{row["model"]}|{row["lora_r"]}|{row["method"]}|{row["seed"]}': row
        for row in selected_results
    }
    with open(os.path.join(out_dir, "selected_run_results.jsonl"), "w") as f:
        for row in selected_by_id.values():
            f.write(json.dumps(row, sort_keys=True) + "\n")

    def aggregate(metric_name: str, output_prefix: str) -> None:
        metric_rows = []
        for row in selected_by_id.values():
            metric = row.get(metric_name, {})
            selected_lr = selected_lookup[(row["model"], row["lora_r"], row["method"])]
            trainable_params = first_present(
                row,
                ["trainable_params", "adapter_trainable_params_estimate", "budget_trainable_params"],
            )
            metric_row = {
                "model": row["model"],
                "lora_r": row["lora_r"],
                "method": row["method"],
                "selected_lr": selected_lr,
                "seed": row["seed"],
                "trainable_params": trainable_params,
            }
            for dataset in datasets + ["Average"]:
                metric_row[dataset] = metric.get(dataset)
            metric_rows.append(metric_row)

        if not metric_rows:
            return

        metric_df = pd.DataFrame(metric_rows)
        index_cols = ["model", "lora_r", "method", "selected_lr"]
        mean_df = metric_df.groupby(index_cols)[datasets + ["Average"]].mean().sort_index()
        std_df = metric_df.groupby(index_cols)[datasets + ["Average"]].std().sort_index()
        trainable_params = metric_df.groupby(index_cols)["trainable_params"].first().sort_index()
        mean_df.insert(0, "trainable_params", trainable_params)
        std_df.insert(0, "trainable_params", np.nan)
        formatted_df = format_mean_std(mean_df, std_df)

        mean_df.to_csv(os.path.join(out_dir, f"{output_prefix}_mean.csv"))
        std_df.to_csv(os.path.join(out_dir, f"{output_prefix}_std.csv"))
        formatted_df.to_csv(os.path.join(out_dir, f"{output_prefix}_mean_std.csv"))
        with open(os.path.join(out_dir, f"{output_prefix}_mean_std.tex"), "w") as f:
            f.write(formatted_df.to_latex(escape=False))

    aggregate("accuracy", "selected_accuracy")
    aggregate("ppl", "selected_ppl")


def save_tables(out_dir: str, results: Iterable[dict], datasets: List[str]) -> None:
    accuracy_df, ppl_df, summary_df = build_tables(results, datasets)
    os.makedirs(out_dir, exist_ok=True)

    for name, df in [
        ("accuracy_table", accuracy_df),
        ("ppl_table", ppl_df),
        ("summary_table", summary_df),
    ]:
        df.to_csv(os.path.join(out_dir, f"{name}.csv"))
        save_pickle(df, os.path.join(out_dir, f"{name}.pkl"))
        with open(os.path.join(out_dir, f"{name}.tex"), "w") as f:
            f.write(df.to_latex(float_format=lambda value: f"{value:.2f}" if pd.notna(value) else ""))
    save_selected_lr_tables(out_dir, results, datasets)


def iter_run_specs(args) -> Iterable[RunSpec]:
    all_specs = []
    for seed in parse_csv_list(args.seeds, int):
        for model in parse_csv_list(args.models, str):
            for lora_r in parse_csv_list(args.lora_rs, int):
                for method in parse_csv_list(args.methods, str):
                    lrs = [0.0] if method == "base" else parse_csv_list(args.lrs, float)
                    for lr in lrs:
                        all_specs.append(RunSpec(seed=seed, model=model, lora_r=lora_r, lr=lr, method=method))

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")

    for idx, spec in enumerate(all_specs):
        if idx % args.num_shards == args.shard_id:
            yield spec


def run(args) -> None:
    print_environment()
    if not 0.0 <= args.rosa_lora_budget_ratio <= 1.0:
        raise ValueError("--rosa_lora_budget_ratio must be in [0, 1].")
    if args.budget_tolerance_pct < 0.0:
        raise ValueError("--budget_tolerance_pct must be nonnegative.")

    target_modules = LEGACY_TARGET_MODULES if args.legacy_target_modules else parse_csv_list(args.target_modules, str)
    datasets = parse_csv_list(args.datasets, str)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "run_results.jsonl")
    failures_path = os.path.join(args.out_dir, "failed_runs.jsonl")
    existing = load_existing_results(results_path)
    completed_results = list(existing.values())

    print("Training data:", args.train_data)
    print("Wanda calibration data:", args.train_data if args.calibration_data == "same_as_train" else args.calibration_data)
    print("Wanda calibration samples:", args.calibration_nsamples)
    print("Wanda calibration seed:", args.calibration_seed)
    print("Evaluation datasets:", datasets)
    print("Target modules:", target_modules)
    print("Validation split seed:", args.val_split_seed)
    print("Output directory:", args.out_dir)

    specs = list(iter_run_specs(args))
    pending_specs = [
        spec
        for spec in specs
        if not (args.only_missing and spec.run_id in existing)
    ]
    if args.max_runs is not None:
        pending_specs = pending_specs[:args.max_runs]
    wandb_state = init_wandb(
        args=args,
        datasets=datasets,
        target_modules=target_modules,
        total_specs=len(specs),
        pending_specs=len(pending_specs),
    )

    lr_tuning_records = None
    if not args.skip_lr_tuning_metric:
        lr_tuning_records = load_lr_tuning_records(
            train_data=args.train_data,
            val_set_size=args.val_set_size,
            split_seed=args.val_split_seed,
        )
        print("LR tuning metric: validation perplexity on", len(lr_tuning_records), "held-out Math10K examples")

    run_count = 0
    failure_count = 0
    try:
        for spec in specs:
            if args.only_missing and spec.run_id in existing:
                print("Already computed:", spec.run_id)
                continue
            if args.max_runs is not None and run_count >= args.max_runs:
                break

            budget_plan = build_budget_plan(args, spec, target_modules)
            print("=" * 100)
            print("Run:", asdict(spec))
            print("rank-equivalent global sparse_rate:", budget_plan["total_sparse_rate"])
            print(
                "adapter budget estimate:",
                budget_plan["adapter_trainable_params_estimate"],
                "reference LoRA params:",
                budget_plan["reference_lora_params"],
                f"error={budget_plan['adapter_budget_error_pct']:.4f}%",
            )
            print("train_lora_r:", budget_plan["train_lora_r"])
            print("train_sparse_rate:", budget_plan["train_sparse_rate"])
            log_wandb_run_started(wandb_state, spec, budget_plan)
            set_seed(spec.seed)

            model = tokenizer = None
            try:
                model, tokenizer, checkpoint_dir = train_one_run(args, spec, budget_plan, target_modules)
                if args.dry_run:
                    run_count += 1
                    continue
                trainable_param_report = collect_trainable_param_report(model, budget_plan)
                print("actual optimizer trainable params:", trainable_param_report["trainable_params"])
                print("requires-grad params:", trainable_param_report["requires_grad_params"])
                print(
                    "actual trainable budget error:",
                    f"{trainable_param_report['trainable_budget_error_pct']:.4f}%",
                )
                if abs(trainable_param_report["trainable_budget_error_pct"]) > args.budget_tolerance_pct:
                    raise ValueError(
                        f"Actual optimizer trainable params differ from the rank-{spec.lora_r} LoRA reference by "
                        f"{trainable_param_report['trainable_budget_error_pct']:.2f}% "
                        f"({trainable_param_report['trainable_params']} vs {budget_plan['reference_lora_params']} params)."
                    )

                lr_tuning = {}
                if lr_tuning_records is not None:
                    tune_ppl, tune_nll, tune_count = evaluate_perplexity_on_records(
                        model=model,
                        tokenizer=tokenizer,
                        records=lr_tuning_records,
                        max_length=args.ppl_max_length,
                        max_examples=args.lr_tuning_max_examples,
                    )
                    lr_tuning = {"ppl": tune_ppl, "nll": tune_nll, "examples": tune_count}
                    print(f"LR tuning validation ppl: {tune_ppl:.4f} (nll={tune_nll:.4f}, examples={tune_count})")

                accuracy: Dict[str, float] = {}
                for dataset in datasets:
                    score = eval_model(dataset_name=dataset, model=model, tokenizer=tokenizer) * 100.0
                    accuracy[dataset] = score
                    print(f"{dataset} accuracy: {score:.4f}")
                accuracy["Average"] = float(np.mean([accuracy[dataset] for dataset in datasets]))

                ppl, nll, ppl_examples = evaluate_perplexity(
                    model=model,
                    tokenizer=tokenizer,
                    datasets=datasets,
                    max_length=args.ppl_max_length,
                    max_examples=args.ppl_max_examples,
                )

                row = {
                    "run_id": spec.run_id,
                    **asdict(spec),
                    "sparse_rate": budget_plan["total_sparse_rate"],
                    **budget_plan,
                    **trainable_param_report,
                    "target_modules": target_modules,
                    "train_data": args.train_data,
                    "calibration_data": args.train_data if args.calibration_data == "same_as_train" else args.calibration_data,
                    "calibration_nsamples": args.calibration_nsamples,
                    "calibration_seed": args.calibration_seed,
                    "val_split_seed": args.val_split_seed,
                    "lr_tuning": lr_tuning,
                    "checkpoint_dir": checkpoint_dir,
                    "accuracy": accuracy,
                    "ppl": ppl,
                    "nll": nll,
                    "ppl_examples": ppl_examples,
                }
                append_jsonl(results_path, row)
                existing[spec.run_id] = row
                completed_results.append(row)
                save_tables(args.out_dir, completed_results, datasets)
                run_count += 1
                log_wandb_result(wandb_state, row, run_count)
            except Exception as exc:
                failure_count += 1
                traceback_text = traceback.format_exc()
                failure_row = {
                    "run_id": spec.run_id,
                    **asdict(spec),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback_text,
                    "target_modules": target_modules,
                    **budget_plan,
                }
                append_jsonl(failures_path, failure_row)
                log_wandb_failure(wandb_state, spec, exc, failure_count)
                if not args.continue_on_error:
                    raise
                print(traceback_text)
                print(f"Run failed and will be skipped: {spec.run_id} ({type(exc).__name__}: {exc})")
            finally:
                del model
                del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        save_tables(args.out_dir, existing.values(), datasets)
        print("Finished. Tables written to:", args.out_dir)
    finally:
        finish_wandb(wandb_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PEFT methods on Math10K and construct math accuracy/perplexity tables."
    )
    parser.add_argument("--models", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--lora_rs", default="8")
    parser.add_argument("--lrs", default=DEFAULT_LRS)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--datasets", default=",".join(MATH_BENCHMARKS))
    parser.add_argument("--target_modules", default=",".join(FULL_LLAMA_TARGET_MODULES))
    parser.add_argument("--legacy_target_modules", action="store_true")
    parser.add_argument("--train_data", default="ft-training_set/math_10k.json")
    parser.add_argument("--calibration_data", default="same_as_train")
    parser.add_argument("--calibration_nsamples", type=int, default=128)
    parser.add_argument("--calibration_seed", type=int, default=228)
    parser.add_argument("--out_dir", default="out_math_experiments")
    parser.add_argument("--checkpoint_dir", default="checkpoints_math")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--val_set_size", type=int, default=120)
    parser.add_argument("--eval_step", type=int, default=50)
    parser.add_argument("--save_step", type=int, default=50)
    parser.add_argument("--compile", type=int, default=0)
    parser.add_argument("--optimizer_name", default="adam")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--ppl_max_length", type=int, default=256)
    parser.add_argument("--ppl_max_examples", type=int, default=None)
    parser.add_argument("--lr_tuning_max_examples", type=int, default=None)
    parser.add_argument("--val_split_seed", "--lr_tuning_split_seed", dest="val_split_seed", type=int, default=42)
    parser.add_argument("--skip_lr_tuning_metric", action="store_true")
    parser.add_argument("--sparse_rate_override", type=float, default=None)
    parser.add_argument("--rosa_lora_budget_ratio", type=float, default=0.5)
    parser.add_argument("--budget_tolerance_pct", type=float, default=3.0)
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", ""))
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_group", default="math-experiments")
    parser.add_argument("--wandb_run_name", default="")
    parser.add_argument("--wandb_tags", default="")
    parser.add_argument("--wandb_notes", default="")
    parser.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--only_missing", action="store_true", default=True)
    parser.add_argument("--rerun_existing", dest="only_missing", action="store_false")
    parser.add_argument("--continue_on_error", action="store_true", default=True)
    parser.add_argument("--stop_on_error", dest="continue_on_error", action="store_false")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
