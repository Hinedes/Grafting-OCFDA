import argparse
import gc
import json
import math
import os
import pickle
import random
import re
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(1, REPO_DIR)

try:
    from .evaluate import eval_model  # noqa: E402
    from .finetune import train  # noqa: E402
except ImportError:
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
DEFAULT_METHODS = "supra-0.8-bottom"
DEFAULT_LRS = "5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1"
DEFAULT_TRAIN_DATA = os.path.join(SCRIPT_DIR, "ft-training_set", "math_17k.json")


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


def resolve_dataset_path(dataset_name: str, dataset_dir: Optional[str] = None) -> str:
    return os.path.join(dataset_dir or os.path.join(SCRIPT_DIR, "dataset"), dataset_name, "test.json")


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


def get_gold_answer(record: dict) -> str:
    answer = record.get("answer")
    if answer is not None:
        return str(answer).strip()
    return get_gold_output(record).strip()


def answer_text_variants(answer: str) -> List[str]:
    answer = str(answer).strip()
    variants = {answer}
    try:
        numeric_answer = float(answer.replace(",", ""))
        if math.isfinite(numeric_answer):
            if numeric_answer.is_integer():
                integer_answer = int(numeric_answer)
                variants.update(
                    {
                        str(integer_answer),
                        f"{integer_answer:,}",
                        f"{float(integer_answer):.1f}",
                        f"{float(integer_answer):,.1f}",
                    }
                )
            else:
                compact_answer = ("%f" % numeric_answer).rstrip("0").rstrip(".")
                variants.update({str(numeric_answer), f"{numeric_answer:,}", compact_answer})
                if compact_answer:
                    variants.add(f"{float(compact_answer):,}")
    except ValueError:
        pass
    return sorted((variant for variant in variants if variant), key=lambda item: (len(item), item), reverse=True)


def find_answer_span_in_output(output: str, answer: str) -> Optional[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    for variant in answer_text_variants(answer):
        if re.fullmatch(r"[A-Ea-e]", variant):
            patterns = [
                rf"\b{re.escape(variant)}\b(?=\s*\)|[\s.,:;!?]|$)",
                rf"\b{re.escape(variant)}\)",
            ]
            flags = re.IGNORECASE
        else:
            patterns = [rf"(?<!\d){re.escape(variant)}(?!\d)"]
            flags = 0
        for pattern in patterns:
            for match in re.finditer(pattern, output, flags):
                variant_match = re.search(re.escape(variant), match.group(0), flags)
                if variant_match is None:
                    candidates.append(match.span())
                else:
                    start = match.start() + variant_match.start()
                    candidates.append((start, start + len(variant)))
    if not candidates:
        return None
    return max(candidates, key=lambda span: (span[1], span[0]))


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
    target_mode: str = "gold_output",
    eval_batch_size: int = 1,
) -> Tuple[float, float, int]:
    model.eval()
    device = get_model_input_device(model)
    eos = tokenizer.eos_token or ""
    total_nll = 0.0
    total_tokens = 0
    used_examples = 0
    batch_features = []

    if max_examples is not None:
        records = records[:max_examples]

    for record in records:
        prompt = generate_prompt(record.get("instruction", ""), record.get("input"))
        if target_mode == "gold_output":
            target = get_gold_output(record)
            full_text = prompt + target + eos
            label_char_span = None
        elif target_mode == "answer":
            target = get_gold_answer(record)
            output = get_gold_output(record)
            answer_span = find_answer_span_in_output(output, target)
            if answer_span is None:
                continue
            full_text = prompt + output + eos
            label_char_span = (len(prompt) + answer_span[0], len(prompt) + answer_span[1])
        elif target_mode == "direct_answer":
            target = get_gold_answer(record)
            full_text = prompt + target
            label_char_span = None
        else:
            raise ValueError(f"Unsupported perplexity target mode: {target_mode}")
        if not target:
            continue

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encode_kwargs = {}
        if label_char_span is not None:
            encode_kwargs["return_offsets_mapping"] = True
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False,
            **encode_kwargs,
        )
        offsets = encoded.pop("offset_mapping", None)
        input_ids = encoded["input_ids"]
        if label_char_span is None:
            labels = input_ids.clone()
            prompt_len = min(len(prompt_ids), labels.shape[1])
            labels[:, :prompt_len] = -100
        else:
            labels = torch.full_like(input_ids, -100)
            span_start, span_end = label_char_span
            token_offsets = offsets[0].tolist()
            max_observed_end = max((end for _, end in token_offsets), default=0)
            if max_observed_end < span_end:
                continue
            for token_idx, (token_start, token_end) in enumerate(token_offsets):
                if token_end <= token_start:
                    continue
                if token_end > span_start and token_start < span_end:
                    labels[0, token_idx] = input_ids[0, token_idx]
        token_count = int((labels != -100).sum().item())
        if token_count == 0:
            continue

        batch_features.append(
            {
                "input_ids": input_ids[0].tolist(),
                "attention_mask": encoded["attention_mask"][0].tolist(),
                "labels": labels[0].tolist(),
                "token_count": token_count,
            }
        )

        if len(batch_features) >= eval_batch_size:
            batch_nll, batch_tokens, batch_count = score_perplexity_batch(model, batch_features, tokenizer, device)
            total_nll += batch_nll
            total_tokens += batch_tokens
            used_examples += batch_count
            batch_features = []

    if batch_features:
        batch_nll, batch_tokens, batch_count = score_perplexity_batch(model, batch_features, tokenizer, device)
        total_nll += batch_nll
        total_tokens += batch_tokens
        used_examples += batch_count

    if total_tokens == 0:
        return float("nan"), float("nan"), used_examples

    mean_nll = total_nll / total_tokens
    return float(math.exp(min(mean_nll, 50.0))), mean_nll, used_examples


def score_perplexity_batch(model, features: List[dict], tokenizer, device: torch.device) -> Tuple[float, int, int]:
    max_len = max(len(feature["input_ids"]) for feature in features)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = []
    attention_mask = []
    labels = []
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(feature["attention_mask"] + [0] * pad_len)
        labels.append(feature["labels"] + [-100] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
    }
    label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    token_count = int((label_tensor != -100).sum().item())
    if token_count == 0:
        return 0.0, 0, 0

    with torch.no_grad():
        outputs = model(**batch, labels=label_tensor, use_cache=False)

    return float(outputs.loss.item()) * token_count, token_count, len(features)


def evaluate_perplexity(
    model,
    tokenizer,
    datasets: Iterable[str],
    max_length: int,
    max_examples: Optional[int],
    target_mode: str = "gold_output",
    eval_batch_size: int = 1,
    dataset_dir: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    ppl_by_dataset: Dict[str, float] = {}
    nll_by_dataset: Dict[str, float] = {}
    count_by_dataset: Dict[str, int] = {}

    total_nll_weighted = 0.0
    total_examples = 0

    for dataset in datasets:
        records = load_json(resolve_dataset_path(dataset, dataset_dir=dataset_dir))
        ppl, nll, count = evaluate_perplexity_on_records(
            model=model,
            tokenizer=tokenizer,
            records=records,
            max_length=max_length,
            max_examples=max_examples,
            target_mode=target_mode,
            eval_batch_size=eval_batch_size,
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


def evaluate_perplexity_on_data_file(
    model,
    tokenizer,
    data_path: str,
    data_name: str,
    max_length: int,
    max_examples: Optional[int],
    target_mode: str = "gold_output",
    eval_batch_size: int = 1,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    records = load_json(data_path)
    ppl, nll, count = evaluate_perplexity_on_records(
        model=model,
        tokenizer=tokenizer,
        records=records,
        max_length=max_length,
        max_examples=max_examples,
        target_mode=target_mode,
        eval_batch_size=eval_batch_size,
    )
    print(f"{data_name} perplexity: {ppl:.4f} (nll={nll:.4f}, examples={count})")
    return (
        {data_name: ppl, "Average": ppl},
        {data_name: nll, "Average": nll},
        {data_name: count},
    )


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


def parse_method(method: str) -> Tuple[str, str, float]:
    if method == "base":
        return "base", "none", 0.0
    if method in {"full", "full-ft", "fft"}:
        return "full", "none", 0.0
    if method == "lora":
        return "lora", "none", 0.0
    if method == "rosa":
        return "rosa", "none", 0.0
    if method.startswith("sift"):
        return "sift", "random" if "rand" in method else "super", 0.0
    if method in {"magnitude-topk", "magnitude-top", "magnitude", "super-magnitude", "mag-topk", "mag-top"}:
        return "super", "magnitude", 0.0
    if method in {
        "magnitude-bottomk",
        "magnitude-bottom",
        "super-magnitude-bottom",
        "super-magnitude-bottomk",
        "mag-bottomk",
        "mag-bottom",
    }:
        return "super", "magnitude-bottom", 0.0
    if method in {"super-delta-naive", "super-full-delta-naive", "super-fft-delta-naive", "super-ft-delta-naive"}:
        return "super", "full-delta-naive", 0.0
    if method in {
        "super-delta",
        "super-delta-wanda",
        "super-wanda-delta",
        "super-full-delta",
        "super-fft-delta",
        "super-ft-delta",
    }:
        return "super", "full-delta", 0.0
    if method.startswith("super-wanda-hybrid-") or method.startswith("super-hybrid-"):
        beta_text = (
            method.removeprefix("super-wanda-hybrid-")
            if method.startswith("super-wanda-hybrid-")
            else method.removeprefix("super-hybrid-")
        )
        beta = float(beta_text)
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Super Wanda hybrid beta must be in [0, 1].")
        return "super", f"super-hybrid-{beta:g}", 0.0
    if method in {"super-bottom-structured", "super-wanda-bottom-structured", "super-row-bottom"}:
        return "super", "super-bottom-structured", 0.0
    if method.startswith("super"):
        if "rand" in method:
            return "super", "random", 0.0
        if "bottom" in method:
            return "super", "super-bottom", 0.0
        return "super", "super", 0.0
    if method.startswith("supra-magnitude-"):
        ratio_text = method.removeprefix("supra-magnitude-")
        ratio = float(ratio_text)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Supra lambda must be in [0, 1].")
        return "supra", "magnitude-bottom", ratio
    if method.startswith("supra"):
        pieces = method.split("-", 1)
        if len(pieces) != 2:
            raise ValueError("Supra method names must look like 'supra-0.3'.")
        mask_choice = "random" if "rand" in method else "super"
        ratio_text = pieces[1].replace("rand-", "")
        if ratio_text.endswith("-bottom"):
            mask_choice = "super-bottom"
            ratio_text = ratio_text.removesuffix("-bottom")
        ratio = float(ratio_text)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Supra lambda must be in [0, 1].")
        return "supra", mask_choice, ratio
    raise ValueError(f"Unknown method: {method}")


def build_budget_plan(args, spec: RunSpec, target_modules: List[str]) -> dict:
    config = AutoConfig.from_pretrained(spec.model, trust_remote_code=True)
    shapes = llama_module_shapes(config, target_modules)
    target_dense_params = sum(out_features * in_features for out_features, in_features in shapes)
    reference_lora_params = lora_param_count(shapes, spec.lora_r)
    total_sparse_rate = args.sparse_rate_override
    if total_sparse_rate is None:
        total_sparse_rate = reference_lora_params / target_dense_params

    adapter_name, mask_choice, lora_params_ratio = parse_method(spec.method)
    train_sparse_rate = total_sparse_rate
    train_lora_r = spec.lora_r
    component_lora_ratio = None

    if adapter_name == "base":
        train_sparse_rate = 0.0
        train_lora_r = 0
        adapter_params = 0
    elif adapter_name == "full":
        train_sparse_rate = 0.0
        train_lora_r = 0
        adapter_params = None
    elif adapter_name == "lora":
        train_sparse_rate = 0.0
        adapter_params = reference_lora_params
    elif adapter_name in {"super", "sift"}:
        adapter_params = sparse_param_count(
            shapes,
            total_sparse_rate,
            add_one=(adapter_name != "super" or mask_choice != "super-bottom-structured"),
        )
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

    if adapter_name in {"base", "full"}:
        budget_error_pct = 0.0
    elif reference_lora_params == 0:
        budget_error_pct = 0.0
    else:
        budget_error_pct = 100.0 * (adapter_params - reference_lora_params) / reference_lora_params
    if adapter_name not in {"base", "full"} and abs(budget_error_pct) > args.budget_tolerance_pct:
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
        "is_unbudgeted": adapter_name == "full",
    }


def import_rosa_train():
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from rosa.finetune_rosa import train as train_rosa  # noqa: PLC0415

    return train_rosa


def collect_trainable_param_report(model, budget_plan: dict) -> dict:
    requires_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_params = getattr(model, "optimizer_trainable_params", None)
    if optimizer_params is None:
        optimizer_params = requires_grad_params
    reference_params = budget_plan["reference_lora_params"]
    if budget_plan.get("is_baseline") or budget_plan.get("is_unbudgeted"):
        trainable_budget_error_pct = 0.0
    elif reference_params:
        trainable_budget_error_pct = 100.0 * (optimizer_params - reference_params) / reference_params
    else:
        trainable_budget_error_pct = 0.0

    return {
        "trainable_params": int(optimizer_params),
        "requires_grad_params": int(requires_grad_params),
        "budget_trainable_params": int(
            budget_plan["adapter_trainable_params_estimate"]
            if budget_plan["adapter_trainable_params_estimate"] is not None
            else optimizer_params
        ),
        "trainable_budget_error_pct": float(trainable_budget_error_pct),
    }


def merge_adapter_for_evaluation(model, adapter_name: str) -> Tuple[object, bool]:
    if adapter_name != "rosa":
        return model, False
    merge_and_unload = getattr(model, "merge_and_unload", None)
    if not callable(merge_and_unload):
        print("RoSA merge_and_unload is unavailable; evaluating with live adapter layers.")
        model.eval()
        return model, False
    print("Merging RoSA adapters into base weights for evaluation.")
    merged_model = merge_and_unload(progressbar=False)
    merged_model.eval()
    if hasattr(merged_model, "config"):
        merged_model.config.use_cache = True
    return merged_model, True


def train_one_run(args, spec: RunSpec, budget_plan: dict, target_modules: List[str]):
    adapter_name, mask_choice, lora_params_ratio = parse_method(spec.method)
    random_indices = mask_choice == "random"
    output_dir = os.path.join(args.checkpoint_dir, spec.run_id)
    calibration_data = args.train_data if args.calibration_data == "same_as_train" else args.calibration_data
    if adapter_name == "rosa":
        lora_params_ratio = args.rosa_lora_budget_ratio
    train_adapter_name = "no" if adapter_name == "full" else adapter_name

    if args.dry_run and adapter_name == "base":
        print(
            "DRY RUN:",
            json.dumps(
                {
                    "base_model": spec.model,
                    "adapter_name": "base",
                    "method_name": spec.method,
                    "output_dir": output_dir,
                },
                indent=2,
            ),
        )
        return None, None, output_dir

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
        adapter_name=train_adapter_name,
        method_name=spec.method,
        random_indices=random_indices,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        optimizer_name=args.optimizer_name,
        save_model=args.save_adapters,
    )
    training_curve_dir = getattr(args, "training_curve_dir", "")
    if training_curve_dir:
        training_curve_path = os.path.join(training_curve_dir, f"{spec.run_id}.jsonl")
        common_kwargs["training_curve_path"] = training_curve_path
        common_kwargs["training_curve_metadata"] = {
            "run_id": spec.run_id,
            "model": spec.model,
            "method": spec.method,
            "lr": spec.lr,
            "seed": spec.seed,
            "budget_lora_r": spec.lora_r,
            "train_lora_r": budget_plan["train_lora_r"],
            "train_sparse_rate": budget_plan["train_sparse_rate"],
            "component_lora_ratio": budget_plan["component_lora_ratio"],
            "target_modules": target_modules,
            "calibration_data": calibration_data if adapter_name != "rosa" else None,
            "full_ft_checkpoint": args.full_ft_checkpoint if mask_choice.startswith("full-delta") else None,
        }
    logging_steps = getattr(args, "logging_steps", None)
    if logging_steps is not None:
        common_kwargs["logging_steps"] = logging_steps
    if getattr(args, "bf16", False) or adapter_name == "full":
        common_kwargs["bf16"] = True

    if adapter_name != "rosa":
        common_kwargs.update(
            calibration_data=calibration_data,
            calibration_nsamples=args.calibration_nsamples,
            calibration_seed=args.calibration_seed,
            full_ft_checkpoint=args.full_ft_checkpoint,
        )
        if adapter_name in {"super", "supra"}:
            common_kwargs["mask_choice"] = mask_choice
    else:
        common_kwargs.update(
            rosa_schedule=args.rosa_schedule,
            rosa_spa_num_grads=args.rosa_spa_num_grads,
            rosa_dtype=args.rosa_dtype,
        )
        if args.rosa_dtype == "bf16":
            common_kwargs["bf16"] = True

    if args.dry_run:
        print("DRY RUN:", json.dumps({**common_kwargs, "target_modules": target_modules}, indent=2, default=str))
        return None, None, output_dir

    if adapter_name == "rosa":
        train_rosa = import_rosa_train()
        rosa_kwargs = dict(common_kwargs)
        rosa_kwargs.pop("lora_params_ratio")
        rosa_kwargs.pop("random_indices")
        model, tokenizer = train_rosa(**rosa_kwargs)
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
    define_metric("progress/completed_lr_tuning_runs")
    define_metric("progress/completed_full_eval_runs")
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
        "progress/completed_full_eval_runs": completed_runs,
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


def log_wandb_tuning_result(wandb_state, row: dict, tuned_runs: int, total_tuning_runs: int) -> None:
    if wandb_state is None:
        return
    lr_tuning = row.get("lr_tuning", {})
    metrics = {
        "progress/completed_lr_tuning_runs": tuned_runs,
        "progress/total_lr_tuning_runs": total_tuning_runs,
        "progress/lr_tuning_fraction": tuned_runs / total_tuning_runs if total_tuning_runs else 1.0,
        "hparams/lr": row.get("lr"),
        "hparams/budget_lora_r": row.get("budget_lora_r"),
        "budget/trainable_params": row.get("trainable_params"),
        "budget/requires_grad_params": row.get("requires_grad_params"),
        "budget/reference_lora_params": row.get("reference_lora_params"),
        "budget/trainable_error_pct": row.get("trainable_budget_error_pct"),
        "lr_tuning/ppl": lr_tuning.get("ppl"),
        "lr_tuning/nll": lr_tuning.get("nll"),
    }
    metrics = {key: finite_or_none(value) for key, value in metrics.items()}
    wandb_state["run"].log(metrics)


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


def build_tables(
    results: Iterable[dict],
    datasets: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index_cols = ["seed", "model", "lora_r", "lr", "method"]
    accuracy_rows = []
    ppl_rows = []
    nll_rows = []
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

        nll_row = dict(index_values)
        nll_row["trainable_params"] = trainable_params
        for dataset in datasets:
            nll_row[dataset] = row.get("nll", {}).get(dataset)
        nll_row["Average"] = row.get("nll", {}).get("Average")
        nll_rows.append(nll_row)

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
            full_ft_checkpoint=row.get("full_ft_checkpoint"),
            val_split_seed=row.get("val_split_seed"),
            lr_tuning_ppl=row.get("lr_tuning", {}).get("ppl"),
            lr_tuning_nll=row.get("lr_tuning", {}).get("nll"),
            lr_tuning_examples=row.get("lr_tuning", {}).get("examples"),
            average_accuracy=row.get("accuracy", {}).get("Average"),
            average_ppl=row.get("ppl", {}).get("Average"),
            average_nll=row.get("nll", {}).get("Average"),
            ppl_target=row.get("ppl_target"),
            ppl_eval_data=row.get("ppl_eval_data"),
            ppl_eval_name=row.get("ppl_eval_name"),
            accuracy_eval_skipped=row.get("accuracy_eval_skipped"),
        )
        summary_rows.append(summary_row)

    accuracy_df = pd.DataFrame(accuracy_rows)
    ppl_df = pd.DataFrame(ppl_rows)
    nll_df = pd.DataFrame(nll_rows)
    summary_df = pd.DataFrame(summary_rows)
    if not accuracy_df.empty:
        accuracy_df = accuracy_df.set_index(index_cols).sort_index()
    if not ppl_df.empty:
        ppl_df = ppl_df.set_index(index_cols).sort_index()
    if not nll_df.empty:
        nll_df = nll_df.set_index(index_cols).sort_index()
    if not summary_df.empty:
        summary_df = summary_df.set_index(index_cols).sort_index()
    return accuracy_df, ppl_df, nll_df, summary_df


def format_mean_std(mean_df: pd.DataFrame, std_df: pd.DataFrame) -> pd.DataFrame:
    formatted = mean_df.copy().astype(object)
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


def save_selected_lr_tables(
    out_dir: str,
    results: Iterable[dict],
    datasets: List[str],
    tuning_results: Optional[Iterable[dict]] = None,
) -> None:
    result_rows = list(results)
    selection_source = list(tuning_results) if tuning_results is not None else result_rows
    rows = [
        row
        for row in selection_source
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
    for row in result_rows:
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
    aggregate("nll", "selected_nll")


def save_tables(
    out_dir: str,
    results: Iterable[dict],
    datasets: List[str],
    tuning_results: Optional[Iterable[dict]] = None,
) -> None:
    result_rows = list(results)
    accuracy_df, ppl_df, nll_df, summary_df = build_tables(result_rows, datasets)
    os.makedirs(out_dir, exist_ok=True)

    for name, df in [
        ("accuracy_table", accuracy_df),
        ("ppl_table", ppl_df),
        ("nll_table", nll_df),
        ("summary_table", summary_df),
    ]:
        df.to_csv(os.path.join(out_dir, f"{name}.csv"))
        save_pickle(df, os.path.join(out_dir, f"{name}.pkl"))
        with open(os.path.join(out_dir, f"{name}.tex"), "w") as f:
            f.write(df.to_latex(float_format=lambda value: f"{value:.2f}" if pd.notna(value) else ""))
    save_selected_lr_tables(out_dir, result_rows, datasets, tuning_results=tuning_results)


def resolved_calibration_data(args) -> str:
    return args.train_data if args.calibration_data == "same_as_train" else args.calibration_data


def make_result_row(
    args,
    spec: RunSpec,
    budget_plan: dict,
    trainable_param_report: dict,
    target_modules: List[str],
    checkpoint_dir: str,
    lr_tuning: dict,
    eval_stage: str,
) -> dict:
    return {
        "run_id": spec.run_id,
        **asdict(spec),
        "eval_stage": eval_stage,
        "sparse_rate": budget_plan["total_sparse_rate"],
        **budget_plan,
        **trainable_param_report,
        "target_modules": target_modules,
        "train_data": args.train_data,
        "dataset_dir": args.dataset_dir,
        "calibration_data": resolved_calibration_data(args),
        "calibration_nsamples": args.calibration_nsamples,
        "calibration_seed": args.calibration_seed,
        "full_ft_checkpoint": args.full_ft_checkpoint,
        "val_split_seed": args.val_split_seed,
        "ppl_target": args.ppl_target,
        "ppl_eval_data": args.ppl_eval_data,
        "ppl_eval_name": args.ppl_eval_name,
        "lr_tuning": lr_tuning,
        "checkpoint_dir": checkpoint_dir,
    }


def print_spec_header(spec: RunSpec, budget_plan: dict, stage: str) -> None:
    print("=" * 100)
    print("Stage:", stage)
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


def eval_progress_path(out_dir: str, spec: RunSpec, dataset: str) -> str:
    return os.path.join(out_dir, "eval_progress", spec.run_id, f"{dataset}.jsonl")


def save_full_model_checkpoint(model, tokenizer, checkpoint_dir: str) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    previous_use_cache = getattr(model.config, "use_cache", None)
    if previous_use_cache is not None:
        model.config.use_cache = True
    try:
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
    print("Saved full-model checkpoint to:", checkpoint_dir)


def run_spec_once(
    args,
    spec: RunSpec,
    budget_plan: dict,
    target_modules: List[str],
    datasets: List[str],
    lr_tuning_records: Optional[List[dict]],
    full_eval: bool,
    stage: str,
) -> Optional[dict]:
    set_seed(spec.seed)
    model = tokenizer = None
    try:
        model, tokenizer, checkpoint_dir = train_one_run(args, spec, budget_plan, target_modules)
        if args.dry_run:
            return None

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

        adapter_name, _, _ = parse_method(spec.method)
        full_model_checkpoint_saved = False
        if adapter_name == "full" and full_eval:
            save_full_model_checkpoint(model, tokenizer, checkpoint_dir)
            full_model_checkpoint_saved = True

        model, adapter_merged_for_eval = merge_adapter_for_evaluation(model, adapter_name)

        lr_tuning = {}
        if lr_tuning_records is not None:
            tune_ppl, tune_nll, tune_count = evaluate_perplexity_on_records(
                model=model,
                tokenizer=tokenizer,
                records=lr_tuning_records,
                max_length=args.ppl_max_length,
                max_examples=args.lr_tuning_max_examples,
                target_mode=args.ppl_target,
                eval_batch_size=args.ppl_eval_batch_size,
            )
            lr_tuning = {"ppl": tune_ppl, "nll": tune_nll, "examples": tune_count}
            print(f"LR tuning validation ppl: {tune_ppl:.4f} (nll={tune_nll:.4f}, examples={tune_count})")

        row = make_result_row(
            args=args,
            spec=spec,
            budget_plan=budget_plan,
            trainable_param_report=trainable_param_report,
            target_modules=target_modules,
            checkpoint_dir=checkpoint_dir,
            lr_tuning=lr_tuning,
            eval_stage=stage,
        )
        if full_model_checkpoint_saved:
            row["full_model_checkpoint_saved"] = True
        if adapter_merged_for_eval:
            row["adapter_merged_for_eval"] = True

        if full_eval:
            accuracy: Dict[str, float] = {}
            row["eval_progress_dir"] = os.path.join(args.out_dir, "eval_progress", spec.run_id)
            if args.skip_accuracy_eval:
                row["accuracy_eval_skipped"] = True
                print("Skipping generation accuracy evaluation; computing NLL/PPL only.")
            else:
                for dataset in datasets:
                    score = eval_model(
                        dataset_name=dataset,
                        model=model,
                        tokenizer=tokenizer,
                        dataset_dir=args.dataset_dir,
                        max_examples=args.accuracy_max_examples,
                        max_new_tokens=args.generation_max_new_tokens,
                        num_beams=args.generation_num_beams,
                        verbose=args.verbose_generation_eval,
                        progress_path=eval_progress_path(args.out_dir, spec, dataset),
                    ) * 100.0
                    accuracy[dataset] = score
                    print(f"{dataset} accuracy: {score:.4f}")
                accuracy["Average"] = float(np.mean([accuracy[dataset] for dataset in datasets]))

            ppl_eval_start = time.perf_counter()
            if args.ppl_eval_data:
                ppl, nll, ppl_examples = evaluate_perplexity_on_data_file(
                    model=model,
                    tokenizer=tokenizer,
                    data_path=args.ppl_eval_data,
                    data_name=args.ppl_eval_name,
                    max_length=args.ppl_max_length,
                    max_examples=args.ppl_max_examples,
                    target_mode=args.ppl_target,
                    eval_batch_size=args.ppl_eval_batch_size,
                )
            else:
                ppl, nll, ppl_examples = evaluate_perplexity(
                    model=model,
                    tokenizer=tokenizer,
                    datasets=datasets,
                    max_length=args.ppl_max_length,
                    max_examples=args.ppl_max_examples,
                    target_mode=args.ppl_target,
                    eval_batch_size=args.ppl_eval_batch_size,
                    dataset_dir=args.dataset_dir,
                )
            ppl_eval_time_sec = time.perf_counter() - ppl_eval_start
            print(f"NLL/PPL evaluation time: {ppl_eval_time_sec:.3f} seconds")
            row.update(
                selected_by_lr_tuning=(stage == "selected_full_eval"),
                accuracy=accuracy,
                ppl=ppl,
                nll=nll,
                ppl_examples=ppl_examples,
                ppl_eval_time_sec=ppl_eval_time_sec,
            )

        return row
    finally:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def merged_tuning_results(tuning_rows: Dict[str, dict], full_rows: Dict[str, dict]) -> Dict[str, dict]:
    merged = dict(tuning_rows)
    for run_id, row in full_rows.items():
        if row.get("lr_tuning", {}).get("nll") is not None:
            merged.setdefault(run_id, row)
    return merged


def selection_group_key(spec: RunSpec) -> Tuple[str, int, str]:
    return spec.model, spec.lora_r, spec.method


def select_specs_from_tuning(specs: List[RunSpec], tuning_lookup: Dict[str, dict]) -> Tuple[List[RunSpec], Dict[Tuple[str, int, str], float]]:
    grouped_specs: Dict[Tuple[str, int, str], List[RunSpec]] = {}
    for spec in specs:
        grouped_specs.setdefault(selection_group_key(spec), []).append(spec)

    selected_specs: List[RunSpec] = []
    selected_lrs: Dict[Tuple[str, int, str], float] = {}
    for group_key, group_specs in grouped_specs.items():
        missing = [spec.run_id for spec in group_specs if spec.run_id not in tuning_lookup]
        if missing:
            print(
                "LR selection pending for",
                group_key,
                f"({len(missing)} missing tuning rows)",
            )
            continue

        lr_rows = []
        for spec in group_specs:
            lr_tuning = tuning_lookup[spec.run_id].get("lr_tuning", {})
            nll = lr_tuning.get("nll")
            if nll is None or not np.isfinite(nll):
                continue
            lr_rows.append({"lr": spec.lr, "nll": float(nll)})
        if not lr_rows:
            print("LR selection skipped because all tuning metrics are invalid for", group_key)
            continue

        lr_df = pd.DataFrame(lr_rows)
        grouped = lr_df.groupby("lr")["nll"].mean()
        selected_lr = float(grouped.idxmin())
        selected_lrs[group_key] = selected_lr
        print("Selected LR for", group_key, "=", f"{selected_lr:g}", "(mean nll", f"{grouped.loc[selected_lr]:.4f})")
        selected_specs.extend([spec for spec in group_specs if spec.lr == selected_lr])

    return selected_specs, selected_lrs


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
    if args.batch_size <= 0 or args.micro_batch_size <= 0:
        raise ValueError("--batch_size and --micro_batch_size must be positive.")
    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("--batch_size must be divisible by --micro_batch_size.")
    if args.calibration_nsamples <= 0:
        raise ValueError("--calibration_nsamples must be positive.")
    if not 0.0 <= args.rosa_lora_budget_ratio <= 1.0:
        raise ValueError("--rosa_lora_budget_ratio must be in [0, 1].")
    if args.budget_tolerance_pct < 0.0:
        raise ValueError("--budget_tolerance_pct must be nonnegative.")
    if args.ppl_eval_batch_size <= 0:
        raise ValueError("--ppl_eval_batch_size must be positive.")
    if not args.eval_all_lrs and args.skip_lr_tuning_metric:
        raise ValueError("--skip_lr_tuning_metric cannot be used with selected-only evaluation.")
    if any(parse_method(method)[1].startswith("full-delta") for method in parse_csv_list(args.methods, str)):
        if not args.full_ft_checkpoint:
            raise ValueError("--full_ft_checkpoint is required for super-delta/full-delta methods.")

    target_modules = LEGACY_TARGET_MODULES if args.legacy_target_modules else parse_csv_list(args.target_modules, str)
    datasets = parse_csv_list(args.datasets, str)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "run_results.jsonl")
    tuning_results_path = os.path.join(args.out_dir, "tuning_results.jsonl")
    failures_path = os.path.join(args.out_dir, "failed_runs.jsonl")
    existing = load_existing_results(results_path)
    tuning_existing = load_existing_results(tuning_results_path)
    tuning_lookup = merged_tuning_results(tuning_existing, existing)
    completed_results = list(existing.values())

    print("Training data:", args.train_data)
    print("Wanda calibration data:", resolved_calibration_data(args))
    print("Wanda calibration samples:", args.calibration_nsamples)
    print("Wanda calibration seed:", args.calibration_seed)
    print("Full-FT checkpoint for delta masks:", args.full_ft_checkpoint or None)
    print("Evaluation datasets:", datasets)
    print("Target modules:", target_modules)
    print("Validation split seed:", args.val_split_seed)
    print("NLL/PPL target:", args.ppl_target)
    if args.ppl_eval_data:
        print("NLL/PPL eval data:", args.ppl_eval_name, "=", args.ppl_eval_data)
    print("Output directory:", args.out_dir)
    print("Evaluation mode:", "all learning rates" if args.eval_all_lrs else "selected LR only")
    if args.skip_accuracy_eval:
        print("Accuracy generation eval: skipped")

    specs = list(iter_run_specs(args))
    if args.eval_all_lrs:
        pending_specs = [
            spec
            for spec in specs
            if not (args.only_missing and spec.run_id in existing)
        ]
    else:
        tuning_pending_specs = [
            spec
            for spec in specs
            if not (args.only_missing and spec.run_id in tuning_lookup)
        ]
        selected_specs, _ = select_specs_from_tuning(specs, tuning_lookup)
        full_eval_pending_specs = [
            spec
            for spec in selected_specs
            if not (args.only_missing and spec.run_id in existing)
        ]
        pending_specs = tuning_pending_specs + full_eval_pending_specs
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
        print(
            "LR tuning metric:",
            args.ppl_target,
            "validation perplexity on",
            len(lr_tuning_records),
            "held-out examples",
        )

    action_count = 0
    failure_count = 0

    def reached_action_limit() -> bool:
        return args.max_runs is not None and action_count >= args.max_runs

    def execute_and_record(spec: RunSpec, stage: str, full_eval: bool) -> None:
        nonlocal action_count, failure_count, completed_results, tuning_lookup
        budget_plan = {}
        try:
            budget_plan = build_budget_plan(args, spec, target_modules)
            print_spec_header(spec, budget_plan, stage)
            log_wandb_run_started(wandb_state, spec, budget_plan)
            row = run_spec_once(
                args=args,
                spec=spec,
                budget_plan=budget_plan,
                target_modules=target_modules,
                datasets=datasets,
                lr_tuning_records=lr_tuning_records,
                full_eval=full_eval,
                stage=stage,
            )
            action_count += 1
            if row is None:
                return

            if full_eval:
                append_jsonl(results_path, row)
                existing[spec.run_id] = row
                completed_results = list(existing.values())
                if row.get("lr_tuning", {}).get("nll") is not None:
                    tuning_lookup.setdefault(spec.run_id, row)
                save_tables(args.out_dir, completed_results, datasets, tuning_results=tuning_lookup.values())
                log_wandb_result(wandb_state, row, len(existing))
            else:
                append_jsonl(tuning_results_path, row)
                tuning_existing[spec.run_id] = row
                tuning_lookup[spec.run_id] = row
                save_tables(args.out_dir, completed_results, datasets, tuning_results=tuning_lookup.values())
                log_wandb_tuning_result(wandb_state, row, len(tuning_lookup), len(specs))
        except Exception as exc:
            failure_count += 1
            traceback_text = traceback.format_exc()
            failure_row = {
                "run_id": spec.run_id,
                **asdict(spec),
                "eval_stage": stage,
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

    try:
        if args.eval_all_lrs:
            for spec in specs:
                if args.only_missing and spec.run_id in existing:
                    print("Already computed:", spec.run_id)
                    continue
                if reached_action_limit():
                    break
                execute_and_record(spec, stage="full_eval_all_lrs", full_eval=True)
        else:
            for spec in specs:
                if args.only_missing and spec.run_id in tuning_lookup:
                    print("Already tuned:", spec.run_id)
                    continue
                if reached_action_limit():
                    break
                execute_and_record(spec, stage="lr_tuning", full_eval=False)

            if not reached_action_limit():
                selected_specs, _ = select_specs_from_tuning(specs, tuning_lookup)
                for spec in selected_specs:
                    if args.only_missing and spec.run_id in existing:
                        print("Already full-evaluated selected LR:", spec.run_id)
                        continue
                    if reached_action_limit():
                        break
                    execute_and_record(spec, stage="selected_full_eval", full_eval=True)

        save_tables(args.out_dir, existing.values(), datasets, tuning_results=tuning_lookup.values())
        print("Finished. Tables written to:", args.out_dir)
    finally:
        finish_wandb(wandb_state)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Tune and evaluate PEFT methods on the Math17K arithmetic setup."
    )
    parser.add_argument("--config", default=config_args.config, help="JSON file containing experiment defaults.")
    parser.add_argument("--models", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--lora_rs", default="8")
    parser.add_argument("--lrs", default=DEFAULT_LRS)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--datasets", default=",".join(MATH_BENCHMARKS))
    parser.add_argument(
        "--dataset_dir",
        default=os.path.join(SCRIPT_DIR, "dataset"),
        help="Directory containing <dataset>/test.json files.",
    )
    parser.add_argument("--target_modules", default=",".join(FULL_LLAMA_TARGET_MODULES))
    parser.add_argument("--legacy_target_modules", action="store_true")
    parser.add_argument("--train_data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--calibration_data", default="c4")
    parser.add_argument("--calibration_nsamples", type=int, default=128)
    parser.add_argument("--calibration_seed", type=int, default=228)
    parser.add_argument("--full_ft_checkpoint", default="")
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
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--ppl_max_length", type=int, default=256)
    parser.add_argument("--ppl_max_examples", type=int, default=None)
    parser.add_argument("--ppl_eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--ppl_target",
        choices=["gold_output", "answer", "direct_answer"],
        default="gold_output",
        help=(
            "Text scored by NLL/PPL. gold_output scores the supervised output; answer scores only the final "
            "answer span inside the supervised output; direct_answer scores record['answer'] immediately after "
            "the prompt."
        ),
    )
    parser.add_argument(
        "--ppl_eval_data",
        default="",
        help="Optional JSON file for full-eval NLL/PPL. When set, NLL/PPL is computed on this file instead of benchmark datasets.",
    )
    parser.add_argument("--ppl_eval_name", default="Math17K")
    parser.add_argument("--lr_tuning_max_examples", type=int, default=None)
    parser.add_argument("--accuracy_max_examples", type=int, default=None)
    parser.add_argument(
        "--skip_accuracy_eval",
        action="store_true",
        help="Skip generation accuracy during full eval and only compute benchmark NLL/PPL.",
    )
    parser.add_argument("--generation_max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_num_beams", type=int, default=4)
    parser.add_argument("--verbose_generation_eval", action="store_true")
    parser.add_argument("--val_split_seed", "--lr_tuning_split_seed", dest="val_split_seed", type=int, default=42)
    parser.add_argument("--skip_lr_tuning_metric", action="store_true")
    parser.add_argument(
        "--eval_selected_only",
        dest="eval_all_lrs",
        action="store_false",
        help="Tune all learning rates cheaply, then run full benchmark generation only for the selected LR.",
    )
    parser.add_argument(
        "--eval_all_lrs",
        dest="eval_all_lrs",
        action="store_true",
        help="Legacy mode: run full benchmark generation for every learning rate.",
    )
    parser.add_argument("--sparse_rate_override", type=float, default=None)
    parser.add_argument("--rosa_lora_budget_ratio", type=float, default=0.5)
    parser.add_argument("--rosa_schedule", default="wl64")
    parser.add_argument("--rosa_spa_num_grads", type=int, default=1)
    parser.add_argument("--rosa_dtype", default="bf16")
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
    parser.add_argument("--save_adapters", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true", default=True)
    parser.add_argument("--stop_on_error", dest="continue_on_error", action="store_false")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.set_defaults(eval_all_lrs=False)

    if config_args.config:
        with open(config_args.config, "r") as config_file:
            config = json.load(config_file)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(config) - valid_keys)
        if unknown_keys:
            parser.error(f"Unknown keys in {config_args.config}: {', '.join(unknown_keys)}")
        parser.set_defaults(**config)

    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
