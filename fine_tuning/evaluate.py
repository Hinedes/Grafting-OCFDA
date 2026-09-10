"""Exact-answer evaluation for the six arithmetic benchmarks."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Optional

import torch
from tqdm import tqdm
from transformers import GenerationConfig


def _model_device(model) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(
    dataset_name: str,
    model,
    tokenizer,
    dataset_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    verbose: bool = False,
    progress_path: Optional[str] = None,
    resume_progress: bool = True,
) -> float:
    device = _model_device(model)

    def generate(instruction: str, input_text: Optional[str] = None) -> str:
        prompt = generate_prompt(instruction, input_text)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        generation_config = GenerationConfig(
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            ),
            eos_token_id=tokenizer.eos_token_id,
        )
        previous_use_cache = getattr(model.config, "use_cache", None)
        model.config.use_cache = True
        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
        finally:
            if previous_use_cache is not None:
                model.config.use_cache = previous_use_cache

        decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        if "### Response:" in decoded:
            return decoded.split("### Response:", 1)[1].strip()
        return decoded.strip()

    dataset = load_data(dataset_name, dataset_dir=dataset_dir)
    if max_examples is not None:
        dataset = dataset[:max_examples]

    total = len(dataset)
    completed = _load_eval_progress(progress_path, total) if resume_progress else {}
    correct = sum(1 for row in completed.values() if row.get("flag"))
    completed_count = len(completed)

    with tqdm(total=total, initial=completed_count, desc=dataset_name) as progress:
        for index, record in enumerate(dataset):
            if index in completed:
                continue

            prediction_text = generate(record.get("instruction", ""), record.get("input"))
            label = record.get("answer")
            if dataset_name.lower() == "aqua":
                prediction = extract_answer_letter(prediction_text)
                is_correct = str(label) == prediction
            else:
                prediction = extract_answer_number(dataset_name, prediction_text)
                is_correct = abs(float(label) - prediction) <= 0.001

            correct += int(is_correct)
            completed_count += 1
            accuracy = correct / completed_count
            result = copy.deepcopy(record)
            result.update(
                output_pred=prediction_text,
                pred=prediction,
                flag=is_correct,
                idx=index,
                dataset=dataset_name,
                total=total,
            )
            _append_eval_progress(progress_path, result)
            if verbose:
                print(f"\n{dataset_name} #{index}: prediction={prediction!r}, label={label!r}")
                print(prediction_text)
            progress.set_postfix(accuracy=f"{accuracy:.4f}")
            progress.update(1)

    return correct / total if total else float("nan")


def _load_eval_progress(progress_path: Optional[str], total: int) -> dict[int, dict]:
    completed = {}
    if not progress_path or not os.path.exists(progress_path):
        return completed
    with open(progress_path, "r", encoding="utf-8") as progress_file:
        for line in progress_file:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            index = row.get("idx")
            if isinstance(index, int) and 0 <= index < total and "flag" in row:
                completed[index] = row
    return completed


def _append_eval_progress(progress_path: Optional[str], row: dict) -> None:
    if not progress_path:
        return
    directory = os.path.dirname(progress_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(progress_path, "a", encoding="utf-8") as progress_file:
        progress_file.write(json.dumps(row, sort_keys=True) + "\n")
        progress_file.flush()


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


def load_data(dataset: str, dataset_dir: Optional[str] = None) -> list[dict]:
    dataset_dir = dataset_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    path = os.path.join(dataset_dir, dataset, "test.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def extract_answer_number(dataset_name: str, sentence: str) -> float:
    if dataset_name.lower() not in {"multiarith", "addsub", "singleeq", "gsm8k", "svamp"}:
        raise NotImplementedError(f"Unsupported numeric benchmark: {dataset_name}")
    matches = re.findall(r"-?\d+\.?\d*", sentence.replace(",", ""))
    return float(matches[-1]) if matches else float("inf")


def extract_answer_letter(sentence: str) -> str:
    explicit = re.findall(
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*[\(\[]?\s*([A-E])\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].upper()
    matches = re.findall(r"\b([A-E])\b", sentence, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else ""
