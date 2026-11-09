"""Two-process batch=1 checkpoint evaluator for frozen B1.

Each worker loads its own fresh model replica from the saved checkpoint, runs
literal batch=1 beam-search generation on half of the examples, and writes a
shard of per-example rows. The parent merges shards into the standard
per-dataset progress files and returns dataset accuracies. There is no
persistent state and no adapter hot-swapping: checkpoint isolation comes from
one fresh process per replica per evaluated checkpoint.
"""

from __future__ import annotations

import copy
import json
import multiprocessing as mp
import os
from typing import Optional, Sequence

try:
    from .evaluate import (
        _model_device,
        _response_from_decoded,
        _run_generation,
        generate_prompt,
        load_data,
        score_prediction,
    )
except ImportError:
    from evaluate import (
        _model_device,
        _response_from_decoded,
        _run_generation,
        generate_prompt,
        load_data,
        score_prediction,
    )


def _worker(
    checkpoint_dir: str,
    items: Sequence,
    shard_path: str,
    dataset_dir: str,
    max_examples: Optional[int],
    max_new_tokens: int,
    num_beams: int,
    dtype: str,
) -> None:
    try:
        from .checkpoints import load_checkpoint
    except ImportError:
        from checkpoints import load_checkpoint

    model, tokenizer, _metadata = load_checkpoint(checkpoint_dir, dtype=dtype)
    device = _model_device(model)
    datasets: dict[str, list[dict]] = {}
    for name in sorted({dataset_name for dataset_name, _ in items}):
        records = load_data(name, dataset_dir=dataset_dir)
        if max_examples is not None:
            records = records[:max_examples]
        datasets[name] = records

    with open(shard_path, "w", encoding="utf-8") as handle:
        for dataset_name, index in items:
            record = datasets[dataset_name][index]
            prompt = generate_prompt(record.get("instruction", ""), record.get("input"))
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            output = _run_generation(model, inputs, tokenizer, num_beams, max_new_tokens)
            text = _response_from_decoded(tokenizer.decode(output.sequences[0], skip_special_tokens=True))
            prediction, is_correct = score_prediction(dataset_name, text, record.get("answer"))
            row = copy.deepcopy(record)
            row.update(
                output_pred=text,
                pred=prediction,
                flag=is_correct,
                idx=index,
                dataset=dataset_name,
                total=len(datasets[dataset_name]),
            )
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _merge_shards(shard_paths: Sequence[str], datasets: Sequence[str]) -> dict[str, dict[int, dict]]:
    rows_by_dataset: dict[str, dict[int, dict]] = {name: {} for name in datasets}
    for path in shard_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows_by_dataset[row["dataset"]][int(row["idx"])] = row
    return rows_by_dataset


def evaluate_checkpoint_parallel(
    checkpoint_dir: str,
    datasets: Sequence[str],
    dataset_dir: str,
    out_dir: str,
    workers: int = 2,
    max_examples: Optional[int] = None,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    dtype: str = "bf16",
    fresh: bool = True,
) -> dict[str, float]:
    """Evaluate a saved checkpoint with fresh batch=1 replicas and return dataset accuracies."""

    if workers < 2:
        raise ValueError("Parallel evaluation requires at least two workers")
    os.makedirs(out_dir, exist_ok=True)
    progress_paths = {name: os.path.join(out_dir, f"{name}.jsonl") for name in datasets}
    if fresh:
        for path in progress_paths.values():
            if os.path.exists(path) and os.path.getsize(path) > 0:
                raise RuntimeError(f"Evaluation progress already exists; use a fresh output directory: {path}")

    datasets_records = {name: load_data(name, dataset_dir=dataset_dir) for name in datasets}
    if max_examples is not None:
        datasets_records = {name: records[:max_examples] for name, records in datasets_records.items()}
    items = [(name, index) for name in datasets for index in range(len(datasets_records[name]))]

    shard_dir = os.path.join(out_dir, "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    shards = [items[worker_id::workers] for worker_id in range(workers)]
    context = mp.get_context("spawn")
    processes = []
    for worker_id in range(workers):
        shard_path = os.path.join(shard_dir, f"worker{worker_id}.jsonl")
        if os.path.exists(shard_path):
            os.remove(shard_path)
        process = context.Process(
            target=_worker,
            args=(
                checkpoint_dir,
                shards[worker_id],
                shard_path,
                dataset_dir,
                max_examples,
                max_new_tokens,
                num_beams,
                dtype,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Parallel evaluation worker failed with exit code {process.exitcode}")

    rows_by_dataset = _merge_shards(
        [os.path.join(shard_dir, f"worker{worker_id}.jsonl") for worker_id in range(workers)],
        datasets,
    )
    scores: dict[str, float] = {}
    for name in datasets:
        rows = rows_by_dataset[name]
        total = len(datasets_records[name])
        if len(rows) != total:
            raise RuntimeError(f"Parallel evaluation produced {len(rows)} rows for {name}, expected {total}")
        with open(progress_paths[name], "w", encoding="utf-8") as handle:
            for index in sorted(rows):
                handle.write(json.dumps(rows[index], sort_keys=True) + "\n")
        scores[name] = sum(1 for row in rows.values() if row.get("flag")) / total if total else float("nan")
    scores["Average"] = float(sum(scores[name] for name in datasets) / len(datasets)) if datasets else float("nan")
    return scores
