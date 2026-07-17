# Super-Tuning

[![arXiv](https://img.shields.io/badge/arXiv-2607.09287-b31b1b.svg)](https://arxiv.org/abs/2607.09287)

Official implementation of **Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning**.

Super turns pruning scores into fixed sparse fine-tuning supports. For a linear-layer weight $W_{ij}$, the Wanda variant ranks coordinates with

$$
s_{ij}=|W_{ij}|\lVert X_{j:}\rVert_2,
$$

where $X_{j:}$ contains calibration activations entering input coordinate $j$. Supra combines the resulting sparse update with LoRA under the same rank-equivalent trainable-scalar budget. The repository also provides the paper baselines and the complete Math17K learning-rate selection and evaluation pipeline.

> **Math17K protocol note:** Math17K contains questions from the first 80% of the six packaged benchmark snapshots. The submitted full-snapshot protocol is reproducible, but it is not a held-out evaluation. Use `supertuning-data-audit` and `--dataset_dir` as described under Data for evaluation on disjoint questions.

## Paper

**Paper:** [Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning](https://arxiv.org/abs/2607.09287)

## What is included

- Super with Wanda TopK, Wanda BottomK, random, and magnitude supports
- Supra with configurable low-rank budget fraction `lambda`
- Supra-Mag, which combines LoRA with a magnitude-BottomK sparse support
- LoRA, RoSA, SIFT, magnitude-only sparse tuning, full fine-tuning, and a frozen base model
- Rank-equivalent parameter accounting against LoRA rank `r0`
- Validation-NLL learning-rate selection followed by exact-answer evaluation
- Math17K presets for Llama-3.2-1B and Meta-Llama-3-8B
- Machine-readable JSON/CSV results and reusable adapter checkpoints
- A fixed-step efficiency profiler used by the paper

## Installation

The tested setup uses Python 3.10 or 3.11, PyTorch 2.5/2.6, and an NVIDIA GPU. RoSA additionally requires Linux/CUDA because it uses bitsandbytes.

```bash
git clone https://github.com/vectozavr/SuperTuning.git
cd SuperTuning

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[rosa,tracking,analysis]"
```

Request access to the gated Meta Llama checkpoints and authenticate with Hugging Face before running the examples. C4 calibration is downloaded through `datasets`; set `HF_HOME` and `HF_DATASETS_CACHE` when cluster storage requires a specific cache location.

For development:

```bash
python -m pip install -e ".[dev,rosa,tracking,analysis]"
pytest
```

## Quick start

This smoke run trains Supra for two optimizer steps, constructs its support from four C4 samples, and evaluates four examples from each benchmark:

```bash
supertuning-math \
  --models meta-llama/Llama-3.2-1B \
  --methods supra-0.8-bottom \
  --lrs 5e-4 \
  --num_epochs 1 \
  --max_steps 2 \
  --calibration_nsamples 4 \
  --accuracy_max_examples 4 \
  --ppl_max_examples 4 \
  --out_dir runs/smoke/results \
  --checkpoint_dir runs/smoke/checkpoints
```

For a normal LR sweep, remove the smoke limits and use the paper grid:

```bash
supertuning-math \
  --models meta-llama/Llama-3.2-1B \
  --methods supra-0.8-bottom \
  --lrs 5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1 \
  --num_epochs 3 \
  --save_adapters \
  --out_dir runs/supra-1b/results \
  --checkpoint_dir runs/supra-1b/checkpoints
```

The default protocol uses Math17K, `r0=8`, batch size 16, micro-batch size 16, sequence length 256, 120 validation examples, 100 warmup steps, seed 0, and 128 C4 calibration samples.

## Paper presets

Four checked-in presets contain the complete method and LR grids:

| Preset | Model | Epochs |
| --- | --- | ---: |
| `configs/math17k/llama-1b-1epoch.json` | Llama-3.2-1B | 1 |
| `configs/math17k/llama-1b-3epoch.json` | Llama-3.2-1B | 3 |
| `configs/math17k/llama-8b-1epoch.json` | Meta-Llama-3-8B | 1 |
| `configs/math17k/llama-8b-3epoch.json` | Meta-Llama-3-8B | 3 |

Run one method per GPU and merge the results automatically:

```bash
supertuning-launch \
  --gpus 0,1,2,3 \
  --config configs/math17k/llama-1b-1epoch.json \
  --base_out_dir runs/llama-1b-1epoch/results \
  --base_checkpoint_dir runs/llama-1b-1epoch/checkpoints
```

Command-line options override preset values. For example, this runs only LoRA and Super-BottomK:

```bash
supertuning-launch \
  --gpus 0,1 \
  --config configs/math17k/llama-1b-1epoch.json \
  --methods lora,super-wanda-bottom \
  --base_out_dir runs/two-methods/results \
  --base_checkpoint_dir runs/two-methods/checkpoints
```

The generic Slurm launcher is documented in [`jobs/README.md`](jobs/README.md). Cluster-specific scripts used during paper development are retained under `jobs/orix/` for provenance.

## Method names

All sparse methods adapt the seven Llama projection matrices: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

| CLI method | Sparse support | Calibration |
| --- | --- | --- |
| `base` | Frozen model | None |
| `full` | All model parameters | None |
| `lora` | None | None |
| `rosa` | RoSA gradient-selected sparse support plus LoRA | During training |
| `sift-topk` | First-gradient TopK | During training |
| `sift-rand` | Uniform random | None |
| `super-wanda` | TopK of `abs(W) * ||X||_2` | C4 or JSON calibration data |
| `super-wanda-bottom` | BottomK of `abs(W) * ||X||_2` | C4 or JSON calibration data |
| `super-rand` | Uniform random | None |
| `magnitude-topk` | TopK of `abs(W)` | None |
| `magnitude-bottomk` | BottomK of `abs(W)` | None |
| `supra-0.3-bottom` | Wanda BottomK plus LoRA; `lambda=0.3` | C4 or JSON calibration data |
| `supra-0.8` | Wanda TopK plus LoRA; `lambda=0.8` | C4 or JSON calibration data |
| `supra-magnitude-0.3` | Magnitude BottomK plus LoRA; `lambda=0.3` | None |

For Supra, `lambda` is the fraction of the matched scalar budget assigned to the low-rank component. The runner accepts `0.3`, `0.5`, and `0.8` for both Wanda support directions and for Supra-Mag.

## Budget matching

The runner computes the sparse rate separately for each model from the selected target matrices:

$$
\rho = \frac{\sum_l r_0(d_{\mathrm{in}}^{(l)} + d_{\mathrm{out}}^{(l)})}
{\sum_l d_{\mathrm{in}}^{(l)}d_{\mathrm{out}}^{(l)}}.
$$

Super and magnitude baselines receive approximately the same number of trainable scalars as rank-`r0` LoRA. Supra divides this total between LoRA and sparse values, and the runner rejects configurations outside the default 3% budget tolerance.

## Learning-rate selection

For every `(model, method, rank, seed)` combination, the pipeline:

1. Trains every candidate learning rate.
2. Measures negative log-likelihood on the fixed held-out Math17K validation split.
3. Selects one learning rate using validation NLL only.
4. Runs the six generation benchmarks for that selected run.

Benchmark accuracy is never used to choose the learning rate. `--eval_all_lrs` is available only for analyses that intentionally evaluate every candidate.

## Results and checkpoints

Each output directory contains:

- `tuning_results.jsonl`: validation metrics for LR selection
- `run_results.jsonl`: completed full-evaluation records
- `selected_lr_by_method.csv`: selected learning rate per method
- `accuracy_table.csv` and `ppl_table.csv`: per-benchmark tables
- `summary_table.csv`: parameters, protocol metadata, and aggregate metrics

Pass `--save_adapters` to retain checkpoints. Every new checkpoint includes `supertuning_config.json`, which records the base model, method, target modules, sparse rate, budget split, and adapter hyperparameters.

Evaluate a saved checkpoint independently:

```bash
supertuning-eval \
  --checkpoint runs/supra-1b/checkpoints/<run-id> \
  --output_dir runs/supra-1b/re-evaluation
```

Sparse Super and Supra checkpoints store trainable values plus integer support indices. Full and SIFT checkpoints contain dense model weights. LoRA uses the standard PEFT checkpoint format.

## Data

The repository contains the exact Math7K and Math17K snapshots and six arithmetic benchmark snapshots used during the experiments. Math7K is the decontaminated split published by LLM-Adapters. Math17K has 17,172 records: it contains the 13,921-record Math14K snapshot plus records drawn from the first 80% of AddSub, MultiArith, SingleEq, GSM8K, AQuA, and SVAMP.

Consequently, evaluating a Math17K-trained model on all records in the six full benchmark snapshots is not a held-out evaluation. A clean evaluation must filter out every record whose normalized instruction occurs in Math17K. This leaves 79 AddSub, 109 MultiArith, 102 SingleEq, 264 GSM8K, 51 AQuA, and 200 SVAMP records. These are the final 20% suffixes except for 11 duplicated MultiArith questions that also occur in its first 80%. This distinction must be preserved when reporting unseen-data performance.

The earlier contaminated Math10K training set is intentionally not distributed. It overlaps with AddSub, MultiArith, and SingleEq. See the data READMEs for file-level provenance and construction details.

Audit a training file, or materialize benchmark subsets that are disjoint by normalized instruction:

```bash
supertuning-data-audit --train_data fine_tuning/ft-training_set/math_17k.json
supertuning-data-audit \
  --train_data fine_tuning/ft-training_set/math_17k.json \
  --write_heldout_dir runs/math17k-heldout
```

Run the experiment pipeline or a saved-checkpoint evaluation on those disjoint subsets with:

```bash
supertuning-math --config configs/math17k/llama-1b-1epoch.json --dataset_dir runs/math17k-heldout
supertuning-eval --checkpoint <checkpoint> --dataset_dir runs/math17k-heldout
```

Custom training or calibration data can be supplied as a JSON list with `instruction`, `input`, `output`, and optional `answer` fields.

## Efficiency profiling

The fixed-step profiler writes both JSONL and CSV and measures trainable parameters, checkpoint size, calibration time, peak allocated/reserved GPU memory, measured optimizer-state storage, steps/s, tokens/s, and wall-clock time:

```bash
supertuning-profile \
  --models meta-llama/Llama-3.2-1B \
  --methods full,rosa,sift-topk,lora,super-wanda-bottom,magnitude-bottomk,supra-0.8-bottom,supra-magnitude-0.3 \
  --profile_steps 20 \
  --output_dir runs/efficiency
```

These are implementation-level measurements. Super and Supra store sparse trainable vectors and sparse optimizer states, but the current layer still constructs a dense effective weight and computes a dense gradient matrix. Matched trainable-scalar counts therefore do not imply sparse-kernel wall-clock speedups.

## Repository layout

```text
configs/math17k/            Reproducible paper presets
fine_tuning/                Training, LR selection, evaluation, and profiling
fine_tuning/baselines/      Baseline integrations
fine_tuning/rosa/           RoSA integration
fine_tuning/dataset/        Six arithmetic test sets
fine_tuning/ft-training_set Training-data snapshots
jobs/                       Generic and Orix Slurm launchers
src/                        Wanda calibration and mask utilities
```

## Citation

Please cite the paper when using Super or Supra. A machine-readable citation is available in [`CITATION.cff`](CITATION.cff).

```bibtex
@article{ilin2026super,
  title={Super-Tuning: From Activation-Aware Pruning to Sparse Fine-Tuning},
  author={Ilin, Ivan and Zmushko, Philip and Richt{\'a}rik, Peter},
  journal={arXiv preprint arXiv:2607.09287},
  year={2026}
}
```

## License and attribution

Original Super-Tuning code is released under the MIT license. RoSA-derived and SparseGPT-derived files remain subject to Apache-2.0 and retain their upstream notices. The SIFT baseline is an in-repository implementation of the published algorithm rather than vendored upstream source. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for details.
