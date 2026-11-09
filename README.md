# Grafting

Grafting adds a small trainable path to a frozen language model. The host
weights never change; the trained part is a separate set of tensors you can
switch off, keep on disk, or replace.

PyTorch, transformers, safetensors. Dense SwiGLU models with `gate_proj`,
`up_proj`, and `down_proj` in each decoder layer.

## What it's for

A team runs Qwen2.5-0.5B behind an arithmetic helper. The model is fine for
chat but unreliable on "what is 47 + 38?". Fine-tuning the whole model would
change the weights they already validated and make every experiment a new
model.

They train a Graft on a small set of arithmetic prompts and attach it once
with `attach_graft`. The host stays Qwen2.5-0.5B. For a request that doesn't
need arithmetic, `detach_graft` turns the Graft off and the model answers from
the host alone. `enable_graft` turns it back on for the next request. Neither
call reloads anything.

The Grafts live in `~/grafts`. `grafting list` shows them, and `swap_graft`
moves between two Grafts trained with the same support layout without
reloading the host. An artifact trained for Qwen2.5-0.5B is refused on
Llama-3.2-1B before it is attached. The error names the artifact model and
both tensor counts.

It won't merge the delta into the host weights, and it won't load onto a host
whose tensors differ. An installed Graft changes the model's answers on every
prompt, including prompts it wasn't trained for. Detaching removes that change
exactly.

## How a Graft sits in the model

A Graft lives on one set of intermediate coordinates per layer. `S` is a list
of `k` indices. In aligned geometry the same `S` feeds gate, up, and down.
`E_S` selects those rows from `gate_proj` and `up_proj`, and those columns from
`down_proj`. The host matrices stay frozen.

```
                        x
                        |
          +-------------+-------------+
          |                           |
   gate_proj (frozen)          up_proj (frozen)
          |                           |
    +-----+-----+               +-----+-----+
    |           |               |           |
 host rows   Graft rows      host rows   Graft rows
    |         E_S G               |         E_S U
    |           |                 |           |
    |        [cut]                |        [cut]
    |           |                 |           |
    +-----+-----+                 +-----+-----+
          |                           |
          +------------+--------------+
                       |
                    SwiGLU
                       |
                down_proj (frozen)
                       |
                +------+------+
                |             |
          host columns   Graft columns
                         D E_S^T
                          [cut]
                |             |
                +------+------+
                       |
                       y
```

`detach_graft()` stops adding `E_S G`, `E_S U`, and `D E_S^T`. The frozen host
branches stay connected, so the output is the plain host output.

### What happens on attach

1. `attach_graft` reads `graft_manifest.json`. It needs `supports`, `k`, and
   `geometry`; if any is missing, it stops.
2. It checks host identity: the manifest's `base_model` against the host's
   model ID, and the exact set of host tensor names. A mismatch stops with
   the artifact model, the host model when known, and both tensor counts. The
   host is left without this Graft, and no host weight is written.
3. It casts an unwrapped host to the manifest's `dtype` if needed, then
   computes SHA-256 for each host tensor and compares it with
   `host_tensor_hashes`. A changed value stops with the tensor name. Wrappers
   are not built.
4. It builds the OCFDA wrappers from the recorded `supports` and `k`. It does
   not draw a new support.
5. `load_graft` checks the Graft tensor names, shapes, and support buffers,
   then copies `graft.safetensors` into the wrappers. `attach_graft` enables
   the Graft.
6. The same model object comes back. `attach_graft` does not reload the host.

A failure at any step returns before the new Graft is enabled. If the host was
already wrapped, it keeps its current Graft.

## Guarantees

Detached output is bitwise equal to the plain host. `detach_graft` calls
`set_graft_enabled(model, False)`, which sets `graft_enabled` to `False` on
every OCFDA wrapper; it does not copy or restore weights.

Training and attach leave every host weight unchanged. The optimizer only sees
`graft_delta` parameters.

A Graft trained for a different host is refused before it is attached, and the
host is left without that Graft.

A failed `swap_graft` leaves the active Graft, its tensors, and its enabled
state unchanged.

Attaching a Graft reproduces the outputs of the model that trained it,
bitwise.

These are checked in `tests/test_graft_manage.py`
(`test_detach_graft_matches_unwrapped_host`,
`test_attach_graft_reproduces_trained_outputs`,
`test_enable_graft_restores_grafted_outputs`,
`test_swap_graft_refuses_other_host_and_preserves_state`,
`test_swap_graft_matches_fresh_attach`, `test_cross_host_refusal_names_models_and_counts`),
`tests/test_ocfda.py`
(`test_pretrained_tensors_are_bit_identical_after_an_adamw_step`,
`test_graft_parameters_learn_and_detach_restores_the_host`,
`test_zero_graft_is_bit_exact_noop_and_host_stays_frozen`), and
`tests/test_graft_artifact.py`
(`test_load_refuses_modified_host_tensor_and_names_it`).

## Run it

Python 3.10 or newer and the dependencies in `pyproject.toml`. The train
commands and Python loads below run on CPU and download Qwen2.5-0.5B into
`HF_HOME` on first use. For an existing cache, add `--local-files-only` to
`grafting train` and `local_files_only=True` to the Python loads to run
offline. `grafting verify` always reads the cache.

```sh
git clone https://github.com/Hinedes/Grafting-OCFDA.git
cd Grafting-OCFDA
python -m pip install -e .
```

Write a small arithmetic set. Any JSON list with `instruction` and `output`
works; this one has 400 rows and the run uses the first 64.

```sh
python - <<'PY'
import json
import os

rows = [
    {"instruction": f"What is {a} + {b}?", "input": "", "output": str(a + b)}
    for a in range(20)
    for b in range(20)
]
path = os.path.expanduser("~/grafts/arithmetic.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(rows, handle)
PY
```

Train two Grafts with the same support and `k`, so one can replace the other.
Ten steps is a smoke run, not a useful arithmetic model.

```sh
COMMON="--model Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --train-data $HOME/grafts/arithmetic.json --k 8 --geometry aligned --support-seed 1001 --steps 10 --batch-size 2 --max-examples 64 --max-length 64 --loss-on response --dtype fp32 --device cpu"
grafting train $COMMON --output-dir ~/grafts/arithmetic-a --seed 2001
grafting train $COMMON --output-dir ~/grafts/arithmetic-b --seed 2002
```

```sh
grafting list ~/grafts
grafting inspect ~/grafts/arithmetic-a
grafting verify ~/grafts/arithmetic-a --model Qwen/Qwen2.5-0.5B
```

The Python API uses the same artifacts. The asserts check that detach and swap
change the logits while enable restores the first Graft.

```python
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from grafting import attach_graft, detach_graft, enable_graft, swap_graft

model_id = "Qwen/Qwen2.5-0.5B"
revision = "060db6499f32faf8b98477b0a26969ef7d8b9987"
grafts = Path.home() / "grafts"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, torch_dtype=torch.float32
)
model.eval()

prompt = "### Instruction:\nWhat is 17 + 25?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt")

def logits():
    with torch.no_grad():
        return model(**inputs).logits

attach_graft(model, str(grafts / "arithmetic-a"))
attached = logits()

detach_graft(model)
detached = logits()

enable_graft(model)
enabled = logits()

swap_graft(model, str(grafts / "arithmetic-b"))
swapped = logits()

assert torch.equal(attached, enabled)
assert not torch.equal(attached, detached)
assert not torch.equal(attached, swapped)
print("attach/enable match; detach and swap change the output")
```

## The artifact

`grafting train` writes a directory with two files. `graft.safetensors` holds
only `graft_delta` and `graft_support` tensors. `graft_manifest.json` holds
`format_version`, `base_model`, `base_model_revision`, `host_tensor_hashes`,
`supports`, `k`, `geometry`, `seed`, `training_seed`, `dtype`, `provenance`,
and `extra_metadata`. `extra_metadata.training` records the steps, losses,
learning rate, and `loss_on`.

Host identity is the pair (`base_model`, exact host tensor names) plus the
SHA-256 of each host tensor. `host_tensor_hashes` is checked in the manifest's
`dtype`. `base_model_revision` is what `grafting verify` loads when `--model`
is the artifact's recorded model ID; it calls `snapshot_download(revision=...,
local_files_only=True)` instead of following `refs/main`. `attach_graft`
itself does not download anything.

Version 1 artifacts from the B1 pipeline still load: `model.safetensors` (or
`pytorch_model.bin`) plus `supertuning_config.json`.
`test_load_accepts_format_version_1_checkpoint` checks the v1 path.

## Which models work

A dense causal LM with SwiGLU and `gate_proj`, `up_proj`, and `down_proj` in
each decoder layer, under `model.layers` or `model.decoder.layers`. Tested on
Llama-3.2-1B and Qwen2.5-0.5B. MoE layers and GeGLU models are not supported.

## Limits

`swap_graft` needs the same `k` and support layout as the active Graft. Two
artifacts trained with the same `--support-seed` and `--k` qualify. A
different layout is refused and the active Graft stays as it was; load the
other artifact with `attach_graft` on a fresh host instead.

An installed Graft affects every prompt. On Llama-3.2-1B, aligned OCFDA, the
macro score over PIQA, HellaSwag, and ARC-Easy fell 3.54 pp against the frozen
base (3 seeds, `b1-evidence-v1`). `detach_graft` returns the host output
exactly, so that change goes away with it.

Loading the original B1 manifests with `strict_b1=True` depends on
machine-specific paths recorded in those manifests. The standalone v2 path
does not.

## Research results

Math17K, Llama-3.2-1B, 3 seeds, 805 held-out questions. Aligned OCFDA reached
52.10% canonical macro accuracy, the unweighted mean of six benchmark
accuracies; the frozen base scored 9.94%. Aligned support beat independent
support by 4.16 pp on average, in 3 of 3 matched pairs. This is one model/task
setting. It does not establish superiority over LoRA or any other method. The
results are tagged `b1-evidence-v1` in this repository; the full run record is
kept separately.

## Credits and license

The training pipeline and Math17K benchmark come from
[Super-Tuning](https://github.com/vectozavr/SuperTuning), pinned at commit
`3e961f0bb7ca49417f3804d7a61b24af58fab21d`. Its MIT notice is kept in
`LICENSE`. Grafting and OCFDA are by Amantur Saliev (Hinedes). MIT.
