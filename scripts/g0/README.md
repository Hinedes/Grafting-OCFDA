# G0 fixed-prefix NLL

G0 checks that a code path reproduces the frozen fixed-prefix NLL of the B1
evidence. It runs the gold-continuation NLL for the aligned-2001 and
independent-2001 endpoints on the same 805 held-out sequences. It does not
train and does not generate.

The frozen values are:

| Endpoint | NLL |
|---|---:|
| aligned-2001 | 0.29259259259259257 |
| independent-2001 | 0.3300037411148522 |

Source: `exact_interventions.json`, sha256
`443d1fc8e7d4b000a1c124159c01c791e4e00e11cd4518d9e81d8f5c33d0c090`. The
values are the `nll_nl` entries for the two endpoints. The earlier
`threeviews.json` values are from a different run and do not match.

## Inputs

- Endpoints: `Llama-3_2-1B_r8_ocfda-aligned_lr0p0005_seed2001_support1001` and
  `Llama-3_2-1B_r8_ocfda-independent_lr0p0005_seed2001_support1001` under an
  endpoints directory. Each has `supertuning_config.json` and
  `model.safetensors`.
- Held-out set: the six `test.json` files under `heldout_dataset`, 805
  sequences in total. Not in this repository.
- Prefix manifest: `fixed_prefixes_manifest.jsonl`, 805 entries, sha256
  `b1b77eba71f9177db107dafbb85af08754b2979170f9ce4098a95b134b6c18ba`. Not in
  this repository.
- Host: `meta-llama/Llama-3.2-1B`, revision
  `5d853ed7d16ac794afa8f5c9c7f59f4e9c950954`, bf16, eager attention, from
  `HF_HOME`.

The measurement uses `fixedprefix_whole.build_fixed_batches`: batch 16, max
length 256, left padding, pad token 0, `add_special_tokens=False`. NLL is
shifted cross-entropy on 53,460 response-token labels.

## Run

These are the commands used for the G0 check. The first loads the endpoints
through the `b1-evidence-v1` model code. The second loads them through
`grafting.attach_graft`.

```sh
export HF_HOME=/home/iclab/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/iclab/grafting-framework
export PATH=/home/iclab/grafting-gpu-venv/bin:$PATH

PYTHONPATH=/home/iclab/grafting-g0-tag:$PWD/scripts/g0 \
python scripts/g0/run_g0.py --mode tag \
  --endpoints-dir /home/iclab/grafting-evidence/endpoints_src \
  --heldout-dir /home/iclab/grafting-evidence-extra/measurement/heldout_dataset \
  --prefix-manifest /home/iclab/grafting-evidence-extra/measurement/fixed_prefixes_manifest.jsonl

PYTHONPATH=$PWD:$PWD/scripts/g0 \
python scripts/g0/run_g0.py --mode readme \
  --endpoints-dir /home/iclab/grafting-evidence/endpoints_src \
  --heldout-dir /home/iclab/grafting-evidence-extra/measurement/heldout_dataset \
  --prefix-manifest /home/iclab/grafting-evidence-extra/measurement/fixed_prefixes_manifest.jsonl
```

Both commands print the two NLLs and `G0_PASS`. They also check the endpoint
canonical hashes against the frozen values.

The endpoints and data are not in this repository.
