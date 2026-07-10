# Fine-tuning data

These are the exact JSON snapshots used during Super-Tuning development.

| File | Records | Use |
| --- | ---: | --- |
| `math_17k.json` | 17,172 | Main Math17K experiments |
| `math_7k.json` | 6,851 | Calibration-source and earlier-scale ablations |

Every record contains `instruction`, `input`, `output`, and `answer` fields. `math_7k.json` is byte-for-byte identical to the decontaminated Math7K snapshot published by [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), whose repository is Apache-2.0 licensed.

Math17K contains all 13,921 Math14K records followed by 3,251 unique records drawn from the first 80% of the six benchmark snapshots under `fine_tuning/dataset/`. The complete benchmark files are therefore not disjoint. A clean held-out evaluation must remove every record whose normalized instruction occurs in Math17K; this additionally removes 11 duplicated MultiArith questions from its final 20% suffix.

The earlier Math10K snapshot is not distributed because it contains overlap with AddSub, MultiArith, and SingleEq.

The files are dataset snapshots rather than original software. Dataset rights and terms remain with their respective creators. Do not assume that the repository's MIT software license relicenses the data.
