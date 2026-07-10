# Arithmetic evaluation data

The paper evaluates exact-answer accuracy on six fixed test snapshots:

| Directory | Examples | Answer type |
| --- | ---: | --- |
| `AddSub/` | 395 | Numeric |
| `MultiArith/` | 600 | Numeric |
| `SingleEq/` | 508 | Numeric |
| `gsm8k/` | 1,319 | Numeric |
| `AQuA/` | 254 | Multiple choice |
| `SVAMP/` | 1,000 | Numeric |

Each directory contains only the `test.json` snapshot consumed by `fine_tuning/evaluate.py`. Records use the common `instruction`, `input`, `output`, and `answer` schema.

These six files are byte-for-byte identical to the corresponding snapshots published by [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), whose repository is Apache-2.0 licensed. Dataset copyrights and licensing terms remain with their original authors; the root MIT license applies to original Super-Tuning software, not to third-party dataset content.

Math17K includes records from the first 80% of every file above. A clean held-out evaluation must filter every record whose normalized instruction occurs in Math17K. This yields 79 AddSub, 109 MultiArith, 102 SingleEq, 264 GSM8K, 51 AQuA, and 200 SVAMP examples. MultiArith has 11 duplicated questions across its nominal 80/20 boundary, so simply taking its final 20% is insufficient.
