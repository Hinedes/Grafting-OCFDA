# Third-party notices

The repository's MIT license applies to original Super-Tuning code. The following components retain their upstream terms and attribution.

## Alpaca-LoRA-derived training utilities

`fine_tuning/finetune.py` is adapted from the training and prompt-processing pipelines in [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora) and [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), with substantial changes for the methods and experimental protocol in this repository. Both upstream repositories are licensed under Apache License 2.0.

License copy: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## RoSA and Hugging Face PEFT-derived code

`fine_tuning/rosa/rosa/` is adapted from the official [IST-DASLab RoSA](https://github.com/IST-DASLab/RoSA) and [PEFT-RoSA](https://github.com/IST-DASLab/peft-rosa) implementations. Those repositories are licensed under Apache License 2.0. Several files also retain Hugging Face copyright and Apache-2.0 headers, and `layer.py` retains its Microsoft LoRA notice.

License copy: [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## Wanda and SparseGPT-derived utilities

The activation-statistics and sequential calibration utilities under `src/` build on the public implementations from [locuslab/wanda](https://github.com/locuslab/wanda) and [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt). Wanda is MIT-licensed; SparseGPT is Apache-2.0.

License copies: [`LICENSES/Wanda-MIT.txt`](LICENSES/Wanda-MIT.txt) and [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).

## SIFT

`fine_tuning/baselines/sift.py` is an in-repository implementation of the update rule described in:

> Weixi Song, Zuchao Li, Lefei Zhang, Hai Zhao, and Bo Du. "Sparse is Enough in Fine-tuning Pre-trained Large Language Models." ICML 2024.

The upstream research repository is [song-wx/SIFT](https://github.com/song-wx/SIFT). No upstream SIFT source files are vendored in this repository.

## Datasets

Math7K and the six evaluation JSON files are snapshots published by [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). Math17K includes the upstream Math14K records plus benchmark-derived records used during this project's experiments. Dataset copyrights and terms remain with their respective authors. See the READMEs under `fine_tuning/ft-training_set/` and `fine_tuning/dataset/` before redistributing the data independently of this research artifact.
