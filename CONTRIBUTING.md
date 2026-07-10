# Contributing

Bug reports and focused pull requests are welcome. Before opening an issue, please search existing issues and verify the problem on the latest `main` branch.

Include the following in reproducibility reports:

- exact command or JSON preset
- commit hash
- Python, PyTorch, Transformers, PEFT, and CUDA versions
- model name and GPU type
- relevant traceback and the corresponding worker log

Set up a development environment and run the local checks with:

```bash
python -m pip install -e ".[dev]"
python -m py_compile \
  fine_tuning/finetune.py \
  fine_tuning/math_experiment_tables.py \
  fine_tuning/launch_math_methods.py
pytest
```

Please keep changes scoped, add CPU tests for mask/budget logic, and avoid committing model checkpoints, W&B directories, caches, or generated tables.
