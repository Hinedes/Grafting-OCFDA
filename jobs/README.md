# Slurm jobs

`math17k.slurm` is cluster-neutral: choose the partition, QoS, account, and GPU type at submission time. It runs one method per visible GPU and queues additional methods as GPUs become available.

```bash
mkdir -p logs
sbatch \
  --partition=<partition> \
  --qos=<qos> \
  --gpus=nvidia_h200:1 \
  --export=ALL,REPO_DIR=$PWD,CONFIG=configs/math17k/llama-1b-1epoch.json \
  jobs/math17k.slurm
```

Set `GPUS=0,1,2,3` and request four GPUs to run four methods concurrently. Set `METHODS` or `LRS` to override the selected JSON preset without editing it.

The `orix/` directory contains the exact cluster-specific launchers used for paper development and auxiliary analyses. They are retained for provenance; new runs should normally start from `math17k.slurm`.
