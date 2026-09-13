# Grafting / OCFDA

This repository contains the experimental implementation of **Grafting** using **Ownership-Constrained FFN Delta Adaptation (OCFDA)**.

## Why Super-Tuning is here

The training and evaluation pipeline comes from [Super-Tuning](https://github.com/vectozavr/SuperTuning), pinned from upstream commit `3e961f0bb7ca49417f3804d7a61b24af58fab21d`.

Super-Tuning already provides the Math17K pipeline, LoRA and sparse-tuning baselines, parameter-budget accounting, learning-rate (LR) selection, evaluation, and checkpoint handling. OCFDA uses the same pipeline so the comparisons do not depend on a separate benchmark implementation.

We retain the upstream MIT license and third-party notices. Super-Tuning itself is not part of the claimed Grafting contribution.

## What Grafting is

Grafting treats adaptation as separately owned state attached to an immutable host model.

$$
W_{\mathrm{eff}} = W_0 + \Delta W
$$

The host weights $W_0$ remain frozen. The optimizer writes only to Graft-owned parameters.

OCFDA applies this to SwiGLU FFNs. OCFDA treats the matching Gate row, Up row, and Down column as one intermediate coordinate:

$$
\{W_{gate,i:},\ W_{up,i:},\ W_{down,:i}\}
$$

For a selected support $S$ (the set of trained coordinates):

$$
\Delta W_g = E_S G,\qquad
\Delta W_u = E_S U,\qquad
\Delta W_d = D E_S^\top
$$

The experiment compares two support modes with the same number of trainable values. In aligned mode, one support $S$ is drawn per layer and reused for Gate, Up, and Down. In independent mode, each projection draws its own support of the same size.

## Results

All results below use Llama-3.2-1B, Math17K training, and 805 evaluation questions that do not appear in training. Macro accuracy is the unweighted mean of the six benchmark accuracies.

### Strict ownership

With the host unchanged bit for bit and the optimizer restricted to Graft state, aligned OCFDA reached 46.34–55.57% macro accuracy across three new-seed runs. The frozen base scored 9.94%.

The 1B configuration uses 5,603,328 trainable Graft values.

### Aligned vs independent support

| Pair | Aligned | Independent | Δ |
|---|---:|---:|---:|
| 1001 / 2001 | 46.34 | 37.53 | +8.81 |
| 1002 / 2002 | 54.40 | 52.96 | +1.44 |
| 1003 / 2003 | 55.57 | 53.35 | +2.22 |

Mean aligned advantage:

$$
+4.16\text{ percentage points}
$$

Aligned won all three matched pairs.

### Competitive gate

Learning rates were selected by held-out Math17K validation NLL (negative log-likelihood) before final evaluation.

| Method | Three-seed mean macro accuracy |
|---|---:|
| **OCFDA aligned** | **52.10** |
| Super-BottomK | 47.95 |
| LoRA | 45.69 |

Under this protocol, OCFDA finished +4.15 pp above Super-BottomK and +6.42 pp above LoRA.

The evidence is limited to one model/task setting with three final replicates per method. Seed variance is large, so these results do not establish general superiority over LoRA or Super-Tuning.

## Next

The next block repeats the comparison on Llama-3.2-3B while keeping the task and evaluation protocol fixed.
