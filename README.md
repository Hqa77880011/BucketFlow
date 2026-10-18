# BucketFlow

Implementation of **BucketFlow: Online Gradient-Readiness-Aware Collective Scheduling for Communication-Computation Overlap**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`configs/small.json` generates a seeded classification workload with labels from a fixed linear rule. For downloaded [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), use `configs/cifar10.json` consistently instead. `prepare` writes the training and test arrays.

```bash
python -m bucketflow prepare --config configs/small.json
```

## Training and execution

`configs/small.json` fixes seed `2026`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON. Two CPU workers use Gloo with the small config; distributed GPU runs require a matching Linux/CUDA environment.

```bash
torchrun --standalone --nproc_per_node=2 -m bucketflow train --config configs/small.json --output outputs/bucketflow
```

## Evaluation

```bash
python -m bucketflow evaluate --config configs/small.json --checkpoint outputs/bucketflow/model.pt --output outputs/bucketflow
```

## Comparisons and ablations

`compare` replays the saved readiness trace and labels results `trace_simulation`.

```bash
python -m bucketflow compare --config configs/small.json --trace outputs/bucketflow/trace.jsonl --output outputs/comparison
python -m bucketflow plot --config configs/small.json --output outputs/comparison
```

## Outputs

Run directories contain `evaluation.json`, `model.pt`, `training.csv`, `metrics.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
