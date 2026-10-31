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

Native DDP uses `train --variant ddp`; set `bucket_bytes=25000000`, `1000000`, or `100000000` with `--set` for the fixed-cap controls. `--variant static --set bucket_bytes=1` gives one collective per parameter. `no_split`, `no_merge`, `no_priority`, `no_stability`, `no_prediction` (zero readiness estimates), and `no_network` (frozen startup/bandwidth estimates) are executable variants. Add them to the JSON `variants` list to include them in replay. Replay is a serialized communication-server model, not a device latency measurement. Its queueing comes from ready-to-launch waiting; it does not fit an additional message-class queue-delay regressor.

For timed runs, set `--steps 230 --set warmup=30`. Use the same global batch, batches, precision and seeds across controls. `readiness_ema` defaults to `0.35`; the small workload and architecture differ from the paper's four cluster workloads. `GradientReducer` is the integration entry for an existing training loop; those ImageNet/SST-2/WikiText training recipes and the eight-A100 setup have not been validated here.

Independent complete-run exports can be aggregated with:

```bash
python -m bucketflow summarize --config configs/small.json --trace data/runs.jsonl --output outputs/aggregate
```

Each input JSONL object needs `workload`, `variant`, `world_size`, `run_id`, `measurement`, `iteration_s`, and `samples_per_s`; optional numeric values include `exposed_s`, `accuracy`, and `perplexity` from the corresponding task evaluator. Supply complete-run observations, not individual iterations. Pair DDP and method records by workload, worker count, run ID and measurement. The summary computes paired throughput gain `T_ddp/T_method-1`, exposed-time reduction, normalized throughput and scaling efficiency `Q_N/(N Q_1)` when the matching observations exist. Means and sample SD are across runs; `ci95_normal` is explicitly a normal-approximation interval and is null for a single run. Hardware and simulation records remain separate.

## Outputs

Run directories contain `evaluation.json`, `model.pt`, `training.csv`, `metrics.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files.

## Verification

Run the small protocol check after installation:

```bash
python tests/smoke.py
```

It covers generated-data I/O, four SGD steps and updates to all model tensors, checkpoint reload, evaluation, replay/plotting, hand-computed overlap/queue/gain examples, and two Gloo ranks against the gradient of the combined batch. Generated samples are functional fixtures, not experimental evidence. Run outputs stay under ignored `outputs/smoke/`.

The existing training, planner and data modules predate this update. This update adds measured communication fields, the two missing controller controls, run aggregation and numerical checks. It does not reconstruct the paper's original experiment records. On CPU, communication completion is observed when the host waits for a work handle; `host_observed` times can overestimate true device completion. CUDA timing uses stream events. Timing conclusions require measurements on the target distributed hardware.

| Paper module or metric | Code entry | Configuration or command |
|---|---|---|
| Robust readiness / split / merge / priority / fallback | `planner.Controller` | `variant`, `readiness_ema`, `gain_s`, `cooldown`, `bandwidth_cv` |
| Exact world-size-normalized all-reduce and shared ordering | `runtime.GradientReducer`, `cli.train` | `torchrun ... train` |
| Iteration time, throughput, controller cost, reconfigurations | `cli.train` | `training.csv`, warm-up filtered `metrics.json` |
| Exposed time, overlap, queue delay, bytes, collective count | `metrics.communication_metrics`, `planner.schedule` | `train` (observed) / `compare` (modeled) |
| Paired throughput/exposed gain, scaling, seed aggregation | `metrics.paired_summary` | `summarize --trace data/runs.jsonl` |
| Classification quality | `cli.evaluate` | `evaluate --checkpoint ...` |

Actual verification environment: Windows, Python 3.13.5, NumPy 2.1.3 (OpenBLAS wheel), PyTorch 2.7.1+cu118, torchvision 0.22.1 and matplotlib 3.10.0. The smoke workload is CPU/Gloo; full training, multi-node NCCL and the four reported cluster workload results are not verified by it.

The GitHub Actions package workflow also executes `python tests/smoke.py` with one CPU thread to check behavior after installation.
