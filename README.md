# BucketFlow

Implementation of **BucketFlow: Online Gradient-Readiness-Aware Collective Scheduling for Communication-Computation Overlap**. A runtime records parameter readiness, adapts gradient bucket boundaries and executes exact all-reduces in a globally agreed order.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Data and single-process workflow

`configs/small.json` uses a seeded generated classification workload for inspecting the execution protocol with little compute. Labels come from a fixed linear decision rule; this dataset is a development workload, separate from the paper's vision and language benchmarks. `configs/cifar10.json` downloads the official [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) splits through torchvision and uses a CIFAR-style ResNet-18. `prepare` writes training and test arrays under the configured data directory. CIFAR test labels are used only by `evaluate`.

```bash
python -m bucketflow prepare --config configs/small.json
python -m bucketflow train --config configs/small.json --output outputs/small
python -m bucketflow evaluate --config configs/small.json --checkpoint outputs/small/model.pt --output outputs/small
python -m bucketflow compare --config configs/small.json --trace outputs/small/trace.jsonl --output outputs/replay
python -m bucketflow plot --config configs/small.json --output outputs/replay
```

A single process exercises packing and the optimizer boundary. It has no cross-worker communication and cannot measure communication overlap improvements.

## Distributed execution

Prepare data before launching workers. For two NVIDIA GPUs on Linux:

```bash
python -m bucketflow prepare --config configs/cifar10.json
torchrun --standalone --nproc-per-node=2 -m bucketflow train --config configs/cifar10.json --output outputs/bucketflow
torchrun --standalone --nproc-per-node=2 -m bucketflow train --config configs/cifar10.json --variant ddp --output outputs/ddp
torchrun --standalone --nproc-per-node=2 -m bucketflow train --config configs/cifar10.json --variant static --output outputs/static
python -m bucketflow evaluate --config configs/cifar10.json --checkpoint outputs/bucketflow/model.pt --output outputs/bucketflow
```

CUDA workers use NCCL, one device per local rank. CPU workers use Gloo; a two-process CPU run can use the small configuration. All ranks must use the same model, step count and batch size. Every trainable parameter must contribute to every backward pass and all parameters must use one dtype.

## Controller and ablations

| Variant | Execution |
| --- | --- |
| `ddp` | Native PyTorch DDP reducer under `torchrun` |
| `static` | Fixed bucket layout with parameter readiness hooks |
| `no_overlap` | Wait for backward before launching all-reduces |
| `bucketflow` | Online split/merge, readiness priority and stability controls |
| `no_split`, `no_merge` | Disable one boundary operation |
| `no_priority` | Launch ready candidates in logical index order |
| `no_stability` | Remove hysteresis, cooldown and rollback observation |

The `compare` command replays a measured readiness trace through the controller. Its output is tagged `trace_simulation`; it does not replace distributed timing runs. Native DDP requires an actual training run rather than replaying its internally different bucket layout.

Commands read the JSON configuration. `--data`, `--output`, `--device`, `--variant`, `--steps`, `--batch-size`, `--seed`, `--limit`, `--trace` and `--checkpoint` override run settings. Other fields use `--set KEY=JSON`, for example `--set gain_s=0.0001`. String values need JSON quotes. Use separate output directories for policies.

`bucket_bytes` is the initial and merge size bound. A single parameter larger than the bound remains indivisible. `min_bucket_bytes`, `max_buckets`, `startup_s` and `bandwidth_bytes_s` constrain candidate costs. `gain_s` and `copy_s` set the minimum predicted benefit and packing penalty. `reconfigure_every` controls the observation epoch; `cooldown` is counted in iterations. The readiness EMA clips outliers using three median absolute deviations. Communication timings fit startup and bandwidth estimates. Two repeated proposals are required before a stable layout change.

## Outputs and timing

`training.csv` records end-to-end iteration time including control and layout exchange, the separate forward/backward/optimizer time (`training_s`), loss, completed samples per second, bucket count and controller cost. `trace.jsonl` contains parameter names, byte sizes, readiness, compute time and observed collective timings. `model.pt` saves model weights. `metrics.json` summarizes measured iteration times and loss; `evaluation.json` contains test accuracy. Replay writes `comparison.csv`, including exposed communication, overlap ratio, collective count and unchanged communicated bytes. `iteration_time.pdf` plots the corresponding measured or simulated series.

## Implementation conventions

Post-accumulate-gradient hooks detect readiness before a complete bucket exists. CUDA events connect packing and communication streams; all completed buffers are copied back before the optimizer update. Rank zero broadcasts the next layout and launch order, so local timing differences cannot reorder collectives across ranks. Rank averaging preserves equal-sized rank batches. Parameters use a fixed reverse-registration logical order.

The planner uses a single communication-server DAG and analytic completion slack for critical-path priority. Split and merge decisions score that schedule directly. CUDA events measure device intervals, while CPU asynchronous durations include host queueing and wait time. Readiness is measured from the training iteration's forward/backward start. There is no gradient compression or change to sample membership. The supplied workloads cover CIFAR classification and the small protocol workload; the paper's other training architectures and cluster/network configurations require their own measured traces.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.
