import torch
from pathlib import Path
import os
import json
import time
import numpy as np
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .io import configure, device, csv_write, json_write, plot
from .data import prepare, tensors, network
from .planner import Controller, VARIANTS
from .runtime import GradientReducer
from .metrics import communication_metrics, summarize


def train(config):
    if not 0 <= config.get("warmup", 0) < config["steps"]:
        raise ValueError("Require 0 <= warmup < steps")
    variant = config["variant"]
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    target = device(config)
    if target.type == "cuda":
        torch.cuda.set_device(local)
        target = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl" if target.type == "cuda" else "gloo")
    model = network(config).to(target)
    named = list(reversed(list(model.named_parameters())))
    controller = Controller([p.numel() * p.element_size() for _, p in named], config)
    reducer = None
    if variant == "ddp" and world > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local] if target.type == "cuda" else None,
            bucket_cap_mb=config["bucket_bytes"] / 1e6,
        )
    else:
        reducer = GradientReducer(named, controller, world)
    x, y = tensors(config)
    rng = np.random.default_rng(config["seed"] + rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    history, traces = [], []
    model.train()
    for step in range(config["steps"]):
        indices = rng.choice(
            len(x), config["batch_size"], replace=len(x) < config["batch_size"]
        )
        inputs, labels = x[indices].to(target), y[indices].to(target)
        optimizer.zero_grad(set_to_none=True)
        if target.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if reducer:
            reducer.begin()
        loss = F.cross_entropy(model(inputs), labels)
        loss.backward()
        if reducer:
            readiness, compute, communications = reducer.finish()
        else:
            readiness, compute, communications = [], 0.0, []
        optimizer.step()
        if target.type == "cuda":
            torch.cuda.synchronize()
        iteration = time.perf_counter() - start
        if world > 1:
            observed = torch.tensor(
                readiness + [iteration, compute], device=target, dtype=torch.float64
            )
            dist.all_reduce(observed, op=dist.ReduceOp.MAX)
            values = observed.cpu().tolist()
            readiness, iteration, compute = values[:-2], values[-2], values[-1]
        row = {
            "variant": variant,
            "step": step + 1,
            "loss": loss.item(),
            "iteration_s": iteration,
            "training_s": iteration,
            "buckets": len(controller.buckets) if reducer else None,
        }
        if reducer:
            old_layout = [list(bucket) for bucket in controller.buckets]
            controller.observe(readiness, compute, communications, iteration)
            row.update(communication_metrics(communications, readiness, reducer.layout, compute))
            control_start = time.perf_counter()
            if rank == 0 and (step + 1) % config["reconfigure_every"] == 0:
                controller.adapt(iteration)
            if world > 1:
                objects = [controller.buckets if rank == 0 else None]
                dist.broadcast_object_list(
                    objects, src=0, device=target if target.type == "cuda" else None
                )
                controller.buckets = objects[0]
                ordered = [controller.predict()["order"] if rank == 0 else None]
                dist.broadcast_object_list(
                    ordered, src=0, device=target if target.type == "cuda" else None
                )
                controller.predict_order = ordered[0]
            row["controller_us"] = (time.perf_counter() - control_start) * 1e6
            row["reconfigurations_per_100"] = 100 * int(old_layout != controller.buckets)
            traces.append(
                {
                    "step": step + 1,
                    "names": [n for n, _ in named],
                    "sizes": controller.sizes,
                    "readiness_s": readiness,
                    "compute_s": compute,
                    "communications": communications,
                    "world_size": world,
                }
            )
        row["iteration_s"] = time.perf_counter() - start
        if world > 1:
            duration = torch.tensor(
                row["iteration_s"], device=target, dtype=torch.float64
            )
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            row["iteration_s"] = duration.item()
        row["samples_per_s"] = world * config["batch_size"] / row["iteration_s"]
        history.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
    if reducer:
        reducer.close()
    if rank == 0:
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": (
                    model.module
                    if isinstance(model, DistributedDataParallel)
                    else model
                ).state_dict(),
                "config": config,
            },
            output / "model.pt",
        )
        csv_write(output / "training.csv", history)
        json_write(output / "config.json", config)
        with (output / "trace.jsonl").open("w", encoding="utf-8") as stream:
            for trace in traces:
                stream.write(json.dumps(trace) + "\n")
        json_write(
            output / "metrics.json",
            {
                "variant": variant,
                "world_size": world,
                "mean_iteration_s": float(
                    np.mean([row["iteration_s"] for row in history[config.get("warmup", 0):]])
                ),
                "p99_iteration_s": float(
                    np.quantile([row["iteration_s"] for row in history[config.get("warmup", 0):]], 0.99)
                ),
                "final_loss": history[-1]["loss"],
                "measurement": "cuda_events" if target.type == "cuda" else "host_observed",
                "warmup": config.get("warmup", 0),
                "summary": summarize(history[config.get("warmup", 0):]),
            },
        )
    if world > 1:
        dist.destroy_process_group()


def replay(config):
    traces = [
        json.loads(line)
        for line in Path(config["trace"]).read_text().splitlines()
        if line.strip()
    ]
    if not traces:
        raise ValueError("Trace has no iterations")
    rows = []
    for variant in config["variants"]:
        if variant == "ddp":
            continue
        controller = Controller(traces[0]["sizes"], dict(config, variant=variant))
        for trace in traces:
            controller.observe(
                trace["readiness_s"],
                trace["compute_s"],
                trace["communications"],
                trace["compute_s"],
            )
            if trace["step"] % config["reconfigure_every"] == 0:
                controller.adapt(trace["compute_s"])
            if variant == "no_overlap":
                value = controller.predict()
                value["iteration_s"] = trace["compute_s"] + sum(
                    r["finish"] - r["start"] for r in value["records"]
                )
                value["exposed_s"] = value["iteration_s"] - trace["compute_s"]
                value["overlap"] = 0.0
            else:
                value = controller.predict()
            rows.append(
                {
                    "variant": variant,
                    "step": trace["step"],
                    "measurement": "trace_simulation",
                    **{
                        key: value[key]
                        for key in [
                            "iteration_s",
                            "exposed_s",
                            "overlap",
                            "collectives",
                            "bytes",
                        ]
                    },
                }
            )
    csv_write(Path(config["output"]) / "comparison.csv", rows)
    json_write(Path(config["output"]) / "config.json", config)


@torch.no_grad()
def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model = network(state["config"]).to(device(config)).eval()
    model.load_state_dict(state["model"])
    x, y = tensors(config, "test")
    correct = 0
    for offset in range(0, len(x), config["batch_size"]):
        predictions = (
            model(x[offset : offset + config["batch_size"]].to(device(config)))
            .argmax(1)
            .cpu()
        )
        correct += (
            (predictions == y[offset : offset + config["batch_size"]]).sum().item()
        )
    json_write(
        Path(config["output"]) / "evaluation.json",
        {"accuracy": 100 * correct / len(y), "samples": len(y)},
    )


def main():
    command, config = configure(["prepare", "train", "evaluate", "compare", "summarize", "plot"])
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        replay(config)
    elif command == "summarize":
        records = [json.loads(line) for line in Path(config["trace"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        from .metrics import paired_summary
        json_write(Path(config["output"]) / "summary.json", paired_summary(records))
    else:
        root = Path(config["output"])
        path = (
            root / "comparison.csv"
            if (root / "comparison.csv").exists()
            else root / "training.csv"
        )
        plot(path, root / "iteration_time.pdf", "step", "iteration_s")
