"""Small functional check, including two-rank gradient equivalence on CPU."""

import json
from pathlib import Path
import sys
import tempfile
import uuid
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from bucketflow.cli import train, evaluate, replay
from bucketflow.data import prepare, network, tensors
from bucketflow.io import plot
from bucketflow.metrics import communication_metrics, paired_summary
from bucketflow.planner import Controller, schedule
from bucketflow.runtime import GradientReducer


def check_reducer(rank, store, config):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2)
    model, reference = network(config), network(config)
    generator = torch.Generator().manual_seed(71)
    x = torch.randn(12, config["input_dim"], generator=generator)
    y = torch.randint(config["classes"], (12,), generator=generator)
    named = list(reversed(list(model.named_parameters())))
    controller = Controller([p.numel() * p.element_size() for _, p in named], config)
    reducer = GradientReducer(named, controller, 2)
    reducer.begin()
    F.cross_entropy(model(x[rank * 6:(rank + 1) * 6]), y[rank * 6:(rank + 1) * 6]).backward()
    reducer.finish()
    F.cross_entropy(reference(x), y).backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)
    reducer.close()
    dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/smoke").resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("configs/small.json").read_text())
    config.update(data=str(output / "data"), output=str(output), steps=4,
                  batch_size=8, samples=64, hidden_dim=16, input_dim=8,
                  warmup=1, bucket_bytes=512, min_bucket_bytes=4)
    prepare(config)
    initial = network(config).state_dict()
    train(config)
    saved = torch.load(output / "model.pt", weights_only=False)
    assert all(torch.isfinite(p).all() and not torch.equal(p, initial[n])
               for n, p in saved["model"].items())
    model = network(config).eval()
    model.load_state_dict(saved["model"])
    other = network(config).eval()
    other.load_state_dict(torch.load(output / "model.pt", weights_only=False)["model"])
    x, _ = tensors(config, "test")
    with torch.no_grad():
        torch.testing.assert_close(model(x), other(x), atol=1e-7, rtol=1e-6)
    config.update(checkpoint=str(output / "model.pt"), trace=str(output / "trace.jsonl"))
    evaluate(config)
    replay(config)
    plot(output / "comparison.csv", output / "comparison.pdf", "step", "iteration_s")
    result = schedule([[0], [1]], [2, 2], [0, 1], 2, 0, 1)
    assert result["iteration_s"] == 4 and result["exposed_s"] == 2 and result["overlap"] == .5
    measured = communication_metrics([
        {"bucket": 1, "bytes": 2, "launch_s": 1, "finish_s": 3, "duration_s": 2},
        {"bucket": 0, "bytes": 2, "launch_s": 3, "finish_s": 5, "duration_s": 2},
    ], [0, 1], [[0], [1]], 3)
    assert measured["overlap"] == .5 and measured["queue_delay_s"] == 1.5
    records = [dict(workload="manual", variant=v, world_size=2, run_id=1,
                    measurement="manual", iteration_s=t, samples_per_s=12 / t,
                    exposed_s=e) for v, t, e in [("ddp", 4, 2), ("bucketflow", 2, 1)]]
    assert paired_summary(records)[1]["metrics"]["throughput_gain"]["mean"] == 1
    # PyTorch's Windows FileStore cannot open a non-ASCII pathname.
    store = Path(tempfile.gettempdir()) / ("bucketflow-store-" + uuid.uuid4().hex)
    mp.spawn(check_reducer, args=(store.as_uri(), config), nprocs=2, join=True)
    print("PASS: data, 4 training steps, all parameter updates, reload, evaluation, replay, hand metrics, 2-rank exact gradients")


if __name__ == "__main__":
    main()
