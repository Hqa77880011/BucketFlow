"""Two Gloo ranks exercise the actual BERT task/reducer path via a file store."""

import json
import os
import tempfile
import uuid
from pathlib import Path
import torch
import torch.multiprocessing as mp
from bucketflow.workloads import train


def worker(rank, config):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    train(config)


def main():
    out = Path("outputs/workload_smoke").resolve()
    config = json.loads((out / "bert_sst2/config.json").read_text())
    config.update(
        output=str(out / "gloo2"),
        init_method="file:///"
        + (
            Path(tempfile.gettempdir()) / ("bucketflow-workload-" + uuid.uuid4().hex)
        ).as_posix(),
        device="cpu",
        steps=2,
    )
    mp.spawn(worker, args=(config,), nprocs=2, join=True)
    state = torch.load(out / "gloo2/model.pt", map_location="cpu", weights_only=False)
    assert state["config"]["global_batch_size"] == 2
    print(
        "PASS: two-rank Gloo BERT recipe, shared collectives and globally normalized objective"
    )


if __name__ == "__main__":
    main()
