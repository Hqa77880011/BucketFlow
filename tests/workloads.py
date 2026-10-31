"""Four task recipe checks with file-backed tiny fixtures, not paper scores."""

from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from bucketflow.workloads import (
    prepare,
    task_data,
    build_model,
    task_loss,
    train,
    evaluate,
    WORKLOADS,
)
from bucketflow.planner import QueueDelayRegressor, schedule, Controller


def main():
    torch.set_num_threads(1)
    out = Path("outputs/workload_smoke").resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(71)
    image_root = out / "imagenet"
    for split in ("train", "val"):
        for label in ("a", "b"):
            folder = image_root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(4):
                Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)).save(
                    folder / f"{i}.png"
                )
    words = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "good",
        "bad",
        "movie",
        "hello",
        "world",
        "a",
        "b",
    ]
    core = Tokenizer(
        WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    core.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.save_pretrained(out / "tokenizer")
    for name in WORKLOADS:
        cfg = json.loads(Path("configs/workloads", name + ".json").read_text())
        cfg.update(
            tiny=True,
            classes=2,
            image_size=32,
            image_root=str(image_root),
            data=str(out / name / "data"),
            output=str(out / name),
            tokenizer=str(out / "tokenizer"),
            vocab_size=len(words),
            sequence_length=8,
            global_batch_size=2,
            steps=2,
            warmup=0,
            device="cpu",
            workers=0,
            threads=1,
            bucket_bytes=10000000,
        )
        if not name.endswith("imagenet"):
            cfg["source_files"] = {}
            for split in ("train", "eval"):
                rows = [
                    {
                        "sentence": "good movie" if i % 2 else "bad movie",
                        "label": i % 2,
                        "text": "hello world good movie a b " * 4,
                    }
                    for i in range(8)
                ]
                file = out / f"{name}_{split}.jsonl"
                file.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
                cfg["source_files"][split] = str(file)
        prepare(cfg)
        data = task_data(cfg, "train")
        batch = [
            torch.stack(
                [
                    data[i][j]
                    if torch.is_tensor(data[i][j])
                    else torch.tensor(data[i][j])
                    for i in range(2)
                ]
            )
            for j in range(len(data[0]))
        ]
        model = build_model(cfg)
        initial = deepcopy(model.state_dict())
        loss, _, count, _, _ = task_loss(model, batch, name)
        assert torch.isfinite(loss) and count > 0
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        cfg["checkpoint"] = train(cfg)
        state = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
        key = {
            "resnet50_imagenet": "conv1.weight",
            "vit_b16_imagenet": "conv_proj.weight",
            "bert_sst2": "bert.embeddings.word_embeddings.weight",
            "gpt2_wikitext": "transformer.wte.weight",
        }[name]
        assert not torch.equal(initial[key], state["model"][key])
        restored = build_model(cfg, state["model_config"]).eval()
        restored.load_state_dict(state["model"])
        second = build_model(cfg, state["model_config"]).eval()
        second.load_state_dict(state["model"])

        def output(current):
            return (
                current(batch[0])
                if name.endswith("imagenet")
                else current(input_ids=batch[0], attention_mask=batch[1]).logits
            )

        torch.testing.assert_close(output(restored), output(second), rtol=0, atol=0)
        metric = evaluate(cfg)
        assert metric.get("samples", metric.get("token_count", 0)) > 0
        if name == "gpt2_wikitext":
            assert math.isfinite(metric["perplexity"])
        print("PASS recipe:", name, flush=True)
    queue = QueueDelayRegressor(0.5)
    queue.observe(4, 0.01)
    queue.observe(5, 0.03)
    queue.observe(64, 0.2)
    assert (
        math.isclose(queue.predict(4), 0.02)
        and queue.predict(64) == 0.2
        and queue.predict(1024) == 0
    )
    value = schedule([[0]], [4], [0.0], 0.0, 0.001, 1000.0, queue_delays=[0.02])
    assert math.isclose(value["iteration_s"], 0.025)
    config = json.loads(Path("configs/small.json").read_text())
    controller = Controller([4], config)
    controller.observe(
        [0.0], 0.01, [{"bytes": 4, "duration_s": 0.1, "queue_s": 0.03}], 0.1
    )
    assert math.isclose(controller.queue.predict(4), 0.03)
    print(
        "PASS: four data/forward/loss/backward/update/checkpoint/evaluation recipes; hand queue classes and schedule"
    )


if __name__ == "__main__":
    main()
