"""Executable ImageNet, SST-2 and WikiText workload recipes for GradientReducer."""

from pathlib import Path
import json
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torchvision import datasets, models, transforms
from .io import configure, device, seed, csv_write, json_write
from .planner import Controller, VARIANTS
from .runtime import GradientReducer
from .metrics import communication_metrics, summarize

WORKLOADS = ("resnet50_imagenet", "vit_b16_imagenet", "bert_sst2", "gpt2_wikitext")


def prepare(config):
    """Prepare native task inputs; official held-out labels are evaluation only."""
    output = Path(config["data"])
    output.mkdir(parents=True, exist_ok=True)
    if config["workload"].endswith("imagenet"):
        train = datasets.ImageFolder(Path(config["image_root"]) / "train")
        test = datasets.ImageFolder(Path(config["image_root"]) / "val")
        if train.classes != test.classes:
            raise ValueError(
                "ImageNet train/val class folders must have identical mappings"
            )
        json_write(
            output / "metadata.json",
            {"classes": train.classes, "train": len(train), "eval": len(test)},
        )
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"], revision=config.get("revision", "main")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.get("source_files"):
        splits = {
            split: [
                json.loads(line)
                for line in Path(file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for split, file in config["source_files"].items()
        }
    else:
        from datasets import load_dataset

        if config["workload"] == "bert_sst2":
            source = load_dataset("nyu-mll/glue", "sst2")
            splits = {"train": source["train"], "eval": source["validation"]}
        else:
            source = load_dataset(
                "Salesforce/wikitext",
                config.get("wikitext_variant", "wikitext-2-raw-v1"),
            )
            splits = {"train": source["train"], "eval": source["test"]}
    for split, rows in splits.items():
        if config.get("limit"):
            rows = list(rows)[: config["limit"]]
        if config["workload"] == "bert_sst2":
            encoded = tokenizer(
                [row["sentence"] for row in rows],
                padding="max_length",
                truncation=True,
                max_length=config["sequence_length"],
                return_tensors="np",
            )
            values = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": np.asarray([row["label"] for row in rows], dtype=np.int64),
            }
        else:
            tokens = []
            for row in rows:
                if row["text"].strip():
                    tokens.extend(
                        tokenizer.encode(row["text"], add_special_tokens=False)
                    )
                    tokens.append(tokenizer.eos_token_id)
            length = config["sequence_length"]
            count = len(tokens) // length
            if not count:
                raise ValueError("WikiText split has no full token block")
            inputs = np.asarray(tokens[: count * length], dtype=np.int64).reshape(
                count, length
            )
            values = {
                "input_ids": inputs,
                "attention_mask": np.ones_like(inputs),
                "labels": inputs.copy(),
            }
        np.savez_compressed(output / (split + ".npz"), **values)
    tokenizer.save_pretrained(output / "tokenizer")


def task_data(config, split):
    if config["workload"].endswith("imagenet"):
        size = config.get("image_size", 224)
        operations = (
            [transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip()]
            if split == "train"
            else [
                transforms.Resize(round(size * 256 / 224)),
                transforms.CenterCrop(size),
            ]
        )
        operation = transforms.Compose(
            operations
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return datasets.ImageFolder(
            Path(config["image_root"]) / ("train" if split == "train" else "val"),
            operation,
        )
    with np.load(Path(config["data"]) / (split + ".npz"), allow_pickle=False) as values:
        return TensorDataset(
            *(
                torch.from_numpy(values[key].copy()).long()
                for key in ("input_ids", "attention_mask", "labels")
            )
        )


def build_model(config, model_config=None):
    seed(config["seed"])
    name = config["workload"]
    tiny = config.get("tiny", False)
    if name == "resnet50_imagenet":
        if tiny:
            return models.ResNet(
                models.resnet.Bottleneck,
                [1, 1, 1, 1],
                num_classes=config["classes"],
                width_per_group=8,
            )
        return models.resnet50(weights=None, num_classes=config["classes"])
    if name == "vit_b16_imagenet":
        if tiny:
            return models.VisionTransformer(
                image_size=config["image_size"],
                patch_size=8,
                num_layers=1,
                num_heads=2,
                hidden_dim=32,
                mlp_dim=64,
                num_classes=config["classes"],
            )
        return models.vit_b_16(
            weights=None, image_size=config["image_size"], num_classes=config["classes"]
        )
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        GPT2Config,
        GPT2LMHeadModel,
    )

    cls, configuration = (
        (BertForSequenceClassification, BertConfig)
        if name == "bert_sst2"
        else (GPT2LMHeadModel, GPT2Config)
    )
    if model_config is not None:
        return cls(configuration.from_dict(model_config))
    if tiny:
        spec = (
            BertConfig(
                vocab_size=config["vocab_size"],
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=64,
                num_labels=config["classes"],
                max_position_embeddings=64,
            )
            if name == "bert_sst2"
            else GPT2Config(
                vocab_size=config["vocab_size"],
                n_embd=32,
                n_layer=1,
                n_head=2,
                n_positions=64,
                n_ctx=64,
            )
        )
        return cls(spec)
    if config.get("pretrained", True):
        return cls.from_pretrained(
            config["model_name"], revision=config.get("revision", "main")
        )
    return cls(
        configuration.from_pretrained(
            config["model_name"], revision=config.get("revision", "main")
        )
    )


def task_loss(model, batch, workload, world=1):
    """Return an exact summed objective and its globally weighted backward loss."""
    if workload.endswith("imagenet"):
        logits, labels = model(batch[0]), batch[1]
    else:
        logits = model(input_ids=batch[0], attention_mask=batch[1]).logits
        labels = batch[2]
        if workload == "gpt2_wikitext":
            logits, labels = logits[:, :-1], labels[:, 1:]
    total = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    count = (labels != -100).sum().to(dtype=torch.float64)
    denominator = count.clone()
    if world > 1:
        torch.distributed.all_reduce(denominator)
    if denominator.item() == 0:
        raise ValueError("Batch contains no scored targets")
    return total * world / denominator, total.detach(), count.detach(), logits, labels


def train(config):
    if config["workload"] not in WORKLOADS or config["variant"] not in VARIANTS:
        raise ValueError("Unknown workload or reducer variant")
    world, rank, local = [
        int(os.environ.get(key, default))
        for key, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0"))
    ]
    if config["global_batch_size"] % world:
        raise ValueError("global_batch_size must be divisible by world size")
    target = device(config)
    if target.type == "cuda":
        torch.cuda.set_device(local)
        target = torch.device("cuda", local)
    if world > 1:
        torch.distributed.init_process_group(
            "nccl" if target.type == "cuda" else "gloo",
            init_method=config.get("init_method", "env://"),
            world_size=world,
            rank=rank,
        )
    base = build_model(config).to(target)
    named = list(reversed(list(base.named_parameters())))
    controller = Controller(
        [parameter.numel() * parameter.element_size() for _, parameter in named], config
    )
    reducer = None
    model = base
    if world > 1 and config["variant"] == "ddp":
        model = DDP(
            base,
            device_ids=[local] if target.type == "cuda" else None,
            bucket_cap_mb=config["bucket_bytes"] / 1e6,
        )
    else:
        reducer = GradientReducer(named, controller, world)
    data = task_data(config, "train")
    sampler = DistributedSampler(
        data, world, rank, shuffle=True, seed=config["seed"], drop_last=True
    )
    loader = DataLoader(
        data,
        batch_size=config["global_batch_size"] // world,
        sampler=sampler,
        drop_last=True,
        num_workers=config.get("workers", 0),
    )
    if not len(loader) or not 0 <= config.get("warmup", 0) < config["steps"]:
        raise ValueError("Need a complete global batch and 0 <= warmup < steps")
    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.01),
        )
        if not config["workload"].endswith("imagenet")
        else torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    )
    history, traces = [], []
    model.train()
    epoch = 0
    while len(history) < config["steps"]:
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = [value.to(target) for value in batch]
            optimizer.zero_grad(set_to_none=True)
            if target.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if reducer:
                reducer.begin()
            loss, _, count, _, _ = task_loss(model, batch, config["workload"], world)
            loss.backward()
            readiness, compute, communications = (
                reducer.finish() if reducer else ([], 0.0, [])
            )
            optimizer.step()
            if target.type == "cuda":
                torch.cuda.synchronize()
            step = len(history) + 1
            if world > 1:
                observation = torch.tensor(
                    readiness + [compute], device=target, dtype=torch.float64
                )
                torch.distributed.all_reduce(
                    observation, op=torch.distributed.ReduceOp.MAX
                )
                readiness, compute = (
                    observation.cpu().tolist()[:-1],
                    observation[-1].item(),
                )
            row = {
                "variant": config["variant"],
                "workload": config["workload"],
                "step": step,
                "loss": loss.item(),
                "targets": int(count),
                "world_size": world,
            }
            if reducer:
                controller.observe(
                    readiness, compute, communications, time.perf_counter() - started
                )
                if rank == 0 and step % config["reconfigure_every"] == 0:
                    controller.adapt(time.perf_counter() - started)
                if world > 1:
                    agreed = [
                        (controller.buckets, controller.predict()["order"])
                        if rank == 0
                        else None
                    ]
                    torch.distributed.broadcast_object_list(
                        agreed, src=0, device=target if target.type == "cuda" else None
                    )
                    controller.buckets, controller.predict_order = agreed[0]
                row.update(
                    communication_metrics(
                        communications, readiness, reducer.layout, compute
                    )
                )
                traces.append(
                    {
                        "step": step,
                        "sizes": controller.sizes,
                        "readiness_s": readiness,
                        "compute_s": compute,
                        "communications": communications,
                        "world_size": world,
                    }
                )
            elapsed = torch.tensor(
                time.perf_counter() - started, device=target, dtype=torch.float64
            )
            if world > 1:
                torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
            row.update(
                iteration_s=elapsed.item(),
                samples_per_s=config["global_batch_size"] / elapsed.item(),
            )
            history.append(row)
            if rank == 0:
                print(json.dumps(row), flush=True)
            if step >= config["steps"]:
                break
        epoch += 1
    if reducer:
        reducer.close()
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        torch.save(
            {
                "model": base.state_dict(),
                "config": config,
                "model_config": base.config.to_dict()
                if hasattr(base, "config")
                else None,
            },
            output / "model.pt",
        )
        csv_write(output / "training.csv", history)
        json_write(
            output / "metrics.json", summarize(history[config.get("warmup", 0) :])
        )
        json_write(output / "config.json", config)
        (output / "trace.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8"
        )
    if world > 1:
        torch.distributed.destroy_process_group()
    return str(output / "model.pt")


@torch.no_grad()
def evaluate(config):
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    trained = state["config"]
    target = device(config)
    model = build_model(trained, state["model_config"]).to(target).eval()
    model.load_state_dict(state["model"])
    data = task_data(
        dict(
            trained,
            data=config["data"],
            image_root=config.get("image_root", trained.get("image_root")),
        ),
        "eval",
    )
    total = count = correct = 0.0
    for batch in DataLoader(
        data, batch_size=config.get("eval_batch_size", 8), num_workers=0
    ):
        _, nll, n, logits, labels = task_loss(
            model, [value.to(target) for value in batch], trained["workload"]
        )
        total += float(nll)
        count += float(n)
        if trained["workload"] != "gpt2_wikitext":
            correct += int((logits.argmax(-1) == labels).sum())
    result = (
        {
            "nll_sum": total,
            "token_count": int(count),
            "perplexity": float(np.exp(total / count)),
        }
        if trained["workload"] == "gpt2_wikitext"
        else {
            "correct": int(correct),
            "samples": int(count),
            "accuracy": 100 * correct / count,
        }
    )
    json_write(Path(config["output"]) / "evaluation.json", result)
    return result


def main():
    command, config = configure(["prepare", "train", "evaluate"])
    {"prepare": prepare, "train": train, "evaluate": evaluate}[command](config)


if __name__ == "__main__":
    main()
