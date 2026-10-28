"""Communication metrics and aggregation over independently measured runs."""

from collections import defaultdict
import math
import numpy as np


def communication_metrics(communications, readiness, layout, compute):
    """Return exposed time, overlap and mean ready-to-launch queue delay."""
    total = sum(row["duration_s"] for row in communications)
    exposed = max(0.0, max((row["finish_s"] for row in communications), default=0) - compute)
    delays = [max(0.0, row["launch_s"] - max(readiness[i] for i in layout[row["bucket"]]))
              for row in communications]
    return {
        "exposed_s": exposed,
        "overlap": 1 - exposed / total if total else 0.0,
        "queue_delay_s": float(np.mean(delays)) if delays else 0.0,
        "collectives": len(communications),
        "bytes": sum(row["bytes"] for row in communications),
    }


def summarize(rows):
    """Average numeric observations; warm-up removal belongs to the caller."""
    if not rows:
        raise ValueError("No timed observations; warmup must be smaller than steps")
    keys = set.intersection(*(set(row) for row in rows))
    return {key: float(np.mean([row[key] for row in rows])) for key in sorted(keys)
            if all(isinstance(row[key], (int, float)) for row in rows)}


def paired_summary(records):
    """Aggregate complete runs without pooling hardware and simulated timing.

    Each JSONL record needs workload, variant, world_size, run_id, measurement,
    iteration_s and samples_per_s. Other numeric metrics, including accuracy
    or perplexity, are summarized in their native units. CI uses a documented
    normal approximation; it is null for a single run.
    """
    groups = defaultdict(list)
    lookup = {}
    for row in records:
        key = (row["workload"], row["variant"], row["world_size"], row["measurement"])
        identity = (*key, row["run_id"])
        if identity in lookup:
            raise ValueError(f"Duplicate complete run: {identity}")
        lookup[identity] = row
        groups[key].append(row)
    result = []
    for (workload, variant, workers, measurement), rows in groups.items():
        values = defaultdict(list)
        for row in rows:
            for key, value in row.items():
                if key not in {"world_size", "run_id"} and isinstance(value, (int, float)):
                    values[key].append(value)
            baseline = lookup.get((workload, "ddp", workers, measurement, row["run_id"]))
            if baseline is not None:
                values["throughput_gain"].append(baseline["iteration_s"] / row["iteration_s"] - 1)
                if baseline.get("exposed_s", 0) > 0:
                    values["exposed_cut"].append(1 - row["exposed_s"] / baseline["exposed_s"])
            single = lookup.get((workload, variant, 1, measurement, row["run_id"]))
            if single is not None:
                normalized = row["samples_per_s"] / single["samples_per_s"]
                values["normalized_throughput"].append(normalized)
                values["scaling_efficiency"].append(normalized / workers)
        stats = {}
        for key, value in values.items():
            mean = float(np.mean(value))
            sd = float(np.std(value, ddof=1)) if len(value) > 1 else None
            half = 1.96 * sd / math.sqrt(len(value)) if sd is not None else None
            stats[key] = {"mean": mean, "sample_sd": sd, "n": len(value),
                          "ci95_normal": [mean - half, mean + half] if half is not None else None}
        result.append({"workload": workload, "variant": variant, "world_size": workers,
                       "measurement": measurement, "metrics": stats})
    return result
