from collections import deque
import numpy as np

VARIANTS = [
    "bucketflow",
    "static",
    "no_overlap",
    "ddp",
    "no_split",
    "no_merge",
    "no_priority",
    "no_stability",
    "no_prediction",
    "no_network",
    "no_queue",
]


def partition(sizes, maximum):
    buckets, current, size = [], [], 0
    for index, value in enumerate(sizes):
        if current and size + value > maximum:
            buckets.append(current)
            current, size = [], 0
        current.append(index)
        size += value
    if current:
        buckets.append(current)
    return buckets


def schedule(
    buckets,
    sizes,
    readiness,
    compute_time,
    alpha,
    bandwidth,
    prioritize=True,
    queue_delays=None,
):
    pending = list(range(len(buckets)))
    releases = [max(readiness[i] for i in bucket) for bucket in buckets]
    durations = [
        alpha
        + sum(sizes[i] for i in bucket) / bandwidth
        + (queue_delays[index] if queue_delays is not None else 0.0)
        for index, bucket in enumerate(buckets)
    ]
    now = 0.0
    records = []
    while pending:
        ready = [i for i in pending if releases[i] <= now]
        if not ready:
            now = min(releases[i] for i in pending)
            ready = [i for i in pending if releases[i] <= now]
        if prioritize:

            def priority(index):
                cost = durations[index]
                slack = max(0.0, compute_time - now - cost)
                age = (now - releases[index]) / max(compute_time, 1e-8)
                return max(0.0, cost - slack) / max(cost, 1e-8) + 0.05 * age

            index = max(ready, key=lambda i: (priority(i), -i))
        else:
            index = min(ready)
        start = now
        now += durations[index]
        records.append(
            {
                "bucket": index,
                "start": start,
                "finish": now,
                "bytes": sum(sizes[i] for i in buckets[index]),
                "blocking": sum(
                    (releases[index] - readiness[i]) * sizes[i] for i in buckets[index]
                ),
            }
        )
        pending.remove(index)
    finish = max(compute_time, now)
    exposed = max(0.0, now - compute_time)
    communication = sum(durations)
    return {
        "iteration_s": finish,
        "exposed_s": exposed,
        "overlap": 1 - exposed / max(communication, 1e-8),
        "collectives": len(buckets),
        "bytes": sum(sizes),
        "order": [r["bucket"] for r in records],
        "records": records,
    }


class QueueDelayRegressor:
    """Online nonnegative conditional-mean regression for log2 message classes.

    Explicit queue_s observations are preferred. Otherwise the target is the
    positive observed-service residual above the current alpha/bandwidth model.
    Only observations from completed previous iterations enter predictions.
    """

    def __init__(self, momentum=0.8):
        self.momentum = momentum
        self.means = {}

    @staticmethod
    def message_class(size):
        return int(np.floor(np.log2(max(int(size), 1))))

    def predict(self, size):
        return self.means.get(self.message_class(size), 0.0)

    def observe(self, size, queue_s):
        if not np.isfinite(queue_s) or queue_s < 0:
            raise ValueError("Observed queue delay must be finite and nonnegative")
        key = self.message_class(size)
        self.means[key] = (
            self.momentum * self.means[key] + (1 - self.momentum) * float(queue_s)
            if key in self.means
            else float(queue_s)
        )


class Controller:
    def __init__(self, sizes, config):
        self.sizes = list(sizes)
        self.config = config
        self.buckets = partition(sizes, config["bucket_bytes"])
        self.readiness = (
            np.zeros(len(sizes))
            if config["variant"] == "no_prediction"
            else np.linspace(0, 0.01, len(sizes))
        )
        self.windows = [deque(maxlen=20) for _ in sizes]
        self.alpha = config["startup_s"]
        self.bandwidth = config["bandwidth_bytes_s"]
        self.bandwidth_window = deque(maxlen=10)
        self.cooldown = 0
        self.pending = None
        self.previous = None
        self.watch = 0
        self.compute = 0.01
        self.queue = QueueDelayRegressor(config.get("queue_momentum", 0.8))

    def predict(self, buckets=None):
        return schedule(
            buckets or self.buckets,
            self.sizes,
            self.readiness,
            self.compute,
            self.alpha,
            self.bandwidth,
            self.config["variant"] != "no_priority",
            [
                self.queue.predict(sum(self.sizes[i] for i in bucket))
                for bucket in (buckets or self.buckets)
            ]
            if self.config["variant"] not in ("no_network", "no_queue")
            else None,
        )

    def observe(self, readiness, compute, communications, iteration):
        for i, value in enumerate(readiness):
            if self.config["variant"] == "no_prediction":
                self.readiness[i] = 0.0
                continue
            recent = np.asarray(self.windows[i])
            if len(recent) >= 3:
                median = np.median(recent)
                mad = np.median(np.abs(recent - median))
                value = float(
                    np.clip(
                        value, median - 3 * max(mad, 1e-6), median + 3 * max(mad, 1e-6)
                    )
                )
            rho = self.config.get("readiness_ema", 0.35)
            self.readiness[i] = (
                value
                if not self.windows[i]
                else rho * value + (1 - rho) * self.readiness[i]
            )
            self.windows[i].append(float(readiness[i]))
        self.compute = compute
        if (
            self.config["variant"] != "no_network"
            and len(communications) >= 2
            and len(set(r["bytes"] for r in communications)) >= 2
        ):
            x = np.asarray([[1.0, row["bytes"]] for row in communications])
            y = np.asarray([row["duration_s"] for row in communications])
            alpha, slope = np.linalg.lstsq(x, y, rcond=None)[0]
            if slope > 0:
                self.alpha = 0.2 * max(0.0, alpha) + 0.8 * self.alpha
                self.bandwidth = 0.2 / slope + 0.8 * self.bandwidth
        elif communications and self.config["variant"] != "no_network":
            duration = sum(row["duration_s"] for row in communications)
            overhead = self.alpha * len(communications)
            if duration > overhead:
                value = sum(row["bytes"] for row in communications) / (
                    duration - overhead
                )
                self.bandwidth = 0.2 * value + 0.8 * self.bandwidth
        if self.config["variant"] not in ("no_network", "no_queue"):
            for row in communications:
                baseline = self.alpha + row["bytes"] / self.bandwidth
                self.queue.observe(
                    row["bytes"],
                    row.get("queue_s", max(0.0, row["duration_s"] - baseline)),
                )
        self.bandwidth_window.append(self.bandwidth)
        if self.watch:
            self.watch -= 1
            if not self.watch and iteration > self.previous[1] * 1.02:
                self.buckets = self.previous[0]
                self.cooldown = self.config["cooldown"]
        if self.cooldown:
            self.cooldown -= 1

    def adapt(self, iteration):
        variant = self.config["variant"]
        if variant in ["static", "ddp", "no_overlap"]:
            return
        stable = variant != "no_stability"
        if stable and (self.cooldown or self.watch):
            return
        if (
            stable
            and len(self.bandwidth_window) >= 3
            and np.std(self.bandwidth_window) / np.mean(self.bandwidth_window)
            > self.config["bandwidth_cv"]
        ):
            self.pending = None
            return
        baseline = self.predict()["iteration_s"]
        candidates = []
        for i, bucket in enumerate(self.buckets):
            if variant == "no_split" or len(self.buckets) >= self.config["max_buckets"]:
                break
            for boundary in range(1, len(bucket)):
                first, second = bucket[:boundary], bucket[boundary:]
                if (
                    min(
                        sum(self.sizes[k] for k in first),
                        sum(self.sizes[k] for k in second),
                    )
                    < self.config["min_bucket_bytes"]
                ):
                    continue
                layout = self.buckets[:i] + [first, second] + self.buckets[i + 1 :]
                gain = (
                    baseline
                    - self.predict(layout)["iteration_s"]
                    - self.alpha
                    - self.config["copy_s"]
                )
                candidates.append((gain, layout))
        if variant != "no_merge":
            for i in range(len(self.buckets) - 1):
                merged = self.buckets[i] + self.buckets[i + 1]
                if sum(self.sizes[k] for k in merged) > self.config["bucket_bytes"]:
                    continue
                layout = self.buckets[:i] + [merged] + self.buckets[i + 2 :]
                gain = (
                    baseline
                    - self.predict(layout)["iteration_s"]
                    - self.config["copy_s"]
                )
                candidates.append((gain, layout))
        if not candidates:
            return
        gain, layout = max(candidates, key=lambda item: item[0])
        if gain <= self.config["gain_s"]:
            self.pending = None
            return
        signature = tuple(tuple(bucket) for bucket in layout)
        if not stable or self.pending == signature:
            self.previous = ([list(bucket) for bucket in self.buckets], iteration)
            self.buckets = layout
            self.cooldown = self.config["cooldown"] if stable else 0
            self.watch = 2 if stable else 0
            self.pending = None
        else:
            self.pending = signature
