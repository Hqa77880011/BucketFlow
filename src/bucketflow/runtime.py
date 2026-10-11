import torch
import time
import torch.distributed as dist


class GradientReducer:
    def __init__(self, named_parameters, controller, world_size):
        self.named = list(named_parameters)
        self.parameters = [parameter for _, parameter in self.named]
        self.controller = controller
        self.world_size = world_size
        dtypes = {parameter.dtype for parameter in self.parameters}
        if len(dtypes) != 1:
            raise ValueError("GradientReducer requires one parameter dtype")
        self.cuda = self.parameters[0].is_cuda
        self.stream = torch.cuda.Stream() if self.cuda else None
        self.handles = [
            parameter.register_post_accumulate_grad_hook(self.make_hook(i))
            for i, parameter in enumerate(self.parameters)
        ]

    def make_hook(self, index):
        def hook(parameter):
            self.ready[index] = True
            if self.cuda:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.events[index] = event
            else:
                self.times[index] = time.perf_counter() - self.start
            if self.controller.config["variant"] != "no_overlap":
                self.launch_ready()

        return hook

    def begin(self):
        self.layout = [list(bucket) for bucket in self.controller.buckets]
        self.order = (
            getattr(self.controller, "predict_order", None)
            or self.controller.predict()["order"]
        )
        self.buffers = [
            torch.empty(
                sum(self.parameters[i].numel() for i in bucket),
                device=self.parameters[0].device,
                dtype=self.parameters[0].dtype,
            )
            for bucket in self.layout
        ]
        self.ready = [False] * len(self.parameters)
        self.events = [None] * len(self.parameters)
        self.times = [0.0] * len(self.parameters)
        self.jobs = []
        self.cursor = 0
        self.start = time.perf_counter()
        if self.cuda:
            self.origin = torch.cuda.Event(enable_timing=True)
            self.origin.record()

    def launch_ready(self):
        while self.cursor < len(self.order):
            index = self.order[self.cursor]
            bucket = self.layout[index]
            if not all(self.ready[i] for i in bucket):
                break
            stream_context = torch.cuda.stream(self.stream) if self.cuda else None
            if stream_context:
                stream_context.__enter__()
                for i in bucket:
                    self.stream.wait_event(self.events[i])
            offset = 0
            for i in bucket:
                grad = self.parameters[i].grad
                self.buffers[index][offset : offset + grad.numel()].copy_(
                    grad.reshape(-1)
                )
                offset += grad.numel()
            started = time.perf_counter()
            if self.cuda:
                first, last = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                first.record(self.stream)
            work = (
                dist.all_reduce(self.buffers[index], async_op=True)
                if self.world_size > 1
                else None
            )
            if self.cuda:
                if work is not None:
                    work.wait()
                self.buffers[index].div_(self.world_size)
                last.record(self.stream)
                stream_context.__exit__(None, None, None)
            self.jobs.append(
                (index, work, started, (first, last) if self.cuda else None)
            )
            self.cursor += 1

    def finish(self):
        if not all(self.ready):
            raise RuntimeError(
                "Every trainable parameter must participate in every backward pass"
            )
        if self.cuda:
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_end.record()
        else:
            compute_time = time.perf_counter() - self.start
        self.launch_ready()
        communications = []
        for index, work, started, events in self.jobs:
            if work is not None:
                work.wait()
            if self.cuda:
                torch.cuda.current_stream().wait_event(events[1])
            else:
                self.buffers[index].div_(self.world_size)
            offset = 0
            for i in self.layout[index]:
                parameter = self.parameters[i]
                parameter.grad.copy_(
                    self.buffers[index][offset : offset + parameter.numel()].view_as(
                        parameter
                    )
                )
                offset += parameter.numel()
            if not self.cuda:
                communications.append(
                    {
                        "bytes": self.buffers[index].numel()
                        * self.buffers[index].element_size(),
                        "duration_s": time.perf_counter() - started,
                    }
                )
        if self.cuda:
            torch.cuda.synchronize()
            readiness = [
                self.origin.elapsed_time(event) / 1000 for event in self.events
            ]
            compute_time = self.origin.elapsed_time(compute_end) / 1000
            communications = [
                {
                    "bytes": self.buffers[i].numel() * self.buffers[i].element_size(),
                    "duration_s": events[0].elapsed_time(events[1]) / 1000,
                }
                for i, _, _, events in self.jobs
            ]
        else:
            readiness = self.times
        return readiness, compute_time, communications

    def close(self):
        for handle in self.handles:
            handle.remove()
