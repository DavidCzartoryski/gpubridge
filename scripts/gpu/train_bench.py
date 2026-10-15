"""Training throughput through gpubridge: DDP with gpubridge.ddp_comm_hook on a real model.

Launched by torchrun. Models, built from torch.nn alone (no torchvision or
transformers), on synthetic data each rank makes for itself:

- ``gpt2-small``: GPT-2 small, 124M parameters (12 layers, 12 heads, width 768,
  vocabulary 50257, tied embedding), next-token loss on random tokens.
- ``resnet50``: ResNet-50 (v1.5), 25.6M parameters, on random 224x224 images
  and 1000 classes.
- ``tiny-gpt``, ``tiny-resnet``: the same shapes, small enough for CPU tests.

Three timed phases, every step synchronized and timed on every rank:

1. **compute**: forward, backward and an AdamW step on the plain model, before
   DDP wraps it: this rank's GPU alone, with no gradient sync.
2. **ddp**: the same steps through DistributedDataParallel on gpubridge's world
   group, with gpubridge.ddp_comm_hook: each gradient bucket is averaged by an
   async gpubridge.all_reduce that overlaps the rest of backward. Wrapping
   broadcasts rank 0's parameters, so every rank starts this phase in step.
3. **comm alone**: the buckets DDP built (their sizes, from the hook), all_reduced
   the same way with no compute around them.

From the medians (the ddp step time is the slowest rank's at each step):

- ``compute_ms``: per rank, from phase 1.
- ``waiting_ms``: per rank, the slowest rank's compute less its own: the time it
  waits for the slower GPU in every step.
- ``exposed_comm_ms``: the ddp step less the slowest rank's compute: the gradient
  sync (and anything else DDP adds) that backward did not hide.
- ``hidden_comm_ms``: comm alone less the exposed part, at least 0.
- ``samples_per_s``: the global batch (every rank's batch) / ddp step. Each rank's
  ``alone_samples_per_s`` (batch / its compute) says what its GPU does by itself;
  with one rank there is no communication and the two agree.

On a GPU each rank's ``memory`` has, for each phase, the caching allocator's
peaks (``torch.cuda.max_memory_allocated`` and ``max_memory_reserved``, MiB) and
the allocations it retried after freeing its cache (``num_alloc_retries``: a
sign it ran short), and the GPU's total; ``memory`` is null on CPU.

Batches per rank. By default every rank takes ``--batch-size`` samples. Unequal
batches, so a faster GPU does more of each step:

- ``--batch-sizes a,b,...``: one per rank, in rank order.
- ``--global-batch N``: N samples per step in all, split evenly (the first ranks
  take one more when N doesn't divide); one rank alone takes all N, so a
  single-GPU baseline runs at the same global batch with the same flags.
- ``--global-batch N --balance``: a short calibration first times each rank's
  steps at ``--calibration-batch`` samples (default N / world size), then N is
  split into whole samples, at least one per rank, in proportion to each rank's
  speed: each next sample goes to the rank whose predicted compute stays lowest.

With either flag the update stays the gradient of the mean loss over all N
samples: each rank sums its samples' losses (each the mean over its tokens) and
scales the sum by world size / N, so the hook's plain average over ranks,
a SUM then a division by the world size, is the sample-weighted mean. Without the
scaling, unequal batches would weight a sample on a small rank more. The record
holds each rank's batch, compute, per-sample compute and waiting, the global
batch, the split and, with --balance, the calibration's measurements.

``--accumulate K``: each rank's batch (its samples per optimizer step) is cut into
K micro-batches as evenly as whole samples allow, forward and backward run on
each in turn, and DDP syncs the gradients once, in the last backward
(``no_sync()`` for the others): one exchange per optimizer step instead of K.
The losses are sample-weighted as above, so the update is the same as one
backward over the rank's whole batch (BatchNorm aside: it normalizes over each
micro-batch). With --balance the calibration runs at
the micro-batch size, and each rank gets at least K samples. The record holds
K, each rank's micro-batch sizes, the samples per step, and the exchanges per
step the hook counted (and per sample).

So, for "does adding a second GPU raise throughput?", compare a run's
samples_per_s with a one-rank run's on the faster GPU. The decomposition is a
model: compute measured alone can differ from compute while gradients sync
(copies and kernels share the GPU), and that difference lands in exposed_comm.

``broadcast_buffers=False``: ResNet-50's BatchNorm statistics stay per rank, so
Gloo carries nothing in the training loop. After the ddp phase every rank
hashes its parameters (``params_identical``: the same bits everywhere) and sums
them (``params_sum_spread``: the largest difference between two ranks' sums).
Gradients arrive with the same bits on every rank, but across vendors the
optimizer step itself may round differently, so this is recorded, not
enforced, unless a value is not finite.

Rank 0 writes --json PATH. On CPU, in simulation mode:

    GPUBRIDGE_VENDOR=nvidia GPUBRIDGE_SPLIT_TEST=half torchrun --nproc-per-node=2 \\
        scripts/gpu/train_bench.py --model tiny-gpt --json /tmp/train_bench.json
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from gpukit import KIT_VERSION, default_timeout, environment, run_info, sync, write_json
from torch import nn
from torch.distributed import ReduceOp
from torch.nn.parallel import DistributedDataParallel

import gpubridge


@dataclass(frozen=True)
class GPTConfig:
    layers: int
    heads: int
    width: int
    vocab: int
    context: int


@dataclass(frozen=True)
class ResNetConfig:
    blocks: tuple[int, ...]
    base: int
    classes: int
    image: int


MODELS: dict[str, GPTConfig | ResNetConfig] = {
    "gpt2-small": GPTConfig(layers=12, heads=12, width=768, vocab=50257, context=1024),
    "resnet50": ResNetConfig(blocks=(3, 4, 6, 3), base=64, classes=1000, image=224),
    "tiny-gpt": GPTConfig(layers=2, heads=2, width=32, vocab=128, context=32),
    "tiny-resnet": ResNetConfig(blocks=(1, 1), base=8, classes=10, image=32),
}
#: Per-rank batch and (GPT) sequence length when not given: they fit a 16 GB GPU.
DEFAULT_BATCH = {"gpt2-small": 4, "resnet50": 32, "tiny-gpt": 4, "tiny-resnet": 4}
DEFAULT_SEQ = {"gpt2-small": 512, "tiny-gpt": 16}


class Block(nn.Module):
    """A GPT-2 block: pre-LayerNorm causal self-attention, then a 4x MLP."""

    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.heads = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.qkv, self.proj = nn.Linear(c.width, 3 * c.width), nn.Linear(c.width, c.width)
        self.fc, self.out = nn.Linear(c.width, 4 * c.width), nn.Linear(4 * c.width, c.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, w = x.shape
        q, k, v = (z.view(b, t, self.heads, w // self.heads).transpose(1, 2)
                   for z in self.qkv(self.ln1(x)).split(w, dim=2))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, w))
        return x + self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig) -> None:
        super().__init__()
        self.tok, self.pos = nn.Embedding(c.vocab, c.width), nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.width)
        self.head = nn.Linear(c.width, c.vocab, bias=False)
        self.head.weight = self.tok.weight  # tied, as in GPT-2

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


class Bottleneck(nn.Module):
    """ResNet v1.5's bottleneck: the stride is on the 3x3 convolution."""

    def __init__(self, inp: int, width: int, stride: int) -> None:
        super().__init__()
        out = 4 * width
        self.conv1, self.bn1 = nn.Conv2d(inp, width, 1, bias=False), nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3, self.bn3 = nn.Conv2d(width, out, 1, bias=False), nn.BatchNorm2d(out)
        self.down = (nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False),
                                   nn.BatchNorm2d(out))
                     if stride != 1 or inp != out else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return F.relu(y + (x if self.down is None else self.down(x)))


def resnet(c: ResNetConfig) -> nn.Module:
    layers: list[nn.Module] = [nn.Conv2d(3, c.base, 7, 2, 3, bias=False),
                               nn.BatchNorm2d(c.base), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
    inp = c.base
    for stage, count in enumerate(c.blocks):
        width = c.base * 2**stage
        for i in range(count):
            layers.append(Bottleneck(inp, width, 2 if i == 0 and stage > 0 else 1))
            inp = 4 * width
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(inp, c.classes)]
    return nn.Sequential(*layers)


def build(name: str) -> nn.Module:
    c = MODELS[name]
    return GPT(c) if isinstance(c, GPTConfig) else resnet(c)


def batches(name: str, batch: int, seq: int, rank: int, device: torch.device,
            count: int = 4) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``count`` synthetic (input, target) batches for this rank, on its device."""
    c = MODELS[name]
    g = torch.Generator().manual_seed(100 + rank)
    out = []
    for _ in range(count):
        if isinstance(c, GPTConfig):
            tokens = torch.randint(c.vocab, (batch, seq + 1), generator=g)
            x, y = tokens[:, :-1], tokens[:, 1:]
        else:
            x = torch.randn(batch, 3, c.image, c.image, generator=g)
            y = torch.randint(c.classes, (batch,), generator=g)
        out.append((x.to(device), y.to(device)))
    return out


def loss_scale(world: int, global_batch: int, units_per_sample: int = 1) -> float:
    """The factor on a rank's summed loss that makes DDP's average over ranks the
    gradient of the mean loss over all ``global_batch`` samples, whatever each rank's
    share: the average divides by ``world``, the mean by ``global_batch``, and each
    sample's loss is a mean over its ``units_per_sample`` (tokens for GPT, 1 otherwise)."""
    return world / (global_batch * units_per_sample)


def loss_of(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
            scale: float | None = None) -> torch.Tensor:
    """Cross entropy: the mean over the batch, or with ``scale`` (:func:`loss_scale`)
    the sum times ``scale``."""
    logits = model(x)
    if scale is None:
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                           reduction="sum") * scale


def split_batch(global_batch: int, per_sample_s: list[float], minimum: int = 1) -> list[int]:
    """``global_batch`` whole samples over the ranks, at least ``minimum`` each, in
    proportion to their speed: each next sample goes to the rank whose predicted
    compute (samples x seconds per sample) stays lowest, the lower rank on a tie."""
    if global_batch < minimum * len(per_sample_s):
        raise ValueError(f"a global batch of {global_batch} can't give each of "
                         f"{len(per_sample_s)} ranks "
                         + ("a sample" if minimum == 1 else f"{minimum} samples"))
    if not all(t > 0 for t in per_sample_s):
        raise ValueError(f"seconds per sample must be positive, got {per_sample_s}")
    shares = [minimum] * len(per_sample_s)
    for _ in range(global_batch - sum(shares)):
        rank = min(range(len(shares)), key=lambda r: ((shares[r] + 1) * per_sample_s[r], r))
        shares[rank] += 1
    return shares


def even_split(global_batch: int, world: int) -> list[int]:
    """``global_batch`` over ``world`` ranks as evenly as whole samples allow, the
    first ranks taking one more."""
    if global_batch < world:
        raise ValueError(f"a global batch of {global_batch} can't give each of {world} "
                         "ranks a sample")
    return [global_batch // world + (r < global_batch % world) for r in range(world)]


def micro_batches(data: list[tuple[torch.Tensor, torch.Tensor]],
                  sizes: list[int]) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """Each (input, target) batch cut into micro-batches of ``sizes`` samples."""
    return [list(zip(x.split(sizes), y.split(sizes), strict=True)) for x, y in data]


def backward(model: nn.Module, micro: list[tuple[torch.Tensor, torch.Tensor]],
             scale: float | None, loss=loss_of) -> tuple[torch.Tensor, int]:
    """Forward and backward over each micro-batch, gradients accumulating. Under DDP
    the gradients sync once, in the last backward (``no_sync`` for the others). The
    summed loss and the targets' element count."""
    total, units = None, 0
    for j, (x, y) in enumerate(micro):
        syncs = j == len(micro) - 1 or not isinstance(model, DistributedDataParallel)
        with contextlib.nullcontext() if syncs else model.no_sync():
            value = loss(model, x, y, scale)
            value.backward()
        total = value.detach() if total is None else total + value.detach()
        units += y.numel()
    assert total is not None
    return total, units


def timed_steps(model: nn.Module, optimizer: torch.optim.Optimizer, data, device,
                warmup: int, steps: int,
                scale: float | None = None) -> tuple[list[float], list[float]]:
    """Seconds per timed step (synchronized before and after), and every step's loss
    (this rank's mean, with or without ``scale``). Each entry of ``data`` is one
    step's micro-batches (:func:`micro_batches`)."""
    times, losses = [], []
    for i in range(warmup + steps):
        micro = data[i % len(data)]
        sync(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total, units = backward(model, micro, scale)
        optimizer.step()
        sync(device)
        if i >= warmup:
            times.append(time.perf_counter() - start)
        value = float(total)
        losses.append(value if scale is None else value / (scale * units))
    return times, losses


def recording_hook(sizes: list[tuple[int, torch.dtype]], exchanges: list[int] | None = None):
    """gpubridge.ddp_comm_hook, noting each bucket's size and dtype in ``sizes`` as DDP
    hands it over (the last step's buckets: DDP may rebuild them after the first), and
    counting gradient exchanges (bucket 0's) in ``exchanges[0]``."""

    def hook(state, bucket):
        if bucket.index() == 0:
            sizes.clear()
            if exchanges is not None:
                exchanges[0] += 1
        buffer = bucket.buffer()
        sizes.append((buffer.numel(), buffer.dtype))
        return gpubridge.ddp_comm_hook(state, bucket)

    # DDP checks these annotations against the real objects, and this module's are
    # strings (``from __future__ import annotations``).
    hook.__annotations__ = {"state": object, "bucket": dist.GradBucket,
                            "return": torch.futures.Future[torch.Tensor]}
    return hook


def comm_alone(sizes: list[tuple[int, torch.dtype]], device: torch.device,
               trials: int) -> list[float]:
    """Seconds to average every bucket (async, as the hook does) with nothing else running."""
    buffers = [torch.ones(n, dtype=dtype, device=device) for n, dtype in sizes]
    out = []
    for i in range(trials + 1):
        sync(device)
        dist.barrier()
        start = time.perf_counter()
        works = [gpubridge.all_reduce(b, op=ReduceOp.AVG, async_op=True) for b in buffers]
        for work in works:
            if work is not None:
                work.wait()
        sync(device)
        if i:  # the first is a warmup
            out.append(time.perf_counter() - start)
    return out


def mib(size: int) -> float:
    return round(size / 2**20, 1)


def memory_mark(device: torch.device) -> int:
    """Start a phase's memory record: reset the caching allocator's peaks; its
    allocation retries so far (0 on CPU)."""
    if device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_stats(device).get("num_alloc_retries", 0)


def memory_peaks(device: torch.device, retries_before: int) -> dict[str, Any] | None:
    """A phase's memory since :func:`memory_mark`: the peaks allocated to tensors and
    reserved from the GPU (MiB), and the allocations retried after freeing the cache.
    None on CPU."""
    if device.type != "cuda":
        return None
    retries = torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
    return {"max_allocated_mib": mib(torch.cuda.max_memory_allocated(device)),
            "max_reserved_mib": mib(torch.cuda.max_memory_reserved(device)),
            "alloc_retries": retries - retries_before}


def gather(value: Any) -> list[Any]:
    everyone: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, value)
    return everyone


def ms(seconds: float) -> float:
    return round(seconds * 1e3, 3)


def decompose(compute: list[list[float]], ddp: list[list[float]], alone: list[list[float]],
              batch: int | list[int]) -> dict[str, Any]:
    """The step-time split, from each rank's compute, ddp and comm-alone times (seconds),
    and its batch (one for every rank, or one per rank)."""
    world = len(compute)
    sizes = [batch] * world if isinstance(batch, int) else list(batch)
    compute_s = [statistics.median(c) for c in compute]
    step_s = statistics.median(max(step) for step in zip(*ddp))
    slowest = max(compute_s)
    exposed = max(0.0, step_s - slowest)
    alone_s = statistics.median(max(trial) for trial in zip(*alone)) if alone[0] else 0.0
    return {
        "step_ms": ms(step_s),
        "samples_per_s": round(sum(sizes) / step_s, 3),
        "slowest_compute_ms": ms(slowest),
        "exposed_comm_ms": ms(exposed),
        "comm_alone_ms": ms(alone_s),
        "hidden_comm_ms": ms(max(0.0, alone_s - exposed)),
        "ranks": [{"rank": r, "batch_size": n, "compute_ms": ms(c),
                   "compute_ms_per_sample": ms(c / n), "waiting_ms": ms(slowest - c),
                   "alone_samples_per_s": round(n / c, 3)}
                  for r, (c, n) in enumerate(zip(compute_s, sizes, strict=True))],
    }


def calibrate(name: str, model: nn.Module, batch: int, seq: int, rank: int,
              device: torch.device, warmup: int, steps: int, lr: float,
              scale: float | None) -> list[float]:
    """Seconds per step of this rank alone at ``batch`` samples: its speed for
    --balance. The optimizer is thrown away, like the compute phase's."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    data = micro_batches(batches(name, batch, seq, rank, device), [batch])
    times, _ = timed_steps(model, optimizer, data, device, warmup, steps, scale)
    del optimizer
    model.zero_grad(set_to_none=True)
    return times


def batch_sizes(args: argparse.Namespace, world: int) -> tuple[list[int], str]:
    """Each rank's batch before any calibration, and how it was chosen."""
    if args.batch_sizes:
        sizes = [int(s) for s in args.batch_sizes.split(",")]
        if len(sizes) != world or min(sizes) < 1:
            raise SystemExit(f"--batch-sizes needs {world} sizes of at least 1, one per "
                             f"rank, got {args.batch_sizes!r}")
        return sizes, "explicit"
    if args.global_batch:
        try:
            return even_split(args.global_batch, world), "even"
        except ValueError as e:
            raise SystemExit(f"--global-batch: {e}") from None
    return [args.batch_size or DEFAULT_BATCH[args.model]] * world, "per-rank"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=MODELS, default="gpt2-small")
    sizes_given = parser.add_mutually_exclusive_group()
    sizes_given.add_argument("--batch-size", type=int, default=None, help="per rank")
    sizes_given.add_argument("--batch-sizes", default=None,
                             help="one batch per rank, comma-separated, in rank order")
    sizes_given.add_argument("--global-batch", type=int, default=None,
                             help="samples per step over all ranks, split evenly or, "
                             "with --balance, by speed")
    parser.add_argument("--balance", action="store_true",
                        help="split --global-batch by each rank's measured speed")
    parser.add_argument("--calibration-batch", type=int, default=None,
                        help="--balance's per-rank batch while timing; default "
                        "global batch / world size")
    parser.add_argument("--calibration-steps", type=int, default=3,
                        help="--balance's timed steps per rank, after one warm-up")
    parser.add_argument("--accumulate", type=int, default=1, metavar="K",
                        help="micro-batches per optimizer step: each rank's batch is cut "
                        "into K, its gradients synced once, after the last")
    parser.add_argument("--seq-len", type=int, default=None, help="GPT models only")
    parser.add_argument("--steps", type=int, default=10, help="timed steps per phase")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--comm-trials", type=int, default=5)
    parser.add_argument("--bucket-mb", type=float, default=25.0, help="DDP's bucket_cap_mb")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--json", type=Path, help="rank 0 writes the result here")
    parser.add_argument("--timeout", type=float, default=default_timeout(600),
                        help="process-group timeout (s); default KIT_TIMEOUT, else 600")
    args = parser.parse_args()
    if args.balance and not args.global_batch:
        raise SystemExit("--balance needs --global-batch")
    if args.accumulate < 1:
        raise SystemExit("--accumulate must be at least 1")
    accumulate = args.accumulate
    config = MODELS[args.model]
    seq = (args.seq_len or DEFAULT_SEQ[args.model]) if isinstance(config, GPTConfig) else 0
    if isinstance(config, GPTConfig) and not 0 < seq <= config.context:
        raise SystemExit(f"--seq-len must be 1 to {config.context} for {args.model}")

    topology = gpubridge.init(timeout=timedelta(seconds=args.timeout))
    device, rank, world = topology.device, topology.rank, topology.world_size
    if topology.split_test and rank == 0:
        print("SPLIT TEST: not a mixed-vendor result.", file=sys.stderr, flush=True)
    shares, split = batch_sizes(args, world)
    global_batch = sum(shares)
    if min(shares) < accumulate:
        raise SystemExit(f"--accumulate {accumulate} needs at least {accumulate} samples "
                         f"per rank, got {shares}")
    # Unequal batches, a global batch asked for or micro-batches: sample-weighted losses
    # (loss_scale), so the update is the mean over every sample of the step.
    scale = (None if split == "per-rank" and accumulate == 1
             else loss_scale(world, global_batch, seq or 1))
    torch.manual_seed(0)  # on CPU, so every rank builds the same model on any device
    model = build(args.model).to(device)
    parameters = sum(p.numel() for p in model.parameters())

    calibration = None
    if args.balance and world > 1:
        # 0. calibration: each rank alone at one batch size, then the split by speed.
        # With micro-batches, at the micro-batch size: what each forward and backward runs.
        cal_batch = args.calibration_batch or max(1, global_batch // (world * accumulate))
        times = calibrate(args.model, model, cal_batch, seq, rank, device, 1,
                          args.calibration_steps, args.lr, scale)
        per_sample = [statistics.median(t) / cal_batch for t in gather(times)]
        shares, split = split_batch(global_batch, per_sample, accumulate), "balanced"
        calibration = {"batch_size": cal_batch, "steps": args.calibration_steps,
                       "warmup_steps": 1,
                       "ranks": [{"rank": r, "compute_ms_per_sample": ms(t),
                                  "predicted_compute_ms": ms(t * n)}
                                 for r, (t, n) in enumerate(zip(per_sample, shares,
                                                                strict=True))]}
        if rank == 0:
            print(f"balanced split of {global_batch}: {shares} from per-sample compute "
                  + ", ".join(f"{ms(t)} ms" for t in per_sample), flush=True)
    elif args.balance:
        split = "balanced"
    batch = shares[rank]
    micro = [even_split(n, accumulate) for n in shares]
    data = micro_batches(batches(args.model, batch, seq, rank, device), micro[rank])

    # 1. compute: this rank's GPU alone. Its optimizer state is thrown away; DDP below
    # starts again from rank 0's parameters.
    memory = {}  # each phase's peaks (GPU only)
    mark = memory_mark(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    compute, _ = timed_steps(model, optimizer, data, device, args.warmup_steps, args.steps,
                             scale)
    memory["compute"] = memory_peaks(device, mark)
    del optimizer
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. ddp: the gradient sync through gpubridge, overlapping backward.
    mark = memory_mark(device)
    ddp = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda"
                                  else None, bucket_cap_mb=args.bucket_mb,
                                  broadcast_buffers=False)
    sizes: list[tuple[int, torch.dtype]] = []
    exchanges = [0]
    ddp.register_comm_hook(None, recording_hook(sizes, exchanges))
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=args.lr)
    ddp_times, losses = timed_steps(ddp, optimizer, data, device, args.warmup_steps, args.steps,
                                    scale)
    exchanges_per_step = exchanges[0] / (args.warmup_steps + args.steps)
    memory["ddp"] = memory_peaks(device, mark)

    # Every rank's parameters: their bits, their sum, and whether they are finite.
    digest, total, finite = hashlib.sha256(), 0.0, True
    for p in model.parameters():
        values = p.detach().cpu().contiguous()
        digest.update(values.view(torch.uint8).numpy().tobytes())
        total += float(values.double().sum())
        finite = finite and bool(torch.isfinite(values).all())
    del optimizer, ddp

    # 3. comm alone: the same buckets, no compute.
    mark = memory_mark(device)
    alone = comm_alone(sizes, device, args.comm_trials)
    memory["comm_alone"] = memory_peaks(device, mark)

    gpu = device.type == "cuda"
    everyone = gather({"compute": compute, "ddp": ddp_times, "alone": alone,
                       "digest": digest.hexdigest(), "sum": total, "finite": finite,
                       "vendor": topology.vendor,
                       "device": torch.cuda.get_device_name(device) if gpu else str(device),
                       "memory": ({"total_mib": mib(torch.cuda.get_device_properties(device)
                                                    .total_memory), **memory}
                                  if gpu else None)})
    environments = gather(environment()) if args.json else []
    result = decompose([e["compute"] for e in everyone], [e["ddp"] for e in everyone],
                       [e["alone"] for e in everyone], shares)
    for entry, mine in zip(result["ranks"], everyone, strict=True):
        entry.update(vendor=mine["vendor"], device=mine["device"],
                     compute_ms_all=[ms(t) for t in mine["compute"]],
                     ddp_step_ms_all=[ms(t) for t in mine["ddp"]], memory=mine["memory"])
    ok = all(e["finite"] for e in everyone)
    if rank == 0:
        record = {
            "kit_version": KIT_VERSION, "script": "scripts/gpu/train_bench.py",
            "argv": sys.argv, "run": run_info(topology), "world_size": world,
            "grad_sync": "DistributedDataParallel + gpubridge.ddp_comm_hook",
            "policy": topology.policy_name,
            "model": {"name": args.model, "config": asdict(config), "parameters": parameters,
                      "gradient_bytes": sum(n * torch.tensor([], dtype=d).element_size()
                                            for n, d in sizes)},
            "batch_size_per_rank": shares[0] if len(set(shares)) == 1 else shares,
            "batch_sizes": shares, "global_batch": global_batch, "batch_split": split,
            "loss_weighting": ("mean per rank, averaged over ranks" if scale is None else
                               "sample-weighted: each rank's summed loss x world size / "
                               "(global batch x units per sample)"),
            "calibration": calibration,
            "accumulate": accumulate, "micro_batch_sizes": micro,
            "samples_per_step": global_batch,
            "exchanges_per_step": exchanges_per_step,
            "samples_per_exchange": (round(global_batch / exchanges_per_step, 3)
                                     if exchanges_per_step else None),
            "exchanges_per_sample": round(exchanges_per_step / global_batch, 6),
            "seq_len": seq or None,
            "steps": args.steps, "warmup_steps": args.warmup_steps,
            "comm_trials": args.comm_trials, "bucket_mb": args.bucket_mb,
            "buckets": len(sizes), "optimizer": f"AdamW(lr={args.lr})",
            **result,
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "params_identical": len({e["digest"] for e in everyone}) == 1,
            "params_sum_spread": max(e["sum"] for e in everyone) - min(e["sum"] for e in everyone),
            "params_finite": ok,
            "environment": environments[0] if environments else None,
            "environments": environments,
        }
        if topology.split_test:
            record["warning"] = "SPLIT TEST: not a mixed-vendor result."
        print(f"{args.model}, {world} rank(s), {topology.run_kind}: "
              f"{result['samples_per_s']} samples/s, step {result['step_ms']} ms = slowest "
              f"compute {result['slowest_compute_ms']} + exposed comm "
              f"{result['exposed_comm_ms']} (comm alone {result['comm_alone_ms']}); per rank "
              + ", ".join(f"{r['vendor']} batch {r['batch_size']} compute {r['compute_ms']} "
                          f"waits {r['waiting_ms']}" for r in result["ranks"])
              + f"; global batch {global_batch}"
              + (f" in {accumulate} micro-batches {micro}, {exchanges_per_step:g} "
                 "exchange(s) per step" if accumulate > 1 else ""), flush=True)
        if not record["params_identical"]:
            print(f"note: the ranks' parameters differ (sums up to "
                  f"{record['params_sum_spread']:.3g} apart)", flush=True)
        if args.json:
            write_json(args.json, record)
    gpubridge.destroy()
    if not ok:
        raise SystemExit("a parameter is not finite after training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
