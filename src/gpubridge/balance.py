"""Pipeline stage balancing, opt-in (``import gpubridge.balance``; nothing else imports it).

Profile a model's layers on each stage's device, then cut them into contiguous stages so
that the slowest stage, its transfers included, is as fast as possible within each
device's memory. For GPUs of different speed, an even split leaves the fast ones idle.

The cost model, per microbatch:

- **A stage's time** is the forward and backward of its layers on its own device, plus,
  at each boundary it shares with a neighbour, the activation crossing forward and its
  gradient crossing back: ``2 x bytes / bandwidth`` of that link. Both stages of a link
  count it, because gpubridge's sends and receives across islands are host-staged and
  each side waits for them.
- **A stage's memory** is its layers' state (parameters, gradients and optimizer state,
  ``state_bytes``) plus the activations it holds for backward (``activation_bytes`` per
  microbatch) times the microbatches it has in flight: all of them with GPipe,
  ``stages - stage`` with 1F1B (see ``in_flight``).
- **A step** is bounded by its slowest stage, so that is what :func:`partition` minimizes:
  exactly, by dynamic programming over the contiguous splits, in ``O(stages x layers^2)``.

It ignores the pipeline's fill and drain (the bubble, the same for every split with the
same stages and microbatches), overlap of transfers with compute, and the optimizer step.
"""

from __future__ import annotations

import copy
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["LayerCost", "Plan", "evaluate", "in_flight", "partition", "profile_layers"]


@dataclass(frozen=True)
class LayerCost:
    """One layer, profiled on one device, for one microbatch."""

    forward_ms: float
    backward_ms: float
    out_bytes: int  # its output: what crosses a stage boundary after it
    state_bytes: int  # parameters, gradients and optimizer state
    activation_bytes: int  # tensors autograd saves for its backward

    @property
    def ms(self) -> float:
        return self.forward_ms + self.backward_ms


@dataclass(frozen=True)
class Plan:
    split: list[int]  # layers per stage, in stage order
    stage_ms: list[float]  # each stage's time per microbatch under the cost model
    bottleneck_ms: float  # the slowest stage's


def in_flight(schedule: str, stages: int, microbatches: int) -> list[int]:
    """Microbatches each stage holds activations for at once: every one with GPipe, at
    most ``stages - stage`` with 1F1B."""
    if schedule == "gpipe":
        return [microbatches] * stages
    if schedule == "1f1b":
        return [min(stages - s, microbatches) for s in range(stages)]
    raise ValueError(f"unknown schedule {schedule!r}")


def _timer(device: torch.device):
    if device.type == "cuda":  # CUDA and HIP events: the GPU's own clock, on its own stream
        def timed(fn):
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                result = fn()
                end.record(stream)
                end.synchronize()
            return result, start.elapsed_time(end)
    else:
        def timed(fn):
            start = time.perf_counter()
            result = fn()
            return result, (time.perf_counter() - start) * 1e3
    return timed


def profile_layers(layers: Sequence[nn.Module], example: torch.Tensor, device: torch.device,
                   *, trials: int = 3, warmup: int = 1, state_bytes_per_parameter: int = 16,
                   ) -> list[LayerCost]:
    """Each layer's forward and backward on ``device`` (medians of ``trials`` after
    ``warmup``), its output's bytes, its state and its saved activations, one microbatch:
    ``example`` is the first layer's input; each later layer gets the one before's output.
    The backward starts from a gradient of ones. Each layer is profiled as a copy on
    ``device``, one at a time; the layers themselves are never moved or run. 16 bytes per
    parameter is float32 weights, gradients and AdamW's two moments. Saved activations
    count each storage once and leave out the layer's own parameters (its state).

    So the layers stay as they were, even if one raises: their parameters (shared ones
    included) and buffers, gradients and devices. The random number generators (dropout)
    are restored too."""
    timed = _timer(device)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        x = example.to(device)
        costs = []
        for layer in layers:
            work = copy.deepcopy(layer)  # the layer itself is never moved or run
            for p in work.parameters():
                p.grad = None
            work.to(device)
            own = {p.untyped_storage().data_ptr() for p in work.parameters()}
            forward, backward, saved = [], [], []
            for i in range(warmup + trials):
                inp = x.detach().requires_grad_(x.is_floating_point())
                held: dict[int, int] = {}

                def pack(t: torch.Tensor) -> torch.Tensor:
                    for key, size in _held(t):
                        if key not in own:
                            held[key] = size
                    return t

                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    out, f_ms = timed(lambda: work(inp))
                grad = torch.ones_like(out)
                _, b_ms = timed(lambda: out.backward(grad))
                if i >= warmup:
                    forward.append(f_ms)
                    backward.append(b_ms)
                    saved.append(sum(held.values()))
            parameters = sum(p.numel() for p in work.parameters())
            costs.append(LayerCost(round(statistics.median(forward), 4),
                                   round(statistics.median(backward), 4),
                                   out.numel() * out.element_size(),
                                   parameters * state_bytes_per_parameter, max(saved)))
            x = out.detach()
    return costs


def _held(t: torch.Tensor) -> list[tuple[int, int]]:
    """The storages a saved tensor keeps alive, each with its bytes: the whole storage, as a
    view keeps all of it. A sparse tensor's are those of its indices and values."""
    if t.layout == torch.sparse_coo:
        parts = [t._indices(), t._values()]
    elif t.layout in (torch.sparse_csr, torch.sparse_bsr):
        parts = [t.crow_indices(), t.col_indices(), t.values()]
    elif t.layout in (torch.sparse_csc, torch.sparse_bsc):
        parts = [t.ccol_indices(), t.row_indices(), t.values()]
    else:
        parts = [t]
    out = []
    for part in parts:
        storage = part.untyped_storage()
        out.append((storage.data_ptr(), storage.nbytes()))
    return out


def _check(layer_ms: Sequence[Sequence[float]], boundary_bytes: Sequence[float],
           bytes_per_ms: Sequence[float]) -> tuple[int, int]:
    stages, layers = len(layer_ms), len(layer_ms[0]) if layer_ms else 0
    if stages < 1 or layers < stages or any(len(row) != layers for row in layer_ms):
        raise ValueError(f"need a time for each of at least {stages} layers on each of "
                         f"{stages} stages")
    if len(boundary_bytes) != layers - 1 or len(bytes_per_ms) != stages - 1:
        raise ValueError("need one boundary size per pair of neighbouring layers and one "
                         "bandwidth per pair of neighbouring stages")
    return stages, layers


def evaluate(split: Sequence[int], layer_ms: Sequence[Sequence[float]],
             boundary_bytes: Sequence[float], bytes_per_ms: Sequence[float]) -> list[float]:
    """Each stage's time per microbatch under the cost model, for a given split.
    ``layer_ms[s][i]`` is layer ``i``'s forward and backward on stage ``s``'s device,
    ``boundary_bytes[i]`` what crosses between layers ``i`` and ``i + 1``, and
    ``bytes_per_ms[s]`` the link between stages ``s`` and ``s + 1``."""
    stages, layers = _check(layer_ms, boundary_bytes, bytes_per_ms)
    if len(split) != stages or sum(split) != layers or min(split) < 1:
        raise ValueError(f"split {list(split)}: need {stages} positive counts adding up to "
                         f"{layers}")
    times, start = [], 0
    for s, n in enumerate(split):
        t = sum(layer_ms[s][start:start + n])
        if s > 0:
            t += 2 * boundary_bytes[start - 1] / bytes_per_ms[s - 1]
        if s < stages - 1:
            t += 2 * boundary_bytes[start + n - 1] / bytes_per_ms[s]
        times.append(t)
        start += n
    return times


def partition(layer_ms: Sequence[Sequence[float]], boundary_bytes: Sequence[float],
              bytes_per_ms: Sequence[float], *, state_bytes: Sequence[int] | None = None,
              activation_bytes: Sequence[int] | None = None,
              in_flight: Sequence[int] | None = None,
              memory_bytes: Sequence[float | None] | None = None) -> Plan:
    """The contiguous split (each stage at least one layer) whose slowest stage is fastest
    under the cost model (arguments as for :func:`evaluate`), with each stage's memory, if
    given, within ``memory_bytes[s]`` (``None``: no limit). Exact, and the same inputs give
    the same split, so every rank computing it agrees. Raises ``ValueError`` if no split
    fits.
    """
    stages, layers = _check(layer_ms, boundary_bytes, bytes_per_ms)
    limited = memory_bytes is not None
    if limited and not (state_bytes and activation_bytes and in_flight):
        raise ValueError("a memory limit needs state_bytes, activation_bytes and in_flight")
    prefix = [[0.0] for _ in range(stages)]
    for s in range(stages):
        for t in layer_ms[s]:
            prefix[s].append(prefix[s][-1] + t)
    state = [0.0]
    saved = [0.0]
    for i in range(layers if limited else 0):
        state.append(state[-1] + state_bytes[i])
        saved.append(saved[-1] + activation_bytes[i])

    def cost(s: int, i: int, j: int) -> float | None:  # stage s holds layers [i, j)
        if limited and memory_bytes[s] is not None and (
                state[j] - state[i] + in_flight[s] * (saved[j] - saved[i]) > memory_bytes[s]):
            return None
        t = prefix[s][j] - prefix[s][i]
        if s > 0:
            t += 2 * boundary_bytes[i - 1] / bytes_per_ms[s - 1]
        if s < stages - 1:
            t += 2 * boundary_bytes[j - 1] / bytes_per_ms[s]
        return t

    inf = float("inf")
    # best[s][j]: the slowest of stages 0..s when they hold layers [0, j); came[s][j]: where
    # stage s starts in that split.
    best = [[inf] * (layers + 1) for _ in range(stages)]
    came = [[0] * (layers + 1) for _ in range(stages)]
    for j in range(1, layers - stages + 2):
        c = cost(0, 0, j)
        if c is not None:
            best[0][j] = c
    for s in range(1, stages):
        for j in range(s + 1, layers - stages + s + 2):
            for i in range(s, j):
                c = cost(s, i, j)
                if c is None or best[s - 1][i] == inf:
                    continue
                v = max(best[s - 1][i], c)
                if v < best[s][j]:
                    best[s][j], came[s][j] = v, i
    if best[stages - 1][layers] == inf:
        raise ValueError("no split fits every stage's memory")
    split, j = [], layers
    for s in range(stages - 1, 0, -1):
        i = came[s][j]
        split.append(j - i)
        j = i
    split.append(j)
    split.reverse()
    times = evaluate(split, layer_ms, boundary_bytes, bytes_per_ms)
    return Plan(split, times, max(times))
