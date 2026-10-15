"""gpubridge.balance: the partitioner against brute force, with injected device speeds,
links and memory limits; the profiler on CPU."""

from __future__ import annotations

import itertools
import random

import pytest
import torch
from torch import nn

from gpubridge import balance


def compositions(layers: int, stages: int):
    for cuts in itertools.combinations(range(1, layers), stages - 1):
        edges = (0, *cuts, layers)
        yield [b - a for a, b in itertools.pairwise(edges)]


def memory(split, state, saved, flight):
    out, start = [], 0
    for s, n in enumerate(split):
        out.append(sum(state[start:start + n]) + flight[s] * sum(saved[start:start + n]))
        start += n
    return out


def random_case(rng: random.Random, with_memory: bool):
    stages = rng.randint(1, 4)
    layers = rng.randint(stages, 9)
    work = [rng.uniform(0.5, 5.0) for _ in range(layers)]
    speed = [rng.choice([1.0, 1.0, 1.7, 3.0]) for _ in range(stages)]  # injected slowness
    layer_ms = [[w * speed[s] for w in work] for s in range(stages)]
    boundary = [rng.choice([1e5, 1e6, 8e6]) for _ in range(layers - 1)]
    link = [rng.choice([1e5, 1e6, 1e7]) for _ in range(stages - 1)]  # bytes per ms
    kwargs = {}
    if with_memory:
        state = [rng.randint(1, 10) for _ in range(layers)]
        saved = [rng.randint(1, 5) for _ in range(layers)]
        flight = balance.in_flight(rng.choice(["gpipe", "1f1b"]), stages, 4)
        limit = [rng.choice([None, rng.randint(5, 60)]) for _ in range(stages)]
        kwargs = {"state_bytes": state, "activation_bytes": saved, "in_flight": flight,
                  "memory_bytes": limit}
    return layer_ms, boundary, link, kwargs


@pytest.mark.parametrize("with_memory", [False, True])
def test_partition_matches_brute_force(with_memory):
    rng = random.Random(7 + with_memory)
    for _ in range(300):
        layer_ms, boundary, link, kw = random_case(rng, with_memory)
        stages, layers = len(layer_ms), len(layer_ms[0])
        feasible = []
        for split in compositions(layers, stages):
            if kw and any(lim is not None and m > lim for m, lim in zip(
                    memory(split, kw["state_bytes"], kw["activation_bytes"], kw["in_flight"]),
                    kw["memory_bytes"])):
                continue
            feasible.append(max(balance.evaluate(split, layer_ms, boundary, link)))
        if not feasible:
            with pytest.raises(ValueError, match="no split fits"):
                balance.partition(layer_ms, boundary, link, **kw)
            continue
        plan = balance.partition(layer_ms, boundary, link, **kw)
        assert plan.bottleneck_ms == pytest.approx(min(feasible))
        assert sum(plan.split) == layers and min(plan.split) >= 1
        assert plan.stage_ms == balance.evaluate(plan.split, layer_ms, boundary, link)


def test_a_slower_second_gpu_gets_fewer_layers_and_the_expected_speedup():
    # 12 equal layers of 3 ms on the fast GPU, 5 ms on the slow one, a link that costs
    # 2 x 1e6 B / 1e6 B/ms = 2 ms per hop, both stages counting it.
    layer_ms = [[3.0] * 12, [5.0] * 12]
    boundary, link = [1e6] * 11, [1e6]
    even = balance.evaluate([6, 6], layer_ms, boundary, link)
    assert even == [6 * 3 + 2, 6 * 5 + 2]
    plan = balance.partition(layer_ms, boundary, link)
    assert plan.split == [8, 4]  # 26 and 22 ms; 7 + 5 gives 23 and 27, 9 + 3 29 and 17
    best = min(max(balance.evaluate(s, layer_ms, boundary, link))
               for s in compositions(12, 2))
    assert plan.bottleneck_ms == best
    assert max(even) / plan.bottleneck_ms == pytest.approx(32 / 26)


def test_a_memory_limit_overrides_speed():
    # The fast GPU would take most layers, but only 3 of them fit on it.
    layer_ms = [[1.0] * 6, [4.0] * 6]
    plan = balance.partition(layer_ms, [0.0] * 5, [1.0], state_bytes=[10] * 6,
                             activation_bytes=[0] * 6, in_flight=[1, 1],
                             memory_bytes=[30, None])
    assert plan.split == [3, 3]


def test_in_flight_per_schedule():
    assert balance.in_flight("gpipe", 3, 8) == [8, 8, 8]
    assert balance.in_flight("1f1b", 3, 8) == [3, 2, 1]
    assert balance.in_flight("1f1b", 4, 2) == [2, 2, 2, 1]


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="at least 3 layers"):
        balance.partition([[1.0, 1.0]] * 3, [0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="positive counts"):
        balance.evaluate([2, 0], [[1.0, 1.0]] * 2, [0.0], [1.0])
    with pytest.raises(ValueError, match="memory limit needs"):
        balance.partition([[1.0, 1.0]] * 2, [0.0], [1.0], memory_bytes=[1, 1])


def test_profile_layers_on_cpu():
    torch.manual_seed(0)
    layers = [nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)]
    costs = balance.profile_layers(layers, torch.randn(5, 8), torch.device("cpu"), trials=2)
    assert len(costs) == 3
    assert all(c.forward_ms >= 0 and c.backward_ms >= 0 and c.ms >= c.forward_ms for c in costs)
    assert [c.out_bytes for c in costs] == [5 * 16 * 4, 5 * 16 * 4, 5 * 4 * 4]
    assert [c.state_bytes for c in costs] == [(8 * 16 + 16) * 16, 0, (16 * 4 + 4) * 16]
    # Each saves its input; the Linears' weights, also saved, are state, not activations.
    assert [c.activation_bytes for c in costs] == [5 * 8 * 4, 5 * 16 * 4, 5 * 16 * 4]
    assert all(p.grad is None for layer in layers for p in layer.parameters())


def test_profiling_leaves_the_layers_as_they_were():
    torch.manual_seed(0)
    layers = [nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5)]
    for layer in layers:
        layer.train()
    for p in layers[0].parameters():
        p.grad = torch.full_like(p, 3.0)
    weights = [p.detach().clone() for layer in layers for p in layer.parameters()]
    buffers = [b.clone() for b in layers[1].buffers()]
    example = torch.randn(6, 4)
    rng = torch.get_rng_state()
    balance.profile_layers(layers, example, torch.device("cpu"), trials=2)
    assert all(torch.equal(p.grad, torch.full_like(p, 3.0)) for p in layers[0].parameters())
    assert all(torch.equal(a, b) for a, b in zip(
        weights, [p.detach() for layer in layers for p in layer.parameters()]))
    assert all(torch.equal(a, b) for a, b in zip(buffers, layers[1].buffers()))
    assert torch.equal(rng, torch.get_rng_state())


def test_a_layer_that_raises_is_still_restored():
    class Boom(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.register_buffer("seen", torch.zeros(1))

        def forward(self, x):
            self.seen += 1
            raise RuntimeError("boom")

    layer = Boom()
    layer.lin.weight.grad = torch.ones_like(layer.lin.weight)
    weight = layer.lin.weight.detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        balance.profile_layers([layer], torch.randn(2, 4), torch.device("cpu"))
    assert torch.equal(layer.seen, torch.zeros(1))
    assert torch.equal(layer.lin.weight.grad, torch.ones_like(weight))
    assert torch.equal(layer.lin.weight.detach(), weight)


def test_a_sparse_input_is_profiled():
    costs = balance.profile_layers([nn.Linear(8, 4)], torch.randn(5, 8).to_sparse(),
                                   torch.device("cpu"), trials=1)
    assert costs[0].activation_bytes > 0 and costs[0].out_bytes == 5 * 4 * 4


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device")
def test_a_layer_without_parameters_goes_back_to_its_own_device():
    layer = nn.BatchNorm1d(4, affine=False).to("mps")
    balance.profile_layers([layer], torch.randn(6, 4), torch.device("cpu"), trials=1)
    assert all(b.device.type == "mps" for b in layer.buffers())
