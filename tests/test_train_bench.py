"""train_bench.py's unequal per-rank batches: DDP through gpubridge.ddp_comm_hook still
takes the gradient of the mean loss over every rank's samples, as one process would
over the concatenated batch."""

import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from _harness import run_ranks
from test_ops import gather
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import gpubridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
import train_bench  # noqa: E402

STEPS = 2
#: Unequal shares of a power-of-two global batch, so 1 / N and world / N are exact.
LAYOUTS = {
    "2-ranks-1+3": (["nvidia", "amd"], [1, 3]),
    "3-ranks-1+2+5": (["nvidia", "amd", "amd"], [1, 2, 5]),
}


def make_model() -> nn.Module:
    """Small integer weights, no bias: every gradient below is a small integer over N."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5, bias=False), nn.ReLU(), nn.Linear(5, 3, bias=False))
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randint(-2, 3, param.shape))
    return model


def full_batch(size: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1000 + step)
    return (torch.randint(-3, 4, (size, 6), generator=g).float(),
            torch.randint(-2, 3, (size, 3), generator=g).float())


def per_sample_loss(model: nn.Module, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (model(x) * w).sum(dim=1)  # integer inputs and weights: integer gradients


def mine(shares: list[int], rank: int, *tensors: torch.Tensor) -> list[torch.Tensor]:
    start = sum(shares[:rank])
    return [t[start:start + shares[rank]] for t in tensors]


def _check_integer(topology, shares, weighted):
    rank, world, n = topology.rank, topology.world_size, sum(shares)
    reference = make_model()  # one process over the concatenated batch, no DDP
    model = DDP(make_model(), bucket_cap_mb=1e-6)  # several buckets, several all_reduces
    model.register_comm_hook(None, gpubridge.ddp_comm_hook)
    mismatches = 0
    for step in range(STEPS):
        x, w = full_batch(n, step)
        reference.zero_grad()
        per_sample_loss(reference, x, w).mean().backward()
        model.zero_grad()
        lx, lw = mine(shares, rank, x, w)
        losses = per_sample_loss(model, lx, lw)
        if weighted:
            (losses.sum() * train_bench.loss_scale(world, n)).backward()
        else:  # each rank's own mean, as train_bench does with equal batches
            losses.mean().backward()
        for i, (param, twin) in enumerate(zip(model.parameters(), reference.parameters(),
                                              strict=True)):
            if weighted:
                assert torch.equal(param.grad, twin.grad), (step, i)
            else:
                mismatches += not torch.allclose(param.grad, twin.grad, rtol=1e-3, atol=1e-3)
        # The same integer update on both, so the next step's sums stay exact.
        with torch.no_grad():
            for param, twin in zip(model.parameters(), reference.parameters(), strict=True):
                twin -= twin.grad.sign()
                param.copy_(twin)
    if not weighted:
        assert mismatches > 0, "the unweighted average matched: the test can't see the bug"


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sample_weighted_losses_give_the_concatenated_batchs_gradient_bit_for_bit(
        layout, tmp_path):
    vendors, shares = LAYOUTS[layout]
    run_ranks(vendors, partial(_check_integer, shares=shares, weighted=True), tmp_path)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_unweighted_means_with_unequal_batches_do_not_match(layout, tmp_path):
    vendors, shares = LAYOUTS[layout]
    run_ranks(vendors, partial(_check_integer, shares=shares, weighted=False), tmp_path)


#: Float data (tiny-gpt's cross entropy): sums in a different order round differently.
#: On CPU (torch 2.14.1) the largest difference was 3.6e-7, on gradients up to 1.3.
FLOAT_RTOL, FLOAT_ATOL = 1e-5, 1e-6


def _check_float(topology, shares):
    rank, world, n = topology.rank, topology.world_size, sum(shares)
    seq = train_bench.DEFAULT_SEQ["tiny-gpt"]
    torch.manual_seed(0)
    reference = train_bench.build("tiny-gpt")
    torch.manual_seed(0)
    model = DDP(train_bench.build("tiny-gpt"), bucket_cap_mb=0.01)
    model.register_comm_hook(None, gpubridge.ddp_comm_hook)
    g = torch.Generator().manual_seed(7)
    tokens = torch.randint(train_bench.MODELS["tiny-gpt"].vocab, (n, seq + 1), generator=g)
    x, y = tokens[:, :-1], tokens[:, 1:]
    train_bench.loss_of(reference, x, y).backward()
    lx, ly = mine(shares, rank, x, y)
    train_bench.loss_of(model, lx, ly, train_bench.loss_scale(world, n, seq)).backward()
    for param, twin in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(param.grad, twin.grad, rtol=FLOAT_RTOL, atol=FLOAT_ATOL)
    # Every rank holds the same gradients bit for bit.
    everyone = gather([p.grad for p in model.parameters()])
    assert all(torch.equal(a, b) for other in everyone for a, b in zip(
        everyone[0], other, strict=True))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_sample_weighted_losses_match_on_float_data_within_tolerance(layout, tmp_path):
    vendors, shares = LAYOUTS[layout]
    run_ranks(vendors, partial(_check_float, shares=shares), tmp_path)


def test_loss_scale_undoes_the_average_and_takes_the_mean():
    assert train_bench.loss_scale(2, 4) == 0.5
    assert train_bench.loss_scale(3, 8, 16) == 3 / 128
    assert train_bench.loss_scale(1, 32) == 1 / 32


@pytest.mark.parametrize(("n", "per_sample", "want"), [
    (32, [0.034, 0.058], [20, 12]),       # the V100 about 1.7x as fast as the MI50
    (8, [1.0, 1.0], [4, 4]),
    (7, [1.0, 1.0], [4, 3]),              # a tie goes to the lower rank
    (3, [1.0, 100.0, 100.0], [1, 1, 1]),  # at least one each
    (10, [1.0, 2.0, 4.0], [6, 3, 1]),
])
def test_split_batch_minimizes_the_slowest_ranks_compute(n, per_sample, want):
    got = train_bench.split_batch(n, per_sample)
    assert got == want and sum(got) == n
    # No single sample moved to another rank would lower the slowest rank's compute.
    slowest = max(s * t for s, t in zip(got, per_sample, strict=True))
    for src in range(len(got)):
        for dst in range(len(got)):
            if src == dst or got[src] == 1:
                continue
            moved = list(got)
            moved[src] -= 1
            moved[dst] += 1
            assert max(s * t for s, t in zip(moved, per_sample, strict=True)) >= slowest


def test_split_batch_needs_a_sample_per_rank_and_positive_times():
    with pytest.raises(ValueError, match="a sample"):
        train_bench.split_batch(1, [1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        train_bench.split_batch(4, [1.0, 0.0])


def test_even_split_gives_the_first_ranks_the_remainder():
    assert train_bench.even_split(32, 2) == [16, 16]
    assert train_bench.even_split(7, 3) == [3, 2, 2]
    assert train_bench.even_split(5, 1) == [5]
    with pytest.raises(ValueError):
        train_bench.even_split(1, 2)


def test_decompose_counts_each_ranks_own_batch():
    out = train_bench.decompose([[1.0], [1.5]], [[2.0], [2.0]], [[0.5], [0.5]], [20, 12])
    assert out["samples_per_s"] == 16.0
    assert [r["batch_size"] for r in out["ranks"]] == [20, 12]
    assert [r["compute_ms_per_sample"] for r in out["ranks"]] == [50.0, 125.0]
    assert [r["alone_samples_per_s"] for r in out["ranks"]] == [20.0, 8.0]
