import pytest
import torch
from _harness import expected_sum, rank_tensor, run_ranks

import gpubridge

SHAPES = [(), (1,), (7,), (3, 4), (2, 3, 5)]
DTYPES = [torch.float32, torch.float16]


def _check_sum(topology):
    tensor = rank_tensor((3, 4), torch.float32, topology.rank)
    data_ptr = tensor.data_ptr()
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((3, 4), torch.float32, topology.world_size))
    assert tensor.data_ptr() == data_ptr, "all_reduce must work in place"


@pytest.mark.parametrize(
    "vendors",
    [
        ["nvidia", "nvidia", "amd", "amd"],
        ["nvidia", "amd", "nvidia", "amd"],
        ["nvidia", "nvidia", "nvidia", "amd"],
        ["amd", "nvidia", "nvidia", "nvidia"],
        ["nvidia"] * 4,
        ["amd"] * 2,
        ["amd"],
    ],
    ids=[
        "2nvidia-2amd",
        "interleaved",
        "3nvidia-1amd",
        "1amd-3nvidia",
        "nvidia-only",
        "amd-only",
        "single-rank",
    ],
)
def test_sum_matches_on_every_rank(vendors, tmp_path):
    run_ranks(vendors, _check_sum, tmp_path)


def _check_shapes_and_dtypes(topology):
    for dtype in DTYPES:
        for shape in SHAPES:
            tensor = rank_tensor(shape, dtype, topology.rank)
            gpubridge.all_reduce(tensor)
            expected = expected_sum(shape, dtype, topology.world_size)
            assert tensor.dtype == dtype
            assert torch.equal(tensor, expected), f"{dtype} {shape}: {tensor} != {expected}"


@pytest.mark.parametrize(
    "vendors",
    [["nvidia", "nvidia", "amd", "amd"], ["nvidia", "nvidia", "nvidia", "amd"]],
    ids=["2nvidia-2amd", "3nvidia-1amd"],
)
def test_shapes_and_dtypes(vendors, tmp_path):
    run_ranks(vendors, _check_shapes_and_dtypes, tmp_path)


def _check_rejects_bad_calls(topology):
    with pytest.raises(ValueError, match="SUM"):
        gpubridge.all_reduce(torch.ones(3), op=gpubridge.ReduceOp.MAX)
    with pytest.raises(ValueError, match="simulation mode keeps tensors on CPU"):
        gpubridge.all_reduce(torch.ones(3, device="meta"))
    with pytest.raises(ValueError, match="contiguous"):
        gpubridge.all_reduce(torch.ones(4, 4).t())
    # The rejected calls must not have left any rank mid-collective.
    tensor = torch.ones(2)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, torch.full((2,), float(topology.world_size)))


def test_rejects_bad_calls_without_communicating(tmp_path):
    run_ranks(["nvidia", "amd"], _check_rejects_bad_calls, tmp_path)


def test_requires_init():
    with pytest.raises(RuntimeError, match="init"):
        gpubridge.all_reduce(torch.ones(1))
