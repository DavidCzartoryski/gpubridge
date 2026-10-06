import pytest
import torch
from _harness import rank_tensor, run_ranks

import gpubridge


def _check_broadcast_from_every_rank(topology):
    # Every rank takes a turn as src, so each island broadcasts from both
    # its leader and its other members.
    for src in range(topology.world_size):
        for dtype in (torch.float32, torch.float16):
            tensor = rank_tensor((3, 4), dtype, topology.rank)
            gpubridge.broadcast(tensor, src)
            assert torch.equal(tensor, rank_tensor((3, 4), dtype, src)), f"src={src} {dtype}"
        gpubridge.barrier()


@pytest.mark.parametrize(
    "vendors",
    [
        ["nvidia", "amd", "nvidia", "amd"],
        ["nvidia", "nvidia", "nvidia", "amd"],
        ["amd"] * 3,
    ],
    ids=["interleaved", "3nvidia-1amd", "amd-only"],
)
def test_broadcast_from_every_rank(vendors, tmp_path):
    run_ranks(vendors, _check_broadcast_from_every_rank, tmp_path)


def _check_rejects_bad_src(topology):
    with pytest.raises(ValueError, match="src"):
        gpubridge.broadcast(torch.ones(1), topology.world_size)
    with pytest.raises(ValueError, match="src"):
        gpubridge.broadcast(torch.ones(1), -1)
    gpubridge.barrier()


def test_rejects_bad_src(tmp_path):
    run_ranks(["nvidia", "amd"], _check_rejects_bad_src, tmp_path)
