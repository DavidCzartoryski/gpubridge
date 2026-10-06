"""CPU-only mode: vendor detected from the PyTorch build, islands on Gloo, tensors on CPU."""

import torch
from _harness import expected_sum, rank_tensor, run_ranks

import gpubridge
from gpubridge.topology import plan_topology

VENDORS = ["nvidia", "amd", "nvidia", "amd"]


def _check_cpu_only(topology):
    assert topology.simulated
    assert topology.device == torch.device("cpu")
    assert [peer.vendor for peer in topology.peers] == VENDORS
    assert topology.layout == plan_topology(VENDORS)

    tensor = rank_tensor((3, 4), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((3, 4), torch.float32, topology.world_size))

    for src in range(topology.world_size):
        tensor = rank_tensor((5,), torch.float16, topology.rank)
        gpubridge.broadcast(tensor, src)
        assert torch.equal(tensor, rank_tensor((5,), torch.float16, src))


def test_vendors_come_from_the_build(tmp_path):
    run_ranks(VENDORS, _check_cpu_only, tmp_path, fake_builds=True)
