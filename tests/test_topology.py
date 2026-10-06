from functools import partial

import pytest
import torch.distributed as dist
from _harness import run_ranks

from gpubridge.topology import Island, PeerInfo, plan_topology, validate_peers


def test_two_by_two():
    layout = plan_topology(["nvidia", "nvidia", "amd", "amd"])
    assert layout.islands == (Island("nvidia", (0, 1)), Island("amd", (2, 3)))
    assert layout.bridge_ranks == (0, 2)
    assert layout.needs_bridge


def test_interleaved_vendors():
    layout = plan_topology(["nvidia", "amd", "nvidia", "amd"])
    assert layout.islands == (Island("nvidia", (0, 2)), Island("amd", (1, 3)))
    assert layout.bridge_ranks == (0, 1)
    assert layout.island_of(3) == Island("amd", (1, 3))


def test_islands_are_ordered_by_leader():
    layout = plan_topology(["amd", "nvidia", "nvidia", "nvidia"])
    assert [island.vendor for island in layout.islands] == ["amd", "nvidia"]
    assert layout.bridge_ranks == (0, 1)


def test_uneven_islands():
    layout = plan_topology(["nvidia", "nvidia", "nvidia", "amd"])
    assert layout.islands == (Island("nvidia", (0, 1, 2)), Island("amd", (3,)))
    assert layout.bridge_ranks == (0, 3)


def test_single_vendor_has_no_bridge():
    layout = plan_topology(["amd"] * 4)
    assert layout.islands == (Island("amd", (0, 1, 2, 3)),)
    assert not layout.needs_bridge


def test_leader_election_is_deterministic():
    vendors = ["amd", "nvidia", "amd", "nvidia", "nvidia", "amd"]
    layouts = {plan_topology(vendors) for _ in range(20)}
    assert len(layouts) == 1
    (layout,) = layouts
    for island in layout.islands:
        assert island.leader == min(island.ranks)
        assert all(vendors[rank] == island.vendor for rank in island.ranks)
    assert sorted(rank for island in layout.islands for rank in island.ranks) == list(range(6))


def test_empty_cluster_is_rejected():
    with pytest.raises(ValueError):
        plan_topology([])


def _peer(simulated: bool = True, torch_version: str = "2.5.1") -> PeerInfo:
    return PeerInfo("nvidia", simulated, torch_version, "node0")


def test_mixed_simulation_modes_are_rejected():
    with pytest.raises(RuntimeError, match=r"ranks \[0, 2\] but not on ranks \[1\]"):
        validate_peers([_peer(simulated=True), _peer(simulated=False), _peer(simulated=True)])


def test_different_torch_releases_warn():
    with pytest.warns(UserWarning, match="2.4.0, 2.5.1"):
        validate_peers([_peer(torch_version="2.5.1+cu124"), _peer(torch_version="2.4.0+rocm6.1")])


def test_same_release_on_both_builds_is_quiet(recwarn):
    validate_peers([_peer(torch_version="2.5.1+cu124"), _peer(torch_version="2.5.1+rocm6.2")])
    assert len(recwarn) == 0


def _check_topology(topology, vendors):
    expected = plan_topology(vendors)
    assert topology.layout == expected
    assert topology.vendor == vendors[topology.rank]
    assert [peer.vendor for peer in topology.peers] == list(vendors)
    assert all(peer.simulated for peer in topology.peers)
    assert topology.is_leader == (topology.rank in expected.bridge_ranks)
    assert dist.get_process_group_ranks(topology.island_group) == list(topology.island.ranks)
    if expected.needs_bridge and topology.is_leader:
        assert dist.get_process_group_ranks(topology.bridge_group) == list(expected.bridge_ranks)
    else:
        assert topology.bridge_group is None

    # Every rank must have elected the same leaders.
    layouts = [None] * topology.world_size
    dist.all_gather_object(layouts, topology.layout)
    assert all(layout == topology.layout for layout in layouts)


@pytest.mark.parametrize(
    "vendors",
    [
        ["nvidia", "nvidia", "amd", "amd"],
        ["amd", "nvidia", "amd", "nvidia"],
        ["nvidia", "nvidia", "nvidia", "amd"],
        ["nvidia", "nvidia", "nvidia"],
    ],
    ids=["2nvidia-2amd", "interleaved", "3nvidia-1amd", "nvidia-only"],
)
def test_every_rank_builds_the_same_topology(vendors, tmp_path):
    run_ranks(vendors, partial(_check_topology, vendors=vendors), tmp_path)
