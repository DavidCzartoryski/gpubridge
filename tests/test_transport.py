"""The bridge transport is swappable, and Gloo stays the default."""

import pytest
import torch
from _harness import expected_sum, rank_tensor, run_failing_init, run_ranks
from _transports import STORE_ENV, StoreTransport

import gpubridge
from gpubridge.config import VENDOR_ENV
from gpubridge.transport import (
    DEFAULT_TRANSPORT,
    BridgeTransport,
    GlooTransport,
    register_transport,
    resolve_transport,
)


def test_gloo_is_the_default():
    assert DEFAULT_TRANSPORT == "gloo"
    assert resolve_transport("gloo") is GlooTransport


def test_resolve_accepts_names_and_classes():
    assert resolve_transport("test-store") is StoreTransport
    assert resolve_transport(StoreTransport) is StoreTransport
    with pytest.raises(ValueError, match="known: gloo"):
        resolve_transport("carrier-pigeon")
    with pytest.raises(TypeError):
        resolve_transport(42)


def test_register_rejects_non_transports_and_name_clashes():
    with pytest.raises(TypeError):
        register_transport(object)

    class Impostor(StoreTransport):
        name = "gloo"

    with pytest.raises(ValueError, match="already registered"):
        register_transport(Impostor)
    assert register_transport(StoreTransport) is StoreTransport  # re-registering is fine


def _check_default_bridge(topology):
    if topology.layout.needs_bridge and topology.is_leader:
        assert isinstance(topology.bridge, GlooTransport)
        assert topology.bridge_group is topology.bridge.group
    else:
        assert topology.bridge is None
        assert topology.bridge_group is None
    assert all(peer.bridge == "gloo" for peer in topology.peers)


@pytest.mark.parametrize("vendors", [["nvidia", "amd", "nvidia", "amd"], ["amd", "amd"]],
                         ids=["mixed", "single-vendor"])
def test_default_bridge_is_gloo(vendors, tmp_path):
    run_ranks(vendors, _check_default_bridge, tmp_path)


def _check_store_bridge(topology):
    cases = [((3, 4), torch.float32), ((5,), torch.float16)]
    for shape, dtype in cases:
        tensor = rank_tensor(shape, dtype, topology.rank)
        gpubridge.all_reduce(tensor)
        assert torch.equal(tensor, expected_sum(shape, dtype, topology.world_size))
    for src in range(topology.world_size):
        tensor = rank_tensor((3, 4), torch.float32, topology.rank)
        gpubridge.broadcast(tensor, src)
        assert torch.equal(tensor, rank_tensor((3, 4), torch.float32, src))

    if topology.is_leader:
        # Every bridge hop went through the store transport, none through Gloo.
        assert isinstance(topology.bridge, StoreTransport)
        assert topology.bridge_group is None
        assert topology.bridge.calls == {"all_reduce": len(cases),
                                         "broadcast": topology.world_size}
    else:
        assert topology.bridge is None


@pytest.mark.parametrize(
    ("vendors", "bridge"),
    [
        (["nvidia", "nvidia", "amd", "amd"], StoreTransport),
        (["nvidia", "nvidia", "nvidia", "amd"], StoreTransport),
        (["amd", "nvidia", "amd"], "test-store"),
    ],
    ids=["2+2-class", "3+1-class", "interleaved-by-name"],
)
def test_collectives_run_over_a_swapped_in_transport(vendors, bridge, tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "bridge-store"))
    run_ranks(vendors, _check_store_bridge, tmp_path, init_kwargs={"bridge": bridge})


def test_ranks_with_different_transports_fail_init_on_every_rank(tmp_path, monkeypatch):
    monkeypatch.setenv(STORE_ENV, str(tmp_path / "bridge-store"))
    run_failing_init(
        [
            {"env": {VENDOR_ENV: "nvidia"}, "init_kwargs": {"bridge": "gloo"}},
            {"env": {VENDOR_ENV: "amd"}, "init_kwargs": {"bridge": StoreTransport}},
        ],
        tmp_path,
        [r"different bridge transports", r"'gloo' on ranks \[0\]",
         r"'test-store' on ranks \[1\]"],
    )


def test_bridge_transport_is_abstract():
    with pytest.raises(TypeError):
        BridgeTransport()


def test_init_rejects_an_unknown_bridge_before_joining():
    with pytest.raises(ValueError, match="unknown bridge transport"):
        gpubridge.init(bridge="carrier-pigeon")
    assert not gpubridge.is_initialized()
