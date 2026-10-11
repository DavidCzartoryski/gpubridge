"""GPUBRIDGE_SPLIT_TEST: test-only split of a single-vendor job into two islands."""

import os
import warnings
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from _harness import TIMEOUT, expected_sum, rank_tensor, run_failing_init, run_ranks

import gpubridge
from gpubridge.config import (
    CPU_ONLY_ENV,
    SIM_HOSTNAME_ENV,
    SPLIT_TEST_ENV,
    VENDOR_ENV,
    gpu_identity,
    hostname,
    resolve_config,
    split_test_mode,
)
from gpubridge.split_test import SplitTestWarning, banner, split_labels
from gpubridge.topology import gpu_clashes, plan_topology, unchecked_islands


@pytest.fixture(autouse=True)
def no_split_env(monkeypatch):
    monkeypatch.delenv(SPLIT_TEST_ENV, raising=False)


@pytest.mark.parametrize(
    ("value", "mode"),
    [("", None), ("0", None), ("off", None), ("1", "half"), ("yes", "half"),
     ("half", "half"), (" Alternate ", "alternate"), ("node", "node")],
)
def test_the_env_var_is_the_only_switch(monkeypatch, value, mode):
    monkeypatch.setenv(SPLIT_TEST_ENV, value)
    assert split_test_mode() == mode


def test_split_mode_is_off_by_default():
    assert split_test_mode() is None


def test_unknown_split_mode_is_rejected(monkeypatch):
    monkeypatch.setenv(SPLIT_TEST_ENV, "thirds")
    with pytest.raises(ValueError, match="thirds"):
        split_test_mode()


@pytest.mark.parametrize(
    ("world", "mode", "flipped"),
    [
        (2, "half", [1]),
        (3, "half", [2]),
        (4, "half", [2, 3]),
        (4, "alternate", [1, 3]),
        (5, "alternate", [1, 3]),
    ],
)
def test_labels_by_rank(world, mode, flipped):
    labels = split_labels(["nvidia"] * world, ["node0"] * world, mode)
    assert [rank for rank, label in enumerate(labels) if label == "amd"] == flipped
    assert all(labels[rank] == "nvidia" for rank in range(world) if rank not in flipped)


def test_labels_by_node_follow_machines():
    hosts = ["b", "b", "a", "a", "c", "c"]  # host order is by lowest rank: b, a, c
    assert split_labels(["nvidia"] * 6, hosts, "node") == [
        "nvidia", "nvidia", "amd", "amd", "nvidia", "nvidia"
    ]


def test_an_amd_job_is_split_the_other_way():
    assert split_labels(["amd"] * 4, ["n"] * 4, "half") == ["amd", "amd", "nvidia", "nvidia"]


def test_split_always_yields_two_islands():
    for world in range(2, 9):
        for mode in ("half", "alternate"):
            layout = plan_topology(split_labels(["nvidia"] * world, ["n"] * world, mode))
            assert len(layout.islands) == 2, (world, mode)


@pytest.mark.parametrize(
    ("vendors", "hosts", "mode", "match"),
    [
        (["nvidia"], ["n"], "half", "at least 2 ranks"),
        (["nvidia", "amd"], ["n", "n"], "half", "already has amd and nvidia ranks"),
        (["nvidia", "nvidia"], ["n", "n"], "node", "every rank runs on n"),
    ],
)
def test_impossible_splits_are_refused(vendors, hosts, mode, match):
    with pytest.raises(RuntimeError, match=match):
        split_labels(vendors, hosts, mode)


def test_banner_says_it_is_not_a_mixed_vendor_run():
    text = banner("half", ["nvidia"] * 4, ["nvidia", "nvidia", "amd", "amd"])
    assert "TEST-ONLY" in text
    assert "Ranks [2, 3] run nvidia builds but are labeled 'amd'" in text
    assert "NOT a mixed-vendor run" in text


def test_split_test_lets_ranks_share_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("LOCAL_RANK", "1")
    with pytest.raises(RuntimeError, match="sees only 1 GPU"):  # off: a GPU per rank
        resolve_config()
    monkeypatch.setenv(SPLIT_TEST_ENV, "half")
    assert resolve_config().device == torch.device("cuda", 0)


def _check_split_run(topology, labels, mode="half"):
    assert topology.split_test == mode
    assert topology.run_kind == "split-test"
    assert topology.detected_vendor == "nvidia"
    assert topology.vendor == labels[topology.rank]
    assert [peer.vendor for peer in topology.peers] == ["nvidia"] * len(labels)
    assert topology.layout == plan_topology(labels)

    tensor = rank_tensor((3, 4), torch.float32, topology.rank)
    gpubridge.all_reduce(tensor)
    assert torch.equal(tensor, expected_sum((3, 4), torch.float32, topology.world_size))
    for src in range(topology.world_size):
        tensor = rank_tensor((5,), torch.float16, topology.rank)
        gpubridge.broadcast(tensor, src)
        assert torch.equal(tensor, rank_tensor((5,), torch.float16, src))


def test_split_run_forms_two_islands_and_collectives_work(tmp_path, monkeypatch):
    monkeypatch.setenv(SPLIT_TEST_ENV, "half")
    labels = ["nvidia", "nvidia", "amd", "amd"]
    run_ranks(["nvidia"] * 4, partial(_check_split_run, labels=labels), tmp_path)


def _two_machines(rank):
    os.environ[SIM_HOSTNAME_ENV] = f"node{rank // 2}"


def test_node_mode_on_one_machine_with_simulated_hostnames(tmp_path, monkeypatch):
    # How the multi-node job scripts run on CPU: two "machines" on this one.
    monkeypatch.setenv(SPLIT_TEST_ENV, "node")
    labels = ["nvidia", "nvidia", "amd", "amd"]
    run_ranks(["nvidia"] * 4, partial(_check_split_run, labels=labels, mode="node"), tmp_path,
              before_init=_two_machines)


def test_split_on_cuda_builds_needs_no_vendor_variable(tmp_path, monkeypatch):
    # As on an NVIDIA node: the vendor comes from the CUDA build, not GPUBRIDGE_VENDOR.
    monkeypatch.setenv(SPLIT_TEST_ENV, "half")
    labels = ["nvidia", "nvidia", "amd", "amd"]
    run_ranks(["nvidia"] * 4, partial(_check_split_run, labels=labels), tmp_path,
              fake_builds=True)


@pytest.mark.parametrize(("gpus", "devices"), [(4, [0, 1, 2, 3]), (2, [0, 1, 0, 1])])
def test_split_on_one_node_maps_ranks_to_gpus(gpus, devices, monkeypatch):
    # One multi-GPU NVIDIA node: each rank its own GPU; with fewer GPUs than
    # ranks, split-test mode wraps around so ranks share them. Both islands use NCCL.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)
    monkeypatch.delenv(VENDOR_ENV, raising=False)
    monkeypatch.delenv(CPU_ONLY_ENV, raising=False)
    monkeypatch.setenv(SPLIT_TEST_ENV, "half")
    for local_rank, index in enumerate(devices):
        monkeypatch.setenv("LOCAL_RANK", str(local_rank))
        config = resolve_config()
        assert config.device == torch.device("cuda", index)
        assert config.island_backend == "nccl" and not config.simulated


def _props(uuid="", domain=0, bus=0, device=0):
    return SimpleNamespace(uuid=uuid, pci_domain_id=domain, pci_bus_id=bus, pci_device_id=device)


@pytest.mark.parametrize(("props", "want"), [
    (_props("GPU-6b3b5f8c", 0, 0x3B, 0), "uuid GPU-6b3b5f8c, pci 0000:3b:00"),
    (_props("00000000-0000-0000-0000-000000000000", 1, 0xC1, 0), "pci 0001:c1:00"),
    (_props("GPU-6b3b5f8c"), "uuid GPU-6b3b5f8c"),  # PCI address all zero: unknown
    (SimpleNamespace(uuid="GPU-6b3b5f8c"), "uuid GPU-6b3b5f8c"),  # no PCI fields
    (_props(), ""),
])
def test_gpu_identity_is_the_physical_gpu(props, want, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: props)
    assert gpu_identity(torch.device("cuda", 0)) == want


def test_gpu_identity_is_empty_when_unknown(monkeypatch):
    assert gpu_identity(torch.device("cpu")) is None  # not a GPU at all

    def fails(device):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "get_device_properties", fails)
    assert gpu_identity(torch.device("cuda", 0)) == ""


@pytest.mark.parametrize(("per_node", "gpus", "mode", "nodes", "clash"), [
    (4, 4, "half", 1, False),       # step 04
    (4, 4, "alternate", 1, False),  # step 04
    (2, 1, "half", 1, False),       # step 02: two single-rank islands share the GPU
    (2, 2, "half", 1, False),       # a 2-GPU node, a GPU per rank (the GPU tests, rung 2)
    (4, 2, "half", 1, False),       # a 2-GPU node, two ranks per GPU: each half has its own
    (4, 2, "alternate", 1, True),   # ... but alternate puts ranks 0 and 2 on GPU 0
    (6, 2, "half", 1, True),        # more than twice as many ranks as GPUs
    (4, 4, "node", 2, False),       # step 05
    (4, 4, None, 2, False),         # one island over two nodes: same PCI address, other host
    (2, 1, None, 1, True),          # no split test: two ranks on one GPU
])
def test_ranks_of_one_island_never_share_a_gpu(per_node, gpus, mode, nodes, clash):
    # Local rank r drives GPU r % gpus, as split-test mode wraps them.
    world = per_node * nodes
    hosts = [f"node{rank // per_node}" for rank in range(world)]
    ids = [f"pci 0000:{rank % per_node % gpus:02x}:00" for rank in range(world)]
    labels = split_labels(["nvidia"] * world, hosts, mode) if mode else ["nvidia"] * world
    lines = gpu_clashes(plan_topology(labels), hosts, ids)
    assert bool(lines) == clash, lines
    if (per_node, gpus, mode) == (4, 2, "alternate"):
        assert lines == ["  island 'nvidia': ranks [0, 2] on node0, GPU pci 0000:00:00",
                         "  island 'amd': ranks [1, 3] on node0, GPU pci 0000:01:00"]


@pytest.mark.parametrize(("gpus", "unchecked"), [
    (["", ""], ["  island 'nvidia': ranks [0, 1]"]),
    (["pci 0000:00:00", "pci 0000:00:00", ""], ["  island 'nvidia': ranks [2]"]),
    ([None, None], []),  # CPU ranks: nothing to check
])
def test_an_island_with_an_unknown_gpu_is_not_checked(gpus, unchecked):
    layout = plan_topology(["nvidia"] * len(gpus))
    assert gpu_clashes(layout, ["node0"] * len(gpus), gpus) == []
    assert unchecked_islands(layout, gpus) == unchecked


def test_only_the_island_with_an_unknown_gpu_goes_unchecked():
    layout = plan_topology(["nvidia", "nvidia", "amd", "amd"])
    gpus = ["pci 0000:00:00", "pci 0000:00:00", "", ""]
    assert gpu_clashes(layout, ["node0"] * 4, gpus) == [
        "  island 'nvidia': ranks [0, 1] on node0, GPU pci 0000:00:00"]
    assert unchecked_islands(layout, gpus) == ["  island 'amd': ranks [2, 3]"]


def test_a_layout_that_shares_a_gpu_fails_init_on_every_rank(tmp_path):
    # Four ranks on a 2-GPU node with alternate: NCCL would refuse both islands.
    env = {VENDOR_ENV: "nvidia", SPLIT_TEST_ENV: "alternate"}
    setups = [{"env": env, "gpu_identity": f"pci 0000:{rank % 2:02x}:00"} for rank in range(4)]
    run_failing_init(setups, tmp_path, [
        "ranks of one island would share a GPU", r"ranks \[0, 2\]", r"ranks \[1, 3\]",
        f"{SPLIT_TEST_ENV}=half with at most twice as many ranks as GPUs"])


def _on_two_gpus(rank):
    gpubridge.topology.gpu_identity = lambda device: f"pci 0000:{rank % 2:02x}:00"


def test_half_on_a_two_gpu_node_passes_the_check(tmp_path, monkeypatch):
    monkeypatch.setenv(SPLIT_TEST_ENV, "half")
    labels = ["nvidia", "nvidia", "amd", "amd"]
    run_ranks(["nvidia"] * 4, partial(_check_split_run, labels=labels), tmp_path,
              before_init=_on_two_gpus)


_WARNINGS: list[str] = []


def _no_gpu_identity(rank):
    # As on a ROCm build whose GPUs report neither a UUID nor a PCI address.
    gpubridge.topology.gpu_identity = lambda device: ""
    warnings.simplefilter("always")
    warnings.showwarning = lambda message, *args, **kwargs: _WARNINGS.append(str(message))


def _check_unchecked_run(topology):
    _check_split_run(topology, ["nvidia", "amd", "nvidia", "amd"], mode="alternate")
    skipped = [w for w in _WARNINGS if "could not check that ranks" in w]
    if topology.rank == 0:
        assert len(skipped) == 1, _WARNINGS
        assert "island 'nvidia': ranks [0, 2]" in skipped[0]
        assert "island 'amd': ranks [1, 3]" in skipped[0]
    else:
        assert skipped == []


def test_init_warns_and_skips_the_check_when_no_rank_knows_its_gpu(tmp_path, monkeypatch):
    # The layout that fails above on a 2-GPU node, but no rank can say which GPU
    # it drives: init() must warn, not fail.
    monkeypatch.setenv(SPLIT_TEST_ENV, "alternate")
    run_ranks(["nvidia"] * 4, _check_unchecked_run, tmp_path, before_init=_no_gpu_identity)


def test_simulated_hostname_is_ignored_on_gpus(monkeypatch):
    monkeypatch.setenv(SIM_HOSTNAME_ENV, "node7")
    assert hostname(simulated=True) == "node7"
    assert hostname(simulated=False) != "node7"


def _check_normal_run(topology):
    assert topology.split_test is None
    assert topology.run_kind == "cpu"
    assert topology.vendor == topology.detected_vendor


def test_normal_runs_are_not_marked_as_split(tmp_path):
    run_ranks(["nvidia", "amd"], _check_normal_run, tmp_path)


def _init_and_record_warnings(rank, world, init_file, results):
    os.environ[VENDOR_ENV] = "nvidia"
    os.environ[SPLIT_TEST_ENV] = "alternate"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        topology = gpubridge.init(init_method=f"file://{init_file}", rank=rank,
                                  world_size=world, timeout=TIMEOUT)
    try:
        split_warnings = [w for w in caught if issubclass(w.category, SplitTestWarning)]
        assert len(split_warnings) == 1, [str(w.message) for w in caught]
        assert "NOT a mixed-vendor run" in str(split_warnings[0].message)
        assert topology.run_kind == "split-test"
    finally:
        gpubridge.destroy()


def test_every_rank_warns(tmp_path):
    mp.spawn(_init_and_record_warnings, args=(3, str(tmp_path / "rdzv"), None), nprocs=3)


def test_ranks_must_agree_on_the_split_mode(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", SPLIT_TEST_ENV: "half"}},
         {"env": {VENDOR_ENV: "nvidia"}}],
        tmp_path,
        [rf"disagree on {SPLIT_TEST_ENV}", r"half on ranks \[0\]", r"off on ranks \[1\]"],
    )


def test_a_real_mixed_job_cannot_be_split(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", SPLIT_TEST_ENV: "half"}},
         {"env": {VENDOR_ENV: "amd", SPLIT_TEST_ENV: "half"}}],
        tmp_path,
        [r"already has amd and nvidia ranks"],
    )


def test_node_mode_needs_two_machines(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", SPLIT_TEST_ENV: "node"}}] * 2,
        tmp_path,
        [r"splits islands by machine"],
    )
