"""GPUBRIDGE_SPLIT_TEST: test-only split of a single-vendor job into two islands."""

import os
import warnings
from functools import partial

import pytest
import torch
import torch.multiprocessing as mp
from _harness import TIMEOUT, expected_sum, rank_tensor, run_failing_init, run_ranks

import gpubridge
from gpubridge.config import SPLIT_TEST_ENV, VENDOR_ENV, resolve_config, split_test_mode
from gpubridge.split_test import SplitTestWarning, banner, split_labels
from gpubridge.topology import plan_topology


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


def _check_split_run(topology, labels):
    assert topology.split_test == "half"
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
