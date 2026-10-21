"""CPU contract tests for benchmark timing, input bounds and payload generation."""

import json
import queue
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gpu"))
import bench_p2p  # noqa: E402


@pytest.mark.parametrize("warmup,trials", [(0, 1), (0, 3), (2, 3)])
@pytest.mark.parametrize("verification_cost", [0.0, 10.0])
@pytest.mark.parametrize("corrupt", [False, True])
def test_verification_is_bidirectional_and_cannot_change_timing(
        monkeypatch, warmup, trials, verification_cost, corrupt):
    # Execute both real benchmark functions. Queue rendezvous fixes the order;
    # this virtual clock charges 1 ms per transport call and arbitrary time to
    # pattern creation/equality, with no wall-clock performance assertions.
    messages = [queue.Queue(), queue.Queue()]
    roles = threading.local()
    clock, calls = [0.0], Counter()
    lock = threading.Lock()
    original_pattern, original_equal = bench_p2p.pattern, torch.equal

    def charge(seconds):
        with lock:
            clock[0] += seconds

    def pattern(numel, device):
        calls[(roles.rank, "pattern")] += 1
        charge(verification_cost)
        return original_pattern(numel, device)

    def equal(a, b):
        calls[(roles.rank, "equal")] += 1
        charge(verification_cost)
        return original_equal(a, b)

    def send(tensor, peer, **kwargs):
        payload = tensor.clone()
        if corrupt and roles.rank == 0 and calls[(0, "send")] == 0:
            payload[0] = -10
        calls[(roles.rank, "send")] += 1
        charge(0.001)
        messages[peer].put(payload)

    def recv(tensor, peer, **kwargs):
        tensor.copy_(messages[roles.rank].get(timeout=5))
        charge(0.001)

    monkeypatch.setattr(bench_p2p, "pattern", pattern)
    monkeypatch.setattr(bench_p2p, "torch", SimpleNamespace(
        empty=torch.empty, empty_like=torch.empty_like, float32=torch.float32, equal=equal,
        arange=torch.arange, int64=torch.int64))
    monkeypatch.setattr(bench_p2p, "dist", SimpleNamespace(send=send, recv=recv))
    monkeypatch.setattr(bench_p2p, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(bench_p2p, "sync", lambda device: None)
    device = torch.device("cpu")

    def peer():
        roles.rank = 1
        return bench_p2p.echo(31, device, rounds=warmup + trials)

    roles.rank = 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(peer)
        samples, outgoing_ok = bench_p2p.ping_pong(1, 31, device, warmup=warmup, trials=trials)
        incoming_ok = future.result(timeout=5)
    assert outgoing_ok is (not corrupt) and incoming_ok is (not corrupt)
    assert samples == pytest.approx([0.004] * trials)
    assert calls[(0, "equal")] == calls[(1, "equal")] == 1
    assert calls[(0, "pattern")] == calls[(1, "pattern")] == 1
    assert calls[(0, "send")] == calls[(1, "send")] == 1 + warmup + trials


@pytest.mark.parametrize("arguments", [
    ["--factor", "1"], ["--factor", "0"], ["--factor", "-1"],
    ["--min-bytes", "0"], ["--min-bytes", "-1"],
    ["--max-bytes", "0"], ["--min-bytes", "4K", "--max-bytes", "1K"],
    ["--trials", "0"], ["--trials", "-1"], ["--warmup", "-1"],
    ["--timeout", "0"], ["--timeout", "-1"], ["--timeout", "nan"], ["--timeout", "inf"],
])
def test_invalid_bounds_fail_before_process_group_initialization(monkeypatch, tmp_path, arguments):
    initialized = []
    monkeypatch.setattr(bench_p2p.dist, "init_process_group", lambda *a, **kw: initialized.append(1))
    monkeypatch.setattr(sys, "argv", ["bench_p2p.py", "--out", str(tmp_path), *arguments])
    with pytest.raises(SystemExit) as error:
        bench_p2p.main()
    assert error.value.code == 2 and initialized == []


@pytest.mark.parametrize("offset", [2**24 - 3, 2**24, 2**28 - 5])
def test_pattern_residues_remain_exact_at_large_offsets_without_large_allocations(offset):
    chunk = bench_p2p._pattern_chunk(offset, 17)
    assert chunk.dtype == torch.float32
    assert chunk.tolist() == [(offset + i) % 251 for i in range(17)]
    assert bool((chunk[1:] != chunk[:-1]).all())


def test_pattern_uses_bounded_chunks_including_the_tail(monkeypatch):
    original = bench_p2p._pattern_chunk
    calls = []

    def chunk(offset, count):
        calls.append((offset, count))
        return original(offset, count)

    monkeypatch.setattr(bench_p2p, "PATTERN_CHUNK", 7)
    monkeypatch.setattr(bench_p2p, "_pattern_chunk", chunk)
    result = bench_p2p.pattern(31, torch.device("cpu"))
    assert calls == [(0, 7), (7, 7), (14, 7), (21, 7), (28, 3)]
    assert result.tolist() == list(range(31))


def test_the_handshake_benchmark_runs_every_phase_on_both_stores():
    # scripts/bench_p2p_handshake.py, this p2p.py against itself, one short run per arm.
    repo = Path(__file__).resolve().parents[1]
    for store in ("tcp", "file"):
        result = subprocess.run(
            [sys.executable, str(repo / "scripts" / "bench_p2p_handshake.py"), "--before",
             str(repo / "src" / "gpubridge" / "p2p.py"), "--store", store, "--iterations", "5",
             "--warmup", "1", "--idle", "0.1", "--missing", "2", "--hold", "0.1",
             "--repeats", "1"], cwd=repo, capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, result.stderr
        out = json.loads(result.stdout)
        assert out["sources"]["before"]["sha256"] == out["sources"]["after"]["sha256"]
        for arm in ("before", "after"):
            summary = out["summary"][arm]
            assert summary["pingpong"]["one_way_us_median"] > 0
            assert summary["pingpong"]["store_calls"] > 0
            assert summary["missing"]["waiting_ended_by_close"] == [2]  # close() ends them
