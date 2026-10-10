"""The pipelined bridge, policy settings checked in discovery, and the auto-tuned policy."""

import json
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest
import torch
from _harness import rank_tensor, run_failing_init, run_ranks
from _staging import LazyPipelined, LazyStaging

import gpubridge
from gpubridge import ReduceOp
from gpubridge.config import CHUNK_BYTES_ENV, SPLIT_TEST_ENV, THRESHOLDS_ENV, VENDOR_ENV
from gpubridge.policies import (
    AutoTuned,
    FlatGloo,
    PipelinedReduceBridgeBroadcast,
    load_thresholds,
)
from gpubridge.staging import DirectStaging, make_staging

PIPELINED = PipelinedReduceBridgeBroadcast.name
REPO = Path(__file__).resolve().parents[1]


# ---- the pipeline's ordering, with copies that run only when waited for ---------------

def _check_lazy_pipeline(topology):
    rank, world = topology.rank, topology.world_size
    reference = FlatGloo()
    # 16-byte chunks: 37 float32 elements make 10 chunks, so all 3 slots are reused.
    for dtype, shape in [(torch.float32, (37,)), (torch.float16, (5, 7)), (torch.int64, (9,)),
                         (torch.int8, (20,)), (torch.float64, (1,))]:
        for op in (ReduceOp.SUM, ReduceOp.MAX, ReduceOp.MIN):
            tensor = rank_tensor(shape, dtype, rank)
            want = tensor.clone()
            gpubridge.all_reduce(tensor, op=op)
            reference.all_reduce(want, topology, op=op)
            assert torch.equal(tensor, want), (dtype, op)
        for src in range(world):
            tensor = rank_tensor(shape, dtype, rank)
            want = tensor.clone()
            gpubridge.broadcast(tensor, src)
            reference.broadcast(want, src, topology)
            assert torch.equal(tensor, want), (dtype, src)

    staging = topology.policy.staging
    if not topology.is_leader:
        assert staging is None  # only leaders touch the bridge
        return
    assert isinstance(staging, LazyStaging)
    assert not staging.out.queue and not staging.back.queue, "a copy never ran"
    log = staging.log
    # Copies out into a reused slot wait (on the device) for the copy back out of it.
    assert any(entry[0] == "to_host" and entry[2] is not None for entry in log)
    # The only host waits are for copies, never for the whole device.
    assert {entry[0] for entry in log} <= {"begin", "to_host", "wait_host", "to_device",
                                           "end", "ran"}


@pytest.mark.parametrize("vendors", [["nvidia", "nvidia", "amd", "amd"],
                                     ["amd", "nvidia", "nvidia", "nvidia"],
                                     ["nvidia", "amd", "nvidia", "amd", "amd"]],
                         ids=["2+2", "1+3", "interleaved-2+3"])
def test_lazy_copies_still_give_flat_gloos_answer(vendors, tmp_path, monkeypatch):
    monkeypatch.setenv(CHUNK_BYTES_ENV, "16")
    run_ranks(vendors, _check_lazy_pipeline, tmp_path, init_kwargs={"policy": LazyPipelined})


class ForgetfulPipelined(LazyPipelined):
    """Skips the device-side wait before reusing a slot: the bug the protocol prevents."""

    name = "test-forgetful-pipelined"

    def setup(self, topology):
        super().setup(topology)
        if self.staging is not None:
            real = self.staging.to_host
            self.staging.to_host = lambda slot, chunk, after: real(slot, chunk, None)


def _check_forgetful(topology, results):
    tensor = rank_tensor((37,), torch.float32, topology.rank)
    want = tensor.clone()
    gpubridge.all_reduce(tensor)
    FlatGloo().all_reduce(want, topology)
    if topology.rank == 0:
        Path(results).write_text(json.dumps(bool(torch.equal(tensor, want))))


def test_the_lazy_staging_catches_a_missing_wait(tmp_path, monkeypatch):
    # Proof that the test above can fail: without the slot-reuse wait, the
    # lazy copies run in the wrong order and the answer is wrong.
    monkeypatch.setenv(CHUNK_BYTES_ENV, "16")
    results = tmp_path / "matched.json"
    run_ranks(["nvidia", "amd"], partial(_check_forgetful, results=str(results)), tmp_path,
              init_kwargs={"policy": ForgetfulPipelined})
    assert json.loads(results.read_text()) is False


def _check_direct_staging(topology):
    policy = topology.policy
    assert policy.chunk_bytes == 24
    if topology.is_leader:
        assert isinstance(policy.staging, DirectStaging)
    tensor = rank_tensor((50,), torch.float32, topology.rank)
    want = tensor.clone()
    gpubridge.all_reduce(tensor)
    FlatGloo().all_reduce(want, topology)
    assert torch.equal(tensor, want)
    gpubridge.destroy()
    assert policy.staging is None  # close() released it


def test_on_cpu_the_pipeline_bridges_chunks_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv(CHUNK_BYTES_ENV, "24")
    run_ranks(["nvidia", "amd", "amd"], _check_direct_staging, tmp_path,
              init_kwargs={"policy": PIPELINED})


def test_cpu_devices_get_direct_staging():
    assert isinstance(make_staging(torch.device("cpu"), 64, 3), DirectStaging)


# ---- settings are part of discovery ---------------------------------------------------

def test_ranks_with_different_chunk_sizes_fail_init_on_every_rank(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", CHUNK_BYTES_ENV: "1M"},
          "init_kwargs": {"policy": PIPELINED}},
         {"env": {VENDOR_ENV: "amd", CHUNK_BYTES_ENV: "2M"},
          "init_kwargs": {"policy": PIPELINED}}],
        tmp_path,
        [rf"disagree on the settings of collective policy '{PIPELINED}'",
         r'\{"chunk_bytes": 1048576\} on ranks \[0\]',
         r'\{"chunk_bytes": 2097152\} on ranks \[1\]'],
    )


def test_a_bad_chunk_size_fails_init_on_every_rank(tmp_path):
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia"}, "init_kwargs": {"policy": PIPELINED}},
         {"env": {VENDOR_ENV: "amd", CHUNK_BYTES_ENV: "12"},
          "init_kwargs": {"policy": PIPELINED}}],
        tmp_path,
        [r"1 of 2 ranks reported problems",
         rf"rank 1 \([^)]*\): Collective policy '{PIPELINED}': {CHUNK_BYTES_ENV}='12' must be a "
         r"positive multiple of 8 bytes"],
    )


@pytest.mark.parametrize(("raw", "expected"), [("4M", 4 * 2**20), ("64K", 65536),
                                               ("65536", 65536), (" 1g ", 2**30)])
def test_chunk_sizes_parse(raw, expected, monkeypatch):
    monkeypatch.setenv(CHUNK_BYTES_ENV, raw)
    assert PipelinedReduceBridgeBroadcast.settings() == {"chunk_bytes": expected}


def test_the_default_chunk_size(monkeypatch):
    monkeypatch.delenv(CHUNK_BYTES_ENV, raising=False)
    assert PipelinedReduceBridgeBroadcast.settings() == {"chunk_bytes": 4 * 2**20}
    monkeypatch.setenv(CHUNK_BYTES_ENV, "lots")
    with pytest.raises(ValueError, match="'lots' is not a size"):
        PipelinedReduceBridgeBroadcast.settings()


# ---- auto-tuned --------------------------------------------------------------------------

class Policies(gpubridge.CollectiveObserver):
    def __init__(self):
        self.used = []

    def on_collective(self, record):
        self.used.append((record.nbytes, record.policy))


def _check_auto_tuned_default(topology):
    assert topology.policy_name == "auto-tuned"
    assert topology.policy_settings == {"rules": [{"max_bytes": None, "policy": "auto"}],
                                        "chunk_bytes": 4 * 2**20}
    seen = Policies()
    gpubridge.add_observer(seen)
    tensor = rank_tensor((300,), torch.float32, topology.rank)
    want = tensor.clone()
    gpubridge.all_reduce(tensor)
    FlatGloo().all_reduce(want, topology)
    assert torch.equal(tensor, want)
    expected = "reduce-bridge-broadcast" if topology.layout.needs_bridge else "native-only"
    assert seen.used == [(1200, expected)]  # what auto would pick


@pytest.mark.parametrize("vendors", [["nvidia", "amd", "amd"], ["amd", "amd"]],
                         ids=["mixed", "single-vendor"])
def test_auto_tuned_without_a_file_does_what_auto_does(vendors, tmp_path, monkeypatch):
    monkeypatch.delenv(THRESHOLDS_ENV, raising=False)
    run_ranks(vendors, _check_auto_tuned_default, tmp_path, init_kwargs={"policy": "auto-tuned"})


def write_thresholds(path, rules, chunk_bytes=16):
    path.write_text(json.dumps({"gpubridge_thresholds": 1, "rules": rules,
                                "chunk_bytes": chunk_bytes}))
    return str(path)


RULES = [{"max_bytes": 64, "policy": "flat-gloo"},
         {"max_bytes": 1024, "policy": "reduce-bridge-broadcast"},
         {"max_bytes": None, "policy": PIPELINED}]


def _check_auto_tuned_rules(topology):
    seen = Policies()
    gpubridge.add_observer(seen)
    reference = FlatGloo()
    for numel in (4, 16, 200, 256, 257, 3000):  # 16, 64, 800, 1024, 1028, 12000 bytes
        tensor = rank_tensor((numel,), torch.float32, topology.rank)
        want = tensor.clone()
        gpubridge.all_reduce(tensor)
        reference.all_reduce(want, topology)
        assert torch.equal(tensor, want), numel
    assert [policy for _, policy in seen.used] == [
        "flat-gloo", "flat-gloo", "reduce-bridge-broadcast", "reduce-bridge-broadcast",
        PIPELINED, PIPELINED]
    pipelined = topology.policy.rules[-1][1]
    assert isinstance(pipelined, PipelinedReduceBridgeBroadcast) and pipelined.chunk_bytes == 16
    out = torch.empty(topology.world_size * 300)
    gpubridge.all_gather_into_tensor(out, torch.full((300,), float(topology.rank)))
    assert out.view(topology.world_size, 300)[:, 0].tolist() == [
        float(r) for r in range(topology.world_size)]


def test_auto_tuned_follows_the_thresholds_file(tmp_path, monkeypatch):
    monkeypatch.setenv(THRESHOLDS_ENV, write_thresholds(tmp_path / "t.json", RULES))
    run_ranks(["nvidia", "amd", "nvidia", "amd"], _check_auto_tuned_rules, tmp_path,
              init_kwargs={"policy": "auto-tuned"})


def test_a_rule_that_does_not_fit_the_cluster_fails_init_on_every_rank(tmp_path):
    path = write_thresholds(tmp_path / "t.json", [{"max_bytes": None, "policy": "native-only"}])
    setup = {"env": {THRESHOLDS_ENV: path}, "init_kwargs": {"policy": "auto-tuned"}}
    run_failing_init(
        [{**setup, "env": {**setup["env"], VENDOR_ENV: v}} for v in ("nvidia", "amd")],
        tmp_path,
        [rf"{THRESHOLDS_ENV}: the rule for sizes up to None bytes can't be used here",
         r"'native-only' doesn't apply to this cluster"],
    )


def test_ranks_with_different_thresholds_fail_init_on_every_rank(tmp_path):
    a = write_thresholds(tmp_path / "a.json", RULES)
    b = write_thresholds(tmp_path / "b.json", RULES[1:])
    run_failing_init(
        [{"env": {VENDOR_ENV: "nvidia", THRESHOLDS_ENV: a},
          "init_kwargs": {"policy": "auto-tuned"}},
         {"env": {VENDOR_ENV: "amd", THRESHOLDS_ENV: b},
          "init_kwargs": {"policy": "auto-tuned"}}],
        tmp_path,
        [r"disagree on the settings of collective policy 'auto-tuned'"],
    )


def test_identical_files_at_different_paths_agree(tmp_path, monkeypatch):
    # Settings hold the rules, not the path, so nodes may keep the file anywhere.
    a = write_thresholds(tmp_path / "a.json", RULES)
    monkeypatch.setenv(THRESHOLDS_ENV, a)
    settings_a = AutoTuned.settings()
    monkeypatch.setenv(THRESHOLDS_ENV, write_thresholds(tmp_path / "copy.json", RULES))
    assert AutoTuned.settings() == settings_a


@pytest.mark.parametrize(
    ("content", "match"),
    [(None, "can't read it"),
     ("{not json", "not valid JSON"),
     ('{"rules": []}', "not a gpubridge thresholds file"),
     ('{"gpubridge_thresholds": 1, "rules": []}', "non-empty list of rules"),
     ('{"gpubridge_thresholds": 1, "rules": [{"max_bytes": 5, "policy": "flat-gloo"}]}',
      "last rule must cover every size"),
     ('{"gpubridge_thresholds": 1, "rules": [{"max_bytes": null, "policy": "chunky"}]}',
      "names policy 'chunky'"),
     ('{"gpubridge_thresholds": 1, "rules": [{"max_bytes": null, "policy": "auto-tuned"}]}',
      "names policy 'auto-tuned'"),
     ('{"gpubridge_thresholds": 1, "rules": [{"max_bytes": 9, "policy": "flat-gloo"}, '
      '{"max_bytes": 4, "policy": "flat-gloo"}, {"max_bytes": null, "policy": "auto"}]}',
      "larger than the previous"),
     ('{"gpubridge_thresholds": 1, "chunk_bytes": 7, '
      '"rules": [{"max_bytes": null, "policy": "auto"}]}', "multiple of 8")],
)
def test_bad_thresholds_files_are_explained(content, match, tmp_path):
    path = tmp_path / "t.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_thresholds(str(path), 64)


def test_a_missing_thresholds_file_fails_init_on_every_rank(tmp_path):
    missing = str(tmp_path / "nowhere.json")
    run_failing_init(
        [{"env": {VENDOR_ENV: v, THRESHOLDS_ENV: missing}, "init_kwargs": {"policy": "auto-tuned"}}
         for v in ("nvidia", "amd")],
        tmp_path,
        [r"2 of 2 ranks reported problems", rf"Collective policy 'auto-tuned': {THRESHOLDS_ENV}="
         r"'.*nowhere\.json': can't read it"],
    )


# ---- bench_all_reduce.py --write-thresholds feeds auto-tuned -----------------------------

def _check_measured_file(topology):
    assert topology.policy_name == "auto-tuned"
    tensor = rank_tensor((1000,), torch.float32, topology.rank)
    want = tensor.clone()
    gpubridge.all_reduce(tensor)
    FlatGloo().all_reduce(want, topology)
    assert torch.equal(tensor, want)


def test_bench_writes_a_thresholds_file_auto_tuned_can_load(tmp_path, monkeypatch):
    out, path = tmp_path / "bench", tmp_path / "thresholds.json"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
           "--master-addr=127.0.0.1", f"--master-port={_free_port()}", "--nproc-per-node=4",
           str(REPO / "scripts" / "gpu" / "bench_all_reduce.py"), "--out", str(out),
           "--max-bytes", "16K", "--warmup", "1", "--trials", "3", "--ops", "gpubridge",
           "--write-thresholds", str(path)]
    env = {**_clean_env(), VENDOR_ENV: "nvidia", SPLIT_TEST_ENV: "half", CHUNK_BYTES_ENV: "1K"}
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    data = json.loads(path.read_text())
    assert data["chunk_bytes"] == 1024 and data["measured"]["split_test"] == "half"
    assert data["rules"][-1]["max_bytes"] is None
    assert {r["policy"] for r in data["rules"]} <= {
        "flat-gloo", "reduce-bridge-broadcast", PIPELINED, "sharded-bridge"}
    rows = json.loads((out / "bench.json").read_text())["rows"]
    assert {r["op"] for r in rows} == {"gpubridge", "policy:flat-gloo",
                                       "policy:reduce-bridge-broadcast", f"policy:{PIPELINED}",
                                       "policy:sharded-bridge"}
    assert load_thresholds(str(path), 64)["rules"] == data["rules"]
    monkeypatch.setenv(THRESHOLDS_ENV, str(path))
    run_ranks(["nvidia", "amd", "amd"], _check_measured_file, tmp_path,
              init_kwargs={"policy": "auto-tuned"})


def _clean_env():
    import os

    return {k: v for k, v in os.environ.items() if not k.startswith("GPUBRIDGE_")} | {
        "OMP_NUM_THREADS": "1"}


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
