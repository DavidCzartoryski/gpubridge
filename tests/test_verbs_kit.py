"""The verbs kit's no-silent-TCP evidence (scripts/gpu/verbs_kit.py).

Port and netdev counters are shared by every process on a node, and a RoCE netdev
carries traffic that isn't TCP, so they can never decide whether verbs carried a job's
payload. The evidence must be positive and attributable: the bridge's own telemetry
(a real provider, its bytes and completions), the process's own TCP sockets idle,
and where the kernel lists them, the process's RC queue pairs. Anything not shown is
``unknown``, never a pass. These run on CPU; the end-to-end check runs the kit on the
MOCK test fabric, which must fail.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "gpu"))
import verbs_kit  # noqa: E402

PAYLOAD = 64 << 20
REAL = {"provider": "libibverbs (/usr/lib64/libibverbs.so.1)", "mock": False,
        "bytes_sent": 0, "bytes_received": 0, "completions": 0, "completion_errors": 0}


def moved(sent: int, received: int, completions: int = 100, errors: int = 0) -> dict:
    return {**REAL, "bytes_sent": sent, "bytes_received": received,
            "completions": completions, "completion_errors": errors}


RC = [{"ifname": "mlx5_0", "lqpn": 77, "type": "RC", "state": "RTS", "pid": 1}]
#: The RDMA port's data counters (4-byte words) and an idle netdev: what the old check
#: took as proof on their own.
PORT = {"mlx5_0/1/counters/port_xmit_data": PAYLOAD // 4,
        "mlx5_0/1/counters/port_rcv_data": PAYLOAD // 4}
NETDEV = {"ib0": {"rx_bytes": 0, "tx_bytes": 0}}


def evidence(tele_after, tcp=0, qps=RC, tele_before=REAL, **scope):
    return verbs_kit.tcp_evidence(PAYLOAD, tele_before, tele_after, tcp, qps, NETDEV, PORT,
                                  "mlx5_0", **scope)


def test_every_positive_part_shown_passes():
    e = evidence(moved(PAYLOAD, PAYLOAD))
    assert (e["verdict"], e["no_silent_tcp"], e["failures"], e["unknown"]) == (
        "pass", True, [], [])
    assert e["bridge_bytes"] == 2 * PAYLOAD and e["process_tcp_bytes"] == 0


def test_port_and_netdev_counters_alone_are_never_proof():
    # The old check passed here: payload on the port, no interface bytes.
    e = evidence(None)
    assert e["verdict"] == "unknown" and not e["no_silent_tcp"]
    assert "never decisive" in e["context_only"]["why"]
    assert e["context_only"]["rdma_port_data_bytes"] == 2 * PAYLOAD


def test_unrelated_port_traffic_does_not_stand_in_for_the_bridges_bytes():
    # Another job filled the shared port; the bridge itself moved a fraction.
    e = evidence(moved(PAYLOAD // 8, 0))
    assert e["verdict"] == "fail"
    assert any("less than the payload" in f for f in e["failures"])


def test_mock_provider_fails():
    e = evidence({**moved(PAYLOAD, PAYLOAD), "mock": True,
                  "provider": "MOCK: cross-process test fabric over Unix sockets (not RDMA)"})
    assert e["verdict"] == "fail" and any("MOCK" in f for f in e["failures"])


def test_unread_tcp_sockets_are_unknown_not_a_pass():
    e = evidence(moved(PAYLOAD, PAYLOAD), tcp=None)
    assert e["verdict"] == "unknown" and any("no ss" in u for u in e["unknown"])


def test_tcp_traffic_of_this_process_fails():
    e = evidence(moved(PAYLOAD, PAYLOAD), tcp=PAYLOAD // 2)
    assert e["verdict"] == "fail" and any("TCP sockets" in f for f in e["failures"])
    # The store's and Gloo's control traffic fits under the slack.
    assert evidence(moved(PAYLOAD, PAYLOAD), tcp=64 << 10)["verdict"] == "pass"


def test_no_completions_errors_or_rc_queue_pairs_fail():
    assert evidence(moved(PAYLOAD, PAYLOAD, completions=0))["verdict"] == "fail"
    assert evidence(moved(PAYLOAD, PAYLOAD, errors=3))["verdict"] == "fail"
    assert evidence(moved(PAYLOAD, PAYLOAD), qps=[])["verdict"] == "fail"
    # Without the rdma tool the queue pairs are recorded as unavailable, not required.
    e = evidence(moved(PAYLOAD, PAYLOAD), qps=None)
    assert e["verdict"] == "pass" and e["qps"].startswith("unavailable")


SS = """\
ESTAB 0 0 10.0.0.1:41000 10.0.0.2:29500 users:(("python",pid=123,fd=7))
\t cubic wscale:7,7 rto:204 bytes_sent:900 bytes_acked:800 bytes_received:500 segs_out:9
ESTAB 0 0 127.0.0.1:41001 127.0.0.1:29500 users:(("python",pid=123,fd=8))
\t cubic bytes_acked:99999 bytes_received:99999
ESTAB 0 0 10.0.0.1:41002 10.0.0.3:22 users:(("sshd",pid=456,fd=3))
\t cubic bytes_acked:7777 bytes_received:7777
ESTAB 0 0 [::ffff:10.0.0.1]:41003 [::ffff:10.0.0.4]:29400 users:(("python",pid=123,fd=9))
\t cubic bytes_received:40
"""


def test_ss_parsing_keeps_this_processs_sockets_to_other_hosts():
    sockets = verbs_kit.parse_ss(SS, 123)
    assert sockets == {"10.0.0.1:41000 10.0.0.2:29500": {"sent": 800, "received": 500},
                       "[::ffff:10.0.0.1]:41003 [::ffff:10.0.0.4]:29400":
                           {"sent": 0, "received": 40}}


@pytest.mark.parametrize(("local", "peer"), [
    ("10.0.0.1:41000", "10.0.0.1:29500"),
    ("[::ffff:10.0.0.1]:41000", "10.0.0.1:29500"),
    ("[2001:db8::1]:41000", "[2001:db8::1]:29500"),
    ("[2001:db8::1]:41000", "[2001:0db8:0:0:0:0:0:1]:29500"),
])
def test_same_host_island_tcp_only_excluded_for_gloo_islands(local, peer):
    before = (f'ESTAB 0 0 {local} {peer} users:(("python",pid=123,fd=7))\n'
              '\t cubic bytes_acked:100 bytes_received:100\n' + SS)
    after = before.replace("bytes_acked:100 bytes_received:100",
                           "bytes_acked:1600928576 bytes_received:1600928576")
    tcp = verbs_kit.tcp_moved(verbs_kit.parse_ss(before, 123, exclude_local_tcp=True),
                             verbs_kit.parse_ss(after, 123, exclude_local_tcp=True))
    assert tcp == 0
    # A remote bridge remains decisive despite local Gloo island traffic.
    assert evidence(moved(PAYLOAD, PAYLOAD), tcp=tcp, exclude_local_tcp=True,
                    peers_co_resident=False)["verdict"] == "pass"
    # GPU/NCCL mode retains the base kit's rejection of non-loopback local TCP.
    gpu_tcp = verbs_kit.tcp_moved(verbs_kit.parse_ss(before, 123), verbs_kit.parse_ss(after, 123))
    assert gpu_tcp > PAYLOAD
    assert evidence(moved(PAYLOAD, PAYLOAD), tcp=gpu_tcp)["verdict"] == "fail"


@pytest.mark.parametrize("co_resident", [True, None])
def test_local_bridge_partial_tcp_fallback_cannot_produce_a_decisive_pass(co_resident):
    # Reviewer's counterexample: genuine verbs all-reduces, local TCP broadcasts.
    before = ('ESTAB 0 0 10.0.0.1:41000 10.0.0.1:42000 users:(("python",pid=123,fd=7))\n'
              '\t bytes_acked:0\n')
    after = before.replace("bytes_acked:0", f"bytes_acked:{PAYLOAD}")
    tcp = verbs_kit.tcp_moved(verbs_kit.parse_ss(before, 123, exclude_local_tcp=True),
                             verbs_kit.parse_ss(after, 123, exclude_local_tcp=True))
    e = evidence(moved(PAYLOAD, PAYLOAD), tcp=tcp, exclude_local_tcp=True,
                 peers_co_resident=co_resident)
    assert tcp == 0
    assert e["verdict"] == "not-decisive" and e["no_silent_tcp"] is None
    assert e["reason"] == ("local bridge: island Gloo traffic and a bridge TCP fallback are "
                           "indistinguishable on one host")
    assert e["verbs_evidence"] == {"verdict": "pass", "failures": [], "unknown": []}


@pytest.mark.parametrize(("telemetry", "tcp", "qps"), [
    ({**moved(PAYLOAD, PAYLOAD), "mock": True}, 0, RC),
    ({**moved(PAYLOAD, PAYLOAD), "provider": "other"}, 0, RC),
    (moved(0, 0), 0, RC),
    (moved(PAYLOAD, PAYLOAD, completions=0), 0, RC),
    (moved(PAYLOAD, PAYLOAD, errors=1), 0, RC),
    (moved(PAYLOAD, PAYLOAD), 0, []),
    (moved(PAYLOAD, PAYLOAD), PAYLOAD, RC),
])
def test_local_bridge_limitation_never_overrides_a_failure(telemetry, tcp, qps):
    e = evidence(telemetry, tcp=tcp, qps=qps, exclude_local_tcp=True, peers_co_resident=True)
    assert e["verdict"] == "fail" and e["no_silent_tcp"] is False


@pytest.mark.parametrize(("peer", "expected"), [
    ({"hostname": "node-a", "boot_id": "boot-a"}, True),
    ({"hostname": "node-b", "boot_id": "boot-b"}, False),
    ({"hostname": "node-b", "boot_id": "boot-a"}, True),
    ({"hostname": "node-a", "boot_id": "boot-b"}, True),
    ({"hostname": "node-b", "boot_id": ""}, None),
    ({}, None),
])
def test_bridge_peer_identity_is_conservative(peer, expected):
    identities = [{"hostname": "node-a", "boot_id": "boot-a", "leader": True},
                  {**peer, "leader": True},
                  {"hostname": "node-a", "boot_id": "boot-a", "leader": False}]
    assert verbs_kit.bridge_peers_co_resident(identities, 0) is expected


def test_every_bridge_peer_must_be_proven_remote():
    local = {"hostname": "node-a", "boot_id": "boot-a", "leader": True}
    remote = {"hostname": "node-b", "boot_id": "boot-b", "leader": True}
    assert verbs_kit.bridge_peers_co_resident([local], 0) is None
    assert verbs_kit.bridge_peers_co_resident([local, remote, {"leader": True}], 0) is None
    assert verbs_kit.bridge_peers_co_resident([local, remote, local], 0) is True


def test_gpu_mode_keeps_same_host_tcp_in_the_strict_gate(tmp_path, monkeypatch):
    """Exercise GPU mode's measurement setup on CPU, without GPU collectives."""
    import torch
    import torch.distributed as dist

    import gpubridge

    telemetry = iter([REAL, REAL, moved(PAYLOAD, PAYLOAD), moved(PAYLOAD, PAYLOAD)])
    topology = SimpleNamespace(device=torch.device("cuda"),
                               config=SimpleNamespace(island_backend="nccl"),
                               bridge=SimpleNamespace(telemetry=lambda: next(telemetry)))
    monkeypatch.setattr(gpubridge, "init", lambda **kwargs: topology)
    monkeypatch.setattr(gpubridge, "destroy", lambda: None)
    monkeypatch.setattr(dist, "new_group", lambda **kwargs: None)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, **kwargs: tensor.fill_(2))

    def gather(identities, local, **kwargs):
        identities[:] = [local, {**local, "leader": False}, local, {**local, "leader": False}]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(verbs_kit, "sizes", lambda max_bytes: [])
    snapshots = iter([0, PAYLOAD])

    def read_tcp(*, exclude_local_tcp):
        text = ('ESTAB 0 0 10.0.0.1:41000 10.0.0.1:42000 users:(("python",pid=123,fd=7))\n'
                f'\t bytes_acked:{next(snapshots)}\n')
        return verbs_kit.parse_ss(text, 123, exclude_local_tcp=exclude_local_tcp)

    monkeypatch.setattr(verbs_kit, "read_process_tcp", read_tcp)
    monkeypatch.setattr(verbs_kit, "read_qps", lambda: RC)
    monkeypatch.delenv("GPUBRIDGE_CPU_ONLY", raising=False)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(sys, "argv", ["verbs_kit.py", "collectives", "--memory", "gpu",
                                      "--require-no-tcp", "--out", str(tmp_path)])
    assert verbs_kit.main() == 1
    record = json.loads((tmp_path / "rank0.json").read_text())
    e = record["window"]["tcp_evidence"]
    assert record["status"] == "failed" and "TCP sockets" in record["error"]
    assert e["verdict"] == "fail" and e["no_silent_tcp"] is False
    assert e["local_tcp_excluded"] is False and e["bridge_peers_co_resident"] is True
    assert e["process_tcp_bytes"] == PAYLOAD and e["verbs_evidence"]["verdict"] == "pass"


def test_ss_uses_all_local_endpoints_but_keeps_unknown_peers():
    # A second interface is known local even when only another process uses it.
    text = SS + '''\
ESTAB 0 0 10.0.0.5:41000 10.0.0.2:29500 users:(("python",pid=456,fd=7))
\t cubic bytes_acked:900
ESTAB 0 0 10.0.0.1:41004 [::ffff:10.0.0.5]:29500 users:(("python",pid=123,fd=9))
\t cubic bytes_acked:3201856952
ESTAB 0 0 10.0.0.1:41005 *:29500 users:(("python",pid=123,fd=10))
\t cubic bytes_acked:700
'''
    parsed = verbs_kit.parse_ss(text, 123, exclude_local_tcp=True)
    assert "10.0.0.1:41004 [::ffff:10.0.0.5]:29500" not in parsed
    assert parsed["10.0.0.1:41005 *:29500"]["sent"] == 700
    # Actual remote payload still fails, even when a local socket has large traffic.
    assert evidence(moved(PAYLOAD, PAYLOAD), tcp=verbs_kit.tcp_moved({}, parsed))[
        "verdict"] == "pass"
    remote = text.replace("bytes_acked:800", f"bytes_acked:{PAYLOAD}")
    assert evidence(moved(PAYLOAD, PAYLOAD), tcp=verbs_kit.tcp_moved(
        {}, verbs_kit.parse_ss(remote, 123, exclude_local_tcp=True)))["verdict"] == "fail"


def test_tcp_moved_counts_new_sockets_in_full_and_unknown_as_unknown():
    before = {"a": {"sent": 10, "received": 10}}
    after = {"a": {"sent": 15, "received": 12}, "b": {"sent": 3, "received": 0}}
    assert verbs_kit.tcp_moved(before, after) == 10
    assert verbs_kit.tcp_moved(None, after) is None
    assert verbs_kit.tcp_moved(before, None) is None


def test_missing_tools_read_as_unknown():
    def missing(*args, **kwargs):
        raise FileNotFoundError(args[0][0])

    assert verbs_kit.read_process_tcp(run=missing) is None
    assert verbs_kit.read_qps(run=missing) is None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kit(mode: str, out: Path, fabric: str, *extra: str,
         split: str | None = "half", cuda_like: bool = False,
         tcp_case: str | None = None) -> list[subprocess.CompletedProcess]:
    """Ranks as plain processes. Under torchrun the agent SIGTERMs the other ranks as
    soon as one exits nonzero, so when every rank fails only the first one's record is
    sure to be written. ``split=None``: one NVIDIA island, so no bridge leader."""
    port = _free_port()
    procs = []
    world = 4 if cuda_like else 2
    for rank in range(world):
        env = {**os.environ, "GPUBRIDGE_VERBS_TEST_FABRIC": fabric,
               "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
               "GPUBRIDGE_VENDOR": "nvidia",
               "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank),
               "LOCAL_RANK": str(rank), "WORLD_SIZE": str(world), "LOCAL_WORLD_SIZE": str(world)}
        env.pop("GPUBRIDGE_SPLIT_TEST", None)
        if split is not None:
            env["GPUBRIDGE_SPLIT_TEST"] = split
        command = [sys.executable]
        driver = ""
        if cuda_like:
            # No vendor simulation: without the kit's CPU-only selection this
            # configuration would select NCCL and cuda:LOCAL_RANK, even on CPU CI.
            env.pop("GPUBRIDGE_VENDOR", None)
            env.pop("GPUBRIDGE_CPU_ONLY", None)
            env["GPUBRIDGE_VERBS_GPU_PATH"] = "staged"
            driver += """
import os
import runpy
import sys
import torch
import torch.distributed as dist
import gpubridge
from gpubridge.config import resolve_config
from gpubridge.verbs.transport import VerbsTransport

torch.version.cuda = "12.6"
torch.version.hip = None
torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 4
original_init = gpubridge.init
def checked_init(**kwargs):
    assert os.environ["GPUBRIDGE_CPU_ONLY"] == "1"
    assert os.environ["GPUBRIDGE_VERBS_GPU_PATH"] == "staged"
    assert kwargs["bridge"] == "verbs" and kwargs["policy"] == "auto"
    config = resolve_config()
    assert config.device == torch.device("cpu") and config.island_backend == "gloo"
    assert not dist.is_initialized()
    topology = original_init(**kwargs)
    assert dist.get_backend() == "gloo"
    assert dist.get_backend(topology.island_group) == "gloo"
    assert topology.config == config
    assert topology.transport is VerbsTransport
    return topology
gpubridge.init = checked_init
"""
        if tcp_case:
            # Run real mock-fabric collectives; replace only diagnostic observations
            # with real-provider-shaped counters and synthetic socket/host evidence.
            # This exercises acceptance, not hardware or actual TCP fallback.
            driver += f"\ntcp_case = {tcp_case!r}\n" + '''
import os
import sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import verbs_kit
from gpubridge.verbs.transport import VerbsTransport

original_telemetry = VerbsTransport.telemetry
def observed_telemetry(self):
    return {**original_telemetry(self), "mock": False, "provider": "libibverbs (synthetic)"}
VerbsTransport.telemetry = observed_telemetry
verbs_kit.read_qps = lambda: [{"type": "RC"}]
snapshots = 0
def observed_tcp(*, exclude_local_tcp):
    global snapshots
    peer = "10.0.0.2" if tcp_case == "remote-tcp" else "10.0.0.1"
    text = (f'ESTAB 0 0 10.0.0.1:41000 {peer}:42000 '
            f'users:(("python",pid={os.getpid()},fd=7))\\n'
            f'\\t bytes_acked:{(64 << 20) if snapshots else 0}\\n')
    snapshots += 1
    return verbs_kit.parse_ss(text, os.getpid(), exclude_local_tcp=exclude_local_tcp)
verbs_kit.read_process_tcp = observed_tcp
if tcp_case.startswith("remote"):
    verbs_kit.host_identity = lambda: {
        "hostname": "node-" + os.environ["RANK"], "boot_id": "boot-" + os.environ["RANK"]}
'''
        if driver:
            driver += '''
import os
import sys
sys.path.insert(0, os.path.dirname(sys.argv[1]))
import verbs_kit
sys.argv = sys.argv[1:]
sys.exit(verbs_kit.main())
'''
            command += ["-c", driver]
        procs.append(subprocess.Popen(
            [*command, str(ROOT / "scripts" / "gpu" / "verbs_kit.py"), mode,
             "--out", str(out), "--max-bytes", "4096", "--iters", "5", "--timeout", "60", *extra],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    done = []
    try:
        for proc in procs:
            stdout, stderr = proc.communicate(timeout=240)
            done.append(subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr))
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
    return done


def test_host_mode_uses_cpu_islands_with_a_cuda_like_build_on_the_mock_fabric(tmp_path):
    from test_verbs import UNAVAILABLE

    if UNAVAILABLE is not None:
        pytest.skip(f"verbs test library unavailable: {UNAVAILABLE}")
    fabric = tempfile.mkdtemp(prefix="gbvk-", dir="/tmp")
    try:
        done = _kit("collectives", tmp_path, fabric, "--memory", "host", cuda_like=True)
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
    assert all(rank.returncode == 0 for rank in done), [rank.stderr[-2000:] for rank in done]
    for r in range(4):
        record = json.loads((tmp_path / f"rank{r}.json").read_text())
        assert record["status"] == "ok", record.get("error")
        assert record["memory"] == "host" and record["cpu_only"] is True
        assert record["island_backend"] == "gloo" and record["leaders"] == 2
        assert record["mismatches"] == 0
        assert {call["op"] for call in record["calls"]} == {"all_reduce", "broadcast"}
        assert all(call["exact"] for call in record["calls"])
        if record["leader"]:
            assert record["telemetry_after"]["mock"] is True
            assert record["telemetry_after"]["gpu_path_setting"] == "staged"
            assert {call["path"] for call in record["calls"]} == {"host"}
            assert record["window"]["tcp_evidence"]["bridge_bytes"] > 0


@pytest.mark.parametrize(("tcp_case", "strict", "verdict", "status"), [
    ("local", False, "not-decisive", "ok"),
    ("local", True, "not-decisive", "failed"),
    ("remote-island", True, "pass", "ok"),
    ("remote-tcp", True, "fail", "failed"),
])
def test_host_tcp_acceptance_on_mock_collectives_with_synthetic_evidence(
        tmp_path, tcp_case, strict, verdict, status):
    from test_verbs import UNAVAILABLE

    if UNAVAILABLE is not None:
        pytest.skip(f"verbs test library unavailable: {UNAVAILABLE}")
    fabric = tempfile.mkdtemp(prefix="gbvk-", dir="/tmp")
    try:
        extra = ["--require-no-tcp"] if strict else []
        done = _kit("collectives", tmp_path, fabric, "--memory", "host", *extra,
                    cuda_like=True, tcp_case=tcp_case)
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
    records = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(4)]
    for rank, record in zip(done, records):
        assert record["mismatches"] == 0 and record["leaders"] == 2
        if not record["leader"]:
            assert rank.returncode == 0 and record["status"] == "ok"
            continue
        assert record["status"] == status, record.get("error")
        assert (rank.returncode == 0) == (status == "ok")
        e = record["window"]["tcp_evidence"]
        assert e["verdict"] == verdict
        assert e["verbs_evidence"]["verdict"] == "pass"
        assert e["local_tcp_excluded"] is True
        assert e["bridge_peers_co_resident"] is (tcp_case == "local")
        if verdict == "not-decisive":
            assert e["no_silent_tcp"] is None
            assert e["reason"] in rank.stderr
            if strict:
                assert "not-decisive" in record["error"] and e["reason"] in record["error"]
        else:
            assert e["no_silent_tcp"] is (verdict == "pass")
        assert e["process_tcp_bytes"] == (PAYLOAD if tcp_case == "remote-tcp" else 0)


def test_host_selection_precedes_the_first_torch_or_gpubridge_import(tmp_path):
    env = {**os.environ}
    env.pop("GPUBRIDGE_CPU_ONLY", None)
    done = subprocess.run([sys.executable, "-c", '''
import builtins
import os
import runpy
import sys

original_import = builtins.__import__
seen = []
def checked_import(name, *args, **kwargs):
    if name.split(".")[0] in ("torch", "gpubridge"):
        assert os.environ.get("GPUBRIDGE_CPU_ONLY") == "1"
        seen.append(name)
        raise RuntimeError("startup ordering checked")
    return original_import(name, *args, **kwargs)
builtins.__import__ = checked_import
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
except SystemExit as exc:
    assert exc.code == 1
assert seen, "the kit never reached its first runtime import"
''', str(ROOT / "scripts/gpu/verbs_kit.py"), "collectives", "--memory", "host",
        "--out", str(tmp_path)], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / f"rank{env.get('RANK', '0')}.json").read_text())
    assert record["error"] == "RuntimeError: startup ordering checked"


@pytest.mark.parametrize("rank", range(4))
def test_host_config_at_init_with_inherited_staged(tmp_path, monkeypatch, rank):
    import torch
    import torch.distributed as dist

    import gpubridge
    from gpubridge.config import resolve_config

    monkeypatch.delenv("GPUBRIDGE_CPU_ONLY", raising=False)
    monkeypatch.delenv("GPUBRIDGE_VENDOR", raising=False)
    monkeypatch.setenv("GPUBRIDGE_VERBS_GPU_PATH", "staged")
    monkeypatch.setenv("GPUBRIDGE_SPLIT_TEST", "half")
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    assert resolve_config().island_backend == "nccl"
    observed = []

    def check_init(**kwargs):
        config = resolve_config()
        assert config.device == torch.device("cpu")
        assert config.island_backend == "gloo" and config.simulated
        assert kwargs["bridge"] == "verbs" and kwargs["policy"] == "auto"
        assert not dist.is_initialized()
        assert os.environ["GPUBRIDGE_VERBS_GPU_PATH"] == "staged"
        observed.append(config)
        raise RuntimeError("host configuration checked")

    monkeypatch.setattr(gpubridge, "init", check_init)
    monkeypatch.setattr(sys, "argv", ["verbs_kit.py", "collectives", "--memory", "host",
                                      "--out", str(tmp_path)])
    assert verbs_kit.main() == 1
    assert len(observed) == 1
    record = json.loads((tmp_path / f"rank{rank}.json").read_text())
    assert record["error"] == "RuntimeError: host configuration checked"


@pytest.mark.parametrize(("memory", "setting"), [
    ("gpu", "1"), ("gpu", " true "), ("gpu", "yes"), ("gpu", "on"),
    ("host", "0"), ("host", "false"), ("host", "no"), ("host", "off"),
])
def test_memory_conflicts_are_refused_before_init(tmp_path, monkeypatch, memory, setting):
    import gpubridge

    def unexpected_init(**kwargs):
        pytest.fail("a contradictory setting must fail before initializing collectives")

    monkeypatch.setattr(gpubridge, "init", unexpected_init)
    monkeypatch.setenv("GPUBRIDGE_CPU_ONLY", setting)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(sys, "argv", ["verbs_kit.py", "collectives", "--memory", memory,
                                      "--out", str(tmp_path)])
    assert verbs_kit.main() == 1
    record = json.loads((tmp_path / "rank0.json").read_text())
    assert record["status"] == "error"
    assert f"--memory {memory} conflicts with GPUBRIDGE_CPU_ONLY={setting!r}" in record["error"]
    assert os.environ["GPUBRIDGE_CPU_ONLY"] == setting


@pytest.mark.parametrize("setting", [None, "", "0", "false"])
def test_gpu_mode_preserves_the_environment_and_init_arguments(tmp_path, monkeypatch, setting):
    import gpubridge

    monkeypatch.delenv("GPUBRIDGE_CPU_ONLY", raising=False)
    if setting is not None:
        monkeypatch.setenv("GPUBRIDGE_CPU_ONLY", setting)

    def check_init(**kwargs):
        from datetime import timedelta

        assert os.environ.get("GPUBRIDGE_CPU_ONLY") == setting
        assert kwargs == {"timeout": timedelta(seconds=60), "bridge": "verbs", "policy": "auto"}
        raise RuntimeError("stop before GPU initialization")

    monkeypatch.setattr(gpubridge, "init", check_init)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(sys, "argv", ["verbs_kit.py", "collectives", "--memory", "gpu",
                                      "--timeout", "60", "--out", str(tmp_path)])
    assert verbs_kit.main() == 1
    record = json.loads((tmp_path / "rank0.json").read_text())
    assert record["error"] == "RuntimeError: stop before GPU initialization"


@pytest.mark.parametrize("mode", ["endpoint", "collectives"])
def test_the_kit_on_the_mock_fabric_never_passes_the_no_tcp_requirement(tmp_path, mode):
    from test_verbs import UNAVAILABLE

    if UNAVAILABLE is not None:
        pytest.skip(f"verbs test library unavailable: {UNAVAILABLE}")
    fabric = tempfile.mkdtemp(prefix="gbvk-", dir="/tmp")
    try:
        done = _kit(mode, tmp_path, fabric, "--require-no-tcp")
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
    assert all(rank.returncode != 0 for rank in done), [rank.stderr[-2000:] for rank in done]
    records = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]
    for record in records:
        evidence = record["window"]["tcp_evidence"]
        assert record["status"] == "failed", record.get("error")
        assert evidence["verdict"] == "fail" and not evidence["no_silent_tcp"]
        assert any("MOCK" in f for f in evidence["failures"])
        if mode == "collectives":
            assert record["mismatches"] == 0 and record["leader"] is True


def test_a_job_without_a_bridge_leader_fails_the_no_tcp_requirement(tmp_path):
    from test_verbs import UNAVAILABLE

    if UNAVAILABLE is not None:
        pytest.skip(f"verbs test library unavailable: {UNAVAILABLE}")
    fabric = tempfile.mkdtemp(prefix="gbvk-", dir="/tmp")
    try:
        done = _kit("collectives", tmp_path, fabric, "--require-no-tcp", split=None)
    finally:
        shutil.rmtree(fabric, ignore_errors=True)
    assert all(rank.returncode != 0 for rank in done), [rank.stderr[-2000:] for rank in done]
    for r in (0, 1):
        record = json.loads((tmp_path / f"rank{r}.json").read_text())
        assert record["leader"] is False and record["leaders"] == 0
        assert record["status"] == "failed" and "no rank is a bridge leader" in record["error"]
        assert record["window"]["tcp_evidence"]["no_silent_tcp"] is False
