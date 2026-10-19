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


def evidence(tele_after, tcp=0, qps=RC, tele_before=REAL):
    return verbs_kit.tcp_evidence(PAYLOAD, tele_before, tele_after, tcp, qps, NETDEV, PORT,
                                  "mlx5_0")


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


def _kit(mode: str, out: Path, fabric: str, *extra: str) -> list[subprocess.CompletedProcess]:
    """Both ranks as plain processes. Under torchrun the agent SIGTERMs the other rank as
    soon as one exits nonzero, so when every rank fails only the first one's record is
    sure to be written."""
    port = _free_port()
    procs = []
    for rank in (0, 1):
        env = {**os.environ, "GPUBRIDGE_VERBS_TEST_FABRIC": fabric,
               "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]),
               "GPUBRIDGE_VENDOR": "nvidia", "GPUBRIDGE_SPLIT_TEST": "half",
               "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "RANK": str(rank),
               "LOCAL_RANK": str(rank), "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"}
        procs.append(subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "gpu" / "verbs_kit.py"), mode,
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
