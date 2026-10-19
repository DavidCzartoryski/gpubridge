"""scripts/verbs_perf.py's decision rules (docs/VERBS_PERF.md, section 1) and its
crossing counter. CPU only; no fabric."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verbs_perf


def row(metric, before, after):
    return verbs_perf._row("host", "economy", 4096, metric, before, after)


def test_a_win_needs_more_than_the_spread_and_most_pairs():
    assert row("p50_us", [100, 101, 102, 100, 101], [90, 91, 90, 92, 90])["verdict"] == "win"
    # Lower, but inside the spread: no change.
    assert row("p50_us", [100, 130, 90, 120, 100], [95, 125, 88, 110, 96])["verdict"] \
        .startswith("no change")
    # Beyond the spread in the median, but better in only 3 of 5 pairs: no win.
    assert row("p50_us", [100, 101, 102, 100, 101], [90, 91, 90, 110, 110])["verdict"] \
        != "win"


def test_regressions_and_higher_is_better():
    assert row("cpu_per_op_us", [100, 101, 100, 100, 101], [120, 121, 119, 120, 122])[
        "verdict"] == "REGRESSION"
    assert row("throughput_mb_s", [100, 101, 100, 100, 101], [90, 89, 90, 91, 90])[
        "verdict"] == "REGRESSION"
    assert row("throughput_mb_s", [100, 101, 100, 100, 101], [120, 121, 119, 120, 122])[
        "verdict"] == "win"


def test_counted_metrics_may_not_increase_at_all():
    assert row("crossings_total_per_call", [3] * 5, [3.02] * 5)["verdict"] == "regression"
    assert row("crossings_total_per_call", [4] * 5, [3] * 5)["verdict"] == "better"
    assert row("mismatches", [0] * 5, [1, 0, 0, 0, 0])["verdict"] == "ok"
    assert row("mismatches", [0] * 5, [1, 1, 1, 0, 0])["verdict"].startswith("FAIL")


def test_the_counter_counts_and_times_every_declared_function_then_restores_them():
    class Lib:
        pass

    lib = Lib()
    lib.gb_verbs_send = lambda *a: 7
    lib.gb_verbs_wait = lambda *a: 0
    lib.other = lambda: 1
    original = lib.gb_verbs_send
    with verbs_perf.Counter(lib) as counter:
        assert lib.gb_verbs_send(1, 2) == 7
        lib.gb_verbs_send()
        lib.gb_verbs_wait()
        lib.other()
    assert counter.counts == {"gb_verbs_send": 2, "gb_verbs_wait": 1}
    assert set(counter.seconds) == {"gb_verbs_send", "gb_verbs_wait"}
    assert lib.gb_verbs_send is original
