"""Async submission backpressure above the native transport's bounded work queues."""

import threading

import pytest
import torch

from gpubridge import work


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_an_invalid_async_queue_bound_is_refused(value, monkeypatch):
    monkeypatch.setenv(work.QUEUE_ENV, value)
    with pytest.raises(ValueError, match=work.QUEUE_ENV):
        work._Worker(torch.device("cpu"))


def test_overflow_is_refused_without_losing_accepted_work_or_fifo_order(monkeypatch):
    monkeypatch.setenv(work.QUEUE_ENV, "2")
    worker = work._Worker(torch.device("cpu"))
    entered, release = threading.Event(), threading.Event()
    ran = []

    def first():
        entered.set()
        assert release.wait(5)
        ran.append(1)

    try:
        one = worker.submit("one", first, [], None)
        assert entered.wait(5)
        two = worker.submit("two", lambda: ran.append(2), [], None)
        three = worker.submit("three", lambda: ran.append(3), [], None)
        with pytest.raises(RuntimeError, match="async queue is full"):
            worker.submit("refused", lambda: pytest.fail("a refused call ran"), [], None)
        release.set()
        for accepted in [one, two, three]:
            assert accepted.wait(5)
        # Backpressure is recoverable; it did not poison or reorder accepted work.
        four = worker.submit("four", lambda: ran.append(4), [], None)
        assert four.wait(5)
        assert ran == [1, 2, 3, 4]
    finally:
        release.set()
        worker.close()


def test_abandon_cancels_queued_work_when_the_bounded_queue_is_full(monkeypatch):
    monkeypatch.setenv(work.QUEUE_ENV, "1")
    worker = work._Worker(torch.device("cpu"))
    entered, release = threading.Event(), threading.Event()

    def first():
        entered.set()
        assert release.wait(5)

    try:
        running = worker.submit("running", first, [], None)
        assert entered.wait(5)
        pending = worker.submit("pending", lambda: pytest.fail("cancelled work ran"), [], None)
        worker.abandon(0.01)
        with pytest.raises(RuntimeError, match="process groups were destroyed"):
            pending.wait(1)
        release.set()
        assert running.wait(5)
        worker.thread.join(5)
        assert not worker.thread.is_alive()
    finally:
        release.set()
        if worker.thread.is_alive():
            worker.thread.join(5)
