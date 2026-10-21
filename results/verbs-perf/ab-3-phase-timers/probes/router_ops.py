"""Per-op cost of Router.send/recv/wait against an endpoint that does nothing (CPU only).

    PYTHONPATH=TREE/src python router_ops.py base|off|timed
"""

import json
import sys
import timeit

from gpubridge.verbs.gpu import Router


class Instant:
    n = 0
    failed = None

    def rndv_bytes(self):
        return 262144

    def send(self, *args):
        self.n += 1
        return self.n

    recv = send

    def wait(self, ids, deadline):
        pass


router = Router(Instant())
router.begin([])


def one():
    a = router.send(1, 0, 64, 1)
    b = router.recv(1, 0, 64, 2)
    router.wait([a, b], 0)


mode = sys.argv[1]
if mode == "timed":
    router.timed = True
t = min(timeit.repeat(one, number=100000, repeat=9)) / 100000
print(json.dumps({"mode": mode, "ns_per_send_recv_wait": round(t * 1e9, 1)}))
