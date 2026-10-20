"""Run the staged default path (overlap env unset) over a recording fake wire against the
tree given as argv[1]; print the full ordered log, wire and results as JSON."""
import ctypes, json, os, sys
tree = sys.argv[1]
sys.path[:0] = [f"{tree}/src", f"{tree}/tests"]
os.environ.pop("GPUBRIDGE_VERBS_STAGED_OVERLAP", None)
from datetime import timedelta
import torch
from torch.distributed import ReduceOp
from test_verbs_gpu import LazyRegistered
from gpubridge.verbs import Registration
from gpubridge.verbs import transport as vt
from gpubridge.verbs.gpu import HostBuffer, Router
print(vt.__file__, file=sys.stderr)

class Wire:
    def __init__(self, peer):
        self.peer, self.log, self.ops, self.next_op, self.failed, self._handle = peer, [], {}, 0, None, 1
    def rndv_bytes(self): return 1024
    def register_host(self, pointer, nbytes, owner): return Registration(100 + owner, pointer, nbytes, "host")
    def release(self, reg): pass
    def invalidate_host(self, owner): return 0
    def telemetry(self): return {"state": "ok"}
    def _post(self, kind, reg, pointer, nbytes, tag):
        self.next_op += 1
        self.ops[self.next_op] = (kind, reg.handle, pointer - reg.addr, nbytes, tag, pointer)
        self.log.append((kind, reg.handle % 100, pointer - reg.addr, nbytes, tag, ctypes.string_at(pointer, nbytes).hex()))
        return self.next_op
    def send_reg(self, m, reg, p, n, t): return self._post("send", reg, p, n, t)
    def recv_reg(self, m, reg, p, n, t): return self._post("recv", reg, p, n, t)
    def wait(self, ids, deadline):
        self.log.append(("wait_ops", [self.ops[i][4] for i in ids]))
        for i in ids:
            kind, _, off, n, tag, ptr = self.ops[i]
            if kind == "recv":
                src = self.peer.view(torch.uint8)[off:off + n]
                ctypes.memmove(ptr, src.data_ptr(), n)
    def abort(self, why="aborted"): self.failed = why

class Staged(vt.VerbsTransport):
    def _is_device(self, t): return True
    def _device_index(self, t): return 0
    def _support_for(self, i): return vt.Support(False, "standing in")
    def _staging_for(self, tensor):
        if 0 not in self._staging:
            self._staging[0] = LazyRegistered(HostBuffer(self.worker, lambda n: torch.zeros(n, dtype=torch.uint8)))
            self.worker.endpoint.log = self._staging[0].log
        return self._staging[0]

out = {}
for numel in (1, 15, 16, 17, 56, 200):
    for op in (ReduceOp.SUM, ReduceOp.MAX):
        peer = torch.arange(numel, dtype=torch.float32) * -3 + 7
        t = Staged(Router(Wire(peer)), [0, 1], 0, timedelta(seconds=5), 64)
        results = []
        for k, stream in enumerate(("caller", "other", "other")):
            tensor = torch.arange(numel, dtype=torch.float32) + 1 + 100 * k
            if k:
                t._staging[0].stream = stream
            t.all_reduce(tensor, op)
            if k == 2:  # a broadcast too
                b = torch.arange(numel, dtype=torch.float32)
                t.broadcast(b, 0)
            results.append(tensor)
        t._staging[0].run_streams()
        out[f"{numel}-{op}"] = {"log": [list(map(str, e)) for e in t._staging[0].log],
                               "results": [r.tolist() for r in results],
                               "copied_back": t._staging[0].copied_back.label,
                               "paths": t.paths}
print(json.dumps(out, sort_keys=True))
