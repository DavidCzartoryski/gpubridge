"""Bridge compression kernels: throughput against the device's memory bandwidth.

Runs on one device, without torchrun. For each mode (bf16, fp8), input dtype
and size (1 MB to 1 GB of input by default, x4 per step) it times, with CUDA
events on GPUs:

- ``quantize``: reads the input and the error feedback residual, writes the
  payload and the new residual;
- ``dequantize``: reads two members' payloads (a two-island bridge step) and
  writes the result in the input's dtype.

GB/s counts the bytes each kernel has to move, so it compares directly with
the device's memory bandwidth: measured here as a large device-to-device copy
(reads plus writes), and the peak from the memory clock and bus width where
PyTorch reports them. The kernels are memory-bound; a kernel far below the
copy rate is worth a look before blaming the bridge.

It also checks, for every mode and dtype, that the device's kernels give the
same bits as the plain-PyTorch reference on CPU, and records a SHA-256 of the
payload and result for one fixed input. Every vendor's GPU must give the same
digests (docs/COMPRESSION.md), so the NVIDIA and AMD runs of a mixed job can
be compared directly.

On a machine without a GPU (simulation mode, the kit's CPU tests) it times the
PyTorch reference on CPU instead. Writes OUT/compression.json,
OUT/compression.csv and OUT/summary.md:

    python scripts/gpu/bench_compression.py --out results/bench_compression

gpubridge compiles the Triton kernels without multiply-add contraction
(``_compression_kernels.LAUNCH_OPTIONS``). ``--fp-fusion`` launches them with
it, to time what that costs: run it once each way, into two OUT directories.
The bits are the same either way, since every product the kernels add is exact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
import time
from pathlib import Path

import torch
from gpukit import KIT_VERSION, environment, sync, write_json

from gpubridge import compression as C

UNITS = {"K": 2**10, "M": 2**20, "G": 2**30}
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
FIELDS = ["mode", "kernel", "dtype", "bytes", "numel", "impl", "trials", "median_us",
          "GBps", "pct_of_copy", "pct_of_peak"]
#: Values in the fixed input whose digests every run records.
DIGEST_NUMEL = (1 << 16) + 37


def parse_size(text: str) -> int:
    text = text.strip().upper().removesuffix("B")
    if text and text[-1] in UNITS:
        return int(float(text[:-1]) * UNITS[text[-1]])
    return int(text)


def human(nbytes: int) -> str:
    for unit in ("G", "M", "K"):
        if nbytes >= UNITS[unit]:
            return f"{nbytes / UNITS[unit]:g}{unit}"
    return f"{nbytes}B"


def time_kernel(fn, device: torch.device, *, warmup: int, trials: int,
                max_seconds: float) -> list[float]:
    """Seconds per call: CUDA events on GPUs, the host timer on CPU."""
    for _ in range(warmup):
        fn()
    sync(device)
    times: list[float] = []
    while len(times) < trials and (sum(times) < max_seconds or len(times) < 3):
        if device.type == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            fn()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) / 1e3)
        else:
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
    return times


def peak_GBps(device: torch.device) -> float | None:
    """Theoretical memory bandwidth from the device's memory clock and bus width, if known."""
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    clock_khz = getattr(props, "memory_clock_rate", None)
    width_bits = getattr(props, "memory_bus_width", None)
    if not clock_khz or not width_bits:
        return None
    return 2 * clock_khz * 1e3 * width_bits / 8 / 1e9  # double data rate


def copy_GBps(device: torch.device, nbytes: int, args) -> float:
    """Device-to-device copy throughput, counting the read and the write."""
    src = torch.empty(nbytes, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    times = time_kernel(lambda: dst.copy_(src), device, warmup=args.warmup, trials=args.trials,
                        max_seconds=args.max_seconds)
    return 2 * nbytes / statistics.median(times) / 1e9


def awkward(n: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """A fixed input over many magnitudes, with the special values the codec handles."""
    g = torch.Generator().manual_seed(seed)
    span = 10 if dtype == torch.float16 else 40
    x = torch.randn(n, generator=g, dtype=torch.float64) * 2.0 ** torch.randint(
        -span, span, (n,), generator=g)
    for i, value in enumerate([0.0, float("nan"), float("inf"), -float("inf"), 1e-35]):
        x[(i * 7919) % n] = value
    return x.to(dtype)


def check_and_digest(mode: str, dtype: torch.dtype, device: torch.device) -> dict:
    """The device's kernels against the CPU reference, and digests of their output."""
    n = DIGEST_NUMEL
    x = awkward(n, dtype, 1234)
    residual = torch.zeros(n)
    want_payload = torch.empty(C.payload_nbytes(mode, n), dtype=torch.uint8)
    C.quantize_reference(mode, x, residual.clone(), want_payload)
    payload = torch.empty_like(want_payload, device=device)
    C.quantize(mode, x.to(device), residual.to(device), payload)
    other = torch.empty_like(payload)  # a second member's payload
    C.quantize(mode, awkward(n, dtype, 4321).to(device), None, other)
    rows = torch.stack([payload, other])
    rows_cpu = rows.cpu()
    want_out = torch.empty(n, dtype=dtype)
    C.dequantize_accumulate_reference(mode, rows_cpu, want_out)
    out = torch.empty(n, dtype=dtype, device=device)
    C.dequantize_accumulate(mode, rows, out)
    out_cpu = out.cpu()
    same_out = torch.equal(out_cpu.isnan(), want_out.isnan()) and torch.equal(
        out_cpu.nan_to_num(0.0), want_out.nan_to_num(0.0))
    return {
        "mode": mode, "dtype": str(dtype).removeprefix("torch."), "numel": n,
        "matches_reference": bool(torch.equal(payload.cpu(), want_payload) and same_out),
        "payload_sha256": hashlib.sha256(payload.cpu().numpy().tobytes()).hexdigest(),
        "result_sha256": hashlib.sha256(out_cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-bytes", type=parse_size, default=parse_size("1M"))
    parser.add_argument("--max-bytes", type=parse_size, default=parse_size("1G"))
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--modes", default=",".join(C.MODES))
    parser.add_argument("--dtypes", default="float32,bfloat16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="stop timing a size after this long (at least 3 trials)")
    parser.add_argument("--fp-fusion", action="store_true",
                        help="launch the Triton kernels with multiply-add contraction on "
                             "(gpubridge's default is off), to time the difference")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    probe = torch.empty(1, device=device)
    triton_kernels = C._triton_kernels(probe)
    impl = "triton" if triton_kernels is not None else "torch"
    fp_fusion = None  # only Triton's kernels have the option
    if triton_kernels is not None:
        if args.fp_fusion:
            triton_kernels.LAUNCH_OPTIONS["enable_fp_fusion"] = True
        fp_fusion = bool(triton_kernels.LAUNCH_OPTIONS.get("enable_fp_fusion", True))
        impl += f", fp fusion {'on' if fp_fusion else 'off'}"
    elif args.fp_fusion:
        print("--fp-fusion: these are the PyTorch kernels, which have no such option; "
              "ignored", file=sys.stderr)
    if device.type == "cpu":
        impl += " (CPU)"
    lo = min(args.min_bytes, args.max_bytes)
    sizes = []
    while lo <= args.max_bytes:
        sizes.append(lo)
        lo *= args.factor
    copy = copy_GBps(device, max(sizes[-1] if sizes else 1, 1 << 20), args)
    peak = peak_GBps(device)
    print(f"device {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}"
          f"), kernels {impl}; copy {copy:.1f} GB/s"
          + (f", peak {peak:.1f} GB/s" if peak else ""), flush=True)

    rows = []
    checks = []
    for mode in args.modes.split(","):
        for name in args.dtypes.split(","):
            dtype = DTYPES[name]
            if not C.applies(mode, dtype):
                continue
            checks.append(check_and_digest(mode, dtype, device))
            element = torch.tensor([], dtype=dtype).element_size()
            for nbytes in sizes:
                n = max(1, nbytes // element)
                x = torch.randn(n, device=device).to(dtype)
                residual = torch.zeros(n, device=device)
                p = C.payload_nbytes(mode, n)
                payloads = torch.empty((2, p), dtype=torch.uint8, device=device)
                C.quantize(mode, x, residual, payloads[0])
                payloads[1].copy_(payloads[0])
                out = torch.empty_like(x)
                kernels = {
                    # input + residual read + residual write + payload
                    "quantize": (lambda: C.quantize(mode, x, residual, payloads[0]),
                                 n * element + 8 * n + p),
                    # two payloads + output
                    "dequantize": (lambda: C.dequantize_accumulate(mode, payloads, out),
                                   2 * p + n * element),
                }
                for kernel, (fn, moved) in kernels.items():
                    times = time_kernel(fn, device, warmup=args.warmup, trials=args.trials,
                                        max_seconds=args.max_seconds)
                    median = statistics.median(times)
                    gbps = moved / median / 1e9
                    rows.append({
                        "mode": mode, "kernel": kernel, "dtype": name, "bytes": n * element,
                        "numel": n, "impl": impl, "trials": len(times),
                        "median_us": round(median * 1e6, 2), "GBps": round(gbps, 2),
                        "pct_of_copy": round(100 * gbps / copy, 1),
                        "pct_of_peak": round(100 * gbps / peak, 1) if peak else None,
                    })
                    print(f"{mode:>4} {kernel:>10} {name:>8} {human(n * element):>6}: "
                          f"{median * 1e6:10.1f} us  {gbps:8.2f} GB/s "
                          f"({100 * gbps / copy:.0f}% of copy)", flush=True)
                x = residual = payloads = out = None  # free this size before the next

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "compression.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.out / "compression.json", {
        "kit_version": KIT_VERSION, "script": "scripts/gpu/bench_compression.py",
        "argv": sys.argv, "environment": environment(), "device": str(device), "impl": impl,
        "copy_GBps": round(copy, 2), "peak_GBps": round(peak, 2) if peak else None,
        "fp_fusion": fp_fusion, "checks": checks, "rows": rows,
    })
    lines = [f"# {args.out.name}: bridge compression kernels\n",
             f"Device {device}, kernels {impl}. Memory bandwidth: copy {copy:.1f} GB/s"
             + (f", peak {peak:.1f} GB/s" if peak else "") + ".\n",
             "| mode | dtype | check | payload sha256 | result sha256 |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {c['mode']} | {c['dtype']} | {'same bits as CPU' if c['matches_reference'] else 'DIFFERS'}"
              f" | {c['payload_sha256'][:16]} | {c['result_sha256'][:16]} |" for c in checks]
    lines += ["", "| mode | kernel | dtype | size | median us | GB/s | % of copy |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['mode']} | {r['kernel']} | {r['dtype']} | {human(r['bytes'])} | "
              f"{r['median_us']} | {r['GBps']} | {r['pct_of_copy']} |" for r in rows]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    bad = [c for c in checks if not c["matches_reference"]]
    for c in bad:
        print(f"FAIL: {c['mode']} {c['dtype']}: the {impl} kernels differ from the CPU reference",
              file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
