# Bridge compression

**Opt-in, changes results, not validated on GPUs.** On the first GPU run, a
V100, two of the kernel tests failed: the fp8 error-feedback residual
differed from the reference, from a codec overflow and a fused multiply-add,
both fixed since (#50). No GPU run has checked the fix yet. Otherwise
everything here has run on CPU only: the Triton kernels in Triton's
interpreter, and whole jobs in simulation mode.
[HARDWARE_VALIDATION.md](../HARDWARE_VALIDATION.md) items 28 and 29 list what
a GPU run must show.

```bash
export GPUBRIDGE_BRIDGE_COMPRESSION=fp8   # or bf16; unset (the default) is off
```

Every rank must set the same value. `init()` fails on every rank if they
differ or if one is invalid, as for the other settings.

## What it does

In a bridged all_reduce, island leaders send their island's sum across the
bridge: through host memory and Gloo (or UCX) between vendors, the slow hop.
Compression sends fewer bytes there, and only there. Island steps
(NCCL/RCCL over NVLink, Infinity Fabric or PCIe) stay in full precision.

- **What it applies to:** the bridge step of SUM and AVG reductions on
  floating-point tensors, in `all_reduce` and `reduce_scatter_tensor`, under
  the three bridged policies: `reduce-bridge-broadcast` (the default for two
  islands), `pipelined-reduce-bridge-broadcast` and `sharded-bridge`, and
  `auto-tuned` when it picks one of them.
- **What it leaves alone:** MAX and MIN, integer dtypes, broadcast and
  all_gather, the `native-only` and `flat-gloo` policies, and single-island
  jobs (no bridge). A mode also skips dtypes it wouldn't shrink: `bf16`
  compresses float32 and float64 only, while `fp8` also compresses float16
  and bfloat16.
- **Both bridge transports:** Gloo (the default) and UCX (`bridge="ucx"`).
  With UCX on a GPU leader the payloads go straight from GPU memory, with no
  host copy.

| mode | per value on the bridge | float32 bytes saved |
| --- | --- | --- |
| `bf16` | bfloat16, rounded to nearest even | 2x |
| `fp8` | OCP FP8 E4M3 code, plus one float32 scale per 256 values | 3.9x |

## How a compressed bridge step runs

On each bridge member (an island leader, or with `sharded-bridge` the rank
that owns a segment of a link):

1. **`compress`**: a Triton kernel adds this member's error-feedback residual
   (below) to its island sum and quantizes it into a byte payload, on the GPU,
   so the copy to host memory already carries fewer bytes.
2. **`bridge`**: every member broadcasts its payload in turn, so each ends up
   with every island's payload. Payloads are uint8 tensors, since Gloo has no
   fp8 dtype: fp8 travels as the block scales (float32 bytes) followed by the
   codes. The copies to and from host memory happen in this step, as they do
   uncompressed.
3. **`decompress`**: a second kernel dequantizes every payload and adds them
   in float32, in member order, then writes the result in the tensor's dtype.

The pipelined policy compresses the whole island sum once, exchanges the
payloads chunk by chunk (`GPUBRIDGE_CHUNK_BYTES` of payload per message,
copies overlapping the broadcasts as in its uncompressed pipeline), then
decompresses once. The sharded policy compresses each link's segment on its
own rank.

## Every rank still gets the same bits

Uncompressed, every rank ends a collective with the same bits. Compressed, it
still does. Leaders on different vendors run the kernels on different GPUs,
or one may use the PyTorch fallback, so this is by construction rather than by
luck:

- Every member dequantizes the *same* payloads, its own included (it never
  uses its own exact sum), and adds them in the same order.
- Scales are powers of two, and the scale exponent is capped at 119, so the
  largest code times its scale, 448 · 2^119, still fits in float32: every
  product of a code and its scale is exact. A fused multiply-add and a
  separate multiply then round the same way. The kernels are compiled without
  multiply-add contraction all the same (`enable_fp_fusion=False`), so no
  product is ever fused into an addition, and PyTorch's rounding on CPU is
  the kernels' too. Before the cap, a code that decoded beyond float32's range
  (256 · 2^120 = 2^128) left a residual of -inf on CPU and a finite one from a
  fused multiply-add on a GPU.
- fp8 codes and bfloat16 rounding are computed with integer arithmetic on
  float32 bit patterns, not with a GPU's fp8 instructions, which differ
  between vendors (NVIDIA's E4M3 against AMD's FNUZ variants on MI300).
- Magnitudes below 2^-100 count as zero, before quantizing and after every
  addition. So no value a kernel handles is a subnormal, and GPUs that flush
  subnormals to zero agree with those that don't.

The tests check this. The Triton kernels (in Triton's interpreter) and the
plain-PyTorch reference give the same bytes, residuals and results, NaN, ±inf
and tiny values included. In a simulated job with one leader on Triton and
the other on PyTorch, every rank still matches. The GPU kit records digests
of a fixed input on each GPU, so a mixed job's NVIDIA and AMD results can be
compared bit for bit.

## Error bounds

For one value v going into a quantization (the island sum plus the residual,
in float32), the value dequantized from its code differs from v by at most:

- **`bf16`:** 2^-8 |v|, half a bfloat16 unit in the last place.
- **`fp8`:** 2^-4 |v| + 2^-10 s, where s is its block's scale. s is the
  smallest power of two with max|v| / s <= 448 over the block, so
  s <= max|v| / 224, up to its cap of 2^119. The first term is E4M3's half
  unit (3 mantissa bits); the second covers values small enough to land in
  E4M3's subnormal range.
- **Beyond a mode's largest value:** `fp8` sends a value beyond 448 · 2^119
  (about 2.98e38, 87.5% of float32's maximum) as ±448 · 2^119, an error of up
  to 2^-3 |v|. `bf16` sends one that would round past bfloat16's largest
  (from about 3.3961e38) as that largest, 3.3895e38, within its bound above.
  The residual keeps the excess, but a value that stays that large is sent
  as the largest again next step: then what crosses falls short every step.
- **Either mode:** values below 2^-100 become 0.

The result's error, against the exact sum of the island sums, is the sum of
those bounds over the islands, plus float32 rounding while adding (at most
(islands - 1) * 2^-24 of the summed magnitudes), plus one rounding to the
output dtype. That holds while everything stays in range: not for a float64
value beyond float32's, nor for a sum that overflows float32 or an output
dtype (below). The tests check results against exactly this bound, and that
it isn't loose: one quantization of 4,096 values over many magnitudes comes
within 0.94 (`fp8`) and 0.97 (`bf16`) of it.

In practice: `bf16` keeps about three significant decimal digits per value;
`fp8` keeps about one, relative to the largest value in each block of 256.
Neither suits values that must add exactly, which is why integer dtypes, MAX
and MIN are never compressed.

**Non-finite values** stay non-finite, so AMP's overflow check still sees
them: NaN and ±inf become NaN in `fp8`, and stay themselves in `bf16`. Their
residual is 0, so one bad step doesn't poison the next. **Finite values stay
finite:** one beyond a mode's largest is sent as that largest (above), and a
finite float64 beyond float32's range, or a value its residual pushes past it,
counts as float32's largest. So every dequantized value and every residual
is a finite float32. Two things can still overflow, as uncompressed: adding
the islands' values in float32 (two at 1.7e38 round up to 2^127 each and sum
to inf, where the exact sum fits), and the result's dtype (a float16 value
near 65504 can be sent as 65536, inf in float16). Negative zero arrives as
zero.

## Error feedback

Each member keeps a float32 residual per buffer: what its last payload left
out of each value. It adds that to the next call's values before quantizing.
Summed over steps, what crossed the bridge is then the sum of the values
minus the last residual. So the accumulated error stays about one step's
bound instead of growing with the number of steps. The one exception is
saturation at float32's largest: once a value plus its residual passes it,
the part beyond is dropped, every step it does. The tests check that over
500 steps with the same values every step: without error feedback the error
ends 240 (`bf16`) to 325 (`fp8`) times larger.

- **Keyed by buffer.** A residual is found by its buffer's device, address,
  size and dtype (and segment, for `sharded-bridge`). Training loops and DDP
  reuse their gradient buffers, so the residual reaches the next step of the
  same data.
- **Reallocated buffers:** a new buffer at an old address with the same size
  would inherit an old residual. After reallocating gradient buffers, call
  `gpubridge.compression.reset_error_feedback()`. `destroy()` drops every
  residual.
- **Memory:** 4 bytes per value compressed, on bridge members only, held
  until `destroy()`.

## More than two islands

gpubridge makes one island per vendor, so real jobs have at most two today.
Tests relabel ranks to run three and four.

A ring all_reduce over compressed data would dequantize, add and quantize
again at every hop. Each hop adds a fresh quantization error, so with n
islands the last hop's sum carries up to n - 1 compounded roundings, and error
feedback can't correct the intermediate ones: each member only knows its own
residual.

gpubridge doesn't do that. **It accumulates in float32 and never quantizes
twice:** each island's sum is quantized once, its payload reaches every member
unchanged, and each member adds all of them in float32. The error is one
quantization per island, the bound above, however many islands there are.
This is also what lets every rank end with the same bits.

The cost is bandwidth: with n islands each member receives n - 1 payloads,
where an uncompressed ring would move about 2(n - 1)/n tensors. For float32,
`fp8` (payloads about a quarter of the tensor) still sends fewer bytes up to 7
islands, and `bf16` (half) up to 3; at 4 islands `bf16` breaks even. Every
real layout today has 2.

## Phases and observers

On a bridge member, a compressed all_reduce marks `compress`, `bridge` and
`decompress` as their own phases, between the island steps. Observers see
them in `CollectiveRecord.phases`, and `gpubridge.in_flight()` names them
while a thread is inside one. So the bridge phase stays the time spent moving
bytes, copies included as uncompressed, and kernel time is never mistaken for
a slow link. Non-members see only the island phases, as without compression.

## Kernels

`src/gpubridge/_compression_kernels.py` is one Triton source for CUDA and ROCm
builds. gpubridge imports it only when it compresses a GPU tensor. Without
Triton it warns once and runs the PyTorch reference on the GPU instead:
slower, same bits. `GPUBRIDGE_COMPRESSION_KERNELS=triton|torch` forces one
implementation, for testing. It is local: ranks may differ, since the results
don't. Every launch passes `LAUNCH_OPTIONS` (`enable_fp_fusion=False`): a CPU
test compiles each kernel for sm_70 and sm_90, without a GPU, and checks that
its PTX has no fused multiply-add.

- **quantize:** one program per block of 256 values. It reads the values and
  the residual, finds the block's largest finite magnitude and its scale,
  writes the codes, the scale and the new residual. It is a single pass over
  memory.
- **dequantize-accumulate:** one program per block. It reads that block's
  codes and scale from every member's payload, adds in float32 and writes
  the output.

## Measuring it

- `scripts/gpu/bench_compression.py` times both kernels per mode, dtype and
  size, in GB/s of the bytes they must move, against the GPU's memory
  bandwidth (a measured device-to-device copy, and the peak where PyTorch
  reports the memory clock and bus width). It also checks the GPU's kernels
  against the CPU reference and records the digests. `--fp-fusion` launches
  the kernels with multiply-add contraction on: one run each way times what
  gpubridge's default (off) costs.
- `scripts/gpu/bench_all_reduce.py --compression none,bf16,fp8 --phases` times
  the active policy with each mode in one job, with the compress, bridge and
  decompress times. For one random tensor per size it records the error
  against the uncompressed result and whether every rank matched. summary.md
  says from which size each mode is faster than uncompressed.
- The GPU kit runs both (steps `bench_compression_kernels` and
  `bench_compressed` in `amd/run_all.sh`, Explorer steps 04 and 07, and the
  two-machine `mixed/node.sh`). The first GPU run shows where compression
  starts paying off.

## Not done

- No GPU run has checked #50's fix yet. Kernel speed, where compression pays
  off, the cost of compiling without fused multiply-adds, and the kernels'
  bit-identity across NVIDIA and AMD are open (items 28 and 29).
- No training-accuracy study: loss curves with and without compression over
  many steps (`examples/train_synthetic.py` would do).
- Compressed broadcast, all_gather, or a compressed island step.
