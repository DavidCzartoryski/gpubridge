# Deterministic numerical comparison before throughput

Candidate frozen2026-10-20. **Tiny GPU gate accepted at `f4146a4`: 48/48 positive rank records, zero violations; all three bad controls rejected. See [hardware record](../CHANGELOG.md#unreleased). Representative GPT equivalence remains unproven.** CPU/Gloo and MOCK verbs
only test the harness and software paths. A four-V100 split test cannot establish
mixed-vendor, cross-node or GPU-direct correctness.

## What the older300-step benchmark controlled

The hardware-tested source pin was616f2f7b591ef014663f1eb6e6150f7d208d75a1;
the candidate's baseline is mergecd9fbac1dc8654ccf4f502330019d22943c9dda6.
Both have tree05dd9cc0bd76f9b2235c051ed77ae8ca7b678cbb. These are distinct refs.

The native105.938samples/s and staged-verbs82.485samples/s runs both used GPT2-small,
FP32, global batch32 split8/8/8/8, seq512, AdamW1e-4, seed0 model initialization,
and four repeating rank-specific batches generated on CPU with seed100+rank. GPT
has no dropout and attention uses no nonzero dropout probability. Both start DDP
with recorded local loss34.8388. No sampler or shuffled external dataset is involved.

However, `train_bench.py` first performs305 **actual local optimizer updates**
(five warmup plus300 measured compute steps). It then broadcasts rank0's evolved
parameters through DDP and creates a fresh AdamW state. The two executions did
not fingerprint this boundary or request deterministic CUDA algorithms. Losses
are rank0's local mean, not the global batch mean. Equal first losses cannot
prove equal full parameter states. The final11.3349 versus13.1502 loss gap neither
proves a reduction bug nor establishes equivalent training. The topology differs:
one native NCCL world versus two NCCL islands joined by staged verbs. The missing
Gloo bridge with that same two-island wiring is an essential control.

Original native/verbs JSON SHA256, respectively:
`993792ceb152f5f48bc20537966ee48ab5f31d7de8877102cd55ba456005700d` and
`4bc29e7593402f5f718fff8e504ffd5b37db388262641f62f3c316f68dac4308`.
Their standalone libibverbs probe is not attestation of the training endpoint.

## Frozen small gate

`scripts/gpu/numerical_gate.py` uses the existing PyTorch/DDP and GPUBridge hook
APIs; transport and native source are unchanged. Four ranks, two equal islands
for each bridge, compression OFF. Seed20261020, global batch32, rank-contiguous
eight-row slices, CPU-generated dyadic inputs/targets,6→5→3 tanh MLP without
bias/dropout/buffers. Three optimizer steps, no pretraining or timed warmup.
Both SGD(lr1/32,momentum0.5) and AdamW(lr1e-3,weight_decay0.01,eps1e-8) are checked.
Each arm starts independently from the same fingerprinted FP32 parameters and
per-rank data. Candidate parameters are never replaced with oracle parameters.

An independent CPU FP64 model trains on each concatenated global batch. Check
every gradient, actual optimizer delta, optimizer state (including AdamW moments)
and parameter after every step. Snapshots are small numerical vectors, not private
checkpoints. Compare all arms and all ranks as well as each arm versus the oracle.
The CPU oracle is software evidence, not distributed GPU/RDMA acceptance.

Before training, element-distinct/rank-distinct integer-valued FP32 SUM and queued
async AVG calls check17,65535,65537 and1048577 elements: below/above256KiB rendezvous
and above4MiB chunk size. These sums and division by four are exactly representable;
**every element must match exactly**. Hardware attestation refuses different
rendezvous/chunk settings, which would invalidate these boundary probes.

Gradient, state and parameter bounds: `abs(error) <= 1e-6 + 1e-5*abs(reference)`,
matching the existing tiny-GPT float gradient bounds in `tests/test_train_bench.py`.
Small FP32 dot products and four-rank summation introduce rounding, not a systematic
factor or missing contribution. Update bounds are tighter near zero:
`abs(error) <= 2e-8 + 1e-4*abs(reference_update)`. Initial weights are at most3/16,
so the absolute allowance covers about two FP32 subtraction roundoffs; updates are
also checked directly to prevent large unchanged parameters hiding missed work.
Report max absolute and relativeL2 errors and element violations. These bounds
were fixed before GPU runs and must not be widened after seeing failures. A native
control repeat outside bounds invalidates this workload gate; diagnose first.

Three deliberately bad controls must fail: `sum` returns SUM where AVG is expected;
`scale` doubles backward loss; `data` changes rank0 inputs. They must respectively
violate collective, gradient and input/gradient gates. AdamW's normalization can
hide a common gradient factor in updates, so gradient evidence is mandatory.

Hardware mode requires CUDA, clean committed source and imports from that exact
checkout; records include actual process-group backends, GPU UUID/PCI identity,
source/tree/file hashes, fingerprints, seeds, workload and frozen bounds. It rejects
simulation and a mock fabric before initialization. Verbs must name an explicit
existing `GPUBRIDGE_VERBS_LIB`; no automatic alternate bridge is accepted.
The actual leaders0/2 record endpoint telemetry before/after, non-mock libibverbs
provider/path, positive completion/byte/staged-call deltas, zero completion errors,
and the loaded library's absolute path/hash. Both leaders must use the same hash.
Comparison rechecks required numerical checks, layout/backends and endpoint proof;
a cached `passed=true` alone is insufficient. Records use exclusive file creation,
and all12 files must share the same fresh run ID, source and workload.

## Owner-only hardware commands

Artemis first reviews the immutable local commits. Only the existing tester may
upload, select prefixes, allocate or submit jobs. Run this serially **inside its
existing exclusive four-GPU allocation** with its prepared CUDA Python environment.
No submission or other owner communication is authorized by these instructions.
`GB` is the reviewed clean checkout, `EVIDENCE` a new directory outside it, and
`GPUBRIDGE_VERBS_LIB` the tester's actual production build (record its source ref).
No native source changed in this candidate; a new native build is not required
solely for these Python changes. Never relabel an old binary as rebuilt.

```bash
set -euo pipefail
: "${GB:?reviewed checkout}" "${EVIDENCE:?new directory outside checkout}"
: "${GPUBRIDGE_VERBS_LIB:?actual production native library}"
test ! -e "$EVIDENCE"
mkdir -p "$EVIDENCE"
export PYTHONPATH="$GB/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 GPUBRIDGE_VERBS_GPU_PATH=staged
export GPUBRIDGE_BRIDGE_COMPRESSION= GPUBRIDGE_VERBS_RNDV_BYTES=262144 GPUBRIDGE_VERBS_CHUNK_BYTES=4194304
unset GPUBRIDGE_VENDOR GPUBRIDGE_CPU_ONLY GPUBRIDGE_VERBS_TEST_FABRIC
RUN_ID=numerics-reviewed-20261020-v1
for opt in sgd adamw; do
  for arm in native gloo verbs; do
    timeout 300 python -m torch.distributed.run --standalone --nproc-per-node=4 \
      "$GB/scripts/gpu/numerical_gate.py" --arm "$arm" --optimizer "$opt" \
      --run-id "$RUN_ID-$opt" --out "$EVIDENCE/$opt-$arm" \
      >"$EVIDENCE/$opt-$arm.stdout" 2>"$EVIDENCE/$opt-$arm.stderr"
  done
  files=()
  for arm in native gloo verbs; do
    for rank in 0 1 2 3; do files+=("$EVIDENCE/$opt-$arm/rank$rank.json"); done
  done
  python "$GB/scripts/gpu/numerical_gate.py" --compare "${files[@]}" \
    --out "$EVIDENCE/$opt-comparison.json"
done
# Reduced deliberately incorrect controls: their nonzero exit is required.
for bad in sum scale data; do
  if timeout 300 python -m torch.distributed.run --standalone --nproc-per-node=4 \
      "$GB/scripts/gpu/numerical_gate.py" --arm gloo --negative-control "$bad" \
      --run-id "$RUN_ID-negative-$bad" --out "$EVIDENCE/negative-$bad" \
      >"$EVIDENCE/negative-$bad.stdout" 2>"$EVIDENCE/negative-$bad.stderr"; then
    echo "negative control unexpectedly passed: $bad" >&2; exit 1
  fi
done
```

Inspect all four negative-control rank records: a crash/timeout or missing file
does not prove detection. Require the intended AVG, gradient or fingerprint
violations above. Repeat both positive optimizer matrices with a new runID and
reversed arm order (verbs,gloo,native); compare each repeat independently. Gate:
24 complete rank records per optimizer across two repeats, all exact probes and
numerical bounds pass, no provider/path violations, every reduced negative fails
for its intended reason. Preserve every result, including failures, with a SHA256
manifest. A failed gate stops representative training/performance acceptance.

## Representative training and fair timing, only after the gate

`train_bench.py --controlled` enables deterministic algorithms, disables TF32,
fingerprints parameters and per-rank batches, and rebuilds the seeded model after
the compute probe. It fails if the DDP boundary differs from the initial model.
`--balance` is refused in this fixed-batch mode. Default benchmark behavior is
unchanged without the flag. Fingerprints are per rank in `ranks`; these must match
between corresponding arms/repeats, with all ranks sharing the same initial state.

First use the command below with `--model tiny-gpt --seq-len 32 --steps 20`; retain
losses/parameter evidence for diagnosis. Then use the fixed representative workload:

```bash
# One arm inside each serial repeat; fresh result paths, existing allocation.
# Gloo/verbs share exactly the same two-island configuration. Native is homogeneous.
arm=gloo  # repeat for verbs and native in the balanced orders below
sync=gpubridge; bridge=$arm; split=half
if [ "$arm" = native ]; then sync=native; bridge=gloo; split=; fi
GPUBRIDGE_SPLIT_TEST="$split" timeout 1800 \
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
  "$GB/scripts/gpu/train_bench.py" --controlled --model gpt2-small \
  --global-batch 32 --seq-len 512 --warmup-steps 10 --steps 100 \
  --lr 0.0001 --bucket-mb 25 --comm-trials 5 --timeout 120 \
  --grad-sync "$sync" --bridge "$bridge" --json "$EVIDENCE/$arm-repeat1.json"
```

Six blocks: native/gloo/verbs; gloo/verbs/native; verbs/native/gloo;
native/verbs/gloo; verbs/gloo/native; gloo/native/verbs. Use fresh paths per block,
capture stdout/stderr and actual source/binary/provider evidence. Require matching
initial/data fingerprints, fixed global batch/schedule and complete rank records.
Warmup is separate; report median step time/throughput for each repeat, paired
verbs−Gloo differences and a95% paired bootstrap interval over the **six blocks**
(not over correlated individual steps). NCCL is the homogeneous baseline. A
speedup claim needs an interval excluding zero and reproducible numerical gates;
one timing or final-loss sample cannot support it. No timing outcome is predicted.
Per-rank `final_sha256` fingerprints expose final-state differences; unequal
hashes require numerical investigation, since reduction order can change bits.
Passing the tiny gate does not prove numerical equivalence of representative
GPT training; full representative gradient/parameter error norms remain unproven.
