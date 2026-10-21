# GPUBridge Day 1 handoff: reproducible baseline and native-runtime audit

2026-10-21. Branch `day1/bridge-baseline-personal-20261020`, starting from reviewed `main`
**`38b1e2017c07541cd57f0eeb9d05cd7288b94ae2`** (PR #102), integrated through normal
merge with upstream `main` **`26da9f9a12f762bd49f135c2a7167431a0317fc4`** (PR #101, #103)
at merge commit **`34fed0fa81d8ae79887532258e24c3e763e8e1f3`**, with Enterprise review
findings resolved at **`910aa96e488ce5f1e590def48e9e8c83fdf6cb25`**. Everything here ran
on CPU on the Mac mini: **no GPU, RDMA, Slurm or cloud job was run**, and a pass on the
MOCK verbs fabric is not native GPU or RDMA validation.

## 1. Findings

1. **The baseline is clean where it can run.** At `38b1e20`, both Rust crates pass
   CI's checks (fmt, clippy `-D warnings`, tests: verbs 93 on the mock provider, 96
   with the test fabric; multitcp 12), the Python builds work, and the CPU suite gave
   **1280 passed, 3 failed, 62 skipped** (21 min 32 s). None of the three failures is
   in a collective, a bridge or the verbs path: two were macOS-only (GNU-vs-BSD
   behaviour in the GPU kit's Triton-cache code and one test's host paths, both fixed
   here) and one was a fake-cluster `torchrun` rendezvous timeout under host load
   (passed when rerun alone). Every skip is a platform or hardware requirement and is
   kept (section 4).
2. **The source that ran on hardware is the source tested here.** `38b1e20`'s `src/`
   tree is `ea6c77b9` and its verbs crate `63a58a08`, the trees of the single-host
   hardware runs at `c92684b`/`4c13375`/`4ffcf93`. This branch changes Python under
   `src/` only; both crate trees are unchanged prior to PR #101 integration.
3. **A genuine build defect, fixed:** `GPUBRIDGE_VERBS_LIB_DIR` meant both "where
   `libibverbs.so.1` is" (build.rs, docs) and "where the built library goes" (Python).
   Following the docs built into the rdma-core directory. The cache is now
   `GPUBRIDGE_VERBS_CACHE_DIR` (`d46c25b`).
4. **The measured repair is small, and is reported as small.** `import gpubridge`
   loaded the three opt-in bridges in every process. Lazy registration: 25 → 15
   gpubridge modules, **7.04 → 4.75 ms** with bytecode (-33%), **31.3 → 19.6 ms**
   without (-37%), -0.45 MiB peak RSS, against a torch import of about 390 ms
   (`b5c396f`). API, registration and failure semantics are kept and tested.
5. **Where Python and Gloo run** is mapped in [docs/RUNTIME_INVENTORY.md](docs/RUNTIME_INVENTORY.md)
   with a keep/demote/remove decision per part. Short version: keep the Gloo
   world group (discovery, barrier, subgroups: nothing else exists before a bridge),
   the Gloo bridge as default and `flat-gloo` as reference and real-TCP negative
   controls, NCCL/RCCL islands and the Python integration; qualify verbs as the bulk
   path; demote UCX (never on hardware), multitcp (missed its own gate), the eager
   imports (done), `tagged.py`'s coupling to UCX helpers and build-on-first-use;
   **remove nothing** (no unreferenced code). The highest-value measured next repair
   is the P2P store polling: cross-island P2P took 2.85 ms one way at 1 KiB on
   hardware, against 0.14 ms native and 0.56 ms for a whole Gloo-bridge all_reduce.
6. **Stale status fixed.** The README still said the verbs bridge was untested on
   hardware ("prepared, not run"), "No GPU numbers yet", one GPU run (CONTRIBUTING),
   and that multitcp hadn't run between machines; docs/BENCHMARKS.md had no verbs
   numbers; the roadmap and four design documents still said "tested on CPU only".
   They now state the accepted single-host evidence with its boundaries (`c0b27d5`,
   `e676acc`). HARDWARE_VALIDATION.md and the CHANGELOG were already current and keep
   their per-item rule (status lines describe CPU work; GPU verdicts are in tables).
7. **Upstream PR #101 / #103 reconciled & integrated.** Merged upstream at `26da9f9`
   and integrated here at `34fed0f`. PR #101 adds staged phase timers and direct
   write-window telemetry, while PR #103 introduces opt-in staged copy overlap
   (`GPUBRIDGE_VERBS_STAGED_OVERLAP=1`). Both are opt-in under the verbs transport,
   do not affect Gloo or default imports, leave the `38b1e20` baseline intact, and
   do not introduce multi-node hardware qualification. Default-off bit-for-bit
   identity of staged overlap was verified upstream in PR #103
   (`results/verbs-perf/ab-4-staged-overlap/probes/default-path.json`, commit `d153bf3`),
   not re-executed on hardware here.

## 2. Commits

| Commit | What | Checked by |
| --- | --- | --- |
| `b5c396f` | Load the opt-in bridges only when a job names them | `tests/test_transport.py` (+5 tests, fresh interpreters; the laziness test fails on the old `__init__.py`), full suite |
| `d46c25b` | Give the verbs build cache its own variable | `tests/test_verbs.py::test_the_linker_directory_is_not_where_the_library_goes`, full suite |
| `f21972e` | Make the Triton cache fill and two kit tests portable to BSD userlands | the two tests; the fill checked by hand on BSD `cp` (fixed form `node/triton/oldkernel/k.json`, old form flattened to `node/triton/k.json`) |
| `e3b7375` | Add a one-command CPU baseline with an identity record | `tests/test_baseline.py`, shellcheck, and the post-change runs below, which used it |
| `c0b27d5` | Bring the top-level status in line with the single-host hardware evidence | every number against the accepted report and its 4c13375 results file |
| `7d89e18` | Map where Python and Gloo run, with a keep, demote or remove inventory; `scripts/bench_import.py` | `tests/test_baseline.py` |
| `e676acc` | Retire the remaining blanket "tested on CPU only" status labels | as `c0b27d5`; documentation only |
| `938f5c9` | Reconcile upstream PR 101/103 in runtime inventory and handle redacted test path | `docs/RUNTIME_INVENTORY.md`, `tests/test_gpu_kit.py`, ruff, full suite |
| `899d4a9` | Document Day 1 reproducible baseline audit and handoff | full documentation and JUnit record |
| `34fed0f` | Merge remote-tracking branch 'origin/main' into day1/bridge-baseline-personal-20261020 | 147 focused tests, 96 Rust tests, ruff, shellcheck |
| `ce33d4d` | Update handoff and inventory with integrated HEAD trees, requalification and attribution | `docs/RUNTIME_INVENTORY.md`, `DAY1-HANDOFF.md` |
| `910aa96` | Address Enterprise review findings: ENV_CODE variable, RST docstrings, completion record and BSD cache fill test | `scripts/gpu/verbs_staged_ab.py`, `tests/test_gpu_kit.py`, `scripts/baseline.sh`, `src/gpubridge/transport.py` |

Detector source, its environment and its jobs were not touched. Nothing from the
earlier Personal detector branch is integrated. No shared venv was changed; rustup's
global default was not set (the toolchain is pinned per command).

## 3. Identity

| | |
| --- | --- |
| Machine | Mac mini, Apple silicon (arm64), macOS 27.0.1 (Darwin 27.0.0), 10 CPUs, 16 GiB |
| Python | CPython 3.12.15 (Homebrew), isolated venvs in the worktree (`.venv`, `.day1/venv-torch2.8.0`) |
| Packages | torch 2.14.1 (CPU wheel; Gloo yes, NCCL no) or 2.8.0; numpy 2.5.4; pytest 9.1.1; pytest-timeout 2.4.0; Triton not installable on macOS |
| Rust | rustc 1.99.0 (b940084d7 2026-09-28), cargo 1.99.0 (5f94df478 2026-08-27), clippy 0.1.99, rustfmt 1.10.0, via `RUSTUP_TOOLCHAIN=1.99.0` |
| C | Apple clang 17.0.0 (clang-1700.0.13.5) |
| Lint | ruff 0.17.0 (CI's rule set), shellcheck 0.11.0 (`shellcheck-py`), in `.day1/venv-tools` |
| Trees at `38b1e20` | `src` `ea6c77b9`, verbs crate `63a58a08`, multitcp crate `9b3e9cdb` |
| Trees at `34fed0f` | `src` `fcc9e50e`, verbs crate `afb1034c`, multitcp crate `9b3e9cdb` |
| Trees at `910aa96` | `src` `93b34ecd`, verbs crate `afb1034c`, multitcp crate `9b3e9cdb` |
| Built at `38b1e20` | `libgpubridge_multitcp-afcdd80dde3353f2.dylib` sha256 `97fbfbcc…17235ed`; `libgpubridge_verbs-test-8de9ec89bae1dfcd.dylib` sha256 `080ab804…2a042e` |
| Built at `34fed0f` | `libgpubridge_verbs-test-0406e9b4d36d053a.dylib` |

The multitcp build has the same content key as one cached on Oct 18 but not the same
bytes (different LC_UUID; cargo builds from a temporary copy of the crate, and the
path feeds symbol hashes). The key identifies the source and toolchain, not the
binary: record the SHA-256 of the library a job actually loads
(`scripts/baseline_identity.py` does).

## 4. Results

### Baseline at `38b1e20`, before any change (torch 2.14.1)

| Step | Result |
| --- | --- |
| verbs crate: fmt, clippy `-D warnings`, test | pass; 93 tests |
| verbs crate, `--features test-fabric`: clippy, test | pass; 96 tests |
| verbs crate, `--features ibverbs` | **not run: build.rs refuses non-Linux** ("needs Linux (rdma-core)"); CI's `verbs crate` job covers it on Ubuntu |
| multitcp crate: fmt, clippy, test | pass; 12 tests |
| `python -m gpubridge.multitcp build` / `info` | pass; protocol 1 |
| verbs test-fabric build | pass |
| `python -m gpubridge.verbs build` | refused as designed: "needs Linux with rdma-core (libibverbs); this is darwin" |
| ruff (CI rules), shellcheck (CI's file set) | pass |
| pytest, whole suite (1344 collected + 1 module skipped) | **1280 passed, 3 failed, 62 skipped**, 21 min 32 s |

The 3 failures:

| Test | Cause | Disposition |
| --- | --- | --- |
| `test_gpu_kit.py::test_time_triton_cache_finds_each_directorys_filesystem` | `filesystem()` resolves the path on the host; on macOS `/home` and `/tmp` are symlinks, so a made-up mount table can't match | test fixed (`f21972e`); kit code unchanged (correct on Linux nodes) |
| `test_gpu_kit.py::test_a_job_builds_triton_kernels_in_a_node_local_cache_kept_in_the_store` | BSD `cp -a dir/ dest/` copies the contents, flattening the cache; the store step also needs GNU `mv -T` | `cp` fixed portably; the test skips where `mv` has no `-T` (`f21972e`) |
| `test_gpu_jobs.py::test_step_07_mixed_hetjob` | `verify_fsdp2_backend` in the fake cluster: torchrun's c10d rendezvous timed out at 60 s while the host also ran a venv install and import timings | load-sensitive, not a code failure: passed alone (70.7 s) and in the post-change runs below |

The 62 skips, all kept: UCX bridge is Linux only (39: 36 in `test_ucx.py`, 3 in
`test_p2p.py`); needs a GPU or two (10); the UCX bridge can't be built here (2);
bf16 doesn't compress 16-bit inputs, by design (8); Triton isn't installed (2: one test
and the kernel module); shellcheck not on PATH (1).

### After this branch (pre-merge baseline)

#### PyTorch 2.14.1 (CPython 3.12.15, `.venv`)

All steps recorded in `.day1/after-py312-torch2.14.1/steps.jsonl`:

| Step | Result |
| --- | --- |
| identity & identity_built | pass (`.day1/after-py312-torch2.14.1/identity.json`) |
| verbs crate: fmt, clippy `-D warnings`, test | pass; 93 tests (18 s) |
| verbs crate, `--features test-fabric`: clippy, test | pass; 96 tests (20 s) |
| verbs crate, `--features ibverbs` | skipped: needs Linux with rdma-core |
| multitcp crate: fmt, clippy, test | pass; 12 tests (1 s) |
| `python -m gpubridge.multitcp build` / `info` | pass; protocol 1 (1 s) |
| verbs test-fabric build | pass (1 s) |
| `python -m gpubridge.verbs build` | skipped: needs Linux with rdma-core |
| ruff (CI rules), shellcheck (CI's file set) | pass; 0 errors (`.day1/venv-tools`) |
| pytest, full suite (1353 collected + 1 module skipped) | **1291 passed, 0 failed, 63 skipped**, ~20 min 03 s total |
| `examples/simulate_two_islands.py` | pass; exact tensor sum 10.0 matched on all 4 ranks |

Suite breakdown:
- `tests/test_gpu_jobs.py`: 54 passed in 584.75 s (including `test_step_07_mixed_hetjob` in 66.22 s).
- `tests/test_gpu_kit.py`: 297 passed, 2 skipped in 213.22 s (both portability fixes verified; `test_import_check` path redaction handled).
- Remaining 39 test modules: 940 passed, 61 skipped in 405.17 s.
- 0 failures across all 1354 collected items (+9 new tests from this branch: `test_transport.py` +5, `test_baseline.py` +3, `test_verbs.py` +1).
- All 63 skips are documented hardware/platform requirements (no GPU, no Linux RDMA/UCX, no Triton on macOS).

#### PyTorch 2.8.0 (CPython 3.12.15, `.day1/venv-torch2.8.0`)

Checked against modified/new transport, baseline, and verbs tests:
- `pytest tests/test_transport.py tests/test_baseline.py tests/test_verbs.py`: **25 passed, 31 skipped, 0 failed** in 8.48 s.

#### Import benchmark (`scripts/bench_import.py`)

Fresh interpreter runs alternating between `38b1e20`'s `src` and this branch:
- **With bytecode (30 runs)**: 25 → 15 gpubridge modules (-10), **7.04 → 4.75 ms** median (-2.29 ms, -33%), -0.45 MiB peak RSS.
- **Without bytecode (20 runs)**: 25 → 15 gpubridge modules (-10), **31.31 → 19.57 ms** median (-11.74 ms, -37%), -0.59 MiB peak RSS.

### Requalification on integrated HEAD (`34fed0f` and `910aa96`, including PR #101 & PR #103)

- **Rust Verbs Crate Checks**:
  - `cargo fmt --check --manifest-path src/gpubridge/verbs/crate/Cargo.toml`: pass.
  - `cargo clippy --offline --manifest-path src/gpubridge/verbs/crate/Cargo.toml --all-targets --features test-fabric -- -D warnings`: pass.
  - `cargo test --offline --manifest-path src/gpubridge/verbs/crate/Cargo.toml --features test-fabric`: **96 tests passed** (including write window timing in `direct.rs`).
- **Verbs Test-Fabric Build**: Built `libgpubridge_verbs-test-0406e9b4d36d053a.dylib`.
- **Focused Python Integration Suite**:
  - `pytest tests/test_transport.py tests/test_verbs_staged_ab.py tests/test_verbs_staged_overlap.py tests/test_verbs_gpu.py tests/test_verbs.py tests/test_baseline.py`:
  - **147 passed, 0 failed in 97.13 s**.
  - Requalifies: lazy default loading, opt-in transport resolution, verbs staged phase breakdown, copy overlap, cache/link-dir separation, and identity verification.
- **Linters**: `ruff` and `shellcheck` pass with 0 warnings.

## 5. Reproducing it

From the worktree root, nothing outside it modified:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install "torch==2.14.1" numpy "pytest>=8" pytest-timeout
.venv/bin/python -m pip install -e . --no-deps
export RUSTUP_TOOLCHAIN=1.99.0 PATH="$PWD/.venv/bin:$HOME/.cargo/bin:$PATH"
export GPUBRIDGE_MULTITCP_LIB_DIR="$PWD/.day1/libcache" GPUBRIDGE_VERBS_CACHE_DIR="$PWD/.day1/libcache"
PYTHON=.venv/bin/python scripts/baseline.sh .day1/run      # steps.jsonl, logs, junit.xml, identity.json
```

The oldest supported torch: the same with a second venv on `torch==2.8.0` and
`PYTHON` pointing at it. CI's Python 3.10 leg could not be reproduced here (no 3.10
interpreter; the system 3.9 is below the floor). Lint as CI:
`ruff@0.17.0 check --isolated --select E4,E7,E9,F --extend-exclude "*.ipynb" .` and
`shellcheck scripts/*.sh scripts/gpu/*.sh scripts/gpu/*/*.sh scripts/gpu/explorer/*.sbatch tests/fake_cluster/*`.

The import measurement:

```bash
git archive 38b1e20 src | tar -x -C /tmp/before
python scripts/bench_import.py --before /tmp/before/src --runs 30
python scripts/bench_import.py --before /tmp/before/src --runs 20 --no-bytecode
```

Raw logs, JUnit files and JSON records are in the worktree's untracked `.day1/`.

## 6. Remaining gates

None of these can be settled on this machine; each needs the hardware named.

| Gate | Needs | Where it is tracked |
| --- | --- | --- |
| This branch on Linux CI (3.10 + torch 2.8.0, 3.12 + newest, UCX job, verbs `ibverbs` build and tests) | a pushed branch or PR | `.github/workflows/tests.yml` |
| GPU tests (`pytest -m gpu`), Triton kernels, stream ordering on CUDA and ROCm; the uncaught `wait-skips-stream` mutation needs a test that can fail | an NVIDIA and an AMD GPU | HARDWARE_VALIDATION.md items 16, 28 |
| Verbs between two hosts, host and staged GPU, with a decisive no-silent-TCP verdict and real-TCP negatives | two RDMA-connected nodes | items 38, 39 |
| Verbs direct GPU memory: NVIDIA dma-buf or peer memory, AMD dma-buf | nodes with dma-buf support or `nvidia_peermem` (the 4-V100 node had neither) | items 40, 41 |
| Training over verbs against Gloo across nodes; the matched native-DDP vs gpubridge 1/2/4-GPU series | multi-node GPUs | item 42; BENCHMARKS.md |
| Mixed vendor over verbs (NVIDIA + AMD leaders on one fabric) | an RDMA path between an NVIDIA and an AMD node (none between Explorer's V100 and MI50 nodes) | items 38, 41; BENCHMARKS.md \"RDMA between the node types\" |
| UCX bridge on GPUs and RDMA | UCX on GPU nodes | items 26, 27 |
| More than 4 GPUs; representative-model numerical equivalence over the bridge | 8+ GPUs; a frozen representative workload | NUMERICAL_GATE.md |

Software follow-ups, in order of measured value: replace the P2P store polling
(`p2p.py:259`, `:181`) with a blocking wait of the same timeout and failure behaviour,
measured with `bench_p2p.py` on TCPStore and FileStore; profile the verbs path at 1 to
64 KiB, where it loses to Gloo, before moving `tagged.py`'s reductions native; move
the helpers `tagged.py` imports from `ucx.transport` to a neutral module; a
deprecation notice for multitcp after a usage check.

## 7. Upstream PR #101 and PR #103 Reconciliation (`26da9f9`)

Upstream `main` was checked at commit `26da9f9`, incorporating:
- **PR #101** (`195d7b9`, `friend/verbs-measure-20261019`): Added instrumentation for verbs write window registration and release timing (`direct.rs`), breakdown of host staged collective time into phases (`staged_phases` in `verbs/transport.py`, `verbs/gpu.py`), and a matched one-node A/B benchmark harness (`scripts/gpu/verbs_staged_ab.py`).
- **PR #103** (`26da9f9`, `friend/verbs-pipeline-20261019`): Added opt-in staged copy overlap (`GPUBRIDGE_VERBS_STAGED_OVERLAP=1`) for the two-leader verbs `all_reduce` path, overlapping host-to-device and device-to-host copies with bridge execution where streams permit.

**Architectural Reconciliation:**
1. **Isolated Opt-in Scope**: Both changes live strictly inside the opt-in `verbs` transport path. Neither impacts default `gpubridge` initialization, discovery, barriers, or the default `gloo` transport.
2. **Import Integrity & Evidence Attribution**: Staged timers and copy overlap do not alter the lazy import boundaries established in this branch. Opt-in transports continue to load only when requested. Default-off bit-for-bit identity of staged copy overlap was verified upstream in PR #103 (`results/verbs-perf/ab-4-staged-overlap/probes/default-path.json`, commit `d153bf3`), and is attributed to that run rather than re-executed on hardware here.
3. **Qualification Boundaries**: Neither PR constitutes multi-node, cross-node, or cross-vendor hardware qualification. The `38b1e20` baseline remains the reproducible reference point for this audit, and the integrated HEAD `910aa96` preserves all Day 1 audit guarantees and evidence.

## 8. Enterprise Review Resolution (`DAY1-GPUBRIDGE-REVIEW.md`)

The independent read-only review from Enterprise Opus (`DAY1-GPUBRIDGE-REVIEW.md`) confirmed:
- "Verdict: For 899d4a9, there is no blocking defect in its own scope... For 34fed0f, the merge resolution is sound."
- All concrete in-scope suggestions were addressed at `910aa96`:
  1. **Finding 1 (Cross-repo pin bump)**: Noted for detector pin-bump owner (`GPUBRIDGE_VERBS_CACHE_DIR` usage).
  2. **Finding 2 (`verbs_staged_ab.py`)**: `ENV_CODE` updated to include `GPUBRIDGE_VERBS_CACHE_DIR`; `GPUBRIDGE_VERBS_LIB_DIR` falls under `ENV_SETTINGS`.
  3. **Finding 3 (Doc precision)**: Comment in `src/gpubridge/transport.py` clarifies verbs coupling to `ucx.transport` helpers.
  4. **Finding 5 (BSD `cp` fill test)**: Added `test_triton_cache_begin_fills_node_local_cache_portably_from_store` in `tests/test_gpu_kit.py`, providing automated coverage of the portable `cp -a "${entry%/}"` fill on BSD without requiring GNU `mv -T`.
  5. **Finding 6 (`steps.jsonl` self-attestation)**: Appended final `{"completed": true, "failed": ...}` completion record in `scripts/baseline.sh`.
  6. **Finding 7 (RST docstrings)**: Restored double backticks around `telemetry()["staged_phases"]` in `src/gpubridge/verbs/transport.py`.
