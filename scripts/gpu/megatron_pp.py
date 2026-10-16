"""Megatron-Core's GPT with its own pipeline parallelism across vendors, through
backend="gpubridge" (experimental; docs/MEGATRON.md).

Launched by torchrun, one process per GPU. With ``--pipeline 2`` the world is Megatron's
pipeline group (TP=1, DP=1 per stage): Megatron's 1F1B schedule sends activations and
gradients between the stages with batch_isend_irecv on that group, which on
``backend="gpubridge"`` is native within an island and host-staged across the bridge.
With ``--pipeline 1`` on one GPU it is the baseline. Nothing in Megatron is patched; the
configuration avoids what doesn't run on both vendors: no Transformer Engine, Apex or
fused kernels (the local layer spec, every fusion off), float32 (V100s have no bf16),
no dropout, and untied embeddings (so no embedding all-reduce between the first and last
stages).

Every rank builds the whole model on the CPU from the seed with Megatron set to one
pipeline stage, then each stage keeps its layers' weights, so every split starts from
the weights the one-GPU run has. Tokens follow a fixed rule with noise, as llama.py's:
``(prev * 7 + 13 + e) % vocab``, ``e`` uniform in 0..3, the same for every world size.
Recorded (rank 0 writes ``--json``): every step's loss (the mean over the global batch,
from the last stage), every step's time (the slowest rank's), tokens/s from the median
step after ``--warmup-steps``, and which layers each rank held.

Needs Megatron-Core (``megatron.core``) importable; 0.14.0 has been tried.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from gpukit import KIT_VERSION, default_timeout, environment, run_info, sync, write_json

import gpubridge  # noqa: F401  registers backend="gpubridge"


def batch(step: int, sequences: int, seq: int, vocab: int, seed: int
          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Step ``step``'s (input, target) tokens, ``sequences`` rows of ``seq``."""
    g = torch.Generator().manual_seed(1_000_003 * seed + step)
    tokens = torch.empty(sequences, seq + 1, dtype=torch.int64)
    tokens[:, 0] = torch.randint(vocab, (sequences,), generator=g)
    noise = torch.randint(4, (sequences, seq), generator=g)
    for i in range(seq):
        tokens[:, i + 1] = (tokens[:, i] * 7 + 13 + noise[:, i]) % vocab
    return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def stage_weights(full: dict[str, torch.Tensor], stage: dict[str, torch.Tensor],
                  offset: int) -> dict[str, torch.Tensor]:
    """The whole model's weights a stage holds: its decoder layers (local index i is
    layer offset + i of the whole model) and whatever else it has by name (the
    embedding on the first stage, the final norm and output layer on the last)."""
    out = {}
    for name in stage:
        source = name
        if name.startswith("decoder.layers."):
            local, rest = name[len("decoder.layers."):].split(".", 1)
            source = f"decoder.layers.{offset + int(local)}.{rest}"
        # Megatron's per-module _extra_state (empty with the local spec) isn't a weight.
        out[name] = full[source] if source in full else stage[name]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pipeline", type=int, default=2, help="pipeline stages (= ranks)")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=2, help="sequences per microbatch")
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=5,
                        help="steps left out of the median step time")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="rank 0 writes the result here")
    parser.add_argument("--timeout", type=float, default=default_timeout(1800),
                        help="process-group timeout (s); default KIT_TIMEOUT, else 1800")
    args = parser.parse_args()
    from megatron.core import __version__ as megatron_version
    from megatron.core import parallel_state as ps
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    dist.init_process_group(backend="gpubridge", timeout=timedelta(seconds=args.timeout))
    dist.barrier()  # gpubridge's setup needs every rank before any point-to-point
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != args.pipeline:
        raise SystemExit(f"--pipeline {args.pipeline} needs as many ranks; this run has {world}")
    topology = gpubridge.get_topology()
    device = topology.device
    torch.cuda.set_device(device)
    def gpt(first: bool, last: bool, stages: int) -> GPTModel:
        # Megatron sizes each stage from the config's pipeline_model_parallel_size, not only
        # from parallel_state's groups.
        config = TransformerConfig(
            num_layers=args.layers, hidden_size=args.hidden, num_attention_heads=args.heads,
            hidden_dropout=0.0, attention_dropout=0.0, use_cpu_initialization=True,
            params_dtype=torch.float32, pipeline_dtype=torch.float32,
            pipeline_model_parallel_size=stages)
        return GPTModel(config=config, transformer_layer_spec=get_gpt_layer_local_spec(),
                        vocab_size=args.vocab, max_sequence_length=args.seq_len,
                        pre_process=first, post_process=last,
                        share_embeddings_and_output_weights=False)

    # The whole model, from the seed, as the one-stage run builds it.
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(args.seed)
    torch.manual_seed(args.seed)
    full = gpt(True, True, 1).state_dict()
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(tensor_model_parallel_size=1,
                                 pipeline_model_parallel_size=args.pipeline)
    model_parallel_cuda_manual_seed(args.seed)
    first, last = ps.is_pipeline_first_stage(), ps.is_pipeline_last_stage()
    model = gpt(first, last, args.pipeline)
    layers = [layer.layer_number - 1 for layer in model.decoder.layers]  # 0-based, global
    model.load_state_dict(stage_weights(full, model.state_dict(), layers[0] if layers else 0))
    del full
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    forward_backward = get_forward_backward_func()
    positions = torch.arange(args.seq_len, device=device).unsqueeze(0).expand(
        args.micro_batch, -1)
    sequences = args.micro_batch * args.microbatches

    def forward_step(data, model):
        x, y = next(data)
        out = model(x.to(device), positions, None, labels=y.to(device) if last else None)

        def loss(per_token: torch.Tensor):
            mean = per_token.float().mean()
            # A copy: Megatron divides the returned loss by the microbatches in place.
            return mean, {"loss": mean.detach().clone()}
        return out, loss

    losses, step_s = [], []
    for step in range(args.steps):
        x, y = batch(step, sequences, args.seq_len, args.vocab, args.seed)
        micro = iter(zip(x.split(args.micro_batch), y.split(args.micro_batch), strict=True))
        sync(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        result = forward_backward(forward_step_func=forward_step, data_iterator=micro,
                                  model=[model], num_microbatches=args.microbatches,
                                  seq_length=args.seq_len, micro_batch_size=args.micro_batch,
                                  forward_only=False)
        optimizer.step()
        sync(device)
        step_s.append(time.perf_counter() - start)
        if last:
            losses.append(float(sum(r["loss"] for r in result)) / len(result))

    def gather(value: Any) -> list[Any]:
        everyone: list[Any] = [None] * world
        dist.all_gather_object(everyone, value)
        return everyone

    everyone = gather({"step_s": step_s, "losses": losses, "layers": layers,
                       "vendor": topology.vendor, "device": torch.cuda.get_device_name(device)})
    environments = gather(environment()) if args.json else []
    held = sorted(layer for e in everyone for layer in e["layers"])
    if held != list(range(args.layers)):
        raise SystemExit(f"the stages hold layers {held}, not each of 0 to {args.layers - 1} once")
    losses_all = next(e["losses"] for e in everyone if e["losses"])
    steps = [max(times) for times in zip(*(e["step_s"] for e in everyone))]
    median_step = statistics.median(steps[args.warmup_steps:] or steps)
    ok = all(math.isfinite(v) for v in losses_all)
    if rank == 0:
        record = {
            "kit_version": KIT_VERSION, "script": "scripts/gpu/megatron_pp.py",
            "argv": sys.argv, "run": run_info(topology), "megatron_core": megatron_version,
            "mode": f"megatron-pp{args.pipeline}", "world_size": world, "compression": "none",
            "model": {"layers": args.layers, "hidden": args.hidden, "heads": args.heads,
                      "vocab": args.vocab},
            "seq_len": args.seq_len, "micro_batch": args.micro_batch,
            "microbatches": args.microbatches, "global_batch": sequences,
            "steps": args.steps, "warmup_steps": args.warmup_steps, "seed": args.seed,
            "lr": args.lr, "optimizer": f"AdamW(lr={args.lr})",
            "ranks": [{"rank": r, "vendor": e["vendor"], "device": e["device"],
                       "layers": [e["layers"][0], e["layers"][-1] + 1] if e["layers"] else [],
                       "share": f"{len(e['layers'])} layers"}
                      for r, e in enumerate(everyone)],
            "losses": losses_all, "step_ms": [round(s * 1e3, 3) for s in steps],
            "median_step_ms": round(median_step * 1e3, 3),
            "tokens_per_s": round(sequences * args.seq_len / median_step, 1),
            "loss_first": round(losses_all[0], 4), "loss_last": round(losses_all[-1], 4),
            "environment": environments[0] if environments else None,
            "environments": environments,
        }
        if topology.split_test:
            record["warning"] = "SPLIT TEST: not a mixed-vendor result."
        if args.json:
            write_json(args.json, record)
        print(f"megatron-core {megatron_version}, pp{args.pipeline}: loss "
              f"{record['loss_first']} -> {record['loss_last']}, median step "
              f"{record['median_step_ms']} ms, {record['tokens_per_s']} tokens/s", flush=True)
    ps.destroy_model_parallel()
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
