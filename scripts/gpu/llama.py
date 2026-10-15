"""A small LLaMA-style decoder and its synthetic data, for the training runs that compare
parallel layouts across vendors (train_dp.py; the pipeline runs later).

Built from torch.nn alone: token embedding, then pre-norm blocks of RMSNorm, causal
self-attention with rotary position embeddings (grouped-query when ``kv_heads`` is
smaller than ``heads``) and a SwiGLU MLP, then a final RMSNorm and an untied output
layer. ``blocks`` is a plain ModuleList, so a pipeline can cut it into contiguous
stages. Sizes (parameters counted by :func:`parameters`):

- ``llama-125m``: width 768, 12 blocks, 12 heads, MLP 2048, vocabulary 32000: 134M.
- ``llama-350m``: width 1024, 24 blocks, 16 heads, MLP 2816: 374M.
- ``llama-1b``: width 2048, 20 blocks, 32 heads with 8 key/value heads, MLP 5632: 1.03B.
- ``llama-tiny``: width 32, 4 blocks (enough for a 4-stage pipeline), for CPU tests.

Each fits a 32 GB V100 or MI50 with float32 AdamW states at a few sequences per GPU
(llama-1b: 16 GB of weights, gradients and optimizer state).

:func:`build` makes the same weights on every rank and every vendor: they are drawn on
the CPU from one seed, then moved. :func:`global_batch` makes step ``s``'s whole batch the
same way for any world size, so a single GPU and several ranks splitting it see the same
tokens. The tokens follow a fixed rule with a little noise (each next token is
``(prev * 7 + 13 + e) % vocab``, ``e`` uniform in 0..3), so the loss falls from about
ln(vocab) towards ln(4) as the model learns it, and loss curves can be compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LlamaConfig:
    width: int
    blocks: int
    heads: int
    kv_heads: int
    mlp: int
    vocab: int
    context: int
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0


MODELS: dict[str, LlamaConfig] = {
    "llama-125m": LlamaConfig(width=768, blocks=12, heads=12, kv_heads=12, mlp=2048,
                              vocab=32000, context=2048),
    "llama-350m": LlamaConfig(width=1024, blocks=24, heads=16, kv_heads=16, mlp=2816,
                              vocab=32000, context=2048),
    "llama-1b": LlamaConfig(width=2048, blocks=20, heads=32, kv_heads=8, mlp=5632,
                            vocab=32000, context=2048),
    "llama-tiny": LlamaConfig(width=32, blocks=4, heads=4, kv_heads=2, mlp=64, vocab=128,
                              context=64),
}


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotary(seq: int, head_width: int, theta: float, device: torch.device,
           dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin for rotary embeddings, (seq, head_width / 2). Computed per call, not
    kept as buffers, so DDP and FSDP have nothing to broadcast or shard for them."""
    inv = 1.0 / theta ** (torch.arange(0, head_width, 2, device=device, dtype=torch.float32)
                          / head_width)
    angles = torch.outer(torch.arange(seq, device=device, dtype=torch.float32), inv)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary embedding of ``x`` (batch, heads, seq, head_width), pairs (even, odd)."""
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: LlamaConfig) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_width = c.heads, c.kv_heads, c.width // c.heads
        self.theta = c.rope_theta
        self.q = nn.Linear(c.width, c.width, bias=False)
        self.kv = nn.Linear(c.width, 2 * c.kv_heads * self.head_width, bias=False)
        self.out = nn.Linear(c.width, c.width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, w = x.shape
        q = self.q(x).view(b, t, self.heads, self.head_width).transpose(1, 2)
        k, v = (z.view(b, t, self.kv_heads, self.head_width).transpose(1, 2)
                for z in self.kv(x).chunk(2, dim=-1))
        cos, sin = rotary(t, self.head_width, self.theta, x.device, x.dtype)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        if self.kv_heads != self.heads:
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, t, w))


class SwiGLU(nn.Module):
    def __init__(self, c: LlamaConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(c.width, c.mlp, bias=False)
        self.up = nn.Linear(c.width, c.mlp, bias=False)
        self.down = nn.Linear(c.mlp, c.width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: LlamaConfig) -> None:
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.width, c.norm_eps), RMSNorm(c.width, c.norm_eps)
        self.attention, self.mlp = Attention(c), SwiGLU(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Llama(nn.Module):
    def __init__(self, c: LlamaConfig) -> None:
        super().__init__()
        self.config = c
        self.embed = nn.Embedding(c.vocab, c.width)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.blocks))
        self.norm = RMSNorm(c.width, c.norm_eps)
        self.head = nn.Linear(c.width, c.vocab, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build(name: str, seed: int = 0, device: torch.device | str = "cpu") -> Llama:
    """Model ``name`` with weights drawn on the CPU from ``seed`` (the same bits on every
    rank and vendor), then moved to ``device``. Linear and embedding weights are normal
    with std 0.02, the output projections of each block scaled by 1/sqrt(2 * blocks)."""
    c = MODELS[name]
    g = torch.Generator().manual_seed(seed)
    model = Llama(c)
    with torch.no_grad():
        for module_name, p in model.named_parameters():
            if p.dim() == 1:
                continue  # RMSNorm weights stay at one
            std = 0.02
            if module_name.endswith(("attention.out.weight", "mlp.down.weight")):
                std /= math.sqrt(2 * c.blocks)
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return model.to(device)


def parameters(name: str) -> int:
    """How many parameters model ``name`` has, without building it."""
    c = MODELS[name]
    kv = 2 * c.kv_heads * (c.width // c.heads)
    block = c.width * (2 * c.width + kv) + 3 * c.width * c.mlp + 2 * c.width
    return 2 * c.vocab * c.width + c.blocks * block + c.width


def global_batch(name: str, step: int, sequences: int, seq: int, seed: int = 0
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Step ``step``'s whole batch: ``sequences`` (input, target) rows of ``seq`` tokens,
    the same for any world size (see the module docstring for the rule)."""
    vocab = MODELS[name].vocab
    g = torch.Generator().manual_seed(1_000_003 * seed + step)
    tokens = torch.empty(sequences, seq + 1, dtype=torch.int64)
    tokens[:, 0] = torch.randint(vocab, (sequences,), generator=g)
    noise = torch.randint(4, (sequences, seq), generator=g)
    for i in range(seq):
        tokens[:, i + 1] = (tokens[:, i] * 7 + 13 + noise[:, i]) % vocab
    return tokens[:, :-1], tokens[:, 1:]


def loss_of(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy over every token of the batch."""
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
