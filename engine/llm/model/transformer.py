"""A decoder-only transformer.

The architecture is carried over from the Rust implementation, where each choice
was already paid for:

  * RoPE, rotating adjacent pairs (2i, 2i+1) of each head's dimensions — no
    learned positional table, and the context window is not baked into a weight.
  * Pre-LN blocks, each normalizing its own input and adding into the residual.
  * A final LayerNorm over the residual stream. The blocks are Pre-LN, so
    nothing normalizes the stream itself and its scale grows with depth; without
    this the logits grow with it.
  * Tied embeddings: one [vocab, d] matrix gathers rows going in and multiplies
    transposed coming out. Worth roughly one doubling of the vocabulary at equal
    parameter count, and near-free in compute.

What PyTorch supplies and the Rust had to write by hand — the backward pass, the
optimizer, the CUDA workspace, the attention kernel — is simply absent here.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# The frequency base for RoPE. 10000 is the value from the original paper and
# what the Rust side uses; changing it changes what every trained model expects.
ROPE_BASE = 10000.0


def rope_tables(head_dim: int, ctx_window: int, base: float = ROPE_BASE):
    """Precomputed cos/sin tables, shape [ctx_window, head_dim/2]."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(ctx_window).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent dimension pairs of `x` [batch, heads, seq, head_dim]."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, dimensions: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dimensions // heads
        # Diagnostics only. `scaled_dot_product_attention` never materializes the
        # attention matrix — that is most of why it is fast — so measuring where
        # the heads look needs an explicit softmax. Off by default and flipped on
        # for a single step by the diagnostics collector, so the training path
        # keeps the fused kernel.
        self.record_attention = False
        self.last_attention: torch.Tensor | None = None
        # Q, K and V as one [d, 3d] projection rather than three [d, d] ones.
        # Same arithmetic — the weight is their vertical concatenation and the
        # output is sliced back apart — but one GEMM instead of three, and on
        # this hardware that is not a wash. A [4096,256]x[256,256] matmul only
        # reaches 6.9 TFLOPS on an RTX 3060 against a 18.7 peak, because 256
        # output columns tile into too few thread blocks to fill 28 SMs; at 768
        # columns the same kernel reaches 15.4. Measured 3.7% off the step.
        self.wqkv = nn.Linear(dimensions, 3 * dimensions, bias=False)
        self.wo = nn.Linear(dimensions, dimensions, bias=False)

    def init_projections(self) -> None:
        """Initialize `wqkv` as the three separate projections it stands in for.

        Xavier's bound is sqrt(6 / (fan_in + fan_out)), so initializing a [d, 3d]
        weight as one matrix would draw from a visibly narrower distribution than
        three [d, d] ones — a quieter Q, K and V at step 0 for no reason but the
        layout. Each third gets the fan it would have had on its own.
        """
        for third in self.wqkv.weight.data.chunk(3, dim=0):
            nn.init.xavier_uniform_(third)

    def forward(self, x, cos, sin, cache=None):
        batch, seq, _ = x.shape

        # [batch, seq, 3d] -> 3 x [batch, heads, seq, head_dim]. The projection's
        # output is laid out q|k|v, so this is a view and an unbind, not a copy.
        q, k, v = (
            self.wqkv(x)
            .view(batch, seq, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            past_k, past_v = cache
            if past_k is not None:
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            cache = (k, v)

        # Causal only when attending over a fresh multi-token sequence. A cached
        # single-token decode step must attend to ALL of the past, so masking it
        # causally would hide every previous token from the one being generated.
        if self.record_attention:
            out = self._attention_recording(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=seq > 1)

        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(out), cache

    def _attention_recording(self, q, k, v):
        """The same attention, computed explicitly so the weights can be read."""
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        seq, keys = scores.shape[-2], scores.shape[-1]
        if seq > 1:
            causal = torch.ones(seq, keys, dtype=torch.bool, device=scores.device).tril(
                diagonal=keys - seq
            )
            scores = scores.masked_fill(~causal, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        self.last_attention = weights.detach()
        return weights @ v

    def _load_from_state_dict(self, state_dict, prefix, *args):
        """Accept checkpoints written before Q, K and V shared one projection.

        `wqkv` is exactly `cat([wq, wk, wv])` — nn.Linear stores its weight
        [out, in], and the fused output is laid out q|k|v — so an old checkpoint
        converts here rather than needing a migration pass over every file on
        disk. Weights saved by this build already have the fused key and skip it.
        """
        legacy = [f"{prefix}w{name}.weight" for name in ("q", "k", "v")]
        if all(key in state_dict for key in legacy):
            state_dict[f"{prefix}wqkv.weight"] = torch.cat(
                [state_dict.pop(key) for key in legacy], dim=0
            )
        super()._load_from_state_dict(state_dict, prefix, *args)


class FeedForward(nn.Module):
    def __init__(self, dimensions: int, ff_multiplier: int):
        super().__init__()
        width = dimensions * ff_multiplier
        self.w1 = nn.Linear(dimensions, width)
        self.w2 = nn.Linear(width, dimensions)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class Block(nn.Module):
    """One Pre-LN transformer block."""

    def __init__(self, dimensions: int, heads: int, ff_multiplier: int):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dimensions)
        self.attn = Attention(dimensions, heads)
        self.ln_ff = nn.LayerNorm(dimensions)
        self.ff = FeedForward(dimensions, ff_multiplier)
        self.record_activations = False
        self.last_input: torch.Tensor | None = None

    def forward(self, x, cos, sin, cache=None):
        # The residual stream as it enters this block. Kept detached and only
        # while diagnostics ask for it; see `Transformer.record_activations`.
        if self.record_activations:
            self.last_input = x.detach()
        attended, cache = self.attn(self.ln_attn(x), cos, sin, cache)
        x = x + attended
        x = x + self.ff(self.ln_ff(x))
        return x, cache


class Transformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        dimensions: int,
        heads: int,
        layers: int,
        ff_multiplier: int,
        ctx_window: int,
    ):
        super().__init__()
        self.vocab = vocab
        self.dimensions = dimensions
        self.heads = heads
        self.layers = layers
        self.ff_multiplier = ff_multiplier
        self.ctx_window = ctx_window
        self.head_dim = dimensions // heads

        self.embedding = nn.Embedding(vocab, dimensions)
        self.blocks = nn.ModuleList(
            Block(dimensions, heads, ff_multiplier) for _ in range(layers)
        )
        self.ln_final = nn.LayerNorm(dimensions)

        cos, sin = rope_tables(self.head_dim, ctx_window)
        # Buffers, not parameters: derived from the architecture, and they move
        # with .to(device) instead of having to be passed around.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # After the sweep, not instead of it: `apply` sees an nn.Linear and
        # cannot know that this one stands for three.
        for block in self.blocks:
            block.attn.init_projections()

    @staticmethod
    def _init_weights(module):
        # Glorot/Xavier, matching the Rust `embed()` initializer.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, caches=None, pos_offset: int = 0):
        """Logits for `idx` [batch, seq], plus the updated KV caches.

        `pos_offset` is where this chunk starts in the sequence, so a cached
        decode step rotates by its true position rather than by 0.
        """
        seq = idx.shape[1]
        if pos_offset + seq > self.ctx_window:
            raise ValueError(
                f"position {pos_offset + seq} exceeds context window {self.ctx_window}"
            )

        cos = self.rope_cos[pos_offset : pos_offset + seq]
        sin = self.rope_sin[pos_offset : pos_offset + seq]

        x = self.embedding(idx)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, caches[i] if caches else None)
            new_caches.append(cache)

        x = self.ln_final(x)
        # Tied: the embedding table IS the output projection, transposed.
        logits = x @ self.embedding.weight.T
        return logits, (new_caches if caches is not None else None)

    def empty_caches(self):
        """Fresh per-layer KV cache slots for an incremental decode."""
        return [(None, None) for _ in range(self.layers)]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
