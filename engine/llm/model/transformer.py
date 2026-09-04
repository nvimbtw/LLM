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
# what the Rust side uses; changing it changes what every trained model expects,
# so it travels in the checkpoint's arch block rather than being read from here
# at load time. This constant is only the default for a fresh model.
#
# The base sets how fast each channel rotates, and therefore how far apart two
# positions can be before the highest-frequency channels have completed a full
# lap and stopped distinguishing them at all. Raising it pushes that horizon out
# at the cost of fine-grained local resolution.
ROPE_BASE = 10000.0


def rope_tables(
    head_dim: int, ctx_window: int, base: float = ROPE_BASE, max_seq_len: int | None = None
):
    """Precomputed cos/sin tables, shape [max_seq_len or ctx_window, head_dim/2].

    `max_seq_len` extrapolates past `ctx_window` — the length the architecture
    actually trained on — rather than sizing the tables to it. NTK-aware: `base`
    is scaled up by the length ratio raised to `head_dim / (head_dim - 2))`, so
    the low-frequency channels stretch to cover the extra range while the
    highest-frequency channel (the one adjacent tokens are told apart by) barely
    moves. Plain linear position scaling stretches every channel equally and
    blurs exactly that local resolution instead.
    """
    actual_len = max_seq_len or ctx_window
    if max_seq_len and max_seq_len > ctx_window:
        scale = max_seq_len / ctx_window
        base = base * (scale ** (head_dim / (head_dim - 2)))

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(actual_len).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent dimension pairs of `x` [batch, heads, seq, head_dim].

    The tables are [seq, head_dim/2] when every row of the batch sits at the same
    position, and [batch, seq, head_dim/2] when they do not — which is what a
    batch of rollouts at different points in their own episodes looks like.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos.unsqueeze(-3) if cos.dim() == 3 else cos[None, None]
    sin = sin.unsqueeze(-3) if sin.dim() == 3 else sin[None, None]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def attention_mask(seq: int, keys: int, key_mask: torch.Tensor) -> torch.Tensor:
    """Which keys each query may attend to, [batch, 1, seq, keys]. True attends,
    which is `scaled_dot_product_attention`'s own convention.

    Built only when some keys are padding. Without `key_mask` the causal case is
    `is_causal=True`, which is the fused kernel's fast path, and it stays that.

    Query *t* is key `keys - seq + t` — a decode step's queries are the tail of
    the cache — so the causal band is a `tril` shifted by that offset. **A
    padding query is allowed to attend to itself**: a row masked everywhere
    softmaxes to NaN, and that NaN reaches every later query through the cache,
    so the one position whose output nobody reads has to stay finite.
    """
    device = key_mask.device
    allowed = key_mask[:, None, None, :].expand(-1, 1, seq, -1).clone()
    allowed &= torch.ones(seq, keys, dtype=torch.bool, device=device).tril(
        diagonal=keys - seq
    )
    rows = torch.arange(seq, device=device)
    allowed[:, :, rows, keys - seq + rows] = True
    return allowed


class Attention(nn.Module):
    def __init__(self, dimensions: int, heads: int, learned_temperature: bool = False):
        super().__init__()
        self.heads = heads
        self.head_dim = dimensions // heads
        # One softmax temperature per head, learned. The fixed 1/sqrt(head_dim)
        # scale is a single temperature shared by every head, so heads cannot
        # individually choose to look at one token or average over many — they can
        # only move the *content* of their attention, not its sharpness.
        #
        # Held in log space so the temperature stays positive under any update and
        # so zero-init means exactly tau=1: with the parameter off, and at step 0
        # with it on, this is bit-for-bit the arithmetic that came before.
        #
        # None rather than a disabled parameter: an unused nn.Parameter would still
        # enter the state_dict and make every checkpoint incompatible with a build
        # that has the flag off.
        self.log_temperature = (
            nn.Parameter(torch.zeros(heads)) if learned_temperature else None
        )
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

    def forward(self, x, cos, sin, cache=None, key_mask=None):
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

        # Scaling q scales the scores, and scaling the scores IS the temperature:
        # softmax(t*q.k/sqrt(d)) is what a per-head temperature t means. Folding it
        # into q instead of into the scores keeps the whole thing inside
        # `scaled_dot_product_attention` — the fused kernel never has to
        # materialize an attention matrix for this, so per-head temperature costs
        # one elementwise multiply and no flash-attention path.
        if self.log_temperature is not None:
            q = q * self.log_temperature.exp().view(1, -1, 1, 1).to(q.dtype)

        if cache is not None:
            past_k, past_v = cache
            if past_k is not None:
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            cache = (k, v)

        # Causal only when attending over a fresh multi-token sequence. A cached
        # single-token decode step must attend to ALL of the past, so masking it
        # causally would hide every previous token from the one being generated.
        mask = attention_mask(seq, k.shape[2], key_mask) if key_mask is not None else None
        if self.record_attention:
            out = self._attention_recording(q, k, v, mask)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=mask is None and seq > 1
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(out), cache

    def _attention_recording(self, q, k, v, mask=None):
        """The same attention, computed explicitly so the weights can be read."""
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        seq, keys = scores.shape[-2], scores.shape[-1]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        elif seq > 1:
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

    def __init__(
        self,
        dimensions: int,
        heads: int,
        ff_multiplier: int,
        learned_temperature: bool = False,
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dimensions)
        self.attn = Attention(dimensions, heads, learned_temperature)
        self.ln_ff = nn.LayerNorm(dimensions)
        self.ff = FeedForward(dimensions, ff_multiplier)
        self.record_activations = False
        self.last_input: torch.Tensor | None = None

    def forward(self, x, cos, sin, cache=None, key_mask=None):
        # The residual stream as it enters this block. Kept detached and only
        # while diagnostics ask for it; see `Transformer.record_activations`.
        if self.record_activations:
            self.last_input = x.detach()
        attended, cache = self.attn(self.ln_attn(x), cos, sin, cache, key_mask)
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
        rope_base: float = ROPE_BASE,
        learned_temperature: bool = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.dimensions = dimensions
        self.heads = heads
        self.layers = layers
        self.ff_multiplier = ff_multiplier
        self.ctx_window = ctx_window
        self.rope_base = rope_base
        self.learned_temperature = learned_temperature
        self.head_dim = dimensions // heads

        self.embedding = nn.Embedding(vocab, dimensions)
        self.blocks = nn.ModuleList(
            Block(dimensions, heads, ff_multiplier, learned_temperature)
            for _ in range(layers)
        )
        self.ln_final = nn.LayerNorm(dimensions)

        cos, sin = rope_tables(self.head_dim, ctx_window, rope_base)
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

    def extended_rope(self, target_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """NTK-scaled cos/sin tables reaching `target_len`, for a caller that
        wants to run past `ctx_window` without touching the checkpoint.

        Computed fresh and returned rather than written into `rope_cos`/
        `rope_sin`: those buffers are shared by every request against a cached
        model, and overwriting them would race two requests extending to
        different lengths against each other.
        """
        cos, sin = rope_tables(self.head_dim, self.ctx_window, self.rope_base, target_len)
        return cos.to(self.rope_cos.device), sin.to(self.rope_cos.device)

    def forward(self, idx, caches=None, pos_offset: int = 0, rope=None, positions=None,
                key_mask=None):
        """Logits for `idx` [batch, seq], plus the updated KV caches.

        `pos_offset` is where this chunk starts in the sequence, so a cached
        decode step rotates by its true position rather than by 0.

        `rope` overrides the trained `(cos, sin)` tables — with the pair
        `extended_rope` returns, a caller can run past `ctx_window` at whatever
        length it built those tables for.

        `positions` [batch, seq] and `key_mask` [batch, keys] are what a **ragged
        batch** needs, and are meaningless apart: rows padded to a common length
        sit at different true positions and must not attend to each other's
        padding. `pos_offset` cannot express that — it is one number for the
        whole batch. Both default to None and every existing caller passes
        neither, which is why this stays the arithmetic it was.
        """
        rope_cos, rope_sin = rope if rope is not None else (self.rope_cos, self.rope_sin)
        limit = rope_cos.shape[0]
        seq = idx.shape[1]

        if positions is None:
            if pos_offset + seq > limit:
                raise ValueError(f"position {pos_offset + seq} exceeds context window {limit}")
            cos = rope_cos[pos_offset : pos_offset + seq]
            sin = rope_sin[pos_offset : pos_offset + seq]
        else:
            furthest = int(positions.max())
            if furthest >= limit:
                raise ValueError(f"position {furthest + 1} exceeds context window {limit}")
            cos = rope_cos[positions]
            sin = rope_sin[positions]

        x = self.embedding(idx)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cos, sin, caches[i] if caches else None, key_mask)
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
