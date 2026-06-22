# Architecture

GPT-Simple ships one model, `SimpleLLM` (`src/gpt_simple/model.py`): a
pre-norm, decoder-only transformer in the modern Llama style. The
defaults describe an RMSNorm + RoPE + SwiGLU decoder with a tied output
head and no biases, but the same code expresses other dense decoder-only
families through configuration alone.

All architecture is driven by `ModelConfig` — see
[Configuration](configuration.md) for the full field list.

## Block structure

Each of the `n_layer` transformer blocks is pre-norm with two residual
branches:

```
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

A final norm precedes the output projection. Pre-norm (normalizing the
branch input rather than its output) is the standard choice for training
stability in deep decoders.

## Components

### Normalization
`norm: rmsnorm` (default) uses RMSNorm; `norm: layernorm` uses
`nn.LayerNorm`. RMSNorm runs its reduction in float32 and casts back, for
numerical stability under mixed precision.

### Positional encoding — RoPE
Rotary Position Embeddings are applied to queries and keys each layer
(there is no learned position embedding table). Cos/sin tables are cached
per `(sequence_length, device)`. Two optional scaling modes extend the
trained context length:

- `rope_scaling_type: linear` — divides the rotation frequencies by
  `rope_scaling_factor`.
- `rope_scaling_type: ntk` — scales the RoPE base by `rope_scaling_factor`.

### Attention
Multi-head self-attention with separate Q/K/V projections. The number of
key/value heads is configurable via `n_kv_head`:

- `n_kv_head = n_head` (default) — standard multi-head attention.
- `1 < n_kv_head < n_head` — grouped-query attention (must divide
  `n_head`).
- `n_kv_head = 1` — multi-query attention.

Grouped/multi-query heads are stored compactly and expanded to the query
head count only inside the attention computation, so the KV-cache keeps
its memory advantage during generation.

### Feed-forward (MLP)
Two variants, selected by `mlp_type`:

- `gated` (default) — a gated MLP (`down(up * act(gate))`). With
  `activation: swish` this is SwiGLU; `gelu`/`relu` give GeGLU/ReGLU.
  Inner width defaults to `round_to_256(8 * n_embd / 3)` (Llama sizing).
- `mlp` — a vanilla FFN (`proj(act(fc(x)))`). Inner width defaults to
  `4 * n_embd` (GPT sizing).

`intermediate_size` overrides the derived inner width in either case.

### Output head and weight tying
By default the output projection reuses the token-embedding matrix
(`tie_word_embeddings: true`). Set it to `false` to allocate a dedicated
`lm_head` (for a faithful Llama-1/2 replica, which does not tie).

### Biases
All linear layers are bias-free by default (`use_bias: false`). Biases
can be enabled globally or per projection group via `qkv_bias`,
`attn_out_bias`, and `mlp_bias` (e.g. Qwen2-style Q/K/V-only biases).

## Attention backends

`attention_mode` selects how attention is computed. This is a
performance/capability trade-off, not an architecture change — see
[Hardware tuning](hardware-tuning.md) for the speed implications.

| Mode | Behavior | When to use |
| ---- | -------- | ----------- |
| `causal` (default) | `scaled_dot_product_attention(..., is_causal=True)` — no mask tensor, Flash-Attention eligible, fastest. | Almost always. |
| `sdpa_mask` | Materializes an additive mask (causal + per-document + padding). | Packed sequences that need strict per-document masking. |
| `flex` | `flex_attention` with a compiled block mask. | Custom mask patterns; requires PyTorch ≥ 2.5. |

A manual attention path is used as a fallback on devices without SDPA.

When sequences are packed (see [Data pipeline](data.md)), `causal`
relies on the packing scheme; `sdpa_mask` and `flex` additionally consume
`doc_ids` to prevent attention across document boundaries.

## Initialization

Linear and embedding weights use a normal `std=0.02` init. Residual
output projections (attention `c_proj`, MLP `c_down`) are additionally
scaled by `1/sqrt(2 * n_layer)` — the GPT-2/GPT-3 residual scaling that
keeps activation variance stable in deep stacks.

## Memory and inference features

- **Gradient checkpointing** — when enabled, each block recomputes its
  attention and MLP in the backward pass instead of storing activations
  (trades compute for memory). Automatically disabled during cached
  generation.
- **KV-cache** — `generate` caches per-layer keys/values so each new
  token only attends over the growing cache rather than recomputing the
  full prefix.

## Loss

Causal language-modeling loss (next-token cross-entropy) with label
shifting. Positions labeled `-100` (padding, post-EOD tokens, and
windowing overlaps) are excluded. The loss reduction runs in float32.

---

Authoritative source: `src/gpt_simple/model.py`,
`src/gpt_simple/config.py`.
