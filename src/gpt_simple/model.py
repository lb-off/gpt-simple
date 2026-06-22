"""
SimpleLLM - Educational GPT Implementation
A simplified but functional implementation of the GPT (Generative Pre-trained Transformer) architecture.

This implementation follows the GPT-2 architecture with some simplifications for educational purposes:
- Standard multi-head attention with causal masking
- Feed-forward networks with GELU activation
- Layer normalization and residual connections
- Rotary Position Embeddings (RoPE) for positional encoding

Key design choices:
1. Pre-norm architecture (layer norm before attention/FFN) for better training stability
2. Rotary Position Embeddings (RoPE) for better length extrapolation
3. Flash Attention when available for memory efficiency
4. Gradient checkpointing support for memory-compute tradeoff
"""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple
from dataclasses import dataclass

from gpt_simple.config import ModelConfig

logger = logging.getLogger("gpt_simple")


@dataclass
class CausalLMOutput:
    """Lightweight replacement for transformers.CausalLMOutputWithPast.

    Supports both attribute access (``out.loss``) and dict-style access
    (``out["loss"]``, ``"loss" in out``) for backward compatibility.
    """
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    More stable than LayerNorm and used in modern LLMs like LLaMA.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        # Ensure weight is in float32 for computation, then convert result back to input dtype
        return (self.weight.to(torch.float32) * x).to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) with optional scaling and caching.
    Generates cos/sin tables of shape (1,1,seq_len,dim) and caches them for efficiency.
    """
    def __init__(self, dim: int, base: float = 10000.0,
                 scaling_type: Optional[str] = None, scaling_factor: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.base = float(base)
        self.scaling_type = scaling_type
        self.scaling_factor = float(scaling_factor)

        # Build inverse frequencies (half-dim)
        self.register_buffer("inv_freq_base",
                             (torch.arange(0, dim, 2).float() / dim),
                             persistent=False)
        
        # Cache for cos/sin tables to avoid recomputation
        # Format: {(seq_len, device_str): (cos_tensor, sin_tensor)}
        self._cos_sin_cache = {}

    def _inv_freq(self, device):
        # Apply scaling: linear -> divide theta by factor (equiv to inv_freq / factor)
        # NTK-ish -> increase base by factor (simple, popular approximation)
        if self.scaling_type == "linear" and self.scaling_factor > 1.0:
            base = self.base
            inv = (base ** (-self.inv_freq_base)) / self.scaling_factor
        elif self.scaling_type == "ntk" and self.scaling_factor > 1.0:
            base = self.base * self.scaling_factor
            inv = base ** (-self.inv_freq_base)
        else:
            base = self.base
            inv = base ** (-self.inv_freq_base)
        return inv.to(device)

    def forward(self, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create cache key using sequence length and device string
        device_str = str(device)
        cache_key = (seq_len, device_str)
        
        # Check if we have cached cos/sin tables for this configuration
        if cache_key in self._cos_sin_cache:
            cos, sin = self._cos_sin_cache[cache_key]
            # Verify the cached tensors are still on the correct device
            if cos.device == device and sin.device == device:
                return cos, sin
            else:
                # Remove stale cache entry if device changed
                del self._cos_sin_cache[cache_key]
        
        # Compute cos/sin tables if not cached
        inv_freq = self._inv_freq(device)                               # (half,)
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)  # (L,)
        freqs = torch.outer(t, inv_freq)                                # (L, half)
        emb = torch.cat((freqs, freqs), dim=-1)                         # (L, dim)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)                       # (1,1,L,dim)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        # Cache the computed tables
        self._cos_sin_cache[cache_key] = (cos, sin)
        
        return cos, sin
    
    def clear_cache(self):
        """Clear the cos/sin cache. Useful for memory management or when changing devices."""
        self._cos_sin_cache.clear()


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped key/value heads to match the number of query heads.

    x: (B, n_kv_head, T, D) -> (B, n_kv_head * n_rep, T, D).
    Used for grouped-query / multi-query attention; ``n_rep == 1`` is a no-op
    (plain multi-head attention).
    """
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, H, n_rep, T, D)
    return x.reshape(B, H * n_rep, T, D)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """
    Apply rotary position embedding to query and key tensors.

    q,k: (B, H, T, D)
    cos,sin: (1,1,L, D) where L >= max(position_ids)+1 (if provided) or >= T
    position_ids: (B, T) absolute positions for current tokens only
    """
    if position_ids is not None:
        # index cos/sin -> (B,1,T,D)
        cos = cos.squeeze(0).squeeze(0)         # (L, D)
        sin = sin.squeeze(0).squeeze(0)         # (L, D)
        cos = cos[position_ids].unsqueeze(1)    # (B,1,T,D)
        sin = sin[position_ids].unsqueeze(1)    # (B,1,T,D)
    # else broadcast (1,1,T,D) against q,k
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.

    Supports three attention backends selected via ``config.attention_mode``:
      - ``"causal"``: ``is_causal=True`` (Flash Attention eligible, fastest)
      - ``"sdpa_mask"``: materialized additive mask with doc-boundary masking
      - ``"flex"``: PyTorch ``flex_attention`` with compiled block masks (experimental)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.kv_heads
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_head

        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        q_dim = self.n_head * self.head_dim       # == n_embd
        kv_dim = self.n_kv_head * self.head_dim    # smaller under GQA/MQA

        # Separate Q/K/V projections so key/value can have fewer heads than
        # query (grouped-/multi-query attention).  With n_kv_head == n_head
        # this is exactly standard multi-head attention.
        self.q_proj = nn.Linear(self.n_embd, q_dim, bias=config.resolved_qkv_bias)
        self.k_proj = nn.Linear(self.n_embd, kv_dim, bias=config.resolved_qkv_bias)
        self.v_proj = nn.Linear(self.n_embd, kv_dim, bias=config.resolved_qkv_bias)
        self.c_proj = nn.Linear(q_dim, self.n_embd, bias=config.resolved_attn_out_bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    # ------------------------------------------------------------------
    # Attention backends
    # ------------------------------------------------------------------

    def _attn_causal(self, q, k, v):
        """Pure causal SDPA – enables the Flash Attention kernel."""
        return F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )

    def _attn_sdpa_mask(self, q, k, v, B, attention_mask, doc_ids):
        """SDPA with explicit additive mask (causal + doc-boundary + padding)."""
        NEG_INF = torch.finfo(q.dtype).min

        T_q, T_k = q.size(-2), k.size(-2)
        past_kv = T_k - T_q
        i = torch.arange(T_q, device=q.device)
        j = torch.arange(T_k, device=q.device)
        causal_bool = j.unsqueeze(0) > (past_kv + i.unsqueeze(1))

        add_mask = torch.zeros((T_q, T_k), device=q.device, dtype=q.dtype)
        add_mask = add_mask.masked_fill(causal_bool, NEG_INF)

        if doc_ids is not None and doc_ids.dim() == 2 and doc_ids.size(1) >= T_k:
            doc_ids_q = doc_ids[:, :T_q]
            doc_ids_k = doc_ids[:, :T_k]
            cross_doc_mask = doc_ids_q[:, :, None] != doc_ids_k[:, None, :]
            add_mask = add_mask.unsqueeze(0).expand(B, T_q, T_k)
            add_mask = add_mask.masked_fill(cross_doc_mask, NEG_INF)
            add_mask = add_mask.unsqueeze(1).expand(B, self.n_head, T_q, T_k)
        else:
            add_mask = add_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_head, T_q, T_k)

        if attention_mask is not None and attention_mask.dim() == 2 and attention_mask.size(1) == T_k:
            key_pad = ~attention_mask.to(torch.bool)
            pad_mask = key_pad[:, None, None, :].expand(B, self.n_head, T_q, T_k)
            add_mask = add_mask.masked_fill(pad_mask, NEG_INF)

        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=add_mask,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=False,
        )

    def _attn_flex(self, q, k, v, B, doc_ids):
        """flex_attention with compiled block mask (PyTorch 2.5+)."""
        try:
            from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        except ImportError:
            raise RuntimeError(
                "attention_mode='flex' requires PyTorch >= 2.5. "
                "Upgrade PyTorch or use 'causal' / 'sdpa_mask'."
            )

        T = q.size(-2)

        if doc_ids is None:
            def mask_fn(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
        else:
            _doc_ids = doc_ids
            def mask_fn(b, h, q_idx, kv_idx):
                causal = q_idx >= kv_idx
                same_doc = _doc_ids[b, q_idx] == _doc_ids[b, kv_idx]
                return causal & same_doc

        block_mask = create_block_mask(mask_fn, B=B, H=1, Q_LEN=T, KV_LEN=T, device=q.device)
        return flex_attention(q, k, v, block_mask=block_mask)

    def _attn_manual(self, q, k, v, B, attention_mask, doc_ids):
        """Manual attention fallback for CPU / older GPUs without SDPA."""
        scale = 1.0 / math.sqrt(k.size(-1))
        att = (q @ k.transpose(-2, -1)) * scale

        T_q = q.size(-2)
        T_k = k.size(-2)
        past_kv = T_k - T_q

        i = torch.arange(T_q, device=q.device)
        j = torch.arange(T_k, device=q.device)
        causal_mask = j.unsqueeze(0) > (past_kv + i.unsqueeze(1))
        att = att.masked_fill(causal_mask, float('-inf'))

        if doc_ids is not None and doc_ids.dim() == 2 and doc_ids.size(1) >= T_k:
            doc_ids_q = doc_ids[:, :T_q]
            doc_ids_k = doc_ids[:, :T_k]
            cross_doc_mask = doc_ids_q[:, :, None] != doc_ids_k[:, None, :]
            cross_doc_mask = cross_doc_mask.unsqueeze(1).expand(B, self.n_head, T_q, T_k)
            att = att.masked_fill(cross_doc_mask, float('-inf'))

        if attention_mask is not None:
            if attention_mask.size(1) == T_k:
                key_pad_mask = attention_mask.view(B, 1, 1, T_k) == 0
                att = att.masked_fill(key_pad_mask, float('-inf'))

        att_max = att.max(dim=-1, keepdim=True)[0]
        att_max = torch.clamp(att_max, max=0.0)
        att = att - att_max

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos_sin: [Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        B, T, C = hidden_states.size()

        q = self.q_proj(hidden_states).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        cos, sin = rope_cos_sin
        if rope_cos_sin[0].dtype != q.dtype:
            cos = cos.to(dtype=q.dtype)
            sin = sin.to(dtype=q.dtype)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)

        # Cache the compact (pre-expansion) K/V so GQA keeps its memory win.
        present = (k, v) if use_cache else None

        # Expand grouped K/V up to the query head count (no-op for plain MHA).
        if self.n_rep > 1:
            k = repeat_kv(k, self.n_rep)
            v = repeat_kv(v, self.n_rep)

        if hasattr(F, 'scaled_dot_product_attention'):
            mode = self.config.attention_mode
            has_kv_cache = past_key_value is not None

            if mode == "causal" and not has_kv_cache:
                y = self._attn_causal(q, k, v)
            elif mode == "flex" and not has_kv_cache:
                y = self._attn_flex(q, k, v, B, doc_ids)
            else:
                # sdpa_mask explicitly, or fallback for KV-cache decode
                y = self._attn_sdpa_mask(q, k, v, B, attention_mask, doc_ids)
        else:
            y = self._attn_manual(q, k, v, B, attention_mask, doc_ids)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, present


def _make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    elif name == "relu":
        return nn.ReLU()
    elif name == "swish":
        return nn.SiLU()  # SiLU is the same as Swish
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class GatedMLP(nn.Module):
    """
    Gated feed-forward network (SwiGLU / GeGLU / ReGLU depending on activation).

    Used by Llama, Mistral, Qwen, Gemma, ...  Output is ``c_down(up * act(gate))``.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = config.intermediate_dim()
        bias = config.resolved_mlp_bias
        self.c_up = nn.Linear(config.n_embd, hidden_size, bias=bias)
        self.c_gate = nn.Linear(config.n_embd, hidden_size, bias=bias)
        self.c_down = nn.Linear(hidden_size, config.n_embd, bias=bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act = _make_activation(config.activation)

    def forward(self, x):
        u = self.c_up(x)
        g = self.act(self.c_gate(x))
        x = self.c_down(u * g)
        x = self.dropout(x)
        return x


class VanillaMLP(nn.Module):
    """
    Standard (non-gated) feed-forward network: ``c_proj(act(c_fc(x)))``.

    Used by GPT-2, GPT-NeoX/Pythia, OPT, Falcon, ...
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_size = config.intermediate_dim()
        bias = config.resolved_mlp_bias
        self.c_fc = nn.Linear(config.n_embd, hidden_size, bias=bias)
        self.c_proj = nn.Linear(hidden_size, config.n_embd, bias=bias)
        self.dropout = nn.Dropout(config.dropout)
        self.act = _make_activation(config.activation)

    def forward(self, x):
        x = self.c_proj(self.act(self.c_fc(x)))
        x = self.dropout(x)
        return x


def build_mlp(config: ModelConfig) -> nn.Module:
    """Construct the MLP variant selected by ``config.mlp_type``."""
    if config.mlp_type == "gated":
        return GatedMLP(config)
    elif config.mlp_type == "mlp":
        return VanillaMLP(config)
    raise ValueError(f"Unsupported mlp_type: {config.mlp_type}")


class TransformerBlock(nn.Module):
    """
    A single transformer block with pre-norm architecture.
    """
    def __init__(self, config: ModelConfig, parent_model: "SimpleLLM" = None):
        super().__init__()
        self.config = config
        # Bypass nn.Module.__setattr__ to avoid registering the parent as a
        # child module, which would create a circular reference and infinite
        # recursion in train()/eval()/apply().
        object.__setattr__(self, '_parent_model', parent_model)
        if config.norm == "rmsnorm":
            self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
            self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.mlp = build_mlp(config)

    def _attention_forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos_sin: [Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        doc_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention forward pass for gradient checkpointing."""
        attn_output, _ = self.attn(
            self.ln_1(hidden_states),
            rope_cos_sin=rope_cos_sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            doc_ids=doc_ids,
            use_cache=False,
            past_key_value=None,
        )
        return attn_output

    def _mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP forward pass for gradient checkpointing."""
        return self.mlp(self.ln_2(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos_sin: [Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        doc_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        x = hidden_states
        
        # Use gradient checkpointing if enabled and in training mode
        # Note: gradient checkpointing is incompatible with use_cache=True
        gc_enabled = (
            self._parent_model is not None
            and self._parent_model._gradient_checkpointing
        )
        if (gc_enabled and
            self.training and
            not use_cache and
            past_key_value is None):
            
            # Checkpoint attention
            a = checkpoint(
                self._attention_forward,
                x,
                rope_cos_sin,
                attention_mask,
                position_ids,
                doc_ids,
                use_reentrant=False
            )
            x = x + a
            
            # Checkpoint MLP
            m = checkpoint(
                self._mlp_forward,
                x,
                use_reentrant=False
            )
            x = x + m
            present = None
            
        else:
            # Normal forward pass
            a, present = self.attn(
                self.ln_1(x),
                rope_cos_sin=rope_cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                doc_ids=doc_ids,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
            x = x + a
            m = self.mlp(self.ln_2(x))
            x = x + m
            
        return x, present


class SimpleLLM(nn.Module):
    """
    A simplified but functional GPT implementation for educational purposes.
    
    This model follows the transformer architecture with:
    - Token embeddings
    - Multiple transformer blocks with self-attention
    - Language modeling head for next token prediction
    - RoPE (Rotary Position Embeddings) for positional encoding
    """
    
    def __init__(self, config: ModelConfig, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self._gradient_checkpointing = gradient_checkpointing

        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Output head: tied to the embedding matrix by default (weight tying),
        # or a dedicated projection when ``tie_word_embeddings`` is False
        # (e.g. a faithful Llama-1/2 replica, which does not tie).
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config, self) for _ in range(config.n_layer)])

        if config.norm == "rmsnorm":
            self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd, eps=config.norm_eps)

        head_dim = config.n_embd // config.n_head
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=head_dim,
            base=config.rope_base,
            scaling_type=config.rope_scaling_type,
            scaling_factor=config.rope_scaling_factor,
        )

        for m in self.modules():
            self._init_weights(m)
        self._apply_residual_scaling()

        logger.debug(f"Non-embedding parameters: {self.get_num_params():,}")

        if self._gradient_checkpointing:
            logger.debug("Gradient checkpointing: enabled")
        else:
            logger.debug("Gradient checkpointing: disabled")

    def clear_rope_cache(self):
        """Clear RoPE cache. Useful for memory management or when changing devices."""
        self.rotary_emb.clear_cache()
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all transformer blocks."""
        self._gradient_checkpointing = True
        logger.debug("Gradient checkpointing: enabled")

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for all transformer blocks."""
        self._gradient_checkpointing = False
        logger.debug("Gradient checkpointing: disabled")

    def is_gradient_checkpointing_enabled(self) -> bool:
        """Check if gradient checkpointing is enabled."""
        return self._gradient_checkpointing

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # Exclude token embeddings when counting non-embedding params
            n_params -= self.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initialize weights following GPT-2/GPT-3 initialization scheme.
        Standard initialization for all layers (residual scaling applied separately).
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if isinstance(module, nn.LayerNorm) and hasattr(module, 'bias'):
                torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _apply_residual_scaling(self):
        """
        Apply residual scaling to output projections for training stability.
        
        This is CRITICAL for deep networks. Without this, gradients explode!
        
        For a gated MLP (SwiGLU-style), we scale down c_down because:
        - The gating operation (u * g) increases variance
        - We need to compensate to maintain stable forward pass variance
        
        Scaling factor: 1/sqrt(2*n_layers) as per GPT-2/GPT-3 papers
        """
        scaling_factor = (2 * self.config.n_layer) ** -0.5
        scaled_count = 0
        
        for name, module in self.named_modules():
            # Scale residual projections:
            # - attn.c_proj: attention output projection
            # - mlp.c_down: MLP output projection (especially important for gated MLPs)
            if isinstance(module, nn.Linear) and ('c_proj' in name or 'c_down' in name):
                with torch.no_grad():
                    module.weight.data.normal_(mean=0.0, std=0.02 * scaling_factor)
                scaled_count += 1
        
        logger.debug(f"Applied residual scaling (factor={scaling_factor:.4f}) to {scaled_count} layers")
        if scaled_count == 0:
            logger.warning("No layers were scaled by residual scaling. Check your model architecture.")

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        doc_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token ids of shape (batch_size, sequence_length)
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position indices for each token (continuous 0...T-1 for packed seqs)
            doc_ids: Document IDs for each token (for block-diagonal attention in packed seqs)
            past_key_values: Cached key-value pairs for efficient generation
            use_cache: Whether to return key-value pairs for caching
            labels: Labels for computing language modeling loss
            return_dict: Whether to return a dict or tuple
            
        Returns:
            If labels are provided, returns loss and logits.
            Otherwise returns logits and optionally cached key-value pairs.
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # Build (or extend) absolute position_ids
        if position_ids is None:
            past_length = 0
            if past_key_values is not None and len(past_key_values) > 0:
                # past_key_values[i][0] is k : (B, H, T_past, D)
                past_length = past_key_values[0][0].size(-2)
            position_ids = torch.arange(
                past_length, past_length + seq_length, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)  # (B, T)

        # Token embeddings
        inputs_embeds = self.wte(input_ids)
        hidden_states = self.drop(inputs_embeds)

        # Precompute RoPE tables once per forward for the *current chunk* indices
        # Use a more compile-friendly approach that avoids .item() calls
        # Pre-compute RoPE for the maximum possible sequence length to avoid dynamic shapes
        past_length = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]
        needed_len = seq_length + past_length
        # Add some buffer to ensure we have enough
        needed_len = max(needed_len, self.config.n_positions)
        rope_cos_sin = self.rotary_emb(needed_len, device=device)  # (cos,sin)

        # Transformer blocks
        presents = () if use_cache else None
        for i, block in enumerate(self.h):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = block(
                hidden_states,
                rope_cos_sin=rope_cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                doc_ids=doc_ids,
                use_cache=use_cache,
                past_key_value=past_key_value,
            )
            if use_cache:
                presents = presents + (present,)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Output projection.  When tied, reuse the embedding weights directly
        # (equivalent to a bias-free Linear sharing wte.weight); otherwise use
        # the dedicated head.
        if self.config.tie_word_embeddings:
            lm_logits = F.linear(hidden_states, self.wte.weight)
        else:
            lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Use ignore_index=-100 to exclude padding tokens from loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            with torch.amp.autocast(device_type="cuda", enabled=False):
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32),
                                shift_labels.view(-1))
            

        if return_dict:
            return CausalLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=presents,
            )
        else:
            out = (lm_logits,)
            if presents is not None:
                out = out + (presents,)
            if loss is not None:
                out = (loss,) + out
            return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.LongTensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token ids
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token id
            eos_token_id: End of sequence token id
            
        Returns:
            Generated token ids including the input
        """
        self.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        generated = input_ids
        past_key_values = None
        # Build initial attention mask from pad_token_id if provided (1 = valid, 0 = pad)
        if pad_token_id is not None:
            attention_mask = (generated != pad_token_id).to(dtype=torch.long)
        else:
            attention_mask = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_inputs = generated
            else:
                model_inputs = generated[:, -1:]
                # As we add a new token, grow the attention mask with ones
                if attention_mask is not None:
                    ones = torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
                    attention_mask = torch.cat([attention_mask, ones], dim=1)

            outputs = self.forward(
                input_ids=model_inputs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs["logits"][:, -1, :]  # (B,V)
            past_key_values = outputs["past_key_values"]

            # Repetition penalty (HF-style)
            if repetition_penalty != 1.0:
                prompt_len = input_ids.shape[1]
                for b in range(generated.size(0)):
                    # Only apply penalty to generated tokens, not prompt
                    generated_part = generated[b, prompt_len:]
                    if generated_part.numel() > 0:
                        seen = generated_part.unique()
                        logits_b = logits[b, seen]
                        # if logit < 0 multiply by penalty else divide
                        penalized = torch.where(
                            logits_b < 0,
                            logits_b * repetition_penalty,
                            logits_b / repetition_penalty,
                        )
                        logits[b, seen] = penalized

            if temperature != 1.0:
                logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k)[0][..., -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative = torch.cumsum(probs, dim=-1)

                sorted_rm = cumulative > top_p
                sorted_rm[..., 1:] = sorted_rm[..., :-1].clone()
                sorted_rm[..., 0] = 0

                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_rm)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
