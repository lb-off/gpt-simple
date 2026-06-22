#!/usr/bin/env python3
"""
Tests for the architecture knobs added to express more model families:
GQA/MQA, vanilla (non-gated) MLP, explicit FFN width, per-projection bias,
and an untied LM head.

These check shapes, parameter counts, forward/backward sanity, and that the
analytic counter in ``validate._count_params`` stays in sync with the real
module.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).parent.parent))

from gpt_simple.config import ModelConfig
from gpt_simple.errors import ConfigError
from gpt_simple.model import GatedMLP, SimpleLLM, VanillaMLP
from gpt_simple.validate import _count_params


def _tiny(**overrides):
    base = dict(vocab_size=256, n_embd=64, n_layer=2, n_head=4, n_positions=32)
    base.update(overrides)
    return ModelConfig(**base)


def _forward(config, B=2, T=8):
    torch.manual_seed(0)
    model = SimpleLLM(config, gradient_checkpointing=False)
    ids = torch.randint(0, config.vocab_size, (B, T))
    out = model(ids, labels=ids, return_dict=True)
    return model, out


# ---------------------------------------------------------------------------
# Defaults / backward compatibility
# ---------------------------------------------------------------------------

def test_defaults_are_llama_style():
    c = _tiny()
    assert c.kv_heads == c.n_head            # MHA
    assert c.mlp_type == "gated"
    assert c.tie_word_embeddings is True
    assert c.resolved_qkv_bias is False
    model, out = _forward(c)
    assert out.logits.shape == (2, 8, c.vocab_size)
    assert torch.isfinite(out.loss)
    # Tied head: no separate lm_head module.
    assert not hasattr(model, "lm_head")


# ---------------------------------------------------------------------------
# Grouped-/multi-query attention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_gqa_forward_and_shapes(n_kv_head):
    c = _tiny(n_kv_head=n_kv_head)
    model, out = _forward(c)
    assert torch.isfinite(out.loss)
    attn = model.h[0].attn
    head_dim = c.n_embd // c.n_head
    assert attn.k_proj.weight.shape == (n_kv_head * head_dim, c.n_embd)
    assert attn.v_proj.weight.shape == (n_kv_head * head_dim, c.n_embd)
    assert attn.q_proj.weight.shape == (c.n_embd, c.n_embd)


def test_gqa_reduces_parameters():
    mha = _count_params(_tiny(), 256)["total"]
    mqa = _count_params(_tiny(n_kv_head=1), 256)["total"]
    assert mqa < mha


def test_gqa_requires_divisor():
    with pytest.raises(ConfigError):
        _tiny(n_kv_head=3)  # 4 % 3 != 0


def test_gqa_kv_cache_generate():
    c = _tiny(n_kv_head=2)
    torch.manual_seed(0)
    model = SimpleLLM(c)
    ids = torch.randint(0, c.vocab_size, (1, 5))
    gen = model.generate(ids, max_new_tokens=4, do_sample=False)
    assert gen.shape == (1, 9)


# ---------------------------------------------------------------------------
# Vanilla (non-gated) MLP
# ---------------------------------------------------------------------------

def test_vanilla_mlp_selected_and_runs():
    c = _tiny(mlp_type="mlp")
    model, out = _forward(c)
    assert isinstance(model.h[0].mlp, VanillaMLP)
    assert torch.isfinite(out.loss)


def test_gated_mlp_default():
    model, _ = _forward(_tiny())
    assert isinstance(model.h[0].mlp, GatedMLP)


def test_vanilla_default_width_is_4x():
    c = _tiny(mlp_type="mlp")
    assert c.intermediate_dim() == 4 * c.n_embd


def test_explicit_intermediate_size():
    c = _tiny(intermediate_size=512)
    assert c.intermediate_dim() == 512
    model = SimpleLLM(c)
    assert model.h[0].mlp.c_up.weight.shape == (512, c.n_embd)


# ---------------------------------------------------------------------------
# Bias overrides
# ---------------------------------------------------------------------------

def test_qkv_bias_only_qwen_style():
    c = _tiny(use_bias=False, qkv_bias=True)
    model = SimpleLLM(c)
    attn = model.h[0].attn
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.c_proj.bias is None          # output proj stays bias-free
    assert model.h[0].mlp.c_up.bias is None  # MLP stays bias-free


def test_global_use_bias_propagates():
    c = _tiny(use_bias=True)
    model = SimpleLLM(c)
    assert model.h[0].attn.q_proj.bias is not None
    assert model.h[0].attn.c_proj.bias is not None
    assert model.h[0].mlp.c_up.bias is not None


# ---------------------------------------------------------------------------
# Untied head
# ---------------------------------------------------------------------------

def test_untied_head_allocates_separate_matrix():
    c = _tiny(tie_word_embeddings=False)
    model, out = _forward(c)
    assert hasattr(model, "lm_head")
    assert model.lm_head.weight.data_ptr() != model.wte.weight.data_ptr()
    assert torch.isfinite(out.loss)


def test_untied_head_increases_param_count():
    tied = _count_params(_tiny(), 256)["total"]
    untied = _count_params(_tiny(tie_word_embeddings=False), 256)["total"]
    assert untied == tied + 256 * 64


# ---------------------------------------------------------------------------
# Analytic counter matches the real module across the new knobs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    {},
    {"n_kv_head": 1},
    {"n_kv_head": 2},
    {"mlp_type": "mlp"},
    {"mlp_type": "mlp", "use_bias": True},
    {"intermediate_size": 200},
    {"tie_word_embeddings": False},
    {"qkv_bias": True},
    {"use_bias": True},
])
def test_analytic_param_count_matches_module(overrides):
    c = _tiny(**overrides)
    model = SimpleLLM(c)
    real = sum(p.numel() for p in model.parameters())
    # The module shares wte with the tied head (one matrix); the analytic
    # counter likewise counts the embedding once when tied.
    analytic = _count_params(c, c.vocab_size)["total"]
    assert analytic == real, f"{overrides}: analytic {analytic} != real {real}"
