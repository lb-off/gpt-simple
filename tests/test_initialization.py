"""
Model initialization and numerical-stability tests.

Verifies that residual output projections are down-scaled by the
GPT-2/GPT-3 factor (1/sqrt(2*n_layer)) and that forward/backward passes
stay finite and well-conditioned, including at the depth used by the
example pretraining configs.
"""

import torch
import torch.nn as nn

from gpt_simple.config import ModelConfig
from gpt_simple.model import SimpleLLM


def _model(**overrides):
    base = dict(vocab_size=1000, n_embd=256, n_layer=6, n_head=4, n_positions=128)
    base.update(overrides)
    torch.manual_seed(0)
    return SimpleLLM(ModelConfig(**base), gradient_checkpointing=False)


def test_residual_scaling_applied():
    """Residual projections (c_proj, c_down) are scaled by ~1/sqrt(2*n_layer)."""
    model = _model()
    expected = (2 * model.config.n_layer) ** -0.5

    residual_norms, other_norms = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            norm = module.weight.norm().item()
            if "c_proj" in name or "c_down" in name:
                residual_norms.append(norm)
            else:
                other_norms.append(norm)

    assert residual_norms, "expected residual projections to scale"
    ratio = (sum(residual_norms) / len(residual_norms)) / (
        sum(other_norms) / len(other_norms)
    )
    # Stochastic init: allow a 2x band around the expected ratio.
    assert 0.5 * expected < ratio < 2.0 * expected, (
        f"residual/other norm ratio {ratio:.4f} not near expected {expected:.4f}"
    )


def test_forward_pass_is_stable():
    model = _model().eval()
    ids = torch.randint(0, model.config.vocab_size, (2, 32))
    with torch.no_grad():
        logits = model(ids, return_dict=True).logits

    assert torch.isfinite(logits).all(), "forward produced NaN/Inf"
    assert logits.abs().max().item() < 50, "logits exploding"
    assert 0.01 < logits.std().item() < 100, "logit variance unhealthy"


def test_backward_pass_is_stable():
    model = _model().train()
    ids = torch.randint(0, model.config.vocab_size, (2, 32))
    labels = torch.randint(0, model.config.vocab_size, (2, 32))

    loss = model(ids, labels=labels, return_dict=True).loss
    loss.backward()

    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert grad_norms, "no gradients were produced"
    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    ), "backward produced NaN/Inf gradients"
    assert max(grad_norms) < 100, "gradients exploding"
    assert 0.0001 < (sum(grad_norms) / len(grad_norms)) < 100, "grad magnitude unhealthy"


def test_deep_model_is_stable():
    """Scaling keeps a deep stack finite at the example configs' depth."""
    model = _model(n_embd=256, n_layer=34, n_head=8).eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 16))
    with torch.no_grad():
        logits = model(ids, return_dict=True).logits

    assert torch.isfinite(logits).all(), "deep model produced NaN/Inf"
    assert logits.abs().max().item() < 50, "deep model logits exploding"
