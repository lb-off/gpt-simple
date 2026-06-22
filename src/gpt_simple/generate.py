"""
Inference utilities for gpt_simple.

Two entry points:

* :func:`load_for_inference` — build a :class:`SimpleLLM` from a checkpoint
  directory (or a run's ``output_dir``; the latest checkpoint is picked
  automatically), load weights, attach the tokenizer.
* :func:`generate` — run prompts through a loaded model and return the
  decoded completions.

The on-disk layout expected here is the canonical one written by
:class:`gpt_simple._checkpoint.CheckpointManager`:

    <run>/
    |-- tokenizer/
    |-- checkpoints/
        |-- checkpoint-<step>/
            |-- model/
            |   |-- config.json          # ModelConfig as a flat dict
            |   |-- pytorch_model.bin    # plain torch.save(state_dict)
            |-- trainer_state.json

Passing either ``<run>`` or ``<run>/checkpoints/checkpoint-<step>`` to
``load_for_inference`` works; the function disambiguates by looking for
``model/config.json`` (checkpoint) vs ``checkpoints/`` (run).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

import torch

from gpt_simple._checkpoint import CheckpointManager
from gpt_simple.config import ModelConfig
from gpt_simple.errors import CheckpointError
from gpt_simple.model import SimpleLLM
from gpt_simple.tokenizer import SimpleLLMTokenizer

logger = logging.getLogger("gpt_simple")


_DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "fp32": torch.float32, "float32": torch.float32, "f32": torch.float32,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
}


def _parse_dtype(dtype: Union[None, str, torch.dtype]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower().strip()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        raise ValueError(
            f"unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)} "
            "or a torch.dtype"
        )
    raise TypeError(f"dtype must be str | torch.dtype | None, got {type(dtype)}")


# ---------------------------------------------------------------------------
# Checkpoint + tokenizer discovery
# ---------------------------------------------------------------------------


def _expand(path: Path) -> Path:
    """Expand ``~`` and ``$VAR`` / ``${VAR}`` in a path.

    Paths embedded in a JSONL (``batch-generate``) or a config are never
    seen by the shell, so ``$WORK`` / ``$SCRATCH`` / ``~`` would otherwise be
    taken literally. Expanding here makes them behave like paths typed on the
    command line. A path with no variables passes through unchanged.
    """
    return Path(os.path.expanduser(os.path.expandvars(str(path))))


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "model" / "config.json").is_file()


def _is_run_dir(path: Path) -> bool:
    return (path / "checkpoints").is_dir()


def _resolve_checkpoint(path: Path) -> Path:
    """Return a specific checkpoint dir from either a checkpoint or a run dir."""
    path = _expand(path)
    if _is_checkpoint_dir(path):
        return path
    if _is_run_dir(path):
        mgr = CheckpointManager(path)
        ckpts = mgr.list_checkpoints()
        if not ckpts:
            raise CheckpointError(
                f"{path} looks like a run directory but contains no "
                "completed checkpoints under checkpoints/."
            )
        latest = ckpts[-1][2]
        logger.info("Auto-selected latest checkpoint: %s", latest)
        return latest
    raise CheckpointError(
        f"{path} is neither a checkpoint directory (missing model/config.json) "
        "nor a run directory (missing checkpoints/)."
    )


def _resolve_tokenizer_dir(checkpoint: Path, override: Optional[Path]) -> Path:
    """Find the tokenizer directory associated with a checkpoint.

    Search order:

    1. Explicit ``override`` argument (any directory).
    2. ``<checkpoint>/tokenizer`` (some checkpoints bundle their own copy).
    3. ``<checkpoint>/../../tokenizer`` (canonical run-root location written
       by :meth:`CheckpointManager.save_tokenizer`).
    """
    if override is not None:
        p = _expand(Path(override))
        if not p.is_dir():
            raise CheckpointError(f"tokenizer override path does not exist: {p}")
        return p
    candidate = checkpoint / "tokenizer"
    if candidate.is_dir():
        return candidate
    run_root = checkpoint.parent.parent
    candidate = run_root / "tokenizer"
    if candidate.is_dir():
        return candidate
    raise CheckpointError(
        f"could not locate tokenizer for {checkpoint}. Tried "
        f"{checkpoint / 'tokenizer'} and {run_root / 'tokenizer'}. "
        "Pass tokenizer_path= explicitly."
    )


# ---------------------------------------------------------------------------
# Public: validate_checkpoint
# ---------------------------------------------------------------------------


class CheckpointInfo(NamedTuple):
    """Resolved, validated locations for a checkpoint (no weights loaded)."""
    checkpoint_dir: Path
    tokenizer_dir: Path
    model_config: ModelConfig


def validate_checkpoint(
    path: Union[str, Path],
    *,
    tokenizer_path: Union[str, Path, None] = None,
    load_tokenizer: bool = True,
) -> CheckpointInfo:
    """Pre-flight check that ``path`` is loadable for inference.

    Does everything :func:`load_for_inference` does to *locate, parse, and
    sanity-check* a checkpoint — resolve a checkpoint vs. run directory,
    confirm the weights file exists, parse ``model/config.json`` into a
    :class:`ModelConfig`, resolve the tokenizer directory and (when
    ``load_tokenizer``) actually instantiate it — but **never loads the model
    weights and never touches CUDA**.  That makes it safe to run on a cluster
    login node as a submission gate (e.g. ``batch-generate --dry-run``)
    before spending a GPU allocation.

    Returns a :class:`CheckpointInfo` with the *resolved* checkpoint and
    tokenizer directories (callers can use these to deduplicate references
    that point at the same model two different ways).  Raises
    :class:`CheckpointError` describing the first problem found otherwise.
    """
    ckpt_dir = _resolve_checkpoint(Path(path))

    weights = ckpt_dir / "model" / "pytorch_model.bin"
    if not weights.is_file():
        raise CheckpointError(f"model weights not found: {weights}")

    config_path = ckpt_dir / "model" / "config.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    try:
        model_config = ModelConfig(**cfg_dict)
    except TypeError as exc:
        raise CheckpointError(
            f"model/config.json in {ckpt_dir} is not a valid ModelConfig: {exc}"
        ) from exc

    # Resolves and existence-checks the tokenizer dir; raises if missing.
    tok_dir = _resolve_tokenizer_dir(
        ckpt_dir, Path(tokenizer_path) if tokenizer_path else None
    )

    # Actually instantiate the tokenizer — a directory that exists but is
    # incomplete/corrupt would otherwise pass and only fail once the job
    # dequeues. GPU-free, and cheap for a local tokenizer dir.
    if load_tokenizer:
        try:
            tokenizer = SimpleLLMTokenizer(str(tok_dir))
        except Exception as exc:
            raise CheckpointError(
                f"tokenizer at {tok_dir} could not be loaded: {exc}"
            ) from exc
        if (
            model_config.vocab_size is not None
            and tokenizer.vocab_size != model_config.vocab_size
        ):
            logger.warning(
                "tokenizer vocab=%d but ModelConfig.vocab_size=%d for %s — "
                "usually fine (weights are padded to a multiple of N) but "
                "check before using logits past the tokenizer range.",
                tokenizer.vocab_size, model_config.vocab_size, ckpt_dir,
            )

    return CheckpointInfo(
        checkpoint_dir=ckpt_dir, tokenizer_dir=tok_dir, model_config=model_config
    )


# ---------------------------------------------------------------------------
# Public: load_for_inference
# ---------------------------------------------------------------------------


def load_for_inference(
    path: Union[str, Path],
    *,
    device: Union[str, torch.device, None] = None,
    dtype: Union[str, torch.dtype, None] = None,
    tokenizer_path: Union[str, Path, None] = None,
) -> Tuple[SimpleLLM, SimpleLLMTokenizer, ModelConfig]:
    """Load a checkpoint for inference.

    Parameters
    ----------
    path : str or Path
        Either a specific checkpoint directory (``…/checkpoint-N``) or a
        run's ``output_dir``. In the latter case the latest checkpoint is
        picked automatically (same rule as ``training.resume='auto'``).
    device : str | torch.device | None
        Where to place the model. ``None`` defaults to ``"cuda"`` when
        available, else ``"cpu"``.
    dtype : str | torch.dtype | None
        Optional cast applied after the weights are loaded.  Accepts
        ``"fp32"``, ``"fp16"`` / ``"half"``, ``"bf16"`` /
        ``"bfloat16"``, or a :class:`torch.dtype`.  ``None`` keeps the
        on-disk dtype.
    tokenizer_path : str or Path or None
        Override for the tokenizer location.  When omitted, search order
        is documented in :func:`_resolve_tokenizer_dir`.

    Returns
    -------
    model : SimpleLLM
        In eval mode, on the requested device.
    tokenizer : SimpleLLMTokenizer
        Ready for ``.encode`` / ``.decode``.
    model_config : ModelConfig
        The config recovered from the checkpoint.
    """
    ckpt_dir = _resolve_checkpoint(Path(path))
    config_path = ckpt_dir / "model" / "config.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    try:
        model_config = ModelConfig(**cfg_dict)
    except TypeError as exc:
        # A field present in cfg_dict isn't on the current ModelConfig.
        # Likely a stale or hand-written file; refuse loudly rather than
        # silently dropping fields.
        raise CheckpointError(
            f"model/config.json in {ckpt_dir} is not a valid ModelConfig: {exc}"
        ) from exc

    logger.debug("ModelConfig from %s: %s", config_path, asdict(model_config))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = _parse_dtype(dtype)

    model = SimpleLLM(model_config, gradient_checkpointing=False)
    CheckpointManager.load_model_state(model, ckpt_dir)

    if torch_dtype is not None:
        model = model.to(dtype=torch_dtype)
    model = model.to(device=device)
    model.eval()

    tok_dir = _resolve_tokenizer_dir(ckpt_dir, Path(tokenizer_path) if tokenizer_path else None)
    tokenizer = SimpleLLMTokenizer(str(tok_dir))

    if tokenizer.vocab_size != model_config.vocab_size:
        logger.warning(
            "tokenizer vocab=%d but ModelConfig.vocab_size=%d — usually fine "
            "(weights are padded to a multiple of N) but check before "
            "using logits past the tokenizer range.",
            tokenizer.vocab_size, model_config.vocab_size,
        )

    return model, tokenizer, model_config


# ---------------------------------------------------------------------------
# Public: generate
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate(
    model: SimpleLLM,
    tokenizer: SimpleLLMTokenizer,
    prompts: Union[str, Iterable[str]],
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
    return_full_text: bool = False,
) -> List[str]:
    """Run a list of prompts through ``model`` and return the decoded outputs.

    Prompts are processed one at a time — ragged batched generation with
    KV-cache is fiddly and per-prompt is fast enough for typical batch
    inference jobs (the model itself is on GPU; the loop overhead is
    negligible compared to the forward pass).

    Parameters
    ----------
    return_full_text : bool
        If ``True`` the returned strings include the prompt; otherwise
        only the newly generated text is returned.

    Returns
    -------
    list of str
        One completion per input prompt, in order.
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = list(prompts)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)

    outputs: List[str] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            repetition_penalty=repetition_penalty,
        )
        if return_full_text:
            text = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
        else:
            new_ids = out[0, ids.shape[1]:].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=False)
        outputs.append(text)

    return outputs


__all__ = ["load_for_inference", "generate", "validate_checkpoint", "CheckpointInfo"]
