"""
Configuration dataclasses for gpt_simple.

Four independent configs + a Config compositor that supports YAML/JSON loading:
  - ModelConfig:     architecture (defines checkpoint compatibility)
  - DataConfig:      data pipeline (paths, format, packing)
  - OptimizerConfig: optimizer & LR schedule
  - TrainingConfig:  training loop (batch, checkpointing, wandb, runtime)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from gpt_simple.errors import ConfigError

logger = logging.getLogger("gpt_simple")

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Pure architecture — everything that defines the weight/checkpoint format.

    Defaults produce a small 125M-class model suitable for quick experiments.

    The defaults describe a modern Llama-style decoder (RMSNorm + RoPE +
    SwiGLU + tied head, no biases).  The extra knobs below let the same code
    express other dense decoder-only families:

      - ``n_kv_head``           grouped-/multi-query attention (Llama-3, Mistral, Qwen2)
      - ``mlp_type``            vanilla (non-gated) FFN (GPT-2, NeoX, OPT, Falcon)
      - ``intermediate_size``   explicit FFN width (Gemma, exact Mistral/Qwen widths)
      - ``tie_word_embeddings`` untied output head (real Llama-1/2, GPT-2)
      - ``qkv_bias`` / ``attn_out_bias`` / ``mlp_bias``  per-projection bias (Qwen2)
    """

    vocab_size: Optional[int] = None  # inferred from tokenizer, padded to 128
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    # Number of key/value heads.  None -> n_head (standard multi-head
    # attention).  < n_head enables grouped-query attention (must divide
    # n_head); 1 is multi-query attention.
    n_kv_head: Optional[int] = None
    n_positions: int = 2048

    dropout: float = 0.0

    # -- bias -----------------------------------------------------------------
    # ``use_bias`` is the global default for every linear layer.  The three
    # per-projection overrides take precedence when not None, so e.g. Qwen2
    # (bias on Q/K/V only) is ``use_bias=False, qkv_bias=True``.
    use_bias: bool = False
    qkv_bias: Optional[bool] = None        # Q/K/V projections
    attn_out_bias: Optional[bool] = None   # attention output projection
    mlp_bias: Optional[bool] = None        # MLP linears

    activation: str = "swish"          # "swish", "gelu", "relu"

    # -- feed-forward ---------------------------------------------------------
    # "gated" -> SwiGLU/GeGLU/ReGLU (gate * up); "mlp" -> vanilla act(fc) proj.
    mlp_type: Literal["gated", "mlp"] = "gated"
    # Explicit FFN inner width.  None derives it: gated -> round_256(8*n_embd/3)
    # (the Llama sizing); vanilla -> 4*n_embd (the GPT sizing).
    intermediate_size: Optional[int] = None

    norm: str = "rmsnorm"              # "rmsnorm", "layernorm"
    norm_eps: float = 1e-5

    # Tie the LM head to the token-embedding matrix.  False allocates a
    # separate output projection (real Llama-1/2, GPT-2 both tie; Llama ties
    # nothing — set False for a faithful replica).
    tie_word_embeddings: bool = True

    rope_base: float = 10000.0
    rope_scaling_type: Optional[Literal["linear", "ntk"]] = None
    rope_scaling_factor: float = 1.0

    attention_mode: Literal["causal", "sdpa_mask", "flex"] = "causal"

    def __post_init__(self):
        if self.n_embd % self.n_head != 0:
            raise ConfigError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head is not None:
            if self.n_kv_head < 1:
                raise ConfigError(f"n_kv_head must be >= 1, got {self.n_kv_head}")
            if self.n_head % self.n_kv_head != 0:
                raise ConfigError(
                    f"n_head ({self.n_head}) must be divisible by "
                    f"n_kv_head ({self.n_kv_head})"
                )
        if self.activation not in ("gelu", "relu", "swish"):
            raise ConfigError(f"Unsupported activation: {self.activation}")
        if self.mlp_type not in ("gated", "mlp"):
            raise ConfigError(f"Unsupported mlp_type: {self.mlp_type}")
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ConfigError(
                f"intermediate_size must be >= 1 or None, got {self.intermediate_size}"
            )
        if self.attention_mode not in ("causal", "sdpa_mask", "flex"):
            raise ConfigError(f"Unsupported attention_mode: {self.attention_mode}")
        if self.rope_scaling_type is not None and self.rope_scaling_factor < 1.0:
            raise ConfigError("rope_scaling_factor should be >= 1.0")

    # -- derived architecture helpers ----------------------------------------

    @property
    def kv_heads(self) -> int:
        """Effective number of key/value heads (n_head for plain MHA)."""
        return self.n_head if self.n_kv_head is None else self.n_kv_head

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def resolved_qkv_bias(self) -> bool:
        return self.use_bias if self.qkv_bias is None else self.qkv_bias

    @property
    def resolved_attn_out_bias(self) -> bool:
        return self.use_bias if self.attn_out_bias is None else self.attn_out_bias

    @property
    def resolved_mlp_bias(self) -> bool:
        return self.use_bias if self.mlp_bias is None else self.mlp_bias

    def intermediate_dim(self) -> int:
        """Resolve the FFN inner width from config + mlp_type."""
        if self.intermediate_size is not None:
            return self.intermediate_size
        if self.mlp_type == "gated":
            h = (8 * self.n_embd) // 3
            return ((h + 255) // 256) * 256
        return 4 * self.n_embd


# Keys that define weight/checkpoint compatibility.  Any change to one of
# these changes the parameter shapes (or the loss-bearing graph), so the
# trainer hashes exactly these to detect incompatible resumes.  Centralised
# here so the trainer, validator, and tests cannot drift apart.
ARCH_KEYS = (
    "vocab_size", "n_embd", "n_layer", "n_head", "n_kv_head", "n_positions",
    "activation", "mlp_type", "intermediate_size", "norm",
    "tie_word_embeddings",
    "use_bias", "qkv_bias", "attn_out_bias", "mlp_bias",
)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@dataclass
class CurriculumPhase:
    """A single curriculum phase: train on a specific bucket mix for a token budget."""

    duration_tokens: int
    mix: Dict[str, float]

    def __post_init__(self):
        if self.duration_tokens <= 0:
            raise ConfigError(f"duration_tokens must be positive, got {self.duration_tokens}")
        if not self.mix:
            raise ConfigError("mix must contain at least one bucket")
        if any(v < 0 for v in self.mix.values()):
            raise ConfigError("mix weights must be non-negative")
        total = sum(self.mix.values())
        if total <= 0:
            raise ConfigError("mix weights must sum to a positive value")
        self.mix = {k: v / total for k, v in self.mix.items()}


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """What data to feed the model and how to load it."""

    path: str = ""                                          # root data directory (required)
    tokenizer: str = "gpt2"                                 # name or local path
    format: Literal["pretokenized", "jsonl"] = "pretokenized"
    max_length: int = 2048
    overlap_size: int = 256                                 # overlap between windows of long docs
    packing: bool = True
    num_workers: int = 4
    curriculum: Optional[List[CurriculumPhase]] = None      # None = uniform sampling

    # Curriculum validation + runtime policy.  Describes *intent*: turn these
    # on when the experiment is deliberately designed around an exhausted
    # bucket or a curriculum whose total token budget does not match what the
    # training loop will consume (e.g. a cooldown phase outside the curriculum).
    # Default off, so accidental misconfigurations block the run.
    #
    # ``allow_bucket_exhaustion`` governs BOTH stages:
    #   * validation: a predicted shortfall is a blocking error when False,
    #     a warning when True (see validate.py).
    #   * runtime: when a bucket actually runs dry mid-run, True lets the
    #     loader drop it and renormalise the mix; False instead halts with a
    #     checkpoint (run status "halted") so the unintended mix change is
    #     surfaced rather than applied silently.  Resume with the flag set to
    #     True to continue with the renormalised mix (see _train_loop.py).
    allow_bucket_exhaustion: bool = False
    allow_budget_mismatch: bool = False

    def __post_init__(self):
        if self.format not in ("pretokenized", "jsonl"):
            raise ConfigError(f"Unsupported data format: {self.format}")
        if self.curriculum is not None and self.format != "pretokenized":
            raise ConfigError("curriculum is only supported with format='pretokenized'")
        if self.overlap_size > self.max_length // 2:
            raise ConfigError(
                f"overlap_size ({self.overlap_size}) must be at most half of "
                f"max_length ({self.max_length})"
            )


# ---------------------------------------------------------------------------
# Optimizer & LR schedule
# ---------------------------------------------------------------------------

@dataclass
class OptimizerConfig:
    """Optimizer hyperparameters and learning-rate schedule.

    ``decay_steps`` controls the length of the cosine decay phase.  When
    *None* (the default) it is set to ``max_steps - warmup_steps`` so that
    the cosine finishes exactly at the end of training.  Set it explicitly
    to decouple the schedule from the training length (e.g. for a min-LR
    cooldown phase).
    """

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    warmup_steps: int = 100
    decay_steps: Optional[int] = None   # None -> max_steps - warmup_steps
    min_lr_ratio: float = 0.1

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ConfigError(f"warmup_steps must be non-negative, got {self.warmup_steps}")


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """How the training run is executed — batch sizes, logging, output.

    The resume / checkpoint-retention knobs (``resume``, ``keep_last_k``,
    ``keep_milestone_every``) make stop/resume behave uniformly across
    backends.  Default is ``resume="auto"``: rerunning the same command
    picks up from the latest checkpoint, or starts fresh if none exists.
    """

    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000

    gradient_checkpointing: bool = True
    compile: bool = True
    seed: int = 42

    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    max_eval_batches: Optional[int] = None

    output_dir: str = "./outputs"

    # -- resume / checkpoint retention --------------------------------------
    # "auto":    resume from latest checkpoint if any, else from scratch
    # "scratch": train from scratch; errors if checkpoints already exist
    # "<path>":  resume from this specific checkpoint directory
    resume: str = "auto"
    keep_last_k: Optional[int] = 3
    keep_milestone_every: Optional[int] = None

    # -- graceful shutdown / walltime budget --------------------------------
    # ``max_runtime_seconds`` is an explicit wall-clock budget after which
    # the trainer saves a shutdown checkpoint and exits cleanly.  Leave as
    # ``None`` to auto-detect from the SLURM_JOB_END_TIME environment
    # variable (set by SLURM) or to disable the watchdog entirely.
    # ``walltime_reserve_seconds`` is the buffer before the deadline used
    # to give the loop enough time to save before the orchestrator kills
    # the job.  Increase for very large checkpoints / slow disks.
    max_runtime_seconds: Optional[int] = None
    walltime_reserve_seconds: int = 300

    # -- mixed precision ----------------------------------------------------
    # ``None`` auto-detects: ``bf16`` on Ampere+ (A100, RTX 30xx, H100, …);
    # ``fp16`` on older CUDA GPUs (V100, T4, …); ``no`` on CPU.  Set
    # explicitly when you want to force a backend.
    mixed_precision: Optional[str] = None  # "bf16" | "fp16" | "no" | None

    wandb_project: Optional[str] = None      # None = wandb disabled
    wandb_run_name: Optional[str] = None     # auto-generated if None

    def __post_init__(self):
        if self.per_device_batch_size < 1:
            raise ConfigError(f"per_device_batch_size must be >= 1, got {self.per_device_batch_size}")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        if self.max_steps < 1:
            raise ConfigError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.logging_steps < 1:
            raise ConfigError(f"logging_steps must be >= 1, got {self.logging_steps}")
        if self.logging_steps > self.max_steps:
            logger.warning(
                f"logging_steps ({self.logging_steps}) > max_steps ({self.max_steps}): "
                "no training metrics will be logged"
            )
        if self.eval_steps > self.max_steps:
            logger.warning(
                f"eval_steps ({self.eval_steps}) > max_steps ({self.max_steps}): "
                "no evaluation will run during training"
            )
        if self.save_steps > self.max_steps:
            logger.warning(
                f"save_steps ({self.save_steps}) > max_steps ({self.max_steps}): "
                "no intermediate checkpoints will be saved"
            )
        if self.keep_last_k is not None and self.keep_last_k < 1:
            raise ConfigError(
                f"keep_last_k must be >= 1 or None, got {self.keep_last_k}"
            )
        if self.keep_milestone_every is not None and self.keep_milestone_every < 1:
            raise ConfigError(
                f"keep_milestone_every must be >= 1 or None, got "
                f"{self.keep_milestone_every}"
            )
        if not isinstance(self.resume, str) or not self.resume:
            raise ConfigError(
                "resume must be 'auto', 'scratch', or a checkpoint path"
            )
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ConfigError(
                f"max_runtime_seconds must be > 0 or None, got "
                f"{self.max_runtime_seconds}"
            )
        if self.walltime_reserve_seconds < 0:
            raise ConfigError(
                f"walltime_reserve_seconds must be >= 0, got "
                f"{self.walltime_reserve_seconds}"
            )
        if self.mixed_precision is not None and self.mixed_precision not in {
            "bf16",
            "fp16",
            "no",
        }:
            raise ConfigError(
                f"mixed_precision must be 'bf16', 'fp16', 'no', or None, "
                f"got {self.mixed_precision!r}"
            )


# ---------------------------------------------------------------------------
# Top-level compositor
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Composes all sub-configs into a single object.

    Can be built programmatically or loaded from a YAML / JSON file via
    ``Config.from_file(path)``.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self._validate_cross_config()
        self._validate_schedule()

    def _validate_cross_config(self):
        """Validate constraints that span multiple sub-configs."""
        if self.optimizer.warmup_steps >= self.training.max_steps:
            raise ConfigError(
                f"warmup_steps ({self.optimizer.warmup_steps}) must be < "
                f"max_steps ({self.training.max_steps})"
            )
        if self.data.max_length > self.model.n_positions:
            logger.warning(
                f"data.max_length ({self.data.max_length}) > model.n_positions "
                f"({self.model.n_positions}): sequences will exceed the model's "
                "positional encoding range"
            )

    def _validate_schedule(self):
        """Warn when the LR schedule length differs from max_steps."""
        decay = (
            self.optimizer.decay_steps
            if self.optimizer.decay_steps is not None
            else self.training.max_steps - self.optimizer.warmup_steps
        )
        schedule_steps = self.optimizer.warmup_steps + decay
        if schedule_steps != self.training.max_steps:
            if schedule_steps < self.training.max_steps:
                tail = self.training.max_steps - schedule_steps
                msg = (
                    f"LR schedule length ({schedule_steps}) < max_steps ({self.training.max_steps}). "
                    f"LR will hold at min_lr for the last {tail} steps."
                )
            else:
                excess = schedule_steps - self.training.max_steps
                msg = (
                    f"LR schedule length ({schedule_steps}) > max_steps ({self.training.max_steps}). "
                    f"Cosine decay will not complete — {excess} schedule steps remain unused."
                )
            logger.warning(msg)

    # -- serialisation -------------------------------------------------------

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Config":
        """Load a Config from a YAML or JSON file.

        Top-level keys ``model``, ``data``, ``optimizer``, ``training`` are
        each forwarded to the corresponding sub-config constructor.
        Unrecognised keys are ignored with a warning.
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")

        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML is required to load .yaml configs. "
                    "Install it with: pip install pyyaml"
                )
            raw = yaml.safe_load(text) or {}
        else:
            raw = json.loads(text)

        known_keys = {"model", "data", "optimizer", "training"}
        for key in set(raw) - known_keys:
            logger.warning(f"Ignoring unknown config key: {key!r}")

        data_raw = dict(raw.get("data", {}))
        if "curriculum" in data_raw and data_raw["curriculum"] is not None:
            data_raw["curriculum"] = [
                CurriculumPhase(**phase) for phase in data_raw["curriculum"]
            ]

        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**data_raw),
            optimizer=OptimizerConfig(**raw.get("optimizer", {})),
            training=TrainingConfig(**raw.get("training", {})),
        )

    def to_dict(self) -> dict:
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "optimizer": asdict(self.optimizer),
            "training": asdict(self.training),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Persist the config as JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
