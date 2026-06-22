"""
Pre-flight validation for gpt_simple training runs.

Used in two places:
  - ``gpt-simple validate <config>`` runs the offline checks on the login
    node before a job is submitted (catches typos before queueing).
  - At the start of ``gpt-simple train`` we run the same offline checks
    PLUS hardware-dependent runtime probes (GPU available, mixed precision
    supported, a throwaway forward+backward+step to catch OOM).

Both surfaces share a :class:`ValidationReport` that renders to a single
human-readable summary block.  The training preamble routes findings
through the existing exception types so the trainer fails fast with a
clear exit code.

Severity levels:
  - INFO     informational, never blocks
  - WARNING  surfaced loudly, only blocks with ``--strict``
  - ERROR    always blocks
  - BUDGET   token-budget shortfall / curriculum-vs-max_steps mismatch;
             blocks unless opted out via flags
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import struct
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gpt_simple.config import ARCH_KEYS, Config

logger = logging.getLogger("gpt_simple")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------


class Severity(enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BUDGET = "budget"


@dataclass
class Finding:
    severity: Severity
    code: str
    message: str


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Structured result of a validation run.

    ``sections`` keeps the data needed to render each block of the summary;
    ``findings`` collects severity-tagged messages used to decide the exit
    code (and that the formatter inlines next to the relevant section).
    """

    sections: Dict[str, Any] = field(default_factory=dict)
    findings: List[Finding] = field(default_factory=list)

    def add(self, severity: Severity, code: str, message: str) -> None:
        self.findings.append(Finding(severity, code, message))

    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

    def has_warnings(self) -> bool:
        return any(f.severity == Severity.WARNING for f in self.findings)

    def has_budget_issues(self) -> bool:
        return any(f.severity == Severity.BUDGET for f in self.findings)

    def filter(self, severity: Severity) -> List[Finding]:
        return [f for f in self.findings if f.severity == severity]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_tokens(n: float) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{int(n)}"


def _format_bytes(n: float) -> str:
    if n >= 1024 ** 4:
        return f"{n / 1024 ** 4:.1f} TiB"
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GiB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.1f} MiB"
    if n >= 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{int(n)} B"


def _human_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m ago"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"{d}d {h}h ago"


# ---------------------------------------------------------------------------
# Model parameter accounting
# ---------------------------------------------------------------------------


def _count_params(model_cfg, vocab_size: int) -> Dict[str, int]:
    """Analytic parameter count for the SimpleLLM architecture.

    Matches :class:`gpt_simple.model.SimpleLLM` layer for layer (including
    GQA, the gated/vanilla MLP split, per-projection bias, and an untied
    output head).  We use the analytic count (rather than instantiating the
    model) so the validate command stays cheap even for 7B+ configs.
    """
    n_embd = model_cfg.n_embd
    n_layer = model_cfg.n_layer
    head_dim = model_cfg.head_dim
    kv_dim = model_cfg.kv_heads * head_dim
    norm_params_per_ln = n_embd if model_cfg.norm == "rmsnorm" else 2 * n_embd

    qkv_bias = bool(model_cfg.resolved_qkv_bias)
    attn_out_bias = bool(model_cfg.resolved_attn_out_bias)
    mlp_bias = bool(model_cfg.resolved_mlp_bias)

    embedding = vocab_size * n_embd
    h_ff = model_cfg.intermediate_dim()

    # Attention: q_proj (n_embd x n_embd) + k_proj/v_proj (n_embd x kv_dim each)
    # + c_proj (n_embd x n_embd).
    attn_per_layer = 2 * n_embd * n_embd + 2 * n_embd * kv_dim
    if qkv_bias:
        attn_per_layer += n_embd + 2 * kv_dim
    if attn_out_bias:
        attn_per_layer += n_embd

    # MLP: gated has 3 matrices (up, gate, down); vanilla has 2 (fc, proj).
    if model_cfg.mlp_type == "gated":
        mlp_per_layer = 3 * n_embd * h_ff
        if mlp_bias:
            mlp_per_layer += 2 * h_ff + n_embd
    else:
        mlp_per_layer = 2 * n_embd * h_ff
        if mlp_bias:
            mlp_per_layer += h_ff + n_embd

    ln_per_layer = 2 * norm_params_per_ln  # ln_1 + ln_2

    per_block = attn_per_layer + mlp_per_layer + ln_per_layer
    blocks_total = n_layer * per_block
    final_ln = norm_params_per_ln

    # Untied head allocates a separate vocab x n_embd projection.
    lm_head = 0 if model_cfg.tie_word_embeddings else vocab_size * n_embd

    non_embedding = blocks_total + final_ln + lm_head
    total = embedding + non_embedding
    return {
        "total": total,
        "embedding": embedding,
        "non_embedding": non_embedding,
        "attn_per_layer": attn_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "mlp_hidden": h_ff,
    }


def _bytes_per_param(mixed_precision: Optional[str]) -> Tuple[int, int]:
    """Return (param_bytes, master_bytes) for a mixed-precision setting."""
    mp = (mixed_precision or "").lower()
    if mp in ("bf16", "fp16"):
        return 2, 4
    return 4, 4


# ---------------------------------------------------------------------------
# .idx introspection
# ---------------------------------------------------------------------------


def _read_idx_header_and_total_tokens(idx_path: Path) -> Tuple[int, int, int]:
    """Return (dtype_code, num_docs, total_tokens) by reading only the header.

    The total-tokens sentinel sits at the END of the offsets array
    (entry ``num_docs``).  We seek directly to it to avoid loading the
    full offsets table — this matters when validating against thousands
    of shards.
    """
    from gpt_simple.pretokenize import IDX_MAGIC, IDX_VERSION

    with open(idx_path, "rb") as f:
        magic = f.read(4)
        if magic != IDX_MAGIC:
            raise ValueError(f"bad magic {magic!r} (expected {IDX_MAGIC!r})")
        version = struct.unpack("<I", f.read(4))[0]
        if version != IDX_VERSION:
            raise ValueError(
                f"unsupported version {version} (expected {IDX_VERSION})"
            )
        dtype_code = struct.unpack("<I", f.read(4))[0]
        num_docs = struct.unpack("<I", f.read(4))[0]
        # Skip to the sentinel offset (num_docs entries in, each 8 bytes).
        f.seek(num_docs * 8, 1)
        total_tokens = struct.unpack("<q", f.read(8))[0]
    return dtype_code, num_docs, total_tokens


@dataclass
class BucketInfo:
    name: str
    shards: int
    tokens: int
    docs: int
    bin_size_bytes: int


def _inspect_bucket(bucket_dir: Path) -> BucketInfo:
    """Sum token/doc counts across every .bin/.idx pair in *bucket_dir*."""
    bin_files = sorted(bucket_dir.glob("*.bin"))
    total_tokens = 0
    total_docs = 0
    total_bytes = 0
    for bf in bin_files:
        idx = bf.with_suffix(".idx")
        if not idx.is_file():
            raise FileNotFoundError(f"Missing .idx for {bf}")
        _, ndocs, ntok = _read_idx_header_and_total_tokens(idx)
        total_docs += ndocs
        total_tokens += ntok
        total_bytes += bf.stat().st_size
    return BucketInfo(
        name=bucket_dir.name,
        shards=len(bin_files),
        tokens=total_tokens,
        docs=total_docs,
        bin_size_bytes=total_bytes,
    )


def _collect_data_info(cfg: Config, report: ValidationReport) -> Dict[str, Any]:
    """Walk the configured data directory; record per-bucket stats.

    Adds ERROR findings for missing/empty buckets and any .bin without a
    matching .idx (or with a corrupt .idx header).  Returns a dict the
    formatter consumes — even on failure, partial info is recorded so
    the summary still renders.
    """
    info: Dict[str, Any] = {
        "path": cfg.data.path,
        "format": cfg.data.format,
        "train": {},   # bucket -> BucketInfo
        "val": {},
        "discovered_buckets": [],
        "train_total_tokens": 0,
        "val_total_tokens": 0,
    }

    if not cfg.data.path:
        report.add(Severity.ERROR, "data.path.empty", "data.path is not set")
        return info

    root = Path(cfg.data.path)
    if not root.is_dir():
        report.add(
            Severity.ERROR,
            "data.path.missing",
            f"data.path does not exist: {root}",
        )
        return info

    if cfg.data.format != "pretokenized":
        # JSONL path doesn't expose token counts cheaply; the budget
        # check is skipped but the structural validation still runs.
        report.add(
            Severity.INFO,
            "data.format.jsonl",
            "JSONL format detected; token-budget checks are skipped "
            "(only pretokenized data exposes exact token counts).",
        )

    train_dir = root / "train"
    val_dir = root / "val"
    for d, label in [(train_dir, "train"), (val_dir, "val")]:
        if not d.is_dir():
            report.add(
                Severity.ERROR,
                f"data.{label}.missing",
                f"Expected {label}/ directory not found under {root}",
            )
            return info

    bucket_names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not bucket_names:
        report.add(
            Severity.ERROR,
            "data.buckets.empty",
            f"No bucket subdirectories found in {train_dir}",
        )
        return info
    info["discovered_buckets"] = bucket_names

    for bname in bucket_names:
        for split, parent, dest_key in [
            ("train", train_dir, "train"),
            ("val", val_dir, "val"),
        ]:
            sub = parent / bname
            if not sub.is_dir():
                report.add(
                    Severity.ERROR,
                    f"data.{split}.{bname}.missing",
                    f"Expected bucket dir not found: {sub}",
                )
                continue
            if cfg.data.format != "pretokenized":
                # For JSONL we don't compute token counts, just check files.
                files = sorted(sub.glob("*.jsonl"))
                if not files:
                    report.add(
                        Severity.ERROR,
                        f"data.{split}.{bname}.no_shards",
                        f"No .jsonl files in {sub}",
                    )
                info[dest_key][bname] = BucketInfo(
                    name=bname,
                    shards=len(files),
                    tokens=0,
                    docs=0,
                    bin_size_bytes=sum(f.stat().st_size for f in files),
                )
                continue

            try:
                binfo = _inspect_bucket(sub)
            except (FileNotFoundError, ValueError, OSError) as exc:
                report.add(
                    Severity.ERROR,
                    f"data.{split}.{bname}.idx_error",
                    f"{sub}: {exc}",
                )
                continue
            if binfo.shards == 0:
                report.add(
                    Severity.ERROR,
                    f"data.{split}.{bname}.no_shards",
                    f"No .bin files in {sub}",
                )
            info[dest_key][bname] = binfo

    info["train_total_tokens"] = sum(
        b.tokens for b in info["train"].values()
    )
    info["val_total_tokens"] = sum(
        b.tokens for b in info["val"].values()
    )
    return info


# ---------------------------------------------------------------------------
# Curriculum demand check
# ---------------------------------------------------------------------------


def _compute_curriculum_demand(
    cfg: Config,
    data_info: Dict[str, Any],
    report: ValidationReport,
) -> Dict[str, Any]:
    """Cross-reference curriculum phase token demands against bucket sizes.

    Bucket consumption is computed PER phase (so the formatter can show
    a per-phase breakdown), and CUMULATIVELY across phases (since
    bucket A drained by phase 1 leaves nothing for phase 3).  Adds a
    BUDGET finding for every bucket whose cumulative demand exceeds
    what's on disk.
    """
    out: Dict[str, Any] = {
        "phases": [],            # list of dicts
        "total_demand": 0,
        "cumulative_by_bucket": {},
        "shortfall_by_bucket": {},
    }

    train_buckets = data_info.get("train", {})

    if cfg.data.curriculum is None:
        # Uniform sampling → a single virtual "phase" sized by max_steps.
        # The total tokens consumed depends on training plan; we'll cross-
        # check that in the training-plan section, not here.
        return out

    cumulative: Dict[str, float] = {b: 0.0 for b in train_buckets}
    for i, phase in enumerate(cfg.data.curriculum):
        per_bucket = {b: phase.mix.get(b, 0.0) * phase.duration_tokens for b in phase.mix}
        rows = []
        for bname, demand in per_bucket.items():
            cumulative[bname] = cumulative.get(bname, 0.0) + demand
            available = train_buckets.get(bname).tokens if bname in train_buckets else 0
            rows.append({
                "bucket": bname,
                "demand_this_phase": demand,
                "available": available,
                "cumulative_demand": cumulative[bname],
            })
        out["phases"].append({
            "index": i,
            "duration_tokens": phase.duration_tokens,
            "mix": dict(phase.mix),
            "rows": rows,
        })
        out["total_demand"] += phase.duration_tokens

    allow_exhaustion = bool(cfg.data.allow_bucket_exhaustion)
    for bname, demand in cumulative.items():
        available = train_buckets.get(bname).tokens if bname in train_buckets else 0
        if demand > available:
            shortfall = demand - available
            out["shortfall_by_bucket"][bname] = (demand, available, shortfall)
            sev = Severity.WARNING if allow_exhaustion else Severity.BUDGET
            msg = (
                f"Curriculum demands {_format_tokens(demand)} from bucket "
                f"{bname!r} but only {_format_tokens(available)} available "
                f"({_format_tokens(shortfall)} short)."
            )
            if allow_exhaustion:
                msg += (
                    " data.allow_bucket_exhaustion=true: the loader will "
                    "drop this bucket and renormalise the mix when it runs dry."
                )
            report.add(sev, f"curriculum.shortfall.{bname}", msg)

    out["cumulative_by_bucket"] = cumulative
    return out


# ---------------------------------------------------------------------------
# Training plan
# ---------------------------------------------------------------------------


def _compute_training_plan(
    cfg: Config,
    world_size: int,
    data_info: Dict[str, Any],
    curriculum_info: Dict[str, Any],
    report: ValidationReport,
) -> Dict[str, Any]:
    """Translate config into effective batch / total tokens; flag mismatches.

    Two cross-checks land here:
      * decay_steps + warmup_steps != max_steps (warning; flagged in
        Config but also surfaced in the summary).
      * curriculum total demand vs. tokens the loop will actually see —
        if they're off by >5%, that's almost always a misconfig.
    """
    eff_batch = (
        cfg.training.per_device_batch_size
        * cfg.training.gradient_accumulation_steps
        * world_size
    )
    tokens_per_step = eff_batch * cfg.data.max_length
    total_loop_tokens = tokens_per_step * cfg.training.max_steps

    plan: Dict[str, Any] = {
        "world_size": world_size,
        "per_device_batch_size": cfg.training.per_device_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "effective_batch_size": eff_batch,
        "tokens_per_step": tokens_per_step,
        "max_steps": cfg.training.max_steps,
        "total_loop_tokens": total_loop_tokens,
        "warmup_steps": cfg.optimizer.warmup_steps,
        "decay_steps": (
            cfg.optimizer.decay_steps
            if cfg.optimizer.decay_steps is not None
            else cfg.training.max_steps - cfg.optimizer.warmup_steps
        ),
        "learning_rate": cfg.optimizer.learning_rate,
        "min_lr": cfg.optimizer.learning_rate * cfg.optimizer.min_lr_ratio,
    }

    # Curriculum vs. training-loop budget cross-check.
    if cfg.data.curriculum is not None and curriculum_info["total_demand"] > 0:
        demand = curriculum_info["total_demand"]
        ratio = total_loop_tokens / demand if demand > 0 else 0.0
        plan["curriculum_demand"] = demand
        plan["curriculum_vs_loop_ratio"] = ratio
        if ratio < 0.95 or ratio > 1.05:
            allow_mismatch = bool(cfg.data.allow_budget_mismatch)
            sev = Severity.WARNING if allow_mismatch else Severity.BUDGET
            direction = "fewer" if ratio < 1.0 else "more"
            msg = (
                f"Training loop will see {_format_tokens(total_loop_tokens)} "
                f"tokens but curriculum schedules {_format_tokens(demand)} "
                f"({direction} than planned, ratio={ratio:.2f}).  Adjust "
                f"max_steps or phase durations."
            )
            if allow_mismatch:
                msg += " data.allow_budget_mismatch=true: continuing."
            report.add(sev, "training.budget_mismatch", msg)

    return plan


# ---------------------------------------------------------------------------
# Output / tokenizer / curriculum config checks
# ---------------------------------------------------------------------------


def _check_output_dir(cfg: Config, report: ValidationReport) -> Dict[str, Any]:
    out = {"path": cfg.training.output_dir, "writable": False, "exists": False}
    try:
        os.makedirs(cfg.training.output_dir, exist_ok=True)
        out["exists"] = True
        fd, tmp = tempfile.mkstemp(dir=cfg.training.output_dir, prefix=".validate_")
        os.close(fd)
        os.unlink(tmp)
        out["writable"] = True
    except OSError as exc:
        report.add(
            Severity.ERROR,
            "output.not_writable",
            f"Output directory not writable: {cfg.training.output_dir} ({exc})",
        )
    return out


def _check_tokenizer(cfg: Config, report: ValidationReport) -> Dict[str, Any]:
    info: Dict[str, Any] = {"name": cfg.data.tokenizer, "loaded": False}
    try:
        from gpt_simple.tokenizer import SimpleLLMTokenizer
        tok = SimpleLLMTokenizer(cfg.data.tokenizer)
        info["loaded"] = True
        info["vocab_size"] = tok.vocab_size
        info["padded_vocab_size"] = ((tok.vocab_size + 127) // 128) * 128
        info["pad_token_id"] = tok.pad_token_id
        info["eod_token_id"] = tok.eod_token_id
    except Exception as exc:
        report.add(
            Severity.ERROR,
            "tokenizer.load_failed",
            f"Could not load tokenizer {cfg.data.tokenizer!r}: {exc}",
        )
    return info


def _check_curriculum_buckets(
    cfg: Config,
    data_info: Dict[str, Any],
    report: ValidationReport,
) -> None:
    if cfg.data.curriculum is None:
        return
    discovered = set(data_info.get("discovered_buckets") or [])
    referenced: set = set()
    for phase in cfg.data.curriculum:
        referenced.update(phase.mix.keys())
    missing = referenced - discovered
    if missing:
        report.add(
            Severity.ERROR,
            "curriculum.unknown_buckets",
            f"Curriculum references buckets not found on disk: "
            f"{sorted(missing)}.  Discovered: {sorted(discovered)}.",
        )
    unused = discovered - referenced
    if unused:
        report.add(
            Severity.INFO,
            "curriculum.unused_buckets",
            f"Buckets discovered but never used in curriculum: {sorted(unused)}",
        )


def _check_shard_counts_vs_world(
    data_info: Dict[str, Any],
    world_size: int,
    cfg: Config,
    report: ValidationReport,
) -> None:
    """Each rank needs at least one shard per bucket or the dataloader
    exhausts at step 0; this enforces that without instantiating the
    DataModule."""
    train_buckets = data_info.get("train", {})
    if not train_buckets:
        return
    requested_workers = max(0, int(cfg.data.num_workers))
    min_shards = min(b.shards for b in train_buckets.values())
    shortest = min(train_buckets.values(), key=lambda b: b.shards).name
    if min_shards < world_size:
        report.add(
            Severity.ERROR,
            "data.shards.fewer_than_world",
            f"Bucket {shortest!r} has only {min_shards} shard(s) but "
            f"world_size={world_size}.  Every rank needs at least one "
            f"shard or the run exhausts at step 0.  Re-shard the bucket "
            f"or reduce GPU count.",
        )
    elif requested_workers > 0:
        max_safe = min_shards // world_size
        if requested_workers > max_safe:
            report.add(
                Severity.WARNING,
                "data.num_workers.will_clamp",
                f"num_workers={requested_workers} will be clamped to "
                f"{max_safe} at runtime (bucket {shortest!r}: "
                f"{min_shards} shards / world_size {world_size}).",
            )


def _check_resume_path(cfg: Config, report: ValidationReport) -> None:
    if cfg.training.resume in ("auto", "scratch"):
        return
    p = Path(cfg.training.resume)
    if not p.is_absolute():
        alt = Path(cfg.training.output_dir) / p
        if alt.is_dir():
            p = alt
    if not p.is_dir():
        report.add(
            Severity.ERROR,
            "resume.path_missing",
            f"training.resume={cfg.training.resume!r} but no such "
            f"directory exists.",
        )
    elif not (p / "trainer_state.json").is_file():
        report.add(
            Severity.ERROR,
            "resume.no_trainer_state",
            f"Resume path {p} is missing trainer_state.json.",
        )


# ---------------------------------------------------------------------------
# Resume summary (read existing checkpoint state)
# ---------------------------------------------------------------------------


def _collect_resume_info(
    cfg: Config,
    data_info: Dict[str, Any],
    report: ValidationReport,
) -> Optional[Dict[str, Any]]:
    """Inspect the resume candidate's TrainerState + dataloader state."""
    from gpt_simple._checkpoint import CheckpointManager

    mgr = CheckpointManager(output_dir=cfg.training.output_dir)
    try:
        resume_dir = mgr.resolve_resume(cfg.training.resume)
    except Exception as exc:
        report.add(
            Severity.ERROR,
            "resume.resolve_failed",
            f"Could not resolve resume target: {exc}",
        )
        return None
    if resume_dir is None:
        return None

    from gpt_simple._checkpoint import TrainerState
    try:
        ts = TrainerState.load(resume_dir)
    except Exception as exc:
        report.add(
            Severity.ERROR,
            "resume.trainer_state_unreadable",
            f"Cannot read trainer_state.json in {resume_dir}: {exc}",
        )
        return None

    # Drift detection (same hashes the trainer would compute).
    cfg_hash = "sha256:" + hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    arch_payload = {k: getattr(cfg.model, k, None) for k in ARCH_KEYS}
    arch_hash = "sha256:" + hashlib.sha256(
        json.dumps(arch_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    arch_match = ts.model_arch_hash == arch_hash or not ts.model_arch_hash
    cfg_match = ts.config_hash == cfg_hash or not ts.config_hash
    if not arch_match:
        report.add(
            Severity.ERROR,
            "resume.arch_drift",
            f"Model architecture hash differs from checkpoint at "
            f"{resume_dir}: checkpoint={ts.model_arch_hash} vs "
            f"current={arch_hash}.  Weight loading will fail.",
        )
    elif not cfg_match:
        report.add(
            Severity.INFO,
            "resume.config_drift",
            "Config hash differs from checkpoint (non-architectural "
            "field changed — usually intentional).",
        )

    # Topology compare.
    saved_topologies: List[Tuple[int, int]] = []
    try:
        states = CheckpointManager.load_all_dataloader_states(resume_dir)
    except Exception:
        states = []
    for s in states:
        ws = int(s.get("world_size", 0))
        nw = int(s.get("num_workers", 0))
        saved_topologies.append((ws, nw))

    # Union per-bucket consumed tokens across saved slots.
    consumed_by_bucket: Dict[str, int] = {}
    for s in states:
        for _, ws_dict in (s.get("worker_states") or {}).items():
            if hasattr(ws_dict, "bucket_cursors"):
                buckets = ws_dict.bucket_cursors
            else:
                buckets = (ws_dict or {}).get("bucket_cursors") or {}
            for bname, bc in buckets.items():
                fp = getattr(bc, "file_progress", None)
                if fp is None and isinstance(bc, dict):
                    fp = bc.get("file_progress") or {}
                if not fp:
                    continue
                # "items emitted" is a rough proxy for tokens consumed —
                # translating exactly would require re-running the packer
                # against each .idx file, which the summary doesn't need.
                consumed_by_bucket[bname] = consumed_by_bucket.get(bname, 0) + sum(
                    max(0, int(v)) for v in fp.values() if int(v) >= 0
                )

    age_seconds: Optional[float] = None
    if ts.timing.saved_at:
        try:
            saved = datetime.fromisoformat(ts.timing.saved_at)
            age_seconds = (datetime.now(timezone.utc) - saved).total_seconds()
        except (ValueError, TypeError):
            pass

    return {
        "checkpoint_dir": str(resume_dir),
        "is_shutdown": ts.lineage.is_shutdown_checkpoint,
        "step": ts.step,
        "tokens_trained": ts.tokens_trained,
        "wandb_run_id": ts.wandb_run_id,
        "loss": ts.metrics.loss,
        "lr": ts.metrics.learning_rate,
        "grad_norm": ts.metrics.grad_norm,
        "tokens_per_sec": ts.metrics.tokens_per_sec,
        "phase_idx": ts.curriculum.phase_idx,
        "phase_tokens_consumed": ts.curriculum.phase_tokens_consumed,
        "phase_mix": dict(ts.curriculum.current_mix or {}),
        "config_hash_checkpoint": ts.config_hash,
        "config_hash_current": cfg_hash,
        "arch_hash_checkpoint": ts.model_arch_hash,
        "arch_hash_current": arch_hash,
        "arch_match": arch_match,
        "cfg_match": cfg_match,
        "saved_at": ts.timing.saved_at,
        "age_seconds": age_seconds,
        "saved_topologies": sorted(set(saved_topologies)),
        "consumed_items_by_bucket": consumed_by_bucket,
    }


# ---------------------------------------------------------------------------
# Model section
# ---------------------------------------------------------------------------


def _build_model_section(
    cfg: Config,
    tokenizer_info: Dict[str, Any],
    report: ValidationReport,
) -> Dict[str, Any]:
    # vocab_size for the display/param-count comes from the tokenizer when
    # the user left model.vocab_size unset (the trainer will pad it later
    # the same way).  We do NOT write the padded value back into
    # ``cfg.model.vocab_size`` — that would change the model arch hash
    # (which the trainer captures from ``cfg.model`` before the tokenizer
    # loads) and cause spurious drift errors on resume.
    vocab = cfg.model.vocab_size or tokenizer_info.get("padded_vocab_size")
    if not vocab:
        vocab = 50304  # gpt2 padded; only used so the summary still renders
        report.add(
            Severity.WARNING,
            "model.vocab_unknown",
            "Could not determine vocab_size; param count is a rough estimate.",
        )
    counts = _count_params(cfg.model, vocab)
    param_bytes, master_bytes = _bytes_per_param(cfg.training.mixed_precision)
    return {
        "n_embd": cfg.model.n_embd,
        "n_layer": cfg.model.n_layer,
        "n_head": cfg.model.n_head,
        "n_positions": cfg.model.n_positions,
        "attention_mode": cfg.model.attention_mode,
        "vocab_size": vocab,
        "params": counts,
        "param_bytes": param_bytes,
        "master_bytes": master_bytes,
        "param_memory_bytes": counts["total"] * param_bytes,
        "master_memory_bytes": counts["total"] * master_bytes,
    }


# ---------------------------------------------------------------------------
# Top-level offline validation
# ---------------------------------------------------------------------------


def run_offline_validation(
    cfg: Config,
    *,
    world_size: int = 1,
) -> ValidationReport:
    """Run every check that doesn't need a GPU.

    Safe to call from a login node.  The returned :class:`ValidationReport`
    accumulates findings AND structured per-section data the formatter
    consumes — even if a section adds an ERROR finding, downstream
    sections still attempt to populate so the user sees as much of the
    summary as possible.

    Validation strictness for the two curriculum/budget classes is driven
    by ``cfg.data.allow_bucket_exhaustion`` and
    ``cfg.data.allow_budget_mismatch``.  Set them in the config (or via
    a CLI override) when those conditions are part of the experiment
    design.
    """
    report = ValidationReport()

    if world_size < 1:
        report.add(
            Severity.ERROR,
            "preflight.world_size",
            f"world_size must be >= 1 (got {world_size})",
        )

    tokenizer_info = _check_tokenizer(cfg, report)
    # NB: do not write tokenizer_info["padded_vocab_size"] back into
    # ``cfg.model.vocab_size``.  The trainer captures the architecture
    # hash from ``cfg.model`` BEFORE the tokenizer loads (see
    # ``_model_arch_hash`` callsite in ``train.py``), so any mutation
    # here would make our drift check on resume report bogus mismatches
    # against checkpoints that were saved with ``vocab_size`` still unset.
    # The model section's param-count display uses the padded value via
    # ``tokenizer_info`` directly.

    data_info = _collect_data_info(cfg, report)
    _check_curriculum_buckets(cfg, data_info, report)
    _check_shard_counts_vs_world(data_info, world_size, cfg, report)

    curriculum_info = _compute_curriculum_demand(cfg, data_info, report)
    plan = _compute_training_plan(
        cfg, world_size, data_info, curriculum_info, report,
    )
    output_info = _check_output_dir(cfg, report)
    _check_resume_path(cfg, report)

    model_section = _build_model_section(cfg, tokenizer_info, report)

    # Optional resume summary — only if a checkpoint actually exists for
    # the configured resume target.
    resume_info = _collect_resume_info(cfg, data_info, report)

    report.sections["config_path_display"] = None  # filled in by CLI layer
    report.sections["model"] = model_section
    report.sections["data"] = data_info
    report.sections["tokenizer"] = tokenizer_info
    report.sections["curriculum"] = curriculum_info
    report.sections["training_plan"] = plan
    report.sections["output"] = output_info
    report.sections["resume"] = resume_info
    return report


# ---------------------------------------------------------------------------
# Runtime probe (GPU + synthetic forward/backward/step)
# ---------------------------------------------------------------------------


def _resolve_mixed_precision_for_probe(requested: Optional[str]) -> str:
    """Pick the autocast dtype the trainer will use, for an honest probe."""
    import torch

    if requested in ("bf16", "fp16", "no"):
        return requested
    if not torch.cuda.is_available():
        return "no"
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


def _run_synthetic_probe(cfg: Config) -> Dict[str, Any]:
    """Build a *throwaway* model, run one fwd/bwd/step, then tear it down.

    Snapshots and restores RNG state around the probe so it cannot
    perturb the real training run that follows.  Uses random inputs (not
    real data) so the dataloader cursor stays at zero.  Runs the
    fwd/bwd inside the same autocast mode the real loop will use, so
    bf16-unsupported-op errors surface here instead of mid-step.
    """
    import gc
    import numpy as _np
    import torch

    from gpt_simple._checkpoint import _collect_rng_state, _restore_rng_state
    from gpt_simple.model import SimpleLLM

    out: Dict[str, Any] = {"ran": False}
    rng_snapshot = _collect_rng_state()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mp = _resolve_mixed_precision_for_probe(cfg.training.mixed_precision)
        autocast_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "no": torch.float32,
        }[mp]

        # Build a model with the user's exact config (vocab is whatever the
        # trainer would set via tokenizer-padding; if cfg.model.vocab_size
        # is unset we fall back to gpt2-padded so the probe still runs).
        vocab = cfg.model.vocab_size or 50304
        # Keep every architecture knob (GQA, mlp_type, biases, untied head, …)
        # so the memory probe reflects the real model; only override vocab and
        # disable dropout for the probe.
        import dataclasses
        probe_cfg = dataclasses.replace(cfg.model, vocab_size=vocab, dropout=0.0)
        model = SimpleLLM(
            probe_cfg,
            gradient_checkpointing=cfg.training.gradient_checkpointing,
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-9)

        B = cfg.training.per_device_batch_size
        T = cfg.data.max_length
        rng = _np.random.default_rng(0)
        input_ids = torch.from_numpy(
            rng.integers(low=0, high=vocab - 1, size=(B, T), dtype=_np.int64)
        ).to(device)
        labels = input_ids.clone()
        attention_mask = torch.ones((B, T), dtype=torch.long, device=device)
        position_ids = (
            torch.arange(T, dtype=torch.long, device=device)
            .unsqueeze(0).expand(B, T).contiguous()
        )

        with torch.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            dtype=autocast_dtype,
            enabled=(mp != "no"),
        ):
            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )
            loss = output.loss

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
            out["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        out["mixed_precision_used"] = mp
        out["probe_loss"] = float(loss.detach().to(torch.float32).item())
        out["batch_shape"] = f"({B}, {T})"
        out["ran"] = True

        del loss, output, input_ids, labels, attention_mask, position_ids
        del opt, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    finally:
        _restore_rng_state(rng_snapshot)
    return out


def run_runtime_validation(
    cfg: Config,
    report: ValidationReport,
    *,
    skip_probe: bool = False,
) -> None:
    """Augment an offline report with GPU + synthetic-batch checks.

    Always runs on rank 0 only; caller is responsible for not invoking
    this on other ranks (the trainer does this by gating on
    ``accelerator.is_main_process``).  Findings are added in-place and
    a new ``runtime`` section is populated for the formatter.
    """
    import torch

    runtime: Dict[str, Any] = {}
    if torch.cuda.is_available():
        runtime["cuda"] = f"available ({torch.cuda.device_count()} device(s))"
        runtime["device_name"] = torch.cuda.get_device_name(0)
        bf16_ok = torch.cuda.is_bf16_supported()
        runtime["bf16_supported"] = "yes" if bf16_ok else "no"
        if cfg.training.mixed_precision == "bf16" and not bf16_ok:
            report.add(
                Severity.ERROR,
                "runtime.bf16_unsupported",
                f"mixed_precision='bf16' but GPU "
                f"{torch.cuda.get_device_name(0)!r} does not support bf16. "
                f"Switch to 'fp16' (V100/T4) or 'no'.",
            )
    else:
        runtime["cuda"] = "NOT AVAILABLE"
        if cfg.training.mixed_precision in ("bf16", "fp16"):
            report.add(
                Severity.ERROR,
                "runtime.no_cuda",
                f"mixed_precision={cfg.training.mixed_precision!r} but no "
                "CUDA device is visible from this process.",
            )
        else:
            report.add(
                Severity.WARNING,
                "runtime.cpu_only",
                "No CUDA device — training will fall back to CPU (very slow).",
            )

    if skip_probe:
        runtime["probe"] = "skipped (--skip-runtime-probe)"
        report.sections["runtime"] = runtime
        return

    try:
        probe = _run_synthetic_probe(cfg)
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        runtime["probe"] = "OOM"
        report.add(
            Severity.ERROR,
            "runtime.probe.oom",
            f"Probe OOMed at batch ({cfg.training.per_device_batch_size}, "
            f"{cfg.data.max_length}): {exc}.  Reduce per_device_batch_size "
            f"or enable gradient_checkpointing.",
        )
    except Exception as exc:
        runtime["probe"] = f"failed: {type(exc).__name__}"
        report.add(
            Severity.ERROR,
            "runtime.probe.failed",
            f"Synthetic forward/backward/step failed: {exc}",
        )
    else:
        runtime["probe"] = (
            f"ok (loss={probe['probe_loss']:.3f}, "
            f"shape={probe['batch_shape']}, mp={probe['mixed_precision_used']})"
        )
        if "peak_memory_bytes" in probe:
            runtime["probe_peak_memory"] = _format_bytes(probe["peak_memory_bytes"])

    report.sections["runtime"] = runtime


# ---------------------------------------------------------------------------
# Plain-text formatter
# ---------------------------------------------------------------------------


def _findings_for_section(report: ValidationReport, prefix: str) -> List[Finding]:
    return [f for f in report.findings if f.code.startswith(prefix)]


def _bullet(line: str) -> str:
    return f"  {line}"


def _format_findings(findings: List[Finding], indent: str = "    ") -> List[str]:
    """Render findings with a one-char severity marker."""
    marker = {
        Severity.INFO: "i",
        Severity.WARNING: "!",
        Severity.ERROR: "x",
        Severity.BUDGET: "$",
    }
    return [f"{indent}[{marker[f.severity]}] {f.message}" for f in findings]


def format_report(report: ValidationReport, *, config_path: Optional[str] = None) -> str:
    """Render a :class:`ValidationReport` as the plain-text summary block."""
    lines: List[str] = []
    lines.append("=" * 72)
    title = "gpt-simple validate"
    if config_path:
        title = f"{title} — {config_path}"
    lines.append(title)
    lines.append("=" * 72)

    # -- Model ---------------------------------------------------------------
    m = report.sections.get("model") or {}
    if m:
        lines.append("")
        lines.append("== Model ==")
        lines.append(_bullet(
            f"Arch: {m['n_layer']}L / {m['n_embd']}d / {m['n_head']}h "
            f"/ ctx={m['n_positions']}  attention={m['attention_mode']}"
        ))
        p = m["params"]
        lines.append(_bullet(
            f"Params: {p['total']:,} total  "
            f"({p['non_embedding']:,} non-embedding, "
            f"{p['embedding']:,} embedding)"
        ))
        lines.append(_bullet(
            f"Vocab: {m['vocab_size']:,}  "
            f"MLP hidden: {p['mlp_hidden']:,}"
        ))
        lines.append(_bullet(
            f"Weight mem: {_format_bytes(m['param_memory_bytes'])} "
            f"({m['param_bytes']} B/param)  "
            f"+ optim master copy: {_format_bytes(m['master_memory_bytes'])}"
        ))
    findings = _findings_for_section(report, "model.")
    lines.extend(_format_findings(findings))

    # -- Data ----------------------------------------------------------------
    d = report.sections.get("data") or {}
    if d:
        lines.append("")
        lines.append("== Data ==")
        lines.append(_bullet(f"Path: {d.get('path')}"))
        lines.append(_bullet(f"Format: {d.get('format')}"))
        tk = report.sections.get("tokenizer") or {}
        if tk.get("loaded"):
            lines.append(_bullet(
                f"Tokenizer: {tk['name']}  "
                f"vocab={tk['vocab_size']:,} -> padded {tk['padded_vocab_size']:,}"
            ))
        elif tk.get("name"):
            lines.append(_bullet(f"Tokenizer: {tk['name']} (not loaded)"))

        train = d.get("train") or {}
        if train:
            lines.append(_bullet(f"Train buckets ({len(train)}):"))
            for bname in sorted(train):
                b = train[bname]
                lines.append(
                    f"      - {bname:<20}  "
                    f"{b.shards:>4} shards  "
                    f"{_format_tokens(b.tokens):>8} tok  "
                    f"{b.docs:>10,} docs"
                )
            lines.append(_bullet(
                f"Total train tokens: "
                f"{_format_tokens(d.get('train_total_tokens', 0))}"
            ))
        val = d.get("val") or {}
        if val:
            val_tokens = d.get('val_total_tokens', 0)
            lines.append(_bullet(
                f"Total val tokens:   {_format_tokens(val_tokens)} "
                f"across {len(val)} bucket(s)"
            ))
    findings = _findings_for_section(report, "data.")
    lines.extend(_format_findings(findings))
    findings = _findings_for_section(report, "tokenizer.")
    lines.extend(_format_findings(findings))

    # -- Curriculum ----------------------------------------------------------
    c = report.sections.get("curriculum") or {}
    lines.append("")
    lines.append("== Curriculum ==")
    if not c.get("phases"):
        lines.append(_bullet("(no curriculum configured — uniform sampling)"))
    else:
        for phase in c["phases"]:
            mix_str = "  ".join(
                f"{b}={w:.2f}" for b, w in sorted(phase["mix"].items())
            )
            lines.append(_bullet(
                f"Phase {phase['index'] + 1}  "
                f"({_format_tokens(phase['duration_tokens'])} tokens):  {mix_str}"
            ))
            for row in phase["rows"]:
                pct = (
                    100 * row["cumulative_demand"] / row["available"]
                    if row["available"] > 0 else float("inf")
                )
                marker = "x" if row["cumulative_demand"] > row["available"] else " "
                lines.append(
                    f"        [{marker}] {row['bucket']:<18}  "
                    f"this phase: {_format_tokens(row['demand_this_phase']):>8}  "
                    f"cumulative: {_format_tokens(row['cumulative_demand']):>8} / "
                    f"{_format_tokens(row['available']):>8}  "
                    f"({pct:>5.1f}% used)"
                )
        lines.append(_bullet(
            f"Total curriculum demand: "
            f"{_format_tokens(c['total_demand'])}"
        ))
    findings = _findings_for_section(report, "curriculum.")
    lines.extend(_format_findings(findings))

    # -- Training plan -------------------------------------------------------
    plan = report.sections.get("training_plan") or {}
    if plan:
        lines.append("")
        lines.append("== Training plan ==")
        lines.append(_bullet(
            f"world_size={plan['world_size']}  "
            f"per_device_bs={plan['per_device_batch_size']}  "
            f"grad_accum={plan['gradient_accumulation_steps']}  "
            f"->  effective bs={plan['effective_batch_size']}"
        ))
        lines.append(_bullet(
            f"Tokens/step: {plan['tokens_per_step']:,}  "
            f"max_steps: {plan['max_steps']:,}  "
            f"->  total loop tokens: "
            f"{_format_tokens(plan['total_loop_tokens'])}"
        ))
        lines.append(_bullet(
            f"LR: {plan['learning_rate']:.2e}  "
            f"warmup: {plan['warmup_steps']}  "
            f"cosine decay over {plan['decay_steps']} steps to "
            f"{plan['min_lr']:.2e}"
        ))
        if "curriculum_demand" in plan:
            lines.append(_bullet(
                f"Curriculum schedules "
                f"{_format_tokens(plan['curriculum_demand'])} tokens  "
                f"(loop will see {_format_tokens(plan['total_loop_tokens'])}, "
                f"ratio={plan['curriculum_vs_loop_ratio']:.2f})"
            ))
    findings = _findings_for_section(report, "training.")
    lines.extend(_format_findings(findings))

    # -- Output --------------------------------------------------------------
    out = report.sections.get("output") or {}
    if out:
        lines.append("")
        lines.append("== Output ==")
        marker = "ok" if out["writable"] else "!!"
        lines.append(_bullet(
            f"output_dir: {out['path']}  ({marker} writable)"
        ))
    findings = _findings_for_section(report, "output.")
    lines.extend(_format_findings(findings))

    # -- Resume --------------------------------------------------------------
    r = report.sections.get("resume")
    if r:
        lines.append("")
        lines.append("== Resume ==")
        age = (
            f" ({_human_age(r['age_seconds'])})"
            if r.get("age_seconds") is not None else ""
        )
        lines.append(_bullet(f"Resuming from: {r['checkpoint_dir']}"))
        lines.append(_bullet(
            f"Saved at: {r.get('saved_at') or '?'}{age}  "
            f"({'shutdown' if r['is_shutdown'] else 'regular'} checkpoint)"
        ))
        lines.append(_bullet(
            f"Previous run: step {r['step']:,}  "
            f"tokens trained: {_format_tokens(r['tokens_trained'])}  "
            f"loss: {r['loss']:.3f}  lr: {r['lr']:.2e}  "
            f"throughput: {r['tokens_per_sec']:,.0f} tok/s"
        ))
        if r.get("wandb_run_id"):
            lines.append(_bullet(
                f"W&B: continuing run id {r['wandb_run_id']}"
            ))
        if r.get("phase_mix"):
            mix = "  ".join(
                f"{b}={w:.2f}" for b, w in sorted(r['phase_mix'].items())
            )
            lines.append(_bullet(
                f"Curriculum: in phase {r['phase_idx'] + 1}  "
                f"phase tokens consumed: "
                f"{_format_tokens(r['phase_tokens_consumed'])}  ({mix})"
            ))
        arch_marker = "ok" if r["arch_match"] else "x"
        cfg_marker = "ok" if r["cfg_match"] else "i"
        lines.append(_bullet(
            f"Drift: arch hash {arch_marker}  config hash {cfg_marker}"
        ))
        topo = r.get("saved_topologies") or []
        if topo:
            lines.append(_bullet(
                f"Saved topology(ies): "
                f"{', '.join(f'ws={w}/nw={n}' for w, n in topo)}  "
                f"(current world_size in CLI: see --world-size)"
            ))
        consumed = r.get("consumed_items_by_bucket") or {}
        if consumed:
            lines.append(_bullet("Bucket items consumed so far:"))
            for bname in sorted(consumed):
                lines.append(
                    f"      - {bname:<20}  "
                    f"{consumed[bname]:,} items emitted"
                )
    findings = _findings_for_section(report, "resume.")
    lines.extend(_format_findings(findings))

    # -- Runtime (only present when probe was run) ---------------------------
    rt = report.sections.get("runtime")
    if rt:
        lines.append("")
        lines.append("== Runtime ==")
        for k, v in rt.items():
            lines.append(_bullet(f"{k}: {v}"))
    findings = _findings_for_section(report, "runtime.")
    lines.extend(_format_findings(findings))

    # -- Summary footer ------------------------------------------------------
    errors = report.filter(Severity.ERROR)
    warnings = report.filter(Severity.WARNING)
    budget = report.filter(Severity.BUDGET)
    infos = report.filter(Severity.INFO)
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"Summary: {len(errors)} error(s)  "
        f"{len(budget)} budget issue(s)  "
        f"{len(warnings)} warning(s)  "
        f"{len(infos)} info"
    )
    lines.append("=" * 72)
    return "\n".join(lines)


__all__ = [
    "Severity",
    "Finding",
    "ValidationReport",
    "BucketInfo",
    "run_offline_validation",
    "run_runtime_validation",
    "format_report",
]
