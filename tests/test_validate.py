"""
Tests for the pre-flight validate module.

Covers:
  - Per-bucket token-budget shortfall detection (BUDGET vs. WARNING modes)
  - Curriculum-vs-max_steps mismatch
  - Missing bucket on disk -> ERROR
  - Resume drift detection (arch hash + config hash)
  - format_report runs cleanly on populated reports
  - .idx header reader matches the full reader
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.append(_root)

from gpt_simple.config import (
    Config,
    CurriculumPhase,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from gpt_simple.pretokenize import write_idx
from gpt_simple.validate import (
    Severity,
    _read_idx_header_and_total_tokens,
    format_report,
    run_offline_validation,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_bucket(bucket_dir: Path, n_shards: int, tokens_per_shard: int) -> None:
    """Write *n_shards* .bin/.idx pairs each containing *tokens_per_shard* tokens.

    One synthetic document per shard.  Token values are zeros; the validator
    only reads .idx headers so the contents don't matter for these tests.
    """
    bucket_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        bin_path = bucket_dir / f"shard_{i:03d}.bin"
        idx_path = bucket_dir / f"shard_{i:03d}.idx"
        tokens = np.zeros(tokens_per_shard, dtype=np.uint16)
        tokens.tofile(str(bin_path))
        offsets = np.array([0, tokens_per_shard], dtype=np.int64)
        overlap = np.array([0], dtype=np.uint16)
        write_idx(idx_path, offsets, overlap, dtype_code=2)


def _build_data_tree(
    root: Path,
    train_buckets: dict[str, tuple[int, int]],
    val_buckets: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Build root/train/<bucket>/*.bin and root/val/<bucket>/*.bin."""
    val_buckets = val_buckets or {b: (1, 100) for b in train_buckets}
    for b, (ns, tps) in train_buckets.items():
        _make_bucket(root / "train" / b, ns, tps)
    for b, (ns, tps) in val_buckets.items():
        _make_bucket(root / "val" / b, ns, tps)


def _make_cfg(tmp: Path, **overrides) -> Config:
    """Build a minimal valid Config rooted at *tmp*."""
    cfg = Config(
        model=ModelConfig(n_embd=64, n_layer=2, n_head=4, n_positions=128),
        data=DataConfig(
            path=str(tmp / "data"),
            tokenizer="gpt2",
            format="pretokenized",
            max_length=128,
            overlap_size=16,
        ),
        optimizer=OptimizerConfig(
            learning_rate=3e-4, warmup_steps=10
        ),
        training=TrainingConfig(
            per_device_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=100,
            output_dir=str(tmp / "outputs"),
        ),
    )
    for k, v in overrides.items():
        section, attr = k.split(".", 1)
        setattr(getattr(cfg, section), attr, v)
    return cfg


# ---------------------------------------------------------------------------
# .idx header reader
# ---------------------------------------------------------------------------


def test_idx_header_reader_total_tokens():
    """_read_idx_header_and_total_tokens must agree with full read_idx."""
    from gpt_simple.pretokenize import read_idx

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        offsets = np.array([0, 100, 250, 400, 999], dtype=np.int64)
        overlap = np.array([0, 16, 0, 32], dtype=np.uint16)
        idx_path = tmp / "shard.idx"
        write_idx(idx_path, offsets, overlap, dtype_code=2)

        dtype_full, off_full, _ = read_idx(idx_path)
        dtype_h, ndocs_h, total_h = _read_idx_header_and_total_tokens(idx_path)
        assert dtype_h == dtype_full
        assert ndocs_h == len(overlap)
        assert total_h == int(off_full[-1])


# ---------------------------------------------------------------------------
# Clean config
# ---------------------------------------------------------------------------


def test_clean_config_no_blockers():
    """A correctly-sized curriculum should produce no errors or budget findings."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 2 buckets, each with 5 shards of 12k tokens => 60k tokens/bucket
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (5, 12_000), "code": (5, 12_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.training.per_device_batch_size = 2
        cfg.training.gradient_accumulation_steps = 4
        cfg.data.max_length = 128
        # 80 steps × (2 × 4 × 1 × 128) tok/step = 81_920 tokens
        cfg.training.max_steps = 80
        # Match the curriculum to the loop so the budget-cross-check passes.
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=81_920, mix={"web": 0.5, "code": 0.5}),
        ]

        report = run_offline_validation(cfg, world_size=1)
        # No ERROR / BUDGET findings expected.
        assert not report.has_errors(), [f.message for f in report.findings if f.severity == Severity.ERROR]
        assert not report.has_budget_issues(), [
            f.message for f in report.findings if f.severity == Severity.BUDGET
        ]


# ---------------------------------------------------------------------------
# Token-budget shortfall
# ---------------------------------------------------------------------------


def test_token_budget_shortfall_is_budget_finding():
    """A bucket sized below curriculum demand emits a BUDGET finding by default."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # qa bucket is tiny relative to curriculum demand
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 100_000), "qa": (1, 1_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=200_000, mix={"web": 0.5, "qa": 0.5}),
        ]

        report = run_offline_validation(cfg, world_size=1)
        budget_findings = [
            f for f in report.findings if f.severity == Severity.BUDGET
        ]
        codes = [f.code for f in budget_findings]
        assert any("curriculum.shortfall.qa" == c for c in codes), (
            f"expected curriculum.shortfall.qa BUDGET finding, got: {codes}"
        )


def test_allow_bucket_exhaustion_downgrades_to_warning():
    """data.allow_bucket_exhaustion=True turns shortfalls into warnings."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 100_000), "qa": (1, 1_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=200_000, mix={"web": 0.5, "qa": 0.5}),
        ]
        cfg.data.allow_bucket_exhaustion = True
        # Also allow the curriculum-vs-loop mismatch so the only BUDGET signal
        # we're checking for is the bucket shortfall.
        cfg.data.allow_budget_mismatch = True

        report = run_offline_validation(cfg, world_size=1)
        # No BUDGET findings any more.
        assert not report.has_budget_issues()
        # But a WARNING with the same code should be present.
        warn_codes = [
            f.code for f in report.findings if f.severity == Severity.WARNING
        ]
        assert "curriculum.shortfall.qa" in warn_codes


def test_cumulative_demand_across_phases_detects_drained_bucket():
    """Bucket fully consumed by phase 1 should fail phase 2's demand."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # web has exactly 10k tokens — phase 1 uses it all, phase 2 wants more
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (2, 5_000), "code": (4, 100_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=10_000, mix={"web": 1.0}),
            CurriculumPhase(duration_tokens=5_000, mix={"web": 1.0}),
        ]
        report = run_offline_validation(cfg, world_size=1)
        assert report.has_budget_issues()
        web = next(
            f for f in report.findings
            if f.severity == Severity.BUDGET and f.code.endswith(".web")
        )
        assert "short" in web.message.lower()


# ---------------------------------------------------------------------------
# Curriculum-vs-training-loop budget mismatch
# ---------------------------------------------------------------------------


def test_curriculum_vs_max_steps_mismatch():
    """When the loop sees far fewer tokens than the curriculum schedules."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 1_000_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=1_000_000, mix={"web": 1.0}),
        ]
        # 100 steps × 1024 tok/step = ~100k tokens vs curriculum 1M.
        cfg.training.max_steps = 100

        report = run_offline_validation(cfg, world_size=1)
        codes = [f.code for f in report.findings]
        assert "training.budget_mismatch" in codes


def test_allow_budget_mismatch_downgrades_to_warning():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 1_000_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=1_000_000, mix={"web": 1.0}),
        ]
        cfg.training.max_steps = 100
        cfg.data.allow_budget_mismatch = True

        report = run_offline_validation(cfg, world_size=1)
        # No BUDGET findings for the mismatch.
        budgets = [f.code for f in report.findings if f.severity == Severity.BUDGET]
        assert "training.budget_mismatch" not in budgets


# ---------------------------------------------------------------------------
# Curriculum references unknown bucket
# ---------------------------------------------------------------------------


def test_curriculum_unknown_bucket_is_error():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 100_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=10_000, mix={"qa": 1.0}),
        ]
        report = run_offline_validation(cfg, world_size=1)
        codes = [f.code for f in report.findings if f.severity == Severity.ERROR]
        assert "curriculum.unknown_buckets" in codes


# ---------------------------------------------------------------------------
# Per-rank shard count check
# ---------------------------------------------------------------------------


def test_fewer_shards_than_world_is_error():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (2, 100_000)},
        )
        cfg = _make_cfg(tmp)
        report = run_offline_validation(cfg, world_size=8)
        codes = [f.code for f in report.findings if f.severity == Severity.ERROR]
        assert "data.shards.fewer_than_world" in codes


# ---------------------------------------------------------------------------
# Resume drift
# ---------------------------------------------------------------------------


def _write_fake_checkpoint(out_dir: Path, *, arch_hash: str, config_hash: str, step: int = 1000) -> Path:
    """Lay down enough of the checkpoint structure for resume validation."""
    ck = out_dir / "checkpoints" / f"checkpoint-{step}"
    ck.mkdir(parents=True, exist_ok=True)
    ts = {
        "schema_version": 1,
        "step": step,
        "tokens_trained": step * 1024,
        "wandb_run_id": "abc123",
        "config_hash": config_hash,
        "model_arch_hash": arch_hash,
        "curriculum": {"phase_idx": 0, "phase_tokens_consumed": 0, "current_mix": {}},
        "timing": {
            "wallclock_seconds_elapsed": 100.0,
            "last_save_duration_seconds": 1.0,
            "saved_at": "2026-01-01T00:00:00+00:00",
        },
        "metrics": {"loss": 2.5, "learning_rate": 1e-4, "grad_norm": 0.5, "tokens_per_sec": 1000.0},
        "lineage": {"resumed_from": None, "is_shutdown_checkpoint": False},
    }
    (ck / "trainer_state.json").write_text(json.dumps(ts))
    return ck


def test_validator_does_not_mutate_vocab_size():
    """Regression: the validator must not write the tokenizer's padded
    vocab back into ``cfg.model.vocab_size``.  The trainer captures the
    model arch hash from ``cfg.model`` BEFORE the tokenizer loads, so
    any in-place mutation here would change the hash the resume drift
    check computes and trigger spurious 'arch hash differs' errors on
    every resume."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 100_000)},
        )
        cfg = _make_cfg(tmp)
        assert cfg.model.vocab_size is None, (
            "test premise: vocab_size starts unset"
        )
        run_offline_validation(cfg, world_size=1)
        assert cfg.model.vocab_size is None, (
            "validator must not mutate cfg.model.vocab_size"
        )


def test_resume_arch_hash_stable_when_vocab_unset():
    """A resume against a checkpoint that stored the arch hash with
    ``vocab_size=None`` (which is what the trainer does for any config
    that doesn't explicitly set it) must NOT trip the arch-drift check."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 100_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.training.resume = "auto"
        assert cfg.model.vocab_size is None

        import hashlib as _h
        import json as _j
        from gpt_simple.config import ARCH_KEYS as arch_keys
        # Compute the hash the way the trainer would: directly from
        # cfg.model with vocab_size still None.
        payload = {k: getattr(cfg.model, k, None) for k in arch_keys}
        arch_hash = "sha256:" + _h.sha256(
            _j.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

        cfg_hash = "sha256:" + _h.sha256(
            _j.dumps(cfg.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]

        _write_fake_checkpoint(
            Path(cfg.training.output_dir),
            arch_hash=arch_hash,
            config_hash=cfg_hash,
        )
        report = run_offline_validation(cfg, world_size=1)
        codes_err = [f.code for f in report.findings if f.severity == Severity.ERROR]
        assert "resume.arch_drift" not in codes_err, (
            f"unexpected arch drift error; findings: "
            f"{[(f.severity.value, f.code, f.message) for f in report.findings]}"
        )


def test_resume_arch_drift_is_error():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 100_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.training.resume = "auto"
        _write_fake_checkpoint(
            Path(cfg.training.output_dir),
            arch_hash="sha256:deadbeefdeadbeef",  # intentionally wrong
            config_hash="sha256:0000000000000000",
        )
        report = run_offline_validation(cfg, world_size=1)
        codes = [f.code for f in report.findings if f.severity == Severity.ERROR]
        assert "resume.arch_drift" in codes


def test_resume_config_drift_is_info_only():
    """A non-arch config change should produce INFO, not ERROR."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (4, 100_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.training.resume = "auto"

        # The validator computes the arch hash directly from ``cfg.model``
        # without padding the vocab — same as the trainer does when it
        # first writes the checkpoint.  Mirror that here so the hash we
        # bake into the fake checkpoint matches what the validator
        # will recompute.
        import hashlib as _h
        import json as _j
        from gpt_simple.config import ARCH_KEYS as arch_keys
        payload = {k: getattr(cfg.model, k, None) for k in arch_keys}
        arch_hash = "sha256:" + _h.sha256(_j.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

        _write_fake_checkpoint(
            Path(cfg.training.output_dir),
            arch_hash=arch_hash,
            config_hash="sha256:deadbeefdeadbeef",  # mismatched
        )
        report = run_offline_validation(cfg, world_size=1)
        codes_err = [f.code for f in report.findings if f.severity == Severity.ERROR]
        codes_info = [f.code for f in report.findings if f.severity == Severity.INFO]
        assert "resume.arch_drift" not in codes_err
        assert "resume.config_drift" in codes_info


# ---------------------------------------------------------------------------
# Formatter does not crash
# ---------------------------------------------------------------------------


def test_format_report_renders():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build_data_tree(
            tmp / "data",
            train_buckets={"web": (2, 100_000), "code": (2, 100_000)},
        )
        cfg = _make_cfg(tmp)
        cfg.data.curriculum = [
            CurriculumPhase(duration_tokens=50_000, mix={"web": 0.5, "code": 0.5}),
        ]
        report = run_offline_validation(cfg, world_size=1)
        text = format_report(report, config_path="dummy.yaml")
        assert "== Model ==" in text
        assert "== Data ==" in text
        assert "== Curriculum ==" in text
        assert "== Training plan ==" in text
        assert "== Output ==" in text
        assert "Summary:" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
