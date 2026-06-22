"""
Unit tests for ``gpt_simple._checkpoint``.

Covers:
  - ``TrainerState`` JSON round-trip + tolerant deserialisation
  - ``CheckpointManager.resolve_resume`` (auto / scratch / explicit path)
  - ``apply_retention`` (keep_last_k, milestones, always-keep rules)
  - Atomic save with a small dummy model (no Accelerator-prepare needed)

Distributed and accelerator-prepared behaviour is exercised by the
larger integration tests in Phase E.  Here we keep the surface small
and CPU-only.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from gpt_simple._checkpoint import (
    SCHEMA_VERSION,
    CheckpointManager,
    CurriculumState,
    TrainerState,
    _collect_rng_state,
    _parse_checkpoint_step,
    _restore_rng_state,
)
from gpt_simple.errors import CheckpointError


# ---------------------------------------------------------------------------
# Fake Accelerator for unit tests
# ---------------------------------------------------------------------------


class _FakeAccelerator:
    """Minimal stub matching the bits of Accelerator used by CheckpointManager."""

    def __init__(self, is_main: bool = True, rank: int = 0):
        self.is_main_process = is_main
        self.process_index = rank

    def wait_for_everyone(self):
        pass

    def get_state_dict(self, model):
        return model.state_dict()

    def unwrap_model(self, model):
        return model


@pytest.fixture
def tiny_model():
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


@pytest.fixture
def tiny_optimizer(tiny_model):
    return torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)


@pytest.fixture
def tiny_scheduler(tiny_optimizer):
    return torch.optim.lr_scheduler.LambdaLR(tiny_optimizer, lr_lambda=lambda step: 1.0)


# ---------------------------------------------------------------------------
# TrainerState
# ---------------------------------------------------------------------------


class TestTrainerState:
    def test_roundtrip_defaults(self, tmp_path):
        ts = TrainerState()
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps(ts.to_dict()))

        loaded = TrainerState.load(ckpt)
        assert loaded.schema_version == SCHEMA_VERSION
        assert loaded.step == 0
        assert loaded.tokens_trained == 0
        assert loaded.curriculum.phase_idx == 0
        assert loaded.metrics.loss == float("inf")
        assert loaded.lineage.resumed_from is None

    def test_roundtrip_populated(self, tmp_path):
        ts = TrainerState(
            step=1234,
            tokens_trained=987_654,
            wandb_run_id="abc123xyz",
            config_hash="sha256:cafef00d",
            model_arch_hash="sha256:deadbeef",
        )
        ts.curriculum = CurriculumState(
            phase_idx=2,
            phase_tokens_consumed=42_000_000,
            current_mix={"web": 0.6, "code": 0.4},
        )
        ts.timing.wallclock_seconds_elapsed = 7200.5
        ts.timing.last_save_duration_seconds = 1.25
        ts.timing.saved_at = "2026-05-19T10:00:00+00:00"
        ts.metrics.loss = 2.1
        ts.metrics.learning_rate = 3e-4
        ts.metrics.grad_norm = 0.5
        ts.metrics.tokens_per_sec = 12_345.0
        ts.lineage.resumed_from = "/tmp/old"
        ts.lineage.is_shutdown_checkpoint = True

        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps(ts.to_dict()))

        loaded = TrainerState.load(ckpt)
        assert loaded.step == 1234
        assert loaded.tokens_trained == 987_654
        assert loaded.wandb_run_id == "abc123xyz"
        assert loaded.curriculum.phase_idx == 2
        assert loaded.curriculum.current_mix == {"web": 0.6, "code": 0.4}
        assert loaded.timing.wallclock_seconds_elapsed == 7200.5
        assert loaded.metrics.loss == pytest.approx(2.1)
        assert loaded.lineage.is_shutdown_checkpoint is True

    def test_tolerates_missing_optional_fields(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "step": 5,
        }))
        loaded = TrainerState.load(ckpt)
        assert loaded.step == 5
        assert loaded.curriculum.phase_idx == 0
        assert loaded.lineage.resumed_from is None

    def test_rejects_unknown_schema(self, tmp_path):
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps({
            "schema_version": 999,
        }))
        with pytest.raises(CheckpointError, match="schema"):
            TrainerState.load(ckpt)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(CheckpointError, match="not found"):
            TrainerState.load(tmp_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestParseCheckpointStep:
    @pytest.mark.parametrize("name,expected", [
        ("checkpoint-0", 0),
        ("checkpoint-1000", 1000),
        ("checkpoint-1234-shutdown", 1234),
        ("checkpoint-42-stop", 42),
        ("final", None),
        ("checkpoint-1000.partial", None),
        ("scratch", None),
        ("checkpoint-abc", None),
        ("checkpoint-", None),
    ])
    def test_parse(self, name, expected):
        assert _parse_checkpoint_step(name) == expected


class TestRngRoundtrip:
    def test_python_numpy_torch(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Snapshot
        snapshot = _collect_rng_state()

        # Burn some randomness
        _ = [random.random() for _ in range(50)]
        _ = np.random.randn(10)
        _ = torch.randn(8)

        post_burn_py = random.random()
        post_burn_np = np.random.randn(1)[0]
        post_burn_torch = torch.randn(1).item()

        # Restore
        _restore_rng_state(snapshot)

        # Sequence should match what we burned
        _ = [random.random() for _ in range(50)]
        _ = np.random.randn(10)
        _ = torch.randn(8)

        assert random.random() == post_burn_py
        assert np.random.randn(1)[0] == post_burn_np
        assert torch.randn(1).item() == post_burn_torch


# ---------------------------------------------------------------------------
# CheckpointManager: discovery + resume resolution
# ---------------------------------------------------------------------------


def _make_fake_checkpoint(output_dir: Path, name: str, step: int, *, missing_state: bool = False):
    ckpt = output_dir / "checkpoints" / name
    ckpt.mkdir(parents=True)
    (ckpt / "model").mkdir()
    (ckpt / "rng").mkdir()
    if not missing_state:
        ts = TrainerState(step=step)
        (ckpt / "trainer_state.json").write_text(json.dumps(ts.to_dict()))
    return ckpt


class TestResolveResume:
    def test_scratch_returns_none(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        assert mgr.resolve_resume("scratch") is None

    def test_auto_with_no_checkpoints(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        assert mgr.resolve_resume("auto") is None

    def test_auto_picks_latest(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        _make_fake_checkpoint(tmp_path, "checkpoint-500", 500)
        _make_fake_checkpoint(tmp_path, "checkpoint-200", 200)
        mgr = CheckpointManager(tmp_path)
        resolved = mgr.resolve_resume("auto")
        assert resolved is not None
        assert resolved.name == "checkpoint-500"

    def test_auto_prefers_final(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        _make_fake_checkpoint(tmp_path, "checkpoint-500", 500)
        _make_fake_checkpoint(tmp_path, "final", 500)
        mgr = CheckpointManager(tmp_path)
        resolved = mgr.resolve_resume("auto")
        assert resolved is not None
        assert resolved.name == "final"

    def test_auto_skips_incomplete_checkpoint(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        _make_fake_checkpoint(tmp_path, "checkpoint-500", 500, missing_state=True)
        mgr = CheckpointManager(tmp_path)
        resolved = mgr.resolve_resume("auto")
        assert resolved is not None
        assert resolved.name == "checkpoint-100"

    def test_auto_skips_partial_directories(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        (tmp_path / "checkpoints" / "checkpoint-500.partial").mkdir()
        mgr = CheckpointManager(tmp_path)
        resolved = mgr.resolve_resume("auto")
        assert resolved.name == "checkpoint-100"

    def test_auto_handles_shutdown_checkpoints(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        _make_fake_checkpoint(tmp_path, "checkpoint-300-shutdown", 300)
        mgr = CheckpointManager(tmp_path)
        resolved = mgr.resolve_resume("auto")
        assert resolved.name == "checkpoint-300-shutdown"

    def test_explicit_path_absolute(self, tmp_path):
        ckpt = _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        mgr = CheckpointManager(tmp_path)
        resolved = mgr.resolve_resume(str(ckpt))
        assert resolved == ckpt

    def test_explicit_path_missing(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        with pytest.raises(CheckpointError, match="does not exist"):
            mgr.resolve_resume("/nonexistent/foo")

    def test_explicit_path_missing_trainer_state(self, tmp_path):
        ckpt = tmp_path / "checkpoints" / "bare"
        ckpt.mkdir(parents=True)
        mgr = CheckpointManager(tmp_path)
        with pytest.raises(CheckpointError, match="missing trainer_state"):
            mgr.resolve_resume(str(ckpt))


class TestAssertCanTrainFromScratch:
    def test_empty_output(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        mgr.assert_can_train_from_scratch()  # no raise

    def test_existing_checkpoint_blocks(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        mgr = CheckpointManager(tmp_path)
        with pytest.raises(CheckpointError, match="already contains checkpoints"):
            mgr.assert_can_train_from_scratch()


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


class TestRetention:
    def test_keep_last_k_drops_old(self, tmp_path):
        for step in (100, 200, 300, 400, 500):
            _make_fake_checkpoint(tmp_path, f"checkpoint-{step}", step)
        mgr = CheckpointManager(tmp_path, keep_last_k=2, keep_milestone_every=None)
        deleted = mgr.apply_retention()
        names = {n for _, n, _ in mgr.list_checkpoints()}
        assert names == {"checkpoint-400", "checkpoint-500"}
        assert set(deleted) == {"checkpoint-100", "checkpoint-200", "checkpoint-300"}

    def test_keep_last_k_none_keeps_all(self, tmp_path):
        for step in (100, 200, 300):
            _make_fake_checkpoint(tmp_path, f"checkpoint-{step}", step)
        mgr = CheckpointManager(tmp_path, keep_last_k=None, keep_milestone_every=None)
        deleted = mgr.apply_retention()
        assert deleted == []
        assert len(mgr.list_checkpoints()) == 3

    def test_milestones_preserved(self, tmp_path):
        # keep_last_k=1 would drop 100,200,300,400 but milestones every 200 saves 200,400
        for step in (100, 200, 300, 400, 500):
            _make_fake_checkpoint(tmp_path, f"checkpoint-{step}", step)
        mgr = CheckpointManager(tmp_path, keep_last_k=1, keep_milestone_every=200)
        mgr.apply_retention()
        names = {n for _, n, _ in mgr.list_checkpoints()}
        assert names == {"checkpoint-200", "checkpoint-400", "checkpoint-500"}

    def test_final_always_kept(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        _make_fake_checkpoint(tmp_path, "final", 1000)
        mgr = CheckpointManager(tmp_path, keep_last_k=1, keep_milestone_every=None)
        # Make several more regular checkpoints to force retention
        _make_fake_checkpoint(tmp_path, "checkpoint-200", 200)
        _make_fake_checkpoint(tmp_path, "checkpoint-300", 300)
        mgr.apply_retention()
        names = {n for _, n, _ in mgr.list_checkpoints()}
        assert "final" in names

    def test_shutdown_always_kept(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        _make_fake_checkpoint(tmp_path, "checkpoint-150-shutdown", 150)
        _make_fake_checkpoint(tmp_path, "checkpoint-200", 200)
        _make_fake_checkpoint(tmp_path, "checkpoint-300", 300)
        mgr = CheckpointManager(tmp_path, keep_last_k=1, keep_milestone_every=None)
        mgr.apply_retention()
        names = {n for _, n, _ in mgr.list_checkpoints()}
        assert "checkpoint-150-shutdown" in names

    def test_only_latest_shutdown_kept(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-100-shutdown", 100)
        _make_fake_checkpoint(tmp_path, "checkpoint-200-shutdown", 200)
        _make_fake_checkpoint(tmp_path, "checkpoint-300-shutdown", 300)
        mgr = CheckpointManager(tmp_path, keep_last_k=0, keep_milestone_every=None)
        mgr.apply_retention()
        names = {n for _, n, _ in mgr.list_checkpoints()}
        # All three are -shutdown; only the most-recent shutdown is always-kept
        assert names == {"checkpoint-300-shutdown"}


# ---------------------------------------------------------------------------
# Save / load round-trip (CPU, no Accelerator.prepare)
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_full_save_load(self, tmp_path, tiny_model, tiny_optimizer, tiny_scheduler):
        # Take a training step so optimizer state is non-trivial
        x = torch.randn(4, 4)
        target = torch.randint(0, 2, (4,))
        for _ in range(3):
            out = tiny_model(x)
            loss = nn.functional.cross_entropy(out, target)
            loss.backward()
            tiny_optimizer.step()
            tiny_scheduler.step()
            tiny_optimizer.zero_grad()

        accelerator = _FakeAccelerator()
        mgr = CheckpointManager(tmp_path)

        ts = TrainerState(
            step=42,
            tokens_trained=1_000_000,
            wandb_run_id="run-xyz",
            config_hash="sha256:foo",
            model_arch_hash="sha256:bar",
        )
        ts.curriculum.phase_idx = 1
        ts.metrics.loss = float(loss.item())

        ckpt_dir = mgr.save(
            accelerator=accelerator,
            model=tiny_model,
            optimizer=tiny_optimizer,
            scheduler=tiny_scheduler,
            trainer_state=ts,
            model_config_dict={"n_embd": 4},
            tag=None,
        )

        assert ckpt_dir.name == "checkpoint-42"
        assert (ckpt_dir / "trainer_state.json").is_file()
        assert (ckpt_dir / "model" / "pytorch_model.bin").is_file()
        assert (ckpt_dir / "model" / "config.json").is_file()
        assert (ckpt_dir / "optimizer.bin").is_file()
        assert (ckpt_dir / "scheduler.bin").is_file()
        assert (ckpt_dir / "rng" / "rank_0.pkl").is_file()

        # Load into fresh model + optimizer
        fresh_model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
        fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lr_lambda=lambda s: 1.0)

        CheckpointManager.load_model_state(fresh_model, ckpt_dir)
        CheckpointManager.load_optimizer_state(fresh_opt, ckpt_dir)
        CheckpointManager.load_scheduler_state(fresh_sched, ckpt_dir)
        CheckpointManager.load_rng_state(0, ckpt_dir)

        # Parameters should match
        for p_orig, p_loaded in zip(tiny_model.parameters(), fresh_model.parameters()):
            assert torch.allclose(p_orig, p_loaded)

        # Trainer state round-trip
        ts_loaded = TrainerState.load(ckpt_dir)
        assert ts_loaded.step == 42
        assert ts_loaded.tokens_trained == 1_000_000
        assert ts_loaded.wandb_run_id == "run-xyz"
        assert ts_loaded.curriculum.phase_idx == 1

    def test_partial_directory_cleaned_up_on_overwrite(self, tmp_path, tiny_model, tiny_optimizer, tiny_scheduler):
        accelerator = _FakeAccelerator()
        mgr = CheckpointManager(tmp_path)

        # Plant a stale partial directory
        stale = tmp_path / "checkpoints" / "checkpoint-10.partial"
        stale.mkdir(parents=True)
        (stale / "garbage").write_text("x")

        ts = TrainerState(step=10)
        ckpt_dir = mgr.save(
            accelerator=accelerator,
            model=tiny_model,
            optimizer=tiny_optimizer,
            scheduler=tiny_scheduler,
            trainer_state=ts,
            model_config_dict={},
            tag=None,
        )
        assert ckpt_dir.is_dir()
        # Stale partial should be gone after rename
        assert not stale.exists()

    def test_save_with_shutdown_tag(self, tmp_path, tiny_model, tiny_optimizer, tiny_scheduler):
        accelerator = _FakeAccelerator()
        mgr = CheckpointManager(tmp_path)
        ts = TrainerState(step=99)
        ckpt_dir = mgr.save(
            accelerator=accelerator,
            model=tiny_model,
            optimizer=tiny_optimizer,
            scheduler=tiny_scheduler,
            trainer_state=ts,
            model_config_dict={},
            tag="shutdown",
        )
        assert ckpt_dir.name == "checkpoint-99-shutdown"
        # is_shutdown_checkpoint should be persisted
        loaded = TrainerState.load(ckpt_dir)
        assert loaded.lineage.is_shutdown_checkpoint is True

    def test_save_with_final_tag(self, tmp_path, tiny_model, tiny_optimizer, tiny_scheduler):
        accelerator = _FakeAccelerator()
        mgr = CheckpointManager(tmp_path)
        ts = TrainerState(step=1000)
        ckpt_dir = mgr.save(
            accelerator=accelerator,
            model=tiny_model,
            optimizer=tiny_optimizer,
            scheduler=tiny_scheduler,
            trainer_state=ts,
            model_config_dict={},
            tag="final",
        )
        assert ckpt_dir.name == "final"

    def test_retention_runs_on_save(self, tmp_path, tiny_model, tiny_optimizer, tiny_scheduler):
        accelerator = _FakeAccelerator()
        mgr = CheckpointManager(tmp_path, keep_last_k=2)
        for step in (10, 20, 30, 40):
            ts = TrainerState(step=step)
            mgr.save(
                accelerator=accelerator,
                model=tiny_model,
                optimizer=tiny_optimizer,
                scheduler=tiny_scheduler,
                trainer_state=ts,
                model_config_dict={},
                tag=None,
            )
        names = {n for _, n, _ in mgr.list_checkpoints()}
        assert names == {"checkpoint-30", "checkpoint-40"}


# ---------------------------------------------------------------------------
# Tokenizer save (idempotency)
# ---------------------------------------------------------------------------


class _DummyTokenizer:
    def __init__(self):
        self.save_count = 0

    def save_pretrained(self, path):
        self.save_count += 1
        os.makedirs(path, exist_ok=True)
        Path(path, "tokenizer.json").write_text("{}")


class TestSaveTokenizer:
    def test_first_save_writes(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        tok = _DummyTokenizer()
        path = mgr.save_tokenizer(tok)
        assert path.is_dir()
        assert tok.save_count == 1

    def test_second_save_is_noop(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        tok = _DummyTokenizer()
        mgr.save_tokenizer(tok)
        mgr.save_tokenizer(tok)
        assert tok.save_count == 1


# ---------------------------------------------------------------------------
# list_checkpoints ordering
# ---------------------------------------------------------------------------


class TestListCheckpoints:
    def test_empty_dir(self, tmp_path):
        mgr = CheckpointManager(tmp_path)
        assert mgr.list_checkpoints() == []

    def test_sorts_by_step_with_final_last(self, tmp_path):
        _make_fake_checkpoint(tmp_path, "checkpoint-300", 300)
        _make_fake_checkpoint(tmp_path, "checkpoint-100", 100)
        _make_fake_checkpoint(tmp_path, "final", 100)  # step value irrelevant
        _make_fake_checkpoint(tmp_path, "checkpoint-200", 200)
        mgr = CheckpointManager(tmp_path)
        names = [n for _, n, _ in mgr.list_checkpoints()]
        assert names == ["checkpoint-100", "checkpoint-200", "checkpoint-300", "final"]
