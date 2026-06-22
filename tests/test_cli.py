"""
Comprehensive tests for gpt_simple Phase 5: CLI, logging, errors, and run-state.

Covers:
  - errors.py: exception hierarchy, exit codes
  - _logging.py: setup_logging levels and handler selection
  - _run_state.py: write/read round-trip, atomicity, new_run_state
  - cli/main.py: subcommand dispatch, error handling, help output
  - cli/init_cmd.py: template generation, presets, file output, YAML validity
  - cli/status_cmd.py: running/completed/error/crashed display, plain fallback
  - cli/stop_cmd.py: graceful and force stop logic
  - cli/train_cmd.py: config override application, single vs distributed routing
  - cli/tokenize_cmd.py: sys.argv forwarding
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# errors.py
# ---------------------------------------------------------------------------


class TestErrors:
    def test_hierarchy(self):
        from gpt_simple.errors import (
            CheckpointError,
            ConfigError,
            DataError,
            GptSimpleError,
        )
        assert issubclass(ConfigError, GptSimpleError)
        assert issubclass(DataError, GptSimpleError)
        assert issubclass(CheckpointError, GptSimpleError)
        assert issubclass(GptSimpleError, Exception)

    def test_exit_codes(self):
        from gpt_simple.errors import (
            CheckpointError,
            ConfigError,
            DataError,
            GptSimpleError,
        )
        assert GptSimpleError.exit_code == 1
        assert ConfigError.exit_code == 2
        assert DataError.exit_code == 3
        assert CheckpointError.exit_code == 4

    def test_catchable_as_base(self):
        from gpt_simple.errors import ConfigError, GptSimpleError

        with pytest.raises(GptSimpleError):
            raise ConfigError("bad config")

    def test_message_preserved(self):
        from gpt_simple.errors import DataError

        exc = DataError("missing shards")
        assert str(exc) == "missing shards"


# ---------------------------------------------------------------------------
# _logging.py
# ---------------------------------------------------------------------------


class TestLogging:
    @pytest.fixture(autouse=True)
    def _clean_logger(self):
        """Remove all handlers from the gpt_simple logger between tests."""
        lgr = logging.getLogger("gpt_simple")
        lgr.handlers.clear()
        lgr.setLevel(logging.WARNING)
        yield
        lgr.handlers.clear()
        lgr.setLevel(logging.WARNING)

    def test_default_level_is_info(self):
        from gpt_simple._logging import setup_logging

        setup_logging(use_rich=False)
        lgr = logging.getLogger("gpt_simple")
        assert lgr.level == logging.INFO

    def test_verbose_sets_debug(self):
        from gpt_simple._logging import setup_logging

        setup_logging(verbose=True, use_rich=False)
        lgr = logging.getLogger("gpt_simple")
        assert lgr.level == logging.DEBUG

    def test_quiet_sets_warning(self):
        from gpt_simple._logging import setup_logging

        setup_logging(quiet=True, use_rich=False)
        lgr = logging.getLogger("gpt_simple")
        assert lgr.level == logging.WARNING

    def test_plain_handler_format(self):
        from gpt_simple._logging import setup_logging

        setup_logging(use_rich=False)
        lgr = logging.getLogger("gpt_simple")
        assert len(lgr.handlers) == 1
        fmt = lgr.handlers[0].formatter._fmt
        assert "gpt_simple" in fmt

    def test_idempotent(self):
        from gpt_simple._logging import setup_logging

        setup_logging(use_rich=False)
        setup_logging(use_rich=False)
        lgr = logging.getLogger("gpt_simple")
        assert len(lgr.handlers) == 1

    def test_rich_handler_when_available(self):
        from gpt_simple._logging import setup_logging

        setup_logging(use_rich=True)
        lgr = logging.getLogger("gpt_simple")
        assert len(lgr.handlers) == 1
        handler_type = type(lgr.handlers[0]).__name__
        try:
            import rich  # noqa: F401
            assert handler_type == "RichHandler"
        except ImportError:
            assert handler_type == "StreamHandler"

    def test_rich_fallback_when_unavailable(self):
        from gpt_simple._logging import setup_logging

        with mock.patch.dict(sys.modules, {"rich": None, "rich.logging": None}):
            lgr = logging.getLogger("gpt_simple")
            lgr.handlers.clear()
            setup_logging(use_rich=True)
            assert len(lgr.handlers) == 1
            assert type(lgr.handlers[0]).__name__ == "StreamHandler"


# ---------------------------------------------------------------------------
# _run_state.py
# ---------------------------------------------------------------------------


class TestRunState:
    def test_write_read_roundtrip(self, tmp_path):
        from gpt_simple._run_state import RunState

        state = RunState(
            status="running",
            pid=12345,
            started_at="2026-01-01T00:00:00+00:00",
            updated_at="",
            global_step=100,
            max_steps=1000,
            loss=3.14,
            learning_rate=1e-4,
            tokens_trained=500_000,
            tokens_per_sec=10_000.0,
            latest_checkpoint="./ckpt-100",
        )
        state.write(str(tmp_path))

        loaded = RunState.read(str(tmp_path))
        assert loaded is not None
        assert loaded.status == "running"
        assert loaded.pid == 12345
        assert loaded.global_step == 100
        assert loaded.loss == pytest.approx(3.14)
        assert loaded.latest_checkpoint == "./ckpt-100"
        assert loaded.updated_at != ""

    def test_read_missing_returns_none(self, tmp_path):
        from gpt_simple._run_state import RunState

        assert RunState.read(str(tmp_path)) is None

    def test_read_corrupt_json_returns_none(self, tmp_path):
        from gpt_simple._run_state import RunState

        state_file = tmp_path / ".run_state.json"
        state_file.write_text("not valid json{{{")
        assert RunState.read(str(tmp_path)) is None

    def test_read_ignores_unknown_keys(self, tmp_path):
        from gpt_simple._run_state import RunState

        state_file = tmp_path / ".run_state.json"
        data = {"status": "completed", "pid": 1, "unknown_field": "ignored"}
        state_file.write_text(json.dumps(data))

        loaded = RunState.read(str(tmp_path))
        assert loaded is not None
        assert loaded.status == "completed"
        assert not hasattr(loaded, "unknown_field")

    def test_write_creates_directory(self, tmp_path):
        from gpt_simple._run_state import RunState

        nested = tmp_path / "deep" / "nested"
        state = RunState(status="running", pid=1)
        state.write(str(nested))
        assert (nested / ".run_state.json").exists()

    def test_new_run_state_defaults(self):
        from gpt_simple._run_state import new_run_state

        state = new_run_state(max_steps=5000, config_path="/path/to/config.json")
        assert state.status == "running"
        assert state.pid == os.getpid()
        assert state.max_steps == 5000
        assert state.config_path == "/path/to/config.json"
        assert state.started_at != ""
        assert state.global_step == 0

    def test_state_path(self, tmp_path):
        from gpt_simple._run_state import RunState

        p = RunState.state_path(str(tmp_path))
        assert p.endswith(".run_state.json")

    def test_error_state_with_traceback(self, tmp_path):
        from gpt_simple._run_state import RunState

        state = RunState(
            status="error",
            pid=999,
            global_step=42,
            max_steps=100,
            error="Traceback (most recent call last):\n  RuntimeError: CUDA OOM",
        )
        state.write(str(tmp_path))

        loaded = RunState.read(str(tmp_path))
        assert loaded.status == "error"
        assert "CUDA OOM" in loaded.error


# ---------------------------------------------------------------------------
# cli/main.py -- subcommand dispatch
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_no_args_prints_help(self, capsys):
        from gpt_simple.cli.main import main

        main([])
        out = capsys.readouterr().out
        assert "gpt-simple" in out or "usage" in out.lower()

    def test_init_subcommand_exists(self, capsys):
        from gpt_simple.cli.main import main

        main(["init"])
        out = capsys.readouterr().out
        assert "model:" in out

    def test_gpt_simple_error_caught(self):
        from gpt_simple.cli.main import main
        from gpt_simple.errors import ConfigError

        with mock.patch(
            "gpt_simple.cli.init_cmd.InitCommand.run",
            side_effect=ConfigError("test error"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main(["init"])
            assert exc_info.value.code == 2

    def test_keyboard_interrupt_exit_130(self):
        from gpt_simple.cli.main import main

        with mock.patch(
            "gpt_simple.cli.init_cmd.InitCommand.run",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(SystemExit) as exc_info:
                main(["init"])
            assert exc_info.value.code == 130

    def test_verbose_flag_accepted(self, capsys):
        from gpt_simple.cli.main import main

        main(["-v", "init"])
        out = capsys.readouterr().out
        assert "model:" in out

    def test_quiet_flag_accepted(self, capsys):
        from gpt_simple.cli.main import main

        main(["-q", "init"])
        out = capsys.readouterr().out
        assert "model:" in out


# ---------------------------------------------------------------------------
# cli/init_cmd.py
# ---------------------------------------------------------------------------


class TestInitCommand:
    def test_default_template_stdout(self, capsys):
        from gpt_simple.cli.main import main

        main(["init"])
        out = capsys.readouterr().out
        assert "model:" in out
        assert "data:" in out
        assert "optimizer:" in out
        assert "training:" in out

    def test_default_template_is_valid_yaml(self):
        from gpt_simple.cli.init_cmd import _template

        yaml = pytest.importorskip("yaml")
        text = _template()
        parsed = yaml.safe_load(text)
        assert isinstance(parsed, dict)
        assert "model" in parsed
        assert "data" in parsed
        assert "optimizer" in parsed
        assert "training" in parsed

    def test_preset_small(self):
        from gpt_simple.cli.init_cmd import _template

        text = _template(preset_name="small")
        assert "n_embd: 768" in text
        assert "n_layer: 12" in text
        assert "~125M" in text

    def test_preset_medium(self):
        from gpt_simple.cli.init_cmd import _template

        text = _template(preset_name="medium")
        assert "n_embd: 1024" in text
        assert "n_layer: 24" in text

    def test_preset_large(self):
        from gpt_simple.cli.init_cmd import _template

        text = _template(preset_name="large")
        assert "n_embd: 2048" in text
        assert "n_head: 32" in text

    def test_preset_template_is_valid_yaml(self):
        from gpt_simple.cli.init_cmd import _template

        yaml = pytest.importorskip("yaml")
        for preset in ("small", "medium", "large"):
            text = _template(preset_name=preset)
            parsed = yaml.safe_load(text)
            assert parsed["model"]["n_embd"] > 0

    def test_output_to_file(self, tmp_path):
        from gpt_simple.cli.main import main

        out_file = tmp_path / "my_config.yaml"
        main(["init", "--output", str(out_file)])
        assert out_file.exists()
        content = out_file.read_text()
        assert "model:" in content

    def test_unknown_preset_rejected(self):
        from gpt_simple.cli.main import main

        with pytest.raises(SystemExit):
            main(["init", "--preset", "nonexistent"])


# ---------------------------------------------------------------------------
# cli/status_cmd.py
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def _write_state(self, tmp_path, **overrides):
        from gpt_simple._run_state import RunState

        defaults = dict(
            status="running",
            pid=os.getpid(),
            started_at="2026-04-06T14:30:00+00:00",
            updated_at="2026-04-06T14:35:00+00:00",
            global_step=500,
            max_steps=10000,
            loss=3.24,
            learning_rate=3e-4,
            tokens_trained=50_000_000,
            tokens_per_sec=125_000.0,
            latest_checkpoint="./outputs/checkpoint-500",
        )
        defaults.update(overrides)
        state = RunState(**defaults)
        state.write(str(tmp_path))
        return state

    def _display(self, state):
        """Call the rewritten _display_state with only RunState (Phase D)."""
        from gpt_simple.cli.status_cmd import _display_state

        _display_state(
            state,
            trainer_state=None,
            latest_ckpt_dir=None,
            recent_ckpts=[],
            shutdown_requested=None,
        )

    def test_running_state_display(self, tmp_path, capsys):
        self._write_state(tmp_path, status="running")

        from gpt_simple._run_state import RunState

        state = RunState.read(str(tmp_path))
        self._display(state)

        out = capsys.readouterr().out
        assert "RUNNING" in out
        assert "500" in out
        assert "10000" in out

    def test_completed_state_display(self, tmp_path, capsys):
        self._write_state(tmp_path, status="completed", global_step=10000)

        from gpt_simple._run_state import RunState

        state = RunState.read(str(tmp_path))
        self._display(state)

        out = capsys.readouterr().out
        assert "COMPLETED" in out

    def test_error_state_shows_traceback(self, tmp_path, capsys):
        self._write_state(
            tmp_path,
            status="error",
            global_step=42,
            error="Traceback (most recent call last):\n  File ...\nRuntimeError: CUDA OOM",
        )

        from gpt_simple._run_state import RunState

        state = RunState.read(str(tmp_path))
        self._display(state)

        out = capsys.readouterr().out
        assert "ERROR" in out
        assert "42" in out
        assert "CUDA OOM" in out

    def test_crashed_detection(self, tmp_path, capsys):
        """A 'running' state with a dead PID should show CRASHED."""
        self._write_state(tmp_path, status="running", pid=999999999)

        from gpt_simple._run_state import RunState

        state = RunState.read(str(tmp_path))
        self._display(state)

        out = capsys.readouterr().out
        assert "CRASHED" in out

    def test_no_state_file_exits(self, tmp_path):
        from gpt_simple.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["status", "--output_dir", str(tmp_path)])
        assert exc_info.value.code == 1

    def test_stopped_state_display(self, tmp_path, capsys):
        self._write_state(tmp_path, status="stopped")

        from gpt_simple._run_state import RunState

        state = RunState.read(str(tmp_path))
        self._display(state)

        out = capsys.readouterr().out
        assert "STOPPED" in out

    def test_plain_fallback_display(self, tmp_path, capsys):
        """Force plain text output by hiding rich."""
        self._write_state(tmp_path, status="running")

        from gpt_simple._run_state import RunState
        from gpt_simple.cli.status_cmd import _display_plain

        state = RunState.read(str(tmp_path))
        _display_plain(
            state,
            None,
            None,
            [],
            None,
            "RUNNING (PID 123)",
            "2026-04-06",
            " (1h ago)",
        )

        out = capsys.readouterr().out
        assert "Status:" in out
        assert "RUNNING" in out
        assert "Progress:" in out


# ---------------------------------------------------------------------------
# cli/status_cmd.py -- helper functions
# ---------------------------------------------------------------------------


class TestStatusHelpers:
    def test_human_duration_seconds(self):
        from gpt_simple.cli.status_cmd import _human_duration

        assert _human_duration(30) == "30s"

    def test_human_duration_minutes(self):
        from gpt_simple.cli.status_cmd import _human_duration

        assert _human_duration(300) == "5m"

    def test_human_duration_hours(self):
        from gpt_simple.cli.status_cmd import _human_duration

        result = _human_duration(7200)
        assert "2h" in result

    def test_human_duration_days(self):
        from gpt_simple.cli.status_cmd import _human_duration

        result = _human_duration(90000)
        assert "1d" in result

    def test_format_tokens_billions(self):
        from gpt_simple.cli.status_cmd import _format_tokens

        assert _format_tokens(1_500_000_000) == "1.5B"

    def test_format_tokens_millions(self):
        from gpt_simple.cli.status_cmd import _format_tokens

        assert _format_tokens(50_000_000) == "50.0M"

    def test_format_tokens_thousands(self):
        from gpt_simple.cli.status_cmd import _format_tokens

        assert _format_tokens(5_000) == "5.0K"

    def test_format_tokens_small(self):
        from gpt_simple.cli.status_cmd import _format_tokens

        assert _format_tokens(42) == "42"


# ---------------------------------------------------------------------------
# cli/status_cmd.py -- Phase D: TrainerState, checkpoint history, shutdown flag
# ---------------------------------------------------------------------------


class TestStatusPhaseD:
    """Tests for the Phase-D status enhancements."""

    def _write_ckpt(self, output_dir: Path, step: int, **trainer_state_fields) -> Path:
        from gpt_simple._checkpoint import TrainerState

        ckpt = output_dir / "checkpoints" / f"checkpoint-{step}"
        ckpt.mkdir(parents=True)
        ts = TrainerState(step=step, **trainer_state_fields)
        (ckpt / "trainer_state.json").write_text(json.dumps(ts.to_dict()))
        return ckpt

    def test_load_latest_trainer_state_returns_none_when_empty(self, tmp_path):
        from gpt_simple.cli.status_cmd import _load_latest_trainer_state

        ts, ckpt = _load_latest_trainer_state(str(tmp_path))
        assert ts is None and ckpt is None

    def test_load_latest_trainer_state_picks_highest_step(self, tmp_path):
        from gpt_simple.cli.status_cmd import _load_latest_trainer_state

        for step in (100, 500, 250):
            self._write_ckpt(tmp_path, step=step, tokens_trained=step * 1000)

        ts, ckpt = _load_latest_trainer_state(str(tmp_path))
        assert ts is not None
        assert ts.step == 500
        assert ts.tokens_trained == 500_000
        assert ckpt is not None and ckpt.name == "checkpoint-500"

    def test_load_latest_trainer_state_handles_missing_dir(self, tmp_path):
        from gpt_simple.cli.status_cmd import _load_latest_trainer_state

        missing = tmp_path / "does" / "not" / "exist"
        ts, ckpt = _load_latest_trainer_state(str(missing))
        assert ts is None and ckpt is None

    def test_list_recent_checkpoints(self, tmp_path):
        from gpt_simple.cli.status_cmd import _list_recent_checkpoints

        for step in (100, 200, 300, 400, 500, 600, 700):
            self._write_ckpt(tmp_path, step=step)

        history = _list_recent_checkpoints(str(tmp_path), limit=3)
        assert len(history) == 3
        assert [s for s, _ in history] == [500, 600, 700]

    def test_list_recent_checkpoints_handles_missing_dir(self, tmp_path):
        from gpt_simple.cli.status_cmd import _list_recent_checkpoints

        assert _list_recent_checkpoints(str(tmp_path / "missing")) == []

    def test_shutdown_flag_path_set(self, tmp_path):
        from gpt_simple.cli.status_cmd import _shutdown_flag_path

        flag = tmp_path / ".shutdown_requested"
        flag.write_text("")
        assert _shutdown_flag_path(str(tmp_path)) == flag

    def test_shutdown_flag_path_unset(self, tmp_path):
        from gpt_simple.cli.status_cmd import _shutdown_flag_path

        assert _shutdown_flag_path(str(tmp_path)) is None

    def test_status_picks_up_checkpoint_only_run(self, tmp_path, capsys):
        """Job crashed before writing .run_state.json; only checkpoint exists."""
        from gpt_simple.cli.main import main

        self._write_ckpt(tmp_path, step=42, tokens_trained=123_456)

        main(["status", "--output_dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "NO RUN STATE" in out or "checkpoint only" in out.lower()
        assert "42" in out
        # tokens_trained should be visible (formatted by _format_tokens)
        assert "123" in out or "checkpoint-42" in out

    def test_status_shows_history_row(self, tmp_path, capsys):
        from gpt_simple.cli.main import main
        from gpt_simple._run_state import RunState

        for step in (100, 200, 300):
            self._write_ckpt(tmp_path, step=step)

        RunState(
            status="running",
            pid=os.getpid(),
            global_step=300,
            max_steps=1000,
            loss=1.5,
        ).write(str(tmp_path))

        main(["status", "--output_dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "checkpoint-100" in out
        assert "checkpoint-200" in out
        assert "checkpoint-300" in out

    def test_status_shows_shutdown_requested_flag(self, tmp_path, capsys):
        from gpt_simple.cli.main import main
        from gpt_simple._run_state import RunState

        RunState(status="running", pid=os.getpid(), global_step=1, max_steps=10).write(
            str(tmp_path)
        )
        (tmp_path / ".shutdown_requested").write_text("")

        main(["status", "--output_dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "requested" in out.lower() or "shutdown" in out.lower()

    def test_status_no_run_state_no_checkpoint_exits(self, tmp_path):
        from gpt_simple.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["status", "--output_dir", str(tmp_path)])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# cli/train_cmd.py -- Phase D: --force flag
# ---------------------------------------------------------------------------


class TestForceFlag:
    """Tests for the ``--force`` clobber flag."""

    def _make_output_with_state(self, tmp_path: Path) -> Path:
        """Create an output_dir with a checkpoint + run state + shutdown flag."""
        from gpt_simple._checkpoint import TrainerState
        from gpt_simple._run_state import RunState

        out = tmp_path / "out"
        out.mkdir(parents=True)

        # Checkpoint
        ckpt = out / "checkpoints" / "checkpoint-100"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(
            json.dumps(TrainerState(step=100).to_dict())
        )

        # Run state
        RunState(status="stopped", pid=1, global_step=100, max_steps=1000).write(
            str(out)
        )

        # Shutdown flag
        (out / ".shutdown_requested").write_text("")

        # Tokenizer dir
        (out / "tokenizer").mkdir()
        (out / "tokenizer" / "tokenizer.json").write_text("{}")

        # User-owned thing that should NOT be deleted
        (out / "training.log").write_text("step 100 loss 1.5\n")

        return out

    def test_clobber_wipes_state_but_keeps_user_files(self, tmp_path):
        from gpt_simple.cli.train_cmd import _clobber_output_dir

        out = self._make_output_with_state(tmp_path)
        _clobber_output_dir(str(out))

        assert not (out / "checkpoints").exists()
        assert not (out / ".run_state.json").exists()
        assert not (out / ".shutdown_requested").exists()
        assert not (out / "tokenizer").exists()
        # Unrelated user files survive
        assert (out / "training.log").exists()

    def test_clobber_on_missing_dir_is_noop(self, tmp_path):
        from gpt_simple.cli.train_cmd import _clobber_output_dir

        _clobber_output_dir(str(tmp_path / "does_not_exist"))

    def test_force_flag_registered(self):
        """--force should be a known flag on `gpt-simple train`."""
        import argparse

        from gpt_simple.cli.train_cmd import TrainCommand

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        TrainCommand.register(subparsers)

        ns = parser.parse_args(
            ["train", "--config", "x.yaml", "--force"]
        )
        assert ns.force is True

    def test_force_overrides_resume_to_scratch(self, tmp_path):
        """In _launch_single, --force forces resume=scratch and wipes state."""
        yaml = pytest.importorskip("yaml")

        out = self._make_output_with_state(tmp_path)
        config = {
            "model": {"n_embd": 768, "n_layer": 12, "n_head": 12},
            "data": {"path": "/data", "tokenizer": "gpt2"},
            "optimizer": {"learning_rate": 3e-4, "warmup_steps": 10},
            "training": {
                "max_steps": 1000,
                "output_dir": str(out),
                "resume": "auto",  # we'll show --force overrides this
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        import argparse

        from gpt_simple.cli.train_cmd import _launch_single

        ns = argparse.Namespace(
            config=str(config_path),
            force=True,
            # The argparse-level None defaults we'd see in real usage.
            **{
                "training.max_steps": None,
                "training.output_dir": None,
                "training.resume": None,
                "training.keep_last_k": None,
                "training.max_runtime_seconds": None,
                "training.walltime_reserve_seconds": None,
                "training.seed": None,
                "training.wandb_project": None,
                "training.wandb_run_name": None,
                "optimizer.learning_rate": None,
                "data.path": None,
                "data.tokenizer": None,
                "data.format": None,
            },
        )

        # Patch train() on the gpt_simple.train *module* object. A plain
        # mock.patch("gpt_simple.train.train") fails because the package
        # attribute gpt_simple.train is the re-exported train() function, not
        # the submodule (so the dotted lookup finds the function and it has no
        # .train). Fetch the real module from sys.modules to disambiguate.
        import importlib

        train_module = importlib.import_module("gpt_simple.train")

        with mock.patch.object(train_module, "train") as mock_train:
            mock_train.return_value = mock.MagicMock(
                total_steps=1, final_loss=1.0
            )
            _launch_single(ns)

        # Side effects from _clobber_output_dir
        assert not (out / "checkpoints").exists()
        assert not (out / ".run_state.json").exists()
        assert not (out / ".shutdown_requested").exists()

        # train() was called with resume forced to scratch
        cfg_passed = mock_train.call_args.kwargs.get("config")
        if cfg_passed is None:
            cfg_passed = mock_train.call_args.args[0]
        assert cfg_passed.training.resume == "scratch"

    def test_force_injected_into_distributed_overrides(self, tmp_path):
        """In _launch_distributed, --force injects training.resume=scratch into the script_args."""
        yaml = pytest.importorskip("yaml")

        out = self._make_output_with_state(tmp_path)
        config = {
            "model": {"n_embd": 768, "n_layer": 12, "n_head": 12},
            "data": {"path": "/data", "tokenizer": "gpt2"},
            "optimizer": {"learning_rate": 3e-4, "warmup_steps": 10},
            "training": {
                "max_steps": 1000,
                "output_dir": str(out),
                "resume": "auto",
            },
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))

        import argparse

        from gpt_simple.cli.train_cmd import _launch_distributed

        ns = argparse.Namespace(
            config=str(config_path),
            force=True,
            nproc_per_node=2,
            nnodes=1,
            node_rank=0,
            master_addr="127.0.0.1",
            master_port="29500",
            **{
                "training.max_steps": None,
                "training.output_dir": None,
                "training.resume": None,
                "training.keep_last_k": None,
                "training.max_runtime_seconds": None,
                "training.walltime_reserve_seconds": None,
                "training.seed": None,
                "training.wandb_project": None,
                "training.wandb_run_name": None,
                "optimizer.learning_rate": None,
                "data.path": None,
                "data.tokenizer": None,
                "data.format": None,
            },
        )

        # Mock the torchrun entrypoint so we just inspect the argv it would
        # have been called with.
        with mock.patch(
            "gpt_simple.cli.train_cmd._ensure_hostname_resolves"
        ), mock.patch(
            "torch.distributed.run.run"
        ), mock.patch(
            "torch.distributed.run.get_args_parser"
        ) as mock_parser:
            parsed = mock.MagicMock()
            mock_parser.return_value.parse_args.return_value = parsed
            _launch_distributed(ns)

            # parse_args was called with the constructed argv list
            argv = mock_parser.return_value.parse_args.call_args.args[0]
            assert "--training.resume" in argv
            i = argv.index("--training.resume")
            assert argv[i + 1] == "scratch"

        # Clobber side effects happen on the launcher process.
        assert not (out / "checkpoints").exists()
        assert not (out / ".run_state.json").exists()


# ---------------------------------------------------------------------------
# cli/stop_cmd.py
# ---------------------------------------------------------------------------


class TestStopCommand:
    def _write_running_state(self, tmp_path, pid=None):
        from gpt_simple._run_state import RunState

        state = RunState(
            status="running",
            pid=pid or os.getpid(),
            started_at="2026-04-06T14:30:00+00:00",
            max_steps=1000,
        )
        state.write(str(tmp_path))
        return state

    def test_no_state_file_exits(self, tmp_path):
        from gpt_simple.cli.main import main

        with pytest.raises(SystemExit) as exc_info:
            main(["stop", "--output_dir", str(tmp_path)])
        assert exc_info.value.code == 1

    def test_dead_process_warns(self, tmp_path, caplog):
        self._write_running_state(tmp_path, pid=999999999)

        from gpt_simple.cli.main import main

        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            main(["stop", "--output_dir", str(tmp_path)])
        assert "not running" in caplog.text.lower() or "already exited" in caplog.text.lower()

    def test_graceful_creates_flag_file(self, tmp_path):
        """Verify that graceful stop creates the .shutdown_requested file."""
        self._write_running_state(tmp_path, pid=os.getpid())

        from gpt_simple.cli.stop_cmd import StopCommand

        import argparse
        args = argparse.Namespace(
            output_dir=str(tmp_path),
            force=False,
            verbose=False,
            quiet=False,
        )

        with mock.patch("gpt_simple.cli.stop_cmd._pid_alive", return_value=True):
            with mock.patch("os.kill"):
                with mock.patch("gpt_simple.cli.stop_cmd._pid_alive", side_effect=[True, False]):
                    with mock.patch("time.sleep"):
                        StopCommand.run(args)

        flag = tmp_path / ".shutdown_requested"
        assert flag.exists()

    def test_force_sends_sigkill(self, tmp_path):
        self._write_running_state(tmp_path, pid=12345)

        from gpt_simple.cli.stop_cmd import StopCommand

        import argparse
        args = argparse.Namespace(
            output_dir=str(tmp_path),
            force=True,
            verbose=False,
            quiet=False,
        )

        with mock.patch("gpt_simple.cli.stop_cmd._pid_alive", return_value=True):
            with mock.patch("os.kill") as mock_kill:
                StopCommand.run(args)

        mock_kill.assert_called_once_with(12345, signal.SIGKILL)

        from gpt_simple._run_state import RunState
        updated = RunState.read(str(tmp_path))
        assert updated.status == "stopped"


# ---------------------------------------------------------------------------
# cli/train_cmd.py -- config overrides
# ---------------------------------------------------------------------------


class TestTrainCommandOverrides:
    def _make_config_file(self, tmp_path) -> str:
        yaml = pytest.importorskip("yaml")
        config = {
            "model": {"n_embd": 768, "n_layer": 12, "n_head": 12},
            "data": {"path": "/data", "tokenizer": "gpt2"},
            "optimizer": {"learning_rate": 3e-4},
            "training": {"max_steps": 1000, "output_dir": str(tmp_path / "out")},
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(config))
        return str(p)

    def test_apply_overrides(self, tmp_path):
        from gpt_simple.config import Config
        from gpt_simple.cli.train_cmd import _apply_overrides, _collect_overrides

        config_path = self._make_config_file(tmp_path)
        cfg = Config.from_file(config_path)

        import argparse
        ns = argparse.Namespace(**{
            "training.max_steps": 5000,
            "training.output_dir": None,
            "training.resume": None,
            "training.keep_last_k": None,
            "training.seed": None,
            "training.wandb_project": None,
            "training.wandb_run_name": None,
            "optimizer.learning_rate": 1e-3,
            "data.path": "/new/data",
            "data.tokenizer": None,
            "data.format": None,
        })

        _apply_overrides(cfg, _collect_overrides(ns))

        assert cfg.training.max_steps == 5000
        assert cfg.optimizer.learning_rate == pytest.approx(1e-3)
        assert cfg.data.path == "/new/data"

    def test_single_vs_distributed_routing(self):
        import argparse
        from gpt_simple.cli.train_cmd import TrainCommand

        args_single = argparse.Namespace(nproc_per_node=1, nnodes=1)
        args_multi = argparse.Namespace(nproc_per_node=4, nnodes=1)

        with mock.patch("gpt_simple.cli.train_cmd._launch_single") as m_single:
            with mock.patch("gpt_simple.cli.train_cmd._launch_distributed") as m_dist:
                TrainCommand.run(args_single)
                m_single.assert_called_once()
                m_dist.assert_not_called()

        with mock.patch("gpt_simple.cli.train_cmd._launch_single") as m_single:
            with mock.patch("gpt_simple.cli.train_cmd._launch_distributed") as m_dist:
                TrainCommand.run(args_multi)
                m_dist.assert_called_once()
                m_single.assert_not_called()

    def test_multi_node_routes_to_distributed(self):
        import argparse
        from gpt_simple.cli.train_cmd import TrainCommand

        args = argparse.Namespace(nproc_per_node=1, nnodes=2)

        with mock.patch("gpt_simple.cli.train_cmd._launch_single") as m_single:
            with mock.patch("gpt_simple.cli.train_cmd._launch_distributed") as m_dist:
                TrainCommand.run(args)
                m_dist.assert_called_once()
                m_single.assert_not_called()


# ---------------------------------------------------------------------------
# cli/tokenize_cmd.py
# ---------------------------------------------------------------------------


class TestTokenizeCommand:
    def test_argv_forwarding(self):
        from gpt_simple.cli.tokenize_cmd import TokenizeCommand

        import argparse
        args = argparse.Namespace(
            input_dir="/in",
            output_dir="/out",
            tokenizer_path="gpt2",
            max_length=2048,
            overlap_size=256,
            probabilistic_overlap=True,
            overlap_probability=0.7,
            min_text_length=200,
            num_workers=4,
            verbose=False,
            quiet=False,
        )

        captured_argv = []

        def fake_main():
            captured_argv.extend(sys.argv)

        with mock.patch("gpt_simple.pretokenize.main", side_effect=fake_main):
            TokenizeCommand.run(args)

        assert "--input_dir" in captured_argv
        assert "/in" in captured_argv
        assert "--output_dir" in captured_argv
        assert "/out" in captured_argv
        assert "--probabilistic_overlap" in captured_argv
        assert "--tokenizer_path" in captured_argv

    def test_argv_restored_after_run(self):
        original = sys.argv.copy()

        from gpt_simple.cli.tokenize_cmd import TokenizeCommand

        import argparse
        args = argparse.Namespace(
            input_dir="/in",
            output_dir="/out",
            tokenizer_path="gpt2",
            max_length=2048,
            overlap_size=256,
            probabilistic_overlap=False,
            overlap_probability=0.7,
            min_text_length=200,
            num_workers=1,
            verbose=False,
            quiet=False,
        )

        with mock.patch("gpt_simple.pretokenize.main"):
            TokenizeCommand.run(args)

        assert sys.argv == original

    def test_argv_restored_on_error(self):
        original = sys.argv.copy()

        from gpt_simple.cli.tokenize_cmd import TokenizeCommand

        import argparse
        args = argparse.Namespace(
            input_dir="/in",
            output_dir="/out",
            tokenizer_path="gpt2",
            max_length=2048,
            overlap_size=256,
            probabilistic_overlap=False,
            overlap_probability=0.7,
            min_text_length=200,
            num_workers=1,
            verbose=False,
            quiet=False,
        )

        with mock.patch("gpt_simple.pretokenize.main", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                TokenizeCommand.run(args)

        assert sys.argv == original


# ---------------------------------------------------------------------------
# config.py -- typed exceptions
# ---------------------------------------------------------------------------


class TestConfigErrors:
    def test_invalid_n_embd_raises_config_error(self):
        from gpt_simple.config import ModelConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="n_embd"):
            ModelConfig(n_embd=100, n_head=12)

    def test_invalid_activation_raises_config_error(self):
        from gpt_simple.config import ModelConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="activation"):
            ModelConfig(activation="tanh")

    def test_invalid_data_format_raises_config_error(self):
        from gpt_simple.config import DataConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="format"):
            DataConfig(format="csv")


# ---------------------------------------------------------------------------
# CurriculumPhase validation
# ---------------------------------------------------------------------------


class TestCurriculumPhase:
    def test_valid_phase(self):
        from gpt_simple.config import CurriculumPhase

        phase = CurriculumPhase(duration_tokens=1_000_000, mix={"web": 0.6, "code": 0.4})
        assert phase.duration_tokens == 1_000_000
        assert phase.mix["web"] == pytest.approx(0.6)
        assert phase.mix["code"] == pytest.approx(0.4)

    def test_weights_are_normalised(self):
        from gpt_simple.config import CurriculumPhase

        phase = CurriculumPhase(duration_tokens=100, mix={"a": 3, "b": 7})
        assert phase.mix["a"] == pytest.approx(0.3)
        assert phase.mix["b"] == pytest.approx(0.7)

    def test_negative_duration_rejected(self):
        from gpt_simple.config import CurriculumPhase
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="duration_tokens"):
            CurriculumPhase(duration_tokens=-1, mix={"a": 1.0})

    def test_zero_duration_rejected(self):
        from gpt_simple.config import CurriculumPhase
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="duration_tokens"):
            CurriculumPhase(duration_tokens=0, mix={"a": 1.0})

    def test_empty_mix_rejected(self):
        from gpt_simple.config import CurriculumPhase
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="at least one bucket"):
            CurriculumPhase(duration_tokens=100, mix={})

    def test_all_zero_weights_rejected(self):
        from gpt_simple.config import CurriculumPhase
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="positive"):
            CurriculumPhase(duration_tokens=100, mix={"a": 0.0, "b": 0.0})

    def test_negative_weight_rejected(self):
        from gpt_simple.config import CurriculumPhase
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="non-negative"):
            CurriculumPhase(duration_tokens=100, mix={"a": 1.0, "b": -0.5})

    def test_curriculum_with_jsonl_rejected(self):
        from gpt_simple.config import CurriculumPhase, DataConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="pretokenized"):
            DataConfig(
                format="jsonl",
                curriculum=[CurriculumPhase(duration_tokens=100, mix={"a": 1.0})],
            )

    def test_from_file_with_inline_curriculum(self, tmp_path):
        yaml = pytest.importorskip("yaml")
        from gpt_simple.config import Config

        cfg_dict = {
            "data": {
                "path": "/data",
                "curriculum": [
                    {"duration_tokens": 1000, "mix": {"web": 0.8, "code": 0.2}},
                    {"duration_tokens": 2000, "mix": {"web": 0.5, "code": 0.5}},
                ],
            },
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg_dict))

        cfg = Config.from_file(p)
        assert cfg.data.curriculum is not None
        assert len(cfg.data.curriculum) == 2
        assert cfg.data.curriculum[0].duration_tokens == 1000
        assert cfg.data.curriculum[0].mix["web"] == pytest.approx(0.8)
        assert cfg.data.curriculum[1].mix["code"] == pytest.approx(0.5)

    def test_config_roundtrip_with_curriculum(self, tmp_path):
        from gpt_simple.config import Config, CurriculumPhase, DataConfig

        cfg = Config(
            data=DataConfig(
                path="/data",
                curriculum=[
                    CurriculumPhase(duration_tokens=500, mix={"a": 0.7, "b": 0.3}),
                ],
            ),
        )
        out = tmp_path / "config.json"
        cfg.save(out)

        loaded = json.loads(out.read_text())
        assert loaded["data"]["curriculum"][0]["duration_tokens"] == 500
        assert loaded["data"]["curriculum"][0]["mix"]["a"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# write_error_state in train.py
# ---------------------------------------------------------------------------


class TestWriteErrorState:
    def test_writes_error_from_exception(self, tmp_path):
        from gpt_simple._run_state import RunState, new_run_state

        state = new_run_state(max_steps=100)
        state.write(str(tmp_path))

        from gpt_simple.train import write_error_state

        try:
            raise RuntimeError("CUDA exploded")
        except RuntimeError as exc:
            write_error_state(str(tmp_path), exc)

        loaded = RunState.read(str(tmp_path))
        assert loaded.status == "error"
        assert "CUDA exploded" in loaded.error

    def test_writes_error_when_no_prior_state(self, tmp_path):
        from gpt_simple.train import write_error_state
        from gpt_simple._run_state import RunState

        try:
            raise ValueError("bad value")
        except ValueError as exc:
            write_error_state(str(tmp_path), exc)

        loaded = RunState.read(str(tmp_path))
        assert loaded.status == "error"
        assert "bad value" in loaded.error


# ---------------------------------------------------------------------------
# Config-level validations (checks 10-13)
# ---------------------------------------------------------------------------


class TestConfigValidations:
    def test_learning_rate_must_be_positive(self):
        from gpt_simple.config import OptimizerConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="learning_rate"):
            OptimizerConfig(learning_rate=0.0)
        with pytest.raises(ConfigError, match="learning_rate"):
            OptimizerConfig(learning_rate=-1e-4)

    def test_negative_warmup_steps_rejected(self):
        from gpt_simple.config import OptimizerConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="warmup_steps"):
            OptimizerConfig(warmup_steps=-1)

    def test_batch_size_must_be_positive(self):
        from gpt_simple.config import TrainingConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="per_device_batch_size"):
            TrainingConfig(per_device_batch_size=0)

    def test_gradient_accumulation_must_be_positive(self):
        from gpt_simple.config import TrainingConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="gradient_accumulation_steps"):
            TrainingConfig(gradient_accumulation_steps=0)

    def test_max_steps_must_be_positive(self):
        from gpt_simple.config import TrainingConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="max_steps"):
            TrainingConfig(max_steps=0)

    def test_logging_steps_must_be_positive(self):
        from gpt_simple.config import TrainingConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="logging_steps"):
            TrainingConfig(logging_steps=0)

    def test_logging_steps_gt_max_steps_warns(self, caplog):
        from gpt_simple.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            TrainingConfig(max_steps=10, logging_steps=100)
        assert "logging_steps" in caplog.text

    def test_eval_steps_gt_max_steps_warns(self, caplog):
        from gpt_simple.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            TrainingConfig(max_steps=10, eval_steps=100)
        assert "eval_steps" in caplog.text

    def test_save_steps_gt_max_steps_warns(self, caplog):
        from gpt_simple.config import TrainingConfig

        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            TrainingConfig(max_steps=10, save_steps=100)
        assert "save_steps" in caplog.text

    def test_warmup_ge_max_steps_errors(self):
        from gpt_simple.config import Config, OptimizerConfig, TrainingConfig
        from gpt_simple.errors import ConfigError

        with pytest.raises(ConfigError, match="warmup_steps"):
            Config(
                optimizer=OptimizerConfig(warmup_steps=1000),
                training=TrainingConfig(max_steps=1000),
            )
        with pytest.raises(ConfigError, match="warmup_steps"):
            Config(
                optimizer=OptimizerConfig(warmup_steps=2000),
                training=TrainingConfig(max_steps=1000),
            )

    def test_max_length_gt_n_positions_warns(self, caplog):
        from gpt_simple.config import Config, ModelConfig, DataConfig

        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            Config(
                model=ModelConfig(n_positions=1024),
                data=DataConfig(max_length=2048),
            )
        assert "max_length" in caplog.text

    def test_valid_config_no_warnings(self, caplog):
        from gpt_simple.config import Config

        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            Config()
        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_msgs) == 0


# ---------------------------------------------------------------------------
# Pre-flight checks (_preflight_checks in train.py)
# ---------------------------------------------------------------------------


class TestPreflightChecks:
    def _make_bucket(self, bucket_dir, tokens_per_shard=1000):
        """Write a real .bin/.idx pair the validator can introspect."""
        import numpy as np
        from gpt_simple.pretokenize import write_idx
        bucket_dir.mkdir(parents=True, exist_ok=True)
        bin_path = bucket_dir / "shard_000.bin"
        idx_path = bucket_dir / "shard_000.idx"
        np.zeros(tokens_per_shard, dtype=np.uint16).tofile(str(bin_path))
        offsets = np.array([0, tokens_per_shard], dtype=np.int64)
        overlap = np.array([0], dtype=np.uint16)
        write_idx(idx_path, offsets, overlap, dtype_code=2)

    def _make_config(self, tmp_path, **overrides):
        """Build a Config pointing at a valid (minimal) data directory."""
        from gpt_simple.config import Config, DataConfig, TrainingConfig

        data_dir = tmp_path / "data"
        # The validator requires train/<bucket>/*.bin and val/<bucket>/*.bin.
        self._make_bucket(data_dir / "train" / "default")
        self._make_bucket(data_dir / "val" / "default")

        cfg = Config(
            data=DataConfig(path=str(data_dir), num_workers=0),
            training=TrainingConfig(
                output_dir=str(tmp_path / "out"),
                max_steps=10,
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
            ),
            optimizer=__import__("gpt_simple.config", fromlist=["OptimizerConfig"]).OptimizerConfig(
                warmup_steps=1,
            ),
        )
        cfg.data.max_length = 128
        for key, val in overrides.items():
            parts = key.split(".")
            if len(parts) == 2:
                setattr(getattr(cfg, parts[0]), parts[1], val)
            else:
                setattr(cfg, key, val)
        return cfg

    def test_missing_data_path_raises(self, tmp_path):
        from gpt_simple.config import Config, DataConfig, TrainingConfig
        from gpt_simple.train import _preflight_checks
        from gpt_simple.errors import DataError

        cfg = Config(
            data=DataConfig(path="/nonexistent/path"),
            training=TrainingConfig(output_dir=str(tmp_path / "out")),
        )
        with pytest.raises(DataError, match="does not exist"):
            _preflight_checks(cfg)

    def test_no_bin_shards_raises(self, tmp_path):
        from gpt_simple.config import Config, DataConfig, TrainingConfig
        from gpt_simple.train import _preflight_checks
        from gpt_simple.errors import DataError

        empty_data = tmp_path / "empty_data"
        empty_data.mkdir()

        cfg = Config(
            data=DataConfig(path=str(empty_data), format="pretokenized"),
            training=TrainingConfig(output_dir=str(tmp_path / "out")),
        )
        # Validator surfaces this as a generic "Validation failed" DataError;
        # the underlying message names the missing train/ dir.
        with pytest.raises(DataError, match="Validation failed"):
            _preflight_checks(cfg)

    def test_jsonl_format_with_proper_layout_passes(self, tmp_path):
        """JSONL needs train/<bucket>/*.jsonl and val/<bucket>/*.jsonl, same
        as pretokenized; the bin-existence check no longer fires."""
        from gpt_simple.config import (
            Config, DataConfig, OptimizerConfig, TrainingConfig,
        )
        from gpt_simple.train import _preflight_checks

        data_dir = tmp_path / "jsonl_data"
        for split in ("train", "val"):
            sub = data_dir / split / "default"
            sub.mkdir(parents=True)
            (sub / "data.jsonl").write_text('{"text":"hello"}\n')

        cfg = Config(
            data=DataConfig(path=str(data_dir), format="jsonl", num_workers=0),
            optimizer=OptimizerConfig(warmup_steps=1),
            training=TrainingConfig(
                output_dir=str(tmp_path / "out"),
                max_steps=10,
                per_device_batch_size=1,
                gradient_accumulation_steps=1,
            ),
        )
        cfg.data.max_length = 128
        cfg.data.allow_budget_mismatch = True
        # Should not raise.
        _preflight_checks(cfg)

    def test_resume_scratch_with_existing_checkpoints_raises(self, tmp_path):
        from gpt_simple._checkpoint import CheckpointManager, TrainerState
        from gpt_simple.errors import CheckpointError

        cfg = self._make_config(tmp_path)
        cfg.training.resume = "scratch"

        ckpt = Path(cfg.training.output_dir) / "checkpoints" / "checkpoint-100"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(json.dumps(TrainerState(step=100).to_dict()))

        mgr = CheckpointManager(cfg.training.output_dir)
        with pytest.raises(CheckpointError, match="already contains checkpoints"):
            mgr.assert_can_train_from_scratch()

    def test_resume_auto_with_no_checkpoints_starts_fresh(self, tmp_path):
        from gpt_simple._checkpoint import CheckpointManager

        cfg = self._make_config(tmp_path)
        cfg.training.resume = "auto"
        mgr = CheckpointManager(cfg.training.output_dir)
        assert mgr.resolve_resume("auto") is None

    def test_resume_auto_with_checkpoints_picks_latest(self, tmp_path):
        from gpt_simple._checkpoint import CheckpointManager, TrainerState

        cfg = self._make_config(tmp_path)
        out = Path(cfg.training.output_dir)
        for step in (100, 500, 300):
            ckpt = out / "checkpoints" / f"checkpoint-{step}"
            ckpt.mkdir(parents=True)
            (ckpt / "trainer_state.json").write_text(json.dumps(TrainerState(step=step).to_dict()))

        mgr = CheckpointManager(cfg.training.output_dir)
        resolved = mgr.resolve_resume("auto")
        assert resolved is not None
        assert resolved.name == "checkpoint-500"

    def test_stale_running_state_dead_pid_warns(self, tmp_path, caplog):
        from gpt_simple._run_state import RunState
        from gpt_simple.train import _preflight_checks

        cfg = self._make_config(tmp_path)
        out = Path(cfg.training.output_dir)
        out.mkdir(parents=True)

        state = RunState(status="running", pid=999999999, max_steps=100)
        state.write(str(out))

        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            _preflight_checks(cfg)
        assert "crashed" in caplog.text.lower() or "dead" in caplog.text.lower()

    def test_stale_running_state_live_pid_errors(self, tmp_path):
        from gpt_simple._run_state import RunState
        from gpt_simple.train import _preflight_checks
        from gpt_simple.errors import GptSimpleError

        cfg = self._make_config(tmp_path)
        out = Path(cfg.training.output_dir)
        out.mkdir(parents=True)

        state = RunState(status="running", pid=os.getpid(), max_steps=100)
        state.write(str(out))

        with pytest.raises(GptSimpleError, match="Another training job"):
            _preflight_checks(cfg)

    def test_completed_state_no_error(self, tmp_path):
        from gpt_simple._run_state import RunState
        from gpt_simple.train import _preflight_checks

        cfg = self._make_config(tmp_path)
        out = Path(cfg.training.output_dir)
        out.mkdir(parents=True)

        state = RunState(status="completed", pid=1, max_steps=100)
        state.write(str(out))

        _preflight_checks(cfg)

    def test_wandb_login_check_warns(self, tmp_path, caplog):
        from gpt_simple.train import _preflight_checks

        cfg = self._make_config(tmp_path)
        cfg.training.wandb_project = "test-project"

        mock_api = mock.MagicMock()
        mock_api.api_key = None
        mock_wandb = mock.MagicMock()
        mock_wandb.api = mock_api

        with mock.patch.dict(sys.modules, {"wandb": mock_wandb}):
            with caplog.at_level(logging.WARNING, logger="gpt_simple"):
                _preflight_checks(cfg)
        assert "api key" in caplog.text.lower() or "wandb login" in caplog.text.lower()

    def test_cuda_not_available_warns(self, tmp_path, caplog):
        """CUDA checks moved to _runtime_preflight; verify it logs a warning."""
        from gpt_simple.train import _runtime_preflight

        cfg = self._make_config(tmp_path)

        # Build a minimal accelerator stand-in.  We force is_main_process=True
        # and short-circuit the all_reduce barrier by leaving torch.distributed
        # uninitialized; the function falls through.
        class _FakeAcc:
            is_main_process = True
            device = "cpu"

        with mock.patch("torch.cuda.is_available", return_value=False):
            with caplog.at_level(logging.INFO, logger="gpt_simple"):
                _runtime_preflight(cfg, _FakeAcc(), skip_runtime_probe=True)
        assert "cuda" in caplog.text.lower() or "cpu" in caplog.text.lower()

    def test_resolve_mixed_precision_explicit(self):
        from gpt_simple.train import _resolve_mixed_precision

        for choice in ("bf16", "fp16", "no"):
            assert _resolve_mixed_precision(choice) == choice

    def test_resolve_mixed_precision_no_cuda(self):
        from gpt_simple.train import _resolve_mixed_precision

        with mock.patch("torch.cuda.is_available", return_value=False):
            assert _resolve_mixed_precision(None) == "no"

    def test_resolve_mixed_precision_ampere_picks_bf16(self):
        from gpt_simple.train import _resolve_mixed_precision

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
                assert _resolve_mixed_precision(None) == "bf16"

    def test_resolve_mixed_precision_volta_picks_fp16(self):
        from gpt_simple.train import _resolve_mixed_precision

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
                assert _resolve_mixed_precision(None) == "fp16"

    def test_bf16_auto_fallback_handled_by_mixed_precision_resolver(self):
        """When mixed_precision=None and bf16 isn't supported, the resolver
        picks fp16.  This is now the single source of truth; _runtime_preflight
        only reports — it does not mutate config."""
        from gpt_simple.train import _resolve_mixed_precision

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
                assert _resolve_mixed_precision(None) == "fp16"

    def test_explicit_bf16_on_unsupported_gpu_runtime_error(self, tmp_path):
        """Explicit bf16 on a non-Ampere GPU triggers a runtime ConfigError."""
        from gpt_simple.train import _runtime_preflight
        from gpt_simple.errors import ConfigError

        cfg = self._make_config(tmp_path, **{"training.mixed_precision": "bf16"})

        class _FakeAcc:
            is_main_process = True
            device = "cpu"

        with mock.patch("torch.cuda.is_available", return_value=True):
            with mock.patch("torch.cuda.get_device_name", return_value="Tesla V100"):
                with mock.patch("torch.cuda.device_count", return_value=1):
                    with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
                        with pytest.raises(ConfigError, match="Runtime preflight failed"):
                            _runtime_preflight(cfg, _FakeAcc(), skip_runtime_probe=True)
