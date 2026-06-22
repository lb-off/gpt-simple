"""
Unit tests for ``gpt_simple._shutdown.ShutdownCoordinator``.

Covers:
  - signal handler installation/uninstallation (round-trip with prior handlers)
  - signal-based shutdown sets the flag and reason
  - double-SIGINT triggers ``os._exit``
  - walltime watchdog (explicit ``max_runtime_seconds`` + ``SLURM_JOB_END_TIME``)
  - flag file detection
  - programmatic ``request_shutdown(reason)``
  - ``clear_flag_file`` is a no-op when missing
  - ``remaining_seconds`` reporting
  - ``should_shutdown`` is idempotent / sticky once triggered

Distributed all-reduce coordination is exercised by the multi-rank
integration tests in Phase E; here we cover the single-process path.
"""

from __future__ import annotations

import os
import signal
import time

import pytest

from gpt_simple._shutdown import ShutdownCoordinator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coord(tmp_path):
    """A ShutdownCoordinator with no walltime, no flag file polling override."""
    return ShutdownCoordinator(
        accelerator=None,
        output_dir=tmp_path,
    )


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """Save and restore SIGTERM/SIGINT/SIGUSR1/SIGUSR2 around each test.

    Some tests install handlers and forget to uninstall; this guards the
    next test from inheriting them.
    """
    saved = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2)
    }
    yield
    for sig, handler in saved.items():
        try:
            signal.signal(sig, handler)
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Default behaviour: no triggers ⇒ no shutdown
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_should_not_shutdown_without_any_trigger(self, coord):
        assert coord.should_shutdown() is False
        assert coord.reason is None

    def test_no_walltime_when_no_input(self, coord):
        assert coord.deadline_monotonic is None
        assert coord.remaining_seconds() is None

    def test_walltime_logged_when_armed(self, tmp_path, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="gpt_simple"):
            ShutdownCoordinator(
                accelerator=None,
                output_dir=tmp_path,
                max_runtime_seconds=60,
            )
        assert "Walltime watchdog armed" in caplog.text


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    def test_install_then_uninstall_restores_prior_handlers(self, coord):
        sentinel = signal.getsignal(signal.SIGTERM)
        coord.install_signal_handlers()
        # Bound methods don't satisfy `is` but compare equal.
        assert signal.getsignal(signal.SIGTERM) == coord._handle_signal
        coord.uninstall_signal_handlers()
        assert signal.getsignal(signal.SIGTERM) is sentinel

    def test_sigterm_sets_flag_and_reason(self, coord):
        coord._handle_signal(signal.SIGTERM, None)
        assert coord.should_shutdown() is True
        assert coord.reason == "signal:SIGTERM"

    def test_sigusr1_sets_flag_and_reason(self, coord):
        coord._handle_signal(signal.SIGUSR1, None)
        assert coord.should_shutdown() is True
        assert coord.reason == "signal:SIGUSR1"

    def test_sigusr2_sets_flag_and_reason(self, coord):
        coord._handle_signal(signal.SIGUSR2, None)
        assert coord.should_shutdown() is True
        assert coord.reason == "signal:SIGUSR2"

    def test_first_sigint_is_graceful(self, coord, monkeypatch):
        exit_calls: list[int] = []
        monkeypatch.setattr(os, "_exit", lambda code: exit_calls.append(code))
        coord._handle_signal(signal.SIGINT, None)
        assert exit_calls == []
        assert coord.should_shutdown() is True
        assert coord.reason == "signal:SIGINT"

    def test_second_sigint_forces_exit(self, coord, monkeypatch):
        exit_calls: list[int] = []
        monkeypatch.setattr(os, "_exit", lambda code: exit_calls.append(code))
        coord._handle_signal(signal.SIGINT, None)
        coord._handle_signal(signal.SIGINT, None)
        assert exit_calls == [130]

    def test_repeated_signals_do_not_change_reason(self, coord):
        coord._handle_signal(signal.SIGTERM, None)
        coord._handle_signal(signal.SIGUSR1, None)
        # First-wins
        assert coord.reason == "signal:SIGTERM"


# ---------------------------------------------------------------------------
# Walltime watchdog
# ---------------------------------------------------------------------------


class TestWalltime:
    def test_explicit_deadline_in_past_triggers(self, tmp_path):
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=1,
            walltime_reserve_seconds=0,
            loop_start_monotonic=time.monotonic() - 100,
        )
        assert coord.should_shutdown() is True
        assert coord.reason == "walltime"

    def test_explicit_deadline_in_future_does_not_trigger(self, tmp_path):
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=3600,
            walltime_reserve_seconds=300,
        )
        assert coord.should_shutdown() is False
        assert coord.reason is None

    def test_reserve_eats_into_budget(self, tmp_path):
        # budget = 10s, reserve = 30s ⇒ threshold is 20s in the past
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=10,
            walltime_reserve_seconds=30,
        )
        assert coord.should_shutdown() is True

    def test_slurm_env_var_used(self, tmp_path, monkeypatch):
        # Job ended 5 seconds ago
        monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) - 5))
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            walltime_reserve_seconds=0,
        )
        assert coord.should_shutdown() is True
        assert coord.reason == "walltime"

    def test_slurm_env_var_in_future_no_trigger(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 3600))
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            walltime_reserve_seconds=10,
        )
        assert coord.should_shutdown() is False

    def test_invalid_slurm_env_var_disables_watchdog(self, tmp_path, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("SLURM_JOB_END_TIME", "not-a-number")
        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            coord = ShutdownCoordinator(
                accelerator=None,
                output_dir=tmp_path,
            )
        assert coord.deadline_monotonic is None
        assert "SLURM_JOB_END_TIME" in caplog.text

    def test_explicit_overrides_slurm_env(self, tmp_path, monkeypatch):
        # SLURM says 1h, but explicit says 1s — explicit wins
        monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 3600))
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=1,
            walltime_reserve_seconds=0,
            loop_start_monotonic=time.monotonic() - 100,
        )
        assert coord.should_shutdown() is True

    def test_generic_env_var_used(self, tmp_path, monkeypatch):
        """GPT_SIMPLE_MAX_RUNTIME is honoured when no SLURM_JOB_END_TIME / explicit value."""
        monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
        monkeypatch.setenv("GPT_SIMPLE_MAX_RUNTIME", "1")
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            walltime_reserve_seconds=0,
            loop_start_monotonic=time.monotonic() - 100,
        )
        assert coord.should_shutdown() is True
        assert coord.reason == "walltime"

    def test_slurm_env_var_takes_precedence_over_generic(self, tmp_path, monkeypatch):
        """SLURM_JOB_END_TIME wins over GPT_SIMPLE_MAX_RUNTIME (cluster signal trusted)."""
        monkeypatch.setenv("SLURM_JOB_END_TIME", str(int(time.time()) + 3600))
        monkeypatch.setenv("GPT_SIMPLE_MAX_RUNTIME", "1")
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            walltime_reserve_seconds=10,
            loop_start_monotonic=time.monotonic(),
        )
        assert coord.should_shutdown() is False
        assert coord.remaining_seconds() > 600

    def test_invalid_generic_env_var_disables_watchdog(self, tmp_path, monkeypatch, caplog):
        import logging
        monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
        monkeypatch.setenv("GPT_SIMPLE_MAX_RUNTIME", "not-a-number")
        with caplog.at_level(logging.WARNING, logger="gpt_simple"):
            coord = ShutdownCoordinator(
                accelerator=None,
                output_dir=tmp_path,
            )
        assert coord.deadline_monotonic is None
        assert "GPT_SIMPLE_MAX_RUNTIME" in caplog.text

    def test_remaining_seconds_reports_positive_budget(self, tmp_path):
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=600,
        )
        rem = coord.remaining_seconds()
        assert rem is not None
        assert 590 < rem <= 600

    def test_remaining_seconds_clamps_at_zero(self, tmp_path):
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=1,
            loop_start_monotonic=time.monotonic() - 100,
        )
        assert coord.remaining_seconds() == 0.0


# ---------------------------------------------------------------------------
# Flag file
# ---------------------------------------------------------------------------


class TestFlagFile:
    def test_present_triggers(self, tmp_path):
        (tmp_path / ".shutdown_requested").touch()
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
        )
        assert coord.should_shutdown() is True
        assert coord.reason == "flag_file"

    def test_absent_no_trigger(self, coord):
        assert coord.should_shutdown() is False

    def test_disable_check_ignores_file(self, tmp_path):
        (tmp_path / ".shutdown_requested").touch()
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            check_flag_file=False,
        )
        assert coord.should_shutdown() is False

    def test_clear_when_present(self, tmp_path):
        flag = tmp_path / ".shutdown_requested"
        flag.touch()
        coord = ShutdownCoordinator(accelerator=None, output_dir=tmp_path)
        coord.clear_flag_file()
        assert not flag.exists()

    def test_clear_when_absent_is_noop(self, coord):
        coord.clear_flag_file()  # must not raise


# ---------------------------------------------------------------------------
# Programmatic shutdown
# ---------------------------------------------------------------------------


class TestRequestShutdown:
    def test_request_sets_flag_and_reason(self, coord):
        coord.request_shutdown("bucket_exhausted:web")
        assert coord.should_shutdown() is True
        assert coord.reason == "bucket_exhausted:web"

    def test_request_is_idempotent(self, coord):
        coord.request_shutdown("first_reason")
        coord.request_shutdown("second_reason")
        assert coord.reason == "first_reason"

    def test_should_shutdown_is_sticky(self, coord):
        coord.request_shutdown("user")
        assert coord.should_shutdown() is True
        # Subsequent checks must still report True (even though no new trigger)
        assert coord.should_shutdown() is True
        assert coord.should_shutdown() is True


# ---------------------------------------------------------------------------
# Priority: signal beats walltime if it arrives first
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    def test_signal_then_walltime_keeps_signal_reason(self, tmp_path):
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=1,
            walltime_reserve_seconds=0,
            loop_start_monotonic=time.monotonic() - 100,
        )
        coord._handle_signal(signal.SIGTERM, None)
        coord.should_shutdown()
        assert coord.reason == "signal:SIGTERM"

    def test_walltime_then_signal_keeps_walltime_reason(self, tmp_path):
        coord = ShutdownCoordinator(
            accelerator=None,
            output_dir=tmp_path,
            max_runtime_seconds=1,
            walltime_reserve_seconds=0,
            loop_start_monotonic=time.monotonic() - 100,
        )
        coord.should_shutdown()  # consumes walltime
        coord._handle_signal(signal.SIGTERM, None)
        assert coord.reason == "walltime"
