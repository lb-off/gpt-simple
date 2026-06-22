"""
Unit tests for the bucket-exhaustion policy in ``gpt_simple._train_loop``.

When a bucket runs dry mid-run, ``data.allow_bucket_exhaustion`` decides what
happens:
  - ``True``  -> the loader drops the bucket and renormalises the mix; training
    continues.
  - ``False`` (default) -> the trainer halts with a checkpoint rather than
    silently changing the mix, writing run status ``"halted"`` so the resume
    chain stops resubmitting.

The training loop is monolithic (only exercised end-to-end elsewhere), so the
decision is factored into pure helpers that are tested directly here, plus one
integration check that the helpers drive a real ``ShutdownCoordinator`` the way
the loop does.
"""

from __future__ import annotations

import pytest

from gpt_simple._shutdown import ShutdownCoordinator
from gpt_simple._train_loop import (
    _bucket_for_code,
    _local_bucket_code,
    _shutdown_status,
)


# A stable name<->id mapping like the one ``data.build_bucket_mappings`` builds
# from the sorted bucket names.
BUCKET_TO_ID = {"code": 0, "math": 1, "qa": 2, "web": 3, "wiki": 4}
ID_TO_BUCKET = {v: k for k, v in BUCKET_TO_ID.items()}


# ---------------------------------------------------------------------------
# _local_bucket_code: encode newly-exhausted buckets for an all_reduce(MAX)
# ---------------------------------------------------------------------------


class TestLocalBucketCode:
    def test_none_exhausted_is_zero(self):
        # 0 is reserved for "nothing exhausted on this rank".
        assert _local_bucket_code([], BUCKET_TO_ID) == 0

    def test_single_bucket_is_id_plus_one(self):
        assert _local_bucket_code(["code"], BUCKET_TO_ID) == 1   # id 0 + 1
        assert _local_bucket_code(["wiki"], BUCKET_TO_ID) == 5   # id 4 + 1

    def test_multiple_uses_max_id(self):
        # A rank that has lost several buckets reports the highest id (+1);
        # the all_reduce MAX then yields a representative bucket name.
        assert _local_bucket_code(["code", "qa"], BUCKET_TO_ID) == 3  # max(0,2)+1

    def test_unknown_bucket_ignored(self):
        # An unmapped name contributes -1, so on its own it stays "none".
        assert _local_bucket_code(["nope"], BUCKET_TO_ID) == 0
        # ...but a known bucket alongside it still reports.
        assert _local_bucket_code(["nope", "math"], BUCKET_TO_ID) == 2  # id 1 + 1


# ---------------------------------------------------------------------------
# _bucket_for_code: decode the reduced code back to a bucket name
# ---------------------------------------------------------------------------


class TestBucketForCode:
    def test_zero_means_no_exhaustion(self):
        assert _bucket_for_code(0, ID_TO_BUCKET) is None

    def test_negative_means_no_exhaustion(self):
        assert _bucket_for_code(-1, ID_TO_BUCKET) is None

    def test_decodes_known_code(self):
        assert _bucket_for_code(1, ID_TO_BUCKET) == "code"
        assert _bucket_for_code(5, ID_TO_BUCKET) == "wiki"

    def test_out_of_range_code_is_unknown_not_error(self):
        # The halt should still fire even if the id map is somehow stale.
        assert _bucket_for_code(99, ID_TO_BUCKET) == "<unknown>"

    @pytest.mark.parametrize("bucket", list(BUCKET_TO_ID))
    def test_roundtrip(self, bucket):
        code = _local_bucket_code([bucket], BUCKET_TO_ID)
        assert _bucket_for_code(code, ID_TO_BUCKET) == bucket


# ---------------------------------------------------------------------------
# _shutdown_status: bucket exhaustion is terminal ("halted"), others "stopped"
# ---------------------------------------------------------------------------


class TestShutdownStatus:
    def test_bucket_exhaustion_is_halted(self):
        assert _shutdown_status("bucket_exhausted:wiki") == "halted"
        assert _shutdown_status("bucket_exhausted:<unknown>") == "halted"

    @pytest.mark.parametrize("reason", ["walltime", "signal:SIGUSR1", "flag_file", ""])
    def test_other_reasons_are_stopped(self, reason):
        assert _shutdown_status(reason) == "stopped"

    def test_none_reason_is_stopped(self):
        assert _shutdown_status(None) == "stopped"


# ---------------------------------------------------------------------------
# Integration: the helpers drive a real ShutdownCoordinator like the loop does
# ---------------------------------------------------------------------------


def _enact_policy(newly_exhausted, *, allow, shutdown):
    """Mirror the loop's per-step decision (minus the cross-rank all-reduce,
    which is a MAX no-op in a single process).  Returns the halted bucket name
    when the step should be skipped, else None."""
    if allow:
        return None
    code = _local_bucket_code(newly_exhausted, BUCKET_TO_ID)
    rep = _bucket_for_code(code, ID_TO_BUCKET)
    if rep is not None:
        shutdown.request_shutdown(f"bucket_exhausted:{rep}")
    return rep


class TestPolicyIntegration:
    def test_halt_requests_shutdown_with_halted_status(self, tmp_path):
        coord = ShutdownCoordinator(accelerator=None, output_dir=tmp_path)

        halted = _enact_policy(["wiki"], allow=False, shutdown=coord)

        assert halted == "wiki"
        assert coord.should_shutdown() is True
        assert coord.reason == "bucket_exhausted:wiki"
        # This is what the loop writes to run_state and the chain greps for.
        assert _shutdown_status(coord.reason) == "halted"

    def test_renormalise_does_not_shut_down(self, tmp_path):
        coord = ShutdownCoordinator(accelerator=None, output_dir=tmp_path)

        halted = _enact_policy(["wiki"], allow=True, shutdown=coord)

        assert halted is None
        assert coord.should_shutdown() is False

    def test_no_exhaustion_does_not_shut_down(self, tmp_path):
        coord = ShutdownCoordinator(accelerator=None, output_dir=tmp_path)

        halted = _enact_policy([], allow=False, shutdown=coord)

        assert halted is None
        assert coord.should_shutdown() is False
