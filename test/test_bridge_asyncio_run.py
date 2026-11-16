"""Regression test: ``asyncio.run`` must not be called with a running loop.

Before this fix, ``Bridge._scatter_full`` and the single-bridge publish path called ``asyncio.run(...)`` directly on the
no-client path. ``asyncio.run`` raises ``RuntimeError: asyncio.run() cannot be called from a running event loop`` when
the calling thread already has a running loop, which happens during bridge teardown.
That surfaced as ``ERROR:TestSimulator:Error while closing bridges: asyncio.run() cannot be called from a running event
loop`` and left the ``test_register_callback`` family reporting spurious teardown errors.

These tests exercise the helpers directly and through the real code paths, so a
regression fails here instead of surfacing as flaky teardown noise.
"""

import asyncio

import numpy as np
import pytest

from deisa.dask.bridge import Bridge
from deisa.dask.utils import run_coro_on_private_loop


async def _noop():
    return "done"


def test_run_async_without_a_running_loop_uses_asyncio_run():
    """No loop on this thread: the helper takes the cheap asyncio.run path."""
    bridge = Bridge.__new__(Bridge)  # no __init__: no cluster, no comm
    assert bridge._run_async(_noop()) == "done"


def test_run_async_with_a_running_loop_does_not_raise():
    """A loop IS running: asyncio.run would raise. The helper must still work."""
    bridge = Bridge.__new__(Bridge)
    result = {}

    async def main():
        # Deliberately call the SYNCHRONOUS helper from inside a running loop.
        # This is the situation that used to raise RuntimeError.
        result["value"] = bridge._run_async(_noop())

    asyncio.run(main())
    assert result["value"] == "done"


def test_scatter_full_no_client_path_under_a_running_loop():
    """The real call site: ``_scatter_full`` must survive a running loop.

    ``_scatter_full`` is what bridge teardown reaches on the no-client path, so it is where ``asyncio.run`` used to
    raise. Covering only ``_run_async`` is not enough: a change that fixed the helper while leaving a call site on bare
    ``asyncio.run`` would still pass and still fail in production.
    """
    bridge = Bridge.__new__(Bridge)  # no __init__: no cluster, no comm
    bridge.client = None  # force the no-client branch
    bridge.id = "test-bridge"  # _scatter_full logs f"[{self.id}]"
    bridge.workers = ["w-0"]  # workers=None path reads this

    captured = {}

    async def fake_scatter_payload(workers, payload):
        captured["called"] = True
        return {k: list(workers) for k in payload}, {}

    bridge._scatter_payload = fake_scatter_payload

    result = {}

    async def main():
        # Synchronous bridge call from inside a running loop: the failure condition.
        result["value"] = bridge._scatter_full(np.zeros(4), workers=None)

    asyncio.run(main())
    assert captured.get("called") is True, "the no-client scatter path was not taken"
    assert len(result["value"]["future"]) == 1


def test_scatter_partials_no_client_path_under_a_running_loop():
    """Second real call site: ``_scatter_partials`` must survive a running loop.

    ``_scatter_partials`` also scattered via bare ``asyncio.run`` on the no-client path (the precomputed-partials send
    used by the mergeable reductions). Covering only ``_scatter_full`` leaves this site unprotected: a change that
    reintroduced ``asyncio.run`` here would pass the other tests and still fail in production.
    """
    from deisa.dask.branch import BranchSpec

    bridge = Bridge.__new__(Bridge)  # no __init__: no cluster, no comm
    bridge.client = None  # force the no-client branch
    bridge.id = "test-bridge"
    bridge._branch_by_key = {}  # cache dict built in __init__
    bridge.arrays_metadata = {"a": {"chunk_position": (0,)}}

    captured = {}

    def fake_get_branch_by_key(array_name, branches):
        return {b.output_key: b for b in branches}

    bridge._get_branch_by_key = fake_get_branch_by_key

    async def fake_scatter_payload(target_worker, payload):
        captured["called"] = True
        return {k: ["who"] for k in payload}, {}

    bridge._scatter_payload = fake_scatter_payload

    branch = BranchSpec(
        output_key="f-sum",
        input_name="a",
        output_kind="scalar",
        branch_func=lambda chunk: float(chunk.sum()),
        chunk_axis=(0, 1),
        finalize=None,
        partial_shape=(),
        partial_dtype="float64",
    )

    result = {}

    async def main():
        # Synchronous bridge call from inside a running loop: the failure condition.
        result["value"] = bridge._scatter_partials({"f-sum": 3.0}, [branch], "a", ["w-0"])

    asyncio.run(main())
    assert captured.get("called") is True, "the no-client partials path was not taken"
    # The scattered keys are namespaced (KEY_PREFIX + array + output_key + uuid), so assert on membership rather than
    # exact equality.
    scatter_result, futures_entries = result["value"]
    future_keys = scatter_result["future"]
    assert len(future_keys) == 1 and "f-sum" in future_keys[0]
    assert futures_entries[0]["future"] == future_keys[0]
    assert futures_entries[0]["chunk_position"] == (0,)


def test_private_loop_helper_propagates_exceptions():
    """An exception inside the coroutine must surface on the caller, not vanish."""

    async def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_coro_on_private_loop(boom())


def test_private_loop_helper_returns_value():
    """Sanity check on the private-loop path outside a running loop too."""

    async def compute():
        await asyncio.sleep(0)
        return 42

    assert run_coro_on_private_loop(compute()) == 42
