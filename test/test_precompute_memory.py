# =============================================================================
# Memory-measurement tests for the precompute feature.
#
# Goal: prove that when a callback is registered and the analyzer detects
# reducible operations, the *full* array chunk never lands on a worker. Only
# the small per-bridge reduction partials should appear in worker memory.
# The full chunk stays on the bridge process (which is the simulator here).
#
# We use Dask's built-in memory counters (worker.data + nbytes) instead of
# psutil so the test has no external dependency. The test runs against a
# LocalCluster so worker memory is observable from the test process.
# =============================================================================
import logging
import os
import textwrap
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pytest
from deisa.core.types import DeisaArray
from distributed import Client, LocalCluster
from TestSimulator import TestSimulation
from utils import wait_for

from deisa.dask import Deisa

logging.basicConfig(level=logging.DEBUG)

# Skip on the github-only windowless context
pytestmark = pytest.mark.timeout(60)


def _worker_bytes_per_key(client: Client) -> Dict[str, Dict[str, Any]]:
    """Return, per worker, a dict ``{key: nbytes}`` for every key currently
    resident in that worker's in-memory data store. Includes zeros -- only in-memory keys are visible; spilled-to-disk
    keys are ignored on purpose so we capture what is actually consuming RAM.
    """

    def inspect(dask_worker):
        out = {}
        for k, v in dask_worker.data.items():
            if hasattr(v, "nbytes"):
                out[k] = v.nbytes
            else:
                # numpy scalars / dicts / etc.
                out[k] = None
        return out

    return client.run(inspect)


def _total_bytes_per_worker(per_worker: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Sum nbytes per worker (None entries contribute 0)."""
    out = {}
    for worker, keymap in per_worker.items():
        out[worker] = sum((v or 0) for v in keymap.values())
    return out


def _largest_key_per_worker(per_worker: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """Max nbytes per worker (None entries contribute 0)."""
    out = {}
    for worker, keymap in per_worker.items():
        out[worker] = max((v or 0) for v in keymap.values()) if keymap else 0
    return out


def _make_callback(op: str, callback_results: List[float]) -> Callable:
    """Compile ``def _cb(window): ...`` reducing ``window[-1]`` via ``arr.<op>()``.

    Mirrors ``test_chain.py::_make_callback``: the snippet is built with ``compile``/``exec`` and the source is
    attached via ``__source__`` so the AST-based precompute analyzer can read the reduction op. The extra
    ``callback_results`` override lets the closure append into the test's list (callbacks run in-process on the Deisa
    event loop).
    """
    src = textwrap.dedent(
        f"def _cb(window):\n    arr = window[-1]\n    s = arr.{op}().compute()\n    callback_results.append(float(s))\n"
    )
    scope: Dict[str, Any] = {"callback_results": callback_results}
    code = compile(src, f"<test_precompute_memory:{op}>", "exec")
    exec(code, scope)
    fn = scope["_cb"]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


def _make_compute_callback(body: str, results: List[Any]) -> Callable:
    """Compile ``def _cb(window): <body>`` appending into ``results``.

    ``body`` is the dedented statement list (``textwrap.indent`` re-indents it inside the function). Used by the
    multi-reduction / axis e2e tests whose callbacks call several reductions and store their outputs.
    """
    src = textwrap.dedent(f"def _cb(window):\n{textwrap.indent(body, '    ')}")
    scope: Dict[str, Any] = {"results": results, "np": np}
    code = compile(src, "<test_precompute_memory>", "exec")
    exec(code, scope)
    fn = scope["_cb"]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


@pytest.fixture(scope="function")
def env_setup_2workers():
    """Two-worker LocalCluster + matching client for end-to-end tests."""
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        dashboard_address=":0",
        worker_dashboard_address=":0",
    )
    cluster.wait_for_workers(2, timeout=20)
    os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
    client = Client(cluster, name="env_setup_2workers")
    yield client, cluster
    client.close()
    cluster.close()


class TestPrecomputeMemory:
    """End-to-end tests that measure worker memory to confirm the full chunk
    never crosses the bridge -> worker boundary on the precompute path.

        Chain folding (expressions like ``(arr * arr).sum()`` being folded into a single per-bridge branch_func) is NOT
        exercised here end-to-end. It is covered at the unit level by test/test_chain.py (TestWalkChain /
        TestChainBranchFunc). The memory layer cannot exercise it through the register path: the registered callback
        body is both what the analyzer inspects to detect the chain AND what runs on the already-chain-folded partials,
        so a callback that expresses the chain source cannot correctly consume the folded scalar partials. See
        test_chain.py for the mechanism.
    """

    @pytest.mark.parametrize(
        "op",
        ["sum", "mean", "var", "std"],
    )
    def test_precompute_worker_only_sees_partials(self, env_setup_2workers, op):
        """With a callback that reduces the global chunk to a scalar via
        ``arr.<op>()``, only the per-bridge partial (scalar/dict-blob size, ~8 bytes) should appear on workers. The
        full chunk (~32 MB) must NOT.

        Parametrized over ``sum`` / ``mean`` / ``var`` / ``std``; the callback result is asserted against the true
        global value computed from the same data ``generate_data`` returned (a loose ``x >= 0.0`` predicate would
        satisfy a buggy ``0.0`` result).
        """
        client, cluster = env_setup_2workers
        # Use a chunk big enough that "big" vs "small" is unmistakable.
        # 2048 * 2048 * 8 = 32 MB per chunk. Two bridges => 64 MB total
        # in the legacy path, ~16 bytes (two scalars / dict-blobs) on the
        # precompute path.
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )

        deisa = Deisa(wait_for_go=False)

        callback_results: List[float] = []

        # Build (and register) the callback whose reduction op matches the
        # parametrization. The AST analyzer reads ``op`` from the source.
        deisa.register(array_name)(_make_callback(op, callback_results))

        # Wait for bridges and deisa to handshake.
        time.sleep(0.5)

        # Snapshot worker memory BEFORE we send any data.
        before = _worker_bytes_per_key(client)
        before_max = max(_largest_key_per_worker(before).values())
        assert before_max == 0, f"Workers should start empty, but found max key of {before_max} bytes: {before}"

        # Send one iteration and keep the global ground-truth array.
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        # Wait for the callback to fire (it sets callback_results).
        assert wait_for(lambda: len(callback_results) >= 1, timeout=30), "callback was not called within 30s"

        # Inspect worker memory AFTER the send.
        after = _worker_bytes_per_key(client)
        after_max_per_worker = _largest_key_per_worker(after)
        logging.warning(f"PRECOMPUTE TEST ({op}): per-worker max key nbytes: {after_max_per_worker}")
        logging.warning(f"PRECOMPUTE TEST ({op}): per-worker keys: {after}")

        # The full chunk is 32 MB; the partial is a scalar/dict-blob (~8-32
        # bytes). Allow some slack for numpy wrapping, but keep it ~5000x
        # smaller than the chunk.
        max_allowed = 64 * 1024  # 64 KB
        for worker, max_nbytes in after_max_per_worker.items():
            assert max_nbytes < max_allowed, (
                f"Worker {worker} holds a key of {max_nbytes} bytes; "
                f"expected only the small partial (< {max_allowed} bytes). "
                f"Full chunk appears to have landed on the worker -- precompute "
                f"is not doing its job. Keys: {after}"
            )

        # The callback must have fired exactly once (one iteration) and
        # returned the true reduction value for this op (var/std
        # used to deliver 0.0, which the old ``x >= 0.0`` predicates accepted).
        # ``generate_data`` fills the array with random values in [0, 1).
        assert len(callback_results) == 1, f"Expected exactly one callback invocation, got {len(callback_results)}"
        truth = {"sum": np.sum, "mean": np.mean, "var": np.var, "std": np.std}[op](global_data)
        assert np.isclose(callback_results[0], truth, rtol=1e-5, atol=1e-9), (
            f"callback result {callback_results[0]!r} is not the true {op!r} of the global data "
            f"(expected {truth!r}). The precompute delivery must return the combined FINAL value, "
            f"never a value a re-applied callback op computes from a mis-shaped delivered array."
        )

        # NOTE: we deliberately do NOT call deisa.execute_callbacks() here:
        # TestSimulation.__del__ closes the bridges via async_close_bridges,
        # which handles the lifecycle. Calling execute_callbacks() here would
        # hang waiting for a bridge-done event that only fires on close.

    def test_no_precompute_worker_sees_full_chunk(self, env_setup_2workers):
        """Control test: WITHOUT precompute, the full chunk should land on
        workers. Confirms the memory-measurement harness is sensitive enough to actually see the difference.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )

        deisa = Deisa(wait_for_go=False)

        callback_results: List[float] = []

        @deisa.register(array_name, precompute=False)
        # NOTE: precompute=False opts out of precompute and falls back to the
        # legacy full-chunk scatter path. The callback has a reduction
        # (``arr.sum()``), so without precompute=False it would be precomputed.
        # This control test confirms the memory harness can detect the
        # full chunk on workers when precompute=False is used.
        def _cb(window: list[DeisaArray]) -> None:
            arr = window[-1]
            logging.warning(f"NO-PRECOMPUTE TEST: callback received shape={arr.shape}")
            # The dask array should be the full global shape.
            assert arr.shape == global_shape, f"Expected full global shape {global_shape}, got {arr.shape}"
            callback_results.append(float(arr.sum().compute()))

        time.sleep(0.5)
        sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(callback_results) >= 1, timeout=30)

        after = _worker_bytes_per_key(client)
        after_max_per_worker = _largest_key_per_worker(after)
        logging.warning(f"NO-PRECOMPUTE TEST: per-worker max key nbytes: {after_max_per_worker}")

        # The full chunk (32 MB) MUST appear on the workers when precompute
        # is disabled -- otherwise the test harness is broken.
        chunk_bytes = int(np.prod(chunk_shape) * 8)  # float64
        # At least one worker should hold a key close to the chunk size.
        max_seen = max(after_max_per_worker.values())
        assert max_seen >= chunk_bytes // 2, (
            f"Expected at least one worker to hold ~{chunk_bytes} bytes "
            f"(the full chunk), but max key seen was {max_seen} bytes. "
            f"Per-worker: {after_max_per_worker}. Harness is broken -- "
            f"this control confirms the precompute test above is meaningful."
        )

        assert len(callback_results) == 1
        # See note in test_precompute_worker_only_sees_partials about
        # why we do not call execute_callbacks() here.

    def test_multi_array_precompute_end_to_end(self, env_setup_2workers):
        """Two arrays registered in ONE callback with precompute=True.

        Each array must get its OWN precompute branch (``x-sum`` for the reduction on ``x``, ``y-sum`` for the
        reduction on ``y``), so the callback receives per-bridge partial stacks -- shape ``(2,)`` with two bridges --
        for BOTH arrays.

        The ``(2,)`` shape asserts are the real discriminator: a value-only assert would pass even if both arrays
        delivered a legacy tiled full-chunk result (a full-chunk sum equals the global sum).
        """
        client, cluster = env_setup_2workers
        chunk_shape = (8, 16)
        global_shape = (16, 16)

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                "x": {"global_shape": global_shape, "chunk_shape": chunk_shape},
                "y": {"global_shape": global_shape, "chunk_shape": chunk_shape},
            },
            wait_for_go=False,
        )

        deisa = Deisa(wait_for_go=False)

        results: List[Any] = []

        @deisa.register("x", "y")
        def _cb(x_windows, y_windows):
            x = x_windows[-1]
            y = y_windows[-1]
            results.append((float(x.sum().compute()), float(y.sum().compute()), tuple(x.shape), tuple(y.shape)))

        time.sleep(0.5)

        x_global, y_global = sim.generate_data("x", "y", iteration=1, update_workers=True)

        assert wait_for(lambda: len(results) >= 1, timeout=30), "callback was not called within 30s"

        x_sum, y_sum, x_shape, y_shape = results[0]
        # Float64 summation ORDER differs between the per-bridge partial stack and a
        # direct full-array sum, so compare with a loose relative tolerance.
        assert np.isclose(x_sum, float(np.sum(x_global)))
        assert np.isclose(y_sum, float(np.sum(y_global)))
        # Both arrays go through precompute: the callback sees the stack of
        # per-bridge partials. The window[-1] subscript makes the analyzed reduction
        # axis-0 (the stub is sliced to its last row before .sum()), so each bridge
        # ships a (1, 16) row-sum partial and the stack over 2 bridges is (2, 1, 16).
        # A legacy full-chunk delivery would hand the callback the tiled (16, 16)
        # chunk; the shape assert is the real discriminator (a value-only assert
        # would pass by accident -- a full-chunk sum equals the global sum).
        assert x_shape == (2, 1, 16)
        assert y_shape == (2, 1, 16)

    def test_multi_reduction_callback_receives_all_reductions(self, env_setup_2workers):
        """A 3-reduction callback receives all three true values.

        Every reduction's partials must be delivered to the callback: if only the FIRST reduction's were (`darr =
        darr_chunks[0]`), the other reductions would run on the sum-stack artifact and return values far from the
        truth.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )
        deisa = Deisa(wait_for_go=False)

        results: List[Any] = []
        deisa.register(array_name)(
            _make_compute_callback(
                "arr = window[-1]\n"
                "s = arr.sum().compute()\n"
                "m = arr.mean().compute()\n"
                "mx = arr.max().compute()\n"
                "results.append((float(s), float(m), float(mx)))",
                results,
            )
        )

        time.sleep(0.5)
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(results) >= 1, timeout=30), "callback was not called within 30s"
        s, m, mx = results[0]
        assert np.isclose(s, float(np.sum(global_data)), rtol=1e-5, atol=1e-9), f"sum {s} != {np.sum(global_data)}"
        assert np.isclose(m, float(np.mean(global_data)), rtol=1e-5, atol=1e-9), f"mean {m} != {np.mean(global_data)}"
        assert np.isclose(mx, float(np.max(global_data)), rtol=1e-5, atol=1e-9), f"max {mx} != {np.max(global_data)}"

    def test_same_op_axis_pairs_survive_end_to_end(self, env_setup_2workers):
        """Precomputed delivery: ``arr.sum()`` + ``arr.sum(axis=0)`` both correct.

        Two ``sum`` hints on the same array MUST carry distinct output_keys (``f-sum`` and ``f-sum-axis0``): if both
        shared ``f-sum``, one branch would overwrite the other in the bridge's ``output_key`` index and the callback
        would receive a wrong shape (or an exception) for the axis reduction.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )
        deisa = Deisa(wait_for_go=False)

        results: List[Any] = []
        deisa.register(array_name)(
            _make_compute_callback(
                "arr = window[-1]\n"
                "s = arr.sum().compute()\n"
                "s0 = arr.sum(axis=0).compute()\n"
                "results.append((float(s), np.asarray(s0)))",
                results,
            )
        )

        time.sleep(0.5)
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(results) >= 1, timeout=30), "callback was not called within 30s"
        s, s0 = results[0]
        truth0 = np.sum(global_data, axis=0)
        assert np.isclose(s, float(np.sum(global_data)), rtol=1e-5, atol=1e-9), f"sum {s} != {np.sum(global_data)}"
        assert s0.shape == truth0.shape, f"sum(axis=0) shape {s0.shape} != truth {truth0.shape}"
        assert np.allclose(s0, truth0, rtol=1e-5, atol=1e-9), "sum(axis=0) values differ from truth"

    def test_axis_reductions_end_to_end(self, env_setup_2workers):
        """Precomputed delivery: ``arr.mean(axis=0)`` and ``arr.sum(axis=1)``,
        (2, 1) grid.

                ``mean(axis=0)`` reduces over the 2 row-strips (red grid level 0, kept level extent 1); ``sum(axis=1)``
                keeps grid level 0 (second combine stage concatenates the 2 row-strips along data axis 0). A crash or a
                silently wrong shape/value here means the red/kept-level geometry was misread.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )
        deisa = Deisa(wait_for_go=False)

        results: List[Any] = []
        deisa.register(array_name)(
            _make_compute_callback(
                "arr = window[-1]\n"
                "m0 = arr.mean(axis=0).compute()\n"
                "s1 = arr.sum(axis=1).compute()\n"
                "results.append((np.asarray(m0), np.asarray(s1)))",
                results,
            )
        )

        time.sleep(0.5)
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(results) >= 1, timeout=30), "callback was not called within 30s"
        m0, s1 = results[0]
        truth_m0 = np.mean(global_data, axis=0)
        truth_s1 = np.sum(global_data, axis=1)
        assert m0.shape == truth_m0.shape, f"mean(axis=0) shape {m0.shape} != truth {truth_m0.shape}"
        assert np.allclose(m0, truth_m0, rtol=1e-5, atol=1e-9), "mean(axis=0) values differ from truth"
        assert s1.shape == truth_s1.shape, f"sum(axis=1) shape {s1.shape} != truth {truth_s1.shape}"
        assert np.allclose(s1, truth_s1, rtol=1e-5, atol=1e-9), "sum(axis=1) values differ from truth"

    def test_multiple_callbacks_same_array_each_correct(self, env_setup_2workers):
        """Two callbacks on the same array each get their own result.

        The per-array branch list must MERGE across callbacks: if ``set_task_branches`` blindly overwrote it, only the
        LAST registered callback's branches would be executed and the first callback would compute ``sum()`` of the
        other's mean blobs.
        """
        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )
        deisa = Deisa(wait_for_go=False)

        sum_results: List[float] = []
        mean_results: List[float] = []
        deisa.register(array_name)(_make_callback("sum", sum_results))
        deisa.register(array_name)(_make_callback("mean", mean_results))

        time.sleep(0.5)
        global_data = sim.generate_data(array_name, iteration=1, update_workers=True)

        assert wait_for(lambda: len(sum_results) >= 1 and len(mean_results) >= 1, timeout=30), (
            "callbacks were not both called within 30s"
        )
        assert np.isclose(sum_results[0], float(np.sum(global_data)), rtol=1e-5, atol=1e-9), (
            f"sum callback {sum_results[0]} != {np.sum(global_data)}"
        )
        assert np.isclose(mean_results[0], float(np.mean(global_data)), rtol=1e-5, atol=1e-9), (
            f"mean callback {mean_results[0]} != {np.mean(global_data)}"
        )

    @pytest.mark.parametrize("expr", ["(arr * arr).sum()", "arr[2:5].sum()"])
    def test_registration_refuses_chained_reduction(self, env_setup_2workers, expr):
        """Registration REFUSES non-direct reductions loudly.

        ``(arr * arr).sum()`` and ``arr[2:5].sum()`` cannot be reconstructed on the callback side (the unrewritten
        callback re-applies the chain on the partials, e.g. ``(sum x)^2`` instead of ``sum x^2``), so registration
        raises ``UnsupportedReductionError`` instead of shipping a silently wrong value end-to-end.
        """
        from deisa.dask.precompute_analyzer import UnsupportedReductionError

        client, cluster = env_setup_2workers
        chunk_shape = (2048, 2048)
        global_shape = (chunk_shape[0] * 2, chunk_shape[1])
        array_name = "temperature"

        # The bridges must be alive for Deisa(wait_for_go=False) to handshake;
        # they are never sent data in this test (registration itself raises).
        _sim = TestSimulation(
            client,
            mpi_parallelism=(2, 1),
            arrays_metadata={
                array_name: {
                    "global_shape": global_shape,
                    "chunk_shape": chunk_shape,
                },
            },
            wait_for_go=False,
        )
        deisa = Deisa(wait_for_go=False)

        results: List[Any] = []
        cb = _make_compute_callback(
            f"arr = window[-1]\ns = ({expr}).compute()\nresults.append(float(s))",
            results,
        )
        with pytest.raises(UnsupportedReductionError):
            deisa.register(array_name)(cb)
