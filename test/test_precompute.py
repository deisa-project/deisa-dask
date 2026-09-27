# =============================================================================
# Copyright (C) 2026 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written  permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
Tests for the compute-boundary precompute analyzer.

The analyzer finds compute boundaries (.compute(), client.compute(...),
client.submit(...), etc.) in the user's callback source, symbolically
evaluates the dask arrays being computed (no execution), and extracts
reduction hints from each dask array's task graph.

Crucial contract: the user's callback is NEVER called during analysis.
"""

import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import dask.array as da
import numpy as np
import pytest

from deisa.dask.precompute_analyzer import (
    IncompatibleCallbackError,
    MaterializationError,
    NoComputeBoundaryError,
    NoPrecomputableReductionError,
    UnsupportedReductionError,
    analyze_callback,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _simple_stub() -> da.Array:
    return da.zeros((10, 10), chunks=(5, 5), dtype=np.float64)


def _hint_keys(hints: List[Dict[str, Any]]) -> List[str]:
    return sorted(h["output_key"] for h in hints)


def _op_names(hints: List[Dict[str, Any]]) -> List[str]:
    return sorted(h["op_name"] for h in hints)


# ---------------------------------------------------------------------------
# Per-reduction support: each reduction found in a graph that crosses a
# compute boundary should be detected.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "op_name",
    ["sum", "mean", "max", "min", "std", "var", "prod"],
)
def test_compute_direct_reduction(op_name: str) -> None:
    """``result = arr.op(); result.compute()`` should produce a hint for op_name."""
    arr = _simple_stub()
    src = f"""
        def callback(arr):
            result = arr.{op_name}()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    assert len(hints) == 1
    assert hints[0]["output_key"] == f"f-{op_name}"
    assert hints[0]["op_name"] == op_name


@pytest.mark.parametrize(
    "source,expected_hint_keys,expected_axis",
    [
        pytest.param(
            """\
        def callback(arr):
            result = arr[2:5].sum()
            result.compute()""",
            ["f-sum"],
            None,
            id="sliced_reduction",
        ),
        pytest.param(
            """\
        def callback(arr):
            result = arr[:, 0].sum()
            result.compute()""",
            ["f-sum"],
            None,
            id="column_slice_reduction",
        ),
        pytest.param(
            """\
        def callback(arr):
            result = (arr ** 2).sum()
            result.compute()""",
            ["f-sum"],
            None,
            id="expression_pow",
        ),
        pytest.param(
            """\
        def callback(arr):
            result = da.sum(arr)
            result.compute()""",
            ["f-sum"],
            None,
            id="dask_module_sum",
        ),
        pytest.param(
            """\
        def callback(arr):
            result = arr.sum(axis=0)
            result.compute()""",
            ["f-sum"],
            0,
            id="axis_kwarg_int",
        ),
        pytest.param(
            """\
        def callback(arr):
            result = arr.sum(axis=(0, 1))
            result.compute()""",
            ["f-sum"],
            (0, 1),
            id="axis_kwarg_tuple",
        ),
    ],
)
def test_compute_single_array_hints(source, expected_hint_keys, expected_axis) -> None:
    arr = _simple_stub()
    src = textwrap.dedent(source)
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == expected_hint_keys
    if expected_axis is not None:
        assert len(hints) == 1
        assert hints[0]["chunk_kwargs"].get("axis") == expected_axis


@pytest.mark.parametrize(
    "source,expected_hint_key",
    [
        pytest.param(
            """\
        def callback(arr_a, arr_b):
            result = (arr_a - arr_b).max()
            result.compute()""",
            "a-max",
            id="sub",
        ),
        pytest.param(
            """\
        def callback(arr_a, arr_b):
            result = (arr_a * arr_b).sum()
            result.compute()""",
            "a-sum",
            id="mul",
        ),
    ],
)
def test_compute_two_array_hints(source, expected_hint_key) -> None:
    a = _simple_stub()
    b = _simple_stub()
    src = textwrap.dedent(source)
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"a": a, "b": b})
    assert _hint_keys(hints) == [expected_hint_key]


def test_compute_multiple_reductions() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            s = arr.sum()
            m = arr.mean()
            mx = arr.max()
            s.compute()
            m.compute()
            mx.compute()
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-max", "f-mean", "f-sum"]


def test_same_op_different_axes_distinct_output_keys() -> None:
    """``arr.sum()`` + ``arr.sum(axis=0)`` must produce distinct output keys.

    On the pre-fix code both hints carried ``f-sum`` (the key was
    ``f"{array_name}-{op_name}"``), so the topic handler grouped the two
    reductions into one and one of them was silently overwritten
    (``bridge.py`` indexed branches by ``output_key``). The full reduction
    keeps the stable ``f-sum`` key (existing tests pin it); the axis
    reduction gets a deterministic discriminator suffix.
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            result = arr.sum()
            result.compute()
            result0 = arr.sum(axis=0)
            result0.compute()
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    keys = [h["output_key"] for h in hints]
    assert len(keys) == 2, f"expected two hints, got {keys}"
    assert len(set(keys)) == 2, f"expected distinct output keys, got {keys}"
    assert _hint_keys(hints) == ["f-sum", "f-sum-axis0"]


def test_same_op_same_axis_keeps_single_output_key() -> None:
    """Two IDENTICAL reductions (``arr.sum()`` twice) keep ONE output key.

    They are semantically identical: one branch, one bridge execution,
    shared by every callback's dispatch view. Distinct keys would file two
    branches that compute the same thing.
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            a = arr.sum()
            a.compute()
            b = arr.sum()
            b.compute()
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    keys = [h["output_key"] for h in hints]
    assert len(keys) == 2, f"expected two hints, got {keys}"
    # Identical reductions share ONE key (the dedup unit is the output_key:
    # merge_branches and the topic handler group by it).
    assert set(keys) == {"f-sum"}, f"expected a single shared key, got {keys}"


def test_compute_helper_same_file() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr, grid_dvx):
            result = density(arr, grid_dvx)
            result.compute()

        def density(f, dvx):
            return da.sum(f, axis=(0, 1)) * dvx
        """
    cb = _make_function("callback", src)
    helpers = {"density": _make_function("density", src)}
    hints, _ = analyze_callback(cb, {"f": arr}, helpers=helpers)
    assert _hint_keys(hints) == ["f-sum"]


def test_compute_loop_static_range() -> None:
    arr = da.zeros((4, 10, 10), chunks=(1, 5, 5), dtype=np.float64)
    src = """
        def callback(arr):
            Nsp = 4
            for isp in range(Nsp):
                sp = arr[isp]
                m = sp.sum()
                m.compute()
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    # Loop unrolls statically; we get at least one reduction hint
    assert any(h["op_name"] == "sum" for h in hints)


def test_compute_window_subscript_negative_one() -> None:
    """``window[-1].sum().compute()`` should map to the last registered array."""
    arr = _simple_stub()
    src = """
        def callback(window):
            result = window[-1].sum()
            result.compute()
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    assert _hint_keys(hints) == ["f-sum"]


# ---------------------------------------------------------------------------
# client.compute / client.submit boundaries
# ---------------------------------------------------------------------------
def test_client_compute_single_array() -> None:
    """``client.compute(arr)`` should register the dask array as a boundary."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            client.compute(da.sum(arr))
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr, "client": client_stub})
    assert _hint_keys(hints) == ["f-sum"]


def test_client_compute_list_of_arrays() -> None:
    """``client.compute([a, b, c])`` should register each array as a boundary."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            client.compute([
                da.sum(arr),
                da.mean(arr),
                da.max(arr),
            ])
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr, "client": client_stub})
    assert _hint_keys(hints) == ["f-max", "f-mean", "f-sum"]


def test_client_submit_with_dask_array() -> None:
    """``client.submit(func, arr)`` should register the dask array as a boundary."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            client.submit(float, da.sum(arr))
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr, "client": client_stub})
    assert _hint_keys(hints) == ["f-sum"]


def test_client_compute_inside_helper() -> None:
    """The compute boundary can live inside a helper function."""
    arr = _simple_stub()
    client_stub = _FakeClient()
    src = """
        def callback(arr, client):
            measure(client, arr)

        def measure(client, f):
            client.compute(da.sum(f))
        """
    cb = _make_function("callback", src)
    helpers = {"measure": _make_function("measure", src)}
    hints, _ = analyze_callback(cb, {"f": arr, "client": client_stub}, helpers=helpers)
    assert _hint_keys(hints) == ["f-sum"]


# ---------------------------------------------------------------------------
# Materialization (errors)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,arg_key",
    [
        pytest.param(
            """\
        def callback(arr):
            full = np.array(arr)
            full.compute()""",
            "f",
            id="np_array",
        ),
        pytest.param(
            """\
        def callback(arr):
            full = np.asarray(arr)
            full.compute()""",
            "f",
            id="np_asarray",
        ),
        pytest.param(
            """\
        def callback(arr):
            full = np.array(arr[0])""",
            "fdistribu_offline",
            id="offline_compression",
        ),
    ],
)
def test_materialization_error(source, arg_key) -> None:
    src = textwrap.dedent(source)
    cb = _make_function("callback", src)
    with pytest.raises(MaterializationError):
        analyze_callback(cb, {arg_key: _simple_stub()})


# ---------------------------------------------------------------------------
# Non-reduction compute boundaries
# ---------------------------------------------------------------------------
def test_compute_fft_only_raises_no_precomputable_reduction() -> None:
    """A callback with .compute() but no reductions should raise NoPrecomputableReductionError."""
    arr = da.zeros((10, 10), chunks=(10, 10), dtype=np.float64)  # single chunk so FFT works
    src = """
        def callback(arr):
            phi = da.fft.fft2(arr)
            phi.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(NoPrecomputableReductionError):
        analyze_callback(cb, {"f": arr})


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """\
        def callback(arr):
            result = arr.sum()
            # result not computed - no boundary""",
            id="no_boundary",
        ),
        pytest.param(
            """\
        def callback(arr):
            x = 1 + 2""",
            id="no_dask",
        ),
    ],
)
def test_no_compute_boundary_error(source) -> None:
    arr = _simple_stub()
    src = textwrap.dedent(source)
    cb = _make_function("callback", src)
    with pytest.raises(NoComputeBoundaryError):
        analyze_callback(cb, {"f": arr})


# ---------------------------------------------------------------------------
# Incompatible patterns
# ---------------------------------------------------------------------------
def test_getattr_raises_incompatible_callback() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            op = 'sum'
            result = getattr(arr, op)()
            result.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(IncompatibleCallbackError):
        analyze_callback(cb, {"f": arr})


def test_dynamic_loop_raises_incompatible_callback() -> None:
    arr = _simple_stub()
    src = """
        def callback(arr):
            for x in get_items():
                s = x.sum()
                s.compute()
        """
    cb = _make_function("callback", src)
    with pytest.raises(IncompatibleCallbackError):
        analyze_callback(cb, {"f": arr})


# ---------------------------------------------------------------------------
# precompute=False
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,arr_factory,check_warning",
    [
        pytest.param(
            """\
        def callback(arr):
            phi = da.fft.fft2(arr)
            phi.compute()""",
            lambda: da.zeros((10, 10), chunks=(10, 10), dtype=np.float64),
            True,
            id="fft_no_reductions",
        ),
        pytest.param(
            """\
        def callback(arr):
            op = 'sum'
            result = getattr(arr, op)()
            result.compute()""",
            _simple_stub,
            False,
            id="incompatible_getattr",
        ),
        pytest.param(
            """\
        def callback(arr):
            return (arr - arr.mean()).sum().compute()""",
            _simple_stub,
            True,
            id="cross_reduction_refusal",
        ),
    ],
)
def test_precompute_false_returns_empty(source, arr_factory, caplog, check_warning) -> None:
    """``precompute=False`` returns [] and logs a warning instead of raising.

    The user is explicitly opting out of the precompute safety net: the
    analyzer swallows the refusal and returns zero hints, which the
    registration layer turns into a legacy full-chunk scatter (or raises,
    depending on the registration-time policy).
    """
    arr = arr_factory()
    src = textwrap.dedent(source)
    cb = _make_function("callback", src)
    with caplog.at_level("WARNING"):
        hints, _ = analyze_callback(cb, {"f": arr}, precompute=False)
    assert hints == []
    if check_warning:
        # At least one warning emitted about the refusal.
        assert any(
            "precompute" in str(rec.message).lower() or "reduc" in str(rec.message).lower() for rec in caplog.records
        )


# ---------------------------------------------------------------------------
# Gysela-style patterns
# ---------------------------------------------------------------------------
def test_gysela_density_helper() -> None:
    """``density(f, grid) = da.sum(f, axis=(0, 3, 4)) * grid.dvx * grid.dvy``."""
    arr = da.zeros((4, 5, 10, 10, 10), chunks=(1, 5, 5, 5, 5), dtype=np.float64)
    client_stub = _FakeClient()
    grid_obj = _FakeGrid(dvx=0.5, dvy=0.5)
    src = """
        def callback(arr, grid, client):
            n = density(arr, grid)
            client.compute(n)

        def density(f, grid):
            return da.sum(f, axis=(0, 3, 4)) * grid.dvx * grid.dvy
        """
    cb = _make_function("callback", src)
    helpers = {"density": _make_function("density", src)}
    hints, _ = analyze_callback(cb, {"f": arr, "grid": grid_obj, "client": client_stub}, helpers=helpers)
    assert _hint_keys(hints) == ["f-sum"]


def test_gysela_measure_helper_loop() -> None:
    """The measure() helper has 5 reductions and is called in a loop over species.

    The compute boundary is the ``client.compute([...])`` inside measure.
    """
    fdistribu = da.zeros((4, 5, 10, 10, 10), chunks=(1, 5, 5, 5, 5), dtype=np.float64)
    client_stub = _FakeClient()
    grid_obj = _FakeGrid(
        dvx=0.5,
        dvy=0.5,
        vx=da.zeros((10,), chunks=(10,)),
        vy=da.zeros((10,), chunks=(10,)),
        dV_4D=0.25,
    )
    src = """
        def callback(fdistribu, grid, Nsp, client):
            for isp in range(Nsp):
                measure(client, grid, fdistribu[isp])

        def measure(client, cfg, f):
            vx_bc = cfg.vx[(None, None, slice(None), None)]
            vy_bc = cfg.vy[(None, None, None, slice(None))]
            v2 = vx_bc ** 2 + vy_bc ** 2
            ek = 0.5 * da.sum(f * v2) * cfg.dV_4D
            l2 = da.sum(f ** 2) * cfg.dV_4D
            ms = da.sum(f) * cfg.dV_4D
            mx = da.sum(f * vx_bc) * cfg.dV_4D
            my = da.sum(f * vy_bc) * cfg.dV_4D
            client.compute([ek, l2, ms, mx, my])
        """
    cb = _make_function("callback", src)
    helpers = {"measure": _make_function("measure", src)}
    hints, _ = analyze_callback(
        cb,
        {"fdistribu": fdistribu, "grid": grid_obj, "Nsp": 4, "client": client_stub},
        helpers=helpers,
    )
    op_names = sorted(h["op_name"] for h in hints)
    # We expect 5 unique reductions per call site, possibly repeated per species loop iteration
    assert "sum" in op_names
    # At minimum we have at least 5 sum hints (one per reduction in measure)
    sum_count = op_names.count("sum")
    assert sum_count >= 5


# ---------------------------------------------------------------------------
# Cross-reduction refusal (multi-reduction branches)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,helper_src,check_msg",
    [
        pytest.param(
            """\
        def callback(arr):
            return (arr - arr.mean()).sum().compute()""",
            None,
            True,
            id="simple_cross",
        ),
        pytest.param(
            """\
        def callback(arr):
            return (arr - arr.mean().sum()).sum().compute()""",
            None,
            False,
            id="nested_cross",
        ),
        pytest.param(
            """\
        def callback(arr):
            return drift(arr).compute()

        def drift(arr):
            return (arr - arr.mean()).sum()""",
            """\
        def drift(arr):
            return (arr - arr.mean()).sum()""",
            False,
            id="helper_cross",
        ),
    ],
)
def test_unsupported_reduction_error(source, helper_src, check_msg) -> None:
    """A reduction depending on another reduction's aggregate must be refused.

    The naive per-reduction hint extraction would emit both an ``f-mean`` and
    an ``f-sum`` hint, but ``f-sum`` is WRONG in multi-bridge setups: the
    bridge would compute ``(chunk - chunk.mean()).sum()`` locally, which is
    always 0. We refuse the whole expression (or with ``precompute=False`` the
    analyzer swallows it and returns no hints, so the user gets the legacy
    full-chunk scatter path if they explicitly opt out).
    """
    arr = _simple_stub()
    src = textwrap.dedent(source)
    cb = _make_function("callback", src)
    helpers = None
    if helper_src is not None:
        helpers = {"drift": _make_function("drift", textwrap.dedent(helper_src))}
    with pytest.raises(UnsupportedReductionError) as exc:
        analyze_callback(cb, {"f": arr}, helpers=helpers)
    if check_msg:
        # The error message must name the offending reduction and the
        # cross-reduction dependency so the user can fix the callback.
        msg = str(exc.value).lower()
        assert "sum" in msg  # the outer reduction
        assert "mean" in msg  # the inner reduction it depends on
        assert "bridge" in msg or "global" in msg or "all bridges" in msg


def test_independent_reductions_not_refused() -> None:
    """Two INDEPENDENT reductions on the same array must NOT be refused.

    The walker only refuses when one reduction's chunk-stage depends
    on ANOTHER reduction's aggregate layer. Two independent
    reductions (no shared sub-expression) are emitted as separate
    hints and precomputed independently.
    """
    arr = _simple_stub()
    src = """
        def callback(arr):
            a = arr.sum().compute()
            b = arr.mean().compute()
        """
    cb = _make_function("callback", src)
    hints, _ = analyze_callback(cb, {"f": arr})
    # Both reductions detected -- the walker does not refuse.
    op_names = _op_names(hints)
    assert "sum" in op_names
    assert "mean" in op_names


# ---------------------------------------------------------------------------
# Safety: callback is NEVER executed
# ---------------------------------------------------------------------------
def test_callback_never_executed_during_analysis() -> None:
    """The user's callback must never be invoked by analyze_callback."""
    call_count = {"n": 0}

    def cb(arr):
        call_count["n"] += 1
        result = arr.sum()
        result.compute()

    arr = _simple_stub()
    # Even if the analyzer is broken and calls the callback, we want to detect it.
    analyze_callback(cb, {"f": arr})
    assert call_count["n"] == 0, "analyze_callback invoked the user's callback — AST-only contract broken"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _FakeGrid:
    """A grid config stub with dask-array attributes."""

    dvx: float = 0.5
    dvy: float = 0.5
    dV_4D: float = 0.25
    vx: Any = None
    vy: Any = None


class _FakeClient:
    """Stub for a distributed.Client.

    The analyzer treats ``client.compute(...)`` and ``client.submit(...)`` as
    compute boundaries; the receiver's actual class doesn't matter, just its
    attributes. Returning ``None`` from these methods lets the analysis
    continue without raising.
    """

    def compute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def submit(self, *args: Any, **kwargs: Any) -> None:
        return None


def _make_function(name: str, src: str) -> Callable:
    """Compile a small Python snippet and return the named function."""
    import linecache

    src = textwrap.dedent(src)
    scope: Dict[str, Any] = {}
    code = compile(src, f"<test_precompute:{name}>", "exec")
    exec(code, scope)
    fn = scope[name]
    # Register the source so ``inspect.getsource`` can find it.
    fn.__source__ = src  # type: ignore[attr-defined]
    # Also pin linecache so tools like inspect that rely on file/line lookup
    # can resolve the source.
    linecache.cache[fn.__code__.co_filename] = (
        len(src),
        None,
        [line + "\n" for line in src.splitlines()],
        fn.__code__.co_filename,
    )
    return fn
