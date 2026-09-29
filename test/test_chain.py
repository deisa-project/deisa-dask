"""
Unit tests for the Stage 2B chain-folding walker.

These tests exercise the chain walker helpers directly on synthetic
dask graphs. They do NOT go through ``analyze_branch`` end-to-end
(see the design doc -- wiring is Stage 2B follow-up). They are
sanity checks that the walker correctly identifies single-input
chains, refuses constants and cross-array chains, and produces
branch_func callables that match naive numpy computations.
"""

import functools
import textwrap
from typing import Any, Callable, Dict

import dask.array as da
import numpy as np
import pytest

from deisa.dask.branch import (
    _apply_full_local_fold,
    _build_chain_branch_func,
    _find_chunk_layer,
    _find_single_upstream,
    _walk_chain,
)


def _make_callback(name: str, body: str) -> Callable:
    """Compile a small snippet ``def <name>(arr): <body>`` and return it.

    Mirrors the helper used in test_precompute.py so ``analyze_callback`` can walk the source if needed.
    """
    src = textwrap.dedent(f"def {name}(arr):\n{textwrap.indent(body, '    ')}")
    scope: Dict[str, Any] = {}
    code = compile(src, f"<test_chain:{name}>", "exec")
    exec(code, scope)
    fn = scope[name]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


def _find_agg_layer(graph) -> str:
    """Return the first ``*-aggregate-*`` layer name in a graph."""
    for ln in graph.layers:
        if "-aggregate-" in ln:
            return ln
    raise AssertionError("no aggregate layer in graph")


# ---------------------------------------------------------------------------
# _walk_chain
# ---------------------------------------------------------------------------
class TestWalkChain:
    @pytest.mark.parametrize(
        "expr,expected_length,should_refuse",
        [
            pytest.param(
                lambda arr: (arr * arr).sum(),
                2,
                False,
                id="self-ref-mul-sum",
            ),
            pytest.param(
                lambda arr: np.log(np.exp(arr)).sum(),
                3,
                False,
                id="ufunc-exp-log-sum",
            ),
            pytest.param(
                lambda arr: (arr**arr).sum(),
                2,
                False,
                id="self-ref-pow-sum",
            ),
            pytest.param(
                lambda arr: np.sin(arr).sum(axis=0),
                2,
                False,
                id="sin-axis0-sum",
            ),
            pytest.param(
                lambda arr: (arr - arr.mean()).sum(),
                None,
                True,
                id="cross-array-refused",
            ),
            pytest.param(
                lambda arr: (arr + 1).sum(),
                None,
                True,
                id="scalar-constant-refused",
            ),
            # Dedicated chunk/aggregate pairs must resolve via
            # _chunk_base_for_aggregate_base (mean_chunk/mean_agg,
            # chunk_max/max, chunk_min/min) instead of requiring an
            # exact base match.
            pytest.param(
                lambda arr: (arr * arr).mean(),
                2,
                False,
                id="self-ref-mul-mean",
            ),
            pytest.param(
                lambda arr: (arr * arr).max(),
                2,
                False,
                id="self-ref-mul-max",
            ),
            pytest.param(
                lambda arr: (arr * arr).min(),
                2,
                False,
                id="self-ref-mul-min",
            ),
        ],
    )
    def test_walk_chain_parametrized(self, expr, expected_length, should_refuse):
        arr = da.zeros((4, 4), chunks=2)
        g = expr(arr).__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        if should_refuse:
            assert chain is None
        else:
            assert chain is not None
            assert len(chain) == expected_length


# ---------------------------------------------------------------------------
# _build_chain_branch_func -- numerical correctness
# ---------------------------------------------------------------------------
class TestChainBranchFunc:
    """The composed branch_func must match the dask-computed value for
    every foldable chain. This is the core correctness property.
    """

    @pytest.mark.parametrize(
        "expr,seed,data_generator,expected_expr,rtol",
        [
            pytest.param(
                lambda arr: (arr * arr).sum(),
                0,
                lambda: np.random.random((4, 4)),
                lambda real: (real * real).sum(),
                None,
                id="squared-sum",
            ),
            pytest.param(
                lambda arr: np.log(np.exp(arr)).sum(),
                1,
                lambda: np.random.random((4, 4)),
                lambda real: real.sum(),
                1e-6,
                id="ufunc-chain",
            ),
            pytest.param(
                lambda arr: (arr**arr).sum(),
                2,
                lambda: np.random.random((4, 4)) * 0.5,
                lambda real: (real**real).sum(),
                None,
                id="pow-self",
            ),
        ],
    )
    def test_chain_branch_func_parametrized(self, expr, seed, data_generator, expected_expr, rtol):
        arr = da.zeros((4, 4), chunks=2)
        g = expr(arr).__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        branch_func = _build_chain_branch_func(chain)

        np.random.seed(seed)
        real = data_generator()
        result = float(branch_func(real).sum())
        expected = float(expected_expr(real))

        if rtol is not None:
            assert np.isclose(result, expected, rtol=rtol)
        else:
            assert np.isclose(result, expected)

    def test_picklable(self):
        """The composed branch_func must be picklable so it can cross
        the bridge process boundary.
        """
        import pickle

        arr = da.zeros((4, 4), chunks=2)
        expr = (arr * arr).sum()
        g = expr.__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        branch_func = _build_chain_branch_func(chain)
        # Round-trip pickle
        restored = pickle.loads(pickle.dumps(branch_func))
        real = np.arange(16, dtype=np.float64).reshape(4, 4)
        assert np.allclose(restored(real), branch_func(real))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestFindChunkLayer:
    def test_finds_chunk_stage_layer(self):
        arr = da.zeros((4, 4), chunks=2)
        g = (arr * arr).sum().__dask_graph__()
        agg_name = _find_agg_layer(g)
        agg_base = agg_name.split("-aggregate-", 1)[0]  # "sum"
        chunk_layer = _find_chunk_layer(g, agg_base)
        assert chunk_layer is not None
        assert chunk_layer.startswith("sum-") and "aggregate" not in chunk_layer


class TestFindSingleUpstream:
    @pytest.mark.parametrize(
        "expr,layer_selector,expected_name_start,expected_count",
        [
            pytest.param(
                lambda arr: (arr * arr).sum(),
                lambda g: next(ln for ln in g.layers if ln.startswith("sum-") and "aggregate" not in ln),
                "mul-",
                1,
                id="sum-layer-single-input",
            ),
            pytest.param(
                lambda arr: (arr * arr).sum(),
                lambda g: next(ln for ln in g.layers if ln.startswith("mul-")),
                "zeros_like-",
                2,
                id="mul-layer-self-ref",
            ),
            pytest.param(
                lambda arr: (arr + 1).sum(),
                lambda g: next(ln for ln in g.layers if ln.startswith("add-")),
                None,
                None,
                id="add-layer-constant-refused",
            ),
        ],
    )
    def test_find_single_upstream_parametrized(self, expr, layer_selector, expected_name_start, expected_count):
        arr = da.zeros((4, 4), chunks=2)
        g = expr(arr).__dask_graph__()
        layer = layer_selector(g)
        result = _find_single_upstream(g.layers[layer])
        if expected_count is None:
            # Refused case (scalar constant): walker returns None.
            assert result is None
        else:
            assert result is not None
            name, count = result
            assert name.startswith(expected_name_start)
            assert count == expected_count


# ---------------------------------------------------------------------------
# analyze_branch -- not wired in (Stage 2B follow-up), but the
# length-1 path still works and emits BranchSpec objects.
# ---------------------------------------------------------------------------
class TestAnalyzeBranchLength1:
    def test_analyze_branch_emits_branches(self):
        """analyze_branch produces length-1 branches for the
        per-reduction path. The chain walker folds multi-layer chains into a single branch_func.
        """
        from deisa.dask.branch import _analyze_branch

        # A multi-layer chain callback: (arr * arr).sum(). The chain
        # walker must fold {mul, sum} into a single branch_func; the
        # composed branch's _chain exposes the folded layers.
        cb = _make_callback("test_analyze_branch_cb", "return (arr * arr).sum().compute()")

        arrs = {"f": da.zeros((4, 4), chunks=2)}
        branches = _analyze_branch(cb, arrs)
        assert len(branches) == 1
        assert branches[0].output_key == "f-sum"
        assert branches[0].output_kind == "scalar"

        # The branch_func is a functools.partial over _chain_branch_func;
        # the folded layers live under its keywords["_chain"].
        branch_func = branches[0].branch_func
        chain = branch_func.keywords["_chain"]
        assert len(chain) == 2  # mul + sum genuinely folded into one branch

        def _layer_name(layer):
            func = layer[0]
            # The reduction's chunk-stage layer is itself a partial
            # wrapping numpy.sum (carrying dtype); unwrap it for the name.
            if isinstance(func, functools.partial):
                # A full-fold compositor partial carries the tier func in its keywords; surface the tier (the
                # compositor itself is operator-agnostic, so the op's identity lives on the tier layer).
                tier = getattr(func, "keywords", {}).get("tier_func")
                if tier is not None:
                    if isinstance(tier, functools.partial):
                        tier = tier.func
                    return getattr(tier, "__name__", repr(tier))
                return getattr(func.func, "__name__", repr(func))
            return getattr(func, "__name__", repr(func))

        names = [_layer_name(layer) for layer in chain]
        assert any(n in {"mul", "multiply"} for n in names)
        assert "sum" in names


# ---------------------------------------------------------------------------
# Multi-reduction callbacks: each branch must fold its OWN aggregate
# ---------------------------------------------------------------------------
class TestMultiReductionBranches:
    """Chain folding is hint-aware: a hint must be folded with the aggregate
    layer of its OWN reduction, never the first aggregate of the first walked graph. Regression for the
    aggregate-mismatch bug where ``s=arr.sum(); m=arr.mean(); mx=arr.max()`` folded the SUM chain into all three
    branches (f-mean and f-max then shipped wrong partials).
    """

    def _analyze(self, body: str) -> Dict[str, Any]:
        from deisa.dask.branch import _analyze_branch

        cb = _make_callback("multi_reduction_cb", body)
        branches = _analyze_branch(cb, {"f": da.zeros((4, 4), chunks=2, dtype="float64")})
        return {b.output_key: b for b in branches}

    def _chain_names(self, branch) -> list:
        chain = branch.branch_func.keywords["_chain"]
        names = []
        for layer in chain:
            func = layer[0]
            if isinstance(func, functools.partial):
                names.append(getattr(func.func, "__name__", repr(func)))
                # A full-fold compositor partial carries the tier func in its keywords: surface it so tests can
                # assert WHICH tier was folded (the compositor itself is operator-agnostic by design).
                tier = getattr(func, "keywords", {}).get("tier_func")
                if tier is not None:
                    if isinstance(tier, functools.partial):
                        tier = tier.func
                    names.append(getattr(tier, "__name__", repr(tier)))
            else:
                names.append(getattr(func, "__name__", repr(func)))
        return names

    @staticmethod
    def _scalar(value) -> float:
        return float(np.asarray(value).reshape(-1)[0])

    def test_multi_reduction_each_branch_uses_own_reduction(self):
        # T1 repro: three compute boundaries, one per reduction. Before the
        # fix, all three hints folded the SUM chain (walker_dask_arrays[0] +
        # first aggregate), so f-mean/f-max were wrong.
        branches = self._analyze(
            "s = arr.sum()\nm = arr.mean()\nmx = arr.max()\nreturn s.compute(), m.compute(), mx.compute()"
        )
        assert set(branches) == {"f-sum", "f-mean", "f-max"}
        assert branches["f-sum"].output_kind == "scalar"
        assert branches["f-mean"].output_kind == "mean"
        assert branches["f-max"].output_kind == "scalar"

        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)

        sum_names = self._chain_names(branches["f-sum"])
        assert "sum" in sum_names
        assert np.isclose(self._scalar(branches["f-sum"].branch_func(real)), real.sum())

        # f-mean must compute its OWN reduction: the mean_chunk form
        # ({n, total} dict), not the sum chain (which ships a bare scalar
        # and fails mean_agg at combine time).
        mean_names = self._chain_names(branches["f-mean"])
        assert "mean_chunk" in mean_names
        assert "sum" not in mean_names
        mean_partial = branches["f-mean"].branch_func(real)
        assert isinstance(mean_partial, dict)
        assert set(mean_partial) >= {"n", "total"}
        assert np.isclose(self._scalar(mean_partial["total"] / mean_partial["n"]), real.mean())

        # f-max must compute its OWN reduction via chunk_max, not the sum chain.
        max_names = self._chain_names(branches["f-max"])
        assert "chunk_max" in max_names
        assert "sum" not in max_names
        assert np.isclose(self._scalar(branches["f-max"].branch_func(real)), real.max())

    def test_single_graph_two_aggregates_each_folds_own_chain(self):
        # T2: ONE walker graph with TWO aggregate layers (sum and max).
        # f-max must fold the max chain; it must NOT inherit the 'mul','sum'
        # chain of f-sum.
        branches = self._analyze("return ((arr * arr).sum() + arr.max()).compute()")
        assert set(branches) == {"f-sum", "f-max"}

        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)

        sum_names = self._chain_names(branches["f-sum"])
        assert "mul" in sum_names and "sum" in sum_names
        assert np.isclose(self._scalar(branches["f-sum"].branch_func(real)), (real**2).sum())

        max_names = self._chain_names(branches["f-max"])
        assert "chunk_max" in max_names
        assert "mul" not in max_names and "sum" not in max_names
        assert np.isclose(self._scalar(branches["f-max"].branch_func(real)), real.max())

    def test_chained_mean_max_min_use_dedicated_chunk_pairs(self):
        # The dedicated chunk/aggregate pairs (mean_chunk/mean_agg,
        # chunk_max/max, chunk_min/min) must fold. Before the fix the chunk
        # layer lookup required an exact base match, so these fell back to
        # the length-1 path and silently shipped the RAW reduction:
        # (arr*arr).mean() returned total=136 (sum) instead of 1496
        # (sum of squares).
        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)

        mean = self._analyze("return (arr * arr).mean().compute()")["f-mean"]
        mean_partial = mean.branch_func(real)
        assert isinstance(mean_partial, dict)
        assert np.isclose(self._scalar(mean_partial["total"]), (real**2).sum())
        assert np.isclose(self._scalar(mean_partial["total"] / mean_partial["n"]), (real**2).mean())
        assert "mean_chunk" in self._chain_names(mean)

        mx = self._analyze("return (arr * arr).max().compute()")["f-max"]
        assert np.isclose(self._scalar(mx.branch_func(real)), (real**2).max())
        assert "chunk_max" in self._chain_names(mx)

        # min needs negative data so the raw min (-5) differs from the min of
        # squares (4).
        neg = np.array([[-5.0, 2.0], [3.0, 4.0]])
        mn = self._analyze("return (arr * arr).min().compute()")["f-min"]
        assert np.isclose(self._scalar(mn.branch_func(neg)), (neg**2).min())
        assert "chunk_min" in self._chain_names(mn)

    def test_var_std_folded_partials_combine_correctly(self):
        # var/std fold their pointwise chain AND ship {n, total, M} dict
        # partials that _combine_array_from_partials (kind='moment') can
        # combine into the correct global value.
        from deisa.dask.branch import _combine_array_from_partials

        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        chunks = [real[i : i + 2, j : j + 2] for i in range(0, 4, 2) for j in range(0, 4, 2)]

        for label, expected in [("std", (real**2).std()), ("var", (real**2).var())]:
            branch = self._analyze(f"return (arr * arr).{label}().compute()")[f"f-{label}"]
            assert "moment_chunk" in self._chain_names(branch)
            partials = []
            for c in chunks:
                p = branch.branch_func(c)
                partials.append(
                    {
                        "future": p,
                        "chunk_position": tuple(np.unravel_index(len(partials), (2, 2))),
                        "shape": tuple(np.asarray(p["total"]).shape),
                        "dtype": np.asarray(p["total"]).dtype,
                    }
                )
            combined = _combine_array_from_partials(
                partials,
                kind=branch.output_kind,
                finalize=branch.finalize,
                reduction_axes_hint=(0, 1),
                array_ndim=2,
            )
            # Compute on an explicit scheduler. Earlier tests in the same worker
            # can leave dask's global scheduler set to "dask.distributed" (any
            # Client does), and a bare .compute() then raises
            # "Requested dask.distributed scheduler but no Client active".
            assert np.isclose(self._scalar(combined.compute(scheduler="sync")), float(expected))

    def test_axis_reduction_combine_correct(self):
        # The two-phase, data-axis-ordered combine must produce the
        # TRUE axis reduction for per-bridge partials on a 2x2 grid.
        # A crash here, or mean(axis=1) silently returning the wrong
        # values ([7.5 9.5] vs truth [2.5 6.5 10.5 14.5]), means the
        # red/kept-level geometry was misread.
        from deisa.dask.branch import _combine_array_from_partials

        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        chunks = [real[i : i + 2, j : j + 2] for i in range(0, 4, 2) for j in range(0, 4, 2)]
        truth = {
            "mean": (np.mean(real, axis=0), np.mean(real, axis=1)),
            "sum": (np.sum(real, axis=0), np.sum(real, axis=1)),
        }
        for op in ("mean", "sum"):
            for ax in (0, 1):
                branch = self._analyze(f"r = arr.{op}(axis={ax})\nreturn r.compute()")[f"f-{op}"]
                partials = []
                for c in chunks:
                    p = branch.branch_func(c)
                    rep = np.asarray(p["total"]) if isinstance(p, dict) else np.asarray(p)
                    partials.append(
                        {
                            "future": p,
                            "chunk_position": tuple(np.unravel_index(len(partials), (2, 2))),
                            "shape": tuple(rep.shape),
                            "dtype": str(rep.dtype),
                        }
                    )
                combined = _combine_array_from_partials(
                    partials,
                    kind=branch.output_kind,
                    finalize=branch.finalize,
                    reduction_axes_hint=(ax,),
                    array_ndim=2,
                    op_name=branch.op_name,
                )
                # Explicit scheduler: this module is cluster-free, so it must not
                # depend on whatever default scheduler an earlier test left behind.
                got = np.asarray(combined.compute(scheduler="sync"))
                expected = truth[op][ax]
                assert got.shape == expected.shape, f"{op}(axis={ax}): got shape {got.shape}, expected {expected.shape}"
                assert np.allclose(got, expected, rtol=1e-10, atol=1e-12), (
                    f"{op}(axis={ax}): got {got.tolist()}, expected {expected.tolist()}"
                )


def _make_spec(output_key, op_name="sum", output_kind="scalar", reduction_axes=(), **kw):
    """Minimal BranchSpec for merge_branches unit tests (no cluster needed)."""
    from deisa.dask.branch import BranchSpec

    return BranchSpec(
        output_key=output_key,
        input_name="f",
        output_kind=output_kind,
        branch_func=lambda chunk: chunk,
        chunk_axis=None,
        finalize=None,
        partial_shape=(),
        partial_dtype="float64",
        op_name=op_name,
        reduction_axes=reduction_axes,
        **kw,
    )


def test_merge_branches_dedup_same_key_keeps_all_distinct_keys() -> None:
    """Merging identical signatures dedups; distinct keys survive."""
    from deisa.dask.branch import merge_branches

    existing = [_make_spec("f-sum"), _make_spec("f-mean", op_name="mean", output_kind="mean")]
    new = [_make_spec("f-sum"), _make_spec("f-max", op_name="max")]
    merged = merge_branches(existing, new)
    assert {b.output_key for b in merged} == {"f-sum", "f-mean", "f-max"}
    # The identical f-sum survived exactly once (the existing branch).
    assert sum(1 for b in merged if b.output_key == "f-sum") == 1


def test_merge_branches_refuses_same_key_different_signature() -> None:
    """Merging a same-key/different-signature collision must raise.

    A window read ``x[-1].sum()`` and a true ``x.sum(axis=0)`` can both carry ``f-sum`` from different callbacks (each
    starts a fresh per-callback seen map); their recorded reduction axes (() vs (0,)) differ, so a single shared
    branch cannot serve both callbacks. Refusing loudly at registration beats silently delivering one callback the
    other's result.
    """
    from deisa.dask.branch import merge_branches
    from deisa.dask.precompute_analyzer import PrecomputeRuntimeError

    window_read = _make_spec("f-sum", reduction_axes=())
    axis_zero = _make_spec("f-sum", reduction_axes=(0,))
    with pytest.raises(PrecomputeRuntimeError):
        merge_branches([window_read], [axis_zero])


# ---------------------------------------------------------------------------
# Full local fold: the operator-generic minimal-scalar-partial machinery
# ---------------------------------------------------------------------------
class TestFullFoldLocal:
    """``_apply_full_local_fold`` composes ``tier -> finalizer`` on the bridge-local chain.

    The operator-generic property: whichever scalar op the hint recorded, the finalizer table folds the tier output
    to the minimal partial; the tier (dask's chunk kwargs) is never rebound. mean/moment never set the flag (their
    dict partials are mathematically required), so the gate stays kind-based.
    """

    @pytest.mark.parametrize("op,numpy_op", [("sum", np.sum), ("prod", np.prod), ("max", np.max), ("min", np.min)])
    def test_all_scalar_ops_fold_to_scalar(self, op, numpy_op):
        """All four scalar ops compose a finalizer layer; the executed chain returns a true scalar."""
        dask_op = getattr(da, op)
        arr = da.zeros((4, 4), chunks=2)
        chain = _walk_chain(dask_op(arr).__dask_graph__(), _find_agg_layer(dask_op(arr).__dask_graph__()))
        assert chain is not None
        folded = _apply_full_local_fold(op, True, chain)
        # The compositor REPLACES the tier layer (same length), so the chain shape is unchanged.
        assert len(folded) == len(chain)
        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        value = real
        for func, kwargs, _input_count in folded:
            value = func(value) if isinstance(func, functools.partial) else func(value, **kwargs)
        assert np.isclose(float(value), numpy_op(real))

    def test_mean_moment_flag_composes_nothing(self):
        """mean/moment either don't set the flag or have no table entry: the chain stays UNTOUCHED either way.

        Their ``{n, total[, M]}`` dict partials are required by ``mean_agg``/``moment_agg`` -- no eager fold.
        """
        arr = da.zeros((4, 4), chunks=2)
        g = arr.mean().__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        assert chain is not None
        assert _apply_full_local_fold("mean", False, chain) == chain
        assert _apply_full_local_fold("mean", True, chain) == chain  # no 'mean' finalizer entry

    def test_unknown_op_flag_keeps_chain(self):
        """A scalar-kind op with no finalizer entry degrades to the tier partial (still deliverable)."""
        arr = da.zeros((4, 4), chunks=2)
        g = arr.sum().__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        folded = _apply_full_local_fold("opaque-op", True, chain)
        assert folded == chain

    def test_compositor_picklable(self):
        """The replaced tier layer must survive pickle: the DSL-boundary constraint.

        Regression: a lambda-based first draft raised ``Can't pickle <function <lambda>>`` on the bridge.
        """
        import pickle

        arr = da.zeros((4, 4), chunks=2)
        g = arr.sum().__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        folded = _apply_full_local_fold("sum", True, chain)
        compositor = folded[-1][0]
        assert isinstance(compositor, functools.partial)
        restored = pickle.loads(pickle.dumps(compositor))
        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        assert np.isclose(float(restored(real)), real.sum())

    def test_composed_branch_func_full_fold_is_scalar(self):
        """End-to-end: a full-fold branch_func returns a 0-d (scalar) value, not a tier-shaped array."""
        arr = da.zeros((4, 4), chunks=2)
        g = arr.sum().__dask_graph__()
        chain = _walk_chain(g, _find_agg_layer(g))
        folded = _apply_full_local_fold("sum", True, chain)
        branch_func = _build_chain_branch_func(folded)
        real = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        out = branch_func(real)
        assert np.asarray(out).ndim == 0
        assert np.isclose(float(out), real.sum())
