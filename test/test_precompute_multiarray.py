"""
Unit tests for multi-array precompute: per-array hints / branches / filing.

A callback that reduces TWO registered arrays (``arr_a.sum() + arr_b.sum()``) must
produce ONE branch per array: distinct ``output_key`` (``a-sum`` / ``b-sum``), each
``branch_func`` computing its OWN array's reduction on its OWN array's chunks, and
each ``BranchSpec`` carrying its own ``input_name`` so the Deisa side can file the
branches under the right array. Regression: before per-array attribution, both
reductions were keyed ``a-sum``, the second array got no branches, and the first
array's bridge executed BOTH reduction functions on its own data.

Single-array behavior must stay unchanged (``f-sum``, ``input_name == 'f'``), and
cross-array expressions (``(arr_a - arr_b).max()``) are refused at branch build
under this design because a chunk-local branch cannot rebuild them.
"""

import textwrap
from typing import Any, Callable, Dict

import numpy as np
import pytest

from deisa.dask.branch import _analyze_callback_for_branches
from deisa.dask.precompute_analyzer import UnsupportedReductionError

# Two registered arrays with identical metadata (shape/chunks/dtype). The multi-array
# bug depends on the two arrays being indistinguishable by metadata alone.
META = {
    "a": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
    "b": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
}


def _make_callback(name: str, body: str, params: str = "arr") -> Callable:
    """Compile a small snippet ``def <name>(<params>): <body>`` and return it.

    Mirrors the helper in test_chain.py so ``analyze_callback`` can walk the source.
    ``params`` defaults to the single-array signature; multi-array callbacks pass
    ``params="arr_a, arr_b"``.
    """
    src = textwrap.dedent(f"def {name}({params}):\n{textwrap.indent(body, '    ')}")
    scope: Dict[str, Any] = {}
    code = compile(src, f"<test_precompute_multiarray:{name}>", "exec")
    exec(code, scope)
    fn = scope[name]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


def _analyze(body: str, params: str = "arr", meta: Dict[str, Any] = META) -> Any:
    cb = _make_callback("multiarray_cb", body, params=params)
    return _analyze_callback_for_branches(cb, meta)


def _scalar(value: Any) -> float:
    """Unwrap a (1, 1) keepdims partial (sum chunk layer has keepdims=True) to a scalar."""
    return float(np.asarray(value).reshape(-1)[0])


def test_multiarray_two_reductions_distinct_output_keys():
    """Two-array callback -> distinct output_keys ``a-sum`` and ``b-sum``."""
    branches = _analyze("sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()", params="arr_a, arr_b")
    assert sorted(b.output_key for b in branches) == ["a-sum", "b-sum"]


def test_multiarray_branch_func_computes_own_reduction():
    """Each branch_func equals its OWN array's reduction on a known chunk (real values)."""
    branches = _analyze("sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()", params="arr_a, arr_b")
    by_key = {b.output_key: b for b in branches}
    assert set(by_key) == {"a-sum", "b-sum"}
    chunk_a = np.arange(16.0).reshape(4, 4)
    chunk_b = chunk_a + 16.0
    assert np.isclose(_scalar(by_key["a-sum"].branch_func(chunk_a)), float(chunk_a.sum()))
    assert np.isclose(_scalar(by_key["b-sum"].branch_func(chunk_b)), float(chunk_b.sum()))


def test_multiarray_one_branch_per_array():
    """Grouping: exactly one branch per ``input_name``, each rooted at its own array."""
    branches = _analyze("sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()", params="arr_a, arr_b")
    branches_a = [b for b in branches if b.input_name == "a"]
    branches_b = [b for b in branches if b.input_name == "b"]
    assert len(branches_a) == 1
    assert len(branches_b) == 1
    assert branches_a[0].output_key == "a-sum"
    assert branches_b[0].output_key == "b-sum"


def test_single_array_unchanged():
    """Single-array guard: one branch, output_key ``f-sum``, input_name ``f``."""
    meta = {"f": {"global_shape": (8, 8), "chunk_shape": (4, 4)}}
    branches = _analyze("s = arr.sum()\nreturn s.compute()", meta=meta)
    assert len(branches) == 1
    branch = branches[0]
    assert branch.output_key == "f-sum"
    assert branch.input_name == "f"
    assert branch.output_kind == "scalar"
    chunk = np.arange(16.0).reshape(4, 4)
    assert np.isclose(_scalar(branch.branch_func(chunk)), float(chunk.sum()))


def test_multiarray_mixed_callback_uses_one_array():
    """Mixed callback (only arr_a reduced): all branches rooted at ``a``, none for ``b``."""
    branches = _analyze("sa = arr_a.sum()\nreturn sa.compute()", params="arr_a, arr_b")
    assert len(branches) == 1
    assert branches[0].output_key == "a-sum"
    assert branches[0].input_name == "a"


def test_cross_array_expression_refused():
    """``(arr_a - arr_b).max()`` descends from two arrays -> refused."""
    cb = _make_callback("cross_cb", "r = (arr_a - arr_b).max()\nreturn r.compute()", params="arr_a, arr_b")
    with pytest.raises(UnsupportedReductionError):
        _analyze_callback_for_branches(cb, META)
