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
# * Neither the names of CEA, nor the names of the contributors may be used to
#   endorse or promote products derived from this software without specific
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
"""Regression tests for the precompute analyzer (registration path), cluster-free.

Contract coverage:

- ``arr.map_blocks`` must be analysed as the MAPPED graph: a resolvable
  func (``np.abs``) analyses the mapped graph; an opaque func (``lambda``)
  is refused at registration.
- the cross-reduction guard rejects the ordering-dependent bypass
  ``arr.sum() + (arr - arr.mean()).sum()``: it must be REFUSED.
- ``_Missing`` degrades: unknown operands are UNKNOWN, never an assumed
  branch. Subscripting an unbound callback parameter must not crash either.
- ``register_callback`` must store the callback payload only after the
  analysis succeeds; a failed registration must leak nothing.
"""

import textwrap
from typing import Any, Callable, Dict

import numpy as np
import pytest
from deisa.core import Window

from deisa.dask.branch import _analyze_callback_for_branches
from deisa.dask.deisa import Deisa
from deisa.dask.precompute_analyzer import (
    NoPrecomputableReductionError,
    UnsupportedReductionError,
)

META = {"f": {"global_shape": (8, 8), "chunk_shape": (4, 4)}}
META_A = {"a": {"global_shape": (8, 8), "chunk_shape": (4, 4)}}


def _make_callback(name: str, body: str, params: str = "arr") -> Callable:
    """Compile ``def <name>(<params>): <body>`` and return the function with ``__source__`` set.

    Mirrors the helper in test_chain.py so ``analyze_callback`` can walk the source.
    """
    src = textwrap.dedent(f"def {name}({params}):\n{textwrap.indent(body, '    ')}")
    scope: Dict[str, Any] = {}
    exec(compile(src, f"<analyzer_regression:{name}>", "exec"), scope)
    fn = scope[name]
    fn.__source__ = src  # type: ignore[attr-defined]
    return fn


def _analyze(body: str, params: str = "arr", name: str = "analyze_cb", meta: Dict[str, Any] = META) -> Any:
    cb = _make_callback(name, body, params=params)
    return _analyze_callback_for_branches(cb, meta)


def _scalar(value: Any) -> float:
    """Unwrap a keepdims partial to a scalar."""
    return float(np.asarray(value).reshape(-1)[0])


# ---------------------------------------------------------------------------
# _call_map_blocks must analyze the mapped array, never the receiver
# ---------------------------------------------------------------------------
def test_map_blocks_opaque_lambda_refused():
    """``arr.map_blocks(lambda ...)`` cannot be precomputed: registration refuses.

    An opaque callable cannot be resolved, so the reduction is not precomputable — refuse rather than analyse the
    unmapped receiver.
    """
    with pytest.raises(NoPrecomputableReductionError):
        _analyze("y = arr.map_blocks(lambda b: b * 2)\ns = y.sum()\nreturn s.compute()")


def test_map_blocks_resolvable_function_uses_mapped_array():
    """``arr.map_blocks(np.abs).sum()`` sums the MAPPED chunks (real value).

    The analyzed branch must run the resolvable func on each raw chunk; ``branch_func`` of a ``-2`` chunk sums ``abs``
    of it.
    """
    branches = _analyze("y = arr.map_blocks(np.abs)\ns = y.sum()\nreturn s.compute()")
    assert len(branches) == 1
    chunk = np.full((4, 4), -2.0)
    assert np.isclose(_scalar(branches[0].branch_func(chunk)), abs(chunk).sum())


# ---------------------------------------------------------------------------
# cross-reduction guard must be ordering-independent
# ---------------------------------------------------------------------------
def test_cross_reduction_simple_refused():
    """``(arr - arr.mean()).sum()`` is refused (inner reduction feeds another)."""
    with pytest.raises(UnsupportedReductionError):
        _analyze("s = (arr - arr.mean()).sum()\nreturn s.compute()")


def test_cross_reduction_with_added_mean_refused():
    """``arr.mean() + (arr - arr.mean()).sum()`` is refused."""
    with pytest.raises(UnsupportedReductionError):
        _analyze("s = arr.mean() + (arr - arr.mean()).sum()\nreturn s.compute()")


def test_cross_reduction_ordering_dependent_bypass_refused():
    """``arr.sum() + (arr - arr.mean()).sum()`` is refused.

    The second ``sum``'s guard must inspect ITS OWN subgraph (not the first ``sum``'s, positionally): its chunk depends
    on ``mean_agg`` and must be refused regardless of layer ordering.
    """
    with pytest.raises(UnsupportedReductionError):
        _analyze("s = arr.sum() + (arr - arr.mean()).sum()\nreturn s.compute()")


def test_sibling_reductions_still_emitted():
    """Guard must not over-refuse: two independent reductions stay emitted."""
    branches = _analyze("s = arr.sum()\nm = arr.mean()\nreturn s.compute(), m.compute()")
    assert sorted(b.output_key for b in branches) == ["f-mean", "f-sum"]


# ---------------------------------------------------------------------------
# _Missing comparisons and unbound-param subscripts degrade, never crash
# ---------------------------------------------------------------------------
def test_missing_comparison_walks_both_branches():
    """``if state > 3:`` (state unknown) walks BOTH branches.

    Comparison on an unknown operand must degrade to UNKNOWN and walk both branches, never crash the analysis.
    """
    branches = _analyze("if state > 3:\n    s = arr.sum()\nelse:\n    m = arr.mean()\nreturn s.compute(), m.compute()")
    assert sorted(b.output_key for b in branches) == ["f-mean", "f-sum"]


def test_unbound_param_subscript_does_not_crash():
    """Subscripting an unbound callback parameter is opaque, not a crash.

    ``_apply_subscript`` treats ``_UnboundParam`` as an opaque value; the analysis degrades instead of raising
    ``TypeError``.
    """
    branches = _analyze(
        "x = p[0]\ns = arr.sum()\nreturn s.compute()",
        params="arr, p",
        name="sub_cb",
    )
    assert [b.output_key for b in branches] == ["f-sum"]


# ---------------------------------------------------------------------------
# partial_shape/partial_dtype describe a chunk partial, not the whole array
# ---------------------------------------------------------------------------
def test_partial_metadata_is_chunk_not_whole_array():
    """``arr.sum(axis=0)`` on an (8, 8) array with (4, 4) chunks records (1, 4).

    The partial metadata describes ONE chunk's partial, not the whole array: the value is shipped in the topic event
    and drives ``da.from_delayed``/combine ``out_shape``.
    """
    branches = _analyze("s = arr.sum(axis=0)\nreturn s.compute()")
    assert len(branches) == 1
    assert branches[0].partial_shape == (1, 4), branches[0].partial_shape


def test_partial_metadata_scalar_full_reduction():
    """A full reduction records the chunk partial shape (keepdims=True)."""
    branches = _analyze("s = arr.sum()\nreturn s.compute()")
    assert len(branches) == 1
    assert branches[0].partial_shape == (1, 1)


# ---------------------------------------------------------------------------
# registration stores the payload only after analysis succeeds
# ---------------------------------------------------------------------------
class _FakeClient:
    """Minimal client surface used by Deisa._register_callback_impl."""

    def __init__(self):
        self.subscribed = []

    def subscribe_topic(self, name, handler):  # noqa: D102
        self.subscribed.append(name)

    def close(self):  # noqa: D102
        pass


class _FakeHandshake:
    """Minimal handshake surface used by Deisa._register_callback_impl."""

    def __init__(self):
        self.branches: Dict[str, Any] = {}

    def set_task_branches(self, array_name, hints):  # noqa: D102
        self.branches[array_name] = hints


def _make_deisa_stub() -> Deisa:
    """A Deisa instance with the registration surfaces stubbed (no cluster)."""
    d = Deisa.__new__(Deisa)
    d.client = _FakeClient()
    d.handshake = _FakeHandshake()
    d.arrays_metadata = META_A
    d._callbacks = {}
    d._callbacks_by_array = {}
    d._topic_handlers = {}
    d._callback_reductions = {}
    d._callback_seq = 0
    d._branch_groups = {}
    d._tasks = set()
    d._execute_callbacks_called = False
    return d


def test_registration_success_stores_callback_payload():
    """A successful registration stores the payload in ``_callbacks``.

    The topic handler can only fire a callback whose payload lives in ``_callbacks``; it must be stored once every step
    that can raise has succeeded.
    """
    d = _make_deisa_stub()
    cb = _make_callback("reg_ok", "s = arr.sum()\nreturn s.compute()")
    cid = d._register_callback_impl(cb, [Window("a", size=1)], exception_handler=None, when="AND", precompute=True)
    assert cid in d._callbacks
    assert d._callbacks[cid]["callback"] is cb
    assert d._callbacks[cid]["array_names"] == ["a"]
    assert cid in d._callbacks_by_array["a"]
    # Topic subscription happened for the registered array.
    assert "a" in d.client.subscribed
    # Branches are merged in memory at registration; the handshake actor is
    # filed only by execute_callbacks (the per-cycle filing boundary).
    assert d._branch_groups["a"]
    d._flush_branches_to_handshake()
    assert d.handshake.branches["a"]


def test_registration_failure_leaves_no_trace():
    """A registration whose analysis raises leaks nothing.

    ``_callbacks[callback_id]`` is written only AFTER the analysis; a raising analysis must leave no half-registered
    entry that ``unregister_callback`` could never reach.
    """
    d = _make_deisa_stub()
    cb = _make_callback("reg_fail", "s = (arr - arr.mean()).sum()\nreturn s.compute()")
    with pytest.raises(UnsupportedReductionError):
        d._register_callback_impl(cb, [Window("a", size=1)], exception_handler=None, when="AND", precompute=True)
    assert d._callbacks == {}
    assert d._callbacks_by_array == {}
    assert d.client.subscribed == []
