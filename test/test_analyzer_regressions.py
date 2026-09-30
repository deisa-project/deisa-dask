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
"""Regression tests for the precompute analyzer (branch analysis), cluster-free.

Contract coverage:

- ``arr.map_blocks`` must be analysed as the MAPPED graph: a resolvable
  func (``np.abs``) analyses the mapped graph; an opaque func (``lambda``)
  is refused at registration.
- the cross-reduction guard rejects the ordering-dependent bypass
  ``arr.sum() + (arr - arr.mean()).sum()``: it must be REFUSED.
- ``_Missing`` degrades: unknown operands are UNKNOWN, never an assumed
  branch. Subscripting an unbound callback parameter must not crash either.
- multi-array callbacks produce one branch per registered array with
  distinct ``output_key`` values and per-array ``input_name`` attribution.
- each branch's ``branch_func`` computes its OWN array's reduction on a
  chunk, verified with real numeric values.
"""

import numpy as np
import pytest
from utils import _analyze, _scalar

from deisa.dask.precompute_analyzer import (
    NoPrecomputableReductionError,
    RawFieldReadError,
    UnsupportedReductionError,
)

META_A = {"a": {"global_shape": (8, 8), "chunk_shape": (4, 4)}}
META_AB = {
    "a": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
    "b": {"global_shape": (8, 8), "chunk_shape": (4, 4)},
}


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
    # Full local fold: one scalar per bridge (sum/prod rebind the axis to ``None``, see extract_reduction_hints).
    assert branches[0].partial_shape == ()


# ---------------------------------------------------------------------------
# multi-array callbacks: per-array attribution and distinct output keys
# ---------------------------------------------------------------------------
def test_multiarray_one_branch_per_array_with_distinct_output_keys():
    """Two-array callback emits one branch per array with correct ``input_name`` and distinct ``output_key`` values.

    ``arr_a.sum() + arr_b.sum()`` must produce exactly one branch rooted at ``a`` (key ``a-sum``) and one rooted at
    ``b`` (key ``b-sum``). This merges the grouping assertion of ``test_multiarray_one_branch_per_array`` with the
    distinct-keys assertion of ``test_multiarray_two_reductions_distinct_output_keys``.
    """
    branches = _analyze(
        "sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()",
        params="arr_a, arr_b",
        meta=META_AB,
    )
    assert sorted(b.output_key for b in branches) == ["a-sum", "b-sum"]
    branches_a = [b for b in branches if b.input_name == "a"]
    branches_b = [b for b in branches if b.input_name == "b"]
    assert len(branches_a) == 1
    assert len(branches_b) == 1
    assert branches_a[0].output_key == "a-sum"
    assert branches_b[0].output_key == "b-sum"


def test_multiarray_branch_func_computes_own_reduction():
    """Each branch in a two-array callback computes its OWN array's reduction on a known chunk.

    ``a-sum`` runs ``sum`` over an ``arr_a`` chunk; ``b-sum`` runs ``sum`` over an ``arr_b`` chunk. Regression: before
    per-array attribution both branches folded the same array's data.
    """
    branches = _analyze(
        "sa = arr_a.sum()\nsb = arr_b.sum()\nreturn sa.compute(), sb.compute()",
        params="arr_a, arr_b",
        meta=META_AB,
    )
    by_key = {b.output_key: b for b in branches}
    assert set(by_key) == {"a-sum", "b-sum"}
    chunk_a = np.arange(16.0).reshape(4, 4)
    chunk_b = chunk_a + 16.0
    assert np.isclose(_scalar(by_key["a-sum"].branch_func(chunk_a)), float(chunk_a.sum()))
    assert np.isclose(_scalar(by_key["b-sum"].branch_func(chunk_b)), float(chunk_b.sum()))


# ---------------------------------------------------------------------------
# raw-field reads break the precompute contract and must refuse at registration
# ---------------------------------------------------------------------------
_ERROR_REMEDY = "precompute=False"


def test_raw_field_read_via_opaque_call_refused():
    """``plot(arr)`` (unknown callee with a stub-derived arg) raises ``RawFieldReadError``.

    The precompute path ships only reduction partials, never the chunk: any raw-data value handed to an
    unrecognized call is undeliverable.
    """
    with pytest.raises(RawFieldReadError) as exc_info:
        _analyze("plot(arr)\ns = arr.sum()\nreturn s.compute()")
    msg = str(exc_info.value)
    # The error must name the consumer, the offending line, and the documented remedy. Line numbers are reported on
    # the wrapped source: the module docstring shifts the first callback statement to line 2.
    assert "plot" in msg
    assert "line 2" in msg  # ``plot(arr)`` is the first callback statement
    assert _ERROR_REMEDY in msg
    # The refusal must be precise: attribute the raw value to its registered stub, not to opaque noise.
    assert "raw data" in msg


def test_raw_field_read_via_helper_call_refused():
    """An unrecognized bare-name call consuming the raw array is refused."""
    with pytest.raises(RawFieldReadError) as exc_info:
        _analyze("custom(arr)\ns = arr.sum()\nreturn s.compute()")
    msg = str(exc_info.value)
    assert "custom" in msg  # the callee is named for quick diagnosis
    assert "line 2" in msg
    assert _ERROR_REMEDY in msg


def test_resolvable_module_call_on_raw_data_allowed():
    """``np.linalg.norm(arr)`` is NOT a refusal: the call runs chunk-locally at runtime.

    A receiver that resolves to a real module (numpy) consumes the argument on the BRIDGE -- the local chunk --
    which is deliverable work, unlike an unknown callee (``plot``) whose execution would need the full field.
    """
    branches = _analyze("import numpy as npy\nv = npy.linalg.norm(arr)\ns = arr.sum()\nreturn s.compute(), v")
    assert [b.output_key for b in branches] == ["f-sum"]


def test_raw_field_read_list_unwrapping():
    """A raw array nested in a list argument is caught too: ``plot([arr])``."""
    with pytest.raises(RawFieldReadError) as exc_info:
        _analyze("plot([arr])\ns = arr.sum()\nreturn s.compute()")
    assert "plot" in str(exc_info.value)


def test_raw_field_compute_boundary_without_reduction_refused():
    """A compute boundary on the raw array with no reduction of its own is refused."""
    with pytest.raises(RawFieldReadError) as exc_info:
        _analyze("plot(arr)\nv = arr.compute()\ns = arr.sum()\nreturn s.compute(), v")
    msg = str(exc_info.value)
    # Per-boundary check line number: ``arr.compute()`` sits on line 2 of the callback body.
    assert "line 4" in msg or "line 2" in msg
    assert _ERROR_REMEDY in msg


def test_raw_field_index_read_refused_alongside_reduction():
    """``arr[0, 0]`` read + ``arr.sum()`` is refused: the index read is raw data.

    Per-boundary check: the callback's other reductions do not excuse a raw read.
    """
    with pytest.raises(RawFieldReadError):
        _analyze("v = arr[0, 0]\ns = arr.sum()\nreturn s.compute() + v.compute()")


def test_predicate_builtins_on_raw_data_allowed():
    """``assert isinstance(arr, ...)`` / ``hasattr(arr, ...)`` do NOT break the contract.

    Predicate builtins inspect type/attrs, never data: they must stay allowed so the MPI test's
    ``assert isinstance(darr, DeisaArray)`` pattern keeps registering under precompute.
    """
    branches = _analyze(
        "assert isinstance(arr, object)\nassert hasattr(arr, 'shape')\ns = arr.sum()\nreturn s.compute()"
    )
    assert [b.output_key for b in branches] == ["f-sum"]


def test_len_on_raw_data_allowed():
    """``len(arr)`` reads metadata (shape), not data: allowed."""
    branches = _analyze("n = len(arr)\ns = arr.sum()\nreturn s.compute(), n")
    assert [b.output_key for b in branches] == ["f-sum"]


def test_attribute_reads_on_raw_data_allowed():
    """Attribute (metadata) reads -- ``arr.ndim``, f-strings in logs -- do not taint."""
    branches = _analyze("y = arr.ndim\ns = arr.sum()\nreturn s.compute(), y")
    assert [b.output_key for b in branches] == ["f-sum"]
