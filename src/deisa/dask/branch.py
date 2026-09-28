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
#   prior written permission.
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
Branch-level local compute on the bridge.

A :class:`BranchSpec` describes one chunk-local sub-expression of the user's callback that the bridge can execute on
its local numpy chunk and whose result the Deisa-side topic handler combines across bridges.

Branch of **length-1** case: each BranchSpec corresponds to one detected reduction (``arr.sum()``,
``arr.mean(axis=0)``, ...). The branch is a single chunk callable followed by the reduction's combine aggregator. For
multi-layer chains (length->=2 branches); the data structure below is designed to support that without further changes.

The structure mirrors the prior per-reduction branch metadata (``kind`` / ``finalize`` / ``shape`` / ``dtype`` /
``chunk_axis``) so the bridge and Deisa-side combine code paths can be refactored to consume :class:`BranchSpec`
directly without changing semantics.
"""

from __future__ import annotations

import functools
import itertools
import logging
import pickle
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import dask.array as da
from dask import delayed
from dask.array.reductions import mean_agg, moment_agg
from deisa.dask.precompute_analyzer import (
    PrecomputeRuntimeError,
    UnsupportedReductionError,
    _match_source_arrays,
    analyze_callback,
)
from deisa.dask.task_branches import (
    _aggregate_output_feeds_other_reduction,
    _blockwise_indices_inputs,
    _chunk_func_and_kwargs,
    _chunk_layer_for_aggregate,
    _is_aggregate_layer,
    _normalize_reduction_axis,
    _op_for_aggregate_layer,
)
from deisa.dask.task_branches import (
    # Re-exported for the test suite (test_chain.py imports it from here); the ``as`` alias marks the re-export for ruff
    # (F401).
    _find_chunk_layer as _find_chunk_layer,
)
from deisa.dask.utils import build_deisa_array

logger = logging.getLogger(__name__)


# Output kind values. "scalar" / "mean" / "moment" cover the per-reduction cases. Later, may add "scalar-array" for
# chained pointwise + reduction outputs that are 1-d (e.g. ``arr.mean(axis=0)``).
_BRANCH_KIND_SCALAR = "scalar"
_BRANCH_KIND_MEAN = "mean"
_BRANCH_KIND_MOMENT = "moment"


@dataclass
class BranchSpec:
    """One chunk-local sub-expression the bridge can execute + combine.

        Attributes
        ----------
        output_key : str
            Stable identifier for this branch (e.g. ``"f-mean"``). The bridge uses it to namespace its scatter key and
            the Deisa topic handler uses it to route the per-bridge partials back to the same branch.
        input_name : str
            Registered array name the branch is rooted at (e.g. ``"a"``). The Deisa side groups branches per array using
            this field (never by parsing ``output_key``) and files each group with ``set_task_branches`` under its own
            array name via ``execute_callbacks`` -> ``_flush_branches_to_handshake``.
        output_kind : str
            One of ``"scalar"`` / ``"mean"`` / ``"moment"``. Drives the Deisa-side combine graph: ``scalar`` ->
            ``da.stack`` + dask sum, ``mean`` -> ``mean_agg`` over nested list of dicts, ``moment`` -> ``moment_agg`` (+
            ``np.sqrt`` for ``finalize == "sqrt"``).
        branch_func : Callable
            Python callable that, given a numpy chunk, returns the branch's per-bridge partial value (a scalar /
            ndarray / dict). Pickled across the bridge process boundary. Currently a length-1 callable (``chunk_func``
            from the prior branch); This may produce multi-callable composites.
        chunk_axis : Optional[Tuple[int, ...]]
            For reductions, the tuple of axes being reduced in the chunk (e.g. ``(0, 1)`` for full reduction on a 2-D
            chunk). ``None`` for pointwise-only branches.
        finalize : Optional[str]
            ``"sqrt"`` for std (apply ``np.sqrt`` after combining), else ``None``.
        partial_shape : Tuple[int, ...]
            Shape of the **per-bridge partial** (what the branch_func returns). For ``scalar`` reductions on a 2-D chunk
            with ``keepdims=False`` this is ``()``; with ``keepdims=True`` it is ``(1, 1)``. For axis reductions the
            partial keeps the un-reduced axes' full size (e.g. ``mean(axis=0)`` on ``(M, N)`` partial has shape ``(1,
            N)``). The bridge records this on the topic event so the Deisa side knows what each bridge shipped.
        partial_dtype : str
    NumPy dtype string of the per-bridge partial.
    """

    output_key: str
    input_name: str
    output_kind: str
    branch_func: Callable[[Any], Any]
    chunk_axis: Optional[Tuple[int, ...]]
    finalize: Optional[str]
    partial_shape: Tuple[int, ...]
    partial_dtype: str
    op_name: str = ""
    deliver_direct: bool = True
    window_read: bool = False
    reduction_axes: Tuple[int, ...] = ()


def merge_branches(existing: List[BranchSpec], new: List[BranchSpec]) -> List[BranchSpec]:
    """Merge two branch lists, deduping by ``output_key``.

    Identical signatures (same op, same reduction axes) share one bridge execution; same key with a DIFFERENT
    reduction-axes signature (e.g. a window read vs. a true axis reduction over the same axis) always raises
    :class:`PrecomputeRuntimeError` -- the two deliver different shapes and sharing them silently delivers one callback
    the other one's result.
    """
    merged = list(existing)
    seen: Dict[str, BranchSpec] = {b.output_key: b for b in merged}
    for b in new:
        prev = seen.get(b.output_key)
        if prev is None:
            merged.append(b)
            seen[b.output_key] = b
            continue
        if prev.reduction_axes != b.reduction_axes or prev.op_name != b.op_name or prev.output_kind != b.output_kind:
            raise PrecomputeRuntimeError(
                f"merge_branches: array {b.input_name!r} has two reductions sharing output_key "
                f"{b.output_key!r} but different runtime signatures: "
                f"(op={prev.op_name!r}, axes={prev.reduction_axes!r}, kind={prev.output_kind!r}) vs "
                f"(op={b.op_name!r}, axes={b.reduction_axes!r}, kind={b.output_kind!r}). The precompute "
                f"delivery path cannot serve both callbacks from one branch; rename one of the reductions "
                f"or register them on separate arrays."
            )
        # Identical signatures: keep the first branch (dedup).
    return merged


def _analyze_callback_for_branches(callback: Callable, registered_arrays: Dict[str, Any]):
    """Analyze the callback's source and build a list of :class:`BranchSpec` objects.

    Each spec describes a chunk-local sub-expression the bridge can execute. The callback is NOT executed: the AST is
    parsed and walked symbolically to find compute boundaries (``.compute()``, ``client.compute()``, ...) and the dask
    arrays they reference.

    Analysis is strict: any analysis failure (a ``PrecomputeError`` subclass or an unexpected exception) propagates to
    the caller. The caller decides the fallback policy (skip analysis entirely for ``precompute=False`` callbacks,
    raise otherwise).
    """

    # Build a dask array stub matching the registered array's shape/chunks so the symbolic AST walker has something
    # concrete to operate on. The chunking does not matter for hint extraction. We only read the task graph structure,
    # not the data.
    stubs = {}
    for arr_name in registered_arrays:
        # Get metadata for this array from registered_arrays (which is arrays_metadata), used for shape/chunks of the
        # placeholder stub.
        meta = registered_arrays.get(arr_name, {})
        global_shape = meta.get("global_shape")
        # The metadata exposes ``chunk_shape`` (per-bridge chunk), not a ``chunks`` tuple. ``da.zeros`` tiles the global
        # shape into uniform chunks of ``chunk_shape``.
        chunk_shape = meta.get("chunk_shape")
        if global_shape is not None and chunk_shape is not None:
            array_stub = da.zeros(global_shape, chunks=chunk_shape, dtype=np.float64, name=f"deisa-stub-{arr_name}")
        else:
            # Fallback for arrays without full metadata (e.g. opaque helpers or dynamically constructed arrays). The
            # stub's exact size doesn't matter. Only the graph structure matters for precompute analysis. Shape and
            # chunks must be consistent (same dimensionality).
            array_stub = da.zeros((10, 10), chunks=(5, 5), dtype=np.float64, name=f"deisa-stub-{arr_name}")
        # Wrap in DeisaArray so the analyzer sees the full attribute surface (.t, .timestep, dask Array methods) that
        # the callback uses at runtime.
        stubs[arr_name] = build_deisa_array(array_stub, timestep=0)

    return _analyze_branch(callback, registered_arrays=stubs)


# Scalar-kind (``sum``/``prod``/``max``/``min``) elementwise fold for the first combine stage: scalar-axis partials
# are plain arrays (reduced axes already dropped), combined by binary-ufunc fold over identical shapes.
_SCALAR_FOLDS = {
    "sum": lambda entries: np.sum(entries, axis=0),
    "prod": lambda entries: np.prod(entries, axis=0),
    "max": lambda entries: np.max(entries, axis=0),
    "min": lambda entries: np.min(entries, axis=0),
}


def _nest_partial_dicts_by_grid(
    partials: List[Dict[str, Any]], grid_extent: Optional[Tuple[int, ...]] = None
) -> Tuple[Any, Tuple[int, ...]]:
    """Arrange per-bridge dict-blob partials into a nested list that mirrors
    the MPI chunk grid, so that ``mean_agg`` / ``moment_agg`` (which walk the nested list with ``_concatenate2``) can
    combine them.

    ``partials`` is a list of dicts each carrying a ``chunk_position`` -- the bridge's MPI coords. Returns
    ``(nested_list, grid_shape)`` where ``grid_shape`` is the MPI grid shape (``(N, M, ...)``) and
    ``nested_list[i_0][i_1]...`` is the dict (or future-of-dict) at MPI coords ``(i_0, i_1, ...)``.

    ``grid_extent`` (per data axis, from ``global_shape // chunk_shape``) is validated against the coords when
    provided: a mismatch means the grid layout the partials describe contradicts the array metadata, which would
    silently corrupt any axis combine -- raise instead (F5).

    For a 1-D MPI grid (e.g. ``(2,)`` or ``(4,)``) this returns a flat list of length N. For higher-D grids the list is
    nested.
    """
    # Determine grid shape from the unique coords across all partials.
    coords = [tuple(p["chunk_position"]) for p in partials]
    if not coords:
        raise ValueError("_nest_partial_dicts_by_grid: no partials provided")
    ndim = len(coords[0])
    # Validate uniform ndim
    for c in coords:
        if len(c) != ndim:
            raise ValueError(f"_nest_partial_dicts_by_grid: partials have inconsistent coord dimensions: {coords}")
    # Per-axis sizes
    axis_sizes: Dict[int, set] = {ax: set() for ax in range(ndim)}
    for c in coords:
        for ax, v in enumerate(c):
            axis_sizes[ax].add(v)
    grid_shape = tuple(len(axis_sizes[ax]) for ax in range(ndim))
    if grid_extent is not None and tuple(grid_extent) != grid_shape:
        raise PrecomputeRuntimeError(
            f"_nest_partial_dicts_by_grid: partials' chunk grid {grid_shape} contradicts the array metadata "
            f"grid {tuple(grid_extent)}. The grid<->data axis mapping is ambiguous; refusing to combine "
            f"rather than silently producing a wrong reduction."
        )
    # Validate full grid is filled
    expected = set(itertools.product(*(range(g) for g in grid_shape)))
    actual = set(coords)
    if actual != expected:
        raise ValueError(
            f"_nest_partial_dicts_by_grid: partials do not cover the full grid (expected {expected}, got {actual})"
        )

    # Build a map from coords -> dict (or future)
    by_coord: Dict[Tuple[int, ...], int] = {c: i for i, c in enumerate(coords)}

    def _build_nested(dims_remaining: Tuple[int, ...], prefix: Tuple[int, ...]) -> Any:
        if not dims_remaining:
            return partials[by_coord[prefix]]["future"]
        head, *tail = dims_remaining
        return [_build_nested(tuple(tail), (*prefix, i)) for i in range(head)]

    nested = _build_nested(grid_shape, ())
    return nested, grid_shape


def _flatten_grid_entries(nested: Any) -> List[Any]:
    """Flatten a (possibly nested) grid structure of partial values into a flat list."""
    out: List[Any] = []
    stack = [nested]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _concat_kept_grid(grid: Any, depth: int) -> np.ndarray:
    """np.concatenate a kept-grid (nested over kept axes) along the result axes.

    The Phase-A result for kept-coordinate ``(i0, i1, ...)`` has axes ``(kept data axes in ascending data order)``;
    nesting level ``depth`` of the grid corresponds to result axis ``depth``, so concatenating level by level
    reproduces the full-kept-extent array.
    """
    if isinstance(grid[0], list):
        return np.concatenate([_concat_kept_grid(g, depth + 1) for g in grid], axis=depth)
    return np.concatenate([np.asarray(g) for g in grid], axis=depth)


@delayed
def _combine_two_phase(
    flat_entries: List[Tuple[Tuple[int, ...], Any]],
    kind: str,
    op_name: Optional[str],
    red_axes: Tuple[int, ...],
    kept_axes: Tuple[int, ...],
    grid_shape: Tuple[int, ...],
    out_dtype: str,
    finalize: Optional[str],
) -> np.ndarray:
    """Combine per-bridge partials into the final, correctly-shaped reduction."""
    by_coord = {tuple(c): v for c, v in flat_entries}

    def _full_coord(kept_coord: Tuple[int, ...], red_coord: Tuple[int, ...]) -> Tuple[int, ...]:
        full = [0] * len(grid_shape)
        for pos, ax in enumerate(kept_axes):
            full[ax] = kept_coord[pos]
        for pos, ax in enumerate(red_axes):
            full[ax] = red_coord[pos]
        return tuple(full)

    def _build_sub(idx: int, red_prefix: Tuple[int, ...], kept_coord: Tuple[int, ...]) -> Any:
        if idx == len(red_axes):
            return by_coord[_full_coord(kept_coord, red_prefix)]
        ax = red_axes[idx]
        return [_build_sub(idx + 1, red_prefix + (i,), kept_coord) for i in range(grid_shape[ax])]

    per_kept_coord: Dict[Tuple[int, ...], np.ndarray] = {}
    for kept_coord in itertools.product(*(range(grid_shape[ax]) for ax in kept_axes)):
        sub = _build_sub(0, (), kept_coord)
        if kind in ("mean", "moment"):
            if kind == "mean":
                res = mean_agg(sub, dtype=np.dtype(out_dtype), axis=red_axes)
            else:
                res = moment_agg(sub, order=2, ddof=0, dtype=np.dtype(out_dtype), axis=red_axes)
            per_kept_coord[kept_coord] = np.asarray(res)
        elif kind == "scalar":
            fold = _SCALAR_FOLDS.get(op_name or "")
            if fold is None:
                raise PrecomputeRuntimeError(f"_combine_two_phase: no scalar fold for op {op_name!r}")
            entries = np.stack([np.asarray(v) for v in _flatten_grid_entries(sub)])
            res = np.asarray(fold(entries))
            # Scalar chunk funcs run with keepdims=True (dask's chunk stage), so the RED axes survive as size-1 dims at
            # their data positions. Drop them so the combined result's shape matches the declared kept-extent out_shape
            # (and concatenates cleanly).
            for ax in sorted(red_axes, reverse=True):
                res = np.squeeze(res, axis=ax)
            per_kept_coord[kept_coord] = res
        else:
            raise PrecomputeRuntimeError(f"_combine_two_phase: unknown kind {kind!r}")

    # Second stage: concatenate the per-kept-coordinate results along the kept levels; no kept axes -> single result.
    if not kept_axes:
        result = per_kept_coord[()]
    else:
        kept_grid_shape = tuple(grid_shape[ax] for ax in kept_axes)

        def _build_kept_grid(idx: int, prefix: Tuple[int, ...]) -> Any:
            if idx == len(kept_axes):
                return per_kept_coord[prefix]
            return [_build_kept_grid(idx + 1, prefix + (i,)) for i in range(kept_grid_shape[idx])]

        result = _concat_kept_grid(_build_kept_grid(0, ()), 0)

    if finalize == "sqrt":
        result = np.sqrt(result)
    return result


def _combine_array_from_partials(
    partials: List[Dict[str, Any]],
    kind: str,
    finalize: Optional[str],
    hint_axis: Optional[Tuple[int, ...]],
    array_ndim: int,
    op_name: Optional[str] = None,
    global_shape: Optional[Tuple[int, ...]] = None,
    grid_extent: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Build a single-block dask array that combines per-bridge partials."""
    if not partials:
        raise ValueError("_combine_array_from_partials: no partials provided")
    nested, grid_shape = _nest_partial_dicts_by_grid(partials, grid_extent=grid_extent)
    out_dtype = str(partials[0]["dtype"])

    # Grid <-> data axis mapping must be explicit: each grid level is one data axis (the harness's cart dims == array
    # ndim invariant).
    if len(grid_shape) != array_ndim:
        raise PrecomputeRuntimeError(
            f"_combine_array_from_partials: chunk grid {grid_shape} has {len(grid_shape)} levels but the "
            f"array has {array_ndim} data axes; cannot map grid axes to data axes. Refusing to combine."
        )

    # Red axes = the data axes being reduced. ``hint_axis is None`` is the legacy no-axis form of a full reduction
    # (reduce all data axes).
    if hint_axis is None:
        red_axes = tuple(range(array_ndim))
    else:
        red_axes = tuple(int(a) for a in hint_axis)
        for ax in red_axes:
            if not 0 <= ax < array_ndim:
                raise PrecomputeRuntimeError(
                    f"_combine_array_from_partials: reduction axis {ax} out of range for {array_ndim}-D array"
                )
    kept_axes = tuple(ax for ax in range(array_ndim) if ax not in red_axes)

    # Compute the combined output shape.
    if global_shape is not None:
        out_shape = tuple(int(global_shape[ax]) for ax in kept_axes)
    else:
        # Legacy fallback (no metadata): derive from the per-bridge partial.
        if hint_axis is None:
            out_shape = tuple(partials[0]["shape"])
        else:
            out_shape = tuple(partials[0]["shape"][ax] for ax in kept_axes)

    flat_entries = [(tuple(p["chunk_position"]), p["future"]) for p in partials]
    return da.from_delayed(
        _combine_two_phase(
            flat_entries,
            kind=kind,
            op_name=op_name,
            red_axes=red_axes,
            kept_axes=kept_axes,
            grid_shape=grid_shape,
            out_dtype=out_dtype,
            finalize=finalize,
        ),
        shape=out_shape,
        dtype=out_dtype,
    )


def _discover_partial_metadata(
    branch_func: Callable[[Any], Any],
    placeholder: Optional[Any],
    branch: Dict[str, Any],
) -> Tuple[Tuple[int, ...], str]:
    """Run ``branch_func`` on the placeholder and return ``(partial_shape, partial_dtype)``."""
    if placeholder is None:
        partial_shape: Tuple[int, ...] = tuple(branch.get("shape") or ())
        partial_dtype = str(branch.get("dtype", "float64"))
    else:
        sample = branch_func(placeholder)
        if isinstance(sample, dict):
            # mean / moment : the per-bridge partial is a dict with per-key shape. Pick ``total`` as the representative
            # (it always has the reduction-output shape).
            if "total" in sample:
                rep = np.asarray(sample["total"])
            elif "M" in sample:
                rep = np.asarray(sample["M"])
            else:
                rep = np.asarray(next(iter(sample.values())))
            partial_shape = tuple(rep.shape)
            partial_dtype = str(rep.dtype)
        else:
            arr = np.asarray(sample)
            partial_shape = tuple(arr.shape)
            partial_dtype = str(arr.dtype)
    return partial_shape, partial_dtype


def _analyze_branch(callback: Callable, registered_arrays: Dict[str, Any]) -> List[BranchSpec]:
    """Walk the callback's dask graph and emit a :class:`BranchSpec` per branch."""
    # Single AST walk: analyze_callback returns reduction hints AND the walker's dask_arrays in one pass. The
    # dask_arrays are the walker's expressions at each compute boundary (e.g. ``(arr*arr).sum()``); the registered
    # placeholders only have the root layer, so the chain walker needs these.
    hints, walker_dask_arrays = analyze_callback(callback, registered_arrays)

    if not hints:
        return []

    # Candidate map keyed by (array_name, op_name): from ALL walker graphs, each aggregate layer maps its canonical op
    # name to its (layer, graph). Keying by array avoids collisions when two arrays run the same reduction (each folds
    # its own chain).
    aggregate_candidates: Dict[Tuple[str, str], List[Tuple[str, Any]]] = {}
    for arr_info in walker_dask_arrays:
        candidate = arr_info.get("array")
        if not hasattr(candidate, "__dask_graph__"):
            continue
        graph = candidate.__dask_graph__()
        matched = _match_source_arrays(candidate, registered_arrays)
        candidates_array_name = matched[0] if matched else next(iter(registered_arrays), "f")
        for layer_name in graph.layers:
            if not _is_aggregate_layer(layer_name):
                continue
            op_name = _op_for_aggregate_layer(graph, layer_name)
            if op_name is None:
                continue
            aggregate_candidates.setdefault((candidates_array_name, op_name), []).append((layer_name, graph))

    # Fold each hint with its OWN aggregate layer (matched by op_name via the candidate map) -- folding a shared chain
    # into every hint corrupts multi-reduction callbacks. ``_seen_chains`` memoizes identical (agg-name, length) pairs
    # so identical expressions reuse one partial.
    _seen_chains: Dict[Tuple[str, int], Any] = {}

    branches: List[BranchSpec] = []
    for branch_dict in hints:
        # Cross-array expressions cannot be rebuilt from one array's chunk-local partials (the bridge only owns its
        # array's chunk): refuse them.
        if branch_dict.get("multi_source"):
            raise UnsupportedReductionError(
                f"Cannot precompute reduction {branch_dict.get('output_key')!r} "
                f"(op {branch_dict.get('op_name')!r}): the expression descends from registered array "
                f"{branch_dict.get('array_name')!r} AND at least one other registered array. A chunk-local branch "
                f"cannot compute a cross-array expression because each bridge only owns its own array's chunk. "
                f"Redesign the callback so every reduction descends from exactly one registered array, or register "
                f"with precompute=False to use the legacy full-chunk scatter path."
            )

        try:
            chunk_func = pickle.loads(branch_dict["chunk_func_pickle"])
        except Exception:  # pragma: no cover - safety net
            raise

        # Resolve THIS hint's own registered-array stub (placeholder and ndim are per-array in multi-array callbacks).
        # The placeholder is the dask array's first chunk, materialized via .compute() so the numpy ops return numpy
        # values.
        hint_arr = branch_dict.get("array_name") or next(iter(registered_arrays), None)
        array_stub = registered_arrays.get(hint_arr)
        array_ndim = int(getattr(array_stub, "ndim", 0)) if array_stub is not None else 0
        placeholder = array_stub
        if hasattr(placeholder, "compute") and array_stub is not None:
            try:
                # Materialize the FIRST CHUNK of the stub, not the whole array: the registered partial metadata must
                # describe the real per-bridge chunk partial (a whole-stub run would record ``(1, 8)`` where bridges
                # ship ``(1, 4)``).
                first_chunk_shape = tuple(int(c[0]) for c in array_stub.chunks)
                placeholder = array_stub[tuple(slice(0, s) for s in first_chunk_shape)]
                # Compute on the sync scheduler -- analysis must never touch an ambient distributed client (a stale
                # config would block in its reconnect loop instead of failing fast).
                placeholder = placeholder.compute(scheduler="sync")
            except Exception:
                # If .compute() fails (e.g. no dask client in the CI worker process), fall back to a synthetic numpy
                # array that matches the dask array's shape/dtype/chunks so the chunk_func still produces a
                # representative partial.
                try:
                    placeholder = np.zeros(
                        getattr(array_stub, "shape", (4, 4)),
                        dtype=getattr(array_stub, "dtype", np.float64),
                    )
                except Exception:
                    placeholder = np.zeros((4, 4), dtype=np.float64)

        branch = _try_chain_branch(
            branch=branch_dict,
            array_ndim=array_ndim,
            placeholder=placeholder,
            aggregate_candidates=aggregate_candidates,
            seen_chains=_seen_chains,
        )
        if branch is None:
            # Chain walker refused; fall back to the length-1 path with the ORIGINAL branch dict. The fallback's
            # ``deliver_direct`` is conservative: direct ONLY if every candidate's chunk stage reads from the registered
            # root (see the gate in deisa.py).
            direct, window_read = _candidate_chain_classify(branch_dict, aggregate_candidates)
            branch = _try_length1_branch(
                branch=branch_dict,
                chunk_func=chunk_func,
                array_ndim=array_ndim,
                placeholder=placeholder,
                deliver_direct=direct,
                window_read=window_read,
            )
        if branch is None:
            # Both paths returned None. This shouldn't happen for hints that came out of the analyzer (the length-1 path
            # is supposed to always succeed). Raise defensively.
            raise RuntimeError(
                f"analyze_branch: cannot build branch for branch {branch_dict.get('output_key')!r}. "
                f"The chain walker refused (likely cross-array or constant upstream) AND the length-1 fallback's "
                f"build raised. This usually means the chunk_func rejected the placeholder. "
                f"Inspect with the failing branch's chunk_kwargs."
            )
        branches.append(branch)
    return branches


def _candidate_chain_classify(branch: Dict[str, Any], aggregate_candidates: Dict) -> Tuple[bool, bool]:
    """Conservative ``(deliver_direct, window_read)`` for the length-1 fallback.

    The chain walker can't tell which candidate aggregate belongs to THIS hint when several reductions share
    ``(array_name, op_name)`` (e.g. ``arr.sum()`` + ``arr.sum(axis=0)``). The reduction is deemed DIRECT only when
    EVERY candidate's chain from the chunk stage to the registered root contains just the reduction chunk stage and/or
    window-read getitem layers (``root[-1].op()`` -- DataFrame.region Python-level list indexing). Zero candidates, an
    unwalkable chain, a pointwise chain (``arr*arr``), or a real slice (``arr[2:5]`` / ``arr[:, 0]``) -> not direct
    (refused at registration) so a chained reduction can never sneak past the gate as "direct".

    ``window_read`` is True when every candidate is a window read (the callback's runtime reduction runs on the WHOLE
    delivered array, so its ``reduction_axes`` is the FULL reduction (``()``) regardless of the stub-side chunk axis).
    """
    op_name = branch.get("op_name")
    array_name = branch.get("array_name")
    if op_name is None:
        return False, False
    candidates = aggregate_candidates.get((array_name, op_name), [])
    if not candidates:
        return False, False
    direct = True
    window_read = True
    for agg_name, graph in candidates:
        d, w = _chain_direct_and_window_read(graph, agg_name)
        if not d:
            direct = False
        if not w:
            window_read = False
    return direct, window_read


def _chain_direct_and_window_read(graph, agg_name: str) -> Tuple[bool, bool]:
    """Classify one aggregate's reduction-input chain.

    Walks from the reduction's chunk stage toward the registered root. The chain is DIRECT when every layer between the
    chunk stage and the root is either the chunk stage itself or a window-read getitem layer; ANY other layer
    (pointwise blockwise like ``mul``, a real slice, an unwalkable input) makes it non-direct (``(False, ...)``).
    Returns ``(direct, window_read)`` where ``window_read`` is True when the chain contains a window-read getitem (and
    is otherwise direct).
    """
    from deisa.dask.task_branches import (
        _chain_has_window_read,
        _chunk_layer_for_aggregate,
        _is_stub_layer_name,
        _is_window_read_layer,
        _window_read_upstream_name,
    )

    chunk_layer_name = _chunk_layer_for_aggregate(graph, agg_name)
    if chunk_layer_name is None:
        return False, False
    window_read = _chain_has_window_read(graph, chunk_layer_name)
    current = chunk_layer_name
    seen = set()
    while current is not None and current not in seen:
        seen.add(current)
        layer = graph.layers[current]
        if _is_stub_layer_name(current):
            break  # reached the registered-array root stub
        if _is_window_read_layer(layer):
            upstream = _window_read_upstream_name(layer)
            if upstream is None:
                return False, window_read
            if upstream not in graph.layers:
                break
            current = upstream
            continue
        # Ordinary layer: the reduction chunk stage is the ONLY allowed one.
        if current != chunk_layer_name:
            return False, window_read
        upstream = _find_single_upstream(layer)
        if upstream is None:
            return False, window_read
        upstream_name, _ = upstream
        if upstream_name not in graph.layers:
            break  # root data node
        current = upstream_name
    return True, window_read


def _try_chain_branch(
    branch: Dict[str, Any],
    array_ndim: int,
    placeholder: Optional[Any],
    aggregate_candidates: Dict[Tuple[str, str], List[Tuple[str, Any]]],
    seen_chains: Dict[Tuple[str, int], Any],
) -> Optional[BranchSpec]:
    """Try to fold the branch's reduction into a chain-folded BranchSpec."""
    op_name = branch.get("op_name")
    array_name = branch.get("array_name")
    if op_name is None:
        return None
    candidates = aggregate_candidates.get((array_name, op_name), [])
    if len(candidates) != 1:
        return None
    agg_name, graph = candidates[0]
    chain = _walk_chain(graph, agg_name)
    if chain is None:
        return None
    # The walker returns layers from root-to-chunk. The chain already covers the pointwise steps. The reduction's
    # chunk_func is the last step. We build a single branch_func via ``_build_chain_branch_func``. If a memoized branch
    # exists for this chain, reuse it.
    chain_key = (agg_name, len(chain))
    chain_branch_func = seen_chains.get(chain_key)
    if chain_branch_func is None:
        chain_branch_func = _build_chain_branch_func(chain)
        seen_chains[chain_key] = chain_branch_func
    try:
        return _build_branch(
            branch=branch,
            array_ndim=array_ndim,
            placeholder=placeholder,
            chain_branch_func=chain_branch_func,
            deliver_direct=(len(chain) == 1),
            window_read=False,
        )
    except Exception:
        return None


def _try_length1_branch(
    branch: Dict[str, Any],
    chunk_func: Callable,
    array_ndim: int,
    placeholder: Optional[Any],
    deliver_direct: bool = False,
    window_read: bool = False,
) -> Optional[BranchSpec]:
    """Build a length-1 BranchSpec from a per-reduction branch.

    A length-1 branch is a chain of length 1: represented as ``[(chunk_func, effective_kwargs, 1)]`` and built via the
    unified :func:`_build_branch`. The chain machinery binds the chunk_kwargs (axis, keepdims, dtype, ...) to the
    chunk_func, so the bridge calls ``branch_func(chunk)`` with just the chunk and no extra kwargs.

    ``deliver_direct`` defaults to False because the length-1 path is reached exactly when the chain walker could not
    PROVE the reduction reads the registered root directly (no/ambiguous candidates, unwalkable chain); the caller
    computes the conservative value via :func:`_candidate_chain_classify`. ``window_read`` marks a whole-row-plane
    ``root[-1]`` read (see :mod:`deisa.dask.task_branches`): the callback's runtime reduction runs on the WHOLE
    delivered array, so the branch's ``reduction_axes`` is the full reduction (``()``) even though the stub-side
    chunk axis is partial.
    """
    try:
        # For ``mean`` and ``moment`` the bridge overrides ``keepdims=True`` (see
        # :meth:`Bridge._execute_operations_on_chunk`) so the per-bridge dict values are at least 1-D for ``mean_agg`` /
        # ``moment_agg`` to walk with ``_concatenate2``. Mirror that here.
        chunk_kwargs = branch.get("chunk_kwargs") or {}
        effective_kwargs = dict(chunk_kwargs)
        if branch.get("kind") in (_BRANCH_KIND_MEAN, _BRANCH_KIND_MOMENT):
            effective_kwargs["keepdims"] = True
        chain = [(chunk_func, effective_kwargs, 1)]
        return _build_branch(
            branch=branch,
            chain=chain,
            array_ndim=array_ndim,
            placeholder=placeholder,
            deliver_direct=deliver_direct,
            window_read=window_read,
        )
    except Exception as e:
        # Length-1 is the last-resort fallback; log context for diagnosis (usually a chunk_func/placeholder shape-dtype
        # mismatch).
        logger.debug(
            "_try_length1_branch: build_branch raised for %s with chunk_kwargs=%r, array_ndim=%d, placeholder=%r: %s",
            branch.get("output_key"),
            branch.get("chunk_kwargs"),
            array_ndim,
            type(placeholder).__name__ if placeholder is not None else None,
            e,
        )
        return None


def _build_branch(
    branch: Dict[str, Any],
    chain: Optional[List[Tuple[Callable, dict, int]]] = None,
    array_ndim: int = 0,
    placeholder: Optional[Any] = None,
    chain_branch_func: Optional[Callable] = None,
    deliver_direct: Optional[bool] = None,
    window_read: bool = False,
) -> BranchSpec:
    """Build a :class:`BranchSpec` from a branch and a layer chain.

    The chain (root-to-chunk ``(func, kwargs, input_count)`` triples) is composed into a single ``branch_func``; the
    branch provides the reduction's ``kind``/``finalize``/``chunk_axis`` metadata. A length-1 branch is simply a chain
    of length 1 (``[(chunk_func, effective_kwargs, 1)]``).

    If ``chain_branch_func`` is provided (the memoized / pre-built version), use it directly instead of rebuilding.

    ``deliver_direct`` records whether the reduction's chunk stage reads DIRECTLY from the registered array root (a
    plain ``arr.<op>()`` call, possibly via ``window[-1]``) rather than from a pointwise chain or slice
    (``(arr*arr).sum()``, ``arr[2:5].sum()``). The registration gate in :mod:`deisa.dask.deisa` refuses non-direct
    reductions because the precompute delivery path cannot reconstruct a chain on the callback side. ``None`` defaults
    to ``len(chain) == 1`` (a single layer means the chunk stage is the only layer between root and aggregate).

    ``window_read`` marks a whole-row-plane ``root[-1]`` read: the callback's runtime reduction runs on the WHOLE
    delivered array, so the branch's ``reduction_axes`` is the FULL reduction (``()``) even though the stub-side
    chunk axis is partial.
    """
    kind = branch.get("kind", _BRANCH_KIND_SCALAR)
    finalize = branch.get("finalize")

    chunk_kwargs = branch.get("chunk_kwargs") or {}
    ax = chunk_kwargs.get("axis")
    if isinstance(ax, (list, tuple)):
        chunk_axis = tuple(int(a) for a in ax)
    elif ax is not None:
        chunk_axis = (int(ax),)
    else:
        chunk_axis = None

    if chain_branch_func is None:
        chain_branch_func = _build_chain_branch_func(chain or [])

    partial_shape, partial_dtype = _discover_partial_metadata(chain_branch_func, placeholder, branch)

    # Reduction axes the callback's delivered view will be asked for: what the callback's reduction call on the
    # delivered view must match. A window read runs on the whole array (full reduction, ``()``); any other direct
    # reduction normalizes its chunk axis against the registered array's ndim.
    if window_read:
        reduction_axes: Tuple[int, ...] = ()
    elif chunk_axis is None:
        reduction_axes = ()
    else:
        reduction_axes = _normalize_reduction_axis(chunk_axis, array_ndim)

    return BranchSpec(
        output_key=branch["output_key"],
        input_name=branch["array_name"],
        output_kind=kind,
        branch_func=chain_branch_func,
        chunk_axis=chunk_axis,
        finalize=finalize,
        partial_shape=partial_shape,
        partial_dtype=partial_dtype,
        op_name=branch.get("op_name", ""),
        deliver_direct=(len(chain or []) == 1) if deliver_direct is None else deliver_direct,
        window_read=window_read,
        reduction_axes=reduction_axes,
    )


def _walk_chain(graph, agg_name: str) -> Optional[List[Tuple[Callable, dict, int]]]:
    """Walk from a chunk layer back to the placeholder root."""
    if not _is_aggregate_layer(agg_name):
        return None
    # Match the chunk layer via the dedicated chunk/aggregate pairs (mean_chunk / mean_agg, chunk_max / max, chunk_min /
    # min, moment_agg / var) instead of requiring an exact base name.
    chunk_layer_name = _chunk_layer_for_aggregate(graph, agg_name)
    if chunk_layer_name is None:
        return None
    # Refuse to fold an aggregate whose output feeds ANOTHER reduction's chunk stage (cross-reduction expression like
    # ``(arr - arr.mean()).sum()``). Such an aggregate's global value is required downstream; folding it alone would
    # produce a wrong local partial.
    if _aggregate_output_feeds_other_reduction(graph, agg_name):
        return None
    chain: List[Tuple[Callable, dict, int]] = []
    current = chunk_layer_name
    seen = set()
    while current is not None and current not in seen:
        seen.add(current)
        layer = graph.layers[current]
        if "-aggregate-" in current:
            break
        chunk_info = _chunk_func_and_kwargs(layer)
        if chunk_info is None:
            break
        func, kwargs = chunk_info
        upstream = _find_single_upstream(layer)
        if upstream is None:
            return None
        upstream_name, upstream_input_count = upstream
        # If the upstream isn't a layer in the graph, we've hit the root (placeholder DataNode). The branch_func
        # receives the actual chunk from the bridge, so we don't fold the root.
        if upstream_name not in graph.layers:
            break
        chain.append((func, kwargs, upstream_input_count))
        current = upstream_name
    chain.reverse()
    return chain


def _find_single_upstream(layer) -> Optional[Tuple[str, int]]:
    """Return ``(upstream_layer_name, array_input_count)`` if the layer reads from a single upstream Blockwise
    (one or more times. e.g. ``arr * arr`` reads from ``arr`` twice and is still chunk-local). Returns ``None`` if the
    layer reads from multiple distinct array upstreams (cross-array, can't fold) or contains scalar constants (deferred
    to a later commit).

    Uses the shared Blockwise index-walking primitive :func:`deisa.dask.task_branches._blockwise_indices_inputs` so the
    ``layer.indices`` parsing lives in one place.
    """
    parsed = _blockwise_indices_inputs(layer)
    if parsed is None:
        # Not a new-style Blockwise (no indices) / empty indices.
        return None
    names, array_input_count, has_non_array_input = parsed
    upstream_names = set(names)
    if not upstream_names or len(upstream_names) > 1:
        return None
    if has_non_array_input:
        # Refuse to fold chains with constants. The constant would need to be embedded into the branch_func call (e.g.
        # ``add(chunk, 1)``) which requires extracting the constant value from the Blockwise task's args.
        return None
    return next(iter(upstream_names)), array_input_count


def _build_chain_branch_func(chain: List[Tuple[Callable, dict, int]]) -> Callable[[Any], Any]:
    """Compose a list of ``(func, kwargs, input_count)`` into a single branch_func(chunk).
    Layers reading from a single upstream twice (e.g. ``arr * arr``) get the chunk passed twice.

    Returns a module-level callable (``_chain_branch_func``) bound to the chain tuple via :func:`functools.partial`.
    The closure is on a top-level function so pickle can find it across processes. Building a fresh ``def
    branch_func(chunk, _chain=...)`` inside this helper would produce an unpicklable local function (AttributeError:
    Can't get local object). The ``functools.partial`` + module-level target recipe is the only shape that pickles
    cleanly.
    """
    chain_tuple = tuple(chain)
    return functools.partial(_chain_branch_func, _chain=chain_tuple)


def _chain_branch_func(chunk, _chain=None):
    """Module-level branch callable: apply each (func, kwargs, input_count) in the chain to the chunk,
    threading the result through.

    Pair with :func:`_build_chain_branch_func` which binds ``_chain`` via :func:`functools.partial`. Defined at module
    level so pickle can find it across the bridge process boundary.
    """
    if _chain is None:
        raise RuntimeError("_chain_branch_func called without bound _chain")
    x = chunk
    for func, kwargs, input_count in _chain:
        if input_count == 1:
            x = func(x, **kwargs)
        elif input_count == 2:
            x = func(x, x, **kwargs)
        else:
            # For now refuse; can be extended for N-ary pointwise.
            raise ValueError(f"chain has layer with {input_count} inputs; only 1 or 2 supported")
    return x
