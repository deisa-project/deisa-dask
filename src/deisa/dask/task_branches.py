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
Extract reduction hints from a dask array's task graph.

Given a dask array whose graph contains reductions (sum, mean, std, var, max,
min, prod), walk the graph layers to find the aggregate layer and the matching
chunk layer, then build a branch dict the bridge can execute on a local numpy
chunk before scattering.

branch schema (the contract between the analytics side (precompute analyzer) and
the bridge side (local chunk execution)):

.. code-block:: python

    {
        "output_key": "fdistribu-sum",  # unique key for this reduction
        "op_name": "sum",  # canonical op name
        "chunk_func_pickle": ...,  # pickle of the chunk callable
        "chunk_kwargs": {...},  # kwargs for the chunk callable
        "finalize": "sqrt" | None,  # post-step (sqrt for std)
        "array_name": "a",  # registered array the reduction descends from
        "multi_source": False,  # True when the expression descends from >1 registered array
    }

This module is purely about reading the dask graph; it never executes user
callbacks.
"""

from __future__ import annotations

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import dask.array as da

logger = logging.getLogger(__name__)

# Map an aggregator function name to a canonical op name. sum/prod/max/min/amax/amin are simple; mean_agg and moment_agg
# are the two-stage reducers used by mean/var/std.
_OP_FROM_FUNC_NAME = {
    "sum": "sum",
    "prod": "prod",
    "max": "max",
    "min": "min",
    "amax": "max",
    "amin": "min",
    "mean_agg": "mean",
    "moment_agg": "moment",
}

# Operations supported by ``_combine_reduction_partials`` on the bridge side.
SUPPORTED_OPS = {"sum", "mean", "std", "var", "max", "min", "prod"}

# Reduction kinds -- how the bridge scatters the partial and how the Deisa-side combine graph is built.
#
# "scalar": plain scalar/array partials, combined via dask's natural da.stack + .sum(axis=0);
# "mean": {"n", "total"} dict blobs, the Deisa-side graph calls mean_agg over them;
# "moment": {"n", "total", "M"} blobs, calls moment_agg; "std" adds a trailing sqrt via finalize.
_REDUCTION_KIND = {
    "sum": "scalar",
    "prod": "scalar",
    "max": "scalar",
    "min": "scalar",
    "mean": "mean",
    "var": "moment",
    "std": "moment",
}


def _axis_signature(axis) -> Tuple[int, ...]:
    """Sortable signature of the chunk's axis, used ONLY for output-key uniqueness.

    Deliberately ndim-free: the chunk layer's axis is relative to the input
    (root) array, but ``extract_reduction_hints`` sees the reduction OUTPUT's
    ndim (0 for a scalar full reduction). Dask normalizes reduction axes to
    non-negative before building the chunk layer, so no clamping is needed
    here. Full reductions (``(0, 1)`` on 2-D) and explicit covers-all calls
    share the signature --- they are semantically identical.
    """
    if axis is None:
        return ()
    if isinstance(axis, (int, np.integer)):
        return (int(axis),)
    return tuple(sorted(int(a) for a in axis))


def _normalize_reduction_axis(axis, ndim: int) -> Tuple[int, ...]:
    """Normalize a reduction axis to a signature tuple used for dispatch.

    ``None`` (full reduction with no explicit axis) and an axis that covers
    ALL data axes (dask's chunk layer always carries the full axis tuple for
    a no-axis call, e.g. ``(0, 1)`` for a 2-D ``arr.sum()``) both map to the
    empty tuple ``()`` -- they are the same reduction. Any partial axis maps
    to its sorted tuple of non-negative axes.

    - ``:param axis:`` ``None``, an int, or a tuple/list of ints.
    - ``:param ndim:`` The reduced array's dimensionality.
    - ``:return:`` ``()`` for the full reduction, else the sorted axis tuple.
    """
    if axis is None:
        return ()
    if isinstance(axis, (int, np.integer)):
        axes = (int(axis) % ndim,)
    else:
        axes = tuple(sorted(int(a) % ndim for a in axis))
    if axes == tuple(range(ndim)):
        return ()
    return axes


# --------------------------------------------------------------------------- Window-read detection (the ``param[-1]``
# idiom) --------------------------------------------------------------------------- At runtime every registered-array
# callback parameter is a list of DeisaArrays (the sliding window), so ``param[-1]`` is Python list indexing returning
# the CURRENT iteration's delivered array. The analyzer models the subscript as a dask getitem, but the delivered view
# is the WHOLE array: such getitems must be treated as a WINDOW READ, not a real slice (``arr[2:5]`` / ``arr[:, 0]`` are
# real slices, refused at registration).
#
# A getitem layer produced by ``stub[-1]`` has tasks whose index is an int in the first position and full slices
# elsewhere -- that shape is the window-read signature.
def _is_window_read_index(index) -> bool:
    """True when ``index`` selects a whole row-plane (the ``param[-1]`` idiom).

    An int, or a tuple whose first element is an int and every other element
    is a full slice (``slice(None)`` / ``None``). This is the shape produced
    by ``stub[-1]`` / ``stub[-1, :]``; ``stub[2:5]`` (slice first), ``stub[:,
    0]`` (int in a non-first position) and ``stub[-1, 0]`` (int in a
    non-first position as well) are NOT window reads.
    """
    if isinstance(index, (int, np.integer)):
        return True
    if isinstance(index, tuple) and index:
        first, rest = index[0], index[1:]
        if not isinstance(first, (int, np.integer)):
            return False
        for s in rest:
            if s is None:
                continue
            if not isinstance(s, slice) or (s.start, s.stop, s.step) != (None, None, None):
                return False
        return True
    return False


def _is_window_read_layer(layer) -> bool:
    """True when ``layer`` is a MaterializedLayer of ``getitem`` tasks whose
    index is a whole-row-plane selection (see :func:`_is_window_read_index`).

    Every task in the layer must be a window-read getitem; the layer must
    read from exactly one upstream array.
    """
    mapping = getattr(layer, "mapping", None)
    if mapping is None:
        return False
    upstream_names = set()
    for value in mapping.values():
        func = getattr(value, "func", None)
        if getattr(func, "__name__", "") != "getitem":
            return False
        args = getattr(value, "args", None)
        if not args or len(args) < 2:
            return False
        if not _is_window_read_index(args[1]):
            return False
        # The array operand is the first task argument (a TaskRef key).
        ref = args[0]
        key = getattr(ref, "key", ref)
        if isinstance(key, (list, tuple)) and key:
            upstream_names.add(key[0])
        elif isinstance(key, str):
            upstream_names.add(key)
    return len(upstream_names) == 1


def _window_read_upstream_name(layer) -> Optional[str]:
    """Return the single upstream layer name of a window-read getitem layer
    (the array operand's layer), or ``None`` if the layer cannot be read.
    """
    mapping = getattr(layer, "mapping", None)
    if not mapping:
        return None
    for value in mapping.values():
        args = getattr(value, "args", None)
        if not args:
            return None
        ref = args[0]
        key = getattr(ref, "key", ref)
        if isinstance(key, (list, tuple)) and key:
            return str(key[0])
        if isinstance(key, str):
            return key
        return None
    return None


# Root stub layers created by the analyzer (``_analyze_callback_for_branches`` names the placeholder
# ``deisa-stub-<array>``).
_STUB_LAYER_PREFIX = "deisa-stub-"


def _is_stub_layer_name(layer_name: str) -> bool:
    """True when the layer is the analyzer's registered-array root stub."""
    return layer_name.startswith(_STUB_LAYER_PREFIX)


def _chain_has_window_read(graph, chunk_layer_name: str) -> bool:
    """True when the reduction's chunk stage reads the root through ONLY
    whole-row-plane getitem layers (the window-read idiom).

    Walks upstream from ``chunk_layer_name``. Any layer that is neither the
    reduction chunk stage nor a window-read getitem layer (a pointwise op, a
    real slice, an unwalkable input, ...) makes the chain NOT a window read
    (conservative: such chains are refused at registration anyway).

    - ``:param graph:`` The dask graph containing the reduction.
    - ``:param chunk_layer_name:`` The reduction's chunk layer.
    - ``:return:`` True only for ``root[-1].op()``-style expressions.
    """
    # Imported lazily to avoid a circular import at module load time (branch.py imports task_branches and defines
    # _find_single_upstream).
    from deisa.dask.branch import _find_single_upstream

    current = chunk_layer_name
    seen = set()
    found_window_getitem = False
    while current is not None and current not in seen:
        seen.add(current)
        layer = graph.layers[current]
        if _is_stub_layer_name(current):
            # Reached the registered-array root stub.
            return found_window_getitem
        if _is_window_read_layer(layer):
            found_window_getitem = True
            upstream = _window_read_upstream_name(layer)
            if upstream is None:
                return False
            if upstream not in graph.layers:
                return found_window_getitem  # root data node
            current = upstream
            continue
        # Ordinary layer: only the reduction chunk stage itself is allowed.
        upstream = _find_single_upstream(layer)
        if upstream is None:
            return False
        upstream_name, _ = upstream
        if upstream_name not in graph.layers:
            return found_window_getitem  # root data node
        current = upstream_name
    return False


def _assign_output_key(array_name: str, op_name: str, axes_sig: Tuple[int, ...], seen: Dict) -> str:
    """Return a per-callback-unique ``output_key`` for one reduction.

    The first occurrence of an op keeps the stable ``{array}-{op}`` key
    (existing tests assert ``a-sum`` / ``f-sum`` / ``b-sum``); a later call
    with a DIFFERENT axis signature appends a deterministic discriminator
    (``-axis0``, ``-axis0x1``, ``-axisall`` for a second full reduction).
    Two identical signatures (same op, same axis -- e.g. ``arr.sum()``
    written twice) keep the SAME key: they are semantically identical and
    dedup to one branch.

    - ``:param seen:`` Mutable per-callback dict
        ``{(array_name, op_name): {axes_sig: output_key}}`` shared across
        every call of :func:`extract_reduction_hints` for one callback, so
        keys stay unique across all compute boundaries of the callback.
    """
    per_op = seen.setdefault((array_name, op_name), {})
    if not per_op:
        key = f"{array_name}-{op_name}"
    else:
        existing = per_op.get(axes_sig)
        if existing is not None:
            key = existing
        elif axes_sig == ():
            key = f"{array_name}-{op_name}-axisall"
        else:
            key = f"{array_name}-{op_name}-axis{'x'.join(map(str, axes_sig))}"
    per_op[axes_sig] = key
    return key


# --------------------------------------------------------------------------- Layer name helpers
# ---------------------------------------------------------------------------
def _strip_hash(layer_name: str) -> str:
    """Drop the trailing ``-hash`` from a dask layer name."""
    return layer_name.rsplit("-", 1)[0] if "-" in layer_name else layer_name


def _is_aggregate_layer(layer_name: str) -> bool:
    """An aggregate layer has ``-aggregate-`` in its name."""
    return "-aggregate-" in layer_name


def _is_sqrt_layer(layer_name: str) -> bool:
    """``std`` adds a ``_sqrt-<hash>`` post-step on top of ``var``."""
    return layer_name.startswith("_sqrt-")


def _base_for_aggregate(layer_name: str) -> str:
    """Return the chunk-layer base name for a given aggregate layer.

    For ``<base>-aggregate-<hash>`` returns ``<base>``. The chunk layer may
    use a related base (e.g. ``mean_agg`` aggregate -> ``mean_chunk`` chunk
    layer; ``max`` aggregate -> ``chunk_max`` chunk layer; ``min`` aggregate
    -> ``chunk_min`` chunk layer).
    """
    base = layer_name.split("-aggregate-", 1)[0]
    return base


# Map aggregate base -> possible chunk base names.
_CHUNK_BASE_FOR_AGG = {
    "mean_agg": "mean_chunk",
    "max": "chunk_max",
    "min": "chunk_min",
}


def _chunk_base_for_aggregate_base(agg_base: str) -> List[str]:
    """Return candidate chunk base names for an aggregate base.

    The chunk layer and aggregate layer usually share a base (``sum``,
    ``prod``, ``var``). The exceptions are reductions that use a
    dedicated chunk/aggregate pair: ``mean`` (mean_chunk / mean_agg),
    ``max`` (chunk_max / max), ``min`` (chunk_min / min).
    """
    direct = agg_base
    special = _CHUNK_BASE_FOR_AGG.get(agg_base)
    return [direct, special] if special else [direct]


# --------------------------------------------------------------------------- Aggregate/Chunk layer introspection
# ---------------------------------------------------------------------------
def _is_task(value: Any) -> bool:
    """True if ``value`` is a dask ``Task`` (new task spec)."""
    if isinstance(value, tuple):
        return False
    return hasattr(value, "func") and hasattr(value, "args") and hasattr(value, "kwargs")


def _blockwise_indices_inputs(layer) -> Optional[Tuple[List[str], int, bool]]:
    """Parse a new-style Blockwise layer's ``indices``.

    Returns ``(array_input_names, array_input_count, has_non_array_input)``,
    or ``None`` if the layer has no ``indices`` (not a new-style Blockwise).
    ``array_input_count`` counts the number of array-input references (so a
    self-referential op like ``arr * arr`` yields count 2); ``names`` contains
    one entry per array input. ``has_non_array_input`` is True when a scalar
    constant (non-string-first-element index key) is present.

    This is the shared primitive behind both
    :func:`_blockwise_upstream_layer_names` (names only) and
    :func:`deisa.dask.branch._find_single_upstream` (single distinct
    upstream + array-input count + constant rejection), so the Blockwise
    index-walking logic lives in one place.
    """
    if not (hasattr(layer, "indices") and layer.indices):
        return None
    names: List[str] = []
    array_input_count = 0
    has_non_array_input = False
    for in_key in layer.indices:
        if isinstance(in_key, (list, tuple)) and len(in_key) >= 1 and isinstance(in_key[0], str):
            names.append(in_key[0])
            array_input_count += 1
        else:
            has_non_array_input = True
    return names, array_input_count, has_non_array_input


def _layer_first_task(layer) -> Optional[Any]:
    """Return the first dask ``Task`` in a layer, or None."""
    for value in layer.values():
        if _is_task(value):
            return value
    return None


def _layer_first_tuple(layer) -> Optional[Any]:
    """Return the first tuple-form task in a layer, or None."""
    for value in layer.values():
        if isinstance(value, tuple) and len(value) >= 2:
            return value
    return None


def _unwrap_partial(func: Any) -> Optional[Any]:
    """Return ``func.func`` if ``func`` is a ``functools.partial``."""
    if hasattr(func, "func") and callable(getattr(func, "func", None)):
        return func.func
    return None


def _is_compose(func: Any) -> bool:
    return hasattr(func, "funcs") and isinstance(func.funcs, tuple)


def _op_from_func(func: Any) -> Optional[str]:
    """Identify the canonical op name from an aggregator callable."""
    if _is_compose(func):
        # dask reductions like sum/prod/max/min are wrapped in ``Compose(partial(np_op), partial(_concatenate2))``.
        for f in func.funcs:
            inner = _unwrap_partial(f)
            if inner is not None:
                name = getattr(inner, "__name__", None)
                if name in _OP_FROM_FUNC_NAME:
                    return _OP_FROM_FUNC_NAME[name]
        return None
    inner = _unwrap_partial(func)
    if inner is None:
        return None
    name = getattr(inner, "__name__", None)
    if name is None:
        return None
    return _OP_FROM_FUNC_NAME.get(name)


def _aggregate_layer_func(layer) -> Optional[Any]:
    """Return the aggregator callable of an aggregate layer.

    Legacy tuple form preferred, then the new Task spec.
    """
    tup = _layer_first_tuple(layer)
    if tup is not None:
        return tup[0]
    task = _layer_first_task(layer)
    if task is not None:
        return task.func
    return None


def _op_for_aggregate_layer(graph, layer_name: str) -> Optional[str]:
    """Return the canonical op name for an aggregate layer, matching hint extraction.

    ``moment_agg`` is shared by var and std; the ``_sqrt`` poststep
    disambiguates them exactly like :func:`extract_reduction_hints` does, so
    chain folding selects the same op name the hint carries. Keeping this in
    one place prevents the two op-naming paths from drifting.
    """
    agg_func = _aggregate_layer_func(graph.layers[layer_name])
    if agg_func is None:
        return None
    op_name = _op_from_func(agg_func)
    if op_name is None:
        return None
    if op_name == "moment":
        op_name = "std" if _has_sqrt_poststep(graph) else "var"
    return op_name


def _chunk_func_and_kwargs(chunk_layer) -> Optional[tuple]:
    """Return ``(func, kwargs)`` for a chunk layer.

    Supports:
    - the new dask task spec (Task objects): ``func`` and ``kwargs`` come
      from the Task directly.
    - the legacy tuple form: ``(func, args, kwargs)`` where ``func`` may be
      a partial with extra keywords baked in.
    """
    for value in chunk_layer.values():
        if _is_task(value):
            return value.func, dict(value.kwargs or {})
        if isinstance(value, tuple) and len(value) >= 1:
            func = value[0]
            if hasattr(func, "keywords"):
                return func, dict(func.keywords or {})
            return func, {}
    return None


# --------------------------------------------------------------------------- branch extraction
# ---------------------------------------------------------------------------
def _chunk_layer_for_aggregate(graph, layer_name: str) -> Optional[str]:
    """Locate the chunk-stage layer feeding ``layer_name`` from the
    aggregate layer's OWN task references (Blockwise ``indices`` /
    first-task args).

    The positional name match (``_find_chunk_layer``) returns the FIRST
    graph layer whose stripped base matches the aggregate base, so in
    ``arr.sum() + (arr - arr.mean()).sum()`` the second ``sum`` aggregate
    resolved to the FIRST ``sum``'s chunk layer and the cross-reduction
    guard inspected the wrong subgraph (ordering-dependent bypass).
    Resolving via the aggregate's own upstream references is unambiguous:
    a dask reduction's aggregate layer references exactly its own chunk
    layer (``sum-aggregate-<h>`` -> ``sum-<h>``; ``mean_agg`` ->
    ``mean_chunk``; ``max`` -> ``chunk_max``; ``moment_agg`` ->
    ``moment_chunk``).

    Returns ``None`` when the aggregate references zero, or more than one
    distinct, non-aggregate layer (not a standard reduction). Callers then
    skip the branch, which surfaces as a no-hint refusal at registration
    rather than a guessed chunk layer.
    """
    layer = graph.layers.get(layer_name)
    if layer is None:
        return None
    upstream = sorted(set(_blockwise_upstream_layer_names(layer)))
    candidates = [u for u in upstream if u in graph.layers and not _is_aggregate_layer(u)]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _find_chunk_layer(graph, agg_base: str) -> Optional[str]:
    """Locate the chunk layer that feeds the aggregate layer with the given base.

    Matches candidate chunk base names (including the dedicated mean/max/min
    chunk/aggregate pairs) and layers whose name starts with a candidate base.

    Returns ``None`` if no matching chunk layer exists.
    """
    candidates = _chunk_base_for_aggregate_base(agg_base)
    for layer_name in graph.layers:
        if _is_aggregate_layer(layer_name) or _is_sqrt_layer(layer_name):
            continue
        layer_base = _strip_hash(layer_name)
        if layer_base in candidates:
            return layer_name
        # Also accept names that start with the base (e.g. ``sum-``)
        if any(layer_name.startswith(c + "-") for c in candidates):
            return layer_name
    return None


def _has_sqrt_poststep(graph) -> bool:
    for layer_name in graph.layers:
        if _is_sqrt_layer(layer_name):
            return True
    return False


def _serialize_func(func: Any) -> bytes:
    return pickle.dumps(func)


def _chunk_inputs_reach_other_aggregate(graph, chunk_layer_name: str) -> set:
    """Walk back from ``chunk_layer_name`` through the graph's
    Blockwise layers and return the names of any aggregate layers
    (``-aggregate-`` in name) reachable from the chunk-stage's inputs.

    For a chain like ``(arr - arr.mean()).sum()``:
    - The outer ``sum`` aggregate's chunk-stage reads from a
      ``subtract`` Blockwise
    - That ``subtract`` Blockwise reads from ``arr`` AND from a
      ``mean-aggregate-...`` layer (the inner reduction's output)
    - So walking back from the outer chunk-stage reaches an aggregate
      layer.

    When this happens, computing the outer reduction locally on a
    bridge would silently produce wrong results: the bridge doesn't
    have the global mean, only its chunk's mean. The expression
    requires data from **all bridges**, so we must refuse to
    precompute it -- the legacy path (scatter full chunk, let dask
    workers compute the expression correctly) is the only safe
    behavior.

    Returns the set of aggregate-layer names reached (empty set if
    none). Callers should refuse the entire dask expression when the
    set is non-empty.
    """
    reachable_aggregates: set = set()
    visited: set = set()
    queue: set = {chunk_layer_name}
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        if current not in graph.layers:
            # Upstream root -- not a layer in this graph (typically the registered placeholder). Stop here.
            continue
        if "-aggregate-" in current:
            reachable_aggregates.add(current)
            # Don't keep walking past an aggregate -- its output is already a fully-reduced value (e.g. the inner mean's
            # output is a scalar per chunk, computed in its own aggregate layer). Reaching ANY aggregate is the
            # cross-reduction signal we care about.
            continue
        layer = graph.layers[current]
        # Walk upstream via the Blockwise ``indices`` (new-style) or via the first task's args (legacy-style).
        upstream = _blockwise_upstream_layer_names(layer)
        queue.update(layer for layer in upstream if isinstance(layer, str))
    return reachable_aggregates


def _aggregate_output_feeds_other_reduction(graph, agg_layer_name: str) -> bool:
    """Return True if ``agg_layer_name``'s output feeds another reduction.

    Walks forward from the aggregate's output through the layers consuming
    it. If the output (or a pointwise layer derived from it) reaching another
    ``-aggregate-`` layer, the aggregate participates in a DIFFERENT
    reduction's chunk stage: its consumer needs the aggregate's GLOBAL
    value, which a bridge cannot produce from its own chunk -- so the inner
    reduction of a cross-reduction expression must be refused (mirror of
    :func:`_chunk_inputs_reach_other_aggregate`, which detects the same
    expression from the consumer's side). Returns False when the output
    feeds only terminal layers, e.g. the once-combined sibling sums in
    ``(arr*arr).sum() + arr.max()``.
    """
    dependencies = getattr(graph, "dependencies", None)
    if dependencies is None:
        return False
    # Invert the dependency edges: consumers[layer] = layers that read it.
    consumers: Dict[str, set] = {}
    for layer_name, deps in dependencies.items():
        for dep in deps:
            consumers.setdefault(dep, set()).add(layer_name)
    seen: set = set()
    queue: set = set(consumers.get(agg_layer_name, ()))
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        if _is_aggregate_layer(current):
            return True
        queue.update(consumers.get(current, ()))
    return False


def _blockwise_upstream_layer_names(layer) -> List[str]:
    """Return the upstream layer names referenced by a Blockwise
    layer's task. Falls back to scanning the first task's args if the
    layer isn't a Blockwise.
    """
    # New-style Blockwise: share the index-walking with _find_single_upstream.
    parsed = _blockwise_indices_inputs(layer)
    if parsed is not None:
        names, _count, _has_non_array = parsed
        return names
    # Legacy-style: scan the first task's args for layer-name strings.
    names = []
    for value in layer.values():
        if _is_task(value):
            for arg in value.args:
                if isinstance(arg, str):
                    names.append(arg)
                else:
                    name = getattr(arg, "key", None)
                    if isinstance(name, str):
                        names.append(name.split("(", 1)[0])
            return names
        if isinstance(value, tuple) and len(value) >= 2:
            # legacy form: (func, deps, ...) where deps is nested list/tuple of layer-name strings
            deps = value[1]
            stack = [deps]
            while stack:
                item = stack.pop()
                if isinstance(item, (list, tuple)):
                    stack.extend(item)
                elif isinstance(item, str):
                    names.append(item)
            return names
    return names


def extract_reduction_hints(
    darr: da.Array,
    array_name: str = "f",
    output_key_seen: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Inspect ``darr``'s task graph and return a branch dict per reduction."""
    hints: List[Dict[str, Any]] = []
    if output_key_seen is None:
        output_key_seen = {}  # fresh per-call seen map
    try:
        graph = darr.__dask_graph__()
    except Exception as e:  # pragma: no cover - safety net
        logger.debug("extract_reduction_hints: failed to get graph: %s", e)
        return hints

    # First pass: refuse expressions whose chunk stages depend on another reduction's aggregate -- those are only
    # correct end-to-end on the workers (with the full chunk), never as per-bridge partials. Done once per dask array so
    # a cross-reduction expression yields zero hints, not zero inner hints plus a wrong outer branch.
    for layer_name in list(graph.layers):
        if not _is_aggregate_layer(layer_name):
            continue
        # Resolve the chunk layer from THIS aggregate's own task references (a positional base-name match could hit a
        # same-op aggregate's chunk layer and inspect the wrong subgraph).
        chunk_layer_name = _chunk_layer_for_aggregate(graph, layer_name)
        if chunk_layer_name is None:
            continue
        reachable = _chunk_inputs_reach_other_aggregate(graph, chunk_layer_name)
        # OTHER aggregates in the reachable set = reductions whose output the chunk-stage depends on.
        other_aggregates = reachable - {layer_name}
        if other_aggregates:
            other_ops = sorted({_base_for_aggregate(a) for a in other_aggregates})
            from deisa.dask.precompute_analyzer import UnsupportedReductionError

            raise UnsupportedReductionError(
                f"Reduction '{_base_for_aggregate(layer_name)}' depends on the output of "
                f"other reduction(s) {other_ops}. Bridge-local precompute cannot produce "
                f"correct per-bridge partials for this expression because the input "
                f"requires data from ALL bridges (not just this bridge's chunk). "
                f"Redesign the callback to use a single reduction (e.g. split the "
                f"expression into two callbacks, or pre-compute the inner reduction "
                f"in a separate step). With ``precompute=False``, the legacy full-chunk "
                f"scatter path runs and dask computes the expression correctly on "
                f"the workers (at the cost of placing the full chunk on workers)."
            )

    for layer_name, layer in graph.layers.items():
        if not _is_aggregate_layer(layer_name):
            continue

        op_name = _op_for_aggregate_layer(graph, layer_name)
        if op_name is None:
            continue

        # ``moment_agg`` is shared by var and std; ``_op_for_aggregate_layer`` disambiguates via the _sqrt poststep (std
        # = var followed by sqrt).
        finalize: Optional[str] = "sqrt" if op_name == "std" else None

        if op_name not in SUPPORTED_OPS:
            logger.debug("extract_reduction_hints: unsupported op %s, skipping", op_name)
            continue

        # Find the matching chunk layer via this aggregate's OWN upstream references (base-name matching is
        # ordering-dependent).
        chunk_layer_name = _chunk_layer_for_aggregate(graph, layer_name)
        if chunk_layer_name is None:
            logger.debug("extract_reduction_hints: no chunk layer for %s", layer_name)
            continue

        chunk_layer = graph.layers[chunk_layer_name]
        chunk_info = _chunk_func_and_kwargs(chunk_layer)
        if chunk_info is None:
            continue
        chunk_func, chunk_kwargs = chunk_info

        try:
            chunk_func_pickle = _serialize_func(chunk_func)
        except Exception as e:
            logger.debug("extract_reduction_hints: failed to pickle chunk func: %s", e)
            continue

        # The window-read flag (``root[-1]``): the callback's reduction runs on the WHOLE delivered array, so the
        # branch's runtime dispatch signature is the FULL reduction even though the stub-side chunk layer carries a
        # partial axis (the getitem removed the other axes).
        window_read = _chain_has_window_read(graph, chunk_layer_name)
        output_key = _assign_output_key(
            array_name,
            op_name,
            () if window_read else _axis_signature(chunk_kwargs.get("axis")),
            output_key_seen,
        )
        # Unwrap single-element axis tuples (dask normalizes ``axis=0`` to ``axis=(0,)``) for the bridge's chunk
        # execution path.
        chunk_kwargs = dict(chunk_kwargs) if chunk_kwargs else {}
        if isinstance(chunk_kwargs.get("axis"), tuple) and len(chunk_kwargs["axis"]) == 1:
            chunk_kwargs["axis"] = chunk_kwargs["axis"][0]
        hints.append(
            {
                "output_key": output_key,
                "op_name": op_name,
                "kind": _REDUCTION_KIND.get(op_name, "scalar"),
                "chunk_func_pickle": chunk_func_pickle,
                "chunk_kwargs": chunk_kwargs,
                "finalize": finalize,
                "window_read": window_read,
            }
        )

    return hints
