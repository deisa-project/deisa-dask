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
import asyncio
import logging
import os
import threading

from deisa.core import DeisaArray, ICommunicator
from distributed import Client, Lock, Variable

import dask.array as da
from deisa.dask.precompute_analyzer import PrecomputeRuntimeError, UnsupportedReductionError
from deisa.dask.task_branches import _normalize_reduction_axis

logger = logging.getLogger(__name__)


def get_client(*args, **kwargs):
    addr = os.getenv("DEISA_DASK_SCHEDULER_ADDRESS", "tcp://127.0.0.1:8787")
    logger.info(f"get_client: DEISA_DASK_SCHEDULER_ADDRESS={addr}")
    return get_connection_info(addr, *args, **kwargs)


def get_mpi_comm_world(cart_coord_dims: int = 1) -> ICommunicator:
    """
    Computes and returns an MPI Cartesian communicator based on the number of desired Cartesian coordinate dimensions.

        This function uses the MPI library to calculate and create a Cartesian communicator from the global MPI
        communicator (MPI.COMM_WORLD). The dimensions of the Cartesian coordinate grid are determined dynamically based
        on the size of the Cartesian coordinate dimensions requested by the user and the size of the MPI communicator.


    ``:param cart_coord_dims:`` Number of Cartesian coordinate dimensions used to compute the grid layout for the
    Cartesian communicator. Default is 1. ``:type cart_coord_dims:`` int ``:return:`` A new Cartesian communicator
    created from the MPI world communicator based on the computed dimensions. ``:rtype:`` mpi4py.MPI.Cartcomm
    """
    from mpi4py import MPI

    mpi_comm = MPI.COMM_WORLD
    dims = MPI.Compute_dims(mpi_comm.Get_size(), dims=cart_coord_dims)
    return mpi_comm.Create_cart(dims)


def get_connection_info(dask_scheduler_address: str | Client, *args, **kwargs) -> Client:
    logger.info(f"get_connection_info: {dask_scheduler_address}")
    if isinstance(dask_scheduler_address, Client):
        client = dask_scheduler_address
    elif isinstance(dask_scheduler_address, str):
        try:
            client = Client(address=dask_scheduler_address, *args, **kwargs)
        except ValueError:
            # try scheduler_file
            if os.path.isfile(dask_scheduler_address):
                client = Client(scheduler_file=dask_scheduler_address, *args, **kwargs)
            else:
                raise ValueError(
                    "dask_scheduler_address must be a string containing the address of the scheduler, "
                    "or a string containing a file name to a dask scheduler file, or a Dask Client object."
                )
    else:
        raise ValueError(
            "dask_scheduler_address must be a string containing the address of the scheduler, "
            "or a string containing a file name to a dask scheduler file, or a Dask Client object."
        )

    return client


def _get_actor(client: Client, clazz, **kwargs):
    def check_variable(dask_scheduler, name):
        ext = dask_scheduler.extensions["variables"]
        v = ext.variables.get(name)
        return v is not None

    key = f"deisa_actor_{clazz}"

    with Lock(key):
        is_set = client.run_on_scheduler(check_variable, name=key)
        if is_set:
            return Variable(key, client=client).get().result()
        else:
            actor_future = client.submit(clazz, actor=True, **kwargs)
            Variable(key, client=client).set(actor_future)
            return actor_future.result()


def build_deisa_array(darr: da.Array, timestep: int) -> DeisaArray:
    return DeisaArray(
        t=timestep,
        dask=darr.dask,
        name=darr.name,
        chunks=darr.chunks,
        dtype=darr.dtype,
        meta=darr._meta,
        shape=darr.shape,
    )


class _PrecomputedDeisaArray(DeisaArray):
    """Per-callback dispatch view over the combined per-bridge partials.

        The runtime delivers ONE combined dask array per reduction (per ``output_key``). A callback's source is not
        rewritten, so its reduction calls (``arr.sum()``, ``arr.mean()``, ...) must be routed to the array the analyzer
        recorded for THAT callback and THAT reduction -- never re-applied onto a wrong array. This view routes by the
        recorded signature ``(op_name, normalized_axis)``:

    - signature match + scalar FULL reduction (the per-bridge ``da.stack``): re-apply the op over the stack via dask
    (the callback's op IS the combine step); - signature match + anything else (mean/moment, any axis reduction): the
    stored array is already FINAL and correctly shaped -- return it directly, neutralizing the callback's
    re-application (re-applying ``.var()`` to a one-element array is mathematically forced ``0.0``); - no match (an op
    the callback performs but was never recorded): raise a typed error -- never silently mis-reduce.

        Construction is a class-swap over a REAL ``DeisaArray`` built by :func:`build_deisa_array` (see
        :func:`make_precomputed_view`): the subclass adds only the seven reduction overrides and three dispatch fields,
        so every dask-level attribute keeps its parent behavior without duplicating the ``Array.__new__`` argument
        plumbing here.
    """

    def _dispatch(self, op_name: str, axis, keepdims: bool):
        if keepdims:
            raise PrecomputeRuntimeError(
                f"Precomputed array {self.name!r}: reduction {op_name}(keepdims=True) is not supported on the "
                f"precompute path -- the analyzer cannot distinguish keepdims at registration time and the "
                f"combined result would have a different shape. Use keepdims=False (the default)."
            )
        # Normalize the CALL's axis against the REGISTERED array's ndim: the
        # callback wrote the axis relative to the delivered array's shape
        # semantics at registration time, and the combined arrays have already
        # dropped the reduced axes, so normalizing against the view's own ndim
        # would silently shift axes (e.g. ``mean(axis=0)`` on a 1-D final
        # array would normalize to the FULL reduction).
        sig = (op_name, _normalize_reduction_axis(axis, self._registered_ndim))
        stored = self._signatures.get(sig)
        if stored is None:
            raise UnsupportedReductionError(
                f"Precomputed array {self.name!r}: reduction {op_name} with axis={axis!r} was not recorded for "
                f"this callback by the precompute analyzer (recorded: "
                f"{sorted(((o, a) for (o, a) in self._signatures))}). The precompute delivery path cannot "
                f"compute it correctly on the callback side; register this reduction or use precompute=False."
            )
        if sig in self._reapply:
            # Scalar FULL reduction: the stored array is the stack of per-bridge
            # partials; dask aggregates it (the value is the global reduction,
            # matching the callback's op on the pre-stack behavior). The op is
            # called on the STORED (plain dask) array -- NOT on ``self``, whose
            # custom dispatch would re-enter ``_dispatch``.
            return getattr(stored, op_name)(axis=None, keepdims=False)
        # Already-final combined array: neutralize the callback's re-application.
        return stored

    def sum(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        return self._dispatch("sum", axis, keepdims)

    def prod(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        return self._dispatch("prod", axis, keepdims)

    def max(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        return self._dispatch("max", axis, keepdims)

    def min(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        return self._dispatch("min", axis, keepdims)

    def mean(self, axis=None, dtype=None, keepdims=False, split_every=None, out=None):
        return self._dispatch("mean", axis, keepdims)

    def var(self, axis=None, dtype=None, ddof=0, keepdims=False, split_every=None, out=None):
        if ddof != 0:
            raise PrecomputeRuntimeError(
                f"Precomputed array {self.name!r}: var(ddof={ddof}) is not supported on the precompute path "
                f"(the per-bridge partials are population moments). Use ddof=0 (the default)."
            )
        return self._dispatch("var", axis, keepdims)

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False, split_every=None, out=None):
        if ddof != 0:
            raise PrecomputeRuntimeError(
                f"Precomputed array {self.name!r}: std(ddof={ddof}) is not supported on the precompute path "
                f"(the per-bridge partials are population moments). Use ddof=0 (the default)."
            )
        return self._dispatch("std", axis, keepdims)


def make_precomputed_view(
    first: da.Array,
    *,
    t: int,
    signatures: dict,
    reapply,
    registered_ndim: int,
) -> _PrecomputedDeisaArray:
    """Build the per-callback dispatch view WITHOUT touching ``Array.__new__``.

    Creates a genuine ``DeisaArray`` through :func:`build_deisa_array` (the same factory every other delivery path
    uses), attaches the dispatch fields, then transplants the view class onto the instance.
    ``DeisaArray``/``dask.array.Array`` instances are plain objects whose state lives in ``__dict__``, so the layout is
    identical across subclasses and the swap is safe (dask itself returns plain ``Array`` instances when a derived
    subclass cannot be preserved).
    """
    if not isinstance(signatures, dict) or not signatures:
        got = f"{type(signatures).__name__}={signatures!r}"
        raise PrecomputeRuntimeError(f"make_precomputed_view: signatures must be a non-empty dict, got {got}")
    view = build_deisa_array(first, t)
    view._signatures = dict(signatures)
    view._reapply = frozenset(reapply)
    view._registered_ndim = registered_ndim
    view.__class__ = _PrecomputedDeisaArray
    return view


def run_coro_on_private_loop(coro):
    """Run ``coro`` on a private event loop hosted by a short-lived worker thread.

    Needed when the calling thread already has a running event loop: blocking that thread on
    ``loop.run_until_complete(coro)`` would deadlock, because a running loop cannot be re-entered. A dedicated thread
    with its own loop is unaffected by whatever the caller is doing, so the coroutine completes and the caller blocks
    only on the future's result.
    """
    result: dict = {}

    def _target():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller below
            result["error"] = exc
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    thread = threading.Thread(target=_target, daemon=True, name="deisa-bridge-scatter")
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")
