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
import sys
import time
import uuid
import zlib
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Final, List, Mapping, Optional, Union

import numpy as np
from deisa.core import IBridge, ICommunicator, validate_arrays_metadata
from distributed import Client, Event, Queue
from distributed.protocol import to_serialize
from distributed.sizeof import safe_sizeof as sizeof
from distributed.utils_comm import scatter_to_workers
from distributed.worker import _global_workers
from distributed.worker_state_machine import RemoveReplicasEvent

from deisa.dask.branch import BranchSpec
from deisa.dask.constants import CLIENT_KEY, FEEDBACK_QUEUE_PREFIX, KEY_PREFIX, WAIT_FOR_EXECUTE_CB_EVENT
from deisa.dask.handshake import Handshake
from deisa.dask.precompute_analyzer import PrecomputeRuntimeError
from deisa.dask.utils import called_from_event_loop, get_client, run_coro_on_private_loop

logger = logging.getLogger(__name__)

_COMM_NULL: Final[None] = None
try:
    from mpi4py import MPI

    _UNDEFINED = MPI.UNDEFINED
except Exception:
    logger.warning("mpi4py not available.")
    _UNDEFINED = 2147483647


class Bridge(IBridge):
    def __init__(self, comm: ICommunicator, arrays_metadata: Dict[str, Dict], *args, **kwargs):
        """
        Initializes an instance of the class, setting up communication, metadata validation,
        client connection (for id=0), workers initialization, and handshake configuration for the bridge.

        - ``:param comm:`` An ICommunicator facilitating communication between processes.
            Must provide Get_rank(), Get_size(), gather(), bcast(), barrier(),
            Split(color, key), and Free() — the same API as an MPI communicator.
        - ``:param arrays_metadata:`` Dictionary containing metadata for arrays.
            eg:

            arrays_metadata = {
                    'temperature': {
                        'global_shape': [20, 20],
                        'chunk_shape': [10, 10],
                        'chunk_position': [0, 0],
                    }
                    'pressure': {
                        'global_shape': [20, 20],
                        'chunk_shape': [10, 10],
                        'chunk_position': [0, 0],
                    }
            }

        - ``:type arrays_metadata: Dict[str, Dict]``
        - ``:param args:`` Additional positional arguments for the initialization.
        - ``:param kwargs:`` Additional keyword arguments for the initialization. Can include
            configuration parameters like timeout used during client setup.
        """
        super().__init__(comm, arrays_metadata, *args, **kwargs)
        self.comm: ICommunicator = comm
        self.id = self.comm.Get_rank()
        self.arrays_metadata = validate_arrays_metadata(arrays_metadata)
        self._my_arrays: set[str] = set(arrays_metadata.keys())  # arrays this bridge owns
        self._feedback_queues = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self._has_close_been_called = False
        self.workers = None
        self.handshake = None
        self.client: Optional[Client] = None
        self._array_comms: Dict[str, Any] = {}  # array_name -> sub-comm (from comm.Split)
        self._handshake_metadata = None

        # array_name -> branches for local execution (None = "not fetched yet")
        self._task_branches: Dict[str, Optional[List[BranchSpec]]] = {}
        # array_name -> output_key -> branch (BranchSpec)
        self._branch_by_key: Dict[str, Dict[str, Any]] = {}

        if self.id == 0:
            # only id 0 has a real dask client
            self.client = get_client(timeout=kwargs.get("timeout", 10), name=f"bridge-{self.comm.Get_rank()}")
            assert self.client, "client cannot be None for Bridge id 0."
            # get all workers from scheduler
            self.workers = self.client.scheduler_info(n_workers=-1)["workers"]

        # retrieve workers from rank 0 and bcast
        logger.debug(f"[{self.id}] Bridge __init__(): pre-bcast")
        self.workers = self.comm.bcast(self.workers, root=0)
        logger.debug(f"[{self.id}] Bridge __init__(): post-bcast. workers={self.workers}")

        # Gather each bridge's partial metadata -> global view: each bridge declares only the arrays it sends; merge
        # into a single dict.
        self._gather_global_metadata()

        # Auto-discover array participation and create sub-communicators
        self._setup_array_comms()

        if self.id == 0:
            # all bridges are ready, tell handshake actor
            assert self.client is not None, "client cannot be None for Bridge id 0."
            self.handshake = Handshake(self.client)
            # Send merged metadata (from all bridges) to the handshake actor
            metadata_for_handshake = self._handshake_metadata or self.arrays_metadata
            self.handshake.all_bridges_ready(
                nb_bridge=self.comm.Get_size(), arrays_metadata=metadata_for_handshake, **kwargs
            )

            # The go-event wait is unconditional on the wait_for_go path: ``WAIT_FOR_EXECUTE_CB_EVENT`` is set by
            # ``execute_callbacks()`` only. wait_for_go=False skips the wait and fetches branches lazily on first
            # send().
            if kwargs.get("wait_for_go", True):
                Event(WAIT_FOR_EXECUTE_CB_EVENT, client=self.client).wait()

        # World-wide barrier: rank 0 waited for the go event (all callbacks registered), the other ranks skipped that
        # wait. The barrier aligns every rank past callback registration so the prefetch below reads a consistent
        # handshake state. MUST be unconditional (it is a world collective); deliberately NOT gated on wait_for_go.
        self.comm.barrier()
        logger.debug(f"[{self.id}] Bridge __init__(): post-registration barrier done")

        if kwargs.get("wait_for_go", True) and self.id == 0:
            # Prefetch static branches out of the send() path. Only safe here: the barrier guarantees branches exist in
            # the handshake. On wait_for_go=False they may not be registered yet (--lazy first-send fetch in
            # _get_task_branches).
            self._prefetch_task_branches()

    def _gather_global_metadata(self):
        """
        Gather each bridge's partial arrays_metadata to discover the global
        set of array names.

        Each bridge declares only the arrays it actually sends. This method
        collects all array names across all bridges so that every rank can
        call Split() for every array (arrays not owned get color=_UNDEFINED).

        The bridge's own arrays_metadata (with its chunk_position etc.) is
        preserved for use in send(). Only a lightweight set of global array
        names is stored in self._global_array_names.

        Rank 0 also produces a merged metadata dict for the handshake actor
        (so the Deisa analytics side has the full picture).

        When the communicator size is 1 (single-bridge, no MPI peers), this
        is a no-op: the bridge's own metadata is already the full picture.
        """
        if self.comm.Get_size() == 1:
            # Single-bridge case: nothing to run a collective with
            self._global_array_names = set(self._my_arrays)
            return

        all_metadata = self.comm.gather(self.arrays_metadata, root=0)

        if all_metadata:
            merged = {}
            for partial in all_metadata:
                for name, metadata in partial.items():
                    merged.setdefault(name, metadata)

            self._handshake_metadata = merged
            global_names = set(merged)
        else:
            global_names = None

        # Broadcast the global set of array names to all bridges
        global_names = self.comm.bcast(global_names, root=0)
        self._global_array_names = sorted(global_names)

    def _setup_array_comms(self):
        """
        Create per-array sub-communicators using comm.Split().

        The global array list (from _gather_global_metadata) is known to all
        bridges, so every rank can call Split() for every array. Bridges that
        don't own a given array use color=_UNDEFINED and receive _COMM_NULL.

        The color is derived from zlib.crc32 of the array name so
        that all participating ranks get the same sub-communicator group.
        """
        for array_name in self._global_array_names:
            participates = array_name in self._my_arrays
            # Force into a positive 31-bit integer Reserve 0x7FFFFFFF as _UNDEFINED value.
            color = (zlib.crc32(array_name.encode()) & 0x7FFFFFFE) if participates else _UNDEFINED

            # sub_comm is either an instance of: mpi4py.MPI.Comm, mpi4py.MPI.CommNull or None (FakeComm) Split is a
            # collective. All ranks of parent comm must call this.
            sub_comm = self.comm.Split(color, self.id)

            # convert mpi4py.MPI.CommNull to _COMM_NULL
            sub_comm = _COMM_NULL if not participates else sub_comm

            self._array_comms[array_name] = sub_comm

            # create a new client for sub_comm id==0 if needed
            if not self.client and sub_comm is not _COMM_NULL and sub_comm.Get_rank() == 0:
                self.client = get_client(timeout=10, name=f"bridge-{self.comm.Get_rank()}")
                # Connect to existing handshake actor from analytics side (created by Deisa)
                self.handshake = Handshake(self.client)

            # Seed a "not fetched yet" sentinel: None != [] (a cached branch-less array). Fetched once and cached, []
            # included.
            self._task_branches[array_name] = None

            logger.debug(
                f"[{self.id}] _setup_array_comms: "
                f"array={array_name}, "
                f"participates={participates}, "
                f"sub_comm_size={sub_comm.Get_size() if sub_comm is not _COMM_NULL else 'NULL'}"
            )

    def __del__(self):
        """Clean up resources before destruction.

        Never runs a blocking collective at interpreter shutdown
        (``sys.is_finalizing()``): ``close()`` synchronizes all bridges with an
        MPI barrier, and at shutdown the peer ranks may already be torn down,
        so that barrier would hang forever -- a hang is worse than an
        exception.

        Never runs it from an event-loop callback either. The collector runs
        ``__del__`` on whichever thread dropped the last reference, and for a
        leaked bridge that is a distributed "IO loop" thread. ``close()``
        synchronizes with the world (MPI barrier) and with the scheduler
        (``Handshake.set_bridges_done`` -> ``Client.sync()``), and ``Client.sync()``
        blocks the calling thread until the client's loop runs the callback: from
        inside that very loop the callback can never run, so the thread sits out the
        whole ``sync()`` timeout (60 s measured) and, with an in-process cluster whose
        scheduler, workers and client share that loop, freezes the whole cluster.

        During normal (non-shutdown) garbage collection from any other thread the
        full ``close()`` still runs.
        """
        if sys.is_finalizing():
            logger.debug(f"[{self.id}] Bridge.__del__ during interpreter shutdown: skipping close()")
            return
        if called_from_event_loop():
            logger.debug(
                f"[{self.id}] Bridge.__del__ from inside an event-loop callback: skipping close(), it would "
                f"block the loop that has to service its collectives"
            )
            return
        try:
            self.close(timestep=sys.maxsize)
        except Exception:
            # Never raise from __del__. close() already logs its own errors.
            pass

    def close(self, timestep: int) -> None:
        """
        Close the bridge: synchronize bridges, free sub-comms, shut down client.

        - ``:param timestep:`` The current timestep associated with the closure action.
        - ``:type timestep:`` int
        - ``:return:`` None
        """
        logger.info(f"[{self.id}] Bridge close()")
        try:
            if not self._has_close_been_called:
                self._has_close_been_called = True
                if sys.is_finalizing():
                    # Interpreter shutdown: the world barrier and the sub-comm Free() calls are collectives that can
                    # never complete once peer ranks are gone. Skip the coordinated part; the client (non-collective) is
                    # still closed below.
                    logger.debug(f"[{self.id}] Bridge close() during interpreter shutdown: skipping collectives")
                else:
                    # Barrier on communicator — all bridges must synchronize
                    self.comm.barrier()
                    # Free sub-communicators created via Split()
                    for array_name, sub_comm in self._array_comms.items():
                        if sub_comm is not None and sub_comm != _COMM_NULL:
                            sub_comm.Free()
                            logger.debug(f"[{self.id}] Freed sub-communicator for array '{array_name}'")
                    self._array_comms.clear()
                    if self.id == 0:
                        assert self.handshake, "handshake cannot be None for Bridge id 0."
                        assert self.client, "client cannot be None for Bridge id 0."
                        self.handshake.set_bridges_done(timestep=timestep)

                if self.client:
                    self.client.close()
        except Exception as e:
            logger.error(f"[{self.id}] Could not cleanly close bridge. exception={e}")

    def send(self, array_name: str, chunk: np.ndarray, timestep: Union[int, np.ndarray], *args, **kwargs):
        """
        Distribute a data chunk to a Dask workers.

        Scatters the chunk to the next worker (round-robin), then gathers all chunks metadata to bridge rank 0,
        which updates the Dask scheduler.

        ``send`` is a three-stage pipeline with ONE implementation per stage:

        1) ``_select_workers`` resolves the round-robin target (the only place the ``update_workers``/``filter_workers``
        kwargs are read; the refresh broadcast keeps its position in the collective order).

        2) ``_ship_chunk`` runs the data plane: branch-func reductions on the local numpy chunk, then the
        partials scatter, or the full-chunk scatter when no branch produced a partial (backwards compatibility).

        3) ``_publish_metadata`` runs the control plane: the per-bridge slice is published directly on a single-bridge
        array (sub_comm_size == 1) or gathered first, then a single shared tail merges, updates the scheduler and
        emits the topic event.

        - ``:param array_name:`` The name of the data array being sent as a string.
        - ``:param chunk:`` A numpy ndarray containing the data chunk to be sent to the workers.
        - ``:param timestep:`` The current timestep associated to the sent data chunk.
        - ``:param args:`` Additional positional arguments if required by the method implementation.
        - ``:param kwargs:`` Additional keyword arguments for optional configurations.
        Supported keys include:

        - `update_workers` (bool): If True, updates the workers' list by retrieving it from the scheduler.
        - `filter_workers` (callable): A function that filters the available workers and returns a list of worker names.
        Must return a non-empty list of strings.
        - ``:return:`` None.
        """
        logger.debug(f"[{self.id}] send() array_name={array_name}, data.shape={chunk.shape}, iteration={timestep}")

        if array_name not in self.arrays_metadata:
            raise ValueError(f"array {array_name} is unknown.")

        # handle different types of timestep. Cast to int if necessary.
        if isinstance(timestep, np.ndarray):
            if timestep.ndim == 0:
                timestep = int(timestep)
            elif timestep.ndim == 1 and timestep.size == 1:
                timestep = timestep.item()
            else:
                raise ValueError("timestep must be a scalar, a 0D array or a 1D array of size 1.")
        assert isinstance(timestep, int)

        workers = self._select_workers(array_name, timestep, **kwargs)

        # Determine communicator from cached sub-comms (from comm.Split())
        sub_comm = self._array_comms.get(array_name)

        if sub_comm == _COMM_NULL:
            # This rank doesn't participate in this array
            logger.debug(f"[{self.id}] send() rank not in participating set for '{array_name}', skipping")
            return

        # Data plane: local reductions then ship ONLY the tiny partials when branches produced any (the full chunk
        # never enters worker memory), else the full-chunk scatter for non-reduction callbacks.
        scatter_result, futures_entries = self._ship_chunk(array_name, chunk, workers)

        # Control plane: rank 0 merges every bridge's slice and publishes the topic event.
        self._publish_metadata(array_name, scatter_result, futures_entries, chunk, timestep, sub_comm)

        # TODO: what to do if error ?

    def _select_workers(self, array_name: str, timestep: int, **kwargs) -> List[str]:
        """Pick the ONE worker this bridge's chunk goes to: round-robin over the sorted worker list.

        ``send()``'s whole worker axis lives here, at its original position -- before the participation check -- so
        the optional ``update_workers`` refresh (one sub-comm broadcast, entered by every rank of the array) keeps
        its place in the collective order. The caller's ``filter_workers`` keeps its validation contract: a filter
        that does not return a non-empty list of strings raises ``TypeError``.

        - ``:param array_name:`` Registered array the send is for; its sub-comm carries the refresh broadcast.
        - ``:param timestep:`` The current timestep, which picks the worker together with the bridge id.
        - ``:param kwargs:`` ``send()``'s optional kwargs: ``update_workers`` (bool), ``filter_workers`` (callable).
        """
        assert isinstance(self.workers, dict)
        workers = dict(self.workers)  # make a copy so that the user-defined function does not modify self

        if kwargs.get("update_workers", False):
            # only update worker list if requested
            sub_comm = self._array_comms[array_name]

            if sub_comm.Get_rank() == 0:
                assert self.client is not None, "client cannot be None for Bridge comm id 0."
                # rank 0 retrieve workers and bcast to other bridges
                workers = self.client.scheduler_info(n_workers=-1)["workers"]

            # bcast
            logger.debug(f"[{self.id}] send() pre-bcast workers={workers}")
            self.workers = sub_comm.bcast(workers, root=0)
            logger.debug(f"[{self.id}] send() post-bcast workers={workers}")
            workers = dict(self.workers)

        if kwargs.get("filter_workers", False):
            workers = kwargs["filter_workers"](workers)
            # check return type
            if not isinstance(workers, list):
                raise TypeError(f"worker_filter must return a list, got {type(workers)}")
            if len(workers) == 0:
                raise TypeError("worker_filter must return a non-empty list")
            for w in workers:
                if not isinstance(w, str):
                    raise TypeError(f"worker_filter must return a list of strings, got {type(w)}")
        else:
            workers = list(workers.keys())

        workers = sorted(workers)
        # per bridge id and iteration round-robin over the workers
        index = (timestep + self.id) % len(workers)
        workers = [workers[index]]

        assert len(workers) == 1, "worker list should be of length 1."
        return workers

    def _ship_chunk(self, array_name: str, chunk: np.ndarray, workers: List[str]) -> tuple:
        """Run the data plane of ``send()``: local reductions, then partials, else the full chunk.

        - ``:param array_name:`` Registered array the send is for. Its cached branch list (when any) is computed
          locally on ``chunk``; every produced partial is shipped to the ONE target worker and nothing more.
        - ``:param chunk:`` This bridge's own numpy chunk.
        - ``:param workers:`` The single target worker (see ``_select_workers``).
        - ``:return:`` ``(scatter_result, futures_entries)``: ``scatter_result`` is the normalized scatter result
          ``{"future": List[str], "who_has", "nbytes"}``, and ``futures_entries`` holds the final topic-event
          entries on the precompute path (empty on the full-chunk path, where ``_publish_metadata`` builds one
          entry per bridge from the chunk itself).
        """
        branches = self._get_task_branches(array_name)
        partials = self._execute_operations_on_chunk(chunk, branches)

        if partials:
            logger.debug(
                f"[{self.id}] send() precompute-active: scattering {len(partials)} partials "
                f"instead of full chunk shape={chunk.shape}"
            )
            return self._scatter_partials(partials, branches, array_name, workers)
        logger.debug(f"[{self.id}] send() precompute-inactive: scattering full chunk shape={chunk.shape}")
        return self._scatter_full(chunk, workers=workers), []

    def _publish_metadata(
        self,
        array_name: str,
        scatter_result: dict,
        futures_entries: List[Dict[str, Any]],
        chunk: np.ndarray,
        timestep: int,
        sub_comm: Any,
    ) -> None:
        """Publish one timestep's topic event. This will eventually trigger the user's callback.

        Each side reduces to a ``piece`` (the per-bridge metadata of what was sent on the network (scatter result,
        chunk position, futures entries)) published directly on a single-bridge array or gathered first on a
        multi-bridge one.

        - ``:param array_name:`` Registered array the event is for.
        - ``:param scatter_result:`` This bridge's normalized scatter result
          (``{"future": List[str], "who_has", "nbytes"}``).
        - ``:param futures_entries:`` The final topic-event entries of this bridge's partials on the precompute
          path (empty on the full-chunk path; then one entry per bridge is built here from the chunk).
        - ``:param chunk:`` This bridge's own numpy chunk, read for the payload's shape/dtype.
        - ``:param timestep:`` The current timestep, carried as the event's ``iteration``.
        - ``:param sub_comm:`` The array's sub-communicator on this rank; the gather runs on it when it has peers.
        """
        piece: Dict[str, Any] = {
            "scatter": scatter_result,
            "chunk_position": self.arrays_metadata[array_name]["chunk_position"],
            "futures": futures_entries,
        }

        pieces: List[Dict[str, Any]]
        if sub_comm.Get_size() == 1:
            assert self.client is not None, "client cannot be None for single-bridge publish."
            # Single-bridge array: the one rank IS the whole grid, skip the gather entirely.
            pieces = [piece]
        else:
            logger.debug(f"[{self.id}] send() gather: to_send={piece}")
            gathered = sub_comm.gather(piece, root=0)
            logger.debug(f"[{self.id}] send() gathered_data={gathered}")
            if not gathered:
                # Non-root rank: rank 0 publishes for the array.
                return
            assert self.client is not None, "client cannot be None for Bridge id 0."
            pieces = gathered

        assert self.client is not None, "client cannot be None."

        # Merge the pieces (one per participating bridge) into the array-wide state. Each piece carries its OWN
        # chunk_position: indexing the pieces with ``enumerate`` would misalign when a bridge shipped no partials,
        # letting one bridge's coordinates (or grid count) corrupt another's contribution.
        who_has: Dict[str, List[str]] = {}
        nbytes: Dict[str, int] = {}
        keys: List[str] = []
        for other in pieces:
            info: Dict[str, Any] = other["scatter"]
            who_has.update(info["who_has"])
            nbytes.update(info["nbytes"])
            keys.extend(info["future"])

        # only update the scheduler with who has what and register the futures once
        self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)

        # mimic mechanism from Queue. Keep a reference on keys until reception in topic handler.
        # TODO: can id=0 use a queue instead of manually handling keys ?
        self.client._send_to_scheduler({"op": "client-desires-keys", "keys": keys, "client": CLIENT_KEY})

        # Topic event: with precompute, ``futures`` lists one entry per (bridge, reduction) pair with the partial's
        # reduced shape/dtype; the Deisa side rebuilds the graph from these small partials. Otherwise one entry per
        # bridge names the full chunk.
        precomputed = any(other["futures"] for other in pieces)
        if precomputed:
            futures_payload = [entry for other in pieces for entry in other["futures"]]
        else:
            futures_payload = [
                {
                    "future": other["scatter"]["future"][0],
                    "shape": chunk.shape,
                    "dtype": str(chunk.dtype),
                    "chunk_position": other["chunk_position"],
                }
                for other in pieces
            ]

        to_send = {
            "array_name": array_name,
            "iteration": timestep,
            "precomputed": precomputed,
            "futures": futures_payload,
        }
        logger.debug(
            f"[{self.id}] send() log_event: array={array_name}, timestep={timestep}, n_futures={len(futures_payload)}"
        )
        self.client.log_event(array_name, to_send)

        # TODO: what to do if error ?

    def get(self, key: str, timestep: Optional[int] = None, default: Any = None) -> Optional[Union[Deque, Any]]:
        """
        Retrieve an element associated with a specific key and optional timestep from a feedback queue.
        If a queue for the key does not exist, it initializes the queue for the specified key.

        - ``:param key:`` The unique identifier for the feedback queue.
        - ``:type key:`` str
        - ``:param timestep:`` An optional specific timestep to look for. If None, returns the entire deque.
        - ``:type timestep:`` Optional[int]
        - ``:param default:`` The default value to return if the specified timestep is not found.
        - ``:type default:`` Any
        - ``:return:`` The element associated with the specified timestep if found, the entire deque if no
            timestep is specified, or the default value if the timestep is not found.
        - ``:rtype:`` Optional[Union[Deque, Any]]
        """
        logger.debug(f"[{self.id}] get() key={key}, timestep={timestep}, default={default}")
        fb_state: Dict = self._feedback_queues[key]

        if self.id == 0:
            if len(fb_state) == 0:
                feedback_queue_size = self.handshake.get_feedback_queue_size()
                fb_state[key] = {
                    "q": Queue(f"{FEEDBACK_QUEUE_PREFIX}{key}", client=self.client, maxsize=feedback_queue_size),
                    "deque": deque(maxlen=feedback_queue_size),
                }

            q: Queue = fb_state[key]["q"]
            d: deque = fb_state[key]["deque"]

            if q.qsize() != 0:
                # List[(int, Any), ...]
                full_q = q.get(batch=True)  # get all elements. This pops elements from the Dask queue.
                for v in full_q:
                    d.append(v)  # add all elements to deque
            logger.debug(f"[{self.id}] get() fb_state={fb_state}")

        d = self.comm.bcast(fb_state[key]["deque"], root=0)

        if timestep is None:
            return d

        for t, v in d:
            if timestep == t:
                # found the timestep
                return v

        return default

    def _scatter_full(self, data: np.ndarray, workers: List[str] = None):
        """Scatter the full chunk to ``workers``; returns the normalized ``{"future": [key], "who_has", "nbytes"}``."""
        logger.debug(f"[{self.id}] scatter to {workers}")

        if workers is None:
            workers = list(self.workers)

        key = f"{KEY_PREFIX}-{type(data).__name__}-{uuid.uuid4().hex}"
        who_has, nbytes = self._await_scatter(self._scatter_payload, workers, {key: data})
        return {"future": [key], "who_has": who_has, "nbytes": nbytes}

    def _await_scatter(self, fn, *args):
        """Run a scatter coroutine ``fn(*args)`` through the client's loop, else on a private loop.

        The no-client path (bridge teardown) can reach this from inside a running event loop, where ``asyncio.run``
        raises; ``_run_async`` picks the existing loop or a private one instead.
        """
        if self.client is not None:
            return self.client.sync(fn, *args)
        return self._run_async(fn(*args))

    def _run_async(self, coro):
        """Run ``coro`` to completion from synchronous code.
        ``asyncio.run`` cannot be called when the calling thread already has a running event loop; it raises
        ``RuntimeError: asyncio.run() cannot be called from a running event loop``.
        The no-client path reaches this from bridge teardown, where a loop is frequently still running on this thread.
        Use the existing loop when there is one, and ``asyncio.run`` otherwise.
        """
        try:
            # Probe for a running loop on this thread. We are in synchronous code, so a running loop here cannot make
            # progress while we block on run_until_complete, that would deadlock.
            # Hand the coroutine to a private loop on a worker thread instead, which runs concurrently with the caller's
            # loop.
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop on this thread: asyncio.run is safe and needs no cleanup.
            return asyncio.run(coro)

        return run_coro_on_private_loop(coro)

    async def _scatter_payload(self, workers, payload):
        """Scatter an already-keyed ``{key: value}`` payload to ``workers``; returns ``(who_has, nbytes)``.

        The ONLY scatter primitive: every send path (full chunk, precompute partials) keys its payload and calls
        this. Goes through :meth:`_better_scatter_to_workers`, so an in-process worker receives the values
        zero-copied; ``payload`` must therefore stay RAW -- the remote branch serializes only the entries it
        actually ships.
        """
        assert isinstance(payload, dict)
        _, who_has, nbytes = await self._better_scatter_to_workers(
            workers,
            payload,
            scheduler=self.client.scheduler if self.client else None,
            client_id=self.client.id if self.client else None,
        )
        return who_has, nbytes

    def _fetch_branches_bcast(self, array_name: str) -> Optional[List[BranchSpec]]:
        """Mirror of the per-array bcast protocol: sub-comm rank 0 reads from the
        handshake actor and broadcasts; every other participating rank receives via
        bcast(None). Returns None when this bridge does not participate in the
        array's sub-comm. Callers decide caching (None is NOT a cached result)."""
        sub_comm = self._array_comms.get(array_name)
        if sub_comm is None or sub_comm is _COMM_NULL:
            return None
        if sub_comm.Get_rank() == 0 and self.handshake is not None:
            branches = self.handshake.get_task_branches(array_name)
            sub_comm.bcast(branches, root=0)
        else:
            branches = sub_comm.bcast(None, root=0)
        return branches

    def _get_task_branches(self, array_name: str) -> List[BranchSpec]:
        """
        Retrieve stored task hints for an array.

        Branches are fetched from HandshakeActor on sub_comm rank 0 and broadcast to all ranks.
        If no hints are available, this method returns an empty list (no precomputation).

        - ``:param array_name:`` The array name to get hints for.
        - ``:return:`` List of reduction hints (each carrying a pickled chunk
            callable and the dask kwargs to apply).
        """
        # Cache hit first: after one fetch, sends never touch the handshake.
        cached = self._task_branches.get(array_name)
        if cached is not None:
            return cached

        # Cache miss -> shared per-array bcast protocol (lazy path: wait_for_go=False or a cache miss).
        branches = self._fetch_branches_bcast(array_name)
        if branches is not None:
            # Store unconditionally: [] is a valid cached result meaning "no precompute, full-chunk scatter" -- don't
            # re-fetch per send.
            self._task_branches[array_name] = branches
        return branches or []

    def _prefetch_task_branches(self) -> None:
        """Prefetch static task-branches for every global array into the local cache.

        On the wait_for_go path the handshake is complete by the time this
        runs (go event + world barrier), so broadcasting per array removes
        branch fetches from the send() critical path. The broadcast mirrors
        ``_get_task_branches``: sub-comm rank 0 reads the handshake actor and
        broadcasts; non-participating ranks leave the None sentinel. A
        branch-less array ([]) is cached as-is.
        """
        for array_name in self._global_array_names:
            branches = self._fetch_branches_bcast(array_name)
            if branches is not None:
                self._task_branches[array_name] = branches

    def _get_branch_by_key(self, array_name: str, branches: List[BranchSpec]) -> Dict[str, Any]:
        """Return the ``output_key -> branch`` lookup for an array, building and caching it on first use.

        Branches are static after registration, so the indexed map is not rebuilt on every ``send()``.
        """
        cached = self._branch_by_key.get(array_name)
        if cached is not None:
            return cached

        indexed = {b.output_key: b for b in branches}
        self._branch_by_key[array_name] = indexed
        return indexed

    def _scatter_partials(
        self,
        partials: Dict[str, Any],
        branches: List[BranchSpec],
        array_name: str,
        workers: List[str],
    ) -> tuple:
        """Scatter precompute partials to a worker instead of the full chunk.

        Partial flavors: ``scalar`` (sum/prod/max/min: plain numbers/arrays, one future key per reduction), ``mean``
        (``{"n", "total"}`` dict, combined with ``mean_agg``) and ``moment`` (``{"n", "total", "M"}`` dict for
        var/std, combined with ``moment_agg`` + sqrt).

        One loop keys the payload AND builds the final topic-event entries (each :class:`BranchSpec` carries the
        per-reduction metadata directly), so the event payload is born once instead of being ferried through an
        intermediate metadata sidecar.

        - ``:return:`` ``(scatter_result, futures_entries)``: ``scatter_result`` is the normalized scatter result
          ``{"future": List[str], "who_has", "nbytes"}``; ``futures_entries`` is one entry per reduction with the
          partial's shape/dtype, the reduction's ``chunk_axis`` and this bridge's ``chunk_position``.
        """
        assert len(workers) == 1, "_scatter_partials expects a single target worker"
        target_worker = workers[0]

        # Index branches by output_key (cached; BranchSpec carries the analyzer's per-reduction metadata directly).
        branch_by_key = self._get_branch_by_key(array_name, branches)

        payload: Dict[str, Any] = {}
        futures_entries: List[Dict[str, Any]] = []
        for output_key, value in partials.items():
            branch = branch_by_key.get(output_key)
            if branch is None:
                # Every partial comes from a branch (they are built together in _execute_operations_on_chunk).
                raise KeyError(f"no branch found for precomputed output_key {output_key!r}")
            key = f"{KEY_PREFIX}{array_name}-partial-{output_key}-{uuid.uuid4().hex}"
            payload[key] = value
            futures_entries.append(
                {
                    "future": key,
                    "shape": branch.partial_shape,
                    "dtype": branch.partial_dtype,
                    "kind": branch.output_kind,
                    "finalize": branch.finalize,
                    "chunk_position": self.arrays_metadata[array_name]["chunk_position"],
                    "chunk_axis": branch.chunk_axis,
                    "op_name": branch.op_name,
                    "output_key": output_key,
                }
            )

        # _scatter_payload keeps the payload RAW so an in-process worker receives it zero-copied; pre-serializing
        # here would defeat that.
        who_has, nbytes = self._await_scatter(self._scatter_payload, [target_worker], payload)
        return {"future": list(payload.keys()), "who_has": who_has, "nbytes": nbytes}, futures_entries

    def _execute_operations_on_chunk(self, chunk: np.ndarray, branches: List["BranchSpec"]) -> Dict[str, Any]:
        """
        Execute branch chunk-stage callables locally on the bridge's numpy chunk before scattering.

        Each branch in ``branches`` is a :class:`BranchSpec` whose ``branch_func`` is a pickle-able Python callable
        that takes a numpy chunk and returns the per-bridge partial (a scalar, ndarray, or dict for mean/moment).
        The closure inside ``branch_func`` already binds the analyzer's chunk kwargs (axis, keepdims, dtype, ...),
        so the call here is ``branch_func(chunk)`` with no extra arguments.

        - ``:param array_name:`` The array name being processed.
        - ``:param chunk:`` The numpy ndarray data chunk.
        - ``:param branches:`` List of :class:`BranchSpec` from ``analyze_branch``.
        - ``:return:`` Dict of partial results keyed by output_key.
        """
        partials = {}
        for branch in branches:
            output_key = branch.output_key
            try:
                partial = branch.branch_func(chunk)
            except Exception as e:
                # A dropped partial silently corrupts the combined reduction (scalar stacks get smaller sums,
                # mean_agg/moment_agg miss a whole bridge's n). FAIL LOUDLY instead of skipping.
                raise PrecomputeRuntimeError(
                    f"[{self.id}] _execute_operations_on_chunk: branch {output_key!r} failed on the local "
                    f"chunk: {e!r}. Refusing to ship a partial set that would produce a wrong reduction."
                ) from e
            partials[output_key] = partial

        logger.debug(f"[{self.id}] _execute_operations_on_chunk: {partials}")
        return partials

    async def _better_scatter_to_workers(self, workers, data, scheduler=None, client_id: str = None):
        """Scatter ``data`` to ``workers``, zero-copy for in-process workers.

        Workers found in ``_global_workers`` (same process as this bridge) get the value reference written
        directly into their store via ``worker.update_data`` -- no serialization, no network. All other
        workers fall back to the distributed ``scatter_to_workers`` path with serialized payloads.

        - ``:param workers:`` The worker addresses to scatter to (one key per worker, round-robin).
        - ``:param data:`` The ``{key: value}`` payload to scatter. Values must be RAW: the local branch writes the
            reference itself and only the remote branch wraps its entries in ``to_serialize``.
        - ``:param scheduler:`` The scheduler proxy to report the local keys to (None skips the report).
        - ``:param client_id:`` The reporting client's id, forwarded to ``scheduler.update_data``.
        - ``:return:`` ``(names, who_has, nbytes)`` -- dask's ``scatter_to_workers`` contract. ``nbytes`` comes from
            dask's ``sizeof`` on both branches so the scheduler's accounting matches what the workers report.
        """

        assert isinstance(data, dict)

        local_worker_map = {w.address: w for w in _global_workers}

        # deterministic ordering
        workers = sorted(workers)
        names = list(data.keys())
        values = list(data.values())

        # assign workers round-robin
        entries = [(workers[i % len(workers)], names[i], values[i]) for i in range(len(names))]

        who_has = {}
        nbytes = {}

        local_entries = []
        remote_entries = []

        for w, k, v in entries:
            (local_entries if w in local_worker_map else remote_entries).append((w, k, v))

        # -----------------------
        # Local (zero-copy path)
        # -----------------------
        if local_entries:
            scheduler_who_has = {}
            scheduler_nbytes = {}

            written = []
            try:
                for addr, key, val in local_entries:
                    worker = local_worker_map[addr]
                    worker.update_data({key: val})

                    # Record BEFORE the zero-copy assert: ``update_data`` has already committed, so the key is in the
                    # store from here on. The assert below is one of the ways this loop can raise, and a key missing
                    # from ``written`` would be exactly the orphan the compensation exists to prevent.
                    written.append((addr, key))

                    assert id(worker.data[key]) == id(val), f"In-process scatter copied data: id(orig)={id(val)}"

                    size = sizeof(val)

                    who_has[key] = [addr]
                    nbytes[key] = size

                    scheduler_who_has[key] = [addr]
                    scheduler_nbytes[key] = size

                if scheduler is not None:
                    await scheduler.update_data(
                        who_has=scheduler_who_has,
                        nbytes=scheduler_nbytes,
                        client=client_id,
                    )
            except BaseException:
                # The worker store is written before the scheduler hears about it, so a raise in between (or in the
                # report itself) orphans every key written so far: the worker holds bytes the scheduler never
                # accounts for, and the scheduler may already know about a subset. Compensate by releasing the local
                # writes; a key the scheduler did learn about is then harmlessly redundant and gets dropped.
                #
                # BaseException, not Exception: the divergence window is real for CancelledError / KeyboardInterrupt
                # too (the writes are synchronous and already committed), so those must be compensated as well rather
                # than leaving an orphan behind. ``_release_local_keys`` never re-raises, so the original exception
                # always propagates.
                _release_local_keys(local_worker_map, written, bridge_id=self.id)
                raise

        # -----------------------
        # Remote (distributed path)
        # -----------------------
        if remote_entries:
            remote_workers = sorted({w for w, _, _ in remote_entries})

            remote_data = {k: to_serialize(v) for _, k, v in remote_entries}

            _, remote_who_has, remote_nbytes = await scatter_to_workers(remote_workers, remote_data)

            who_has.update(remote_who_has)
            nbytes.update(remote_nbytes)

        return names, who_has, nbytes


def _release_local_keys(local_worker_map: Mapping[str, Any], written: List[tuple], bridge_id: Any = None) -> None:
    """Drop ``written`` ``(addr, key)`` entries from their in-process worker stores.

    Compensation for a scatter that failed between the local write and the scheduler report. Releasing through the
    worker's own state machine (``RemoveReplicasEvent`` -> ``released`` transition) keeps the worker state, its memory
    accounting and its spill buffer consistent; a bare ``worker.data.pop`` would leave the task in ``memory`` forever.

    - ``:param local_worker_map:`` ``{worker_address: Worker}`` for the in-process workers.
    - ``:param written:`` The ``(addr, key)`` pairs already written to a worker store.
    - ``:param bridge_id:`` The owning bridge's rank, for the log line only.
    - ``:return:`` None.

    Best-effort: a release that itself fails must not mask the original failure, so errors are logged and swallowed.
    """
    by_addr: Dict[str, List[str]] = defaultdict(list)
    for addr, key in written:
        by_addr[addr].append(key)

    for addr, keys in by_addr.items():
        worker = local_worker_map[addr]
        try:
            worker.handle_stimulus(RemoveReplicasEvent(stimulus_id=f"deisa-scatter-rollback-{time.time()}", keys=keys))
        except Exception:
            logger.warning(
                f"[{bridge_id}] _release_local_keys: could not release {len(keys)} key(s) from in-process worker "
                f"{addr}: the worker may hold untracked data",
                exc_info=True,
            )
