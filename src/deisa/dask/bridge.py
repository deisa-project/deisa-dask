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
import uuid
from collections import deque, defaultdict
from numbers import Number
from typing import Any, Iterator, List, Dict, Optional, Union, Deque

import numpy as np
from dask.tokenize import tokenize
from deisa.core import validate_arrays_metadata, IBridge, ICommunicator
from distributed import Queue, Client
from distributed.protocol import to_serialize
from distributed.utils_comm import scatter_to_workers
from tlz import valmap

from deisa.dask.constants import KEY_PREFIX, FEEDBACK_QUEUE_PREFIX, CLIENT_KEY
from deisa.dask.handshake import Handshake
from deisa.dask.utils import get_client

logger = logging.getLogger(__name__)

# Sentinel values for MPI.Comm.Split() — used until ICommunicator is updated
# to expose these constants. These mirror mpi4py's MPI.UNDEFINED and MPI.COMM_NULL.
_UNDEFINED = -1
_COMM_NULL = None


class Bridge(IBridge):
    def __init__(self, comm: ICommunicator, arrays_metadata: Dict[str, Dict], *args, **kwargs):
        """
        Initializes an instance of the class, setting up communication, metadata validation,
        client connection (for id=0), workers initialization, and handshake configuration for the bridge.

        :param comm: An ICommunicator facilitating communication between processes.
            Must provide Get_rank(), Get_size(), gather(), bcast(), barrier(),
            Split(color, key), and Free() — the same API as an MPI communicator.
        :param arrays_metadata: Dictionary containing metadata for arrays.
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

        :type arrays_metadata: Dict[str, Dict]
        :param args: Additional positional arguments for the initialization.
        :param kwargs: Additional keyword arguments for the initialization. Can include
            configuration parameters like timeout used during client setup.
        """
        super().__init__(comm, arrays_metadata, *args, **kwargs)
        self.comm: ICommunicator = comm
        self.id = self.comm.Get_rank()
        self.arrays_metadata = validate_arrays_metadata(arrays_metadata)
        self._feedback_queues = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self._has_close_been_called = False
        self.workers = None
        self.handshake = None
        self.client: Optional[Client] = None
        self._array_comms: Dict[str, Any] = {}  # array_name -> sub-comm (from comm.Split)

        if self.id == 0:
            # only id 0 has a real dask client
            self.client = get_client(timeout=kwargs.get("timeout", 10), name="bridge")
            assert self.client, "client cannot be None for Bridge id 0."
            # get all workers from scheduler
            self.workers = self.client.scheduler_info(n_workers=-1)["workers"]

        # retrieve workers from rank 0 and bcast
        logger.debug(f"[{self.id}] Bridge __init__(): pre-bcast")
        self.workers = self.comm.bcast(self.workers, root=0)
        logger.debug(f"[{self.id}] Bridge __init__(): post-bcast. workers={self.workers}")

        # Auto-discover array participation and create sub-communicators
        self._setup_array_comms()

        if self.id == 0:
            # all bridges are ready, tell handshake actor
            assert self.client is not None, "client cannot be None for Bridge id 0."
            self.handshake = Handshake(self.client)
            self.handshake.all_bridges_ready(nb_bridge=self.comm.Get_size(),
                                             arrays_metadata=self.arrays_metadata, **kwargs)

    def _setup_array_comms(self):
        """Auto-discover array participation and create per-array sub-communicators.

        Uses comm.Split() to create a sub-communicator for each array.
        The color is derived from a consistent hash of the array name so that
        all participating ranks get the same color. Ranks that don't participate
        in an array (chunk_position is None) use _UNDEFINED and get
        _COMM_NULL.
        """
        for array_name, meta in self.arrays_metadata.items():
            chunk_position = meta.get('chunk_position')
            participates = chunk_position is not None

            # All ranks must call Split() — even non-participants
            if participates:
                # Use a consistent hash so all participating ranks get the same color
                color = hash(array_name) % (2**31)
            else:
                color = _UNDEFINED
            key = self.id  # preserve ordering
            sub_comm = self.comm.Split(color, key)
            self._array_comms[array_name] = sub_comm
            logger.debug(
                f"[{self.id}] _setup_array_comms: array={array_name}, "
                f"participates={participates}, sub_comm_size="
                f"{sub_comm.Get_size() if sub_comm is not _COMM_NULL else 'NULL'}"
            )

    def __del__(self):
        """Clean up resources before destruction."""
        self.close(timestep=sys.maxsize)

    def close(self, timestep: int) -> None:
        """Close the bridge: synchronize bridges, free sub-comms, shut down client."""
        logger.info(f"[{self.id}] Bridge close()")
        try:
            if not self._has_close_been_called:
                self._has_close_been_called = True
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
                    self.client.close()
        except Exception as e:
            logger.error(f"[{self.id}] Cloud not cleanly close bridge. exception={e}")

    def send(self, array_name: str, chunk: np.ndarray, timestep: int, *args, **kwargs):
        """Distribute a data chunk to a Dask worker via scatter + gather.

        Scatters the chunk to the next worker (round-robin), then gathers all
        chunks across bridges to rank 0, which updates the Dask scheduler.

        For single-bridge arrays (sub_comm_size == 1), skips the gather entirely
        and updates the scheduler directly.

        Supported kwargs:
            update_workers (bool): refresh the worker list from the scheduler.
            filter_workers (callable): filter available workers before sending.
        """
        logger.debug(f"[{self.id}] send() array_name={array_name}, data.shape={chunk.shape}, iteration={timestep}")

        if array_name not in self.arrays_metadata:
            raise ValueError(f"array {array_name} is unknown.")

        assert isinstance(self.workers, dict)
        workers = dict(self.workers)  # make a copy so that the user-defined function does not modify self

        if kwargs.get('update_workers', False):
            # only update worker list if requested
            if self.id == 0:
                assert self.client is not None, "client cannot be None for Bridge id 0."
                # rank 0 retrieve workers and bcast to other bridges
                workers = self.client.scheduler_info(n_workers=-1)["workers"]

            # bcast
            logger.debug(f"[{self.id}] send() pre-bcast workers={workers}")
            self.workers = self.comm.bcast(workers, root=0)
            logger.debug(f"[{self.id}] send() post-bcast workers={workers}")
            workers = dict(self.workers)

        if kwargs.get('filter_workers', False):
            workers = kwargs['filter_workers'](workers)
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

        # Send data to worker
        res = self._better_scatter(chunk, workers=workers, hash=False)  # send data to workers

        # Get per-array metadata
        meta = self.arrays_metadata[array_name]

        # === Determine communicator from cached sub-comms (from comm.Split()) ===
        sub_comm = self._array_comms.get(array_name)

        if sub_comm == _COMM_NULL:
            # This rank doesn't participate in this array
            logger.debug(f"[{self.id}] send() rank not in participating set for '{array_name}', skipping")
            return

        comm_to_use = sub_comm if sub_comm is not None else self.comm
        sub_comm_size = comm_to_use.Get_size()

        # === Single-bridge fast-path: no collective needed ===
        if sub_comm_size == 1:
            self._direct_send(array_name, res, chunk, timestep)
            return

        to_send = {
            'future-info': res,
            'chunk_position': meta['chunk_position']
        }
        logger.debug(f"[{self.id}] send() gather: to_send={to_send}")

        gathered_data = comm_to_use.gather(to_send, root=0)

        logger.debug(f"[{self.id}] send() gathered_data={gathered_data}")

        if gathered_data is not None:
            assert self.client is not None, "client cannot be None for Bridge id 0."
            # rank 0 (root=0 in comm.gather): aggregate who_has from all chunks
            who_has = {}
            nbytes = {}
            keys = []
            for d in gathered_data:
                who_has.update(d['future-info']['who_has'])
                nbytes.update(d['future-info']['nbytes'])
                keys.append(d['future-info']['future'])

            # only update the scheduler with who has what and register the future once
            self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)

            # mimic mechanism from Queue. Keep a reference on keys until reception in topic handler.
            # TODO: id=0 can use a queue
            self.client._send_to_scheduler({"op": "client-desires-keys", "keys": keys, "client": CLIENT_KEY})

            to_send = {
                'array_name': array_name,
                'iteration': timestep,
                'futures': [{
                    'future': d['future-info']['future'],
                    'shape': chunk.shape,
                    'dtype': str(chunk.dtype),
                    'chunk_position': d['chunk_position']
                } for d in gathered_data]
            }
            logger.debug(f"[{self.id}] send() log_event: array={array_name}, timestep={timestep}, n_futures={len(gathered_data)}")
            self.client.log_event(array_name, to_send)

        # TODO: what to do if error ?

    def _direct_send(self, array_name: str, res: dict, chunk: np.ndarray, timestep: int):
        """Handle single-bridge array send without collective.

        For arrays that exist on only one bridge, we skip the gather() entirely
        and directly update the Dask scheduler.
        """
        assert self.client is not None, "client cannot be None for single-bridge send."

        future_key = res['future']
        who_has = res['who_has']
        nbytes = res['nbytes']

        self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes)
        self.client._send_to_scheduler({
            "op": "client-desires-keys",
            "keys": [future_key],
            "client": CLIENT_KEY
        })
        to_send = {
            'array_name': array_name,
            'iteration': timestep,
            'futures': [{
                'future': future_key,
                'shape': chunk.shape,
                'dtype': str(chunk.dtype),
                'chunk_position': self.arrays_metadata[array_name]['chunk_position']
            }]
        }
        self.client.log_event(array_name, to_send)

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
                    'q': Queue(f'{FEEDBACK_QUEUE_PREFIX}{key}', client=self.client, maxsize=feedback_queue_size),
                    'deque': deque(maxlen=feedback_queue_size)}

            q: Queue = fb_state[key]['q']
            d: deque = fb_state[key]['deque']

            if q.qsize() != 0:
                # List[(int, Any), ...]
                full_q = q.get(batch=True)  # get all elements. This pops elements from the Dask queue.
                for v in full_q: d.append(v)  # add all elements to deque
            logger.debug(f"[{self.id}] get() fb_state={fb_state}")

        d = self.comm.bcast(fb_state[key]['deque'], root=0)

        if timestep is None:
            return d

        for t, v in d:
            if timestep == t:
                # found the timestep
                return v

        return default

    def _better_scatter(self, data: np.ndarray, workers: List[str] = None, hash=False):
        logger.debug(f"[{self.id}] scatter to {workers}")

        if workers is None:
            workers = self.workers

        if self.client:
            return self.client.sync(
                self.__scatter,
                data,
                workers=workers,
                hash=hash)
        else:
            return asyncio.run(self.__scatter(data, workers=workers, hash=hash))

    async def __scatter(self, data, workers=None, hash=False):
        if isinstance(workers, (str, Number)):
            workers = [workers]
        if isinstance(data, type(range(0))):
            data = list(data)

        input_type = type(data)
        names = False
        unpack = False
        if isinstance(data, Iterator):
            data = list(data)
        if isinstance(data, (set, frozenset)):
            data = list(data)
        if not isinstance(data, (dict, list, tuple, set, frozenset)):
            unpack = True
            data = [data]
        if isinstance(data, (list, tuple)):
            if hash:
                names = [KEY_PREFIX + "-" + type(x).__name__ + "-" + tokenize(x) for x in data]
            else:
                names = [KEY_PREFIX + "-" + type(x).__name__ + "-" + uuid.uuid4().hex for x in data]
            data = dict(zip(names, data))

        assert isinstance(data, dict)

        data2 = valmap(to_serialize, data)

        _, who_has, nbytes = await scatter_to_workers(workers, data2)

        out = {
            k: {
                'future': k,
                'who_has': who_has,
                'nbytes': nbytes
            }
            for k in data
        }

        if issubclass(input_type, (list, tuple, set, frozenset)):
            out = input_type(out[k] for k in names)

        if unpack:
            assert len(out) == 1
            out = list(out.values())[0]
        return out
