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


class Bridge(IBridge):
    def __init__(self, comm: ICommunicator, arrays_metadata: Dict[str, Dict], *args, **kwargs):
        """
        Initializes an instance of the class, setting up communication, metadata validation,
        client connection (for id=0), workers initialization, and handshake configuration for the bridge.

        :param comm: An instance of ICommunicator facilitating communication between processes.
        :param arrays_metadata: Dictionary containing metadata for arrays, validated during initialization.
            eg: arrays_metadata = {
                    'temperature': {
                        'global_shape': [20, 20],
                        'chunk_shape': [10, 10],
                        'chunk_position': [0, 0]
                    }
                    'pressure': {
                        'global_shape': [20, 20],
                        'chunk_shape': [10, 10],
                        'chunk_position': [0, 0]
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

        if self.id == 0:
            # all bridges are ready, tell handshake actor
            assert self.client is not None, "client cannot be None for Bridge id 0."
            self.handshake = Handshake(self.client)
            self.handshake.all_bridges_ready(nb_bridge=self.comm.Get_size(),
                                             arrays_metadata=self.arrays_metadata, **kwargs)

    def __del__(self):
        """
        Cleans up resources used by the object before it gets destroyed.

        This method is called when the object is about to be destroyed and ensures that
        any required cleanup operations are performed. The `close` method is invoked
        with a timestep set to the maximum possible value.

        :param timestep: A value to specify the timestep for cleanup operations. This
            is set to the maximum integer value available in Python.
        :type timestep: int
        """
        self.close(timestep=sys.maxsize)

    def close(self, timestep: int) -> None:
        """
        Attempts to close the bridge connection. This involves ensuring the bridge is properly cleaned up,
        orchestrating communication with other bridges, and notifying the analytics of the closure.
        The method ensures that it is only executed once during the lifecycle of the instance.

        :param timestep: The current timestep associated with the closure action.
        :type timestep: int
        :return: None
        """
        logger.info(f"[{self.id}] Bridge close()")
        try:
            if not self._has_close_been_called:
                self._has_close_been_called = True
                self.comm.barrier()
                if self.id == 0:
                    assert self.handshake, "handshake cannot be None for Bridge id 0."
                    assert self.client, "client cannot be None for Bridge id 0."
                    self.handshake.set_bridges_done(timestep=timestep)
                    self.client.close()
        except Exception as e:
            logger.error(f"[{self.id}] Cloud not cleanly close bridge. exception={e}")

    def send(self, array_name: str, chunk: np.ndarray, timestep: int, *args, **kwargs):
        """
        Handles the distribution of the given data chunk to workers in the Dask cluster.
        This method sends the data directly to the workers.

        :param array_name: The name of the data array being sent as a string.
            This should match what is defined in the Bridge arrays_metadata.
        :param chunk: A numpy ndarray containing the data chunk to be sent to the workers.
        :param timestep: The current timestep associated to the sent data chunk.
        :param args: Additional positional arguments if required by the method implementation.
        :param kwargs: Additional keyword arguments for optional configurations.
            Supported keys include:
            - `update_workers` (bool): If True, updates the workers' list by retrieving it from the scheduler.
            - `filter_workers` (callable): A function that filters the available workers
              and returns a list of worker names. Must return a non-empty list of strings.

        :return: None. All operations are internal and side effects include sending data
            to workers, logging the event, and synchronizing worker states.
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

        # Barrier. Wait for all bridges.
        to_send = {
            'future-info': res,
            'chunk_position': self.arrays_metadata[array_name]['chunk_position']
        }
        logger.debug(f"[{self.id}] send() gather: to_send={to_send}")
        gathered_data = self.comm.gather(to_send, root=0)
        logger.debug(f"[{self.id}] send() gathered_data={gathered_data}")

        if gathered_data is not None:
            assert self.client is not None, "client cannot be None for Bridge id 0."
            # rank 0 (root=0 in comm.gather)
            # aggregate who has what
            who_has = {}
            nbytes = {}
            keys = []
            for d in gathered_data:
                who_has = {**who_has, **d['future-info']['who_has']}
                nbytes = {**nbytes, **d['future-info']['nbytes']}
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
            logger.debug(f"[{self.id}] send() log_event: to_send={gathered_data}")
            self.client.log_event(array_name, to_send)

        # TODO: what to do if error ?

    def get(self, key: str, timestep: Optional[int] = None, default: Any = None) -> Optional[Union[Deque, Any]]:
        """
        Retrieve an element associated with a specific key and optional timestep from a feedback queue.
        If a queue for the key does not exist, it initializes the queue for the specified key.

        :param key: The unique identifier for the feedback queue.
        :type key: str
        :param timestep: An optional specific timestep to look for. If None, returns the entire deque.
        :type timestep: Optional[int]
        :param default: The default value to return if the specified timestep is not found.
        :type default: Any
        :return: The element associated with the specified timestep if found, the entire deque if no
            timestep is specified, or the default value if the timestep is not found.
        :rtype: Optional[Union[Deque, Any]]
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
