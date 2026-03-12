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
import uuid
from numbers import Number
from typing import Any, Iterator, List

import numpy as np
from dask.tokenize import tokenize
from deisa.core import validate_system_metadata, validate_arrays_metadata, IBridge
from distributed import Client, Queue, Variable, Lock, Future
from distributed.protocol import to_serialize
from distributed.utils import TimeoutError
from distributed.utils_comm import scatter_to_workers
from tlz import valmap

from deisa.dask.deisa import LOCK_PREFIX, VARIABLE_PREFIX
from deisa.dask.handshake import Handshake


class Bridge(IBridge):

    def __init__(self, id: int, arrays_metadata: dict[str, dict], system_metadata: dict[str, Any], *args, **kwargs):
        """
        Initializes an object to manage communication between an MPI-based distributed
        system and a Dask-based framework. The class ensures proper allocation of workers
        among processes and instantiates the required communication objects like queues.

        :param id: Unique identifier in the computation. This may be the rank of this MPI process.
        :type id: int

        :param mpi_comm_size: Total number of MPI processes involved in the computation.
        :type mpi_comm_size: int

        :param arrays_metadata: A dictionary containing metadata about the Dask arrays
                eg: arrays_metadata = {
                    'global_t': {
                        'size': [20, 20]
                        'subsize': [10, 10]
                    }
                    'global_p': {
                        'size': [100, 100]
                        'subsize': [50, 50]
                    }
        :type arrays_metadata: dict[str, dict]

        :param connection: A function that returns a connected Dask Client.
        :type connection: Callable

        :param kwargs: Passed to Handshake
        :type kwargs: dict
        """
        super().__init__(id, arrays_metadata, system_metadata, *args, **kwargs)
        self.system_metadata = validate_system_metadata(system_metadata)
        self.client: Client = self.system_metadata['connection']
        self.arrays_metadata = validate_arrays_metadata(arrays_metadata)
        self.mpi_rank = id
        self.workers = list(self.client.scheduler_info()["workers"].keys())

        # blocking until analytics is ready
        Handshake('bridge', self.client, id=id, max=self.system_metadata['nb_bridges'],
                  arrays_metadata=self.arrays_metadata, **kwargs)

    def __del__(self):
        self.client.close()
        self.system_metadata.clear()

    def close(self):
        self.__del__()

    def send(self, array_name: str, data: np.ndarray, iteration: int, chunked: bool = True):
        """
        Publishes data to the distributed workers and communicates metadata and data future via a queue. This method is used
        to send data to workers in a distributed computing setup and ensures that both the metadata about the data and the
        data itself (in the form of a future) are made available to the relevant processes. Metadata includes information
        such as iteration number, MPI rank, data shape, and data type.

        :param array_name: Name of the array associated with the data
        :type array_name: str
        :param data: The data to be distributed among the workers
        :type data: numpy.ndarray
        :param iteration: The iteration number associated with the data
        :type iteration: int
        :param chunked: Defines if the data is a chunk.
        :type chunked: bool
        :return: None
        """

        assert self.client.status == 'running', "Client is not connected to a scheduler. Please check your connection."

        # TODO: select workers to send data to.
        f = self._better_scatter(data, workers=self.workers)  # send data to workers

        to_send = {
            'rank': self.mpi_rank,
            'shape': data.shape,
            'dtype': data.dtype,
            'iteration': iteration,
            'future': f
        }

        q = Queue(array_name, client=self.client)
        q.put(to_send)

        # TODO: what to do if error ?

    def get(self, key: str, default: Any = None, chunked: bool = False, delete: bool = True):
        if chunked:
            raise NotImplementedError()  # TODO
        else:
            try:
                with Lock(f'{LOCK_PREFIX}{key}'):
                    return Variable(f'{VARIABLE_PREFIX}{key}', client=self.client).get(timeout=0)
            except TimeoutError:
                return default
            finally:
                if delete:
                    with Lock(f'{LOCK_PREFIX}{key}'):
                        Variable(f'{VARIABLE_PREFIX}{key}', client=self.client).delete()

    def _better_scatter(self, data: np.ndarray, workers: List[str] = None):
        if workers is None:
            workers = self.workers

        return self.client.sync(
            self.__scatter,
            data,
            workers=workers,
            hash=False)

    async def __scatter(self, data, workers=None, hash=True):
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
                names = [type(x).__name__ + "-" + tokenize(x) for x in data]
            else:
                names = [type(x).__name__ + "-" + uuid.uuid4().hex for x in data]
            data = dict(zip(names, data))

        assert isinstance(data, dict)

        types = valmap(type, data)
        data2 = valmap(to_serialize, data)

        _, who_has, nbytes = await scatter_to_workers(workers, data2, self.client.rpc)

        # TODO: maybe the workers can update the scheduler ?
        await self.client.scheduler.update_data(
            who_has=who_has, nbytes=nbytes, client=self.client.id
        )

        out = {k: Future(k, self.client) for k in data}
        for key, typ in types.items():
            self.client.futures[key].finish(type=typ)

        if issubclass(input_type, (list, tuple, set, frozenset)):
            out = input_type(out[k] for k in names)

        if unpack:
            assert len(out) == 1
            out = list(out.values())[0]
        return out
