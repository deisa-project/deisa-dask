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
import logging
import uuid
from numbers import Number
from typing import Any, Iterator, List

import numpy as np
from dask.tokenize import tokenize
from deisa.core import validate_system_metadata, validate_arrays_metadata, IBridge, ICommunicator
from distributed import Client, Variable
from distributed.protocol import to_serialize
from distributed.utils_comm import scatter_to_workers
from tlz import valmap

from deisa.dask.communicator import resolve_comm
from deisa.dask.deisa import VARIABLE_PREFIX, CLIENT_KEY
from deisa.dask.handshake import Handshake

logger = logging.getLogger(__name__)


class Bridge(IBridge):
    def __init__(self, id: int,
                 arrays_metadata: dict[str, dict], system_metadata: dict[str, Any],
                 comm: ICommunicator = None, *args, **kwargs):
        """
        Initializes an object to manage communication between an MPI-based distributed
        system and a Dask-based framework. The class ensures proper allocation of workers
        among processes and instantiates the required communication objects like queues.

        :param id: Unique identifier in the computation. This may be the rank of this MPI process.
        :type id: int

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

        :param args: Passed to Communicator
        type: args: tuple

        :param kwargs: Passed to Handshake and Communicator
        :type kwargs: dict
        """
        super().__init__(id, arrays_metadata, system_metadata, *args, **kwargs)
        self.system_metadata = validate_system_metadata(system_metadata)
        self.client: Client = self.system_metadata['connection']
        self.arrays_metadata = validate_arrays_metadata(arrays_metadata)
        self.id = id
        self.workers = list(self.client.scheduler_info(n_workers=-1)["workers"].keys())
        self.comm: ICommunicator = resolve_comm(comm, use_mpi_if_available=True,
                                                client=self.client,
                                                size=self.system_metadata['nb_bridges'],
                                                *args, **kwargs)
        self._has_close_been_called = False

        logger.debug(f"[{self.id}] Bridge __init__() with:\n"
                     f"comm={self.comm}\n"
                     f"client={self.client}\n"
                     f"arrays_metadata={self.arrays_metadata}\n"
                     f"system_metadata={self.system_metadata}\n"
                     f"workers={self.workers}")

        # blocking until analytics is ready
        self.handshake = Handshake('bridge', self.client, id=id, max=self.system_metadata['nb_bridges'],
                                   arrays_metadata=self.arrays_metadata, **kwargs)

    def __del__(self):
        self.close()

    def close(self):
        logger.info(f"Closing Bridge. id={self.id}")
        if not self._has_close_been_called:
            self._has_close_been_called = True
            self.handshake.stop_bridge(self.id)

    def send(self, array_name: str, data: np.ndarray, iteration: int, chunked: bool = True, *args, **kwargs):
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

        rank = self.comm.Get_rank()
        workers = self.workers

        if 'update_workers' in kwargs and kwargs['update_workers']:
            # only update worker list if requested
            if rank == 0:
                # rank 0 retrieve workers and bcast to other bridges
                workers = self.client.scheduler_info(n_workers=-1)["workers"]
            else:
                workers = None

            # bcast
            logger.debug(f"[{self.id}] send() pre-bcast workers={workers}")
            workers = self.comm.bcast(workers, root=0)
            logger.debug(f"[{self.id}] send() post-bcast workers={workers}")

            # reformat workers to only keep addresses
            self.workers = list(workers.keys())

            if 'filter_workers' in kwargs:
                workers = kwargs['filter_workers'](workers)
                # check return type
                if not isinstance(workers, list):
                    raise TypeError(f"worker_filter must return a list, got {type(workers)}")
                for w in workers:
                    if not isinstance(w, str):
                        raise TypeError(f"worker_filter must return a list of strings, got {type(w)}")

        # Send data to worker
        res = self._better_scatter(data, workers=workers, hash=False)  # send data to workers

        # Barrier. Wait for all bridges.
        to_send = {
            'future-info': res,
            'placement': self.comm.Get_coords(rank) if hasattr(self.comm, 'Get_coords') else self.id
        }
        logger.debug(f"[{self.id}] send() gather: to_send={to_send}")
        gathered_data = self.comm.gather(to_send, root=0)
        logger.debug(f"[{self.id}] send() gathered_data={gathered_data}")

        if gathered_data is not None:
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
            self.client.sync(self.client.scheduler.update_data, who_has=who_has, nbytes=nbytes, client=self.client.id)

            # mimic mechanism from Queue. Keep a reference on keys until reception in topic handler.
            self.client._send_to_scheduler({"op": "client-desires-keys", "keys": keys, "client": CLIENT_KEY})

            to_send = {
                'array_name': array_name,
                'iteration': iteration,

                'futures': [{
                    'future': d['future-info']['future'],
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'placement': d['placement']
                } for d in gathered_data]
            }
            logger.debug(f"[{self.id}] send() log_event: to_send={gathered_data}")
            self.client.log_event(array_name, to_send)

        # TODO: what to do if error ?

    def get(self, key: str, default: Any = None, chunked: bool = False, delete: bool = True):
        def get_variable(dask_scheduler, name):
            ext = dask_scheduler.extensions["variables"]
            v = ext.variables.get(name)
            return v if v is not None else None

        if chunked:
            raise NotImplementedError()  # TODO
        else:
            var_name = f"{VARIABLE_PREFIX}{key}"
            is_set = self.client.run_on_scheduler(get_variable, name=var_name)
            if is_set:
                var = Variable(var_name, client=self.client)
                res = var.get()
                if delete:
                    var.delete()
                return res
            else:
                return default

    def _better_scatter(self, data: np.ndarray, workers: List[str] = None, hash=False):
        if workers is None:
            workers = self.workers

        return self.client.sync(
            self.__scatter,
            data,
            workers=workers,
            hash=hash)

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
                names = [type(x).__name__ + "-" + tokenize(x) for x in data]
            else:
                names = [type(x).__name__ + "-" + uuid.uuid4().hex for x in data]
            data = dict(zip(names, data))

        assert isinstance(data, dict)

        data2 = valmap(to_serialize, data)

        _, who_has, nbytes = await scatter_to_workers(workers, data2, self.client.rpc)

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
