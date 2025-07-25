# =============================================================================
# Copyright (C) 2015-2023 Commissariat a l'energie atomique et aux energies alternatives (CEA)
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

import dask
import dask.array as da
import numpy as np
from dask.distributed import get_client, comm, Queue, Variable

QUEUE_PREFIX = "queue_"


def get_bridge_instance(dask_scheduler_address: str, mpi_comm_size: int, mpi_rank: int,
                        arrays_metadata: dict[str, dict], **kwargs):
    return Bridge(dask_scheduler_address, mpi_comm_size, mpi_rank, arrays_metadata, **kwargs)


class Bridge:
    def __init__(self, dask_scheduler_address: str, mpi_comm_size: int, mpi_rank: int,
                 arrays_metadata: dict[str, dict], **kwargs):
        """
        Initializes an object to manage communication between an MPI-based distributed
        system and a Dask-based framework. The class ensures proper allocation of workers
        among processes and instantiates the required communication objects like queues.

        :param dask_scheduler_address: The address of the Dask scheduler to connect to.
        :param mpi_comm_size: The size of the MPI communicator (total number of MPI
                              processes).
        :param mpi_rank: The rank of the current MPI process.
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
        """
        self.client = get_client(dask_scheduler_address)
        self.mpi_rank = mpi_rank
        self.arrays_metadata = arrays_metadata

        # TODO: check this
        # Note: Blocking call. Simulation will wait for the analysis code to be run.
        # Variable("workers") is set in the Deisa class.
        workers = Variable("workers", client=self.client).get()
        if mpi_comm_size > len(workers):  # more processes than workers
            self.workers = [workers[mpi_rank % len(workers)]]
        else:
            k = len(workers) // mpi_comm_size  # more workers than processes
            self.workers = workers[mpi_rank * k:mpi_rank * k + k]

        if self.mpi_rank == 0:
            Queue("Arrays", client=self.client).put(self.arrays_metadata)

        self.queue = Queue(QUEUE_PREFIX + str(self.mpi_rank), client=self.client)

    def publish_data(self, array_name: str, data: np.array):
        """
        Publishes data to the distributed workers and communicates metadata and data future via a queue. This method is used
        to send data to workers in a distributed computing setup and ensures that both the metadata about the data and the
        data itself (in the form of a future) are made available to the relevant processes. Metadata includes information
        such as iteration number, MPI rank, data shape, and data type.

        :param array_name: Name of the array associated with the data
        :type array_name: str
        :param data: The data to be distributed among the workers
        :type data: numpy.ndarray
        :return: None
        """
        # TODO: check that client is connected

        f = self.client.scatter(data, direct=True, workers=self.workers)  # send data to workers

        metadata = {
            'rank': self.mpi_rank,
            'shape': data.shape,
            'dtype': data.dtype
        }

        # Queue(array_name, client=self.client).put(metadata)
        q = Queue(array_name, client=self.client)
        q.put(metadata)  # put metadata
        q.put(f)  # put future

        # self.queue.put(metadata)  # put metadata
        # self.queue.put(f)  # put future
        # TODO: what to do if error ?


class Deisa(object):

    def __init__(self, dask_scheduler_address: str, mpi_comm_size: int, nb_workers: int):
        # dask.config.set({"distributed.deploy.lost-worker-timeout": 60, "distributed.workers.memory.spill":0.97, "distributed.workers.memory.target":0.95, "distributed.workers.memory.terminate":0.99 })

        self.client = get_client(dask_scheduler_address)

        # Wait for all workers to be available.
        self.workers = [comm.get_address_host_port(i, strict=False) for i in
                        self.client.scheduler_info()["workers"].keys()]
        while len(self.workers) != nb_workers:
            self.workers = [comm.get_address_host_port(i, strict=False) for i in
                            self.client.scheduler_info()["workers"].keys()]

        Variable("workers", client=self.client).set(self.workers)

        # print(self.workers)
        self.mpi_comm_size = mpi_comm_size
        self.queues = [Queue(QUEUE_PREFIX + str(i), client=self.client) for i in range(mpi_comm_size)]
        self.arrays_metadata = None

    def get_array(self, name: str) -> da.Array:

        if self.arrays_metadata is None:
            self.arrays_metadata = Queue("Arrays",
                                         client=self.client).get()  # {'my_array': {'size': (32,32), 'subsize': (16,16), 'dtype': float}}
        # arrays_metadata will look something like this:
        # arrays_metadata = {
        #     'global_t': {
        #         'size': [20, 20]
        #         'subsize': [10, 10]
        #     }
        #     'global_p': {
        #         'size': [100, 100]
        #         'subsize': [50, 50]
        #     }

        if self.arrays_metadata.get(name) is None:
            raise ValueError(f"Array '{name}' is not known.")

        res = []
        l = self.client.sync(self.__get_all_chunks, Queue(name, client=self.client), self.mpi_comm_size)
        for m, f in l:
            m["da"] = da.from_delayed(dask.delayed(f), m["shape"], dtype=m["dtype"])
            res.append(m)

        # create dask array from blocks
        res.sort(key=lambda x: x['rank'])  # sort by mpi rank
        chunks = [item['da'] for item in res]  # extract ordered dask arrays
        darr = self.__tile_dask_blocks(chunks, self.arrays_metadata[name]['size'])
        return darr

    @staticmethod
    async def __get_all_chunks(q: Queue, mpi_comm_size: int):
        res = []
        for _ in range(mpi_comm_size):
            res.append(q.get(batch=2))
        return await asyncio.gather(*res)

    # @staticmethod
    # async def __get_all_chunks(queues):
    #     res = []
    #     for q in queues:
    #         res.append(q.get(batch=2))
    #     return await asyncio.gather(*res)

    @staticmethod
    def __get_from_queue(q: Queue) -> dict:
        metadata, data_future = q.get(batch=2)  # metadata + data future
        metadata['da'] = da.from_delayed(dask.delayed(data_future),
                                         shape=metadata["shape"],
                                         dtype=metadata["dtype"])
        return metadata

    @staticmethod
    def __tile_dask_blocks(blocks: list[da.Array], global_shape: tuple[int, ...]) -> da.Array:
        """
        Given a flat list of N-dimensional Dask arrays, tile them into a single Dask array.
        The tiling layout is inferred from the provided global shape.

        Parameters:
            blocks (list of dask.array): Flat list of Dask arrays. All must have the same shape.
            global_shape (tuple of int): Shape of the full array to reconstruct.

        Returns:
            dask.array.Array: Combined tiled Dask array.
        """
        if not blocks:
            raise ValueError("No blocks provided.")

        block_shape = blocks[0].shape
        ndim = len(block_shape)

        if len(global_shape) != ndim:
            raise ValueError("global_shape must have the same number of dimensions as blocks.")

        # Check that all blocks have the same shape
        for b in blocks:
            if b.shape != block_shape:
                raise ValueError("All blocks must have the same shape.")

        # Compute how many blocks are needed per dimension
        tile_counts = tuple(g // b for g, b in zip(global_shape, block_shape))

        if np.prod(tile_counts) != len(blocks):
            raise ValueError(
                f"Mismatch between number of blocks ({len(blocks)}) and expected number from global_shape {global_shape} "
                f"with block shape {block_shape} (expected {np.prod(tile_counts)} blocks)."
            )

        # Reshape the flat list into an N-dimensional grid of blocks
        def nest_blocks(flat_blocks, shape):
            """Nest a flat list of blocks into a nested list matching the grid shape."""
            if len(shape) == 1:
                return flat_blocks
            else:
                size = shape[0]
                stride = int(len(flat_blocks) / size)
                return [nest_blocks(flat_blocks[i * stride:(i + 1) * stride], shape[1:]) for i in range(size)]

        nested = nest_blocks(blocks, tile_counts)

        # Use da.block to combine blocks
        return da.block(nested)
