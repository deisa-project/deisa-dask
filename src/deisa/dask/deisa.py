# =============================================================================
# Copyright (C) 2025 Commissariat a l'energie atomique et aux energies alternatives (CEA)
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
import collections
import gc
import sys
import threading
import traceback
from typing import Callable

import dask
import dask.array as da
import numpy as np
from dask.array import Array
from dask.distributed import Queue
from distributed import Client, Future

from deisa.dask.handshake import Handshake


class Deisa:
    SLIDING_WINDOW_THREAD_PREFIX = "deisa_sliding_window_callback_"

    def __init__(self, get_connection_info: Callable, *args, **kwargs):
        """
        Initializes the distributed processing environment and configures workers using
        a Dask scheduler. This class handles setting up a Dask client and ensures the
        specified number of workers are available for distributed computation tasks.

        :param get_connection_info: A function that returns a connected Dask Client.
        :type get_connection_info: Callable
        """
        # dask.config.set({"distributed.deploy.lost-worker-timeout": 60, "distributed.workers.memory.spill":0.97, "distributed.workers.memory.target":0.95, "distributed.workers.memory.terminate":0.99 })

        self.client: Client = get_connection_info()

        # blocking until all bridges are ready
        handshake = Handshake('deisa', self.client, **kwargs)

        self.mpi_comm_size = handshake.get_nb_bridges()
        self.arrays_metadata = handshake.get_arrays_metadata()
        self.sliding_window_callback_threads: dict[str, threading.Thread] = {}
        self.sliding_window_callback_thread_lock = threading.Lock()

    def __del__(self):
        if hasattr(self, 'sliding_window_callback_threads'):  # may not be the case if an exception is thrown in ctor
            for thread in self.sliding_window_callback_threads.values():
                self.__stop_join_thread(thread)
            gc.collect()

    @staticmethod
    def __stop_join_thread(thread: threading.Thread):
        thread.stop = True
        thread.join()

        # exc = getattr(thread, "exception", None)
        # if exc:
        #     # print(f"Exception encountered: {exc['traceback']}", file=sys.stderr, flush=True)
        #     # raise exc['exception']
        #     pass

    def close(self):
        self.__del__()

    def get_array(self, name: str, timeout=None) -> tuple[Array, int]:
        """Retrieve a Dask array for a given array name."""

        # if self.arrays_metadata is None:
        #     self.arrays_metadata = Queue("Arrays", client=self.client).get(timeout=timeout)
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
        iteration = 0
        l = self.client.sync(self.__get_all_chunks, Queue(name, client=self.client),
                             self.mpi_comm_size, timeout=timeout)
        for m in l:
            assert type(m) is dict, "Metadata must be a dictionary."
            assert type(m['future']) is Future, "Data future must be a Dask future."
            m["da"] = da.from_delayed(dask.delayed(m['future']), m["shape"], dtype=m["dtype"])
            res.append(m)
            iteration = m["iteration"]

        # create dask array from blocks
        res.sort(key=lambda x: x['rank'])  # sort by mpi rank
        chunks = [item['da'] for item in res]  # extract ordered dask arrays
        darr = self.__tile_dask_blocks(chunks, self.arrays_metadata[name]['size'])
        return darr, iteration

    @staticmethod
    def __default_exception_handler(array_name, e):
        print(f"Exception from {array_name} thread: {e}", file=sys.stderr, flush=True)

    def register_sliding_window_callback(self, array_name: str, callback: Callable[[list[da.Array], int], None],
                                         window_size: int = 1,
                                         exception_handler: Callable[
                                             [str, BaseException], None] = __default_exception_handler):
        """
        Registers a sliding window callback that processes a fixed-size window of arrays over a period.
        This method allows monitoring arrays associated with a specific name and performing operations
        based on a sliding window of recent arrays. The callback will be triggered whenever a new
        array is added to the window.

        The method starts a background thread to watch for updates to the specified array. The thread
        retrieves arrays from the internal system queue with the given `array_name` and manages the
        sliding window. Upon each update, the user-provided `callback` function is invoked with the
        current window and iteration index.

        :param array_name: The name of the array to monitor for updates.
        :param callback: A callable to execute whenever the sliding window is updated. The callable
            receives the current sliding window as a list of arrays and the iteration index as arguments.
        :param window_size: The number of arrays to maintain in the sliding window. Defaults to 1.
        :param exception_handler: A callable to execute whenever the callback raises an exception.
            If exception_handler raises an exception, callback is unregistered.
        :return: None
        """

        def queue_watcher():
            print(f"Starting sliding window callback for array '{array_name}'", flush=True)
            current_window = collections.deque(maxlen=window_size)
            t = threading.current_thread()

            while getattr(t, "stop", False) is False:
                try:
                    darr, iteration = self.get_array(array_name, timeout='1s')
                    current_window.append(darr)
                    callback(list(current_window), iteration)
                except TimeoutError:
                    pass
                except BaseException as e:
                    setattr(t, "exception", (e, traceback.format_exc()))
                    try:
                        if exception_handler:
                            exception_handler(array_name, e)
                    except BaseException as e:
                        with self.sliding_window_callback_thread_lock:
                            print(
                                f"Exception thrown in exception handler for {array_name} thread: {e} Unregistering callback.",
                                file=sys.stderr)
                            self.unregister_sliding_window_callback(array_name)

        if array_name not in self.sliding_window_callback_threads:
            thread = threading.Thread(target=queue_watcher, name=f"{Deisa.SLIDING_WINDOW_THREAD_PREFIX}{array_name}")
            self.sliding_window_callback_threads[array_name] = thread
            thread.start()

    def unregister_sliding_window_callback(self, array_name: str):
        """
        Unregisters a sliding window callback for the specified array name. This method removes the
        callback thread associated with the array name. If the thread exists, it stops the thread and waits
        for it to finish execution.

        :param array_name: The name of the array for which the sliding window callback is to be unregistered.
            Must be a string.
        :return: None
        """
        thread = self.sliding_window_callback_threads.pop(array_name, None)
        if thread:
            self.__stop_join_thread(thread)

    @staticmethod
    async def __get_all_chunks(q: Queue, mpi_comm_size: int, timeout=None) -> list[tuple[dict, Future]]:
        """This will return a list of tuples (metadata, data_future) for all chunks in the queue."""
        try:
            res = []
            for _ in range(mpi_comm_size):
                res.append(q.get(timeout=timeout))
            return await asyncio.gather(*res)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout reached while waiting for chunks in queue '{q.name}'.")

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
