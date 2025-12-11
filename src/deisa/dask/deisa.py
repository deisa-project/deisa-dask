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
from typing import overload, Callable, Union, Tuple, List, Iterable

import dask
import dask.array as da
import numpy as np
from dask.array import Array
from distributed import Client, Future, Queue

from deisa.dask.handshake import Handshake


class Deisa:
    SLIDING_WINDOW_THREAD_PREFIX = "deisa_sliding_window_callback_"

    Window = List[da.Array]
    Callback_args = Union[str, Tuple[str], Tuple[str, int]]  # array_name, window_size
    Callback = Callable[..., None]

    def __init__(self, get_connection_info: Callable[[], Client], *args, **kwargs):
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

    @overload
    def register_sliding_window_callback(self, callback: Callback,
                                         array_name: str, *, window_size: int = 1,
                                         exception_handler: Callable[
                                             [str, BaseException], None] = __default_exception_handler) -> None:
        ...

    @overload
    def register_sliding_window_callback(self, callback: Callback,
                                         *callback_args: Callback_args,
                                         exception_handler: Callable[
                                             [str, BaseException], None] = __default_exception_handler,
                                         when='AND') -> None:
        ...

    def register_sliding_window_callback(self, callback: Callback,
                                         *callback_args: Callback_args,
                                         window_size: int = 1,
                                         exception_handler: Callable[
                                             [str, BaseException], None] = __default_exception_handler,
                                         when: str = 'AND') -> None:
        """
            Supports:
              - (callback, "array_name", window_size=K)
              - (callback, ("name1", k1), ("name2", k2), ..., when='AND')
              - mixed: (callback, "a", ("b", 3)) -> "a" gets default window_size
            """
        if not callback_args:
            raise TypeError("register_sliding_window_callback requires at least one array name or (name, window) tuple")

        parsed: List[Tuple[str, int]] = []

        # single-string convenience: (callback, "my_array", window_size=...)
        if len(callback_args) == 1 and isinstance(callback_args[0], str):
            parsed.append((callback_args[0], int(window_size)))
        else:
            # general case: one or more of str or (str, int)
            for arg in callback_args:
                if isinstance(arg, str):
                    # use the function-level window_size for bare strings
                    parsed.append((arg, int(window_size)))
                elif isinstance(arg, tuple):
                    if len(arg) != 2:
                        raise TypeError("tuple callback_args must be of length 2: (array_name: str, window_size: int)")
                    name, ws = arg
                    if not isinstance(name, str):
                        raise TypeError("first element of tuple must be str (array name)")
                    if not isinstance(ws, int):
                        # allow things that can be casted to int, but be explicit
                        raise TypeError("second element of tuple must be int (window_size)")
                    parsed.append((name, ws))
                else:
                    raise TypeError("callback_args must be either str or (str, int) tuples")

        # validate of `when`
        if when not in ('AND', 'OR'):
            raise ValueError("when must be 'AND' or 'OR'")

        # validate that array_name is known
        for array_name, _ in parsed:
            if array_name not in self.arrays_metadata:
                raise ValueError(f'unknown array name: {array_name}')

        def queue_watcher(arrays: List[Tuple[str, int]]):
            current_windows = {}
            for array_name, window_size in arrays:
                current_windows[array_name] = {'window': collections.deque(maxlen=window_size), 'changed': False}

            t = threading.current_thread()
            while getattr(t, "stop", False) is False:
                for array_name, d in current_windows.items():
                    try:
                        darr, iteration = self.get_array(array_name, timeout='1s')
                        d['window'].append(darr)
                        d['changed'] = True

                        if when == 'OR':
                            windows = [list(dd['window']) for dd in current_windows.values()]
                            callback(*windows, iteration)
                            d['changed'] = False
                        elif when == 'AND':
                            all_changed = all([dd['changed'] for dd in current_windows.values()])
                            if all_changed:
                                windows = [list(dd['window']) for dd in current_windows.values()]
                                callback(*windows, iteration)

                                for dd in current_windows.values():
                                    dd['changed'] = False

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
                                self.unregister_sliding_window_callback(array_name)  # TODO

        callback_id = self.__get_callback_id(*parsed)
        if callback_id not in self.sliding_window_callback_threads:
            thread = threading.Thread(target=queue_watcher,
                                      name=f"{Deisa.SLIDING_WINDOW_THREAD_PREFIX}{callback_id}",
                                      args=(parsed,))
            self.sliding_window_callback_threads[callback_id] = thread
            thread.start()

    def unregister_sliding_window_callback(self, *array_names: Callback_args) -> None:
        """
        Unregisters a sliding window callback for the specified array name. This method removes the
        callback thread associated with the array name. If the thread exists, it stops the thread and waits
        for it to finish execution.

        :param array_name: The name of the array for which the sliding window callback is to be unregistered.
            Must be a string.
        :return: None
        """

        callback_id = self.__get_callback_id(*array_names)
        thread = self.sliding_window_callback_threads.pop(callback_id, None)
        if thread:
            self.__stop_join_thread(thread)

    def set(self, name: str, data: da.Array, chunked=False):
        raise NotImplementedError()  # TODO

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

    @staticmethod
    def __get_callback_id(*callback_args: Callback_args) -> str:
        """Flatten callback_args to a tuple of array names."""
        array_names = []
        for arg in callback_args:
            if isinstance(arg, str):
                array_names.append(arg)
            elif isinstance(arg, tuple):
                if len(arg) == 1 and isinstance(arg[0], str):
                    array_names.append(arg[0])
                elif len(arg) == 2 and isinstance(arg[0], str) and isinstance(arg[1], int):
                    array_names.append(arg[0])
                else:
                    raise TypeError(
                        "Tuple callback_args must be either (array_name,) or (array_name, window_size: int)")
            else:
                raise TypeError("callback_args must be str or a tuple")
        return str(array_names)
