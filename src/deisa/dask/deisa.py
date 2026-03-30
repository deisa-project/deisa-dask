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
import collections
import sys
import threading
import time
from typing import Callable, Union, Tuple, List, Final, Literal, Any

import dask
import dask.array as da
import numpy as np
from dask.array import Array
from deisa.core.interface import IDeisa, SupportsSlidingWindow
from distributed import Client, Future, Queue, Variable

from deisa.dask.handshake import Handshake

LOCK_PREFIX: Final[str] = "deisa_lock_"
VARIABLE_PREFIX: Final[str] = "deisa_variable_"
CALLBACK_PREFIX: Final[str] = "deisa_cb_"
DEFAULT_SLIDING_WINDOW_SIZE: int = 1


class Deisa(IDeisa):
    Callback_args = Union[str, Tuple[str], Tuple[str, int]]  # array_name, window_size
    Callback_id = str

    def __init__(self, get_connection_info: Callable[[], Client], *args, **kwargs):
        """
        Initializes the distributed processing environment and configures workers using
        a Dask scheduler. This class handles setting up a Dask client and ensures the
        specified number of workers are available for distributed computation tasks.

        :param get_connection_info: A function that returns a connected Dask Client.
        :type get_connection_info: Callable
        """
        # dask.config.set({"distributed.deploy.lost-worker-timeout": 60, "distributed.workers.memory.spill":0.97, "distributed.workers.memory.target":0.95, "distributed.workers.memory.terminate":0.99 })

        super().__init__(get_connection_info, *args, **kwargs)
        self.client: Client = get_connection_info()

        # blocking until all bridges are ready
        handshake = Handshake('deisa', self.client, **kwargs)

        self.mpi_comm_size = handshake.get_nb_bridges()
        self.arrays_metadata = handshake.get_arrays_metadata()

        self.received_metadata = dict[str, list[dict[str, Any]]]()  # array_name: list[metadata]
        self.current_sliding_windows = {}
        self._callbacks = {}  # callback_id -> metadata
        self._callback_seq = 0  # unique counter

        # example of arrays_metadata:
        # arrays_metadata = {
        #     'global_t': {
        #         'size': [20, 20],
        #         'subsize': [10, 10],
        #         'sliding_window_callbacks': {}
        #     }
        #     'global_p': {
        #         'size': [100, 100],
        #         'subsize': [50, 50],
        #         'sliding_window_callbacks': {}
        #     }

    def __del__(self):
        print("Deisa.__del__", flush=True)
        self.client.close()

    def close(self):
        self.__del__()

    def get_array(self, name: str, iteration=None, timeout=None) -> tuple[Array, int]:
        """Retrieve a Dask array for a given array name."""

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

        if name not in self.arrays_metadata:
            raise ValueError(f"Array '{name}' is not known.")

        start = time.time()

        while True:
            events = self.client.get_events(name)

            if events:
                # events: List[(ts, payload)]
                payloads = [e[1] for e in events]

                # Filter by iteration if requested
                if iteration is not None:
                    payloads = [p for p in payloads if p["iteration"] == iteration]

                if payloads:
                    # Take latest iteration if not specified
                    payload = max(payloads, key=lambda p: p["iteration"])
                    iteration = payload["iteration"]
                    parts = payload["futures"]

                    # reconstruct array
                    parts = sorted(parts, key=lambda p: p["placement"])

                    darr_chunks = [
                        da.from_delayed(
                            dask.delayed(Future(p["future"], client=self.client)),
                            shape=p["shape"],
                            dtype=p["dtype"],
                        )
                        for p in parts
                    ]

                    darr = self.__tile_dask_blocks(
                        darr_chunks,
                        self.arrays_metadata[name]["size"]
                    )

                    print(f"[ITER {iteration}] {name} shape={darr.shape}", flush=True)
                    return darr, iteration

            # timeout handling
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(f"Timeout waiting for array '{name}' iteration={iteration}")

            time.sleep(0.01)  # avoid busy spin

    @staticmethod
    def __default_exception_handler(callback_id: Callback_id, e):
        print(f"Exception thrown for callback id {callback_id}: {e}", file=sys.stderr, flush=True)

    def register_sliding_window_callback(self,
                                         callback: SupportsSlidingWindow.Callback,
                                         array_name: str, window_size: int = DEFAULT_SLIDING_WINDOW_SIZE,
                                         exception_handler: SupportsSlidingWindow.ExceptionHandler = __default_exception_handler) -> Callback_id:
        """
        Register a sliding-window callback for a single array.
        """
        parsed = [(array_name, window_size)]
        return self._register_sliding_window_callbacks_impl(
            callback,
            parsed,
            exception_handler=exception_handler,
            when='AND')

    def register_sliding_window_callbacks(self,
                                          callback: SupportsSlidingWindow.Callback,
                                          *callback_args: Callback_args,
                                          exception_handler: SupportsSlidingWindow.ExceptionHandler = __default_exception_handler,
                                          when: Literal['AND', 'OR'] = 'AND') -> Callback_id:
        """
        Register a sliding-window callback for one or more arrays.

        Supports:
          - "array"
          - ("array", window_size)
          - mixed forms
        """
        if not callback_args:
            raise TypeError(
                "register_sliding_window_callbacks requires at least one array name "
                "or (name, window_size) tuple"
            )

        parsed: List[Tuple[str, int]] = []

        for arg in callback_args:
            if isinstance(arg, str):
                parsed.append((arg, DEFAULT_SLIDING_WINDOW_SIZE))
            elif isinstance(arg, tuple):
                if len(arg) == 1:
                    parsed.append((arg[0], DEFAULT_SLIDING_WINDOW_SIZE))
                elif len(arg) == 2:
                    name, ws = arg
                    if not isinstance(name, str) or not isinstance(ws, int):
                        raise TypeError("tuple must be (str, int)")
                    parsed.append((name, ws))
                else:
                    raise TypeError("tuple must be (str,) or (str, int)")
            else:
                raise TypeError("callback_args must be str or tuple")

        return self._register_sliding_window_callbacks_impl(
            callback,
            parsed,
            exception_handler=exception_handler,
            when=when)

    def _register_sliding_window_callbacks_impl(self,
                                                callback: SupportsSlidingWindow.Callback,
                                                parsed: List[Tuple[str, int]],
                                                *,
                                                exception_handler: SupportsSlidingWindow.ExceptionHandler,
                                                when: Literal['AND', 'OR']) -> Callback_id:
        """
        Supports:
          - (callback, "array_name", window_size=K)
          - (callback, ("name1", k1), ("name2", k2), ..., when='AND')
          - mixed: (callback, "a", ("b", 3)) -> "a" gets default window_size
        """

        if when not in ('AND', 'OR'):
            raise ValueError("when must be 'AND' or 'OR'")

        for array_name, _ in parsed:
            if array_name not in self.arrays_metadata:
                raise ValueError(f'unknown array name: {array_name}')

        # ---- init sliding windows (shared storage) ----
        for arr_name, window_size in parsed:
            if arr_name not in self.current_sliding_windows:
                self.current_sliding_windows[arr_name] = {
                    "window": collections.deque(maxlen=window_size),
                    "changed": False,
                }

        array_names = self.__get_array_names(*parsed)

        # create unique callback id
        callback_id = self._next_callback_id()

        # store callback metadata
        self._callbacks[callback_id] = {
            "callback": callback,
            "parsed": parsed,
            "when": when,
            "exception_handler": exception_handler,
            "array_names": array_names,
        }

        async def topic_handler(event):
            def call_callback(cb, *args, **kwargs):
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(*args, **kwargs))
                else:
                    cb(*args, **kwargs)

            try:
                _, payload = event
                print(f"[Deisa] topic_handler({callback_id}): {payload}", flush=True)

                array_name = payload["array_name"]
                iteration = payload["iteration"]
                futures = payload["futures"]

                parts = sorted(futures, key=lambda p: p['placement'])

                darr_chunks = [
                    da.from_delayed(
                        dask.delayed(Future(p["future"], client=self.client)),
                        p["shape"],
                        dtype=p["dtype"],
                    )
                    for p in parts
                ]

                darr = self.__tile_dask_blocks(
                    darr_chunks,
                    self.arrays_metadata[array_name]["size"]
                )

                # update sliding window
                d = self.current_sliding_windows[array_name]
                d["window"].append(darr)
                d["changed"] = True

                ordered_array_names = list(self.current_sliding_windows.keys())

                windows = [
                    list(self.current_sliding_windows[name]["window"])
                    for name in ordered_array_names
                ]

                # trigger logic
                if when == "OR":
                    call_callback(callback, *windows, timestep=iteration)
                    d["changed"] = False

                else:  # AND
                    if all(self.current_sliding_windows[name]["changed"] for name in ordered_array_names):
                        call_callback(callback, *windows, timestep=iteration)

                        for name in ordered_array_names:
                            self.current_sliding_windows[name]["changed"] = False

            except BaseException as e:
                try:
                    exception_handler(callback_id, e)
                except BaseException:
                    print(f"Exception in exception handler. Unregistering {callback_id}", file=sys.stderr, flush=True)
                    self.unregister_sliding_window_callback(callback_id)

        print(f"[Deisa] register callback_id={callback_id}", flush=True)

        # subscribe (one handler per registration!)
        for array_name in array_names:
            self.client.subscribe_topic(array_name, topic_handler)

        return callback_id

    def unregister_sliding_window_callback(self, callback_id: Callback_id) -> None:
        meta = self._callbacks.pop(callback_id, None)
        if meta is None:
            return  # already removed or unknown id

        array_names = meta.get("array_names", [])

        # unsubscribe from all topics
        for array_name in array_names:
            try:
                self.client.unsubscribe_topic(array_name)
            except BaseException:
                # don't fail hard during cleanup
                pass

        # cleanup sliding window state (only if no other callback depends on it)
        try:
            for arr_name, _ in meta["parsed"]:
                still_used = any(arr_name in [a for a, _ in other["parsed"]]
                                 for other in self._callbacks.values())

                if not still_used:
                    self.current_sliding_windows.pop(arr_name, None)

        except BaseException:
            pass

    def set(self, key: str, data: Union[Future, object], chunked=False):
        if chunked:
            raise NotImplementedError()  # TODO
        else:
            var = Variable(f'{VARIABLE_PREFIX}{key}', client=self.client)
            if self.client.asynchronous:
                self.client.loop.add_callback(var.set, data)
            else:
                var.set(data)

    def delete(self, key: str) -> None:
        var = Variable(f'{VARIABLE_PREFIX}{key}', client=self.client)
        if self.client.asynchronous:
            self.client.loop.add_callback(var.delete)
        else:
            var.delete()

    def _next_callback_id(self):
        self._callback_seq += 1
        return f"{CALLBACK_PREFIX}{self._callback_seq}"

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
    def __get_array_names(*callback_args: Callback_args) -> List[str]:
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
        return array_names

    @staticmethod
    def __in_client_loop(client):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False

        return loop is client.loop.asyncio_loop

    @staticmethod
    def run_task_sync(coro, loop):
        container = {}
        done = threading.Event()

        def callback():
            task = asyncio.create_task(coro)

            async def wrapper():
                try:
                    container['result'] = await task
                finally:
                    done.set()

            asyncio.create_task(wrapper())

        loop.call_soon_threadsafe(callback)
        done.wait()
        return container.get('result')

    @staticmethod
    def make_topic(arrays, when) -> str:
        return f"{when}|" + "|".join(sorted(arrays))

    # async def topic_handler(event):
    #     """
    #     Runs when the bridge publishes an event.
    #     """
    #     try:
    #         # metadata = {
    #         #     'array_name': array_name,
    #         #     'rank': self.mpi_rank,
    #         #     'shape': data.shape,
    #         #     'dtype': str(data.dtype),
    #         #     'iteration': iteration,
    #         #     'future': f.key
    #         # }
    #         _, metadata = event
    #         array_name = metadata['array_name']
    #
    #         if 'array_name' not in metadata:
    #             raise ValueError(f"Metadata must contain 'array_name' key.")
    #
    #         if array_name not in self.received_metadata:
    #             self.received_metadata[array_name] = []
    #
    #         received_array_metadata = self.received_metadata[array_name]
    #
    #         received_array_metadata.append(metadata)
    #
    #         # will return None if did not receive all the chunks
    #         res = await self.__get_array(array_name, received_array_metadata)
    #         if res:
    #             darr, iteration = res
    #             assert isinstance(darr, Array), "darr must be a Dask array."
    #             assert isinstance(iteration, int), "iteration must be an integer."
    #
    #             print(f"__get_array: {darr.shape=}, {iteration=}", flush=True)
    #
    #             # remove iteration from received_metadata
    #             self.received_metadata[array_name] = [m for m in received_array_metadata if
    #                                                   m["iteration"] != iteration]
    #
    #             d = self.current_sliding_windows[array_name]
    #             d["window"].append(darr)
    #             d["changed"] = True
    #
    #             # convert deque to list
    #             windows = [list(dd["window"]) for dd in self.current_sliding_windows.values()]
    #
    #             if when == "OR":
    #                 callback(*windows, timestep=iteration)
    #                 d["changed"] = False
    #
    #             else:  # AND
    #                 if all(dd["changed"] for dd in self.current_sliding_windows.values()):
    #                     callback(*windows, timestep=iteration)
    #                     for dd in self.current_sliding_windows.values():
    #                         dd["changed"] = False
    #
    #     except BaseException as e:
    #         try:
    #             exception_handler(arr_name, e)
    #         except BaseException:
    #             print(f"Exception thrown in exception handler for {arr_name}", file=sys.stderr)
