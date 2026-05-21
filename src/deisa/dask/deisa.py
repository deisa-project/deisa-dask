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
import logging
import threading
from typing import Callable, Union, Tuple, List, Final, Literal, Any, Dict, Set, Collection

import dask
import dask.array as da
import numpy as np
from deisa.core import CallbackArgs, Window
from deisa.core.interface import IDeisa
from distributed import Client, Future, Queue, Event

from deisa.dask.handshake import Handshake
from deisa.dask.utils import get_client, build_deisa_array

LOCK_PREFIX: Final[str] = "deisa_lock_"
FEEDBACK_QUEUE_PREFIX: Final[str] = "deisa_feedback_queue_"
CALLBACK_PREFIX: Final[str] = "deisa_cb_"
DEFAULT_SLIDING_WINDOW_SIZE: int = 1
CLIENT_KEY: Final[str] = "deisa"

logger = logging.getLogger(__name__)


class Deisa(IDeisa):
    Callback_args = Union[str, Tuple[str], Tuple[str, int]]  # array_name, window_size
    Callback_id = str

    def __init__(self, feedback_queue_size: int = 1024, *args, **kwargs) -> None:
        """
        Initializes the distributed processing environment and configures workers using
        a Dask scheduler. This class handles setting up a Dask client and ensures the
        specified number of workers are available for distributed computation tasks.

        :param get_connection_info: A function that returns a connected Dask Client.
        :type get_connection_info: Callable
        """
        # dask.config.set({"distributed.deploy.lost-worker-timeout": 60, "distributed.workers.memory.spill":0.97, "distributed.workers.memory.target":0.95, "distributed.workers.memory.terminate":0.99 })

        super().__init__(feedback_queue_size, *args, **kwargs)
        self.client: Client = get_client(timeout=kwargs.get("timeout", 10))
        self.feedback_queue_size = feedback_queue_size

        # blocking until all bridges are ready
        self.handshake = Handshake(self.client)
        self.handshake.deisa_ready(feedback_queue_size=feedback_queue_size, **kwargs)

        self.mpi_comm_size = self.handshake.get_nb_bridges()
        self.arrays_metadata = self.handshake.get_arrays_metadata()

        self.received_metadata = dict[str, list[dict[str, Any]]]()  # array_name: list[metadata]
        self.current_sliding_windows = {}

        self._callbacks: Dict[Deisa.Callback_id, Dict] = {}
        self._callbacks_by_array: Dict[str, Set[Deisa.Callback_id]] = {}
        self._topic_handlers: Dict[str, Callable] = {}
        self._callback_seq = 0  # unique counter
        self._received_futures: Set[str] = set()

    @staticmethod
    def __default_exception_handler(exception: BaseException):
        logger.error(f"Exception thrown for callback id: {exception}")

    def register(self, *callback_args: CallbackArgs,
                 exception_handler: IDeisa.ExceptionHandler = __default_exception_handler,
                 when: Literal['AND', 'OR'] = 'AND') -> Callable:
        """
        Decorator that registers the decorated function as a sliding-window callback.

        Supports for callback_args:
        Default window size is 1.
        @deisa.register("arr1")                             # default window size
        @deisa.register("arr1", "arr2")                     # two arrays, default window size
        @deisa.register(Window("arr1"))                     # default window size
        @deisa.register(Window("arr1", 2))                  # window size 2
        @deisa.register(Window("arr1", 2), Window("arr2", 5))   # window size 2 for arr1 and 5 for arr2
        @deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3") # window size 2 for arr1 and 5 for arr2, default window size for arr3
        """

        def decorator(callback: IDeisa.Callback) -> IDeisa.Callback:
            return self.register_callback(callback, *callback_args,
                                          exception_handler=exception_handler,
                                          when=when)

        return decorator

    def register_callback(self, callback: IDeisa.Callback, *callback_args: CallbackArgs,
                          exception_handler: IDeisa.ExceptionHandler = __default_exception_handler,
                          when: Literal['AND', 'OR'] = 'AND') -> Callable:
        """
        Decorator that registers the decorated function as a sliding-window callback.

        Supports for callback_args:
        Default window size is 1.
        @deisa.register("arr1")                             # default window size
        @deisa.register("arr1", "arr2")                     # two arrays, default window size
        @deisa.register(Window("arr1"))                     # default window size
        @deisa.register(Window("arr1", 2))                  # window size 2
        @deisa.register(Window("arr1", 2), Window("arr2", 5))   # window size 2 for arr1 and 5 for arr2
        @deisa.register(Window("arr1", 2), Window("arr2", 5), "arr3") # window size 2 for arr1 and 5 for arr2, default window size for arr3
        """
        logger.debug(f"register_callback: callback={callback}, callback_args={callback_args}")
        if not callback_args:
            raise TypeError("register_callback requires at least one array name or a list of Window(name, window_size)")

        parsed: List[Window] = []

        for arg in callback_args:
            if isinstance(arg, str):
                parsed.append(Window(arg, size=DEFAULT_SLIDING_WINDOW_SIZE))
            elif isinstance(arg, Window):
                parsed.append(arg)
            else:
                raise TypeError("callback_args must be str or tuple")

        callback_id = self._register_callback_impl(
            callback,
            parsed,
            exception_handler=exception_handler,
            when=when)
        callback.callback_id = callback_id
        return callback

    def _register_callback_impl(self,
                                callback: IDeisa.Callback,
                                parsed: List[Window],
                                exception_handler: IDeisa.ExceptionHandler,
                                when: Literal['AND', 'OR']) -> Callback_id:

        if when not in ('AND', 'OR'):
            raise ValueError("when must be 'AND' or 'OR'")

        for array_name, _ in parsed:
            if array_name not in self.arrays_metadata:
                raise ValueError(f'unknown array name: {array_name}')

        array_names = self.__get_array_names(*parsed)
        callback_id = self.__next_callback_id()

        logger.debug(f"_register_callback_impl: register callback_id={callback_id}")

        # per-callback state
        callback_state = {
            arr_name: {
                "window": collections.deque(maxlen=ws),
                "changed": False,
            }
            for arr_name, ws in parsed
        }

        self._callbacks[callback_id] = {
            "callback": callback,
            "when": when,
            "exception_handler": exception_handler,
            "array_names": array_names,
            "state": callback_state,
        }

        for array_name in array_names:
            self._callbacks_by_array.setdefault(array_name, set()).add(callback_id)

            # create handler only once per topic
            if array_name not in self._topic_handlers:
                handler = self._make_topic_handler(array_name)
                self._topic_handlers[array_name] = handler

                logger.debug(f"_register_callback_impl: subscribe_topic() {array_name}")
                self.client.subscribe_topic(array_name, handler)

        return callback_id

    def unregister_callback(self, callback_id: Callback_id) -> None:
        # also accept a decorated callback function, which stores its id in .callback_id
        callback_id = getattr(callback_id, 'callback_id', callback_id)
        cb_data = self._callbacks.pop(callback_id, None)
        if cb_data is None:
            return

        for array_name in cb_data["array_names"]:
            s = self._callbacks_by_array.get(array_name)
            if s:
                s.discard(callback_id)
                if not s:
                    del self._callbacks_by_array[array_name]

    def set(self, key: str, value: Any, timestep: int) -> None:
        logger.debug(f"set() key={key}, value={value}, timestep={timestep}")

        q = Queue(f'{FEEDBACK_QUEUE_PREFIX}{key}', client=self.client, maxsize=self.feedback_queue_size)

        # TODO: check for consistency
        # if q.qsize() > 0:
        #     # check for consistency
        #     t, _ = q.get()
        #     if timestep < t:
        #         raise ValueError(f"timestep {timestep} is smaller than previous timestep {t}")
        #     elif timestep == t:
        #         raise ValueError(f"timestep {timestep} has already been set")

        value = (timestep, value)
        q.put(value)

    def execute_callbacks(self) -> None:
        """
        Blocking call to execute all registered callbacks.
        This method unblocks the simulation and ends when the Bridges are closed.
        """
        logger.info("execute_callbacks()")

        logger.info("Bridges are ready, unblock bridges")
        Event("deisa_wait", client=self.client).set()

        logger.info("execute_callbacks() waiting for bridges")
        self.handshake.wait_for_bridges_to_finish()
        logger.info("execute_callbacks() done")

    def _make_topic_handler(self, array_name):
        async def topic_handler(event):
            try:
                _, payload = event

                logger.debug(f"topic_handler: array_name={array_name}, payload={payload}")

                iteration = payload["iteration"]
                futures = payload["futures"]
                keys = [p["future"] for p in futures]

                self.__update_futures_ownership(keys)
                self._received_futures.update(keys)

                parts = sorted(futures, key=lambda p: p['placement'])

                darr_chunks = [
                    da.from_delayed(
                        dask.delayed(Future(p["future"], client=self.client)),
                        shape=p["shape"], dtype=p["dtype"],
                    ) for p in parts]

                darr = self.__tile_dask_blocks(darr_chunks, self.arrays_metadata[array_name]["global_shape"])

                # dispatch to interested callbacks
                for callback_id in list(self._callbacks_by_array.get(array_name, [])):
                    cb_data = self._callbacks.get(callback_id)
                    if cb_data is None:
                        continue

                    if array_name not in cb_data["state"]:
                        continue

                    try:
                        self._process_callback(callback_id, cb_data, array_name, darr, iteration)
                    except Exception as e:
                        self._handle_callback_exception(callback_id, cb_data, e)

            except Exception as e:
                logger.error(f"topic_handler: topic handler error array_name={array_name}, e={e}")

        return topic_handler

    def _process_callback(self, callback_id, cb_data, array_name, darr, iteration):
        state = cb_data["state"]

        # update sliding window
        d = state[array_name]
        d["window"].append(build_deisa_array(darr, iteration))
        d["changed"] = True

        ordered_array_names = cb_data["array_names"]

        windows = [list(state[name]["window"]) for name in ordered_array_names]

        def _call_callback():
            async def _run():
                try:
                    await asyncio.to_thread(
                        cb_data["callback"],
                        *windows
                    )
                except Exception as ex:
                    self._handle_callback_exception(callback_id, cb_data, ex)

            asyncio.create_task(_run())

        if cb_data["when"] == "OR":
            _call_callback()
            d["changed"] = False

        else:  # AND
            if all(state[name]["changed"] for name in ordered_array_names):
                _call_callback()

                for name in ordered_array_names:
                    state[name]["changed"] = False

    def _handle_callback_exception(self, callback_id, cb_data, ex):
        try:
            handler = cb_data["exception_handler"]
            if handler:
                handler(ex)
        except BaseException:
            logger.info(f"Exception in exception handler. Unregistering callback_id={callback_id}")
            self.unregister_callback(callback_id)

    def __update_futures_ownership(self, futures: Collection[Future]):
        # tell scheduler that my client is using these futures
        self.client._send_to_scheduler({"op": "client-desires-keys", "keys": futures, "client": self.client.id})
        # tell scheduler that deisa client no longer needs the futures
        self.client._send_to_scheduler({"op": "client-releases-keys", "keys": futures, "client": CLIENT_KEY})

    def __next_callback_id(self):
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
