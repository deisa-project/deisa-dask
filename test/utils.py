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
import multiprocessing
import threading
import time
from typing import Optional, Any, Literal, List, Sequence

import dask.array as da
from deisa.core import ICommunicator

from deisa.dask import Bridge


def wait_for(predicate, timeout=5.0, interval=0.01, nb_checks=1):
    """
    Wait until predicate() is True for nb_checks consecutive evaluations.

    Args:
        predicate: callable returning bool
        timeout: max time to wait (seconds)
        interval: delay between checks
        nb_checks: number of consecutive successful checks required

    Returns:
        True if condition satisfied, False otherwise
    """
    start = time.time()
    consecutive = 0

    while time.time() - start < timeout:
        if predicate():
            consecutive += 1
            if consecutive >= nb_checks:
                return True
        else:
            consecutive = 0  # reset if the condition breaks

        time.sleep(interval)

    return False


def dask_array_element_wise_equal(a, b):
    assert isinstance(a, da.Array) or issubclass(type(b), da.Array), "a and b must be dask arrays"
    assert isinstance(b, da.Array) or issubclass(type(b), da.Array), "a and b must be dask arrays"
    return (a.compute() == b.compute()).all(), "a and b are not equal"


def async_map(iterable, func, *args, **kwargs):
    async def _f():
        return await asyncio.gather(*[asyncio.to_thread(func, item, *args, **kwargs) for item in iterable])

    return asyncio.run(_f())


def async_close_bridges(bridges: List[Bridge], timestep: int):
    async def _close_bridges():
        await asyncio.gather(*[asyncio.to_thread(bridge.close, timestep=timestep) for bridge in bridges])

    asyncio.run(_close_bridges())


class FakeComm(ICommunicator):
    class State:
        def __init__(self, size: int, mode: Literal["thread", "process"] = "thread"):
            self.size = size

            if mode == "thread":
                self.condition = threading.Condition()
            elif mode == "process":
                self.condition = multiprocessing.Condition()
            else:
                raise ValueError(f"Invalid mode: {mode}")

            # gather state
            self.gather_data: dict[int, Any] = {}
            self.gather_result: Optional[list[Any]] = None
            self.gather_count = 0

            # bcast state
            self.bcast_value: Any = None
            self.bcast_ready = False
            self.bcast_count = 0

            # barrier state
            self.barrier_count = 0
            self.barrier_generation = 0

    def __init__(self, state: State, rank: int):
        self._state = state
        self._rank = rank

    def Get_rank(self) -> int:
        return self._rank

    def Get_size(self) -> int:
        return self._state.size

    def gather(self, data: Any, root: int = 0) -> Optional[list[Any]]:
        state = self._state

        with state.condition:
            state.gather_data[self._rank] = data

            if len(state.gather_data) == state.size:
                state.gather_result = [
                    state.gather_data[r]
                    for r in range(state.size)
                ]

                state.condition.notify_all()

            else:
                state.condition.wait_for(
                    lambda: state.gather_result is not None
                )

            result = state.gather_result

            state.gather_count += 1

            # Last rank resets collective state
            if state.gather_count == state.size:
                state.gather_data = {}
                state.gather_result = None
                state.gather_count = 0

        if self._rank == root:
            return result

        return None

    def bcast(self, obj: Any, root: int = 0) -> Any:
        state = self._state

        with state.condition:
            # Root publishes value
            if self._rank == root:
                state.bcast_value = obj
                state.bcast_ready = True

                state.condition.notify_all()

            else:
                # Wait for root
                state.condition.wait_for(
                    lambda: state.bcast_ready
                )

            result = state.bcast_value

            state.bcast_count += 1

            # Last rank resets collective state
            if state.bcast_count == state.size:
                state.bcast_value = None
                state.bcast_ready = False
                state.bcast_count = 0

            return result

    def barrier(self) -> None:
        state = self._state
        with state.condition:
            generation = state.barrier_generation
            state.barrier_count += 1
            # Last participant releases everybody
            if state.barrier_count == state.size:
                state.barrier_count = 0
                state.barrier_generation += 1
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.barrier_generation != generation)


class FakeCartComm(FakeComm):
    def __init__(self, state: FakeComm.State, rank: int, dims: Sequence[int], periods: Optional[Sequence[bool]] = None):
        super().__init__(state, rank)

        self._dims = tuple(dims)
        self._periods = tuple(periods or [False] * len(dims))

        size = 1
        for d in self._dims:
            size *= d

        if size != state.size:
            raise ValueError(f"Cartesian dimensions {self._dims} do not match communicator size {state.size}")

    @property
    def dims(self) -> tuple[int, ...]:
        return self._dims

    @property
    def periods(self) -> tuple[bool, ...]:
        return self._periods

    def Get_coords(self, rank: int) -> list[int]:
        """
        Compatible with mpi4py MPI.Cartcomm.Get_coords.

        Converts a linear rank into Cartesian coordinates
        using row-major ordering.
        """
        if not (0 <= rank < self._state.size):
            raise ValueError(f"Invalid rank: {rank}")

        coords = [0] * len(self._dims)
        r = rank
        for i in range(len(self._dims) - 1, -1, -1):
            dim = self._dims[i]
            coords[i] = r % dim
            r //= dim
        return coords

    def Get_cart_rank(self, coords: Sequence[int]) -> int:
        """
        Reverse mapping: Cartesian coordinates -> rank.
        """
        if len(coords) != len(self._dims):
            raise ValueError(f"Expected {len(self._dims)} coordinates, got {len(coords)}")

        rank = 0
        for coord, dim, periodic in zip(coords, self._dims, self._periods):
            c = coord
            if periodic:
                c %= dim
            elif not (0 <= c < dim):
                raise ValueError(f"Coordinate {coord} out of bounds for dim {dim}")
            rank = rank * dim + c
        return rank
