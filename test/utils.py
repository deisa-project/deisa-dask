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


def infer_parallelism_auto(global_size: Sequence[int], max_splits: int = 1) -> tuple[int, ...]:
    ndim = len(global_size)

    # start with 1 split per dimension
    parallelism = [1] * ndim

    # distribute splits as evenly as possible
    base = max_splits // ndim
    remainder = max_splits % ndim

    for i in range(ndim):
        parallelism[i] += base

    # distribute leftover splits to largest dimensions first
    dims = sorted(range(ndim), key=lambda i: global_size[i], reverse=True)

    for i in range(remainder):
        parallelism[dims[i]] += 1

    return tuple(parallelism)


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
            self.gather_phase = 0

            # bcast state
            self.bcast_value: Any = None
            self.bcast_count = 0
            self.bcast_phase = 0

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
                # Phase 0 → 1: pre-barrier (synchronize entry)
                state.gather_count += 1
                if state.gather_count == state.size:
                    state.gather_phase = 1
                    state.gather_count = 0
                    state.condition.notify_all()
                else:
                    state.condition.wait_for(lambda: state.gather_phase == 1)

                # Phase 1 → 2: contribute data, build result
                state.gather_data[self._rank] = data
                if len(state.gather_data) == state.size:
                    state.gather_result = [
                        state.gather_data[r]
                        for r in range(state.size)
                    ]
                    state.gather_phase = 2
                    state.condition.notify_all()
                else:
                    state.condition.wait_for(lambda: state.gather_result is not None)

                result = state.gather_result

                # Phase 2 → 0: post-barrier (synchronize exit / reset)
                state.gather_count += 1
                if state.gather_count == state.size:
                    state.gather_data = {}
                    state.gather_result = None
                    state.gather_phase = 0
                    state.gather_count = 0
                    state.condition.notify_all()
                else:
                    state.condition.wait_for(lambda: state.gather_phase == 0)

            if self._rank == root:
                return result

            return None

    def bcast(self, obj: Any, root: int = 0) -> Any:
        state = self._state

        with state.condition:
            # Phase 0 → 1: pre-barrier to synchronize entry
            state.bcast_count += 1
            if state.bcast_count == state.size:
                state.bcast_phase = 1
                state.bcast_count = 0
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.bcast_phase >= 1)

            # Phase 1 → 2: root publishes, non-roots wait then read.
            if self._rank == root:
                state.bcast_value = obj
                state.bcast_phase = 2
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.bcast_phase >= 2)

            result = state.bcast_value

            # Phase 2 → 0: post-barrier to synchronize exit / reset
            state.bcast_count += 1
            if state.bcast_count == state.size:
                state.bcast_value = None
                state.bcast_phase = 0
                state.bcast_count = 0
                state.condition.notify_all()
            else:
                gen = state.bcast_count  # capture; wait for phase reset
                state.condition.wait_for(lambda: state.bcast_phase == 0)

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

    def Split(self, color: int, key: int = 0) -> 'FakeComm':
        """Mimic MPI.Comm.Split() — creates a sub-communicator for the given color.

        Tracks which ranks share a color so that Get_size() on the sub-comm
        returns the number of participating ranks. Ranks with color == -1
        (MPI.UNDEFINED) get a COMM_NULL sub-comm.

        Collectives on the sub-comm only synchronize with ranks of the same color.
        """
        state = self._state

        # Register this rank's color (all ranks must participate, even with -1)
        with state.condition:
            if not hasattr(state, '_split_colors'):
                state._split_colors = {}
                state._split_sub_states = {}
            state._split_colors[self._rank] = color
            if len(state._split_colors) == state.size:
                state.condition.notify_all()
            else:
                state.condition.wait_for(
                    lambda: len(getattr(state, '_split_colors', {})) == state.size
                )

        # Non-participating ranks return COMM_NULL
        if color == -1:
            # Wait for the group to finish setup before returning
            with state.condition:
                state._split_ready = getattr(state, '_split_ready', 0) + 1
                if state._split_ready == state.size:
                    state._split_colors = {}
                    state._split_ready = 0
                    state.condition.notify_all()
                else:
                    state.condition.wait_for(
                        lambda: len(getattr(state, '_split_colors', {})) == 0
                    )
            return _NullComm()

        # Determine this rank's sub-group
        group_ranks = sorted(r for r, c in state._split_colors.items() if c == color)
        sub_size = len(group_ranks)
        sub_rank = group_ranks.index(self._rank)

        # Create a shared sub-comm state (only the first rank creates it,
        # then all ranks retrieve the same object)
        with state.condition:
            if color not in state._split_sub_states:
                state._split_sub_states[color] = _SubCommState(state, group_ranks)
            sub_state = state._split_sub_states[color]

        sub = _SubComm(parent_comm=self, sub_state=sub_state, color=color,
                       sub_rank=sub_rank, sub_size=sub_size)

        # Wait for all ranks to create their sub-comms, then reset
        with state.condition:
            state._split_ready = getattr(state, '_split_ready', 0) + 1
            if state._split_ready == state.size:
                state._split_colors = {}
                state._split_ready = 0
                state.condition.notify_all()
            else:
                state.condition.wait_for(
                    lambda: len(getattr(state, '_split_colors', {})) == 0
                )

        return sub

    def Free(self):
        """No-op for FakeComm."""


class _SubCommState:
    """Synchronization state for a sub-communicator group."""

    def __init__(self, parent_state, group_ranks):
        self._parent_state = parent_state
        self._group_ranks = group_ranks
        self.condition = threading.Condition()
        self.gather_data = {}
        self.gather_result = None
        self.gather_count = 0
        self.gather_phase = 0
        self.bcast_value = None
        self.bcast_count = 0
        self.bcast_phase = 0
        self.barrier_count = 0
        self.barrier_generation = 0


class _NullComm(ICommunicator):
    """Represents MPI.COMM_NULL — a communicator that is not valid for use."""

    def Get_rank(self) -> int:
        raise RuntimeError("COMM_NULL has no rank")

    def Get_size(self) -> int:
        return 0

    def gather(self, data: Any, root: int = 0) -> None:
        return None

    def bcast(self, obj: Any, root: int = 0) -> None:
        return None

    def barrier(self) -> None:
        pass

    def Split(self, color: int, key: int = 0) -> '_NullComm':
        return self

    def Free(self):
        pass


class _SubComm(ICommunicator):
    """A sub-communicator created by FakeComm.Split().

    Collectives only synchronize with ranks in the same color group.
    """

    def __init__(self, parent_comm, sub_state, color, sub_rank, sub_size):
        self._parent = parent_comm
        self._state = sub_state
        self._color = color
        self._sub_rank = sub_rank
        self._sub_size = sub_size

    def Get_rank(self) -> int:
        return self._sub_rank

    def Get_size(self) -> int:
        return self._sub_size

    def gather(self, data: Any, root: int = 0) -> Optional[list[Any]]:
        state = self._state
        with state.condition:
            # Phase 0 → 1: pre-barrier (synchronize entry)
            state.gather_count += 1
            if state.gather_count == self._sub_size:
                state.gather_phase = 1
                state.gather_count = 0
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.gather_phase >= 1)

            # Phase 1 → 2: contribute data, build result
            state.gather_data[self._sub_rank] = data
            if len(state.gather_data) == self._sub_size:
                state.gather_result = [state.gather_data[r] for r in range(self._sub_size)]
                state.gather_phase = 2
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.gather_result is not None)

            result = state.gather_result

            # Phase 2 → 0: post-barrier (synchronize exit / reset)
            state.gather_count += 1
            if state.gather_count == self._sub_size:
                state.gather_data = {}
                state.gather_result = None
                state.gather_phase = 0
                state.gather_count = 0
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.gather_phase == 0)

        if self._sub_rank == root:
            return result
        return None

    def bcast(self, obj: Any, root: int = 0) -> Any:
        state = self._state
        with state.condition:
            # Phase 0 → 1: pre-barrier
            state.bcast_count += 1
            if state.bcast_count == self._sub_size:
                state.bcast_phase = 1
                state.bcast_count = 0
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.bcast_phase == 1)

            # Phase 1 → 2: root publishes, non-roots read
            if self._sub_rank == root:
                state.bcast_value = obj
                state.bcast_phase = 2
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.bcast_phase == 2)

            result = state.bcast_value

            # Phase 2 → 0: post-barrier
            state.bcast_count += 1
            if state.bcast_count == self._sub_size:
                state.bcast_value = None
                state.bcast_phase = 0
                state.bcast_count = 0
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.bcast_phase == 0)

        return result

    def barrier(self) -> None:
        state = self._state
        with state.condition:
            generation = state.barrier_generation
            state.barrier_count += 1
            if state.barrier_count == self._sub_size:
                state.barrier_count = 0
                state.barrier_generation += 1
                state.condition.notify_all()
            else:
                state.condition.wait_for(lambda: state.barrier_generation != generation)

    def Split(self, color: int, key: int = 0) -> 'FakeComm':
        return self._parent.Split(color, key)

    def Free(self):
        pass


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
