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
import threading
import uuid
from typing import Optional

import numpy as np
from deisa.core import ICommunicator
from distributed import Client

logger = logging.getLogger(__name__)


def is_mpi_comm(comm):
    try:
        from mpi4py import MPI
        return isinstance(comm, MPI.Comm)
    except ImportError:
        return False


def is_running_on_mpi():
    try:
        import mpi4py
        mpi4py.rc.initialize = False
        from mpi4py import MPI
        return MPI.Is_initialized()  # and MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        return False


def resolve_comm(comm, cart_coord_dims=1, use_mpi_if_available=True, *args, **kwargs) -> ICommunicator:
    """
    handle 3 cases to resolve comm:
    - if comm is None: use_mpi_if_available or no MPI
    - if comm is an MPI Comm: use it
    """
    if comm is None:
        if use_mpi_if_available and is_running_on_mpi():
            try:
                from mpi4py import MPI
                mpi_comm = MPI.COMM_WORLD
                dims = MPI.Compute_dims(mpi_comm.Get_size(), dims=cart_coord_dims)
                cart_comm = mpi_comm.Create_cart(dims)
                return cart_comm
            except ImportError:
                return CommClient(*args, **kwargs)
        return CommClient(*args, **kwargs)

    if is_mpi_comm(comm) or isinstance(comm, CommClient):
        return comm

    raise TypeError("Invalid communicator: expected MPI communicator or CommClient, or None")


class CommActor:
    def __init__(self, size: int):
        self.size = size

        # rank management
        self.ranks = {}  # client_id -> rank
        self.next_rank = 0

        # collective state
        self.gathers = {}  # seq -> list[(rank, value)]

        # broadcast state
        self.bcast_data = None

    # rank management
    def register(self, cid: str) -> int:
        if cid not in self.ranks:
            if self.next_rank >= self.size:
                raise RuntimeError("Too many participants")

            self.ranks[cid] = self.next_rank
            self.next_rank += 1

        return self.ranks[cid]

    # gather
    def gather_add(self, seq: int, rank: int, value):
        if seq not in self.gathers:
            self.gathers[seq] = []
        self.gathers[seq].append((rank, value))

    def gather_ready(self, seq: int):
        return len(self.gathers.get(seq, [])) >= self.size

    def gather_get(self, seq: int):
        return self.gathers.pop(seq, [])

    # cartesian topology
    def get_coords(self, rank: int, dims):
        return tuple(int(c) for c in np.unravel_index(rank, dims))

    # broadcast
    def bcast_set(self, obj):
        self.bcast_data = obj

    def bcast_ready(self):
        return self.bcast_data is not None

    def bcast_get(self):
        return self.bcast_data

    def cleanup(self):
        self.bcast_data = None


class CommState:
    def __init__(self, scheduler, size: int):
        self.scheduler = scheduler
        self.size = size

        self.ranks = {}
        self.next_rank = 0

        self.gathers = {}  # seq -> [(rank, data)]
        self.gather_futures = {}  # seq -> Future

        self.bcast_futures = {}  # seq -> Future
        self.bcast_readers = {}  # seq -> set

    def register(self, cid: str) -> int:
        if cid not in self.ranks:
            self.ranks[cid] = self.next_rank
            self.next_rank += 1
        return self.ranks[cid]

    def get_size(self) -> int:
        return self.size

    async def gather(self, seq: str, rank: int, data):
        assert seq is not None, "seq must be provided"
        assert rank is not None, "rank must be provided"
        assert 0 <= rank < self.size, f"Invalid rank: {rank}"

        loop = asyncio.get_running_loop()
        g = self.gathers.setdefault(seq, [])
        fut = self.gather_futures.get(seq)
        if fut is None:
            fut = loop.create_future()
            self.gather_futures[seq] = fut

        g.append((rank, data))

        if len(g) == self.size:
            g.sort(key=lambda x: x[0])
            result = [v for _, v in g]

            if not fut.done():
                fut.set_result(result)

        result = await fut

        if rank == 0:
            self.gathers.pop(seq, None)
            self.gather_futures.pop(seq, None)

        return result

    async def bcast(self, seq: str, rank: int, obj, root: int):
        assert seq is not None, "seq must be provided"
        assert rank is not None, "rank must be provided"
        assert 0 <= rank < self.size, f"Invalid rank: {rank}"

        loop = asyncio.get_running_loop()
        fut = self.bcast_futures.get(seq)
        if fut is None:
            fut = loop.create_future()
            self.bcast_futures[seq] = fut
            self.bcast_readers[seq] = set()

        # root sets value
        if rank == root and not fut.done():
            fut.set_result(obj)

        value = await fut

        readers = self.bcast_readers[seq]
        readers.add(rank)

        if len(readers) == self.size:
            self.bcast_futures.pop(seq, None)
            self.bcast_readers.pop(seq, None)

        return value


def setup_comm(dask_scheduler, size: int):
    logger.debug(f"setup_comm: size={size}")
    if "deisa_register" not in dask_scheduler.handlers:
        comm = CommState(dask_scheduler, size=size)
        dask_scheduler.handlers.update({
            "deisa_register": comm.register,
            "deisa_get_size": comm.get_size,
            "deisa_gather": comm.gather,
            "deisa_bcast": comm.bcast
        })
    else:
        logger.info(f"Deisa DaskComm in already registered. Ignoring.")


class CommClient:
    def __init__(self, comm_state_rpc, client: Optional[Client] = None, *args, **kwargs):
        logger.debug(
            f"CommClient.__init__(): comm_state_rpc={comm_state_rpc}, client={client}, args={args}, kwargs={kwargs}")
        self.comm_state_rpc = comm_state_rpc
        self.client = client
        self._rank = None
        self._seq = 0

        if self.client is None:
            # persistent loop for raw RPC
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
            self._thread.start()

    def _next_seq(self):
        s = self._seq
        self._seq += 1
        return s

    def _run(self, func, *args, **kwargs):
        if self.client:
            return self.client.sync(func, *args, **kwargs)
        else:
            fut = asyncio.run_coroutine_threadsafe(func(*args, **kwargs), self._loop)
            return fut.result()

    async def Get_rank_async(self):
        if self._rank is None:
            cid = str(uuid.uuid4())
            self._rank = await self.comm_state_rpc.deisa_register(cid=cid)
        return self._rank

    def Get_rank(self):
        if self._rank is None:
            cid = str(uuid.uuid4())
            self._rank = self._run(self.comm_state_rpc.deisa_register, cid=cid)
        return self._rank

    def Get_size(self) -> int:
        return self._run(self.comm_state_rpc.deisa_get_size)

    def gather(self, data, root=0):
        async def _gather_async():
            seq = self._next_seq()
            rank = await self.Get_rank_async()

            r = await self.comm_state_rpc.deisa_gather(seq=seq, rank=rank, data=data)

            if rank == root:
                return r
            return None

        return self._run(_gather_async)

    def bcast(self, obj, root=0):
        async def _bcast_async():
            seq = self._next_seq()
            rank = await self.Get_rank_async()
            return await self.comm_state_rpc.deisa_bcast(seq=seq, rank=rank, obj=obj, root=root)

        return self._run(_bcast_async)
