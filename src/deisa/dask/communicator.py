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
import time
import uuid
from typing import Any, Optional, List, Sequence

import numpy as np
from deisa.core import ICommunicator
from distributed import Client

from deisa.dask.utils import _get_actor


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
        return MPI.Is_initialized() and MPI.COMM_WORLD.Get_size() > 1
    except ImportError:
        return False


def resolve_comm(comm, cart_coord_dims=1, use_mpi_if_available=True, *args, **kwargs) -> ICommunicator:
    """
    1 comm per array
    handle 3 cases for Comm:
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
                return DaskComm(*args, **kwargs)
        return DaskComm(*args, **kwargs)

    if is_mpi_comm(comm) or isinstance(comm, DaskComm):
        return comm

    raise TypeError("Invalid communicator: expected MPI communicator or None")


class DaskComm(ICommunicator):
    def __init__(self, client: Client, size: int, cart_coord_dims: Optional[Sequence[int]] = None, *args, **kwargs):
        self.client = client
        self.size = size

        # Cartesian topology
        if cart_coord_dims is None:
            # simple fallback: 1D
            cart_coord_dims = (size,)
        self.dims = tuple(cart_coord_dims)

        self._seq = 0
        self._rank = None
        self._actor = _get_actor(client, CommActor, size=size)

        self.Get_rank()

    def Get_rank(self) -> int:
        if self._rank is None:
            cid = str(uuid.uuid4())
            self._rank = self._actor.register(cid).result()
        return self._rank

    def Get_size(self) -> int:
        return self.size

    def Get_coords(self, rank) -> List[int]:
        return self._actor.get_coords(rank, dims=self.dims).result()

    def gather(self, data: Any, root: int = 0) -> Optional[list[Any]]:
        seq = self._seq
        self._seq += 1

        rank = self.Get_rank()

        self._actor.gather_add(seq, rank, data)

        if rank == root:
            while not self._actor.gather_ready(seq).result():
                time.sleep(0.01)

            result = self._actor.gather_get(seq).result()

            result.sort(key=lambda x: x[0])
            return [v for _, v in result]

        return None

    def bcast(self, obj: Any, root: int = 0) -> Any:

        rank = self.Get_rank()

        if rank == root:
            self._actor.bcast_set(obj)
            result = obj
        else:
            while not self._actor.bcast_ready().result():
                time.sleep(0.01)
            result = self._actor.bcast_get().result()

        if rank == root:
            self._actor.cleanup(root)

        return result


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
