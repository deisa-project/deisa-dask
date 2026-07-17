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
import itertools
import logging
import sys
from typing import Tuple, Any

import numpy as np
from distributed import Client
from numpy import dtype, float64, ndarray

from deisa.dask import Bridge
from utils import FakeComm, async_map, async_close_bridges, FakeCartComm

logger = logging.getLogger(__name__)


class TestSimulation:
    __test__ = False

    def __init__(self, client: Client, arrays_metadata: dict[str, dict], mpi_parallelism: tuple, *args, **kwargs):
        self.client = client
        self.arrays_metadata = arrays_metadata
        self.mpi_parallelism = mpi_parallelism
        nb_mpi_ranks = int(np.prod(mpi_parallelism))
        comm_state = FakeComm.State(nb_mpi_ranks)

        def _make_bridge(rank):
            comm = FakeCartComm(comm_state, rank, dims=mpi_parallelism)
            return Bridge(
                *args,
                comm=comm,
                arrays_metadata={
                    name: {
                        **metadata,
                        "chunk_position": comm.Get_coords(rank),
                    }
                    for name, metadata in arrays_metadata.items()
                },
                **kwargs,
            )

        self.bridges = async_map(range(nb_mpi_ranks), _make_bridge)

    def __del__(self):
        try:
            async_close_bridges(self.bridges, timestep=sys.maxsize)
        except Exception as e:
            logger.error(f"Error while closing bridges: {e}")

    def __gen_data(self, array_name: str, noise_level: int = 0) -> float | ndarray[tuple[Any, ...], dtype[float64]]:
        # Create coordinate grid
        shape = self.arrays_metadata[array_name]['global_shape']
        return np.random.random(shape)

    def __split_array_equal_chunks(self, arr: np.ndarray) -> list[np.ndarray]:
        if arr.ndim != len(self.mpi_parallelism):
            raise ValueError(
                f"Array has {arr.ndim} dimensions but mpi_parallelism has "
                f"{len(self.mpi_parallelism)} entries"
            )

        chunk_shape = []
        for size, n_chunks in zip(arr.shape, self.mpi_parallelism):
            if size % n_chunks != 0:
                raise ValueError(
                    f"Array shape {arr.shape} is not divisible by "
                    f"mpi_parallelism {self.mpi_parallelism}"
                )
            chunk_shape.append(size // n_chunks)

        blocks = []

        for chunk_idx in itertools.product(*(range(n_chunks) for n_chunks in self.mpi_parallelism)):
            slices = tuple(
                slice(i * c, (i + 1) * c)
                for i, c in zip(chunk_idx, chunk_shape)
            )
            blocks.append(arr[slices])

        return blocks

    def generate_data(self, *array_names: str, iteration: int, update_workers: bool = False) \
            -> np.ndarray | Tuple[np.ndarray]:
        global_datas = []
        for array_name in array_names:
            global_data = self.__gen_data(array_name, noise_level=iteration)
            global_datas.append(global_data)
            chunks = self.__split_array_equal_chunks(global_data)

            assert len(chunks) == len(self.bridges), \
                f"There should be as many chunks as bridges ({len(chunks)} != {len(self.bridges)})."

            async def _bridge_send():
                await asyncio.gather(*[asyncio.to_thread(bridge.send, array_name, chunks[i], iteration,
                                                         update_workers=update_workers)
                                       for i, bridge in enumerate(self.bridges)])

            asyncio.run(_bridge_send())

        assert len(global_datas) == len(array_names)
        if len(global_datas) == 1:
            return global_datas[0]
        else:
            return tuple(global_datas)
