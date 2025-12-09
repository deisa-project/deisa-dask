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

import numpy as np
from distributed import Client

from deisa.dask import Bridge


class TestSimulation:
    __test__ = False

    def __init__(self, client: Client, global_grid_size: tuple, mpi_parallelism: tuple,
                 arrays_metadata: dict[str, dict], *args, **kwargs):
        self.client = client
        self.global_grid_size = global_grid_size
        self.mpi_parallelism = mpi_parallelism
        self.local_grid_size = (global_grid_size[0] // mpi_parallelism[0],
                                global_grid_size[1] // mpi_parallelism[1])

        assert global_grid_size[0] % mpi_parallelism[0] == 0, "cannot compute local grid size for x dimension"
        assert global_grid_size[1] % mpi_parallelism[1] == 0, "cannot compute local grid size for y dimension"

        self.nb_mpi_ranks = mpi_parallelism[0] * mpi_parallelism[1]
        self.bridges: list[Bridge] = [
            Bridge(id=rank, arrays_metadata=arrays_metadata,
                   system_metadata={'connection': client, 'nb_bridges': self.nb_mpi_ranks}, *args, **kwargs)
            for rank in range(self.nb_mpi_ranks)]

    def __gen_data(self, noise_level: int = 0) -> np.ndarray:
        # Create coordinate grid
        x = np.linspace(-1, 1, self.global_grid_size[0])
        y = np.linspace(-1, 1, self.global_grid_size[1])
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Generate 2D Gaussian (bell curve)
        sigma = 0.5
        global_data_np = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

        # Add Gaussian noise if requested
        if noise_level > 0:
            noise = np.random.normal(loc=0.0, scale=noise_level, size=global_data_np.shape)
            global_data_np += noise

        # global_data_da = da.from_array(global_data_np)
        return global_data_np

    def __split_array_equal_chunks(self, arr: np.ndarray) -> list[np.ndarray]:
        if arr.ndim != 2:
            raise ValueError("Input must be a 2D array")

        rows, cols = arr.shape
        block_rows, block_cols = rows // self.mpi_parallelism[0], cols // self.mpi_parallelism[1]

        if rows % block_rows != 0 or cols % block_cols != 0:
            raise ValueError(f"Array shape {arr.shape} not divisible by block size ({block_rows}, {block_cols})")

        blocks = []
        for i in range(0, rows, block_rows):
            for j in range(0, cols, block_cols):
                block = arr[i:i + block_rows, j:j + block_cols]
                blocks.append(block)

        return blocks

    def generate_data(self, array_name: str, iteration: int, send_order_fn=None) -> np.ndarray:
        global_data = self.__gen_data(noise_level=iteration)
        chunks = self.__split_array_equal_chunks(global_data)

        assert len(chunks) == len(self.bridges)

        if send_order_fn is None:
            for i, bridge in enumerate(self.bridges):
                bridge.send(array_name, chunks[i], iteration)
        else:
            send_order = send_order_fn(chunks)
            for i, chunk in enumerate(send_order):
                self.bridges[i].send(array_name, chunk, iteration)

        return global_data
