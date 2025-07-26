# =============================================================================
# Copyright (C) 2015-2023 Commissariat a l'energie atomique et aux energies alternatives (CEA)
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
import math
import random

import dask
import dask.array as da
import numpy as np
import pytest
from distributed import Client, LocalCluster, get_client, Queue, Variable

from deisa.deisa import Deisa, get_bridge_instance


@pytest.mark.parametrize('global_shape', [(32, 32), (32, 16), (16, 32)])
@pytest.mark.parametrize('local_shape', [(16, 16), (2, 2), (8, 1), (8, 1)])
def test_reconstruct_global_dask_array_2d(global_shape, local_shape):
    print(f"global_shape={global_shape} local_shape={local_shape}")

    state = da.random.RandomState(42)
    global_data = state.random(global_shape)

    global_len_x, global_len_y = global_shape
    local_len_x, local_len_y = local_shape

    expected_nb_blocks = (global_len_x // local_len_x
                          * global_len_y // local_len_y)

    # create blocks (i.e. 1 block per mpi rank)
    blocks = []
    for x in range(0, global_len_x, local_len_x):
        for y in range(0, global_len_y, local_len_y):
            block = global_data[x:x + local_len_x, y:y + local_len_y]
            blocks.append(block)

    assert len(blocks) == expected_nb_blocks, "number of blocks does not match expected"

    # tested method
    reconstructed_global_data = Deisa._Deisa__tile_dask_blocks(blocks, global_shape)  # access private staticmethod

    assert reconstructed_global_data.shape == global_data.shape, "reconstructed global data shape does not match original"
    assert reconstructed_global_data.chunksize == (local_len_x,
                                                   local_len_y), "reconstructed global data chunksize does not match original"
    assert reconstructed_global_data.all() == global_data.all(), "reconstructed global data does not match original"


@pytest.mark.parametrize('global_shape', [(32, 32, 32), (32, 32, 16), (32, 16, 32), (16, 32, 32), (128, 64, 16)])
@pytest.mark.parametrize('local_shape', [(16, 16, 16), (8, 8, 1), (8, 1, 8), (1, 8, 8)])
def test_reconstruct_global_dask_array_3d(global_shape, local_shape):
    print(f"global_shape={global_shape} local_shape={local_shape}")

    state = da.random.RandomState(42)
    global_data = state.random(global_shape)

    global_len_x, global_len_y, global_len_z = global_shape
    local_len_x, local_len_y, local_len_z = local_shape

    expected_nb_blocks = (global_len_x // local_len_x
                          * global_len_y // local_len_y
                          * global_len_z // local_len_z)

    # create blocks (i.e. 1 block per mpi rank)
    blocks = []
    for x in range(0, global_len_x, local_len_x):
        for y in range(0, global_len_y, local_len_y):
            for z in range(0, global_len_z, local_len_z):
                block = global_data[x:x + local_len_x, y:y + local_len_y, z:z + local_len_z]
                blocks.append(block)

    assert len(blocks) == expected_nb_blocks, "number of blocks does not match expected"

    # tested method
    reconstructed_global_data = Deisa._Deisa__tile_dask_blocks(blocks, global_shape)  # access private staticmethod

    assert reconstructed_global_data.shape == global_data.shape, "reconstructed global data shape does not match original"
    assert reconstructed_global_data.chunksize == (local_len_x, local_len_y,
                                                   local_len_z), "reconstructed global data chunksize does not match original"
    assert reconstructed_global_data.all() == global_data.all(), "reconstructed global data does not match original"


def test_reconstruct_global_dask_array_none():
    with pytest.raises(ValueError):
        Deisa._Deisa__tile_dask_blocks(None, (2, 2))  # access private staticmethod


def test_reconstruct_global_dask_array_empty():
    with pytest.raises(ValueError):
        Deisa._Deisa__tile_dask_blocks([], (2, 2))  # access private staticmethod


class TestSimulation:
    def __init__(self, scheduler_address: str, global_grid_size: tuple, mpi_parallelism: tuple,
                 arrays_metadata: dict[str, dict]):
        self.client = get_client(scheduler_address)
        self.global_grid_size = global_grid_size
        self.mpi_parallelism = mpi_parallelism
        self.local_grid_size = (global_grid_size[0] // mpi_parallelism[0],
                                global_grid_size[1] // mpi_parallelism[1])

        assert global_grid_size[0] % mpi_parallelism[0] == 0, "cannot compute local grid size for x dimension"
        assert global_grid_size[1] % mpi_parallelism[1] == 0, "cannot compute local grid size for y dimension"

        self.nb_mpi_ranks = mpi_parallelism[0] * mpi_parallelism[1]
        self.bridges = [get_bridge_instance(scheduler_address, self.nb_mpi_ranks, rank, arrays_metadata) for rank in
                        range(self.nb_mpi_ranks)]

    def __gen_data(self, noise_level: int = 0) -> np.array:
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

    def __split_array_equal_chunks(self, arr: np.array) -> list[np.array]:
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

    def generate_data(self, array_name: str, iteration: int, send_order_fn=None) -> np.array:
        global_data = self.__gen_data(noise_level=iteration)
        chunks = self.__split_array_equal_chunks(global_data)

        assert len(chunks) == len(self.bridges)

        if send_order_fn is None:
            for i, bridge in enumerate(self.bridges):
                bridge.publish_data(array_name, chunks[i])
        else:
            send_order = send_order_fn(chunks)
            for i, chunk in enumerate(send_order):
                self.bridges[i].publish_data(array_name, chunk)

        return global_data


@pytest.fixture(scope="module")
def env_setup():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    yield client, cluster
    # teardown
    client.close()
    cluster.close()


def test_dask_queue(env_setup):
    client, cluster = env_setup

    q = Queue("Test", client=client)

    np.random.seed(42)

    datas = []
    for _ in range(1):
        data = np.random.random((2, 2))
        datas.append(data)

        f = client.scatter(data, direct=True)
        to_send = {'shape': data.shape,
                   'dtype': data.dtype,
                   'f': f,
                   'f_key': f.key}
        q.put(to_send)

    # get 1
    res = q.get()

    assert res['shape'] == datas[0].shape
    assert res['dtype'] == datas[0].dtype

    darr = da.from_delayed(dask.delayed(res["f"]), res["shape"], dtype=res["dtype"])
    assert darr.compute().all() == datas[0].all()
    assert darr.sum().compute() == datas[0].sum()


def test_dask_variable(env_setup):
    client, cluster = env_setup

    v = Variable("Test", client=client)

    np.random.seed(42)
    data = np.random.random((2, 2))

    f = client.scatter(data, direct=True)
    v.set({'shape': data.shape,
           'dtype': data.dtype,
           'f': f,
           'f_key': f.key})

    res = v.get()
    assert res['shape'] == data.shape
    assert res['dtype'] == data.dtype

    darr = da.from_delayed(dask.delayed(res["f"]), res["shape"], dtype=res["dtype"])
    assert darr.compute().all() == data.all()
    assert darr.sum().compute() == data.sum()


def in_order(original_send_order: list[int]):
    return original_send_order


def reverse_order(original_send_order: list[int]):
    original_send_order.reverse()
    return original_send_order


def random_order(original_send_order: list[int]):
    random.seed(42)
    random.shuffle(original_send_order)
    return original_send_order


@pytest.mark.parametrize('global_grid_size', [(8, 8), (32, 32), (32, 4), (4, 32)])
@pytest.mark.parametrize('mpi_parallelism', [(1, 1), (2, 2), (1, 2), (2, 1)])
@pytest.mark.parametrize('send_order_fn', [in_order, reverse_order, random_order])
@pytest.mark.parametrize('nb_iterations', [1, 2, 5])
def test_get_dask_array(global_grid_size: tuple, mpi_parallelism: tuple, nb_iterations: int, send_order_fn, env_setup):
    print(f"global_grid_size={global_grid_size} mpi_parallelism={mpi_parallelism} nb_iterations={nb_iterations}")

    client, cluster = env_setup

    scheduler_address = cluster.scheduler_address
    nb_mpi_ranks = mpi_parallelism[0] * mpi_parallelism[1]

    deisa = Deisa(scheduler_address, nb_mpi_ranks, 2)
    sim = TestSimulation(scheduler_address,
                         global_grid_size=global_grid_size,
                         mpi_parallelism=mpi_parallelism,
                         arrays_metadata={
                             'my_array': {
                                 'size': global_grid_size,
                                 'subsize': (global_grid_size[0] // mpi_parallelism[0],
                                             global_grid_size[1] // mpi_parallelism[1])
                             }
                         })

    for i in range(nb_iterations):
        global_data = sim.generate_data('my_array', i, send_order_fn)
        global_data_da = da.from_array(global_data, chunks=(global_grid_size[0] // mpi_parallelism[0],
                                                            global_grid_size[1] // mpi_parallelism[1]))
        darr = deisa.get_array('my_array')

        assert math.isclose(global_data_da.sum().compute(), darr.sum().compute(),
                            rel_tol=1e-09), "reconstructed dask array does not match original"
        assert global_data_da.all() == darr.all(), "reconstructed dask array does not match original"
