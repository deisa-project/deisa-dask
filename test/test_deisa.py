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
import math
import os.path
import random
import sys
import time

import dask
import dask.array as da
import numpy as np
import pytest
from distributed import Client, LocalCluster, Queue, Variable, Actor, Future

from TestSimulator import TestSimulation
from deisa.dask import Deisa, get_connection_info


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


class TestDeisaCtor:
    @pytest.fixture(scope="class")
    def env_setup_tcp_cluster(self):
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=True, host='127.0.0.1', scheduler_port=4242)
        yield cluster
        cluster.close()

    def test_deisa_ctor_client(self, env_setup_tcp_cluster):
        cluster = env_setup_tcp_cluster
        client = Client(cluster)
        deisa = Deisa(get_connection_info=lambda: client, wait_for_go=False)
        assert deisa.client is not None, "Deisa should not be None"
        assert deisa.client.scheduler.address == cluster.scheduler_address, "Client should be the same as scheduler"
        deisa.close()

    def test_deisa_ctor_str(self, env_setup_tcp_cluster):
        cluster = env_setup_tcp_cluster
        deisa = Deisa(get_connection_info=lambda: get_connection_info('tcp://127.0.0.1:4242'),
                      wait_for_go=False)
        assert deisa.client is not None, "Deisa should not be None"
        assert deisa.client.scheduler.address == cluster.scheduler_address, "Client should be the same as scheduler"
        deisa.close()

    def test_deisa_ctor_scheduler_file(self, env_setup_tcp_cluster):
        cluster = env_setup_tcp_cluster
        f = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'test-scheduler.json'
        deisa = Deisa(get_connection_info=lambda: get_connection_info(f), wait_for_go=False)
        assert deisa.client is not None, "Deisa should not be None"
        assert deisa.client.scheduler.address == cluster.scheduler_address, "Client should be the same as scheduler"
        deisa.close()

    def test_deisa_ctor_scheduler_file_error(self):
        with pytest.raises(ValueError) as e:
            f = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'test-scheduler-error.json'
            Deisa(get_connection_info=lambda: get_connection_info(f), wait_for_go=False)

    def test_dask_actor(self, env_setup_tcp_cluster):
        cluster = env_setup_tcp_cluster

        class MyActor:
            i = 0

            def __init__(self):
                self.i = 0

            def increment(self):
                self.i += 1

        # actor = Actor(MyActor, name="my-actor")
        # actor.increment()
        client1 = Client(cluster)
        future = client1.submit(MyActor, actor=True)
        print(f"future.key={future.key}")
        Variable('MyActorFuture', client=client1).set(future)
        a1 = future.result()
        a1.increment()
        print(f"a1.i={a1.i}")

        client2 = Client(cluster)

        assert client1.cluster == client2.cluster

        actor_future_key = Variable('MyActorFuture', client=client2).get()
        print(f"actor_future_key={actor_future_key}")
        actor_future = Variable('MyActorFuture', client=client2).get()
        a2 = actor_future.result()
        a2.increment()
        print(f"a2.i={a2.i}")

        assert a2.i == 2


class TestUsingDaskCluster:
    @pytest.fixture(scope="function")
    def env_setup(self):
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False)
        client = Client(cluster)
        yield client, cluster
        # teardown
        client.close()
        cluster.close()

    def test_dask_queue(self, env_setup):
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

    def test_dask_variable(self, env_setup):
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

    @staticmethod
    def in_order(original_send_order: list[int]):
        return original_send_order

    @staticmethod
    def reverse_order(original_send_order: list[int]):
        original_send_order.reverse()
        return original_send_order

    @staticmethod
    def random_order(original_send_order: list[int]):
        random.seed(42)
        random.shuffle(original_send_order)
        return original_send_order

    @pytest.mark.parametrize('global_grid_size', [(8, 8), (32, 32), (32, 4), (4, 32)])
    @pytest.mark.parametrize('mpi_parallelism', [(1, 1), (2, 2), (1, 2), (2, 1)])
    @pytest.mark.parametrize('send_order_fn', [in_order, reverse_order, random_order])
    @pytest.mark.parametrize('nb_iterations', [1, 2, 5])
    def test_get_dask_array(self, global_grid_size: tuple, mpi_parallelism: tuple, nb_iterations: int, send_order_fn,
                            env_setup):
        print(f"global_grid_size={global_grid_size} mpi_parallelism={mpi_parallelism} nb_iterations={nb_iterations}")

        client, cluster = env_setup

        sim = TestSimulation(client,
                             global_grid_size=global_grid_size,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'size': global_grid_size,
                                     'subsize': (global_grid_size[0] // mpi_parallelism[0],
                                                 global_grid_size[1] // mpi_parallelism[1])
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa(get_connection_info=lambda: client)

        for i in range(nb_iterations):
            global_data = sim.generate_data('my_array', i, send_order_fn)
            global_data_da = da.from_array(global_data, chunks=(global_grid_size[0] // mpi_parallelism[0],
                                                                global_grid_size[1] // mpi_parallelism[1]))
            darr, iteration = deisa.get_array('my_array')

            assert iteration == i, "iteration does not match expected"
            assert math.isclose(global_data_da.sum().compute(), darr.sum().compute(),
                                rel_tol=1e-09), "reconstructed dask array does not match original"
            assert global_data_da.all() == darr.all(), "reconstructed dask array does not match original"

    @pytest.mark.parametrize('global_grid_size', [(8, 8), (32, 32), (32, 4), (4, 32)])
    @pytest.mark.parametrize('mpi_parallelism', [(1, 1), (2, 2), (1, 2), (2, 1)])
    @pytest.mark.parametrize('nb_iterations', [1, 5])
    @pytest.mark.parametrize('window_size', [1, 2])
    def test_sliding_window_callback_register(self, global_grid_size: tuple, mpi_parallelism: tuple, nb_iterations: int,
                                              window_size: int, env_setup):
        print(f"global_grid_size={global_grid_size} mpi_parallelism={mpi_parallelism} "
              f"nb_iterations={nb_iterations} window_size={window_size}")

        client, cluster = env_setup

        sim = TestSimulation(client,
                             global_grid_size=global_grid_size,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'size': global_grid_size,
                                     'subsize': (global_grid_size[0] // mpi_parallelism[0],
                                                 global_grid_size[1] // mpi_parallelism[1])
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa(get_connection_info=lambda: client)

        context = {
            'counter': 0
        }

        def window_callback(window: list[da.Array], timestep: int):
            print(f"hello from window_callback. iteration={timestep}", flush=True)
            context['counter'] += 1
            context['latest_timestep'] = timestep
            context['latest_data'] = window[-1]
            context['latest_window_size'] = len(window)

        deisa.register_sliding_window_callback("my_array", window_callback, window_size=window_size)

        for i in range(1, nb_iterations + 1):
            print(f"iteration {i}", flush=True)
            # register an already registered callback. This should not do anything.
            deisa.register_sliding_window_callback("my_array", window_callback, window_size=window_size)

            global_data = sim.generate_data('my_array', i)
            global_data_da = da.from_array(global_data, chunks=(global_grid_size[0] // mpi_parallelism[0],
                                                                global_grid_size[1] // mpi_parallelism[1]))

            time.sleep(.1)  # wait for callback to be called
            assert context['counter'] == i, "callback was not called"
            assert context['latest_timestep'] == i, "callback was not called with correct timestep"
            assert type(context['latest_data']) == da.Array, "callback was not called with correct data"
            assert context['latest_data'].any() == global_data_da.any(), "callback was not called with correct data"
            assert context['latest_window_size'] == min(i,
                                                        window_size), "callback was not called with correct window size"

        assert context['counter'] == nb_iterations, f"callback was not called {nb_iterations} times"
        deisa.close()

    def test_sliding_window_callback_unregister(self, env_setup):
        client, cluster = env_setup
        global_grid_size = (8, 8)
        mpi_parallelism = (2, 2)
        window_size = 1

        sim = TestSimulation(client,
                             global_grid_size=global_grid_size,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'size': global_grid_size,
                                     'subsize': (global_grid_size[0] // mpi_parallelism[0],
                                                 global_grid_size[1] // mpi_parallelism[1])
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa(get_connection_info=lambda: client)

        context = {
            'counter': 0
        }

        def window_callback(window: list[da.Array], timestep: int):
            print(f"hello from window_callback. iteration={timestep}", flush=True)
            context['counter'] += 1
            context['latest_timestep'] = timestep
            context['latest_data'] = window[-1]
            context['latest_window_size'] = len(window)

        # register followed by unregister
        deisa.register_sliding_window_callback("my_array", window_callback, window_size=window_size)
        deisa.unregister_sliding_window_callback("my_array")
        sim.generate_data('my_array', iteration=1)
        time.sleep(1)
        assert context['counter'] == 0, "callback should not be called"

        # unregister an unknown array name
        deisa.register_sliding_window_callback("my_array", window_callback, window_size=window_size)
        deisa.unregister_sliding_window_callback("my_unknown_array")
        sim.generate_data('my_array', iteration=2)
        time.sleep(1)
        assert context['counter'] == 2, "callback should be called"
        assert context['latest_timestep'] == 2, "callback should be called"

        deisa.close()

    def test_sliding_window_callback_throws(self, env_setup):
        client, cluster = env_setup
        global_grid_size = (8, 8)
        mpi_parallelism = (2, 2)

        sim = TestSimulation(client,
                             global_grid_size=global_grid_size,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'size': global_grid_size,
                                     'subsize': (global_grid_size[0] // mpi_parallelism[0],
                                                 global_grid_size[1] // mpi_parallelism[1])
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa(get_connection_info=lambda: client)

        context = {
            'counter': 0,
            'exception_handler': 0
        }

        def window_callback(_: list[da.Array], timestep: int):
            print(f"hello from window_callback. iteration={timestep}", flush=True)
            context['counter'] += 1
            raise RuntimeError("Throw from user callback")

        def custom_exception_handler(array_name, exc):
            print(f"hello from custom_exception_handler. {array_name}={exc}", flush=True)
            context['exception_handler'] += 1

        def custom_exception_handler_raise(array_name, exc):
            print(f"hello from custom_exception_handler. {array_name}={exc}", flush=True)
            context['exception_handler'] += 1
            raise RuntimeError("Throw from user exception handler.")

        # default exception_handler
        deisa.register_sliding_window_callback("my_array", window_callback)
        sim.generate_data('my_array', 1)
        time.sleep(1)  # wait for callback to be called
        assert context['counter'] == 1, "callback was not called"
        assert context['exception_handler'] == 0, "callback was not called"

        # custom error handler
        deisa.unregister_sliding_window_callback("my_array")
        deisa.register_sliding_window_callback("my_array", window_callback,
                                               exception_handler=custom_exception_handler)
        sim.generate_data('my_array', 2)
        time.sleep(1)  # wait for callback to be called
        assert context['counter'] == 2, "callback was not called"
        assert context['exception_handler'] == 1, "callback was not called"

        # custom error handler that throws
        deisa.unregister_sliding_window_callback("my_array")
        deisa.register_sliding_window_callback("my_array", window_callback,
                                               exception_handler=custom_exception_handler_raise)
        sim.generate_data('my_array', 3)
        time.sleep(1)  # wait for callback to be called
        assert context['counter'] == 3, "callback was not called"
        assert context['exception_handler'] == 2, "callback was not called"

        # callback unregistered due to un handled exception in custom_exception_handler_raise. Should no longer be called.
        sim.generate_data('my_array', 4)
        time.sleep(1)  # wait for callback to be called
        assert context['counter'] == 3, "callback was not called"
        assert context['exception_handler'] == 2, "callback was not called"

        deisa.close()

    def test_sliding_window_map_blocks(self, env_setup):
        client, cluster = env_setup
        global_grid_size = (8, 8)
        mpi_parallelism = (2, 2)

        sim = TestSimulation(client,
                             global_grid_size=global_grid_size,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'size': global_grid_size,
                                     'subsize': (global_grid_size[0] // mpi_parallelism[0],
                                                 global_grid_size[1] // mpi_parallelism[1])
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa(get_connection_info=lambda: client)

        def map_block_function(block, block_info=None):
            print("map_block_function() block_info=" + str(block_info), flush=True)
            return np.array([[1]])

        context = {'counter': 0}

        def window_callback(window: list[da.Array], timestep: int):
            print(f"hello from window_callback. iteration={timestep}", flush=True)

            darr = window[-1]

            assert darr.shape == global_grid_size
            assert darr.chunksize == (global_grid_size[0] // mpi_parallelism[0],
                                      global_grid_size[1] // mpi_parallelism[1])

            meta = np.array([[0]])
            res = darr.map_blocks(map_block_function, dtype=int, meta=meta).compute()

            context['counter'] += res.sum()

        def exception_handler(array_name, e):
            print(f"exception_handler. array_name={array_name}, e={e}", flush=True, file=sys.stderr)
            # pytest.fail(str(e))   # TODO

        deisa.register_sliding_window_callback("my_array", window_callback,
                                               window_size=1,
                                               exception_handler=exception_handler)

        for i in range(1, 5):
            sim.generate_data('my_array', i)
            time.sleep(.1)  # wait for callback to be called
            assert context['counter'] == 4 * i, "map_blocks did not run on all blocks"

        deisa.close()
