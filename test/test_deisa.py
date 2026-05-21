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
import abc
import logging
import os.path
import sys
import time
from collections import deque
from typing import List, Dict, Any

import dask
import dask.array as da
import numpy as np
import pytest
from deisa.core.types import DeisaArray, Window
from distributed import Client, LocalCluster, Queue, Variable

from TestSimulator import TestSimulation
from deisa.dask import Deisa, Bridge
from deisa.dask.deisa import DEFAULT_SLIDING_WINDOW_SIZE
from deisa.dask.utils import build_deisa_array
from utils import wait_for, dask_array_element_wise_equal, FakeComm, async_map, async_close_bridges

logging.basicConfig(level=logging.DEBUG)


@pytest.mark.timeout(10)
@pytest.mark.xdist_group(name="serial")
class TestDeisaCtor:
    @pytest.fixture(scope="class")
    def env_setup_tcp_cluster(self):
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=True,
                               dashboard_address=":0", worker_dashboard_address=":0",
                               host='127.0.0.1', scheduler_port=4242)
        client = Client(cluster)
        client.wait_for_workers(1, timeout=20)
        yield cluster
        cluster.close()

    def test_deisa_ctor_str(self, env_setup_tcp_cluster):
        cluster = env_setup_tcp_cluster
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        deisa = Deisa(wait_for_go=False)
        assert deisa.client is not None, "Deisa should not be None"
        assert deisa.client.scheduler.address == cluster.scheduler_address, "Client should be the same as scheduler"

    def test_deisa_ctor_scheduler_file(self, env_setup_tcp_cluster):
        cluster = env_setup_tcp_cluster
        f = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'test-scheduler.json'
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = f
        deisa = Deisa(wait_for_go=False)
        assert deisa.client is not None, "Deisa should not be None"
        assert deisa.client.scheduler.address == cluster.scheduler_address, "Client should be the same as scheduler"

    @pytest.mark.flaky(retries=3, delay=1)
    def test_deisa_ctor_scheduler_file_error(self):
        with pytest.raises(ValueError) as e:
            f = os.path.abspath(os.path.dirname(__file__)) + os.path.sep + 'test-scheduler-error.json'
            os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = f
            Deisa(wait_for_go=False)


class TestUsingDaskCluster:
    @pytest.fixture(scope="function")
    def env_setup(self):
        self.state: Dict[str, Any] = {"counter": 0}
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(2, timeout=10)
        yield client, cluster
        # teardown
        cluster.close()

    @pytest.fixture(scope="class")
    def env_setup_class(self):
        self.state: Dict[str, Any] = {"counter": 0}
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(1, timeout=10)
        yield client, cluster
        # teardown
        cluster.close()

    @pytest.mark.parametrize('global_shape', [(32, 32), (32, 16), (16, 32)])
    @pytest.mark.parametrize('local_shape', [(16, 16), (2, 2), (8, 1), (8, 1)])
    def test_reconstruct_global_dask_array_2d(self, env_setup_class, global_shape, local_shape):
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
        assert dask_array_element_wise_equal(reconstructed_global_data,
                                             global_data), "reconstructed global data does not match original"

    @pytest.mark.parametrize('global_shape', [(32, 32, 32), (32, 32, 16), (32, 16, 32), (16, 32, 32), (128, 64, 16)])
    @pytest.mark.parametrize('local_shape', [(16, 16, 16), (8, 8, 1), (8, 1, 8), (1, 8, 8)])
    def test_reconstruct_global_dask_array_3d(self, env_setup_class, global_shape, local_shape):
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
        assert dask_array_element_wise_equal(reconstructed_global_data,
                                             global_data), "reconstructed global data does not match original"

    def test_reconstruct_global_dask_array_none(self, env_setup_class):
        with pytest.raises(ValueError):
            Deisa._Deisa__tile_dask_blocks(None, (2, 2))  # access private staticmethod

    def test_reconstruct_global_dask_array_empty(self, env_setup_class):
        with pytest.raises(ValueError):
            Deisa._Deisa__tile_dask_blocks([], (2, 2))  # access private staticmethod

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
        darr = build_deisa_array(darr, 0)
        assert isinstance(darr, DeisaArray)
        assert darr.compute().shape == (2, 2)
        assert darr.compute().all() == data.all()
        assert darr.sum().compute() == data.sum()
        assert darr.t == 0

    class RegisterAndCheck(abc.ABC):
        @abc.abstractmethod
        def register_cb(self, state, deisa, expected_window_size): ...

        @abc.abstractmethod
        def check(self, state, i, expected: Dict): ...

        @staticmethod
        def check_array(array_name, state, i, expected):
            assert array_name in state
            assert state[array_name][-1].t == i, "callback was not called with correct timestep"
            assert dask_array_element_wise_equal(state[array_name][-1], expected[array_name]["global_da"]), \
                "callback was not called with correct data"
            assert len(state[array_name]) == min(i, expected[array_name]["window_size"]) \
                if expected[array_name]["window_size"] is not None else DEFAULT_SLIDING_WINDOW_SIZE, \
                "callback was not called with correct window size"

    class SingleArrayName(RegisterAndCheck):
        def register_cb(self, state, deisa, expected_window_size: dict[str, int | None]):
            def cb(temperature: List[DeisaArray]):
                state["counter"] += 1
                state["temperature"] = temperature

            deisa.register_callback(cb,
                                    Window('temperature', size=expected_window_size['temperature'])
                                    if expected_window_size['temperature'] else 'temperature')

        def check(self, state, i, expected):
            self.check_array("temperature", state, i, expected)

    class TwoArrayName(RegisterAndCheck):
        def register_cb(self, state, deisa, expected_window_size: dict[str, int | None]):
            def cb(temperature: List[DeisaArray], pressure: List[DeisaArray]):
                state["counter"] += 1
                state["temperature"] = temperature
                state["pressure"] = pressure

            deisa.register_callback(cb,
                                    Window('temperature', size=expected_window_size['temperature'])
                                    if expected_window_size['temperature'] else 'temperature',
                                    Window('pressure', size=expected_window_size['pressure'])
                                    if expected_window_size['pressure'] else 'pressure')

        def check(self, state, i, expected):
            self.check_array("temperature", state, i, expected)
            self.check_array("pressure", state, i, expected)

    class SingleArrayNameDecorator(RegisterAndCheck):
        def register_cb(self, state, deisa, expected_window_size: dict[str, int | None]):
            @deisa.register(Window('temperature', size=expected_window_size['temperature'])
                            if expected_window_size['temperature']
                            else 'temperature')
            def cb(temperature: List[DeisaArray]):
                state["counter"] += 1
                state["temperature"] = temperature

        def check(self, state, i, expected):
            self.check_array("temperature", state, i, expected)

    class TwoArrayNameDecorator(RegisterAndCheck):
        def register_cb(self, state, deisa, expected_window_size: dict[str, int | None]):
            @deisa.register(Window('temperature', expected_window_size['temperature'])
                            if expected_window_size['temperature']
                            else "temperature",
                            Window('pressure', expected_window_size['pressure'])
                            if expected_window_size['pressure']
                            else "pressure")
            def cb(temperature: List[DeisaArray], pressure: List[DeisaArray]):
                state["counter"] += 1
                state["temperature"] = temperature
                state["pressure"] = pressure

        def check(self, state, i, expected):
            self.check_array("temperature", state, i, expected)
            self.check_array("pressure", state, i, expected)

    class MapBlocks(RegisterAndCheck):
        def register_cb(self, state, deisa, expected_window_size: dict[str, int | None]):
            def map_block_function(block, block_info=None):
                # print(f"map_block_function() block={block}, block_info={block_info}", flush=True)
                return np.array([[1]])

            @deisa.register(Window('temperature', size=expected_window_size['temperature'])
                            if expected_window_size['temperature'] else 'temperature')
            def cb(temperature: List[DeisaArray]):
                state["counter"] += 1
                state["temperature"] = temperature

                meta = np.array([[0]])
                res = temperature[-1].map_blocks(map_block_function, dtype=int, meta=meta).compute()

                if "map_block" not in state:
                    state["map_block"] = 0

                state['map_block'] += res.sum()

        def check(self, state, i, expected):
            self.check_array("temperature", state, i, expected)
            assert state['map_block'] == i * state["temperature"][-1].npartitions, "map_block function was not called"

    @pytest.mark.flaky(retries=5, delay=1)
    @pytest.mark.parametrize('temperature_global_grid_size', [(8, 8)])
    @pytest.mark.parametrize('temperature_window_size', [None, 1, 3])
    @pytest.mark.parametrize('pressure_global_grid_size', [(8, 8)])
    @pytest.mark.parametrize('pressure_window_size', [None, 1])
    @pytest.mark.parametrize('mpi_parallelism', [(1, 1), (2, 2)])
    @pytest.mark.parametrize('nb_iterations', [1, 5])
    @pytest.mark.parametrize('register_fn', [SingleArrayName(), TwoArrayName(),
                                             SingleArrayNameDecorator(), TwoArrayNameDecorator(),
                                             MapBlocks()])
    def test_register_callback(self, temperature_global_grid_size: tuple,
                               pressure_global_grid_size: tuple,
                               mpi_parallelism: tuple,
                               nb_iterations: int,
                               temperature_window_size: int,
                               pressure_window_size: int,
                               register_fn: RegisterAndCheck,
                               env_setup):
        print(f"temperature_global_grid_size={temperature_global_grid_size}, "
              f"pressure_global_grid_size={pressure_global_grid_size}, "
              f"mpi_parallelism={mpi_parallelism}, "
              f"nb_iterations={nb_iterations}, "
              f"temperature_window_size={temperature_window_size}, "
              f"pressure_window_size={pressure_window_size}")

        client, cluster = env_setup

        sim = TestSimulation(client,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'temperature': {
                                     'global_shape': temperature_global_grid_size,
                                     'chunk_shape': (temperature_global_grid_size[0] // mpi_parallelism[0],
                                                     temperature_global_grid_size[1] // mpi_parallelism[1]),
                                     'chunk_position': (0, 0)  # TODO
                                 },
                                 'pressure': {
                                     'global_shape': pressure_global_grid_size,
                                     'chunk_shape': (pressure_global_grid_size[0] // mpi_parallelism[0],
                                                     pressure_global_grid_size[1] // mpi_parallelism[1]),
                                     'chunk_position': (0, 0)  # TODO
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa()

        time.sleep(.2)  # wait for bridges and deisa to be ready

        register_fn.register_cb(self.state, deisa, {
            'temperature': temperature_window_size,
            'pressure': pressure_window_size,
        })

        for i in range(1, nb_iterations + 1):
            print(f"iteration {i}", flush=True)

            global_temperature, global_pressure = sim.generate_data('temperature', 'pressure', iteration=i)
            global_temperature_da = da.from_array(global_temperature,
                                                  chunks=(temperature_global_grid_size[0] // mpi_parallelism[0],
                                                          temperature_global_grid_size[1] // mpi_parallelism[1]))
            global_pressure_da = da.from_array(global_pressure,
                                               chunks=(pressure_global_grid_size[0] // mpi_parallelism[0],
                                                       pressure_global_grid_size[1] // mpi_parallelism[1]))

            assert wait_for(lambda: self.state['counter'] == i, timeout=10), "callback was not called"

            register_fn.check(self.state, i, {
                "temperature": {"global_da": global_temperature_da,
                                "window_size": temperature_window_size},
                "pressure": {"global_da": global_pressure_da,
                             "window_size": pressure_window_size}
            })

    def test_callback_throws(self, env_setup):
        client, cluster = env_setup
        global_grid_size = (8, 8)
        mpi_parallelism = (2, 2)

        sim = TestSimulation(client,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'global_shape': global_grid_size,
                                     'chunk_shape': (global_grid_size[0] // mpi_parallelism[0],
                                                     global_grid_size[1] // mpi_parallelism[1]),
                                     'chunk_position': (0, 0)  # TODO
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa()

        time.sleep(.2)  # wait for bridges and deisa to be ready

        context = {
            'counter': 0,
            'exception_handler': 0
        }

        def window_callback(window: list[DeisaArray]):
            print(f"hello from window_callback. iteration={window[-1].t}", flush=True)
            context['counter'] += 1
            raise RuntimeError("Throw from user callback")

        def custom_exception_handler(exception):
            print(f"hello from custom_exception_handler. exception={exception}", flush=True)
            context['exception_handler'] += 1

        def custom_exception_handler_raise(exception):
            print(f"hello from custom_exception_handler. exception={exception}", flush=True)
            context['exception_handler'] += 1
            raise RuntimeError("Throw from user exception handler.")

        # default exception_handler
        callback_id = deisa.register_callback(window_callback, 'my_array')
        assert callback_id is not None, "callback was not registered"
        time.sleep(.5)
        sim.generate_data('my_array', iteration=1)
        assert wait_for(lambda: context['counter'] == 1, timeout=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 0), "callback was not called"

        # custom error handler
        deisa.unregister_callback(callback_id)
        callback_id = deisa.register_callback(window_callback, 'my_array',
                                              exception_handler=custom_exception_handler)
        assert callback_id is not None, "callback was not registered"
        time.sleep(.5)
        sim.generate_data('my_array', iteration=2)
        assert wait_for(lambda: context['counter'] == 2, timeout=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 1), "callback was not called"

        # custom error handler that throws
        deisa.unregister_callback(callback_id)
        callback_id = deisa.register_callback(window_callback, 'my_array',
                                              exception_handler=custom_exception_handler_raise)
        assert callback_id is not None, "callback was not registered"
        time.sleep(.5)
        sim.generate_data('my_array', iteration=3)
        assert wait_for(lambda: context['counter'] == 3, timeout=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 2), "callback was not called"

        # callback unregistered due to unhandled exception in custom_exception_handler_raise. Should no longer be called.
        sim.generate_data('my_array', iteration=4)
        assert wait_for(lambda: context['counter'] == 3, nb_checks=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 2), "callback was not called"

        async_close_bridges(sim.bridges, 4)
        deisa.execute_callbacks()

    def test_callback_throws_with_decorator(self, env_setup):
        client, cluster = env_setup
        global_grid_size = (8, 8)
        mpi_parallelism = (2, 2)

        sim = TestSimulation(client,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'global_shape': global_grid_size,
                                     'chunk_shape': (global_grid_size[0] // mpi_parallelism[0],
                                                     global_grid_size[1] // mpi_parallelism[1]),
                                     'chunk_position': (0, 0)  # TODO
                                 }
                             },
                             wait_for_go=False)
        deisa = Deisa()

        time.sleep(.2)  # wait for bridges and deisa to be ready

        context = {
            'counter': 0,
            'exception_handler': 0
        }

        @deisa.register("my_array")
        def window_callback(my_array: list[DeisaArray]):
            print(f"hello from window_callback. iteration={my_array[-1].t}", flush=True)
            context['counter'] += 1
            raise RuntimeError("Throw from user callback")

        def custom_exception_handler(exception):
            print(f"hello from custom_exception_handler. exception={exception}", flush=True)
            context['exception_handler'] += 1

        def custom_exception_handler_raise(exception):
            print(f"hello from custom_exception_handler. exception={exception}", flush=True)
            context['exception_handler'] += 1
            raise RuntimeError("Throw from user exception handler.")

        # default exception_handler (set by decorator)
        time.sleep(.5)
        sim.generate_data('my_array', iteration=1)
        assert wait_for(lambda: context['counter'] == 1, timeout=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 0), "callback was not called"

        # custom error handler
        deisa.unregister_callback(window_callback)
        deisa.register_callback(window_callback, 'my_array',
                                exception_handler=custom_exception_handler)
        # assert window_callback.callback_id is not None, "callback was not registered"
        assert 'my_array' in deisa._callbacks_by_array, "callback was not registered for my_array"
        assert len(deisa._callbacks_by_array['my_array']) == 1, "expected exactly one callback registered"
        time.sleep(.5)
        sim.generate_data('my_array', iteration=2)
        assert wait_for(lambda: context['counter'] == 2, timeout=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 1), "callback was not called"

        # custom error handler that throws
        deisa.unregister_callback(window_callback)
        deisa.register_callback(window_callback, 'my_array',
                                exception_handler=custom_exception_handler_raise)
        # assert window_callback.callback_id is not None, "callback was not registered"
        assert 'my_array' in deisa._callbacks_by_array, "callback was not registered for my_array"
        assert len(deisa._callbacks_by_array['my_array']) == 1, "expected exactly one callback registered"
        time.sleep(.5)
        sim.generate_data('my_array', iteration=3)
        assert wait_for(lambda: context['counter'] == 3, timeout=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 2), "callback was not called"

        # callback unregistered due to unhandled exception in custom_exception_handler_raise. Should no longer be called.
        sim.generate_data('my_array', iteration=4)
        assert wait_for(lambda: context['counter'] == 3, nb_checks=10), "callback was not called"
        assert wait_for(lambda: context['exception_handler'] == 2), "callback was not called"

        async_close_bridges(sim.bridges, 4)
        deisa.execute_callbacks()

    @pytest.mark.parametrize('nb_bridges', [1, 4])
    def test_set_get(self, env_setup, nb_bridges: int):
        client, _ = env_setup

        comm_state = FakeComm.State(nb_bridges)

        bridges = [Bridge(comm=FakeComm(comm_state, rank),
                          arrays_metadata={
                              'my_array': {
                                  'global_shape': (0,),
                                  'chunk_shape': (0,),
                                  'chunk_position': (rank,)  # TODO
                              }
                          },
                          wait_for_go=False) for rank in range(nb_bridges)]

        deisa = Deisa()

        for ts in range(5):
            deisa.set('hello', 'world', timestep=ts)

            res = async_map(bridges, Bridge.get, 'hello', timestep=ts)
            assert wait_for(lambda: len(res) == nb_bridges)
            assert all(r == 'world' for r in res)

            # a second call to get should return the same result
            res = async_map(bridges, Bridge.get, 'hello', timestep=ts)
            assert wait_for(lambda: len(res) == nb_bridges)
            assert all(r == 'world' for r in res)

            # without timestep, should return the full queue
            res = async_map(bridges, Bridge.get, 'hello')
            assert wait_for(lambda: len(res) == nb_bridges)
            assert all(r == deque([(i, 'world') for i in range(ts + 1)]) for r in res)

        async_close_bridges(bridges, 4)

    def test_set_from_sliding_window(self, env_setup):
        client, _ = env_setup
        global_grid_size = (8, 8)
        mpi_parallelism = (1, 1)

        sim = TestSimulation(client,
                             mpi_parallelism=mpi_parallelism,
                             arrays_metadata={
                                 'my_array': {
                                     'global_shape': global_grid_size,
                                     'chunk_shape': (global_grid_size[0] // mpi_parallelism[0],
                                                     global_grid_size[1] // mpi_parallelism[1]),
                                     'chunk_position': (0, 0)  # TODO
                                 }
                             },
                             wait_for_go=False)

        deisa = Deisa()

        time.sleep(.2)

        context = {
            'counter': 0
        }

        def window_callback(window: list[DeisaArray]):
            print(f"hello from window_callback. iteration={window[-1].t}", flush=True)
            context['counter'] += 1
            deisa.set('hello', 'world', timestep=window[-1].t)

        deisa.register_callback(window_callback, Window('my_array', size=1))
        sim.generate_data('my_array', iteration=1)
        assert wait_for(lambda: context['counter'] == 1)
        assert wait_for(lambda: sim.bridges[0].get('hello', timestep=1) == 'world')

        async_close_bridges(sim.bridges, 1)

    def test_deisa_array_ctor(self, env_setup):
        dask_arr = da.from_array(np.ones(1))
        deisa_array = build_deisa_array(dask_arr, 0)
        assert dask_array_element_wise_equal(dask_arr, deisa_array)
