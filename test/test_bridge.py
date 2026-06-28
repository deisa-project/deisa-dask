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
import os

import numpy as np
import pytest
from distributed import Client, LocalCluster

from deisa.dask import Bridge
from utils import FakeComm, FakeCartComm

logging.basicConfig(level=logging.DEBUG)


class TestBridge:
    @pytest.fixture(scope="function")
    def env_setup(self):
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=True,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(1, timeout=10)
        yield client, cluster
        cluster.close()

    def get_new_bridge(self):
        arrays_metadata = {
            'temperature': {
                'global_shape': (1,),
                'chunk_shape': (1,),
                'chunk_position': (0,)
            }}
        comm_state = FakeComm.State(1)
        bridge = Bridge(
            comm=FakeComm(comm_state, 0),
            arrays_metadata=arrays_metadata,
            wait_for_go=False
        )
        return bridge, arrays_metadata

    def test_ctor(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata = self.get_new_bridge()
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])
        assert isinstance(bridge.comm, FakeComm)
        assert not bridge._has_close_been_called

    def test__del__(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata = self.get_new_bridge()
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])
        assert isinstance(bridge.comm, FakeComm)
        assert not bridge._has_close_been_called
        bridge.__del__()
        assert bridge._has_close_been_called

    def test_close(self, env_setup):
        client, _ = env_setup
        bridge, _ = self.get_new_bridge()
        assert not bridge._has_close_been_called
        bridge.close(timestep=42)
        assert bridge._has_close_been_called

    @pytest.mark.flaky(retries=3, delay=1)
    def test_send_update_workers(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])

        cluster.scale(2)
        cluster.wait_for_workers(2)

        bridge.send('temperature', np.ones(1), timestep=0, update_workers=True)

        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])

    @pytest.mark.flaky(retries=3, delay=1)
    def test_send_filter_workers_empty(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            return []

        with pytest.raises(TypeError) as e:
            bridge.send('temperature', np.ones(1), timestep=0, filter_workers=filter)

    def test_send_filter_workers_without_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            assert isinstance(workers, dict)
            for addr in workers.keys():
                assert isinstance(addr, str)
                assert addr in [w.worker_address for w in cluster.workers.values()]
            return list(workers.keys())

        bridge.send('temperature', np.ones(1), timestep=0, update_workers=False, filter_workers=filter)

    def test_send_filter_workers_with_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            assert isinstance(workers, dict)
            return list(workers.keys())

        bridge.send('temperature', np.ones(1), timestep=0, update_workers=True, filter_workers=filter)

    def test_cart_comm(self, env_setup):
        client, cluster = env_setup

        arrays_metadata = {
            'temperature': {
                'global_shape': (8, 8),
                'chunk_shape': (4, 4),
                'chunk_position': (0, 0)
            }}
        comm_state = FakeComm.State(4)

        bridges = [Bridge(comm=FakeCartComm(comm_state, rank, dims=(2, 2)),
                          arrays_metadata=arrays_metadata,
                          wait_for_go=False) for rank in range(4)]

        async def _bridge_send():
            await asyncio.gather(*[asyncio.to_thread(bridge.send, 'temperature',
                                                     np.ones(arrays_metadata['temperature']['chunk_shape']),
                                                     timestep=0)
                                   for i, bridge in enumerate(bridges)])

        asyncio.run(_bridge_send())

        event = client.get_events('temperature')
        assert len(event) == 1
        _, info = event[0]
        assert info['array_name'] == 'temperature'
        assert info['iteration'] == 0
        assert len(info['futures']) == 4
        for f in info['futures']:
            assert f['placement'] in [(0, 0), (0, 1), (1, 0), (1, 1)]

        async def _bridge_close():
            await asyncio.gather(*[asyncio.to_thread(bridge.close, 0) for i, bridge in enumerate(bridges)])

        asyncio.run(_bridge_close())


    @pytest.fixture
    def env_setup_inproc(self):
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=False,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(2, timeout=10)
        yield client, cluster
        cluster.close()

    @pytest.fixture
    def env_setup_remote(self):
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(2, timeout=10)
        yield client, cluster
        cluster.close()

    @pytest.fixture
    def env_setup_mixed(self):
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=True,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(1, timeout=10)

        # One in-process worker connecting to the same scheduler
        from distributed import Worker
        async def _start():
            return await Worker(cluster.scheduler.address, nthreads=1)
        async def _stop(w):
            await w.close()

        inproc_worker = client.sync(_start)
        client.wait_for_workers(2, timeout=10)

        yield client, cluster, inproc_worker

        client.sync(_stop, inproc_worker)
        client.close()
        cluster.close()

    def test_send_uses_inprocess_path(self, env_setup_inproc, caplog):
        client, cluster = env_setup_inproc
        bridge, _ = self.get_new_bridge()

        data = np.ones(1)
        original_buffer_addr = data.__array_interface__['data'][0]
        print(f"original buffer address: {hex(original_buffer_addr)}", flush=True)

        with caplog.at_level(logging.DEBUG, logger='deisa.dask.bridge'):
            bridge.send('temperature', data, timestep=0)

        # Verify routing, only in-process workers from this process perspective
        assert any('in_process=' in r.message and 'remote=[]' in r.message
                for r in caplog.records), \
            "Expected all workers to be in-process"

        stored_buffer_addrs = [
            w.data[key].__array_interface__['data'][0]
            for w in cluster.workers.values()
            for key in w.data
            if 'ndarray-' in key
        ]
        print(f"stored buffer addresses: {[hex(a) for a in stored_buffer_addrs]}", flush=True)

        assert len(stored_buffer_addrs) > 0, \
            "No ndarray key found in any worker's data store"
        assert original_buffer_addr in stored_buffer_addrs, \
            f"Zero-copy failed: original buffer {hex(original_buffer_addr)} " \
            f"not found in stored buffers {[hex(a) for a in stored_buffer_addrs]}"

    def test_send_uses_remote_path(self, env_setup_remote, caplog):
        client, cluster = env_setup_remote
        bridge, _ = self.get_new_bridge()

        data = np.ones(1)
        original_buffer_addr = data.__array_interface__['data'][0]
        print(f"original buffer address: {hex(original_buffer_addr)}", flush=True)

        with caplog.at_level(logging.DEBUG, logger='deisa.dask.bridge'):
            bridge.send('temperature', data, timestep=0)

        # Verify routing, no in-process workers from this process perspective
        assert any('in_process=[]' in r.message
                for r in caplog.records), \
            "Expected no in-process workers for remote cluster"

        # Verify data arrived on a worker via the scheduler
        # Buffer address cannot be checked directly since remote workers
        # are in separate OS processes with independent memory spaces
        who_has = client.who_has()
        ndarray_keys = [k for k in who_has if 'ndarray-' in k]
        assert len(ndarray_keys) > 0, \
            "No ndarray key found on any worker after remote scatter"
        assert all(len(who_has[k]) > 0 for k in ndarray_keys), \
            "Some keys have no owner worker"

    def test_send_uses_mixed_path(self, env_setup_mixed, caplog):
        client, cluster, inproc_worker = env_setup_mixed
        bridge, _ = self.get_new_bridge()

        inproc_addr = inproc_worker.address
        remote_addrs = set(bridge.workers) - {inproc_addr}

        # Verify both worker types are visible to the bridge
        assert inproc_addr in bridge.workers, "In-process worker not in bridge.workers"
        assert len(remote_addrs) > 0, "No remote worker in bridge.workers"

        data = np.ones(1)
        original_buffer_addr = data.__array_interface__['data'][0]
        print(f"original buffer address: {hex(original_buffer_addr)}", flush=True)

        with caplog.at_level(logging.DEBUG, logger='deisa.dask.bridge'):
            bridge.send('temperature', data, timestep=0)

        # Verify routing log detected both worker types
        routing_records = [r.message for r in caplog.records if 'in_process=' in r.message]
        assert len(routing_records) > 0, "No routing log found"
        routing_msg = routing_records[0]
        print(routing_msg)
        assert 'in_process=[]' not in routing_msg, "Expected at least one in-process worker"
        assert 'remote=[]' not in routing_msg, "Expected at least one remote worker"

        # Verify the key landed on exactly one worker
        who_has = client.who_has()
        ndarray_keys = [k for k in who_has if 'ndarray-' in k]
        assert len(ndarray_keys) > 0, "No ndarray key found after scatter"
        all_holders = {addr for k in ndarray_keys for addr in who_has[k]}
        assert len(all_holders) == 1, "Key should be on exactly one worker"

        if inproc_addr in all_holders:
            # In-process worker holds the data, verify zero-copy via buffer address
            in_process_keys = [key for key in inproc_worker.data if 'ndarray-' in key]
            stored_buffer_addrs = [
                inproc_worker.data[key].__array_interface__['data'][0]
                for key in in_process_keys
            ]
            print(f"stored buffer addresses: {[hex(a) for a in stored_buffer_addrs]}", flush=True)
            assert len(stored_buffer_addrs) > 0, "No ndarray key found in in-process worker's data store"
            assert original_buffer_addr in stored_buffer_addrs, \
                f"Zero-copy failed: original buffer {hex(original_buffer_addr)} " \
                f"not found in stored buffers {[hex(a) for a in stored_buffer_addrs]}"
        else:
            # Remote worker holds the data: buffer address cannot be checked
            # across process boundaries, verify placement only
            assert all_holders & remote_addrs, \
                "Key holder is neither in-process nor a known remote worker"
