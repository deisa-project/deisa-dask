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
