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
import logging

import numpy as np
import pytest
from distributed import Client, LocalCluster

from deisa.dask import Bridge
from deisa.dask.communicator import DaskComm

logging.basicConfig(level=logging.DEBUG)


class TestBridge:
    @pytest.fixture(scope="function")
    def env_setup(self):
        cluster = LocalCluster(n_workers=1, threads_per_worker=1,
                               processes=False, dashboard_address=None)
        client = Client(cluster)
        client.wait_for_workers(1, timeout=10)
        yield client, cluster
        client.close()
        cluster.close()

    def get_new_bridge(self, client):
        arrays_metadata = {
            'temperature': {
                'size': (1,),
                'subsize': (1,)
            }}
        system_metadata = {'connection': client, 'nb_bridges': 1}
        bridge = Bridge(
            id=0,
            arrays_metadata=arrays_metadata,
            system_metadata=system_metadata,
            wait_for_go=False
        )
        return bridge, arrays_metadata, system_metadata

    def test_ctor(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata, system_metadata = self.get_new_bridge(client)
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.system_metadata == system_metadata
        assert bridge.workers == [w.address for w in cluster.workers.values()]
        assert isinstance(bridge.comm, DaskComm)
        assert not bridge._has_close_been_called

    def test__del__(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata, system_metadata = self.get_new_bridge(client)
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.system_metadata == system_metadata
        assert bridge.workers == [w.address for w in cluster.workers.values()]
        assert isinstance(bridge.comm, DaskComm)
        assert not bridge._has_close_been_called
        bridge.__del__()
        assert bridge._has_close_been_called

    def test_close(self, env_setup):
        client, _ = env_setup
        bridge, _, _ = self.get_new_bridge(client)
        assert not bridge._has_close_been_called
        bridge.close()
        assert bridge._has_close_been_called

    def test_send_update_workers(self, env_setup):
        client, cluster = env_setup
        bridge, _, _ = self.get_new_bridge(client)

        assert bridge.workers == [w.address for w in cluster.workers.values()]
        assert len(bridge.workers) == 1

        cluster.scale(2)
        cluster.wait_for_workers(2)

        bridge.send('temperature', np.ones(1), iteration=0, update_workers=True)

        assert bridge.workers == [w.address for w in cluster.workers.values()]
        assert len(bridge.workers) == 2

    def test_send_filter_workers_empty(self, env_setup):
        client, cluster = env_setup
        bridge, _, _ = self.get_new_bridge(client)

        def filter(workers):
            return []

        with pytest.raises(TypeError) as e:
            bridge.send('temperature', np.ones(1), iteration=0, filter_workers=filter)

    def test_send_filter_workers_without_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _, _ = self.get_new_bridge(client)

        def filter(workers):
            assert isinstance(workers, list)
            for addr in workers:
                assert isinstance(addr, str)
                assert addr in [w.address for w in cluster.workers.values()]
            return workers

        bridge.send('temperature', np.ones(1), iteration=0, update_workers=False, filter_workers=filter)

    def test_send_filter_workers_with_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _, _ = self.get_new_bridge(client)

        def filter(workers):
            assert isinstance(workers, dict)
            return list(workers.keys())

        bridge.send('temperature', np.ones(1), iteration=0, update_workers=True, filter_workers=filter)
