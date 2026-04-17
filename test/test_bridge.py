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
        return Bridge(
            id=0,
            arrays_metadata={},
            system_metadata={'connection': client, 'nb_bridges': 1},
            wait_for_go=False
        )

    def test_ctor(self, env_setup):
        client, cluster = env_setup
        bridge = self.get_new_bridge(client)
        assert bridge.id == 0
        assert bridge.arrays_metadata == {}
        assert bridge.system_metadata == {'connection': client, 'nb_bridges': 1}
        assert bridge.workers == [w.address for w in cluster.workers.values()]
        assert isinstance(bridge.comm, DaskComm)
        assert not bridge._has_close_been_called

    def test__del__(self, env_setup):
        client, cluster = env_setup
        bridge = self.get_new_bridge(client)
        assert bridge.id == 0
        assert bridge.arrays_metadata == {}
        assert bridge.system_metadata == {'connection': client, 'nb_bridges': 1}
        assert bridge.workers == [w.address for w in cluster.workers.values()]
        assert isinstance(bridge.comm, DaskComm)
        assert not bridge._has_close_been_called
        bridge.__del__()
        assert bridge._has_close_been_called

    def test_close(self, env_setup):
        client, _ = env_setup
        bridge = self.get_new_bridge(client)
        bridge.close()
        assert bridge._has_close_been_called
