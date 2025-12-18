import time
from multiprocessing import Process
from typing import List

import pytest
from distributed import LocalCluster, Client

from deisa.dask import get_connection_info
from deisa.dask.handshake import Handshake


@pytest.fixture(scope="function")
def env_setup():
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None)
    client = Client(cluster)
    yield client, cluster
    # teardown
    client.close()
    cluster.close()


def start_deisa_handshake(address: str, nb_bridge: int):
    client = get_connection_info(address)
    handshake = Handshake('deisa', client)
    assert handshake.get_nb_bridges() == nb_bridge
    assert handshake.get_arrays_metadata() == {'hello': 'world'}


def start_bridge_handshake(address: str, id: int, max: int):
    client = get_connection_info(address)
    handshake = Handshake('bridge', client, id=id, max=max, arrays_metadata={'hello': 'world'})


@pytest.mark.parametrize('nb_bridge', [1, 4, 64])
def test_handshake_deisa_first(env_setup, nb_bridge: int):
    client, cluster = env_setup
    addr = cluster.scheduler.address
    print(f"cluster={cluster}, addr={addr}, nb_bridge={nb_bridge}", flush=True)

    processes: List[Process] = []

    p = Process(target=start_deisa_handshake, args=(addr, nb_bridge))
    processes.append(p)
    p.start()

    time.sleep(1)

    for i in range(nb_bridge):
        p = Process(target=start_bridge_handshake, args=(addr, i, nb_bridge))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0


@pytest.mark.parametrize('nb_bridge', [1, 4, 64])
def test_handshake_bridge_first(env_setup, nb_bridge: int):
    client, cluster = env_setup
    addr = cluster.scheduler.address
    print(f"cluster={cluster}, addr={addr}", flush=True)

    processes: List[Process] = []

    for i in range(nb_bridge):
        p = Process(target=start_bridge_handshake, args=(addr, i, nb_bridge))
        processes.append(p)
        p.start()

    time.sleep(1)

    p = Process(target=start_deisa_handshake, args=(addr, nb_bridge))
    processes.append(p)
    p.start()

    for p in processes:
        p.join()
        assert p.exitcode == 0
