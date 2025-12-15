import time
from multiprocessing import Process

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


def start_deisa_handshake(address: str):
    client = get_connection_info(address)
    handshake = Handshake('deisa', client)
    assert handshake.get_nb_bridges() == 1
    assert handshake.get_arrays_metadata() == {'hello': 'world'}


def start_bridge_handshake(address: str):
    client = get_connection_info(address)
    handshake = Handshake('bridge', client, id=0, max=1, arrays_metadata={'hello': 'world'})


def test_handshake_deisa_first(env_setup):
    client, cluster = env_setup
    addr = cluster.scheduler.address
    print(f"cluster={cluster}, addr={addr}", flush=True)

    processes = []

    p = Process(target=start_deisa_handshake, args=(addr,))
    processes.append(p)
    p.start()

    time.sleep(1)

    p = Process(target=start_bridge_handshake, args=(addr,))
    processes.append(p)
    p.start()

    for p in processes:
        p.join()


def test_handshake_bridge_first(env_setup):
    client, cluster = env_setup
    addr = cluster.scheduler.address
    print(f"cluster={cluster}, addr={addr}", flush=True)

    processes = []

    p = Process(target=start_bridge_handshake, args=(addr,))
    processes.append(p)
    p.start()

    time.sleep(1)

    p = Process(target=start_deisa_handshake, args=(addr,))
    processes.append(p)
    p.start()

    for p in processes:
        p.join()
