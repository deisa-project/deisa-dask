import gc
import time
from multiprocessing import Process, Pool
from typing import List

import pytest
from distributed import LocalCluster

from deisa.dask import get_connection_info
from deisa.dask.handshake import Handshake


# logging.basicConfig(level=logging.DEBUG)


@pytest.mark.xdist_group(name="serial")
@pytest.mark.timeout(60)
class TestHandshake:
    @pytest.fixture(scope="function")
    def env_setup(self):
        cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address='localhost:8787')
        cluster.wait_for_workers(2, timeout=10)
        yield cluster
        # teardown
        cluster.close()

    @staticmethod
    def start_deisa_handshake(address: str, nb_bridge: int):

        for _ in range(5):
            try:
                client = get_connection_info(address)
                break
            except Exception as e:
                print(f"\n\n +++++ Failed get_connection_info. Retrying", flush=True)
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Deisa {id} failed to connect")

        handshake = Handshake('deisa', client)
        assert handshake.get_nb_bridges() == nb_bridge
        assert handshake.get_arrays_metadata() == {'hello': 'world'}

    @staticmethod
    def start_bridge_handshake(address: str, id: int, max: int):

        for _ in range(5):
            try:
                client = get_connection_info(address)
                break
            except Exception as e:
                print(f"\n\n +++++ Failed get_connection_info. Retrying", flush=True)
                time.sleep(0.1)
        else:
            raise RuntimeError(f"Bridge {id} failed to connect")

        handshake = Handshake('bridge', client, id=id, max=max, arrays_metadata={'hello': 'world'})

    @staticmethod
    def start_processes(processes: List[Process]):
        for p in processes:
            p.start()

    @staticmethod
    def join_processes(processes: List[Process]):
        for p in processes:
            p.join()
            assert p.exitcode == 0, "process exited with error"

    @pytest.mark.parametrize('nb_bridge', [1, 4, 64])
    def test_handshake_deisa_first(self, env_setup, nb_bridge: int):
        cluster = env_setup
        addr = cluster.scheduler.address
        print(f"cluster={cluster}, addr={addr}, nb_bridge={nb_bridge}", flush=True)

        processes: List[Process] = [Process(target=TestHandshake.start_deisa_handshake, args=(addr, nb_bridge))]

        for i in range(nb_bridge):
            processes.append(Process(target=TestHandshake.start_bridge_handshake, args=(addr, i, nb_bridge)))

        TestHandshake.start_processes(processes)
        TestHandshake.join_processes(processes)

    @pytest.mark.parametrize('nb_bridge', [1, 4, 64])
    def test_handshake_bridge_first(self, env_setup, nb_bridge: int):
        cluster = env_setup
        addr = cluster.scheduler.address
        print(f"cluster={cluster}, addr={addr}", flush=True)

        processes: List[Process] = []

        for i in range(nb_bridge):
            processes.append(Process(target=TestHandshake.start_bridge_handshake, args=(addr, i, nb_bridge)))

        processes.append(Process(target=TestHandshake.start_deisa_handshake, args=(addr, nb_bridge)))

        TestHandshake.start_processes(processes)
        TestHandshake.join_processes(processes)

    @pytest.mark.parametrize('nb_bridge', [128])
    def test_handshake_bridge_deisa_bridge(self, env_setup, nb_bridge: int):
        cluster = env_setup
        addr = cluster.scheduler.address
        print(f"cluster={cluster}, addr={addr}", flush=True)

        processes: List[Process] = []

        for i in range(nb_bridge // 2):
            processes.append(Process(target=TestHandshake.start_bridge_handshake, args=(addr, i, nb_bridge)))

        processes.append(Process(target=TestHandshake.start_deisa_handshake, args=(addr, nb_bridge)))

        for i in range(nb_bridge // 2, nb_bridge):
            processes.append(Process(target=TestHandshake.start_bridge_handshake, args=(addr, i, nb_bridge)))

        TestHandshake.start_processes(processes)
        TestHandshake.join_processes(processes)
