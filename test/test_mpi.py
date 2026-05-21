import argparse
import logging
import os
import shutil
import subprocess
import sys
from multiprocessing.pool import ThreadPool
from typing import Tuple

import pytest

from deisa.core.types import DeisaArray
from utils import wait_for

logging.basicConfig(level=logging.DEBUG)


def mpi_gather_test():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = comm.gather(rank, root=0)

    if rank == 0:
        assert data == list(range(size)), f"Unexpected gather result: {data}"


def mpi_bridge_main(scheduler_address: str, global_size: Tuple, parallelism: Tuple, comm: str):
    from mpi4py import MPI
    import numpy as np
    from deisa.dask import Bridge
    import logging
    logging.basicConfig(level=logging.DEBUG)

    if comm == 'mpi-comm-world':
        bridge_comm = MPI.COMM_WORLD
    elif comm == 'mpi-comm-cart':
        bridge_comm = MPI.COMM_WORLD
        dims = MPI.Compute_dims(bridge_comm.Get_size(), len(global_size))
        bridge_comm = bridge_comm.Create_cart(dims)
    else:
        raise ValueError(f"Invalid comm: {comm}")

    rank = bridge_comm.Get_rank()
    size = bridge_comm.Get_size()

    # global_size = (32, 32)
    # parallelism = (2, 2)

    print(f"global_size={global_size} parallelism={parallelism}", flush=True)

    assert size == np.prod(parallelism), f"comm size={size} should be equal to product of parallelism={parallelism}"

    arrays_metadata = {
        'temperature': {
            'global_shape': global_size,
            'chunk_shape': tuple(g // p for g, p in zip(global_size, parallelism)),
            'chunk_position': (0,) * len(global_size)  # TODO
        }
    }

    print(f"MPI {rank} of {size} started. scheduler_address={scheduler_address}, arrays_metadata={arrays_metadata}",
          flush=True)

    bridge = Bridge(comm=bridge_comm, arrays_metadata=arrays_metadata)

    wait_for(lambda: bridge.get("hello", timestep=1) == "world", timeout=10, interval=1)

    to_send = np.ones(tuple(g // p for g, p in zip(global_size, parallelism)), dtype=np.float64)
    bridge.send('temperature', to_send, iteration=1)

    bridge.close(timestep=1)
    print(f"MPI {rank} of {size} finished", flush=True)


def has_mpirun():
    return shutil.which("mpirun") is not None


def is_xdist():
    import os
    return "PYTEST_XDIST_WORKER" in os.environ


@pytest.mark.skipif(is_xdist(), reason="requires serial execution")
@pytest.mark.skipif(not has_mpirun(), reason="mpirun not available")
@pytest.mark.parametrize('i', [1, 2, 4, 8])
def test_mpi_gather(i):
    cmd = ["mpirun", "-n", str(i), "--oversubscribe", sys.executable, "-u", __file__, "--mpi-gather"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:\n", result.stdout, flush=True)
    print("STDERR:\n", result.stderr, flush=True)

    assert result.returncode == 0, f"MPI test failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.skipif(is_xdist(), reason="requires serial execution")
@pytest.mark.skipif(not has_mpirun(), reason="mpirun not available")
@pytest.mark.parametrize('global_size', [(32, 32), (32, 32, 32)])
@pytest.mark.parametrize('parallelism', [1, 2])  # per dim
@pytest.mark.parametrize('comm', ['mpi-comm-cart', 'mpi-comm-world'])
def test_mpi_bridge(global_size: Tuple, parallelism: int, comm: str):
    from distributed import Client
    import numpy as np

    from distributed import LocalCluster
    from deisa.dask import Deisa

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, host='127.0.0.1', scheduler_port=0,
                           dashboard_address=":0", worker_dashboard_address=":0")
    client = Client(cluster)
    print(f"client={client}", flush=True)

    os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address

    client.wait_for_workers(2, timeout=10)

    def work():
        logging.basicConfig(level=logging.DEBUG)

        deisa = Deisa(timeout=10)

        print(f"deisa={deisa}: deisa.arrays_metadata={deisa.arrays_metadata}", flush=True)

        deisa.set("hello", "world", timestep=1)

        def cb(window):
            print(f"hello from cb. iteration={window[-1].t}", flush=True)
            darr = window[-1]
            assert isinstance(darr, DeisaArray)
            assert darr.dask.sum().compute() == np.prod(
                global_size), f"temperature sum should be the product of {global_size}"

        deisa.register_callback(cb, "temperature")
        deisa.execute_callbacks()
        return 0

    pool = ThreadPool(processes=1)

    def error_callback(e):
        print(f"[ERROR] {e}", flush=True)
        raise e

    async_result = pool.apply_async(work, error_callback=error_callback)

    parallelism = (parallelism,) * len(global_size)

    cmd = ["mpirun", "-n", str(np.prod(parallelism)), "--oversubscribe", sys.executable, "-u", __file__,
           "--mpi-bridge",
           "--scheduler-address", cluster.scheduler.address,
           "--global-size", str(global_size),
           "--parallelism", str(parallelism),
           "--comm", comm
           ]
    print(f"cmd={cmd}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    print(f"result={result}", flush=True)
    print("STDOUT:\n", result.stdout, flush=True)
    print("STDERR:\n", result.stderr, flush=True)

    assert result.returncode == 0, f"MPI test failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert async_result.get(timeout=10) == 0

    cluster.close()


# ENTRY POINT SWITCH
if __name__ == "__main__":
    print(f"sys.argv={sys.argv}", flush=True)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mpi-gather", action="store_true")
    group.add_argument("--mpi-bridge", action="store_true")

    parser.add_argument("--scheduler-address")
    parser.add_argument("--global-size", default="(32, 32)")
    parser.add_argument("--parallelism", default="(2, 2)")
    parser.add_argument("--comm", default="none")

    args = parser.parse_args()

    if args.mpi_bridge and not args.scheduler_address:
        parser.error("--scheduler-address is required when using --mpi-bridge")

    args = parser.parse_args()

    if args.mpi_gather:
        try:
            mpi_gather_test()
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            sys.exit(1)
        sys.exit(0)

    elif args.mpi_bridge:
        try:
            parallelism = eval(args.parallelism)
            global_size = eval(args.global_size)
            print(f"global_size={global_size}, parallelism={parallelism}, comm={args.comm}", flush=True)
            os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = args.scheduler_address
            mpi_bridge_main(scheduler_address=args.scheduler_address,
                            parallelism=parallelism, global_size=global_size,
                            comm=args.comm)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            sys.exit(1)
        sys.exit(0)
