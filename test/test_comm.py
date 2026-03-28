import argparse
import shutil
import subprocess
import sys
from typing import Tuple

import pytest

from deisa.dask.communicator import DaskComm


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
    from deisa.dask import get_connection_info

    if comm == 'mpi':
        comm = MPI.COMM_WORLD
    elif comm == 'dask':
        comm = DaskComm(get_connection_info(scheduler_address), int(np.prod(parallelism)))
    else:
        raise ValueError(f"Invalid comm: {comm}")
    rank = comm.Get_rank()
    size = comm.Get_size()

    # global_size = (32, 32)
    # parallelism = (2, 2)

    print(f"global_size={global_size} parallelism={parallelism}", flush=True)

    assert size == np.prod(parallelism), f"comm size={size} should be equal to product of parallelism={parallelism}"

    arrays_metadata = {
        'temperature': {
            'size': (global_size[0], global_size[1]),
            'subsize': (global_size[0] // parallelism[0],
                        global_size[1] // parallelism[1])
        }
    }

    print(f"MPI {rank} of {size} started. scheduler_address={scheduler_address}", flush=True)

    bridge = Bridge(id=rank,
                    arrays_metadata=arrays_metadata,
                    system_metadata={'connection': get_connection_info(scheduler_address), 'nb_bridges': size},
                    comm=comm,
                    wait_for_go=False)

    to_send = np.ones((global_size[0] // parallelism[0],
                       global_size[1] // parallelism[1]), dtype=np.float64)

    bridge.send('temperature', to_send, iteration=1)

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
@pytest.mark.parametrize('parallelism', [(1, 1), (2, 2)])
@pytest.mark.parametrize('comm', ['mpi', 'dask'])
def test_mpi_bridge(parallelism, comm):
    from distributed import Client
    import numpy as np

    from distributed import LocalCluster

    global_size = (32, 32)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, host='127.0.0.1', scheduler_port=0)
    client = Client(cluster)

    print(f"client={client}", flush=True)

    client.wait_for_workers(2, timeout=10)

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

    # check result
    from deisa.dask import Deisa

    deisa = Deisa(get_connection_info=lambda: client, wait_for_go=False)
    darr, _ = deisa.get_array('temperature', iteration=1)
    assert darr.sum().compute() == np.prod(global_size), f"temperature sum should be the product of {global_size}"

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
    parser.add_argument("--comm", default="dask")

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
            mpi_bridge_main(scheduler_address=args.scheduler_address,
                            parallelism=parallelism, global_size=global_size,
                            comm=args.comm)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            sys.exit(1)
        sys.exit(0)
