import argparse
import logging
import shutil
import subprocess
import sys
import time
from typing import Tuple

import pytest

from deisa.dask.types import DeisaArray
from deisa.dask.communicator import CommClient, resolve_comm

logging.basicConfig(level=logging.DEBUG)


def mpi_gather_test():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = comm.gather(rank, root=0)

    if rank == 0:
        assert data == list(range(size)), f"Unexpected gather result: {data}"


def dask_comm_main(scheduler_address: str, comm_size: int):
    import os
    import time
    from deisa.dask import get_connection_info
    from deisa.dask.communicator import setup_comm
    from distributed import rpc

    rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
    assert rank, "rank cannot be None"
    rank = int(rank)
    client = get_connection_info(scheduler_address)

    print(f">> rank={rank}, client={client}", flush=True)

    if rank == 0:
        client.run_on_scheduler(setup_comm, size=comm_size)
    else:
        time.sleep(.5)

    bridge_comm = CommClient(client=client,
                             comm_state_rpc=client.scheduler if rank == 0 else rpc(scheduler_address))

    #############
    ##  Get_rank
    #############
    r = bridge_comm.Get_rank()
    print(f"rank={rank}, r={r}")
    assert r is not None

    #############
    ##  Get_size
    #############
    size = bridge_comm.Get_size()
    assert size is not None
    assert size == comm_size

    #############
    ##  GATHER
    #############
    ranks = bridge_comm.gather(rank, root=0)
    print(f"ranks={ranks}")
    if ranks:
        # rank 0
        assert len(ranks) == comm_size
        assert sorted(ranks) == list(range(comm_size))

    #############
    ##  BCAST
    #############
    data = None
    if rank == 0:
        data = "hello, world !"
    data = bridge_comm.bcast(data, root=0)
    assert data == "hello, world !"


def mpi_bridge_main(scheduler_address: str, global_size: Tuple, parallelism: Tuple, comm: str):
    from mpi4py import MPI
    import numpy as np
    import os
    from deisa.dask import Bridge, get_connection_info
    from distributed import rpc

    rank = os.environ.get("OMPI_COMM_WORLD_RANK", None)
    assert rank, "rank cannot be None"
    rank = int(rank)
    client = get_connection_info(scheduler_address)

    print(f">> comm={comm}, rank={rank}, client={client}")

    if comm == 'none':
        bridge_comm = resolve_comm(None,
                                   use_mpi_if_available=True,
                                   client=client,
                                   comm_state_rpc=client.scheduler if rank == 0 else rpc(scheduler_address),
                                   size=int(np.prod(parallelism)))
    elif comm == 'mpi-comm-world':
        bridge_comm = MPI.COMM_WORLD
    elif comm == 'mpi-comm-cart':
        bridge_comm = MPI.COMM_WORLD
        dims = MPI.Compute_dims(bridge_comm.Get_size(), len(global_size))
        bridge_comm = bridge_comm.Create_cart(dims)
    elif comm == 'dask':
        bridge_comm = CommClient(client=client,
                                 comm_state_rpc=client.scheduler if rank == 0 else rpc(scheduler_address),
                                 size=int(np.prod(parallelism)))
    elif comm == 'dask-cart':
        bridge_comm = CommClient(client=client,
                                 comm_state_rpc=client.scheduler if rank == 0 else rpc(scheduler_address),
                                 size=int(np.prod(parallelism)),
                                 dims=parallelism)
    else:
        raise ValueError(f"Invalid comm: {comm}")

    print(f">>> bridge_comm={bridge_comm}", flush=True)

    subsize = tuple(g // p for g, p in zip(global_size, parallelism))
    arrays_metadata = {
        "temperature": {
            "size": global_size,
            "subsize": subsize
        }
    }

    bridge = Bridge(id=rank,
                    arrays_metadata=arrays_metadata,
                    system_metadata={'connection': client, 'nb_bridges': int(np.prod(parallelism))},
                    comm=bridge_comm,
                    wait_for_go=False)

    time.sleep(1)

    to_send = np.ones(subsize, dtype=np.float64)
    bridge.send('temperature', to_send, iteration=1)

    print(f"MPI {rank} finished", flush=True)


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
@pytest.mark.parametrize('comm_size', [1, 4, 16])  # per dim
def test_dask_comm(comm_size):
    from distributed import Client
    from distributed import LocalCluster

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, host='127.0.0.1', scheduler_port=0)
    client = Client(cluster)
    client.wait_for_workers(2, timeout=10)

    cmd = ["mpirun", "-n", str(comm_size), "--oversubscribe", sys.executable, "-u", __file__,
           "--dask-comm",
           "--scheduler-address", cluster.scheduler.address,
           "--comm-size", str(comm_size)
           ]
    print(f"cmd={cmd}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    print(f"result={result}", flush=True)
    print("STDOUT:\n", result.stdout, flush=True)
    print("STDERR:\n", result.stderr, flush=True)

    assert result.returncode == 0, f"MPI test failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.skipif(is_xdist(), reason="requires serial execution")
@pytest.mark.skipif(not has_mpirun(), reason="mpirun not available")
@pytest.mark.parametrize('global_size', [(32, 32), (32, 32, 32)])
@pytest.mark.parametrize('parallelism', [1, 2])  # per dim
@pytest.mark.parametrize('comm', ['none', 'mpi-comm-cart', 'mpi-comm-world', 'dask', 'dask-cart'])
def test_mpi_bridge(global_size: Tuple, parallelism: int, comm: str):
    from distributed import LocalCluster, Client
    import numpy as np

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True, host='127.0.0.1', scheduler_port=0)
    client = Client(cluster)
    client.wait_for_workers(2, timeout=10)

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

    # check result
    from deisa.dask import Deisa

    deisa = Deisa(get_connection_info=lambda: client, wait_for_go=False)
    darr = deisa.get_array('temperature', iteration=1)
    assert isinstance(darr, DeisaArray)
    assert darr.dask.sum().compute() == np.prod(global_size), f"temperature sum should be the product of {global_size}"

    cluster.close()


# ENTRY POINT SWITCH
if __name__ == "__main__":
    print(f"sys.argv={sys.argv}", flush=True)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mpi-gather", action="store_true")
    group.add_argument("--dask-comm", action="store_true")
    group.add_argument("--mpi-bridge", action="store_true")

    parser.add_argument("--scheduler-address")
    parser.add_argument("--global-size", default="(32, 32)")
    parser.add_argument("--parallelism", default="(2, 2)")
    parser.add_argument("--comm-size", default="1")
    parser.add_argument("--comm", default="none")

    args = parser.parse_args()

    if args.dask_comm and not args.scheduler_address:
        parser.error("--scheduler-address is required when using --dask-comm")

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

    if args.dask_comm:
        try:
            comm_size = eval(args.comm_size)
            dask_comm_main(scheduler_address=args.scheduler_address, comm_size=comm_size)
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
            print(f"[ERROR] e={e}", flush=True)
            sys.exit(1)
        sys.exit(0)
