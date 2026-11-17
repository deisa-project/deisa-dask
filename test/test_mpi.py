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


def mpi_bridge_main(scheduler_address: str, global_size: Tuple, parallelism: Tuple, comm: str):
    import logging

    import numpy as np
    from mpi4py import MPI

    from deisa.dask import Bridge

    logging.basicConfig(level=logging.DEBUG)

    if comm == "mpi-comm-world":
        bridge_comm = MPI.COMM_WORLD
    elif comm == "mpi-comm-cart":
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
        "temperature": {
            "global_shape": global_size,
            "chunk_shape": tuple(g // p for g, p in zip(global_size, parallelism)),
            "chunk_position": (0,) * len(global_size),
        }
    }

    if rank == 1:
        arrays_metadata["density"] = {"global_shape": (8, 8), "chunk_shape": (8, 8), "chunk_position": (0, 0)}

    print(
        f"MPI {rank} of {size} started. scheduler_address={scheduler_address}, arrays_metadata={arrays_metadata}",
        flush=True,
    )

    bridge = Bridge(comm=bridge_comm, arrays_metadata=arrays_metadata)

    wait_for(lambda: bridge.get("hello", timestep=1) == "world", timeout=10, interval=1)

    to_send = np.ones(tuple(g // p for g, p in zip(global_size, parallelism)), dtype=np.float64)
    bridge.send("temperature", to_send, timestep=1)

    if rank == 1:
        bridge.send("density", np.ones((8, 8), dtype=np.float64), timestep=1)

    bridge.close(timestep=1)
    print(f"MPI {rank} of {size} finished", flush=True)


# Scenario of ``test_mpi_bridge_prefetch_all_ranks``: two ranks, two arrays shared by every rank, sent in opposite
# orders. Kept at module level so the mpirun entry point and the pytest side (analytics expectations) agree.
PREFETCH_ARRAYS = ("alpha", "beta")
PREFETCH_GLOBAL_SHAPE = (4,)
PREFETCH_CHUNK_SHAPE = (2,)


def mpi_prefetch_main(scheduler_address: str):
    """Two ranks sharing two arrays, sending them in OPPOSITE orders.

    ``Bridge.__init__`` prefetches task branches with one bcast per array on that array's sub-communicator, so the
    prefetch MUST run on every rank. With the 0.7.0 rank-0-only gate, rank 0 prefetched in sorted order ("alpha",
    "beta") while the other rank entered the sub-communicator of the array it sent FIRST -> both ranks block in a
    bcast on different communicators and the run never finishes.
    """
    import logging

    import numpy as np
    from mpi4py import MPI

    from deisa.dask import Bridge

    logging.basicConfig(level=logging.DEBUG)

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    assert size == 2, f"this scenario needs exactly 2 ranks, got {size}"

    array_names = PREFETCH_ARRAYS
    chunk_shape = PREFETCH_CHUNK_SHAPE
    arrays_metadata = {
        name: {"global_shape": PREFETCH_GLOBAL_SHAPE, "chunk_shape": chunk_shape, "chunk_position": (rank,)}
        for name in array_names
    }

    print(f"MPI {rank} of {size}: arrays_metadata={arrays_metadata}", flush=True)

    # wait_for_go defaults to True: this is the path that runs the prefetch inside __init__. The analytics side (the
    # pytest process) registers one sum reduction per array before calling execute_callbacks(), which is what
    # eventually unblocks this constructor.
    bridge = Bridge(comm=MPI.COMM_WORLD, arrays_metadata=arrays_metadata)

    # The prefetch must have run on EVERY rank, and it must have picked up the analytics side's registered branches:
    # the registered ``.sum()`` per array gives a NON-EMPTY branch spec list for both arrays on every rank. On an MPI
    # whose bcast blocks for the root this scenario deadlocks in the bcast instead (the field-report symptom); on an
    # eager-bcast MPI it "only" leaves the non-root ranks without their cache, which is what this assertion catches
    # deterministically.
    missing = [name for name in array_names if not bridge._task_branches.get(name)]
    assert not missing, (
        f"__init__ prefetch did not populate task branches for {missing} (cached: {bridge._task_branches}): it must "
        "run on every rank, not only world rank 0"
    )
    print(f"MPI {rank}: bridge ready, cached task branches for {sorted(bridge._task_branches)}", flush=True)

    send_order = array_names if rank == 0 else tuple(reversed(array_names))
    print(f"MPI {rank}: sending {send_order}", flush=True)
    for array_name in send_order:
        bridge.send(array_name, np.ones(chunk_shape, dtype=np.float64), timestep=1)

    bridge.close(timestep=1)
    print(f"MPI {rank} of {size} finished", flush=True)


def has_mpirun():
    return shutil.which("mpirun") is not None


def build_mpirun_cmd(n: int, *extra_args) -> list:
    cmd = ["mpirun", "-n", str(n)]
    mpi_impl = os.environ.get("MPI_IMPL", "openmpi")
    if mpi_impl == "openmpi":
        cmd.append("--oversubscribe")
    cmd += [sys.executable, "-u", __file__] + [*extra_args]
    return cmd


def is_xdist():
    import os

    return "PYTEST_XDIST_WORKER" in os.environ


@pytest.mark.skipif(is_xdist(), reason="requires serial execution")
@pytest.mark.skipif(not has_mpirun(), reason="mpirun not available")
@pytest.mark.parametrize("i", [1, 2, 4, 8])
def test_mpi_gather(i):
    cmd = build_mpirun_cmd(i, "--mpi-gather")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:\n", result.stdout, flush=True)
    print("STDERR:\n", result.stderr, flush=True)

    assert result.returncode == 0, f"MPI test failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"


@pytest.mark.skipif(is_xdist(), reason="requires serial execution")
@pytest.mark.skipif(not has_mpirun(), reason="mpirun not available")
@pytest.mark.parametrize("global_size", [(32, 32), (32, 32, 32)])
@pytest.mark.parametrize("parallelism", [1, 2])  # per dim
@pytest.mark.parametrize("comm", ["mpi-comm-world", "mpi-comm-cart"])
def test_mpi_bridge(global_size: Tuple, parallelism: int, comm: str):
    import numpy as np
    from distributed import Client, LocalCluster

    from deisa.dask import Deisa

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        host="127.0.0.1",
        scheduler_port=0,
        dashboard_address=":0",
        worker_dashboard_address=":0",
    )
    client = Client(cluster)
    print(f"client={client}", flush=True)

    os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address

    client.wait_for_workers(2, timeout=10)

    def work():
        logging.basicConfig(level=logging.DEBUG)

        deisa = Deisa(timeout=10)

        print(f"deisa={deisa}: deisa.arrays_metadata={deisa.arrays_metadata}", flush=True)

        deisa.set("hello", "world", timestep=1)

        # Set instead of raised: ``Deisa._handle_callback_exception`` wraps the handler in
        # ``except BaseException`` and merely unregisters the callback, so an exception raised
        # here would be swallowed and the subprocess would still exit 0. The flag is read after
        # ``execute_callbacks()`` so the exit code actually reflects a failed callback.
        callback_failed = []

        def exception_handler(exception: BaseException):
            print(f"exception_handler: exception={exception}", flush=True)
            callback_failed.append(exception)

        @deisa.register("temperature", exception_handler=exception_handler)
        def cb(window):
            print(f"hello from cb. iteration={window[-1].t}", flush=True)
            darr = window[-1]
            assert isinstance(darr, DeisaArray)
            assert darr.sum().compute() == np.prod(global_size), (
                f"temperature sum should be the product of {global_size}"
            )

        deisa.execute_callbacks()
        if callback_failed:
            print(f"callback raised {callback_failed!r}", flush=True)
            return 1
        return 0

    pool = ThreadPool(processes=1)

    def error_callback(e):
        print(f"[ERROR] {e}", flush=True)
        raise e

    async_result = pool.apply_async(work, error_callback=error_callback)

    parallelism = (parallelism,) * len(global_size)

    cmd = build_mpirun_cmd(
        np.prod(parallelism),
        "--mpi-bridge",
        "--scheduler-address",
        cluster.scheduler.address,
        "--global-size",
        str(global_size),
        "--parallelism",
        str(parallelism),
        "--comm",
        comm,
    )

    print(f"cmd={cmd}", flush=True)
    result = subprocess.run(cmd, timeout=30, stdout=sys.stdout, stderr=sys.stderr)

    print(f"result={result}", flush=True)
    print("STDOUT:\n", result.stdout, flush=True)
    print("STDERR:\n", result.stderr, flush=True)

    try:
        assert result.returncode == 0, f"MPI test failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert async_result.get(timeout=10) == 0
    finally:
        pool.terminate()
        pool.join()
        client.close()
        cluster.close()


@pytest.mark.skipif(is_xdist(), reason="requires serial execution")
@pytest.mark.skipif(not has_mpirun(), reason="mpirun not available")
def test_mpi_bridge_prefetch_all_ranks():
    """Regression (0.7.0): the ``__init__`` prefetch must not deadlock with several shared arrays.

    Two ranks share TWO arrays and send them in opposite orders. The prefetch is a per-array bcast, so running it on
    world rank 0 only makes rank 0 block on the first array's sub-communicator while the other rank blocks on the
    array it sends first -- a permanent deadlock (the field-report bug). The run is bounded by a timeout so a
    deadlock fails the test instead of hanging it.

    The go events are not poked by hand: a REAL ``Deisa`` analytics side (the same ThreadPool pattern as
    ``test_mpi_bridge``) registers one sum reduction per array and calls ``execute_callbacks()``, which is what
    unblocks the bridges' ``__init__`` and files the branches the prefetch must deliver. The callback asserts both
    delivered sums, so the run also proves the data path still works after the prefetch change.
    """
    import signal
    from multiprocessing.pool import ThreadPool

    import numpy as np
    from distributed import Client, LocalCluster

    from deisa.dask import Deisa

    global_shape = PREFETCH_GLOBAL_SHAPE
    array_names = PREFETCH_ARRAYS
    # Two ranks send an all-ones chunk each: the reduced global sum is the number of elements.
    expected_sums = {name: float(np.prod(global_shape)) for name in array_names}

    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
        host="127.0.0.1",
        scheduler_port=0,
        dashboard_address=":0",
        worker_dashboard_address=":0",
    )
    client = Client(cluster)
    os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
    client.wait_for_workers(2, timeout=10)

    def work():
        logging.basicConfig(level=logging.DEBUG)

        deisa = Deisa(timeout=10)

        # Set instead of raised: ``Deisa._handle_callback_exception`` wraps the handler in ``except BaseException``
        # and merely unregisters the callback, so an exception raised here would be swallowed and the subprocess
        # would still exit 0. The list is read after ``execute_callbacks()`` so the exit code reflects a failure.
        callback_failed = []
        sums = {}

        def exception_handler(exception: BaseException):
            print(f"exception_handler: exception={exception}", flush=True)
            callback_failed.append(exception)

        # One callback over both arrays (positional parameters follow the registration order): each array gets its
        # own sum branch, which is what the bridges prefetch in __init__.
        @deisa.register(*array_names, exception_handler=exception_handler)
        def cb(alpha, beta):
            print(f"hello from cb. iterations={alpha[-1].t},{beta[-1].t}", flush=True)
            sums["alpha"] = float(alpha[-1].sum().compute())
            sums["beta"] = float(beta[-1].sum().compute())

        deisa.execute_callbacks()
        if callback_failed:
            print(f"callback raised {callback_failed!r}", flush=True)
            return 1
        if sums != expected_sums:
            print(f"unexpected sums: {sums} != {expected_sums}", flush=True)
            return 1
        return 0

    pool = ThreadPool(processes=1)

    def error_callback(e):
        print(f"[ERROR] {e}", flush=True)
        raise e

    async_result = pool.apply_async(work, error_callback=error_callback)

    cmd = build_mpirun_cmd(2, "--mpi-prefetch", "--scheduler-address", cluster.scheduler.address)
    print(f"cmd={cmd}", flush=True)

    # Own session so that a deadlocked run can be killed as a whole (mpirun plus every rank) instead of leaving
    # orphaned MPI processes behind.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        try:
            output, _ = proc.communicate(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            output, _ = proc.communicate()
            pytest.fail(
                "MPI run did not finish within 90s: the __init__ prefetch deadlocked (it must run on every rank)\n"
                f"output tail:\n{output[-4000:]}"
            )

        print("STDOUT:\n", output, flush=True)
        assert proc.returncode == 0, f"MPI test failed (returncode={proc.returncode})\nSTDOUT:\n{output[-4000:]}"
        # 1 is what ``work()`` returns when a callback failed or a delivered sum was wrong. Bounded so a stuck
        # callback still fails the test instead of hanging the suite on a shared CI box.
        assert async_result.get(timeout=180) == 0
    finally:
        pool.terminate()
        # A deadlocked run leaves the analytics side blocked in ``wait_for_bridges_to_finish`` (its bridges were
        # killed, so they never report done): the pool's worker is a daemon thread that cannot be interrupted, so
        # join only when it actually returned instead of hanging the suite behind it.
        if async_result.ready():
            pool.join()
        client.close()
        cluster.close()


# ENTRY POINT SWITCH
if __name__ == "__main__":
    print(f"sys.argv={sys.argv}", flush=True)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mpi-gather", action="store_true")
    group.add_argument("--mpi-bridge", action="store_true")
    group.add_argument("--mpi-prefetch", action="store_true")

    parser.add_argument("--scheduler-address")
    parser.add_argument("--global-size", default="(32, 32)")
    parser.add_argument("--parallelism", default="(2, 2)")
    parser.add_argument("--comm", default="none")

    args = parser.parse_args()

    if (args.mpi_bridge or args.mpi_prefetch) and not args.scheduler_address:
        parser.error("--scheduler-address is required when using --mpi-bridge or --mpi-prefetch")

    args = parser.parse_args()

    if args.mpi_prefetch:
        try:
            os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = args.scheduler_address
            mpi_prefetch_main(scheduler_address=args.scheduler_address)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            sys.exit(1)
        sys.exit(0)

    if args.mpi_bridge:
        try:
            parallelism = eval(args.parallelism)
            global_size = eval(args.global_size)
            print(f"global_size={global_size}, parallelism={parallelism}, comm={args.comm}", flush=True)
            os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = args.scheduler_address
            mpi_bridge_main(
                scheduler_address=args.scheduler_address,
                parallelism=parallelism,
                global_size=global_size,
                comm=args.comm,
            )
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            sys.exit(1)
        sys.exit(0)
