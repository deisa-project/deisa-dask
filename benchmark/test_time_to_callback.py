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
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# Number of send() -> callback hops performed per benchmark round. Each round
# launches one mpirun process group and loops this many sends, so the reported
# latency is averaged over many hops (better statistics than one-per-round).
N_SENDS = 100

# How long bridge.get() blocks waiting for feedback from the Deisa callback
# (seconds). 10 seconds is generous enough to handle typical callback execution
# time plus scheduler dispatch latency under normal load.
FEEDBACK_TIMEOUT = 10.0

# Interval between polls in the feedback loop (seconds).
# 100ms is a good balance between responsiveness and not flooding the
# Dask scheduler with queue reads.
FEEDBACK_POLL_INTERVAL = 0.1

# How often to print a progress marker during the N_SENDS loop on the
# MPI side. Every N prints avoid slowing down the benchmark with I/O.
MPI_PROGRESS_INTERVAL = 10

# How often to print a progress marker in the Deisa callback.
# Only the first N prints are emitted to keep CI logs manageable.
CB_TRACE_LIMIT = 10


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def _has_mpirun():
    return shutil.which("mpirun") is not None


def _is_xdist():
    return "PYTEST_XDIST_WORKER" in os.environ


def _mpi_bridge_main(array_name: str, n_sends: int):
    """Run MPI bridge processes for benchmarking.

    Performs `n_sends` Bridge.send() calls, one at a time. After each
    send the rank calls ``bridge.get(array_name, timestep=i)`` which
    blocks until the Deisa callback has executed (which calls
    ``deisa.set(array_name, timestep, timestep)``). This guarantees
    exactly one send is in flight at any given moment — the next send
    only fires after the previous callback has completed.

    Synchronization approach: bridge.get() uses a Dask Queue for
    feedback (bridge.get() retrieves feedback put by deisa.set()
    in the Deisa callback). bridge.get() now blocks (blocking Queue.get)
    so there is no polling loop or sleep needed.
    """
    from mpi4py import MPI

    from deisa.dask import Bridge

    bridge_comm = MPI.COMM_WORLD
    dims = MPI.Compute_dims(bridge_comm.Get_size(), 1)
    bridge_comm = bridge_comm.Create_cart(dims)
    rank = bridge_comm.Get_rank()
    size = bridge_comm.Get_size()

    print(f"[{rank}/{size}] Bridge started.", flush=True)

    global_shape = (size,)
    chunk_shape = (1,)

    arrays_metadata = {
        array_name: {
            "global_shape": global_shape,
            "chunk_shape": chunk_shape,
            "chunk_position": bridge_comm.Get_coords(rank),
        }
    }

    # wait_for_go defaults to True: the bridge waits for Deisa to be ready
    # before sending (correct handshake, no premature sends).
    bridge = Bridge(comm=bridge_comm, arrays_metadata=arrays_metadata)

    t0_total = time.monotonic()
    for i in range(n_sends):
        # Build the chunk as int64 so the nanosecond timestamp round-trips
        # exactly (float64 cannot represent ~1.7e18 losslessly). The timestamp
        # lives at element [0]; remaining elements are arbitrary fill.
        data = np.zeros(chunk_shape, dtype=np.int64)
        data[0] = np.int64(time.time_ns())

        bridge.send(array_name, data, timestep=i, update_workers=False, filter_workers=lambda w: list(w.keys()))

        # Block until the Deisa callback has executed for this timestep.
        # bridge.get() is non-blocking (returns None when the queue is
        # empty), so we poll with a sleep between checks. The sleep gives
        # the Deisa callback thread time to execute and call deisa.set()
        # between polls.
        t0 = time.monotonic()
        deadline = t0 + FEEDBACK_TIMEOUT
        while True:
            got = bridge.get(array_name, timestep=i)
            if got is not None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"[{rank}/{size}] timeout waiting for feedback on timestep {i} "
                    f"after {t0 - t0_total:.1f}s total (total send loop {time.monotonic() - t0_total:.1f}s)"
                )
            time.sleep(FEEDBACK_POLL_INTERVAL)

        if (i + 1) % MPI_PROGRESS_INTERVAL == 0 or i == 0 or i == n_sends - 1:
            elapsed_total = time.monotonic() - t0_total
            print(f"[bridge {rank}] progress: {i + 1}/{n_sends} sends done ({elapsed_total:.1f}s total)", flush=True)

    bridge.close(timestep=n_sends)


def _spawn_mpi(scheduler_address: str, nb_bridges: int, array_name: str, n_sends: int):
    """Launch the MPI bridge processes (a fresh process group each call)."""
    cmd = [
        "mpirun",
        "-n",
        str(nb_bridges),
        "--oversubscribe",
        sys.executable,
        "-u",
        __file__,
        "--mpi-bridge",
        "--scheduler-address",
        scheduler_address,
        "--nb-bridges",
        str(nb_bridges),
        "--array-name",
        array_name,
        "--n-sends",
        str(n_sends),
    ]
    return subprocess.run(cmd, timeout=300)


@pytest.fixture(scope="function")
def env_setup():
    import gc

    from distributed import LocalCluster

    # --- setup (not measured): fresh cluster + workers per round ---
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        host="127.0.0.1",
        scheduler_port=0,
        dashboard_address=":0",
        worker_dashboard_address=":0",
    )

    cluster.wait_for_workers(1, timeout=10)
    os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler.address

    yield cluster

    cluster.close()

    gc.collect()


@pytest.mark.benchmark
@pytest.mark.skipif(_is_xdist(), reason="requires serial execution")
@pytest.mark.skipif(not _has_mpirun(), reason="mpirun not available")
@pytest.mark.parametrize("nb_bridges", [1, 2, 4])
def test_time_to_callback_mpi(nb_bridges: int, benchmark, env_setup):
    """Measure the true send() -> Deisa callback latency using real MPI.

    pytest-benchmark's setup pays the cluster spin-up and worker wait once per
    round (pure harness cost, excluded from the timed phase). Each timed round
    launches one mpirun group and loops N_SENDS send() -> callback hops; the
    true per-hop latency (send timestamp embedded in the array payload vs the
    Deisa callback timestamp) is averaged over all hops and stored via
    benchmark.extra_info. No timing data is written to disk.

    Synchronization: after each Bridge.send(), the MPI rank calls bridge.get()
    which blocks until the Deisa callback has fired and called deisa.set().
    This guarantees exactly one send is in flight at any given moment.
    """

    from deisa.dask import Deisa

    _ = env_setup

    array_name = "temperature"
    trace_counter = [0]  # mutable counter, shared across callback invocations

    def run_benchmark():
        results = []  # true send -> callback delta (ns), one per hop

        def deisa_side():
            deisa = Deisa(feedback_queue_size=1024, timeout=60)

            @deisa.register(array_name)
            def timed_callback(window):
                # Deisa passes a list of DeisaArray (one per registered array
                # name); window[0] is the GLOBAL dask array. Materialize it to
                # read the int64 send timestamp embedded at element [0].
                cb_ns = time.time_ns()
                np_arr = window[0].compute()
                send_ns = int(np.min(np_arr))
                delta_ns = cb_ns - send_ns
                results.append(delta_ns)

                # Trace output so we can verify send/receive pairing.
                # Limit traces to avoid I/O overhead on the Dask event loop.
                iteration = window[0].timestep
                if trace_counter[0] < CB_TRACE_LIMIT:
                    trace_counter[0] += 1
                    print(
                        f"[deisa-cb] iter={iteration} send_ns={send_ns} cb_ns={cb_ns} delta_ms={delta_ns / 1e6:.3f}",
                        flush=True,
                    )

                # Signal back to the bridge that this timestep's callback has
                # completed. The bridge.get() call on the MPI side waits for
                # this entry in the feedback queue.
                deisa.set(array_name, iteration, timestep=iteration)  # noqa: F821

            deisa.execute_callbacks()
            del deisa

        thread = threading.Thread(target=deisa_side)
        thread.start()

        result = _spawn_mpi(
            scheduler_address=os.environ["DEISA_DASK_SCHEDULER_ADDRESS"],
            nb_bridges=nb_bridges,
            array_name=array_name,
            n_sends=N_SENDS,
        )
        assert result.returncode == 0, f"MPI bridge failed with returncode {result.returncode}"

        thread.join(timeout=10)
        return results

    results = benchmark.pedantic(run_benchmark, warmup_rounds=0, rounds=1, iterations=1)

    print(f"\n\n>>>> [{len(results)}] {results} \n\n")

    # pytest-benchmark's main column measures the timed phase only (cluster
    # already up, Deisa thread waiting, mpirun send -> callback hops). The
    # number we actually care about -- the true send() -> callback latency --
    # is captured manually inside the callback and surfaced via
    # benchmark.extra_info so it lands in the machine-readable JSON for CI
    # regression tracking.
    benchmark.extra_info["nb_bridges"] = nb_bridges
    benchmark.extra_info["global_shape"] = (nb_bridges,)
    benchmark.extra_info["n_sends_per_round"] = N_SENDS

    if results and len(results) > 0:
        # Report in nanoseconds (true send -> callback latency).
        avg_ms = np.mean(results)
        median_ms = np.median(results)
        min_ms = np.min(results)
        max_ms = np.max(results)
        std_ms = np.std(results)
        seventyfive = np.quantile(results, 0.75)
        ninty = np.quantile(results, 0.90)
        nintynine = np.quantile(results, 0.99)
        nintyninenine = np.quantile(results, 0.999)

        # add results to pytest-benchmark in milliseconds
        benchmark.extra_info["true_latency_ms"] = {
            "avg": avg_ms / 1e6,
            "median": median_ms / 1e6,
            "min": min_ms / 1e6,
            "max": max_ms / 1e6,
            "std": std_ms / 1e6,
            "75": seventyfive / 1e6,
            "90": ninty / 1e6,
            "99": nintynine / 1e6,
            "99.9": nintyninenine / 1e6,
            "n": len(results),
        }

        print(
            f"\nsend->callback ({nb_bridges} MPI bridges, "
            f"{N_SENDS} sends/round): "
            f"avg={avg_ms:.3f}ms, median={median_ms:.3f}ms, "
            f"min={min_ms:.3f}ms, max={max_ms:.3f}ms, std={std_ms:.3f}ms, "
            f"75={seventyfive}ms, 90={ninty}ms, 99={nintynine}ms, 99.9={nintyninenine}ms, "
            f"(n={len(results)})"
        )

        # output the results using Bencher Metric Format (https://bencher.dev/docs/reference/bencher-metric-format)
        result_file = Path("bencher-result.json")

        try:
            with result_file.open("r") as fp:
                bencher = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            bencher = {}

        bencher[f"time-to-callback-{nb_bridges}"] = {
            "latency": {
                "value": avg_ms,
                "lower_value": min_ms,
                "upper_value": max_ms,
            },
            "50": {"value": median_ms},
            "75": {"value": seventyfive},
            "90": {"value": ninty},
            "99": {"value": nintynine},
            "99.9": {"value": nintyninenine},
        }

        with result_file.open("w") as fp:
            json.dump(bencher, fp, indent=2, cls=NpEncoder)


# ENTRY POINT SWITCH
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mpi-bridge", action="store_true")

    parser.add_argument("--scheduler-address")
    parser.add_argument("--nb-bridges", type=int, default=1)
    parser.add_argument("--array-name", default="temperature")
    parser.add_argument("--n-sends", type=int, default=1)

    args = parser.parse_args()

    if args.mpi_bridge and not args.scheduler_address:
        parser.error("--scheduler-address is required when using --mpi-bridge")

    if args.mpi_bridge:
        try:
            os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = args.scheduler_address
            _mpi_bridge_main(array_name=args.array_name, n_sends=args.n_sends)
        except Exception as e:
            print(f"[ERROR] {e}", flush=True)
            import traceback

            traceback.print_exc()
            sys.exit(1)
        sys.exit(0)
