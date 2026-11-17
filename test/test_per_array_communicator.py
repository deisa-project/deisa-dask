# =============================================================================
# Tests for GitHub issues:
#   #17 — Use one communicator per array
#   #109 — Support sending data not present on all bridges (non-distributed arrays)
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
# * Neither the names nor the names of the contributors may be used to
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
import asyncio
import os
import threading

import numpy as np
import pytest
from distributed import Client, LocalCluster
from utils import FakeCartComm, FakeComm, async_map, wait_for

from deisa.dask import Bridge, Deisa


@pytest.fixture(scope="function")
def env_setup():
    cluster = LocalCluster(
        n_workers=1, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0"
    )
    os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
    client = Client(cluster)
    client.wait_for_workers(1, timeout=10)
    yield client, cluster
    client.close()
    cluster.close()


def _meta(array_name, chunk_pos, global_shape=(8,), chunk_shape=(4,)):
    """Helper: build arrays_metadata entry for a single array."""
    return {array_name: {"global_shape": global_shape, "chunk_shape": chunk_shape, "chunk_position": chunk_pos}}


def _close_all(bridges):
    """Close all bridges in parallel."""

    async def _do():
        await asyncio.gather(*[asyncio.to_thread(b.close, 0) for b in bridges])

    asyncio.run(_do())


def _make_bridges_with_timeout(items, func, timeout: float = 30.0):
    """Build bridges in parallel, failing instead of hanging when collectives do not match.

    ``Bridge.__init__`` is collective, so a mismatched collective (the 0.7.0 rank-0-only prefetch) leaves the
    FakeComm ranks waiting on each other forever. Daemon threads plus a bounded join turn that hang into a test
    failure instead of a stuck worker.
    """
    items = list(items)
    results: list = [None] * len(items)
    errors = []

    def run(index, item):
        try:
            results[index] = func(item)
        except BaseException as e:  # noqa: BLE001 - re-raised below as a test failure
            errors.append((item, e))

    threads = [threading.Thread(target=run, args=(i, item), daemon=True) for i, item in enumerate(items)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout)

    assert not errors, f"bridge creation failed: {errors}"
    stuck = [items[i] for i, thread in enumerate(threads) if thread.is_alive()]
    assert not stuck, f"bridge creation deadlocked on ranks {stuck}: mismatched collectives"
    return results


class TestSingleBridge:
    """All tests use FakeComm.State(1) — one bridge, no collectives needed."""

    @pytest.mark.parametrize("narrays", [1, 2, 5])
    def test_setup_array_comms_creates_sub_comms(self, env_setup, narrays):
        """_setup_array_comms creates one sub-comm per declared array."""
        env_setup  # use fixture

        arrays_metadata = {
            "arr_{}".format(i): {"global_shape": (8,), "chunk_shape": (8,), "chunk_position": (0,)}
            for i in range(narrays)
        }
        bridge = Bridge(comm=FakeComm(FakeComm.State(1), 0), arrays_metadata=arrays_metadata, wait_for_go=False)

        assert len(bridge._array_comms) == narrays
        for i in range(narrays):
            key = "arr_{}".format(i)
            assert key in bridge._array_comms
            assert bridge._array_comms[key].Get_size() == 1

    def test_single_bridge_fast_path(self, env_setup):
        """Single-bridge array uses _direct_send (no gather)."""
        client, cluster = env_setup

        arrays_metadata = _meta("temperature", (0,), (1,), (1,))
        bridge = Bridge(comm=FakeComm(FakeComm.State(1), 0), arrays_metadata=arrays_metadata, wait_for_go=False)

        bridge.send("temperature", np.ones(1), timestep=0)

        event = client.get_events("temperature")
        assert len(event) == 1
        _, info = event[0]
        assert info["array_name"] == "temperature"
        assert info["iteration"] == 0
        assert len(info["futures"]) == 1
        assert info["futures"][0]["chunk_position"] == (0,)

    def test_send_unknown_array_raises(self, env_setup):
        """send() raises ValueError for an undeclared array."""
        env_setup  # use fixture

        bridge = Bridge(
            comm=FakeComm(FakeComm.State(1), 0), arrays_metadata=_meta("temperature", (0,)), wait_for_go=False
        )

        with pytest.raises(ValueError, match="unknown"):
            bridge.send("nonexistent", np.ones(1), timestep=0)

    def test_comm_cleanup(self, env_setup):
        """bridge._array_comms populated at init, clearable."""
        env_setup  # use fixture

        bridge = Bridge(
            comm=FakeComm(FakeComm.State(1), 0), arrays_metadata=_meta("temperature", (0,)), wait_for_go=False
        )

        assert "temperature" in bridge._array_comms
        assert bridge._array_comms["temperature"] is not None

        for sub in bridge._array_comms.values():
            sub.Free()
        bridge._array_comms.clear()
        assert len(bridge._array_comms) == 0

    def test_close_calls_cleanup(self, env_setup):
        """bridge.close() frees sub-comms and clears _array_comms."""
        env_setup  # use fixture

        bridge = Bridge(
            comm=FakeComm(FakeComm.State(1), 0), arrays_metadata=_meta("temperature", (0,)), wait_for_go=False
        )

        assert len(bridge._array_comms) > 0
        bridge.close(timestep=0)
        assert bridge._has_close_been_called
        assert len(bridge._array_comms) == 0


class TestMultiBridge:
    """Multi-bridge scenarios with parallel bridge creation."""

    def test_prefetch_populates_every_rank_cache(self, env_setup):
        """Regression: the ``__init__`` prefetch must run on EVERY rank, not just world rank 0.

        ``_prefetch_task_branches`` issues one bcast per array on that array's sub-communicator, so every sub-comm
        member has to enter the same (sorted) sequence of collectives. With the 0.7.0 ``and self.id == 0`` gate only
        rank 0's cache was filled, and any second shared array deadlocked.

        The go events are NOT poked by hand here: a real ``Deisa`` analytics side registers one sum-reduction callback
        over both arrays and runs ``execute_callbacks()`` off-thread, which is what sets
        ``WAIT_FOR_EXECUTE_CB_EVENT`` and unblocks the bridges' ``__init__``. Registering the reduction also makes
        the assertion stronger than "the cache is not None": every rank must hold the NON-EMPTY branch spec list for
        both arrays right after ``__init__``, and the delivered sums are checked end to end.
        """
        env_setup  # use fixture (real cluster + scheduler address for get_client())

        comm_state = FakeComm.State(2)
        array_names = ("alpha", "beta")
        global_shape = (8,)
        chunk_shape = (4,)
        # Two ranks send an all-ones chunk each: the reduced global sum is the number of elements.
        expected_sum = float(np.prod(global_shape))

        # Filled by the callback as the topic events arrive; read under ``wait_for`` below.
        sums: dict = {}
        failed: list = []
        errors: list = []

        def run_analytics():
            try:
                deisa = Deisa(timeout=10)

                def exception_handler(exception):
                    print(f"exception_handler: exception={exception}", flush=True)
                    failed.append(exception)

                # One callback over both arrays (the positional parameters follow the registration order): each array
                # gets its own sum branch, which is what the prefetch must deliver to every rank.
                @deisa.register(*array_names, exception_handler=exception_handler)
                def cb(alpha, beta):
                    print(f"hello from cb. iterations={alpha[-1].t},{beta[-1].t}", flush=True)
                    sums["alpha"] = float(alpha[-1].sum().compute())
                    sums["beta"] = float(beta[-1].sum().compute())

                # Unblocks the bridges (the prefetch runs inside their __init__) and returns once they are closed.
                deisa.execute_callbacks()
            except BaseException as e:  # noqa: BLE001 - re-raised on the main thread below
                print(f"[ERROR] analytics side: {e}", flush=True)
                errors.append(e)

        def make_bridge(rank):
            return Bridge(
                comm=FakeComm(comm_state, rank),
                arrays_metadata={
                    name: {"global_shape": global_shape, "chunk_shape": chunk_shape, "chunk_position": (rank,)}
                    for name in array_names
                },
                # wait_for_go defaults to True: this is the path that runs the prefetch.
            )

        thread = threading.Thread(target=run_analytics, daemon=True)
        thread.start()
        try:
            bridges = _make_bridges_with_timeout(range(2), make_bridge, timeout=30)

            for rank, bridge in enumerate(bridges):
                for name in array_names:
                    cached = bridge._task_branches[name]
                    assert cached, f"rank {rank} did not prefetch task branches for {name!r}: cached={cached!r}"

            # Both ranks send the same array at the same time: the per-array gather inside send() is a collective on
            # that array's sub-communicator. FakeComm emulates one collective at a time per communicator state, so a
            # mismatched per-rank array ORDER is not representable here (the MPI test covers that deadlock shape);
            # what this test pins down is that the prefetch, not send(), filled every rank's cache.
            def send_all(bridge):
                for name in array_names:
                    bridge.send(name, np.ones(chunk_shape, dtype=np.float64), timestep=1)

            async def _send():
                await asyncio.gather(*[asyncio.to_thread(send_all, b) for b in bridges])

            asyncio.run(_send())

            # The callback fires from the analytics client's event loop once both arrays delivered iteration 1; the
            # events are asynchronous, so wait for them before tearing the bridges down.
            assert wait_for(lambda: set(sums) == set(array_names), timeout=30), f"callback did not fire: sums={sums}"
            _close_all(bridges)
        finally:
            thread.join(timeout=30)

        assert not errors, f"analytics side raised: {errors}"
        assert not thread.is_alive(), "the analytics side did not return after the bridges were closed"
        assert not failed, f"callback raised: {failed}"
        assert sums == {name: expected_sum for name in array_names}, sums

    @pytest.mark.parametrize("comm_size", [2, 4])
    def test_all_bridges_same_array(self, env_setup, comm_size):
        """N bridges all declare the same array → gather on full comm."""
        client, cluster = env_setup

        if comm_size == 4:
            dims = (2, 2)
            chunk_shape = (4, 4)
            global_shape = (8, 8)
        else:
            dims = (2,)
            chunk_shape = (4,)
            global_shape = (8,)

        arrays_metadata = {
            "temperature": {
                "global_shape": global_shape,
                "chunk_shape": chunk_shape,
                "chunk_position": tuple([0] * len(dims)),
            }
        }
        comm_state = FakeComm.State(comm_size)

        def make_bridge(rank):
            return Bridge(
                comm=FakeCartComm(comm_state, rank, dims=dims), arrays_metadata=arrays_metadata, wait_for_go=False
            )

        bridges = async_map(range(comm_size), make_bridge)

        async def _send():
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        bridge.send, "temperature", np.ones(arrays_metadata["temperature"]["chunk_shape"]), timestep=0
                    )
                    for bridge in bridges
                ]
            )

        asyncio.run(_send())

        event = client.get_events("temperature")
        assert len(event) == 1
        _, info = event[0]
        assert info["iteration"] == 0
        assert len(info["futures"]) == comm_size

        # Verify all chunk_positions appear
        positions = [f["chunk_position"] for f in info["futures"]]
        assert len(positions) == comm_size

        _close_all(bridges)

    def test_mixed_participation_2_bridges(self, env_setup):
        """Two bridges, two arrays: one shared, one only on bridge 0.

        - temperature: both bridges declare it → sub-comm size 2 → gather
        - pressure: only bridge 0 declares it → sub-comm size 1 on bridge 0 (fast-path), bridge 1 gets _COMM_NULL
        from Split and never sends.
        """
        client, cluster = env_setup

        arrays_metadata_0 = {
            "temperature": {"global_shape": (8,), "chunk_shape": (4,), "chunk_position": (0,)},
            "pressure": {"global_shape": (8,), "chunk_shape": (4,), "chunk_position": (0,)},
        }
        # Bridge 1: only temperature (pressure absent)
        arrays_metadata_1 = {
            "temperature": {"global_shape": (8,), "chunk_shape": (4,), "chunk_position": (1,)},
            "density": {"global_shape": (8,), "chunk_shape": (4,), "chunk_position": (0,)},
        }
        comm_state = FakeComm.State(2)

        def make_bridge(rank, meta):
            return Bridge(comm=FakeComm(comm_state, rank), arrays_metadata=meta, wait_for_go=False)

        bridge0, bridge1 = async_map([(0, arrays_metadata_0), (1, arrays_metadata_1)], lambda args: make_bridge(*args))

        # Both bridges send temperature (gather, sub-comm size 2)
        async def _send_temp():
            await asyncio.gather(
                asyncio.to_thread(bridge0.send, "temperature", np.ones(4), timestep=0),
                asyncio.to_thread(bridge1.send, "temperature", np.ones(4) * 2, timestep=0),
            )

        asyncio.run(_send_temp())

        # Only bridge 0 sends pressure (fast-path)
        bridge0.send("pressure", np.ones(4) * 10, timestep=0)

        # Only bridge 1 sends density (fast-path)
        bridge1.send("density", np.ones(4) * 10, timestep=0)

        # Verify temperature: 2 futures
        event_temp = client.get_events("temperature")
        assert len(event_temp) == 1
        _, info_temp = event_temp[0]
        assert info_temp["iteration"] == 0
        assert len(info_temp["futures"]) == 2
        positions_temp = {f["chunk_position"] for f in info_temp["futures"]}
        assert positions_temp == {(0,), (1,)}

        # Verify pressure: 1 future (only bridge 0)
        event_press = client.get_events("pressure")
        assert len(event_press) == 1
        _, info_press = event_press[0]
        assert info_press["iteration"] == 0
        assert len(info_press["futures"]) == 1
        assert info_press["futures"][0]["chunk_position"] == (0,)

        # Verify density: 1 future (only bridge 1)
        event_press = client.get_events("density")
        assert len(event_press) == 1
        _, info_press = event_press[0]
        assert info_press["iteration"] == 0
        assert len(info_press["futures"]) == 1
        assert info_press["futures"][0]["chunk_position"] == (0,)

        _close_all([bridge0, bridge1])

    @pytest.mark.parametrize("comm_size", [2, 3])
    def test_single_owner_array_in_larger_comm(self, env_setup, comm_size):
        """One bridge owns a 'solo' array; all bridges share 'shared'.

        The owner gets sub-comm size 1 for 'solo' → fast-path. Non-owners get _COMM_NULL for 'solo' → never send it.
        'shared' uses the full comm → gather.
        """
        client, cluster = env_setup

        # Bridge 0 owns both 'solo' and 'shared'
        arrays_metadata_0 = {
            "solo": {"global_shape": (8,), "chunk_shape": (8,), "chunk_position": (0,)},
            "shared": {"global_shape": (8 * comm_size,), "chunk_shape": (8,), "chunk_position": (0,)},
        }

        # Shared state must be created ONCE and shared across all bridge threads
        comm_state = FakeComm.State(comm_size)

        def make_bridge(rank):
            if rank == 0:
                meta = arrays_metadata_0
            else:
                meta = {"shared": {"global_shape": (8 * comm_size,), "chunk_shape": (8,), "chunk_position": (rank,)}}
            return Bridge(comm=FakeComm(comm_state, rank), arrays_metadata=meta, wait_for_go=False)

        bridges = async_map(range(comm_size), make_bridge)

        # Bridge 0 sends 'solo' (fast-path, only participant)
        bridges[0].send("solo", np.ones(8) * 42, timestep=0)

        # All bridges send 'shared'
        async def _send_shared():
            await asyncio.gather(
                *[asyncio.to_thread(b.send, "shared", np.ones(8) * (i + 1), timestep=0) for i, b in enumerate(bridges)]
            )

        asyncio.run(_send_shared())

        # Verify solo: 1 future, bridge 0's data
        event_solo = client.get_events("solo")
        assert len(event_solo) == 1
        _, info_solo = event_solo[0]
        assert len(info_solo["futures"]) == 1
        assert info_solo["futures"][0]["chunk_position"] == (0,)

        # Verify shared: comm_size futures
        event_shared = client.get_events("shared")
        assert len(event_shared) == 1
        _, info_shared = event_shared[0]
        assert len(info_shared["futures"]) == comm_size

        _close_all(bridges)

    @pytest.mark.timeout(30)
    def test_sub_comm_isolation(self, env_setup):
        """Sub-communicators for different arrays are independent.

        With FakeComm.State(1), Split() still createsComm each time, and each sub-comm has its own synchronization
        state.
        """
        env_setup  # use fixture

        arrays_metadata = {
            "temperature": {"global_shape": (8,), "chunk_shape": (8,), "chunk_position": (0,)},
            "pressure": {"global_shape": (8,), "chunk_shape": (8,), "chunk_position": (0,)},
        }
        bridge = Bridge(comm=FakeComm(FakeComm.State(1), 0), arrays_metadata=arrays_metadata, wait_for_go=False)

        assert "temperature" in bridge._array_comms
        assert "pressure" in bridge._array_comms
        assert bridge._array_comms["temperature"].Get_size() == 1
        assert bridge._array_comms["pressure"].Get_size() == 1
