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


def _build_bridges(make_comm, size, metadata_for_rank):
    """Build one bridge per rank in parallel threads -> ``(bridges, errors)``, both keyed by rank.

    A derivation error is per-rank and raised from ``Bridge.__init__`` after the Split collective, so it is recorded
    in ``errors`` instead of propagating out of the thread (a ``pytest.raises`` block cannot span the threads that
    must all reach the collectives).
    """
    bridges: dict = {}
    errors: dict = {}

    def run(rank):
        try:
            bridges[rank] = Bridge(comm=make_comm(rank), arrays_metadata=metadata_for_rank(rank), wait_for_go=False)
        except BaseException as e:
            errors[rank] = e

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return bridges, errors


def _shutdown(bridges, size):
    """Close the bridges, but only when every rank built one: ``close()`` runs a collective and would hang otherwise.

    Called from a ``finally`` so a failed assertion in a test body cannot leave half-built bridges to be collected
    later, where the same barrier would deadlock the garbage collector's thread.
    """
    if len(bridges) == size:
        _close_all([bridges[rank] for rank in range(size)])


def _positions(bridges, size, array_name="temperature"):
    """The resolved positions in rank order, or ``None`` when some rank failed to build its bridge (see ``errors``)."""
    if len(bridges) != size:
        return None
    return [bridges[rank].arrays_metadata[array_name]["chunk_position"] for rank in range(size)]


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

    @pytest.mark.timeout(30)
    def test_uneven_chunks_event_carries_per_piece_shape_and_dtype(self, env_setup):
        """Uneven chunks: the event carries each gathered piece's OWN shape/dtype.

        Two ranks declare the same array with global_shape (3,), but chunk (1,) at chunk_position (0,) on rank 0 and
        chunk (2,) at (1,) on rank 1, with different dtypes. Both send on the full-chunk path. The root used to stamp
        its own ``chunk.shape``/``chunk.dtype`` on every gathered piece, so both entries reported (1,)/float64 and
        the Deisa side inferred 3 tiles against 2 pieces.
        """
        client, cluster = env_setup

        metadata_0 = {"temperature": {"global_shape": (3,), "chunk_shape": (1,), "chunk_position": (0,)}}
        metadata_1 = {"temperature": {"global_shape": (3,), "chunk_shape": (2,), "chunk_position": (1,)}}
        comm_state = FakeComm.State(2)

        def make_bridge(args):
            rank, meta = args
            return Bridge(comm=FakeComm(comm_state, rank), arrays_metadata=meta, wait_for_go=False)

        bridges = async_map([(0, metadata_0), (1, metadata_1)], make_bridge)

        chunks = {0: np.ones(1, dtype=np.float64), 1: np.ones(2, dtype=np.int32)}

        def do_send(rank):
            bridges[rank].send("temperature", chunks[rank], timestep=0)

        async_map(range(2), do_send)

        events = client.get_events("temperature")
        assert len(events) == 1
        _, info = events[0]
        # The base pipeline's "no precompute" sentinel on the full-chunk path is None (v0.7.0), not False.
        assert not info["precomputed"]
        assert len(info["futures"]) == 2

        by_position = {tuple(f["chunk_position"]): f for f in info["futures"]}
        assert set(by_position) == {(0,), (1,)}
        assert tuple(by_position[(0,)]["shape"]) == (1,)
        assert by_position[(0,)]["dtype"] == "float64"
        assert tuple(by_position[(1,)]["shape"]) == (2,)
        assert by_position[(1,)]["dtype"] == "int32"

        _close_all(bridges)


class TestDerivedChunkPosition:
    """``chunk_position`` is optional: the bridge derives a default from the communicator topology, once, at
    registration. A user-specified position stays authoritative and the derivation is never consulted for it."""

    def _assert_derivation_error(self, errors, size, array_name="temperature"):
        """Every rank must raise the loudly-worded derivation error naming the array and pointing at chunk_position."""
        assert len(errors) == size, f"every rank must raise, got {sorted(errors)}"
        for error in errors.values():
            assert isinstance(error, ValueError), f"expected a ValueError, got {error!r}"
            assert array_name in str(error), str(error)
            assert "chunk_position" in str(error), str(error)

    def test_default_position_row_major_on_uniform_grid(self, env_setup):
        """No position + plain comm + 3 ranks over a uniform grid -> the rank's C-order tile index.

        global (6,) with chunk (2,) is 3 tiles, so the 3 ranks derive (0,), (1,), (2,) -- an odd rank count, which
        ``np.unravel_index(rank, grid_extent)`` handles without any even-split assumption.
        """
        client, cluster = env_setup

        metadata = {"temperature": {"global_shape": (6,), "chunk_shape": (2,)}}
        comm_state = FakeComm.State(3)

        bridges, errors = _build_bridges(lambda rank: FakeComm(comm_state, rank), 3, lambda rank: metadata)

        try:
            assert errors == {}, f"derivation must succeed on a uniform grid, got {errors}"
            assert _positions(bridges, 3) == [(0,), (1,), (2,)]

            # The resolved position is what travels to the analytics side: the event carries it per piece.
            async_map(range(3), lambda rank: bridges[rank].send("temperature", np.ones(2), timestep=0))

            events = client.get_events("temperature")
            assert len(events) == 1
            _, info = events[0]
            assert {tuple(f["chunk_position"]) for f in info["futures"]} == {(0,), (1,), (2,)}
        finally:
            _shutdown(bridges, 3)

    def test_default_position_row_major_two_ranks(self, env_setup):
        """The documented 2-rank example: global (4,) with chunk (2,) -> rank 0 at (0,), rank 1 at (1,)."""
        env_setup  # use fixture

        metadata = {"temperature": {"global_shape": (4,), "chunk_shape": (2,)}}
        comm_state = FakeComm.State(2)

        bridges, errors = _build_bridges(lambda rank: FakeComm(comm_state, rank), 2, lambda rank: metadata)

        try:
            assert errors == {}, f"derivation must succeed on a uniform grid, got {errors}"
            assert _positions(bridges, 2) == [(0,), (1,)]
        finally:
            _shutdown(bridges, 2)

    def test_default_position_from_cart_coords(self, env_setup):
        """No position + Cartesian topology -> each rank's Cartesian coordinates.

        3 ranks with the non-square ``MPI.Compute_dims(3, 2) == (3, 1)`` shape -- valid per the user's steer -- and a
        2-D array so the 2 coordinates are one position per axis.
        """
        env_setup  # use fixture

        metadata = {"temperature": {"global_shape": (6, 4), "chunk_shape": (2, 4)}}
        comm_state = FakeComm.State(3)

        bridges, errors = _build_bridges(
            lambda rank: FakeCartComm(comm_state, rank, dims=(3, 1)), 3, lambda rank: metadata
        )

        try:
            assert errors == {}, f"derivation must succeed on a Cartesian topology, got {errors}"
            positions = _positions(bridges, 3)
            assert positions == [(0, 0), (1, 0), (2, 0)]
            # The derived positions are exactly the communicator's own coordinates.
            assert positions == [tuple(bridges[rank].comm.Get_coords(rank)) for rank in range(3)]
        finally:
            _shutdown(bridges, 3)

    def test_default_position_cart_dimension_mismatch_raises(self, env_setup):
        """No position + cart coordinates that do not match the array's axis count -> loud ValueError."""
        env_setup  # use fixture

        metadata = {"temperature": {"global_shape": (6,), "chunk_shape": (2,)}}
        comm_state = FakeComm.State(3)

        bridges, errors = _build_bridges(
            lambda rank: FakeCartComm(comm_state, rank, dims=(3, 1)), 3, lambda rank: metadata
        )

        try:
            self._assert_derivation_error(errors, 3)
        finally:
            _shutdown(bridges, 3)

    def test_default_position_uneven_global_raises(self, env_setup):
        """No position + chunks that do not divide the global shape -> loud ValueError, never repaired."""
        env_setup  # use fixture

        metadata = {"temperature": {"global_shape": (5,), "chunk_shape": (2,)}}
        comm_state = FakeComm.State(2)

        bridges, errors = _build_bridges(lambda rank: FakeComm(comm_state, rank), 2, lambda rank: metadata)

        try:
            self._assert_derivation_error(errors, 2)
        finally:
            _shutdown(bridges, 2)

    def test_default_position_grid_extent_mismatch_raises(self, env_setup):
        """No position + a uniform grid whose tile count is not the rank count -> loud ValueError.

        global (4,) with chunk (2,) is 2 tiles but the communicator has 3 ranks: that distribution needs an explicit
        position, which is what the error asks for.
        """
        env_setup  # use fixture

        metadata = {"temperature": {"global_shape": (4,), "chunk_shape": (2,)}}
        comm_state = FakeComm.State(3)

        bridges, errors = _build_bridges(lambda rank: FakeComm(comm_state, rank), 3, lambda rank: metadata)

        try:
            self._assert_derivation_error(errors, 3)
        finally:
            _shutdown(bridges, 3)

    def test_user_position_wins_over_derivation(self, env_setup):
        """A user-specified position is authoritative: it is used verbatim, not replaced by a derived one."""
        env_setup  # use fixture

        positions = [(2,), (1,), (0,)]
        comm_state = FakeComm.State(3)

        def metadata_for_rank(rank):
            return {"temperature": {"global_shape": (6,), "chunk_shape": (2,), "chunk_position": positions[rank]}}

        bridges, errors = _build_bridges(lambda rank: FakeComm(comm_state, rank), 3, metadata_for_rank)

        try:
            assert errors == {}, f"no derivation is needed when a position is given, got {errors}"
            assert _positions(bridges, 3) == positions
        finally:
            _shutdown(bridges, 3)
