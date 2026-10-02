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
import asyncio
import gc
import logging
import os
import sys
import time
import weakref

import numpy as np
import pytest
from distributed import Client, LocalCluster
from utils import FakeCartComm, FakeComm, async_close_bridges, async_map

from deisa.dask import Bridge

logging.basicConfig(level=logging.DEBUG)


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


class TestBridge:
    def get_new_bridge(self):
        arrays_metadata = {"temperature": {"global_shape": (1,), "chunk_shape": (1,), "chunk_position": (0,)}}
        comm_state = FakeComm.State(1)
        bridge = Bridge(comm=FakeComm(comm_state, 0), arrays_metadata=arrays_metadata, wait_for_go=False)
        return bridge, arrays_metadata

    def test_ctor(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata = self.get_new_bridge()
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])
        assert isinstance(bridge.comm, FakeComm)
        assert not bridge._has_close_been_called

    def test__del__(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata = self.get_new_bridge()
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])
        assert isinstance(bridge.comm, FakeComm)
        assert not bridge._has_close_been_called
        bridge.__del__()
        assert bridge._has_close_been_called

    def test_close(self, env_setup):
        client, _ = env_setup
        bridge, _ = self.get_new_bridge()
        assert not bridge._has_close_been_called
        bridge.close(timestep=42)
        assert bridge._has_close_been_called

    @pytest.mark.flaky(retries=3, delay=1)
    def test_send_update_workers(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])

        cluster.scale(2)
        cluster.wait_for_workers(2)

        bridge.send("temperature", np.ones(1), timestep=0, update_workers=True)

        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])

    @pytest.mark.flaky(retries=3, delay=1)
    def test_send_filter_workers_empty(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            return []

        with pytest.raises(TypeError) as _:
            bridge.send("temperature", np.ones(1), timestep=0, filter_workers=filter)

    @pytest.mark.parametrize("i", [1, np.array(1), np.array([1])])
    def test_send_timestep_type(self, env_setup, i):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        bridge.send("temperature", np.ones(1), timestep=i)

    def test_send_filter_workers_without_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            assert isinstance(workers, dict)
            for addr in workers.keys():
                assert isinstance(addr, str)
                assert addr in [w.worker_address for w in cluster.workers.values()]
            return list(workers.keys())

        bridge.send("temperature", np.ones(1), timestep=0, update_workers=False, filter_workers=filter)

    def test_send_filter_workers_with_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            assert isinstance(workers, dict)
            return list(workers.keys())

        bridge.send("temperature", np.ones(1), timestep=0, update_workers=True, filter_workers=filter)

    def test_cart_comm(self, env_setup):
        client, cluster = env_setup

        arrays_metadata = {"temperature": {"global_shape": (8, 8), "chunk_shape": (4, 4), "chunk_position": (0, 0)}}
        comm_state = FakeComm.State(4)

        def make_bridge(rank):
            return Bridge(
                comm=FakeCartComm(comm_state, rank, dims=(2, 2)), arrays_metadata=arrays_metadata, wait_for_go=False
            )

        # Create bridges in parallel (Split is a collective op)
        bridges = async_map(range(4), make_bridge)

        async def _bridge_send():
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        bridge.send, "temperature", np.ones(arrays_metadata["temperature"]["chunk_shape"]), timestep=0
                    )
                    for i, bridge in enumerate(bridges)
                ]
            )

        asyncio.run(_bridge_send())

        event = client.get_events("temperature")
        assert len(event) == 1
        _, info = event[0]
        assert info["array_name"] == "temperature"
        assert info["iteration"] == 0
        assert len(info["futures"]) == 4
        for f in info["futures"]:
            assert f["chunk_position"] in [(0, 0), (0, 1), (1, 0), (1, 1)]

        async def _bridge_close():
            await asyncio.gather(*[asyncio.to_thread(bridge.close, 0) for i, bridge in enumerate(bridges)])

        asyncio.run(_bridge_close())

    def test_execute_operations_on_chunk_raises_on_failing_branch(self, env_setup):
        """A branch that raises must NOT be dropped silently.

        A caught-and-continued ``branch_func(chunk)`` exception would ship FEWER partials than bridges, and the combine
        would silently produce a wrong reduction (scalar stacks get smaller sums; mean/moment aggregators miss a
        bridge's ``n``). It raises a typed ``PrecomputeRuntimeError`` naming the branch.
        """
        from deisa.dask.branch import BranchSpec
        from deisa.dask.precompute_analyzer import PrecomputeRuntimeError

        bridge, _ = self.get_new_bridge()

        def boom(chunk):
            raise ValueError("boom")

        branch = BranchSpec(
            output_key="f-sum",
            input_name="temperature",
            output_kind="scalar",
            branch_func=boom,
            chunk_axis=None,
            finalize=None,
            partial_shape=(),
            partial_dtype="float64",
            op_name="sum",
        )
        with pytest.raises(PrecomputeRuntimeError) as excinfo:
            bridge._execute_operations_on_chunk(np.ones((1,)), [branch])
        assert "f-sum" in str(excinfo.value)
        assert "boom" in str(excinfo.value)

    @pytest.fixture
    def env_setup_inproc(self):
        cluster = LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False, dashboard_address=":0", worker_dashboard_address=":0"
        )
        os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(2, timeout=10)
        yield client, cluster
        # client.close() BEFORE cluster.close(): closing the cluster out from under a live client leaves the client
        # reconnecting in a background task, which keeps the xdist worker from ever exiting (whole-suite hang).
        client.close()
        cluster.close()

    @pytest.fixture
    def env_setup_remote(self):
        cluster = LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0"
        )
        os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(2, timeout=10)
        yield client, cluster
        cluster.close()

    @pytest.fixture
    def env_setup_mixed(self):
        cluster = LocalCluster(
            n_workers=1, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0"
        )
        os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(1, timeout=10)

        # One in-process worker connecting to the same scheduler
        from distributed import Worker

        async def _start():
            return await Worker(cluster.scheduler.address, nthreads=1)

        async def _stop(w):
            await w.close()

        inproc_worker = client.sync(_start)
        client.wait_for_workers(2, timeout=10)

        yield client, cluster, inproc_worker

        client.sync(_stop, inproc_worker)
        client.close()
        cluster.close()

    def test_send_uses_inprocess_path(self, env_setup_inproc, caplog):
        client, cluster = env_setup_inproc
        bridge, _ = self.get_new_bridge()

        data = np.ones(1)
        original_buffer_addr = data.__array_interface__["data"][0]
        print(f"original buffer address: {hex(original_buffer_addr)}", flush=True)

        bridge.send("temperature", data, timestep=0)

        stored_buffer_addrs = [
            w.data[key].__array_interface__["data"][0]
            for w in cluster.workers.values()
            for key in w.data
            if "ndarray-" in key
        ]
        print(f"stored buffer addresses: {[hex(a) for a in stored_buffer_addrs]}", flush=True)

        assert len(stored_buffer_addrs) > 0, "No ndarray key found in any worker's data store"
        assert original_buffer_addr in stored_buffer_addrs, (
            f"Zero-copy failed: original buffer {hex(original_buffer_addr)} "
            f"not found in stored buffers {[hex(a) for a in stored_buffer_addrs]}"
        )

    def test_send_uses_remote_path(self, env_setup_remote, caplog):
        client, cluster = env_setup_remote
        bridge, _ = self.get_new_bridge()

        data = np.ones(1)
        original_buffer_addr = data.__array_interface__["data"][0]
        print(f"original buffer address: {hex(original_buffer_addr)}", flush=True)

        bridge.send("temperature", data, timestep=0)

        # Verify data arrived on a worker via the scheduler
        # Buffer address cannot be checked directly since remote workers
        # are in separate OS processes with independent memory spaces
        who_has = client.who_has()
        ndarray_keys = [k for k in who_has if "ndarray-" in k]
        assert len(ndarray_keys) > 0, "No ndarray key found on any worker after remote scatter"
        assert all(len(who_has[k]) > 0 for k in ndarray_keys), "Some keys have no owner worker"

    def test_send_uses_mixed_path(self, env_setup_mixed, caplog):
        client, cluster, inproc_worker = env_setup_mixed
        bridge, _ = self.get_new_bridge()

        inproc_addr = inproc_worker.address
        remote_addrs = set(bridge.workers) - {inproc_addr}

        # Verify both worker types are visible to the bridge
        assert inproc_addr in bridge.workers, "In-process worker not in bridge.workers"
        assert len(remote_addrs) > 0, "No remote worker in bridge.workers"

        data = np.ones(1)
        original_buffer_addr = data.__array_interface__["data"][0]
        print(f"original buffer address: {hex(original_buffer_addr)}", flush=True)

        # ``send()`` picks exactly one worker by round-robin over the SORTED worker list -- ``index =
        # (timestep + self.id) % len(workers)`` -- so pick the timestep that lands on the in-process worker instead of
        # relying on the address sort order. The proof that this arithmetic matches ``send()`` is the placement
        # assertion below, not this line.
        sorted_workers = sorted(bridge.workers)
        target_index = sorted_workers.index(inproc_addr)
        timestep = (target_index - bridge.id) % len(sorted_workers)

        bridge.send("temperature", data, timestep=timestep)

        # Verify the key landed on exactly one worker
        who_has = client.who_has()
        ndarray_keys = [k for k in who_has if "ndarray-" in k]
        assert len(ndarray_keys) > 0, "No ndarray key found after scatter"
        all_holders = {addr for k in ndarray_keys for addr in who_has[k]}
        assert len(all_holders) == 1, "Key should be on exactly one worker"

        # The mixed fixture's whole point is that the in-process worker is reachable, so PIN that branch: accepting
        # the remote placement would let a serialized-fallback implementation pass this test. This assertion is
        # also what proves the timestep arithmetic above -- if it were wrong, the key would land on the remote worker
        # and this would fail.
        assert all_holders == {inproc_addr}, (
            f"expected the in-process worker {inproc_addr} to hold the key (got {all_holders}); the timestep "
            f"arithmetic does not round-robin onto it -- inproc at sorted index {target_index} of "
            f"{sorted_workers} (remote workers: {remote_addrs})"
        )

        # Zero-copy: the in-process worker holds the very same buffer the bridge wrote.
        in_process_keys = [key for key in inproc_worker.data if "ndarray-" in key]
        stored_buffer_addrs = [inproc_worker.data[key].__array_interface__["data"][0] for key in in_process_keys]
        print(f"stored buffer addresses: {[hex(a) for a in stored_buffer_addrs]}", flush=True)
        assert len(stored_buffer_addrs) > 0, "No ndarray key found in in-process worker's data store"
        assert original_buffer_addr in stored_buffer_addrs, (
            f"Zero-copy failed: original buffer {hex(original_buffer_addr)} "
            f"not found in stored buffers {[hex(a) for a in stored_buffer_addrs]}"
        )

    def _inproc_workers(self, cluster):
        """``{address: Worker}`` for THIS cluster's in-process workers.

        Built the same way the scatter path does (``_global_workers`` is a WeakSet of every Worker in the process,
        which also holds workers leaked by earlier tests' closed clusters), then filtered down to the addresses the
        given cluster actually owns.
        """
        from distributed.worker import _global_workers

        owned = {w.worker_address: w for w in cluster.workers.values()}
        return {addr: w for addr, w in ((w.address, w) for w in _global_workers) if addr in owned}

    def test_scatter_full_defaults_to_all_workers(self, env_setup_inproc):
        """``_scatter_full(data)`` with no ``workers`` argument must still scatter.

        ``benchmark/scatter/local/connect-clients.py:80`` calls exactly that shape. Dropping the
        ``if workers is None: workers = ...`` default leaves ``None`` as the worker list, and
        ``_better_scatter_to_workers`` then dies on ``sorted(None)`` before anything is written.
        """
        client, cluster = env_setup_inproc
        bridge, _ = self.get_new_bridge()
        assert bridge.workers, "bridge has no workers to default to"
        all_addrs = set(bridge.workers)
        sorted_workers = sorted(bridge.workers)

        data = np.arange(4, dtype=np.float64)
        original_buffer_addr = data.__array_interface__["data"][0]

        # (a) The benchmark call shape: one ndarray, no ``workers`` argument.
        res = bridge._scatter_full(data)
        future = res["future"]
        assert set(res["who_has"]) == {future}
        assert res["nbytes"][future] == data.nbytes
        assert res["who_has"][future] == [sorted_workers[0]], res["who_has"]

        who_has = client.who_has()
        assert who_has[future] == [sorted_workers[0]], "default scatter did not register the key with the scheduler"
        worker = self._inproc_workers(cluster)[sorted_workers[0]]
        assert worker.data[future].__array_interface__["data"][0] == original_buffer_addr, "default scatter copied"

        # (b) The default really is the FULL worker set, not just the first one: a multi-element payload must
        # round-robin across every worker the bridge knows about.
        results = bridge._scatter_full([np.zeros(2), np.ones(2)])
        assert len(results) == 2
        keys = [r["future"] for r in results]
        who_has = client.who_has()
        assert {addr for k in keys for addr in who_has[k]} == all_addrs

    def test_precompute_partial_is_zero_copy_local(self, env_setup_inproc):
        """A LOCAL precompute partial must reach the in-process worker zero-copied AND numerically intact.

        ``_scatter_partials`` no longer pre-serializes its payload (``valmap(to_serialize, ...)`` removed): the raw
        partial goes through ``_better_scatter_to_workers``, whose local branch writes the very same buffer into the
        worker store. This pins both halves of that contract, plus the invariant that the full chunk never becomes
        resident when partials are shipped.
        """
        from deisa.dask.branch import BranchSpec

        client, cluster = env_setup_inproc
        bridge, _ = self.get_new_bridge()

        sorted_workers = sorted(bridge.workers)
        target = sorted_workers[0]
        timestep = (0 - bridge.id) % len(sorted_workers)

        produced = []

        def branch_func(chunk):
            partial = np.asarray(chunk, dtype=np.float64).sum(axis=0, keepdims=True)
            produced.append(partial)
            return partial

        bridge._task_branches["temperature"] = [
            BranchSpec(
                output_key="k-axis0",
                input_name="temperature",
                output_kind="scalar",
                branch_func=branch_func,
                chunk_axis=(0,),
                finalize=None,
                partial_shape=(1, 3),
                partial_dtype="float64",
                op_name="sum",
            )
        ]

        data = np.arange(6, dtype=np.float64).reshape(2, 3)
        bridge.send("temperature", data, timestep=timestep)

        assert len(produced) == 1, f"branch_func ran {len(produced)} times, expected exactly 1"
        partial = produced[0]

        worker = self._inproc_workers(cluster)[target]
        partial_keys = [k for k in worker.data if "-partial-k-axis0-" in k]
        assert len(partial_keys) == 1, f"expected one partial key on {target}, got {partial_keys}"
        stored = worker.data[partial_keys[0]]

        # (a) zero-copy: identical buffer, not an equal copy.
        assert stored.__array_interface__["data"][0] == partial.__array_interface__["data"][0], (
            "precompute partial was copied: stored buffer "
            f"{hex(stored.__array_interface__['data'][0])} != original "
            f"{hex(partial.__array_interface__['data'][0])}"
        )

        # (b) numerically correct.
        np.testing.assert_array_equal(stored, data.sum(axis=0, keepdims=True))

        # The full chunk must NOT be resident anywhere in-process: only partials cross.
        resident_chunks = [k for w in self._inproc_workers(cluster).values() for k in w.data if "ndarray-" in k]
        assert resident_chunks == [], f"full chunk crossed to a worker despite the precompute path: {resident_chunks}"

        # The scheduler knows where the partial is.
        assert client.who_has()[partial_keys[0]] == [target]

    def test_scatter_rolls_back_local_writes_when_report_fails(self, env_setup_inproc):
        """A failure between the local write and the scheduler report must not orphan the key.

        ``worker.update_data`` commits synchronously, so any raise before ``scheduler.update_data`` completes leaves
        the worker holding bytes the scheduler never accounted for. The compensation must actually remove them --
        through the worker's state machine, so the task does not stay in state ``memory``.
        """
        client, cluster = env_setup_inproc
        bridge, _ = self.get_new_bridge()

        target = sorted(bridge.workers)[0]
        payload = {"deisa-probe-a": np.ones(4), "deisa-probe-b": np.ones(4)}

        class _ReportFails:
            async def update_data(self, **kwargs):
                raise RuntimeError("scheduler report exploded")

        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(bridge._better_scatter_to_workers([target], payload, scheduler=_ReportFails(), client_id=None))
        assert "scheduler report exploded" in str(excinfo.value)

        worker = self._inproc_workers(cluster)[target]
        for key in payload:
            assert key not in worker.data, f"compensated scatter left {key} in the worker store"
            ts = worker.state.tasks.get(key)
            assert ts is None or ts.state != "memory", f"{key} is still resident in state {ts and ts.state}"

    def test_scatter_rolls_back_key_written_before_zero_copy_assert(self, env_setup_inproc, monkeypatch):
        """A zero-copy assert failure must still roll the key back.

        ``worker.update_data`` commits the key BEFORE the ``written.append`` that records it for rollback, so the
        implementation's own ``assert id(worker.data[key]) == id(val)`` can fire with the key already resident. If the
        rollback list were appended to after the assert, that key -- the one whose commit the assert is complaining
        about -- would be the single orphan that survives, since it never reaches ``_release_local_keys``.

        Make the worker's store hand back a COPY so the production assert trips for real, with no other failure in
        play, then assert the key is gone afterwards.
        """
        client, cluster = env_setup_inproc
        bridge, _ = self.get_new_bridge()

        target = sorted(bridge.workers)[0]
        payload = {"deisa-assert-probe": np.ones(4)}

        worker = self._inproc_workers(cluster)[target]
        real_update_data = worker.update_data

        def _update_data_but_copy(data, **kwargs):
            # Commit, then replace what the store hands back with an equal-but-distinct object so the bridge's
            # zero-copy assert fires. The key stays committed -- exactly the divergence being pinned.
            result = real_update_data(data, **kwargs)
            for key in list(data):
                worker.data[key] = data[key].copy()
            return result

        monkeypatch.setattr(worker, "update_data", _update_data_but_copy)

        with pytest.raises(AssertionError, match="copied data"):
            asyncio.run(bridge._better_scatter_to_workers([target], payload, scheduler=None, client_id=None))

        monkeypatch.undo()
        for key in payload:
            assert key not in worker.data, f"the key committed before the zero-copy assert was orphaned: {key}"
            ts = worker.state.tasks.get(key)
            assert ts is None or ts.state != "memory", f"{key} is still resident in state {ts and ts.state}"

    def test_local_nbytes_matches_worker_reported_sizeof(self, env_setup_inproc):
        """The local branch's ``nbytes`` must use the SAME convention ``Worker.update_data`` reports.

        ``Worker.update_data`` computes its reported size with ``distributed.sizeof.safe_sizeof``; the local branch
        must not substitute a different convention (the pre-fix code used ``val.nbytes``/``sys.getsizeof``, which
        silently disagrees on a dict-shaped partial). Pinned on both payload flavors a bridge actually ships.
        """
        client, cluster = env_setup_inproc
        bridge, _ = self.get_new_bridge()

        target = sorted(bridge.workers)[0]
        # An ndarray full chunk and a mean/moment-shaped dict partial -- the two flavors the precompute path ships.
        payloads = {
            "deisa-sizeof-ndarray": np.arange(6, dtype=np.float64).reshape(2, 3),
            "deisa-sizeof-mean": {"n": np.array(2), "total": np.arange(3, dtype=np.float64)},
        }

        _, who_has, nbytes = asyncio.run(bridge._better_scatter_to_workers([target], payloads))

        worker = self._inproc_workers(cluster)[target]
        worker_reported = worker.update_data({f"{k}-ref": v for k, v in payloads.items()})["nbytes"]

        assert who_has == {k: [target] for k in payloads}
        for key in payloads:
            expected = worker_reported[f"{key}-ref"]
            assert nbytes[key] == expected, (
                f"{key}: local branch reported {nbytes[key]} bytes, worker reported {expected} -- "
                f"the two sizeof conventions disagree"
            )

        # Sanity: the conventions must not be trivially equal because both are degenerate.
        assert nbytes["deisa-sizeof-ndarray"] == payloads["deisa-sizeof-ndarray"].nbytes
        assert nbytes["deisa-sizeof-mean"] != sys.getsizeof(payloads["deisa-sizeof-mean"]), (
            "the dict-shaped assertion is vacuous: sys.getsizeof would have matched"
        )


class TestPrecomputeRegressions:
    """Regression tests for bridge delivery and interpreter-shutdown teardown."""

    def _meta(self, array_name, chunk_pos, global_shape=(8,), chunk_shape=(4,)):
        return {array_name: {"global_shape": global_shape, "chunk_shape": chunk_shape, "chunk_position": chunk_pos}}

    def test_wait_for_go_false_skips_go_wait(self, env_setup):
        """Bridge.__init__ with wait_for_go=False must not block on the go event.

        The go event is set only by ``Deisa.execute_callbacks()``; when the bridge is constructed before callbacks are
        executed (e.g. a simulation that registers callbacks lazily or not at all), ``wait_for_go=False`` skips the
        wait and leaves branch fetching to the first ``send()``.
        """
        from distributed import Event

        from deisa.dask.handshake import Handshake

        client, cluster = env_setup
        Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=client).set()
        start = time.monotonic()
        bridge = Bridge(
            comm=FakeComm(FakeComm.State(1), 0),
            arrays_metadata=self._meta("temperature", (0,)),
            wait_for_go=False,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 5, f"Bridge.__init__ blocked for {elapsed:.1f}s despite wait_for_go=False"
        # No go signal arrived and no prefetch happened; lazy fetch remains correct.
        bridge.close(timestep=0)

    def test_del_skips_close_at_interpreter_shutdown(self, env_setup, monkeypatch):
        """Teardown: __del__ must not run the blocking close() at shutdown.

        ``close()`` runs a world barrier which can never complete once peer ranks are gone -- ``__del__`` skips
        ``close()`` while ``sys.is_finalizing()``.
        """
        env_setup  # use fixture
        bridge, _ = self.get_plain_bridge()
        monkeypatch.setattr(sys, "is_finalizing", lambda: True)
        calls = []
        orig_close = bridge.close

        def spy_close(timestep):
            calls.append(timestep)
            return orig_close(timestep)

        bridge.close = spy_close
        bridge.__del__()
        assert calls == [], "__del__ must skip close() during interpreter shutdown"

    def test_del_does_not_block_the_event_loop(self, env_setup, caplog):
        """Teardown: __del__ must not make blocking calls from inside an event-loop callback.

        The collector runs ``__del__`` on whichever thread dropped the last reference; for a leaked bridge
        that is a distributed "IO loop" thread. ``close()`` synchronizes with the scheduler through
        ``Client.sync()``, which blocks until *that* loop runs the callback -- from inside the loop the
        callback never runs, so the thread waits out the whole ``sync()`` timeout and, with an in-process
        cluster, freezes the scheduler, its workers and the client that share the loop.

        Deterministic sister of the intermittent serial teardown flake: the collection is forced onto the
        client's loop thread instead of waiting for the garbage collector to pick that thread by chance.
        """
        client, _ = env_setup
        bridge, _ = self.get_plain_bridge()
        # Keep the bridge alive through a reference cycle: refcounting alone cannot free it, so only the
        # gc.collect() below -- running on the client's loop thread -- can collect it (and run __del__).
        bridge._probe_cycle = [bridge]
        collected = weakref.ref(bridge)
        del bridge
        assert collected() is not None, "the reference cycle should keep the bridge alive"

        async def collect_on_the_loop():
            gc.collect()

        with caplog.at_level(logging.DEBUG, logger="deisa.dask.bridge"):
            start = time.monotonic()
            client.sync(collect_on_the_loop)
            elapsed = time.monotonic() - start

        assert collected() is None, "the bridge should have been collected inside the loop callback"
        assert elapsed < 30, f"__del__ blocked the event loop for {elapsed:.1f}s"
        assert any("event-loop callback" in record.getMessage() for record in caplog.records), (
            "close() must be skipped: it cannot complete while the loop that has to service its collectives "
            "is blocked by the very call waiting for it"
        )

    def test_close_skips_collectives_at_interpreter_shutdown(self, env_setup, monkeypatch):
        """Teardown: close() skips the barrier / sub-comm Free() at shutdown.

        At ``sys.is_finalizing()`` the blocking collectives are skipped; a hang is worse than an exception.
        """
        env_setup  # use fixture
        bridge, _ = self.get_plain_bridge()
        monkeypatch.setattr(sys, "is_finalizing", lambda: True)
        barriers = []
        bridge.comm.barrier = lambda: barriers.append(1)
        bridge.close(timestep=0)
        assert bridge._has_close_been_called
        assert barriers == [], "close() must skip the world barrier during interpreter shutdown"

    def test_send_non_participating_rank_skips_branch_work(self, env_setup):
        """A rank whose sub-comm for the array is _COMM_NULL does no branch work.

        The ``_COMM_NULL`` early return comes FIRST: such a rank never fetches task branches nor executes branch funcs
        on the chunk just to discard the result.
        """
        client, cluster = env_setup
        state = FakeComm.State(2)
        meta0 = self._meta("temperature", (0,))
        meta1 = self._meta("pressure", (0,))
        metas = [meta0, meta1]

        def _make(rank):
            return Bridge(comm=FakeComm(state, rank), arrays_metadata=metas[rank], wait_for_go=False)

        b0, b1 = async_map([0, 1], _make)
        # Merged-metadata scenario: the array is known to this bridge but its
        # sub-comm for it is _COMM_NULL (it does not own it).
        b1.arrays_metadata["temperature"] = meta0["temperature"]
        calls = {"branches": 0, "chunk": 0}
        orig_get = b1._get_task_branches
        orig_exec = b1._execute_operations_on_chunk

        def spy_get(array_name):
            calls["branches"] += 1
            return orig_get(array_name)

        def spy_exec(chunk, branches):
            calls["chunk"] += 1
            return orig_exec(chunk, branches)

        b1._get_task_branches = spy_get
        b1._execute_operations_on_chunk = spy_exec
        b1.send("temperature", np.ones(4), timestep=0)
        assert calls == {"branches": 0, "chunk": 0}

        async_close_bridges([b0, b1], 0)

    def test_gather_partial_positions_follow_their_bridge(self, env_setup):
        """A partial's chunk_position comes from its own bridge, not the index.

        Two bridges share array ``temperature``: bridge 0 ships NO partials (legacy i.e. its branch cache is empty) and
        bridge 1 ships one precomputed partial. The metadata travels with its own entry -- indexing filtered metadata
        with ``enumerate`` against ``gathered_data`` would misalign and let one bridge's partial inherit another
        bridge's coordinates. The event must carry bridge 1's ``(1,)`` for its partial.
        """
        from deisa.dask.branch import BranchSpec

        client, cluster = env_setup
        state = FakeComm.State(2)
        meta0 = self._meta("temperature", (0,))
        meta1 = self._meta("temperature", (1,))
        metas = [meta0, meta1]

        def _make(rank):
            return Bridge(comm=FakeComm(state, rank), arrays_metadata=metas[rank], wait_for_go=False)

        b0, b1 = async_map([0, 1], _make)

        # Bridge 0: branch cache empty -> legacy full-chunk path.
        b0._task_branches["temperature"] = []
        # Bridge 1: one precomputed branch (chunk-local sum).
        branch = BranchSpec(
            output_key="k1",
            input_name="temperature",
            output_kind="scalar",
            branch_func=lambda c: float(np.asarray(c).sum()),
            chunk_axis=None,
            finalize=None,
            partial_shape=(),
            partial_dtype="float64",
            op_name="sum",
        )
        b1._task_branches["temperature"] = [branch]
        # Stub the scatters: no worker/network interaction needed for the guard.
        b1._scatter_partials = lambda partials, branches, array_name, workers: {
            "future-info": {"future": ["fut-k1"], "who_has": {}, "nbytes": {}},
            "precomputed": {
                "k1": {"future": "fut-k1", "kind": "scalar", "shape": (), "dtype": "float64", "finalize": None}
            },
        }

        async def _send():
            await asyncio.gather(
                asyncio.to_thread(b0.send, "temperature", np.ones(4), timestep=0),
                asyncio.to_thread(b1.send, "temperature", np.ones(4), timestep=0),
            )

        asyncio.run(_send())

        event = client.get_events("temperature")
        assert len(event) == 1
        _, info = event[0]
        assert info["precomputed"]
        assert len(info["futures"]) == 1
        assert info["futures"][0]["future"] == "fut-k1"
        assert info["futures"][0]["chunk_position"] == (1,), info["futures"][0]["chunk_position"]

        async_close_bridges([b0, b1], 0)

    def get_plain_bridge(self):
        arrays_metadata = {"temperature": {"global_shape": (1,), "chunk_shape": (1,), "chunk_position": (0,)}}
        comm_state = FakeComm.State(1)
        bridge = Bridge(comm=FakeComm(comm_state, 0), arrays_metadata=arrays_metadata, wait_for_go=False)
        return bridge, arrays_metadata
