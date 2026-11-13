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
import logging
import os
import sys
import time

import numpy as np
import pytest
from distributed import Client, LocalCluster
from utils import FakeComm

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
        client.close()
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
        future = res["future"][0]
        assert set(res["who_has"]) == {future}
        assert res["nbytes"][future] == data.nbytes
        assert res["who_has"][future] == [sorted_workers[0]], res["who_has"]

        who_has = client.who_has()
        assert who_has[future] == [sorted_workers[0]], "default scatter did not register the key with the scheduler"
        worker = self._inproc_workers(cluster)[sorted_workers[0]]
        assert worker.data[future].__array_interface__["data"][0] == original_buffer_addr, "default scatter copied"

        # (b) The default really is the FULL worker set, not just the first one: the scatter primitive round-robins
        # a multi-key payload across every worker it is given, and ``_scatter_full`` defaults that list to the full
        # set.
        payload = {"k-a": np.zeros(2), "k-b": np.ones(2)}
        who_has, _nbytes = bridge.client.sync(bridge._scatter_payload, sorted_workers, payload)
        assert {addr for addrs in who_has.values() for addr in addrs} == all_addrs

    def test_precompute_partial_is_zero_copy_local(self, env_setup_inproc):
        """A LOCAL precompute partial reaches the in-process worker numerically intact on an ``inproc://`` cluster.

        NOTE: this test is WEAK BY CONSTRUCTION and is kept only as a cheap regression on the plumbing -- it cannot
        detect a copy. ``env_setup_inproc`` runs with ``processes=False``, so the workers speak ``inproc://``, which
        does not serialize even a ``to_serialize``-wrapped array: the buffer address is preserved whether or not
        zero-copy ran (verified -- this test still passes with the local branch of
        ``_better_scatter_to_workers`` disabled). The real proof is
        ``test_precompute_partial_reaches_inprocess_worker_zero_copy``, which puts the in-process worker on a
        ``processes=True`` (``tcp://``) cluster so a serialized scatter genuinely moves the buffer.

        ``_scatter_partials`` no longer pre-serializes its payload (``valmap(to_serialize, ...)`` removed): the raw
        partial goes through ``_better_scatter_to_workers``, whose local branch writes the very same buffer into the
        worker store. This pins the plumbing half of that contract, plus the invariant that the full chunk never
        becomes resident when partials are shipped.
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

    def test_precompute_partial_reaches_inprocess_worker_zero_copy(self):
        """The precompute path delivers its PARTIAL to an in-process worker ZERO-COPIED.

        ``send()`` with precompute-active routes ``_scatter_partials`` -> ``_scatter_payload`` ->
        ``_better_scatter_to_workers``, so on one send() it asserts: (1) the worker's partial buffer address is
        identical to what the bridge wrote -- nothing serialized; (2) the callback's reduction is the true sum of
        the global data; (3) the full chunk is resident on no worker (only the partial may cross bridge -> worker).
        A MIXED topology (one subprocess worker plus one in-process ``Worker`` on a ``processes=True`` cluster) is
        what gives (1) its meaning: the in-process worker speaks ``tcp://`` so a serialized scatter really moves
        the buffer address; on ``processes=False`` (``inproc://``) nothing serializes and the assert would pass
        vacuously.
        """
        from distributed import Worker
        from test_precompute_memory import _make_compute_callback
        from TestSimulator import TestSimulation
        from utils import wait_for as _wait_for

        from deisa.dask import Deisa

        cluster = LocalCluster(
            n_workers=1, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0"
        )
        os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(1, timeout=20)

        async def _start():
            return await Worker(cluster.scheduler.address, nthreads=1)

        inproc_worker = client.sync(_start)
        client.wait_for_workers(2, timeout=20)

        array_name = "temperature"
        chunk_shape = (512, 512)  # 2 MiB chunk vs ~4 KiB partials: full and partial are unambiguous.

        sim = TestSimulation(
            client,
            mpi_parallelism=(1, 1),
            arrays_metadata={array_name: {"global_shape": chunk_shape, "chunk_shape": chunk_shape}},
            wait_for_go=False,
        )
        bridge = sim.bridges[0]
        assert inproc_worker.address in bridge.workers, "the in-process worker must be one of the bridge's workers"

        # send() round-robins over sorted(workers): pick the timestep that targets the in-process worker.
        sorted_workers = sorted(bridge.workers)
        timestep = (sorted_workers.index(inproc_worker.address) - bridge.id) % len(sorted_workers)

        # Capture the computed partials, and the chunk send() actually handed to the executor.
        recorded: dict = {}
        original_execute = bridge._execute_operations_on_chunk

        def spy_execute(chunk_arg, branches):
            partials = original_execute(chunk_arg, branches)
            recorded.update(partials)
            recorded.setdefault("__chunk__", chunk_arg)
            return partials

        bridge._execute_operations_on_chunk = spy_execute

        # Snapshot the worker-resident state at scatter time, before the topic handler releases the keys.
        scatter_info: dict = {}
        snapshot: dict = {}
        original_scatter_partials = bridge._scatter_partials

        def spy_scatter(partials, branches, array_name_arg, workers):
            def inspect(dask_worker):
                return {k: (v.nbytes if hasattr(v, "nbytes") else None) for k, v in dask_worker.data.items()}

            out = original_scatter_partials(partials, branches, array_name_arg, workers)
            res, futures_entries = out
            scatter_info["future-info"] = res
            scatter_info["entries"] = {entry["output_key"]: entry for entry in futures_entries}
            snapshot["per_worker"] = client.run(inspect)
            snapshot["inproc"] = {k: v for k, v in inproc_worker.data.items()}
            return out

        bridge._scatter_partials = spy_scatter

        results: list = []
        deisa = Deisa(wait_for_go=False)
        deisa.register(array_name)(
            _make_compute_callback(
                "arr = window[-1]\ns0 = arr.sum(axis=0).compute()\nresults.append((np.asarray(s0), tuple(arr.shape)))",
                results,
            )
        )
        time.sleep(0.5)

        global_data = sim.generate_data(array_name, iteration=timestep, update_workers=True)
        assert _wait_for(lambda: len(results) >= 1, timeout=30), "callback was not called within 30s"
        assert snapshot, "the precompute scatter never ran: this test must observe one send()"

        chunk = recorded.pop("__chunk__")
        recorded = {k: v for k, v in recorded.items() if not k.startswith("__")}
        assert recorded, f"no partial was computed for {array_name!r}: the precompute path was not exercised"
        assert bridge._task_branches.get(array_name), "bridge has no cached branch: the precompute path never engaged"

        try:
            future_info = scatter_info["future-info"]
            for output_key, partial in recorded.items():
                # Use the key the scatter reported rather than pattern-matching who_has (may already be released).
                meta = scatter_info["entries"][output_key]
                key = meta["future"]
                assert key in future_info["who_has"], f"scattered key {key!r} missing from the who_has report"
                assert future_info["who_has"][key] == [inproc_worker.address], (
                    f"partial {key!r} must be owned by the in-process worker {inproc_worker.address}"
                )
                assert key in snapshot["inproc"], f"partial {key!r} is absent from the in-process worker's data store"

                stored = snapshot["inproc"][key]
                # (1) zero-copy: the worker holds the very buffer the bridge wrote, not a deserialized copy.
                assert isinstance(stored, np.ndarray), f"expected an ndarray partial, got {type(stored)!r}"
                assert stored.shape == partial.shape == tuple(meta["shape"]), "partial shape changed in transit"
                assert stored.__array_interface__["data"][0] == partial.__array_interface__["data"][0], (
                    f"zero-copy failed for {output_key!r}: the buffer was serialized in transit"
                )
                # Value equality guards against two distinct buffers that happen to alias.
                assert np.array_equal(stored, partial), "worker-held partial differs in VALUE from the one computed"

            # (3) nothing chunk-sized may be resident on any worker: only the partials cross bridge -> worker.
            chunk_bytes = chunk.nbytes
            offenders = [
                (worker, held_key, size)
                for worker, keymap in snapshot["per_worker"].items()
                for held_key, size in keymap.items()
                if size is not None and size >= chunk_bytes
            ]
            assert not offenders, (
                f"the full chunk ({chunk.shape}, {chunk_bytes} bytes) is resident on a worker: {offenders}"
            )

            # (2) numerical correctness: the callback's reduction is the true sum(axis=0) of the global data.
            assert len(results) == 1, f"expected exactly one callback invocation, got {len(results)}"
            computed, delivered_shape = results[0]
            truth = np.sum(global_data, axis=0)
            assert np.allclose(computed, truth, rtol=1e-5, atol=1e-9), "callback reduction is not the true sum"
            assert np.asarray(computed).shape == truth.shape, "delivered reduction shape differs from the truth"
            assert delivered_shape == tuple(truth.shape), "the callback window must be the combined reduction shape"
        finally:

            async def stop_worker(w):
                await w.close()

            client.sync(stop_worker, inproc_worker)
            client.close()
            cluster.close()
            del deisa
            del sim
