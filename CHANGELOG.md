# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Types of changes:

- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

## [Unreleased]

### Added

- CI: continuous benchmark using Bencher.
- Bridge-local precompute: callbacks whose compute operations are chunk-local reductions (`sum`, `mean`, `var`, `std`, 
  `min`, `max`, `prod`) are analyzed at registration time and shipped to Dask workers as small per-bridge partials 
  instead of the full array chunk. Branches are registered per array, so a multi-array callback gets one branch set per 
  array. Expressions whose reduction depends on another reduction's output are refused with`UnsupportedReductionError`.
- `deisa.dask.mergeable_pca`: exact mergeable PCA primitives. A local summary of one data block (`n_samples`, `mean`,
  `components`, `singular_values`) merges with another summary into the exact summary of their
  union, without either source sample ever reaching a merge node, so only a compact summary has to cross a process or
  network boundary. `merge_tree` reduces summaries with a balanced pairwise tree. The merge algebra is classical and is
  attributed as such in the module docstring: the between-block mean-correction term is the Chan-Golub-LeVeque (1979)
  covariance merge and the merge-and-truncate tree shape is Qin & Yan (arXiv:1601.07010) / Kjolstad, Demmel et al.
  (arXiv:1710.02812).
- `deisa.dask.mergeable_pca.MergeablePCA`: a scikit-learn-style PCA estimator for Dask arrays, built on those
  primitives. `fit` reduces one `local_pca` summary per input block with a balanced tree of `merge_pca` tasks, so no
  source sample reaches a merge node and the result is the exact batch PCA at full local rank. `local_rank=R`
  truncates each leaf summary to rank `R` before merging, shrinking what crosses a process boundary at the cost of a
  retained subspace that drifts from the batch one. `fit` refuses rather than approximates: input that is not 2D, a
  feature dimension split across chunks, a zero-size axis, or an `n_components` above the achievable rank each raise a
  `ValueError` naming the condition and the remedy. `_fit_dask_delayed` builds the tree WITHOUT computing and returns a
  real Dask graph, so it can be inspected or `.visualize()`d; `_fit_dask` is the `.compute()` on top. The merged root
  summary is kept as `_summary_` for further merging (and for the bridge to emit), separate from the truncated public
  `components_` / `singular_values_` / `explained_variance_`, so nothing destroys mergeability mid-tree.

### Changed

- `mpi4py` is now a required dependency.

### Fixed

- Bridge: `asyncio.run()` is no longer called with a running event loop. On the no-client path
  (`_scatter_full`, `_scatter_partials`), a loop left running on the calling thread during teardown made
  `asyncio.run` raise `RuntimeError: asyncio.run() cannot be called from a running event loop`, surfacing as
  spurious "Error while closing bridges" errors during teardown. The scatter path now runs the coroutine through
  `Bridge._run_async`: `asyncio.run` when the thread has no running loop, otherwise
  `deisa.dask.utils.run_coro_on_private_loop`, which hosts a fresh loop on a short-lived worker thread (blocking
  the caller's running loop would deadlock, since a running loop cannot be re-entered).
- Missing first callback(s) due to not waiting for `Deisa.execute_callbacks()`
- Deisa: avoid exceptions raised from `__del__()`
- Bridge: a bridge's internal `distributed.Client` is now built with `set_as_default=False`, so it no longer
  pins the process-global Dask scheduler at `dask.distributed` for its whole lifetime (including a 10 s
  reconnect window against a scheduler that is already gone). Previously any module on the same xdist worker
  that pinned `scheduler="threads"` resolved to `dask.distributed`, found no live client, and failed with
  `RuntimeError: Requested dask.distributed scheduler but no Client active.` — the flake that landed on
  unrelated `mergeable_pca` tests in CI. `_get_actor` scopes its client with `client.as_current()` because
  `distributed.Lock(key)` takes no client and reads `Client.current()`. Regression test:
  `TestBridgeClientIsNotTheProcessDefault` (fails on `main`, passes here).
- Bridge: an idle bridge client that no longer finds its scheduler now closes itself cleanly instead of
  holding the process-global scheduler config through the reconnect window.

## [0.6.3]

### Added

- CI: run mpi tests against openmpi and mpich.
- benchmark: time-to-callback. Measure the time between `Bridge.send()` and the execution of a callback.
- Define Ruff code format.
- CI: check code format and style using Ruff linter.
- check monotonically increasing timestep in callbacks.

### Fixed

- Deisa `execute_callbacks` block until all callbacks complete.
- Deisa `_free_windows` concurrency issue.
- inconsistent iteration in callback data.

## [0.6.2]

### Fixed

- Bridge `_gather_global_metadata` now return a sorted list of _global_array_names.
  This fixes `_setup_array_comms` not seeing the same list order.

## [0.6.1]

### Changed

- move `pytest.ini` to the repo root.
- add non-distributed array in mpi test.

### Fixed

- Non-distributed arrays are now handled correctly.
- Bridge.send `update_workers` now uses the sub-communicator.
- CI: install `mpi4py` and `openmpi` together using pip.

## [0.6.0]

### Added

- documentation website
- getting started example code
- feature: per-array communicator.
  This feature also enables the handling of non-distributed arrays.

## [0.5.1]

### Added

- codemeta.json

### Changed

- replace the setup.py by a pyproject.toml
- use chunk position from arrays_metadata instead of the chunk position from the bridge communicator

### Fixed

- UserWarning: Client desires key but key is unknown

## [0.5.0]

### Added

- constants.py: to regroup constants
- Deisa.__del__: to cleanup stored Futures

### Changed

- set minimal dask version to 2024.9.0
- rename Bridge.send's iteration argument to timetep
- bridge.send: per bridge id and iteration round-robin over the workers so that data is not scattered to the same worker
- use barrier instead of gather in bridge close
- bridge.execute_callbacks: wait for all tasks to finish before closing
- prefix Future keys with `KEY_PREFIX` to help identify associated tasks
- use a `weakref` inside the topic handler to avoid circular references
- move constants from `deisa.py` to `constants.py`

## [0.4.1]

### Removed

- dependency on `uuid`

## [0.4.0]

### Added

- logger
- `pytest-timeout` to stop tests that can hang
- MPI tests
- Communicator to Bridge (DaskComm, MPIComm)
- Option to select worker to scatter data to
- `_get_actor` helper function in utils to start a singleton Actor
- `pytest.ini`
- Option to update the worker list, before `send()`, on a Bridge using `update_worker`
- Option to filter the worker list, before `send()`, on a Bridge using `filter_worker`
- DeisaArray wrapper to unify handling of Dask array and iteration
- Dependency on mpi4py
- `Deisa.execute_callbacks`: unblock bridges, execute callbacks and wait for bridges to close
- `utils.build_deisa_array()` to build a DeisaArray from a Dask Array
- Deisa decorator to register callbacks

### Changed

- Bump `deisa-core` to `0.5.0`
- Update Deisa ctor and Bridge ctor to comply with `deisa-core` 0.5.0
- Change callback logic to use native `asyncio` and Dask event mechanism
- Replace client.scatter by custom scatter that limits communication to the scheduler
- Move `get_connection_info` to utils
- set Client `heartbeat_interval` to `sys.maxsize`
- Change `TestSimulation` to use `asyncio` to run `bridge.send`
- Gracefully stop bridges. Deisa waits for all bridges to close
- Feedback is sent from Deisa to the Bridges
- Single Dask Client, no matter how many Bridges. Bridge id=0 handles comm with scheduler
- Bridge.close() now takes a timestep
- `get_client` and `get_connection_info` now pass args and kwargs to the Client ctor

### Fixed

- Handshake protocol throwing a `TimeoutError`.

### Removed

- Deisa.close() is no longer needed due to changes with handshake
- Deisa.delete() is no longer needed

### Deprecated

- Deisa.unregister_callback will be removed in version 1.0.0
