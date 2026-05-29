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

- constants.py: to regroup constants
- Deisa.__del__: to cleanup stored Futures

### Changed

- set minimal dask version to 2024.9.0
- rename Bridge.send's iteration argument to timetep
- bridge.send: per bridge id and iteration round-robin over the workers
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
