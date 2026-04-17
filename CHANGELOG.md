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

- Add logger
- `pytest-timeout` to stop tests that can hang
- Add MPI tests
- Add communicator to Bridge (DaskComm, MPIComm)
- Add option to select worker to scatter data to
- Add `_get_actor` helper function in utils to start a singleton Actor
- Add `pytest.ini`

### Changed

- Change callback logic to use native `asyncio` and Dask event mechanism
- Bump `deisa-core` to `0.4.0`
- Replace client.scatter by custom scatter that limits communication to the scheduler
- Move `get_connection_info` to utils
- set Client `heartbeat_interval` to `sys.maxsize`
- Change `TestSimulation` to use `asyncio` to run `bridge.send`
- Gracefully stop bridges. Deisa waits for all bridges to close

### Fixed

- Handshake protocol throwing a `TimeoutError`.
