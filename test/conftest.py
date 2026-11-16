"""
Conftest for deisa-dask test isolation.

Provides the ``reset_dask_scheduler`` fixture that guarantees
``dask.config["scheduler"]`` is reset after each test, preventing a default
Client from one test from leaking its scheduler config to subsequent tests on
the same xdist worker.
"""

import dask.config
import pytest


@pytest.fixture(autouse=True, scope="function")
def reset_dask_scheduler():
    """Restore ``dask.config['scheduler']`` to its pre-test value after each test.

    A default ``Client`` (``set_as_default=True``) rewrites
    ``dask.config["scheduler"]`` to ``"dask.distributed"`` for as long as it is alive.
    When such a client outlives its test (or its cluster is already gone), the next test
    on the same xdist worker resolves every bare ``.compute()`` against it and fails with
    ``RuntimeError: Requested dask.distributed scheduler but no Client active`` during the
    10 s reconnect window.

    The fixture records the value BEFORE the test and writes it back AFTER the test. The
    restore must be a plain ``dask.config.set`` call: wrapping it in a
    ``with dask.config.set(...)`` block would put the leaked value back on context exit,
    which is exactly why the previous version of this fixture was a no-op. ``scheduler``
    is not defined in every supported dask release's shipped config, so an absent key is
    restored by removing it again rather than writing a value.
    """
    _missing = object()
    previous = dask.config.get("scheduler", _missing)

    yield

    if previous is _missing:
        dask.config.pop("scheduler", None)
    elif dask.config.get("scheduler", _missing) != previous:
        dask.config.set({"scheduler": previous})
