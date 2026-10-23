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
    """Ensure dask.config['scheduler'] is reset after each test.

    This prevents a default Client from one test leaking its scheduler config
    to subsequent tests on the same xdist worker. The fixture:

    1. Records the scheduler config before the test (for reference only)
    2. Yields control to the test
    3. After the test, unconditionally resets ``dask.config["scheduler"]`` to
       ``"threads"`` — this is the critical step that prevents config leakage.
    """
    # Record config before test (for reference; not used after yield)
    _ = dask.config.get("scheduler", None)

    yield

    # CRITICAL: Unconditionally reset scheduler config after the test.
    # Without this, a default Client from a previous test leaves
    # ``dask.config["scheduler"]` set to ``"dask.distributed"``, which causes
    # any subsequent ``.compute()`` or scheduler resolution to raise
    # ``RuntimeError: Requested dask.distributed scheduler but no Client active``
    # during the 10 s reconnect window.
    with dask.config.set({"scheduler": "threads"}):
        pass
