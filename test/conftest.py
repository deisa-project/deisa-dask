"""Test isolation for deisa-dask.

Restores ``dask.config["scheduler"]`` after each test so a leaked default Client cannot
hand its scheduler config to later tests on the same xdist worker.
"""

import dask.config
import pytest


@pytest.fixture(autouse=True, scope="function")
def reset_dask_scheduler():
    """Restore ``dask.config['scheduler']`` to its pre-test value after each test.

    Guards against a leaked default ``Client`` holding ``dask.config["scheduler"]`` at
    ``"dask.distributed"``, which makes later tests on the same xdist worker resolve bare
    ``.compute()`` against a dead client. The restore is a plain ``dask.config.set``: a
    ``with dask.config.set(...)`` block would put the leaked value back on exit. The key is
    absent in some dask versions, so a missing value is restored by ``pop``.
    """
    _missing = object()
    previous = dask.config.get("scheduler", _missing)

    yield

    if previous is _missing:
        dask.config.pop("scheduler", None)
    elif dask.config.get("scheduler", _missing) != previous:
        dask.config.set({"scheduler": previous})
