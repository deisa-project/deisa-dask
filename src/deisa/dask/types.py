import logging

import dask.array as da

from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeisaArray:
    dask: da.Array
    t: int
