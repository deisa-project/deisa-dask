###################################################################################################
# Copyright (c) 2025 Commissariat a l'énergie atomique et aux énergies alternatives (CEA)
# SPDX-License-Identifier: MIT
###################################################################################################
import os

from distributed import Client

from .bridge import Bridge
from .deisa import Deisa


def get_connection_info(dask_scheduler_address: str | Client) -> Client:
    if isinstance(dask_scheduler_address, Client):
        client = dask_scheduler_address
    elif isinstance(dask_scheduler_address, str):
        try:
            client = Client(address=dask_scheduler_address)
        except ValueError:
            # try scheduler_file
            if os.path.isfile(dask_scheduler_address):
                client = Client(scheduler_file=dask_scheduler_address)
            else:
                raise ValueError(
                    "dask_scheduler_address must be a string containing the address of the scheduler, "
                    "or a string containing a file name to a dask scheduler file, or a Dask Client object.")
    else:
        raise ValueError(
            "dask_scheduler_address must be a string containing the address of the scheduler, "
            "or a string containing a file name to a dask scheduler file, or a Dask Client object.")

    return client
